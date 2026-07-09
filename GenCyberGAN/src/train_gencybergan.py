import os, csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from .dataset_loader import WindowDataset
from .models.generator import GenCyberGenerator
from .models.critics import FlowCritic, GraphCritic, CTIClassifier
from .losses import gradient_penalty_flow, gradient_penalty_graph, temporal_smoothness_loss, graph_flow_coherence_loss
from .utils import ensure_dir


def train_gencybergan(X, graph_desc, cti_ids, y, vocab_sizes, cfg, out_dir, device):
    ensure_dir(out_dir)
    ds = WindowDataset(X, graph_desc, cti_ids, y)
    loader = DataLoader(ds, batch_size=cfg['training']['batch_size'], shuffle=True, drop_last=True)
    T, feature_dim = X.shape[1], X.shape[2]
    gdim = graph_desc.shape[1]
    num_labels = int(max(y)) + 1
    mcfg = cfg['model']; tcfg = cfg['training']
    G = GenCyberGenerator(vocab_sizes, mcfg['z_dim'], mcfg['cti_emb_dim'], mcfg['cti_dim'], mcfg['hidden_dim'], T, feature_dim, gdim, mcfg.get('dropout',0.1)).to(device)
    Df = FlowCritic(feature_dim, mcfg['hidden_dim']).to(device)
    Dg = GraphCritic(gdim, mcfg['hidden_dim']).to(device)
    C = CTIClassifier(feature_dim, gdim, num_labels, mcfg['hidden_dim']).to(device)
    optG = torch.optim.Adam(G.parameters(), lr=tcfg['lr_g'], betas=(0.5,0.9))
    optDf = torch.optim.Adam(Df.parameters(), lr=tcfg['lr_d'], betas=(0.5,0.9))
    optDg = torch.optim.Adam(Dg.parameters(), lr=tcfg['lr_d'], betas=(0.5,0.9))
    optC = torch.optim.Adam(C.parameters(), lr=tcfg['lr_ids'])
    ce = nn.CrossEntropyLoss()
    log_path = os.path.join(out_dir, 'gan_training_log.csv')
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch','loss_g','loss_df','loss_dg','loss_cti'])
    for epoch in range(1, tcfg['gan_epochs']+1):
        sums = {'g':0,'df':0,'dg':0,'cti':0}; n=0
        pbar = tqdm(loader, desc=f'GAN epoch {epoch}', leave=False)
        for x_real, g_real, cti, labels in pbar:
            x_real, g_real, cti, labels = x_real.to(device), g_real.to(device), cti.to(device), labels.to(device)
            B = x_real.size(0)
            for _ in range(tcfg['critic_steps']):
                z1 = torch.randn(B, mcfg['z_dim'], device=device); z2 = torch.randn(B, mcfg['z_dim'], device=device)
                with torch.no_grad():
                    x_fake, g_fake = G(z1, z2, cti)
                optDf.zero_grad()
                gp_f = gradient_penalty_flow(Df, x_real, x_fake, device) if tcfg.get('gp_lambda', 0) > 0 else 0.0
                loss_df = Df(x_fake).mean() - Df(x_real).mean() + tcfg.get('gp_lambda', 0) * gp_f
                loss_df.backward(); optDf.step()
                optDg.zero_grad()
                gp_g = gradient_penalty_graph(Dg, g_real, g_fake, device) if tcfg.get('gp_lambda', 0) > 0 else 0.0
                loss_dg = Dg(g_fake).mean() - Dg(g_real).mean() + tcfg.get('gp_lambda', 0) * gp_g
                loss_dg.backward(); optDg.step()
                optC.zero_grad()
                loss_cti_real = ce(C(x_real, g_real), labels.clamp_min(0))
                loss_cti_real.backward(); optC.step()
            z1 = torch.randn(B, mcfg['z_dim'], device=device); z2 = torch.randn(B, mcfg['z_dim'], device=device)
            x_fake, g_fake = G(z1, z2, cti)
            logits_fake = C(x_fake, g_fake)
            loss_g = -Df(x_fake).mean() - Dg(g_fake).mean()
            loss_g = loss_g + tcfg['lambda_cti'] * ce(logits_fake, labels.clamp_min(0))
            loss_g = loss_g + tcfg['lambda_temporal'] * temporal_smoothness_loss(x_fake)
            loss_g = loss_g + tcfg['lambda_graph'] * graph_flow_coherence_loss(x_fake, g_fake)
            optG.zero_grad(); loss_g.backward(); optG.step()
            sums['g'] += float(loss_g.detach()); sums['df'] += float(loss_df.detach()); sums['dg'] += float(loss_dg.detach()); sums['cti'] += float(loss_cti_real.detach()); n += 1
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, sums['g']/max(n,1), sums['df']/max(n,1), sums['dg']/max(n,1), sums['cti']/max(n,1)])
    torch.save({'model': G.state_dict(), 'vocab_sizes': vocab_sizes, 'T': T, 'feature_dim': feature_dim, 'graph_desc_dim': gdim, 'cfg': cfg}, os.path.join(out_dir, 'gencybergan_generator.pt'))
    torch.save(Df.state_dict(), os.path.join(out_dir, 'flow_critic.pt'))
    torch.save(Dg.state_dict(), os.path.join(out_dir, 'graph_critic.pt'))
    return G
