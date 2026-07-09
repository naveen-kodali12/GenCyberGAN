import os
import numpy as np
import torch
from .utils import save_npz, ensure_dir


def mahalanobis_scores(G, ref):
    mu = ref.mean(axis=0)
    cov = np.cov(ref.T) + np.eye(ref.shape[1]) * 1e-5
    inv = np.linalg.pinv(cov)
    diff = G - mu
    return np.sqrt(np.sum(diff @ inv * diff, axis=1))


def generate_synthetic_samples(generator, cti_ids, labels, cfg, out_dir, device, real_graph_desc=None):
    ensure_dir(out_dir)
    generator.eval()
    z_dim = cfg['model']['z_dim']
    per_class = int(cfg['training']['synthetic_per_class'])
    xs, gs, ys, cs = [], [], [], []
    unique = [int(v) for v in np.unique(labels) if int(v) >= 0]
    with torch.no_grad():
        for lab in unique:
            idx = np.where(labels == lab)[0]
            if len(idx) == 0: continue
            reps = np.random.choice(idx, size=per_class, replace=True)
            cti = torch.tensor(cti_ids[reps], dtype=torch.long, device=device)
            z1 = torch.randn(per_class, z_dim, device=device)
            z2 = torch.randn(per_class, z_dim, device=device)
            x_fake, g_fake = generator(z1, z2, cti)
            xs.append(x_fake.cpu().numpy().astype('float32'))
            gs.append(g_fake.cpu().numpy().astype('float32'))
            ys.append(np.full(per_class, lab, dtype='int64'))
            cs.append(cti.cpu().numpy().astype('int64'))
    X = np.concatenate(xs, axis=0) if xs else np.empty((0,))
    Gd = np.concatenate(gs, axis=0) if gs else np.empty((0,))
    y = np.concatenate(ys, axis=0) if ys else np.empty((0,), dtype='int64')
    c = np.concatenate(cs, axis=0) if cs else np.empty((0,3), dtype='int64')
    accept = np.ones(len(y), dtype=bool)
    scores = np.zeros(len(y), dtype='float32')
    if real_graph_desc is not None and len(Gd):
        scores = mahalanobis_scores(Gd, real_graph_desc).astype('float32')
        accept = scores <= float(cfg['training']['mahalanobis_threshold'])
    save_npz(os.path.join(out_dir, 'synthetic_samples.npz'), X=X, graph_desc=Gd, y=y, cti_ids=c, scores=scores, accepted=accept)
    save_npz(os.path.join(out_dir, 'synthetic_samples_filtered.npz'), X=X[accept], graph_desc=Gd[accept], y=y[accept], cti_ids=c[accept], scores=scores[accept])
    return X[accept], Gd[accept], y[accept], c[accept]
