import os, csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from .models.ids_model import IDSClassifier
from .utils import ensure_dir


def train_ids_model(X_train, y_train, X_val, y_val, cfg, out_dir, device, tag='baseline'):
    ensure_dir(out_dir)
    num_classes = int(max(np.max(y_train), np.max(y_val))) + 1
    model = IDSClassifier(X_train.shape[2], num_classes, cfg['model']['hidden_dim']).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['training']['lr_ids'])
    ce = nn.CrossEntropyLoss()
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_x = torch.tensor(X_val, dtype=torch.float32, device=device); val_y = torch.tensor(y_val, dtype=torch.long, device=device)
    loader = DataLoader(train_ds, batch_size=cfg['training']['batch_size'], shuffle=True)
    best = -1.0
    for epoch in range(1, cfg['training']['ids_epochs']+1):
        model.train(); losses=[]
        for xb, yb in tqdm(loader, desc=f'IDS {tag} epoch {epoch}', leave=False):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = ce(model(xb), yb); loss.backward(); opt.step(); losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            pred = model(val_x).argmax(1); acc = (pred == val_y).float().mean().item()
        if acc > best:
            best = acc
            torch.save({'model': model.state_dict(), 'feature_dim': X_train.shape[2], 'num_classes': num_classes, 'cfg': cfg}, os.path.join(out_dir, f'ids_{tag}.pt'))
    return model
