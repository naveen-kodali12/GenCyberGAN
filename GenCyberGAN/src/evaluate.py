import os, csv
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix, balanced_accuracy_score
import matplotlib.pyplot as plt
from .utils import ensure_dir


def predict_model(model, X, device):
    model.eval(); preds=[]
    with torch.no_grad():
        for i in range(0, len(X), 512):
            xb = torch.tensor(X[i:i+512], dtype=torch.float32, device=device)
            preds.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def evaluate_ids(model, X_test, y_test, out_dir, class_names=None, tag='baseline', device='cpu'):
    ensure_dir(out_dir)
    mask = y_test >= 0
    X_test = X_test[mask]; y_test = y_test[mask]
    y_pred = predict_model(model, X_test, device)
    acc = accuracy_score(y_test, y_pred)
    bal = balanced_accuracy_score(y_test, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
    macro = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
    with open(os.path.join(out_dir, f'results_{tag}.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['tag','accuracy','balanced_accuracy','precision_w','recall_w','f1_w','precision_macro','recall_macro','f1_macro'])
        w.writerow([tag, acc, bal, p, r, f1, macro[0], macro[1], macro[2]])
    labels = sorted(np.unique(np.concatenate([y_test, y_pred])).tolist())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(7,6)); plt.imshow(cm); plt.title(f'Confusion Matrix - {tag}'); plt.xlabel('Predicted'); plt.ylabel('True'); plt.colorbar(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'confusion_matrix_{tag}.png'), dpi=200); plt.close()
    names = class_names if class_names is not None else [str(i) for i in labels]
    report = classification_report(y_test, y_pred, zero_division=0)
    with open(os.path.join(out_dir, f'classification_report_{tag}.txt'), 'w') as f: f.write(report)
    return {'accuracy':acc,'balanced_accuracy':bal,'precision':p,'recall':r,'f1':f1,'macro_f1':macro[2]}
