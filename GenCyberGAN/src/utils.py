import os, json, random, yaml
import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device(name='auto'):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_npz(path, **arrays):
    ensure_dir(os.path.dirname(path) or '.')
    np.savez_compressed(path, **arrays)


def load_npz(path):
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}
