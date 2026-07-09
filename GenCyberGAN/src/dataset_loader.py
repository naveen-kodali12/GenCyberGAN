import numpy as np
import torch
from torch.utils.data import Dataset

class WindowDataset(Dataset):
    def __init__(self, X, graph_desc, cti_ids, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.graph_desc = torch.tensor(graph_desc, dtype=torch.float32)
        self.cti_ids = torch.tensor(cti_ids, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long) if y is not None else torch.zeros(len(X), dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.graph_desc[idx], self.cti_ids[idx], self.y[idx]
