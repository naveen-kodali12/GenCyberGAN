import torch
import torch.nn as nn

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)
    def forward(self, h):
        a = torch.softmax(self.score(h), dim=1)
        return (a * h).sum(dim=1)

class IDSClassifier(nn.Module):
    def __init__(self, feature_dim, num_classes, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(feature_dim, hidden_dim, 3, padding=1), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=2, dilation=2), nn.ReLU())
        self.gru = nn.GRU(hidden_dim, hidden_dim//2, batch_first=True, bidirectional=True)
        self.attn = AttentionPooling(hidden_dim)
        self.cls = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))
    def forward(self, x):
        h = self.tcn(x.transpose(1,2)).transpose(1,2)
        h, _ = self.gru(h)
        return self.cls(self.attn(h))
