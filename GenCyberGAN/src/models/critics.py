import torch
import torch.nn as nn

class FlowCritic(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(feature_dim, hidden_dim, 3, padding=1), nn.LeakyReLU(0.2), nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=2, dilation=2), nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.net(x.transpose(1,2)).view(-1)

class GraphCritic(nn.Module):
    def __init__(self, graph_desc_dim=8, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(graph_desc_dim, hidden_dim), nn.LeakyReLU(0.2), nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2), nn.Linear(hidden_dim, 1))
    def forward(self, g):
        return self.net(g).view(-1)

class CTIClassifier(nn.Module):
    def __init__(self, feature_dim, graph_desc_dim, num_labels, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feature_dim + graph_desc_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_labels))
    def forward(self, x, g):
        pooled = x.mean(dim=1)
        return self.net(torch.cat([pooled, g], dim=1))
