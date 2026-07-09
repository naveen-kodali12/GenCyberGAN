import torch
import torch.nn as nn

class CTIEmbedding(nn.Module):
    def __init__(self, vocab_sizes, emb_dim=32, cti_dim=64):
        super().__init__()
        self.tactic = nn.Embedding(max(1, vocab_sizes[0]), emb_dim)
        self.technique = nn.Embedding(max(1, vocab_sizes[1]), emb_dim)
        self.service = nn.Embedding(max(1, vocab_sizes[2]), emb_dim)
        self.mlp = nn.Sequential(nn.Linear(emb_dim*3, cti_dim), nn.ReLU(), nn.Linear(cti_dim, cti_dim), nn.ReLU())
    def forward(self, cti_ids):
        x = torch.cat([self.tactic(cti_ids[:,0]), self.technique(cti_ids[:,1]), self.service(cti_ids[:,2])], dim=1)
        return self.mlp(x)

class GenCyberGenerator(nn.Module):
    def __init__(self, vocab_sizes, z_dim, cti_emb_dim, cti_dim, hidden_dim, T, feature_dim, graph_desc_dim, dropout=0.1):
        super().__init__()
        self.T, self.feature_dim, self.z_dim = T, feature_dim, z_dim
        self.cti = CTIEmbedding(vocab_sizes, cti_emb_dim, cti_dim)
        self.temporal_in = nn.Linear(z_dim + cti_dim, hidden_dim)
        self.tcn = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2), nn.ReLU(), nn.Dropout(dropout)
        )
        self.bigru = nn.GRU(hidden_dim, hidden_dim//2, batch_first=True, bidirectional=True)
        self.flow_head = nn.Sequential(nn.Linear(hidden_dim + cti_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, feature_dim))
        self.graph_head = nn.Sequential(nn.Linear(z_dim + cti_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, graph_desc_dim))
    def forward(self, z_seq, z_graph, cti_ids):
        c = self.cti(cti_ids)
        B = z_seq.size(0)
        h0 = self.temporal_in(torch.cat([z_seq, c], dim=1)).unsqueeze(1).repeat(1, self.T, 1)
        h = self.tcn(h0.transpose(1,2)).transpose(1,2)
        h, _ = self.bigru(h)
        c_rep = c.unsqueeze(1).repeat(1, self.T, 1)
        x_hat = self.flow_head(torch.cat([h, c_rep], dim=2))
        g_raw = self.graph_head(torch.cat([z_graph, c], dim=1))
        # graph descriptors are non-negative for counts/variances; keep density-like terms bounded later by losses/filtering
        g_hat = torch.nn.functional.softplus(g_raw)
        return x_hat, g_hat
