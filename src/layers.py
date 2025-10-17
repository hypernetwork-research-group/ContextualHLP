from torch_geometric.nn import HypergraphConv
import torch.nn as nn
import torch
from torch_geometric.utils import softmax 

class Classifier(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.0):
        super(Classifier, self).__init__()

        self.sequential = nn.Sequential(
            nn.Linear(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.sequential(x)

class StructuralFeatureRefiner(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.0, activation: nn.Module = nn.LeakyReLU()):
        super(StructuralFeatureRefiner, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        self.linear = nn.Linear(in_channels, hidden_channels)
        self.norm = nn.LayerNorm(hidden_channels)

        self.hgcn1 = HypergraphConv(hidden_channels, hidden_channels, use_attention=False)
        self.graph_norm1 = nn.LayerNorm(hidden_channels)
        self.skip1 = nn.Linear(hidden_channels, hidden_channels)
    
    def forward(self, x, edge_index):
        x = self.dropout(x)
        x = self.linear(x)
        x = self.activation(x)
        x = self.norm(x)

        res1 = x
        x = self.hgcn1(x, edge_index)
        x = self.activation(x)
        x = self.graph_norm1(x)
        x = x + self.skip1(res1)

        return x

class SemanticFeatureRefiner(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.0, activation: nn.Module = nn.LeakyReLU()):
        super(SemanticFeatureRefiner, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        self.linear = nn.Linear(in_channels, hidden_channels)
        self.norm = nn.LayerNorm(hidden_channels)

        self.hgcn1 = HypergraphConv(hidden_channels, hidden_channels, use_attention=False)
        self.graph_norm1 = nn.LayerNorm(hidden_channels)
        self.skip1 = nn.Linear(hidden_channels, hidden_channels)
    
    def forward(self, x, edge_index):
        x = self.dropout(x)
        x = self.linear(x)
        x = self.activation(x)
        x = self.norm(x)

        res1 = x
        x = self.hgcn1(x, edge_index)
        x = self.activation(x)
        x = self.graph_norm1(x)
        x = x + self.skip1(res1)

        return x

class SemanticHyperedgeRefiner(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: int = 0.0, activation: nn.Module = nn.LeakyReLU()):
        super(SemanticHyperedgeRefiner, self).__init__()
        
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        self.linear_edge = nn.Linear(in_channels, hidden_channels)
        self.norm_edge = nn.LayerNorm(hidden_channels)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear_edge(x)
        x = self.activation(x)
        x = self.norm_edge(x)

        return x

class NodeHyperedgeAttention(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=None, num_heads=4):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = node_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        self.q_proj = nn.Linear(node_dim, hidden_dim)
        self.k_proj = nn.Linear(edge_dim, hidden_dim)
        self.v_proj = nn.Linear(edge_dim, hidden_dim)

        self.out_proj = nn.Linear(hidden_dim, node_dim)

    def forward(self, x_node, x_edge, edge_index):
        src, dst = edge_index

        Q = self.q_proj(x_node)
        K = self.k_proj(x_edge)
        V = self.v_proj(x_edge)

        head_dim = self.hidden_dim // self.num_heads

        Q = Q.view(Q.size(0), self.num_heads, head_dim)
        K = K.view(K.size(0), self.num_heads, head_dim)
        V = V.view(V.size(0), self.num_heads, head_dim)

        Q_neighbors = Q[src]
        K_neighbors = K[dst]
        V_neighbors = V[dst]

        attn_scores = (Q_neighbors * K_neighbors).sum(dim=-1) / (head_dim ** 0.5)
        attn_weights = softmax(attn_scores, src) 

        attn_weights = attn_weights.unsqueeze(-1)
        out_messages = attn_weights * V_neighbors

        out_node = torch.zeros_like(Q)
        out_node.index_add_(0, src, out_messages)

        out_node = out_node.view(x_node.size(0), -1)
        out_node = self.out_proj(out_node)

        out_node = out_node + x_node
        return out_node

class FusionWithMHA(nn.Module):
    def __init__(self, dim, num_heads=4, num_layers=1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=num_heads,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out = nn.Linear(dim, 1)

    def forward(self, struct_emb, llm_node_emb, llm_edge_emb):
        x = torch.stack([struct_emb, llm_node_emb, llm_edge_emb], dim=1) # (batch, seq_len=3, dim)
        for layer in self.layers:
            x = layer(x)
        x_pooled = x.mean(dim=1)
        logit = self.out(x_pooled).squeeze(-1)
        return logit
