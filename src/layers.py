from torch_geometric.nn import HypergraphConv
import torch.nn as nn
import torch
from torch_geometric.nn import HypergraphConv
from torch_geometric.utils import softmax 
from torch_geometric.nn.norm import GraphNorm

class Classifier(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.0):
        super(Classifier, self).__init__()

        self.sequential = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_channels, hidden_channels),
            nn.LeakyReLU(),
            nn.Linear(hidden_channels, out_channels)
        )
    
    def forward(self, x):
        return self.sequential(x)

class StructuralFeatureRefiner(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.0, activation: nn.Module = nn.LeakyReLU()):
        super(StructuralFeatureRefiner, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        self.linear = nn.Linear(in_channels, hidden_channels)
        self.norm = GraphNorm(hidden_channels)

        self.hgcn1 = HypergraphConv(hidden_channels, hidden_channels, use_attention=False)
        self.graph_norm1 = GraphNorm(hidden_channels)
        self.skip1 = nn.Linear(hidden_channels, hidden_channels)

        self.hgcn2 = HypergraphConv(hidden_channels, hidden_channels, use_attention=False, heads=4, attention_mode="node", concat=False)
        self.graph_norm2 = GraphNorm(hidden_channels)
        self.skip2 = nn.Linear(hidden_channels, hidden_channels)
    
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

        res2 = x
        x = self.hgcn2(x, edge_index)
        x = self.activation(x)
        x = self.graph_norm2(x)
        x = x + self.skip2(res2)

        return x

class SemanticFeatureRefiner(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.0, activation: nn.Module = nn.LeakyReLU()):
        super(SemanticFeatureRefiner, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        self.linear = nn.Linear(in_channels, hidden_channels)
        self.norm = GraphNorm(hidden_channels)

        self.hgcn1 = HypergraphConv(hidden_channels, hidden_channels, use_attention=False)
        self.graph_norm1 = GraphNorm(hidden_channels)
        self.skip1 = nn.Linear(hidden_channels, hidden_channels)

        self.hgcn2 = HypergraphConv(hidden_channels, hidden_channels, use_attention=False, heads=4, attention_mode="node", concat=False)
        self.graph_norm2 = GraphNorm(hidden_channels)
        self.skip2 = nn.Linear(hidden_channels, hidden_channels)
    
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

        res2 = x
        x = self.hgcn2(x, edge_index)
        x = self.activation(x)
        x = self.graph_norm2(x)
        x = x + self.skip2(res2)

        return x

class SemanticHyperedgeRefiner(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: int = 0.0, activation: nn.Module = nn.LeakyReLU()):
        super(SemanticHyperedgeRefiner, self).__init__()
        
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        self.linear_edge = nn.Linear(in_channels, hidden_channels)
        self.norm_edge = GraphNorm(hidden_channels)

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