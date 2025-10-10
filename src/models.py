from torch_geometric.nn import HypergraphConv, TransformerConv
import torch_geometric
import torch.nn as nn
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_score, roc_curve
import numpy as np
from .utils import negative_sampling, sensivity_specificity_cutoff, alpha_beta_negative_sampling
from pytorch_lightning import LightningModule
from torchmetrics.aggregation import RunningMean
from torch.nn.functional import normalize
from torch_geometric.nn import HypergraphConv, MinAggregation, MeanAggregation, MaxAggregation
from torch_geometric.nn.norm import GraphNorm

class ModelBaseline(nn.Module):    
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int = 1):
        super(ModelBaseline, self).__init__()

        self.dropout = nn.Dropout(0.3)
        self.activation = nn.LeakyReLU()
        self.in_norm = nn.LayerNorm(in_channels)
        self.in_proj = nn.Linear(in_channels, hidden_channels)

        for i in range(num_layers):
            setattr(self, f"n_norm_{i}", GraphNorm(hidden_channels))
            setattr(self, f"hgconv_{i}", HypergraphConv(
                hidden_channels,
                hidden_channels,
                use_attention=False,
                concat=False,
                heads=1
            ))
        self.num_layers = num_layers

        self.aggr = MinAggregation()
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, x_e, edge_index):
        x = self.in_norm(x)
        x = self.in_proj(x)
        x = self.activation(x)
        x = self.dropout(x)

        for i in range(self.num_layers):
            n_norm = getattr(self, f"n_norm_{i}")
            hgconv = getattr(self, f"hgconv_{i}")
            x = n_norm(x)
            x = self.activation(hgconv(x, edge_index))
        
        x = self.aggr(x[edge_index[0]], edge_index[1])
        x = self.linear(x)
        return x

class ModelEdge(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int = 1):
        super(ModelEdge, self).__init__()

        self.dropout = nn.Dropout(0.3)
        self.activation = nn.LeakyReLU()

        self.in_norm = nn.LayerNorm(in_channels)
        self.in_proj = nn.Linear(in_channels, hidden_channels)
        self.e_norm = nn.LayerNorm(in_channels)
        self.e_proj = nn.Linear(in_channels, hidden_channels)

        for i in range(num_layers):
            setattr(self, f"n_norm_{i}_llm", GraphNorm(hidden_channels))
            setattr(self, f"hgconv_{i}_llm", HypergraphConv(
                hidden_channels,
                hidden_channels,
                use_attention=False,
                concat=False,
                heads=4,
                attention_mode="node"
            ))
            setattr(self, f"skip_{i}", nn.Linear(hidden_channels, hidden_channels))

            setattr(self, f"n_norm_{i}_llm_d", GraphNorm(hidden_channels))
            setattr(self, f"hgconv_{i}_llm_d", HypergraphConv(
                hidden_channels,
                hidden_channels,
                use_attention=False,
                concat=False,
                heads=4,
                attention_mode="node"
            ))
            setattr(self, f"skip_{i}_d", nn.Linear(hidden_channels, hidden_channels))

        self.num_layers = num_layers

        self.aggr = MinAggregation()
        self.edge_fusion = nn.Linear(hidden_channels * 2, hidden_channels)
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, x_e, edge_index):
        dual_edge_index = edge_index.flip(0)

        x = self.in_norm(x)
        x = self.in_proj(x)
        x = self.activation(x)
        x = self.dropout(x)

        x_e = torch.nn.functional.normalize(x_e, p=2, dim=1)
        x_e = self.e_norm(x_e)
        x_e = self.e_proj(x_e)
        x_e = self.activation(x_e)
        x_e = self.dropout(x_e)

        for i in range(self.num_layers):
            n_norm_llm = getattr(self, f"n_norm_{i}_llm")
            hgconv_llm = getattr(self, f"hgconv_{i}_llm")
            skip_llm = getattr(self, f"skip_{i}")

            n_norm_d = getattr(self, f"n_norm_{i}_llm_d")
            hgconv_llm_d = getattr(self, f"hgconv_{i}_llm_d")
            skip_llm_d = getattr(self, f"skip_{i}_d")

            x = n_norm_llm(x)
            x = self.activation(hgconv_llm(x, edge_index)) + skip_llm(x)

            x_e = n_norm_d(x_e)
            x_e = self.activation(hgconv_llm_d(x_e, dual_edge_index)) + skip_llm_d(x_e)

        x = self.aggr(x[edge_index[0]], edge_index[1])
        x = self.edge_fusion(torch.cat([x, x_e], dim=1))
        x = self.linear(x)
        return x

class ModelNodeSem(nn.Module):
    def __init__(self, num_nodes, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int = 1):
        super(ModelNodeSem, self).__init__()

        self.dropout = nn.Dropout(0.3)
        self.activation = nn.LeakyReLU()
        self.in_norm = nn.LayerNorm(in_channels)
        self.in_proj = nn.Linear(in_channels, hidden_channels)

        self.n_sem_norm = nn.LayerNorm(in_channels)
        self.n_sem_proj = nn.Linear(in_channels, hidden_channels)

        for i in range(num_layers):
            setattr(self, f"n_norm_{i}_llm", GraphNorm(hidden_channels))
            setattr(self, f"hgconv_{i}_llm", HypergraphConv(
                hidden_channels,
                hidden_channels,
                use_attention=False,
                concat=False,
                heads=1
            ))
        self.num_layers = num_layers

        self.aggr = MinAggregation()
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, x_e, edge_index):
        x = normalize(x, p=2, dim=1)
        x = self.n_sem_norm(x)
        x = self.n_sem_proj(x)
        x = self.activation(x)
        x = self.dropout(x)


        for i in range(self.num_layers):
            n_norm_llm = getattr(self, f"n_norm_{i}_llm")
            hgconv_llm = getattr(self, f"hgconv_{i}_llm")
            x = n_norm_llm(x)
            x = self.activation(hgconv_llm(x, edge_index))
        
        x = self.aggr(x[edge_index[0]], edge_index[1])
        x = self.linear(x)
        return x
    
class SemanticStructModel(nn.Module):
    def __init__(self, num_nodes, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int = 1):
        super(SemanticStructModel, self).__init__()

        self.x_struct = torch.randn(num_nodes, in_channels)

        self.dropout = nn.Dropout(0.3)
        self.activation = nn.LeakyReLU()
        self.in_norm = nn.LayerNorm(in_channels)
        self.in_proj = nn.Linear(in_channels, hidden_channels)

        self.e_norm = nn.LayerNorm(in_channels)
        self.e_proj = nn.Linear(in_channels, hidden_channels)

        self.n_sem_norm = nn.LayerNorm(in_channels)
        self.n_sem_proj = nn.Linear(in_channels, hidden_channels)

        for i in range(num_layers):
            setattr(self, f"n_norm_{i}", GraphNorm(hidden_channels))
            setattr(self, f"hgconv_{i}", HypergraphConv(
                hidden_channels,
                hidden_channels,
                use_attention=False,
                concat=False,
                heads=4
            ))
            setattr(self, f"skip_struct_{i}", nn.Linear(hidden_channels, hidden_channels))

            setattr(self, f"n_norm_{i}_llm", GraphNorm(hidden_channels))
            setattr(self, f"hgconv_{i}_llm", HypergraphConv(
                hidden_channels,
                hidden_channels,
                use_attention=False,
                concat=False,
                heads=4
            ))
            setattr(self, f"skip_{i}", nn.Linear(hidden_channels, hidden_channels))
        self.num_layers = num_layers
        self.aggr = MinAggregation()

        self.node_fusion = nn.Sequential(
            nn.LayerNorm(hidden_channels * 2),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_channels),
        )
        
        self.linear = nn.Sequential(
            # nn.Linear(hidden_channels, hidden_channels),
            # nn.LeakyReLU(),
            # nn.Linear(hidden_channels, hidden_channels),
            # nn.LeakyReLU(),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, x_e, edge_index):

        x_struct = self.x_struct.to(x.device)
        x_struct = x_struct[torch.unique(edge_index[0])]
        x_struct = normalize(x_struct, p=2, dim=1)
        x_struct = self.in_proj(x_struct)
        x_struct = self.activation(x_struct)
        x_struct = self.dropout(x_struct)

        x = normalize(x, p=2, dim=1)
        # x = self.n_sem_norm(x)
        x = self.n_sem_proj(x)
        x = self.activation(x)
        x = self.dropout(x)

        for i in range(self.num_layers):
            n_norm = getattr(self, f"n_norm_{i}")
            hgconv = getattr(self, f"hgconv_{i}")
            skip = getattr(self, f"skip_struct_{i}")

            n_norm_llm = getattr(self, f"n_norm_{i}_llm")
            hgconv_llm = getattr(self, f"hgconv_{i}_llm")
            skip_llm = getattr(self, f"skip_{i}")

            x_struct = n_norm(x_struct)
            x_struct = self.activation(hgconv(x_struct, edge_index)) + skip(x_struct)

            x = n_norm_llm(x)
            x = self.activation(hgconv_llm(x, edge_index)) + skip_llm(x)
        
        x = self.node_fusion(torch.cat([x_struct, x], dim=1))
        x = self.aggr(x[edge_index[0]], edge_index[1])
        x = self.linear(x)
        return x
    
class GatedFusion(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate_linear = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fused_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x_struct, x_sem):
        concat = torch.cat([x_struct, x_sem], dim=1)
        z = torch.sigmoid(self.gate_linear(concat))
        # fused = z * x_struct + (1 - z) * x_sem
        fused = self.fused_mlp(torch.cat([z * x_struct, (1 - z) * x_sem], dim=1))
        return fused

class FullModel(nn.Module):
    def __init__(self, num_nodes, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int = 1):
        super(FullModel, self).__init__()

        self.num_layers = num_layers
        self.dropout = nn.Dropout(0.3)
        self.activation = nn.LeakyReLU()

        self.x_struct = torch.randn(num_nodes, in_channels)

        self.in_proj = nn.Linear(in_channels, hidden_channels)
        self.e_proj = nn.Linear(in_channels, hidden_channels)
        self.n_sem_proj = nn.Linear(in_channels, hidden_channels)

        for i in range(num_layers):
            setattr(self, f"n_norm_{i}", nn.LayerNorm(hidden_channels))
            setattr(self, f"hgconv_{i}", HypergraphConv(
                hidden_channels,
                hidden_channels,
                use_attention=False,
                concat=False,
                heads=4,
                attention_mode="node"
            ))
            setattr(self, f"skip_struct_{i}", GraphNorm(hidden_channels, hidden_channels))

            setattr(self, f"n_norm_{i}_llm", GraphNorm(hidden_channels))
            setattr(self, f"hgconv_{i}_llm", HypergraphConv(
                hidden_channels,
                hidden_channels,
                use_attention=False,
                concat=False,
                heads=4,
                attention_mode="node"
            ))
            setattr(self, f"skip_{i}", nn.Linear(hidden_channels, hidden_channels))

            setattr(self, f"n_norm_{i}_llm_d", GraphNorm(hidden_channels))
            setattr(self, f"hgconv_{i}_llm_d", HypergraphConv(
                hidden_channels,
                hidden_channels,
                use_attention=False,
                concat=False,
                heads=4,
                attention_mode="node"
            ))
            setattr(self, f"skip_{i}_d", nn.Linear(hidden_channels, hidden_channels))

        self.aggr = MinAggregation()

        # self.node_fusion = GatedFusion(hidden_channels)
        self.node_fusion = nn.Sequential(
            nn.LayerNorm(hidden_channels * 2),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_channels),
        )
        # self.edge_fusion = GatedFusion(hidden_channels)
        self.edge_fusion = nn.Sequential(
            nn.LayerNorm(hidden_channels * 2),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_channels),
        )

        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, x_e, edge_index):
        dual_edge_index = edge_index.flip(0)

        x_struct = self.x_struct.to(x.device)
        x_struct = x_struct[torch.unique(edge_index[0])]

        x_struct = normalize(x_struct, p=2, dim=1)
        x_struct = self.in_proj(x_struct)
        x_struct = self.activation(x_struct)
        x_struct = self.dropout(x_struct)

        x = normalize(x, p=2, dim=1)
        x = self.n_sem_proj(x)
        x = self.activation(x)
        x = self.dropout(x)

        x_e = normalize(x_e, p=2, dim=1)
        x_e = self.e_proj(x_e)
        x_e = self.activation(x_e)
        x_e = self.dropout(x_e)

        for i in range(self.num_layers):
            n_norm = getattr(self, f"n_norm_{i}")
            hgconv = getattr(self, f"hgconv_{i}")
            skip = getattr(self, f"skip_struct_{i}")

            n_norm_llm = getattr(self, f"n_norm_{i}_llm")
            hgconv_llm = getattr(self, f"hgconv_{i}_llm")
            skip_llm = getattr(self, f"skip_{i}")

            n_norm_d = getattr(self, f"n_norm_{i}_llm_d")
            hgconv_llm_d = getattr(self, f"hgconv_{i}_llm_d")
            skip_llm_d = getattr(self, f"skip_{i}_d")

            x_struct = n_norm(x_struct)
            x_struct = self.activation(hgconv(x_struct, edge_index)) + skip(x_struct)

            x = n_norm_llm(x)
            x = self.activation(hgconv_llm(x, edge_index)) + skip_llm(x)

            x_e = n_norm_d(x_e + self.aggr(x[edge_index[0]], edge_index[1]))
            x_e = self.activation(hgconv_llm_d(x_e, dual_edge_index)) + skip_llm_d(x_e)

        # Fusion
        x_fused = self.node_fusion(torch.cat([x_struct, x], dim=1))
        x_aggr = self.aggr(x_fused[edge_index[0]], edge_index[1])
        x_e_fused = self.edge_fusion(torch.cat([x_aggr, x_e], dim=1))
        pred = self.linear(x_e_fused)
        return pred #, x_aggr, x_e_fused

class LLMNLLMEModel(nn.Module):
    def __init__(self, num_nodes, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int = 1):
        super(LLMNLLMEModel, self).__init__()

        self.dropout = nn.Dropout(0.3)
        self.activation = nn.LeakyReLU()

        self.in_norm = nn.LayerNorm(in_channels)
        self.in_proj = nn.Linear(in_channels, hidden_channels)
        self.e_norm = nn.LayerNorm(in_channels)
        self.e_proj = nn.Linear(in_channels, hidden_channels)

        for i in range(num_layers):
            setattr(self, f"n_norm_{i}_llm", GraphNorm(hidden_channels))
            setattr(self, f"hgconv_{i}_llm", HypergraphConv(
                hidden_channels,
                hidden_channels,
                use_attention=False,
                concat=False,
                heads=4,
                attention_mode="node"
            ))
            setattr(self, f"skip_{i}", nn.Linear(hidden_channels, hidden_channels))

            setattr(self, f"n_norm_{i}_llm_d", GraphNorm(hidden_channels))
            setattr(self, f"hgconv_{i}_llm_d", HypergraphConv(
                hidden_channels,
                hidden_channels,
                use_attention=False,
                concat=False,
                heads=4,
                attention_mode="node"
            ))
            setattr(self, f"skip_{i}_d", nn.Linear(hidden_channels, hidden_channels))

        self.num_layers = num_layers

        self.aggr = MinAggregation()
        self.edge_fusion = nn.Linear(hidden_channels * 2, hidden_channels)
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, x_e, edge_index):
        dual_edge_index = edge_index.flip(0)

        x = self.in_norm(x)
        x = self.in_proj(x)
        x = self.activation(x)
        x = self.dropout(x)

        x_e = torch.nn.functional.normalize(x_e, p=2, dim=1)
        x_e = self.e_norm(x_e)
        x_e = self.e_proj(x_e)
        x_e = self.activation(x_e)
        x_e = self.dropout(x_e)

        for i in range(self.num_layers):
            n_norm_llm = getattr(self, f"n_norm_{i}_llm")
            hgconv_llm = getattr(self, f"hgconv_{i}_llm")
            skip_llm = getattr(self, f"skip_{i}")

            n_norm_d = getattr(self, f"n_norm_{i}_llm_d")
            hgconv_llm_d = getattr(self, f"hgconv_{i}_llm_d")
            skip_llm_d = getattr(self, f"skip_{i}_d")

            x = n_norm_llm(x)
            x = self.activation(hgconv_llm(x, edge_index)) + skip_llm(x)

            x_e = n_norm_d(x_e)
            x_e = self.activation(hgconv_llm_d(x_e, dual_edge_index)) + skip_llm_d(x_e)

        x = self.aggr(x[edge_index[0]], edge_index[1])
        x = self.edge_fusion(torch.cat([x, x_e], dim=1))
        x = self.linear(x)
        return x
from torch_geometric.utils import softmax 

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

import pickle as pkl
from torch_geometric.nn.aggr import SetTransformerAggregation
class TestModel(nn.Module):
    def __init__(self, num_nodes, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.dropout = nn.Dropout(0.3)
        self.activation = nn.Tanh()

        #node
        self.linear_node = nn.Linear(in_channels, hidden_channels)
        self.norm_node = nn.LayerNorm(hidden_channels)
        self.hgcn_llm = HypergraphConv(hidden_channels, hidden_channels, use_attention=False)
        self.graph_norm_llm = GraphNorm(hidden_channels)

        self.hgcn_llm_2 = HypergraphConv(hidden_channels, hidden_channels, use_attention=False, heads=4, attention_mode="node", concat=False)
        self.graph_norm_llm_2 = GraphNorm(hidden_channels)

        #edge
        self.linear_edge = nn.Linear(in_channels, hidden_channels)
        self.norm_edge = nn.LayerNorm(hidden_channels)

        #struct
        self.linear_struct = nn.Linear(768, hidden_channels)
        self.norm_struct = nn.LayerNorm(hidden_channels)
        self.hgcn1 = HypergraphConv(hidden_channels, hidden_channels, use_attention=False)
        self.graph_norm1 = GraphNorm(hidden_channels)

        self.hgcn2 = HypergraphConv(hidden_channels, hidden_channels, use_attention=False, heads=4, attention_mode="node", concat=False)
        self.graph_norm2 = GraphNorm(hidden_channels)

        # Attention and classification
        self.attention_fusion = NodeHyperedgeAttention(hidden_channels, hidden_channels)
        self.aggr_min = SetTransformerAggregation(hidden_channels, heads=4)

        self.node_fusion = nn.Linear(hidden_channels * 2, hidden_channels)
        self.norm_fusion = GraphNorm(hidden_channels)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.LeakyReLU(),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, x_struct, x_e, edge_index):
        #Refine node features
        x = self.dropout(x)
        x = self.linear_node(x)
        x = self.activation(x)
        x = self.norm_node(x)

        #Refine edge features
        x_e = self.dropout(x_e)
        x_e = self.linear_edge(x_e)
        x_e = self.activation(x_e)
        x_e = self.norm_edge(x_e)

        #on node llm
        res_llm = x
        x = self.hgcn_llm(x, edge_index)
        x = self.activation(x)
        x = self.graph_norm_llm(x) + res_llm

        res_llm_2 = x
        x = self.hgcn_llm_2(x, edge_index, hyperedge_attr=x_e)
        x = self.activation(x)
        x = self.graph_norm_llm_2(x) + res_llm_2

        #Refine struct
        x_struct = self.dropout(x_struct)
        x_struct = self.linear_struct(x_struct)
        x_struct = self.activation(x_struct)
        x_struct = self.norm_struct(x_struct)

        #Start
        res = x_struct
        x_struct = self.hgcn1(x_struct, edge_index)
        x_struct = self.activation(x_struct)
        x_struct = self.graph_norm1(x_struct)
        x_struct = x_struct + res

        res2 = x_struct
        x_struct = self.hgcn2(x_struct, edge_index, hyperedge_attr=x_e)
        x_struct = self.activation(x_struct)
        x_struct = self.graph_norm2(x_struct)
        x_struct = x_struct + res2

        #node fusion
        x = torch.cat([x, x_struct], dim=1)
        x = self.node_fusion(x)
        x = self.activation(x)
        x = self.norm_fusion(x)
        x = self.attention_fusion(x, x_e, edge_index)
        
        #Classification
        x = self.aggr_min(x[edge_index[0]], edge_index[1])
        x = torch.cat([x, x_e], dim=1)
        x = self.dropout(x)
        x = self.mlp(x)

        return x

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.dropout = nn.Dropout(0.3)
        self.activation = nn.LeakyReLU()
        self.aggr = MinAggregation()

        self.norm = nn.LayerNorm(hidden_channels)
        self.linear = nn.Linear(in_channels, hidden_channels)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, x_e, edge_index):
        x = self.dropout(x)
        x = self.activation(self.linear(x))
        x = self.norm(x)

        x = self.aggr(x[edge_index[0]], edge_index[1])
        x = self.mlp(x)

        return x

class LitCHLPModel(LightningModule):
    def __init__(self, model, lr=1e-4):
        super().__init__()
        self.model = model
        self.lr = lr
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.cutoff = None
        self.val_preds = []
        self.val_targets = []
        self.test_preds = []
        self.test_targets = []
        self.metric = RunningMean(window=11)

    def training_step(self, batch, batch_idx):
        h = alpha_beta_negative_sampling(batch)
        y_pred = self.model(h.x, h.x_struct, h.edge_attr, h.edge_index)
        y_pred = y_pred.flatten()
        bce_loss = self.criterion(y_pred, h.y.flatten())

        loss = bce_loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        h = alpha_beta_negative_sampling(batch)
        y_pred = self.model(h.x, h.x_struct, h.edge_attr, h.edge_index)
        y_pred = y_pred.flatten()
        bce_loss = self.criterion(y_pred, h.y.flatten())

        loss = bce_loss
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        
        self.val_preds.append(y_pred.detach().cpu())
        self.val_targets.append(h.y.detach().cpu())

    def on_validation_epoch_end(self):
        y_pred = torch.cat(self.val_preds)
        y_true = torch.cat(self.val_targets)
        loss = self.criterion(y_pred, y_true).item()
        y_pred = torch.sigmoid(y_pred)

        cutoff = sensivity_specificity_cutoff(y_true.numpy(), y_pred.numpy())
        self.cutoff = cutoff
        
        y_true = y_true.numpy()
        y_pred = y_pred.numpy()
        self.log("epoch_val_loss", loss, prog_bar=False, on_epoch=True, logger=True)

        self.log("val_roc_auc", roc_auc_score(y_true, y_pred), prog_bar=False, logger=True)
        self.log("val_accuracy", accuracy_score(y_true, (y_pred >= cutoff).astype(int)), prog_bar=False, logger=True)
        self.log("val_precision", precision_score(y_true, (y_pred >= cutoff).astype(int), average='macro'), prog_bar=False, logger=True)
        self.metric(loss)
        running_val = self.metric.compute()
        self.log("running_val", running_val, on_epoch=True, logger=True)
        self.val_preds.clear()
        self.val_targets.clear()

    
    def test_step(self, batch, batch_idx):
        h = alpha_beta_negative_sampling(batch)
        y_pred = self.model(h.x, h.x_struct, h.edge_attr, h.edge_index)
        y_pred = torch.sigmoid(y_pred).flatten()

        self.test_preds.append(y_pred.detach().cpu())
        self.test_targets.append(h.y.detach().cpu())
    
    def on_test_epoch_end(self):
        y_pred = torch.cat(self.test_preds).numpy()
        y_true = torch.cat(self.test_targets).numpy()

        cutoff = self.cutoff if self.cutoff is not None else 0.5

        roc_auc = roc_auc_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, (y_pred >= cutoff).astype(int))
        precision = precision_score(y_true, (y_pred >= cutoff).astype(int), average='macro')

        self.log("test_roc_auc", roc_auc, prog_bar=False)
        self.log("test_accuracy", accuracy, prog_bar=False)
        self.log("test_precision", precision, prog_bar=False)

        print(f"Test ROC AUC: {roc_auc:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test Precision: {precision:.4f}")

        self.test_preds.clear()
        self.test_targets.clear()

        return {
            "test_roc_auc": roc_auc,
            "test_accuracy": accuracy,
            "test_precision": precision,
            "test_cutoff": cutoff,
        }

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
