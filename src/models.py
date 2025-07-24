from torch_geometric.nn import HypergraphConv, TransformerConv
import torch.nn as nn
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_score, roc_curve
import numpy as np
from .utils import negative_sampling, sensivity_specificity_cutoff, alpha_beta_negative_sampling
from pytorch_lightning import LightningModule
from torchmetrics.aggregation import RunningMean
from torch.nn.functional import normalize
from torch_geometric.nn import HypergraphConv, SoftmaxAggregation, MinAggregation, MulAggregation

# class NewModelSemantic(nn.Module):
#     def __init__(self, num_nodes, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int = 1):
#         super(NewModelSemantic, self).__init__()

#         self.dropout = nn.Dropout(0.3)
#         self.activation = nn.LeakyReLU()

#         self.n_sem_norm = nn.LayerNorm(in_channels)
#         self.n_sem_proj = nn.Sequential(
#             nn.Linear(in_channels, hidden_channels),
#         )

#         self.aggr = MinAggregation()
#         self.linear = nn.Linear(hidden_channels, out_channels)

#     def forward(self, x, x_e, edge_index):
#         x = normalize(x, p=2, dim=1)
#         x = self.n_sem_norm(x)
#         x = self.n_sem_proj(x)
#         x = self.activation(x)
        
#         x = self.aggr(x[edge_index[0]], edge_index[1])
#         x = self.linear(x)
#         return x

# class NewLLMn_LLMe(nn.Module):    
#     def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int = 1):
#         super(NewLLMn_LLMe, self).__init__()

#         self.dropout = nn.Dropout(0.3)
#         self.activation = nn.LeakyReLU()

#         self.in_proj = nn.Linear(in_channels, hidden_channels)
#         self.e_norm = nn.LayerNorm(in_channels)
#         self.e_proj = nn.Linear(in_channels, hidden_channels)

#         self.num_layers = num_layers

#         self.aggr = MinAggregation()
#         self.edge_fusion = nn.Linear(hidden_channels * 2, hidden_channels)
#         self.linear = nn.Linear(hidden_channels, out_channels)

#     def forward(self, x, x_e, edge_index):
#         x = torch.nn.functional.normalize(x, p=2, dim=1)
#         x = self.in_proj(x)
#         x = self.activation(x)
#         x = self.dropout(x)

#         x_e = torch.nn.functional.normalize(x_e, p=2, dim=1)
#         x_e = self.e_norm(x_e)
#         x_e = self.e_proj(x_e)
#         x_e = self.activation(x_e)
#         x_e = self.dropout(x_e)

#         x_aggr = self.aggr(x[edge_index[0]], edge_index[1])
#         x_e_fused = self.edge_fusion(torch.cat([x_aggr, x_e], dim=1))
#         x = self.linear(x_e_fused)
#         return x, x_aggr, x_e_fused

# class Struct_LLMn(nn.Module):
#     def __init__(self, num_nodes, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int = 1):
#         super(Struct_LLMn, self).__init__()

#         self.x_struct = torch.randn(num_nodes, in_channels)

#         self.dropout = nn.Dropout(0.3)
#         self.activation = nn.LeakyReLU()
#         self.in_norm = nn.LayerNorm(in_channels)
#         self.in_proj = nn.Linear(in_channels, hidden_channels)

#         self.n_sem_norm = nn.LayerNorm(in_channels)
#         self.n_sem_proj = nn.Linear(in_channels, hidden_channels)

#         for i in range(num_layers):
#             setattr(self, f"n_norm_{i}", nn.LayerNorm(hidden_channels))
#             setattr(self, f"hgconv_{i}", HypergraphConv(
#                 hidden_channels,
#                 hidden_channels,
#                 use_attention=False,
#                 concat=False,
#                 heads=4
#             ))
#             setattr(self, f"skip_struct_{i}", nn.Linear(hidden_channels, hidden_channels))

#         self.num_layers = num_layers
#         self.aggr = MinAggregation()
#         self.node_fusion = nn.Sequential(
#             nn.LayerNorm(hidden_channels * 2),
#             nn.Linear(hidden_channels * 2, hidden_channels),
#             nn.LeakyReLU(),
#             nn.LayerNorm(hidden_channels),
#         )
        
#         self.linear = nn.Sequential(
#             nn.Linear(hidden_channels, hidden_channels),
#             nn.LeakyReLU(),
#             nn.Linear(hidden_channels, hidden_channels),
#             nn.LeakyReLU(),
#             nn.Linear(hidden_channels, out_channels)
#         )

#     def forward(self, x, x_e, edge_index):
#         x_struct = self.x_struct.to(x.device)
#         x_struct = x_struct[torch.unique(edge_index[0])]
#         x_struct = normalize(x_struct, p=2, dim=1)
#         x_struct = self.in_proj(x_struct)
#         x_struct = self.activation(x_struct)
#         x_struct = self.dropout(x_struct)

#         x = normalize(x, p=2, dim=1)
#         x = self.n_sem_proj(x)
#         x = self.activation(x)
#         x = self.dropout(x)

#         for i in range(self.num_layers):
#             n_norm = getattr(self, f"n_norm_{i}")
#             hgconv = getattr(self, f"hgconv_{i}")
#             skip = getattr(self, f"skip_struct_{i}")

#             x_struct = n_norm(x_struct)
#             x_struct = self.activation(hgconv(x_struct, edge_index)) + skip(x_struct)
        
#         x = self.node_fusion(torch.cat([x_struct, x], dim=1))
#         x = self.aggr(x[edge_index[0]], edge_index[1])
#         x = self.linear(x)
#         return x

class ModelBaseline(nn.Module):    
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int = 1):
        super(ModelBaseline, self).__init__()

        self.dropout = nn.Dropout(0.3)
        self.activation = nn.LeakyReLU()
        self.in_norm = nn.LayerNorm(in_channels)
        self.in_proj = nn.Linear(in_channels, hidden_channels)

        for i in range(num_layers):
            setattr(self, f"n_norm_{i}", nn.LayerNorm(hidden_channels))
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
            setattr(self, f"n_norm_{i}_llm", nn.LayerNorm(hidden_channels))
            setattr(self, f"hgconv_{i}_llm", HypergraphConv(
                hidden_channels,
                hidden_channels,
                use_attention=False,
                concat=False,
                heads=4,
                attention_mode="node"
            ))
            setattr(self, f"skip_{i}", nn.Linear(hidden_channels, hidden_channels))

            setattr(self, f"n_norm_{i}_llm_d", nn.LayerNorm(hidden_channels))
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
            setattr(self, f"n_norm_{i}_llm", nn.LayerNorm(hidden_channels))
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
            setattr(self, f"n_norm_{i}", nn.LayerNorm(hidden_channels))
            setattr(self, f"hgconv_{i}", HypergraphConv(
                hidden_channels,
                hidden_channels,
                use_attention=False,
                concat=False,
                heads=4
            ))
            setattr(self, f"skip_struct_{i}", nn.Linear(hidden_channels, hidden_channels))

            setattr(self, f"n_norm_{i}_llm", nn.LayerNorm(hidden_channels))
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
            setattr(self, f"skip_struct_{i}", nn.Linear(hidden_channels, hidden_channels))

            setattr(self, f"n_norm_{i}_llm", nn.LayerNorm(hidden_channels))
            setattr(self, f"hgconv_{i}_llm", HypergraphConv(
                hidden_channels,
                hidden_channels,
                use_attention=False,
                concat=False,
                heads=4,
                attention_mode="node"
            ))
            setattr(self, f"skip_{i}", nn.Linear(hidden_channels, hidden_channels))

            setattr(self, f"n_norm_{i}_llm_d", nn.LayerNorm(hidden_channels))
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
            setattr(self, f"n_norm_{i}_llm", nn.LayerNorm(hidden_channels))
            setattr(self, f"hgconv_{i}_llm", HypergraphConv(
                hidden_channels,
                hidden_channels,
                use_attention=False,
                concat=False,
                heads=4,
                attention_mode="node"
            ))
            setattr(self, f"skip_{i}", nn.Linear(hidden_channels, hidden_channels))

            setattr(self, f"n_norm_{i}_llm_d", nn.LayerNorm(hidden_channels))
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
        y_pred = self.model(h.x, h.edge_attr, h.edge_index)
        y_pred = y_pred.flatten()
        bce_loss = self.criterion(y_pred, h.y.flatten())

        loss = bce_loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        h = alpha_beta_negative_sampling(batch)
        y_pred = self.model(h.x, h.edge_attr, h.edge_index)
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
        y_pred = self.model(h.x, h.edge_attr, h.edge_index)
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
