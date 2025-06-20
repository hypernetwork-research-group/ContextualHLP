from torch_geometric.nn import HypergraphConv
import torch.nn as nn
from torch_geometric.nn.aggr import MeanAggregation, SoftmaxAggregation
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_score, roc_curve
import numpy as np
from .utils import negative_sampling, sensivity_specificity_cutoff, alpha_beta_negative_sampling, negative_samping_mix, negative_samping_mix_2
from pytorch_lightning import LightningModule
from torchmetrics.aggregation import RunningMean
from torch.nn.functional import normalize
from torch_geometric.nn import HypergraphConv, SoftmaxAggregation, MeanAggregation, MinAggregation

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
            setattr(self, f"n_norm_{i}", nn.LayerNorm(hidden_channels))
            setattr(self, f"e_norm_{i}", nn.LayerNorm(hidden_channels))
            setattr(self, f"hgconv_{i}", HypergraphConv(
                hidden_channels,
                hidden_channels,
                use_attention=True,
                concat=False,
                heads=1
            ))
            setattr(self, f"skip_{i}", nn.Linear(hidden_channels, hidden_channels))
        self.num_layers = num_layers

        self.aggr = MinAggregation()
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, x_e, edge_index):
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
            n_norm = getattr(self, f"n_norm_{i}")
            e_norm = getattr(self, f"e_norm_{i}")
            hgconv = getattr(self, f"hgconv_{i}")
            skip = getattr(self, f"skip_{i}")
            x = n_norm(x)
            x_e = e_norm(x_e)
            x = self.activation(hgconv(x, edge_index, hyperedge_attr=x_e)) + \
                skip(x)

        x = self.aggr(x[edge_index[0]], edge_index[1])
        x = self.linear(x)
        return x

class ModelNodeStruct(nn.Module):
    def __init__(self, num_nodes, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int = 1):
        super(ModelNodeStruct, self).__init__()

        self.x_struct = torch.randn(num_nodes, in_channels)

        self.dropout = nn.Dropout(0.3)
        self.activation = nn.LeakyReLU()
        self.in_norm = nn.LayerNorm(in_channels)
        self.in_proj = nn.Linear(in_channels, hidden_channels)

        self.n_sem_norm = nn.LayerNorm(in_channels)
        self.n_sem_proj = nn.Linear(in_channels, hidden_channels)

        for i in range(num_layers):
            # setattr(self, f"n_norm_{i}", nn.LayerNorm(hidden_channels))
            # setattr(self, f"hgconv_{i}", HypergraphConv(
            #     hidden_channels,
            #     hidden_channels,
            #     use_attention=False,
            #     concat=False,
            #     heads=1
            # ))
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
        # x_struct = self.x_struct.to(x.device)
        # x_struct = x_struct[torch.unique(edge_index[0])]
        # x_struct = normalize(x_struct, p=2, dim=1)
        # x_struct = self.in_norm(x_struct)
        # x_struct = self.in_proj(x_struct)
        # x_struct = self.activation(x_struct)
        # x_struct = self.dropout(x_struct)

        x = normalize(x, p=2, dim=1)
        x = self.n_sem_norm(x)
        x = self.n_sem_proj(x)
        x = self.activation(x)
        x = self.dropout(x)


        for i in range(self.num_layers):
            # n_norm = getattr(self, f"n_norm_{i}")
            # hgconv = getattr(self, f"hgconv_{i}")
            # x_struct = n_norm(x_struct)
            # x_struct = self.activation(hgconv(x_struct, edge_index))
            n_norm_llm = getattr(self, f"n_norm_{i}_llm")
            hgconv_llm = getattr(self, f"hgconv_{i}_llm")
            x = n_norm_llm(x)
            x = self.activation(hgconv_llm(x, edge_index))
        
        x = self.aggr(x[edge_index[0]], edge_index[1])
        x = self.linear(x)
        return x

class FullModel(nn.Module):
    def __init__(self, num_nodes, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int = 1):
        super(FullModel, self).__init__()

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
            setattr(self, f"n_norm_{i}_llm", nn.LayerNorm(hidden_channels))
            setattr(self, f"hgconv_{i}_llm", HypergraphConv(
                hidden_channels,
                hidden_channels,
                use_attention=True,
                concat=False,
                heads=4
            ))
        self.num_layers = num_layers
        self.aggr = MinAggregation()
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, x_e, edge_index):
        x_struct = self.x_struct.to(x.device)
        x_struct = x_struct[torch.unique(edge_index[0])]
        x_struct = normalize(x_struct, p=2, dim=1)

        x = normalize(x, p=2, dim=1)
        x = self.n_sem_norm(x)
        x = self.n_sem_proj(x)
        x = self.activation(x)
        x = self.dropout(x)

        x_e = normalize(x_e, p=2, dim=1)
        x_e = self.e_norm(x_e)
        x_e = self.e_proj(x_e)
        x_e = self.activation(x_e)
        x_e = self.dropout(x_e)

        x_struct = self.in_norm(x_struct)
        x_struct = self.in_proj(x_struct)
        x_struct = self.activation(x_struct)
        x_struct = self.dropout(x_struct)

        for i in range(self.num_layers):
            n_norm = getattr(self, f"n_norm_{i}")
            hgconv = getattr(self, f"hgconv_{i}")
            n_norm_llm = getattr(self, f"n_norm_{i}_llm")
            hgconv_llm = getattr(self, f"hgconv_{i}_llm")
            x_struct = n_norm(x_struct)
            x_struct = self.activation(hgconv(x_struct, edge_index))
            x = n_norm_llm(x)
            x = self.activation(hgconv_llm(x, edge_index, hyperedge_attr=x_e))
        
        x = x + x_struct
        x = self.aggr(x[edge_index[0]], edge_index[1])
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
        y_pred = self.model(h.x, h.edge_attr, h.edge_index).flatten()
        loss = self.criterion(y_pred, h.y.flatten())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        h = alpha_beta_negative_sampling(batch)
        y_pred = self.model(h.x, h.edge_attr, h.edge_index).flatten()
        loss = self.criterion(y_pred, h.y.flatten())
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
        self.log("val_precision", precision_score(y_true, (y_pred >= cutoff).astype(int), average='micro'), prog_bar=False, logger=True)
        self.metric(loss)
        running_val = self.metric.compute()
        self.log("running_val", running_val, on_epoch=True, logger=True)
        self.val_preds.clear()
        self.val_targets.clear()

    
    def test_step(self, batch, batch_idx):
        h = alpha_beta_negative_sampling(batch)
        y_pred = torch.sigmoid(self.model(h.x, h.edge_attr, h.edge_index).flatten())

        self.test_preds.append(y_pred.detach().cpu())
        self.test_targets.append(h.y.detach().cpu())
    
    def on_test_epoch_end(self):
        y_pred = torch.cat(self.test_preds).numpy()
        y_true = torch.cat(self.test_targets).numpy()

        cutoff = self.cutoff if self.cutoff is not None else 0.5

        roc_auc = roc_auc_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, (y_pred >= cutoff).astype(int))
        precision = precision_score(y_true, (y_pred >= cutoff).astype(int), average='micro')

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