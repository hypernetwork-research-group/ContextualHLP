from torch_geometric.nn import HypergraphConv
import torch.nn as nn
from torch_geometric.nn.aggr import MeanAggregation
import lightning as L
from datasets import HyperGraphData
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_score, roc_curve
import numpy as np
from utils import negative_sampling, sensivity_specificity_cutoff
from pytorch_lightning import LightningModule

class Model(nn.Module):    
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int = 1):
        super(Model, self).__init__()

        self.dropout = nn.Dropout(0.3)
        self.activation = nn.LeakyReLU()
        self.in_norm = nn.LayerNorm(in_channels)
        self.in_proj = nn.Linear(in_channels, hidden_channels)
        self.e_proj = nn.Linear(in_channels, hidden_channels)
        self.e_norm = nn.LayerNorm(in_channels)

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

        self.aggr = MeanAggregation()
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, x_e, edge_index):
        x = self.in_norm(x)
        x = self.in_proj(x)
        x = self.activation(x)
        x = self.dropout(x)

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

class LitCHLPModel(LightningModule):
    def __init__(self, model, lr=1e-4):
        super().__init__()
        self.model = model
        self.lr = lr
        self.criterion = torch.nn.BCEWithLogitsLoss()
    
    def training_step(self, batch, batch_idx):
        h = negative_sampling(batch)
        y_pred = self.model(h.x, h.edge_attr, h.edge_index)
        loss = self.criterion(y_pred, h.y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        h = negative_sampling(batch)
        y_pred = torch.sigmoid(self.model(h.x, h.edge_attr, h.edge_index))
        loss = self.criterion(y_pred, h.y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        y_true = h.y.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()
        cutoff = sensivity_specificity_cutoff(y_true, y_pred_np)
        self.log("val_roc_auc", roc_auc_score(y_true, y_pred_np), prog_bar=True)
        self.log("val_accuracy", accuracy_score(y_true, (y_pred_np >= cutoff).astype(int)), prog_bar=True)
        self.log("val_precision", precision_score(y_true, (y_pred_np >= cutoff).astype(int), average='micro'), prog_bar=True)
        self.log("val_precision", precision_score(y_true, (y_pred_np >= cutoff).astype(int), average='micro'), prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)