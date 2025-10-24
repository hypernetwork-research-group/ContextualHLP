import torch.nn as nn
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score
from .utils import sensivity_specificity_cutoff, alpha_beta_negative_sampling
from pytorch_lightning import LightningModule
from torchmetrics.aggregation import RunningMean
from torch_geometric.nn import MinAggregation
from dhg.models import HNHN as DHG_HNHN, HyperGCN as HyGCN
from dhg import Hypergraph

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

    def forward(self, x, x_struct, x_e, edge_index):
        x = self.dropout(x)
        x = self.activation(self.linear(x))
        x = self.norm(x)

        x = self.aggr(x[edge_index[0]], edge_index[1])
        x = self.mlp(x)

        return x

class HNHN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.encoder = DHG_HNHN(in_channels, hidden_channels, num_classes=hidden_channels, use_bn=True)
        self.aggr = MinAggregation()
        self.score_fn = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, out_channels),
        )

    def forward(self, x, x_e, x_struct, edge_index):
        node_idx, hyperedge_idx = edge_index
        num_nodes = x.size(0)
        num_hyperedges = int(hyperedge_idx.max().item() + 1)

        e_list = [[] for _ in range(num_hyperedges)]
        for n, e in zip(node_idx.tolist(), hyperedge_idx.tolist()):
            e_list[e].append(n)
        hg = Hypergraph(num_v=num_nodes, e_list=e_list)
        
        node_emb = self.encoder(x, hg)

        x = self.aggr(node_emb[node_idx], hyperedge_idx)
        x = self.score_fn(x)

        return x

class HyperGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()

        self.encoder = HyGCN(
            in_channels,
            hidden_channels,
            num_classes=hidden_channels,
            use_bn=True,
        )

        self.aggr = MinAggregation()
        self.score_fn = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, out_channels),
        )

    def forward(self, x, x_e, x_struct, edge_index):
        
        node_idx, hyperedge_idx = edge_index
        num_nodes = x.size(0)
        num_hyperedges = int(hyperedge_idx.max().item() + 1)

        e_list = [[] for _ in range(num_hyperedges)]
        for n, e in zip(node_idx.tolist(), hyperedge_idx.tolist()):
            e_list[e].append(n)

        hg = Hypergraph(num_v=num_nodes, e_list=e_list)

        if hasattr(self.encoder, "cached_g"):
            self.encoder.cached_g = None

        node_emb = self.encoder(x, hg)

        x = self.aggr(node_emb[node_idx], hyperedge_idx)
        x = self.score_fn(x)

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
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
