import torch
import numpy as np
from sklearn.metrics import roc_curve
from datasets import HyperGraphData

def sensivity_specificity_cutoff(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    idx = np.argmax(tpr - fpr)
    return thresholds[idx]

def negative_sampling(h: HyperGraphData):
    edge_index = h.edge_index.clone()
    edge_index[0] = torch.randint(0, h.num_nodes, (edge_index[1].shape[0], ), device=h.x.device)
    h_edge_attr = torch.vstack((h.edge_attr, h.edge_attr))
    h.y = torch.vstack((
        torch.ones((h.edge_index[1].max() + 1, 1), device=h.x.device),
        torch.zeros((edge_index[1].max() + 1, 1), device=h.x.device),
    ))
    edge_index[1] += edge_index[1].max() + 1
    edge_index = torch.hstack((h.edge_index, edge_index))
    h_ = HyperGraphData(
        x=h.x,
        edge_index=edge_index,
        edge_attr=h_edge_attr,
        y=h.y,
        num_nodes=h.num_nodes,
    )
    return h_

def pre_transform(data: HyperGraphData):
    data.edge_index = data.edge_index[:, torch.isin(data.edge_index[1], (data.edge_index[1].bincount() > 1).nonzero())]
    unique, inverse = data.edge_index[1].unique(return_inverse=True)
    data.edge_attr = data.edge_attr[unique]
    data.edge_index[1] = inverse
    return data

def transform(data: HyperGraphData):
    data.x = torch.rand_like(data.x)
    data.edge_attr = torch.rand_like(data.edge_attr)
    return data