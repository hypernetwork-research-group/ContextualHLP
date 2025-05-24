import torch
import numpy as np
from sklearn.metrics import roc_curve
from datasets import HyperGraphData
from datasets import CHLPBaseDataset, IMDBHypergraphDataset, COURSERAHypergraphDataset, ARXIVHypergraphDataset, train_test_split, collate_fn
from typing import Tuple
from torch.utils.data import DataLoader

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

def transform_baseline(data: HyperGraphData):
    data.x = torch.rand_like(data.x)
    data.edge_attr = torch.rand_like(data.edge_attr)
    return data

def transform_node_features(data: HyperGraphData):
    data.edge_attr = torch.rand_like(data.edge_attr)
    return data

def transform_edge_features(data: HyperGraphData):
    data.x = torch.rand_like(data.x)
    return data

def load_and_prepare_data(
    dataset_name: str,
    mode: str,
    batch_size: int = 512,
    num_workers: int = 8
) -> Tuple[DataLoader, DataLoader, object, int]:
    
    datasets = {
        "IMDB": IMDBHypergraphDataset,
        "COURSERA": COURSERAHypergraphDataset,
        "ARXIV": ARXIVHypergraphDataset
    }

    if dataset_name not in datasets:
        raise NotImplementedError(f"Dataset '{dataset_name}' non supportato")

    if mode == "baseline":
        transform_fn = transform_baseline
    elif mode == "nodes":
        transform_fn = transform_node_features
    elif mode == "edges":
        transform_fn = transform_edge_features
    else:
        raise NotImplementedError(f"Mode '{mode}' non supportato")

    dataset = datasets[dataset_name](
        "./data",
        pre_transform=pre_transform,
        transform=transform_fn,
        force_reload=True
    )

    train_ds, test_ds, _, _, _, _ = train_test_split(dataset, test_size=0.3)
    train_ds, val_ds, _, _, _, _ = train_test_split(train_ds, test_size=0.3)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )

    return train_loader, val_loader, test_ds, dataset.num_features
