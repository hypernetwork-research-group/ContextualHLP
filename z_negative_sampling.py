import torch
import numpy as np
from sklearn.metrics import roc_curve
from .datasets import CHLPBaseDataset, IMDBHypergraphDataset, COURSERAHypergraphDataset, ARXIVHypergraphDataset, PATENTHypergraphDataset, train_test_split, collate_fn, HyperGraphData, IMDBVillainHypergraphDataset, CourseraVillainHypergraphDataset, ArxivVillainHypergraphDataset, PatentVillainHypergraphDataset
from typing import Tuple
from torch.utils.data import DataLoader
import random
import torch.nn.functional as F

def sensivity_specificity_cutoff(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    idx = np.argmax(tpr - fpr)
    return thresholds[idx]

def negative_sampling(h: HyperGraphData):
    edge_index = h.edge_index.clone()
    edge_index[0] = torch.randint(0, h.num_nodes, (edge_index[1].shape[0], ), device=h.x.device)
    h_edge_attr = torch.vstack((h.edge_attr, h.edge_attr[torch.randperm(h.edge_attr.shape[0])]))
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
        y=h.y.flatten(),
        num_nodes=h.num_nodes,
    )
    return h_

import torch
import random
import torch.nn.functional as F
from torch_geometric.utils import degree

def alpha_beta_negative_sampling(h, alpha=0.8, beta=3, similarity='cosine', noise_std=0.05):
    device = h.x.device
    edge_index = h.edge_index.clone()
    real_edges = torch.arange(torch.max(edge_index[1]) + 1)
    real_nodes = torch.arange(torch.max(edge_index[0]) + 1)

    y = [1] * len(real_edges)
    fake_edges = []
    fake_attrs = []
    sample_edges = []

    fake_edge_id = real_edges[-1].item() + 1

    # Precompute edge-to-nodes mapping
    edge_to_nodes = {}
    for eid in real_edges:
        nodes = edge_index[0][edge_index[1] == eid]
        edge_to_nodes[eid.item()] = nodes

    if similarity == 'cosine':
        node_sim = F.normalize(h.x, p=2, dim=-1) @ F.normalize(h.x, p=2, dim=-1).T
    elif similarity == 'degree':
        deg = degree(edge_index[0], num_nodes=len(real_nodes))
        node_sim = -torch.abs(deg.unsqueeze(1) - deg.unsqueeze(0))
    else:
        node_sim = torch.ones(len(real_nodes), len(real_nodes), device=device)

    for eid in real_edges:
        nodes = edge_to_nodes[eid.item()]
        num_nodes = nodes.size(0)
        num_keep = max(1, round(alpha * num_nodes))
        num_replace = max(1, num_nodes - num_keep)

        for _ in range(beta):
            sample_edges.append(eid.item())

            perm = torch.randperm(num_nodes)
            keep_nodes = nodes[perm[:num_keep]]

            sim_scores = node_sim[keep_nodes].mean(dim=0)
            sim_scores[keep_nodes] = -float('inf')
            topk = torch.topk(sim_scores, k=min(num_replace, len(sim_scores))).indices

            if topk.numel() > num_replace:
                replace_nodes = topk[torch.randperm(topk.numel())[:num_replace]]
            else:
                replace_nodes = topk

            new_nodes = torch.cat([keep_nodes, replace_nodes])

            new_edge = torch.stack([
                new_nodes,
                torch.full_like(new_nodes, fake_edge_id)
            ], dim=0)
            fake_edges.append(new_edge)
            y.append(0)

            # Slightly perturb original edge_attr as fake attribute
            fake_attr = h.edge_attr[eid] #+ torch.randn_like(h.edge_attr[eid]) * noise_std
            fake_attrs.append(fake_attr)

            fake_edge_id += 1

    if fake_edges:
        fake_edge_index = torch.cat(fake_edges, dim=1)

        # Shuffle
        perm = torch.randperm(len(fake_attrs), device=device)
        fake_attr_tensor = torch.stack(fake_attrs, dim=0)[perm]

        final_edge_index = torch.cat([edge_index, fake_edge_index], dim=1)
        final_edge_attr = torch.cat([h.edge_attr, fake_attr_tensor], dim=0)
    else:
        final_edge_index = edge_index
        final_edge_attr = h.edge_attr

    h_ = h.__class__(
        x=h.x,
        edge_index=final_edge_index,
        edge_attr=final_edge_attr,
        y=torch.tensor(y, dtype=torch.float, device=device),
        num_nodes=real_nodes.shape[0],
        x_struct=h.x_struct if hasattr(h, "x_struct") else None
    ).to(device)

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
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    
    datasets = {
        "IMDB": IMDBHypergraphDataset,
        "COURSERA": COURSERAHypergraphDataset,
        "ARXIV": ARXIVHypergraphDataset,
        "PATENT": PATENTHypergraphDataset,
        "IMDB_VILLAIN": IMDBVillainHypergraphDataset,
        "COURSERA_VILLAIN": CourseraVillainHypergraphDataset,
        "ARXIV_VILLAIN": ArxivVillainHypergraphDataset,
        "PATENT_VILLAIN": PatentVillainHypergraphDataset,
    }

    if dataset_name not in datasets:
        raise NotImplementedError(f"Dataset '{dataset_name}' non supportato")

    transform_fn = None
    
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

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader, dataset.num_features