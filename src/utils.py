import torch
import numpy as np
from sklearn.metrics import roc_curve
from .datasets import CHLPBaseDataset, IMDBHypergraphDataset, COURSERAHypergraphDataset, ARXIVHypergraphDataset, PROVAHypergraphDataset, PATENTHypergraphDataset, train_test_split, collate_fn, HyperGraphData
from typing import Tuple
from torch.utils.data import DataLoader
import random

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

def alpha_beta_negative_sampling(h: HyperGraphData, alpha=0.8, beta=3):
    edge_index = h.edge_index.clone()
    real_edges = torch.arange(torch.max(edge_index[1]) + 1)
    real_nodes = torch.arange(torch.max(edge_index[0]) + 1)
    
    y = [1] * len(real_edges)
    fake_edges = []
    sample_edges = []

    fake_edge_id = real_edges[-1].item() + 1

    edge_to_nodes = {}
    for eid in real_edges:
        nodes = edge_index[0][edge_index[1] == eid]
        edge_to_nodes[eid.item()] = nodes

    if alpha < 1.0:
        for eid in real_edges:
            nodes = edge_to_nodes[eid.item()]
            num_nodes = nodes.size(0)
            num_keep = max(1, round(alpha * num_nodes))
            num_replace = max(1, round((1 - alpha) * num_nodes))
            
            for _ in range(beta):
                sample_edges.append(eid.item())

                perm = torch.randperm(num_nodes)
                keep_nodes = nodes[perm[:num_keep]]

                mask = torch.ones(len(real_nodes), dtype=torch.bool)
                mask[keep_nodes] = False
                available = real_nodes[mask]

                if available.size(0) < num_replace:
                    replace_nodes = available
                else:
                    idx = torch.randperm(available.size(0))[:num_replace]
                    replace_nodes = available[idx]

                new_nodes = torch.cat([keep_nodes, replace_nodes.to(keep_nodes.device)])

                new_edge = torch.stack([
                    new_nodes,
                    torch.full_like(new_nodes, fake_edge_id)
                ], dim=0)
                fake_edges.append(new_edge)
                y.append(0)

                fake_edge_id += 1

        if fake_edges:
            fake_edge_index = torch.cat(fake_edges, dim=1)
            final_edge_index = torch.cat([edge_index, fake_edge_index], dim=1)
        else:
            final_edge_index = edge_index

        # indices = torch.tensor(random.sample(sample_edges, len(sample_edges)))
        final_edge_attr = torch.vstack([h.edge_attr, h.edge_attr[sample_edges]])

        assert final_edge_attr.size(0) == final_edge_index[1].max() + 1, "Mismatch between edge attributes and edge indices"
    else:
        final_edge_index = torch.cat([edge_index, edge_index], dim=1)
        # indices = torch.randperm(h.edge_attr.size(0))
        final_edge_attr = torch.vstack([h.edge_attr, h.edge_attr])

    h_ = HyperGraphData(
        x=h.x,
        edge_index=final_edge_index,
        edge_attr = final_edge_attr,
        y=torch.tensor(y, dtype=torch.float).flatten(),
        num_nodes=real_nodes.shape[0]
    ).to(h.x.device)

    return h_

def interpolate_negatives(positives, alpha_range=(0.3, 0.7)):
    idx_perm = torch.randperm(positives.size(0))
    alpha = torch.rand(positives.size(0), 1).to(positives.device)
    alpha = alpha * (alpha_range[1] - alpha_range[0]) + alpha_range[0]
    interpolated = positives * alpha + positives[idx_perm] * (1 - alpha)
    return interpolated

def negative_samping_mix(h: HyperGraphData, alpha=0.8, beta=1):
    edge_index = h.edge_index.clone()
    real_edges = torch.arange(torch.max(edge_index[1]) + 1)
    real_nodes = torch.arange(torch.max(edge_index[0]) + 1)
    
    y = [1] * len(real_edges)
    fake_edges = []
    sample_edges = []

    fake_edge_id = real_edges[-1].item() + 1

    edge_to_nodes = {}
    for eid in real_edges:
        nodes = edge_index[0][edge_index[1] == eid]
        edge_to_nodes[eid.item()] = nodes

    if alpha < 1.0:
        for eid in real_edges:
            nodes = edge_to_nodes[eid.item()]
            num_nodes = nodes.size(0)
            num_keep = max(1, round(alpha * num_nodes))
            num_replace = max(1, round((1 - alpha) * num_nodes))
            
            for _ in range(beta):
                sample_edges.append(eid.item())

                perm = torch.randperm(num_nodes)
                keep_nodes = nodes[perm[:num_keep]]

                mask = torch.ones(len(real_nodes), dtype=torch.bool)
                mask[keep_nodes] = False
                available = real_nodes[mask]

                if available.size(0) < num_replace:
                    replace_nodes = available
                else:
                    idx = torch.randperm(available.size(0))[:num_replace]
                    replace_nodes = available[idx]

                new_nodes = torch.cat([keep_nodes, replace_nodes.to(keep_nodes.device)])

                new_edge = torch.stack([
                    new_nodes,
                    torch.full_like(new_nodes, fake_edge_id)
                ], dim=0)
                fake_edges.append(new_edge)
                y.append(0)

                fake_edge_id += 1

        if fake_edges:
            fake_edge_index = torch.cat(fake_edges, dim=1)
            final_edge_index = torch.cat([edge_index, fake_edge_index], dim=1)
        else:
            final_edge_index = edge_index
    else:
        final_edge_index = torch.cat([edge_index, edge_index], dim=1)
    
    negative_embeddings = interpolate_negatives(h.edge_attr, alpha_range=(0.3, 0.7))
    final_edge_attr = torch.vstack([h.edge_attr, negative_embeddings])

    h_ = HyperGraphData(
        x=h.x,
        edge_index=final_edge_index,
        edge_attr = final_edge_attr,
        y=torch.tensor(y, dtype=torch.float).flatten(),
        num_nodes=real_nodes.shape[0]
    ).to(h.x.device)

    return h_

def negative_samping_mix_2(h: HyperGraphData, alpha=0.8, beta=1, semantic_perturb_frac=0.5):
    edge_index = h.edge_index.clone()
    real_edges = torch.arange(torch.max(edge_index[1]) + 1)
    real_nodes = torch.arange(torch.max(edge_index[0]) + 1)

    y = [1] * len(real_edges)
    fake_edges = []
    fake_attrs = []

    fake_edge_id = real_edges[-1].item() + 1

    edge_to_nodes = {}
    for eid in real_edges:
        nodes = edge_index[0][edge_index[1] == eid]
        edge_to_nodes[eid.item()] = nodes

    if alpha < 1.0:
        for eid in real_edges:
            nodes = edge_to_nodes[eid.item()]
            num_nodes = nodes.size(0)
            num_keep = max(1, round(alpha * num_nodes))
            num_replace = max(1, round((1 - alpha) * num_nodes))

            for _ in range(beta):
                perm = torch.randperm(num_nodes)
                keep_nodes = nodes[perm[:num_keep]]

                mask = torch.ones(len(real_nodes), dtype=torch.bool)
                mask[keep_nodes] = False
                available = real_nodes[mask]

                if available.size(0) < num_replace:
                    replace_nodes = available
                else:
                    idx = torch.randperm(available.size(0))[:num_replace]
                    replace_nodes = available[idx]

                new_nodes = torch.cat([keep_nodes, replace_nodes.to(keep_nodes.device)])

                new_edge = torch.stack([
                    new_nodes,
                    torch.full_like(new_nodes, fake_edge_id)
                ], dim=0)
                fake_edges.append(new_edge)
                fake_attrs.append(h.edge_attr[eid].unsqueeze(0))
                y.append(0)
                fake_edge_id += 1

    num_semantic_fakes = int(len(real_edges) * semantic_perturb_frac)
    idxs = torch.randperm(len(real_edges))[:num_semantic_fakes]
    interpolated_embeddings = interpolate_negatives(h.edge_attr, alpha_range=(0.3, 0.7))

    for i in idxs:
        eid = real_edges[i]
        nodes = edge_to_nodes[eid.item()]
        new_edge = torch.stack([
            nodes,
            torch.full_like(nodes, fake_edge_id)
        ], dim=0)
        fake_edges.append(new_edge)
        fake_attrs.append(interpolated_embeddings[i].unsqueeze(0))
        y.append(0)
        fake_edge_id += 1

    if fake_edges:
        fake_edge_index = torch.cat(fake_edges, dim=1)
        final_edge_index = torch.cat([edge_index, fake_edge_index], dim=1)
        fake_edge_attr = torch.cat(fake_attrs, dim=0)
        final_edge_attr = torch.vstack([h.edge_attr, fake_edge_attr])
    else:
        final_edge_index = edge_index
        final_edge_attr = h.edge_attr

    h_ = HyperGraphData(
        x=h.x,
        edge_index=final_edge_index,
        edge_attr=final_edge_attr,
        y=torch.tensor(y, dtype=torch.float).flatten(),
        num_nodes=real_nodes.shape[0]
    ).to(h.x.device)

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
        "PROVA": PROVAHypergraphDataset,
        "PATENT": PATENTHypergraphDataset
    }

    if dataset_name not in datasets:
        raise NotImplementedError(f"Dataset '{dataset_name}' non supportato")

    if mode == "baseline":
        transform_fn = transform_baseline
    elif mode == "nodes":
        transform_fn = transform_node_features
    elif mode == "edges":
        transform_fn = transform_edge_features
    elif mode == "full":
        transform_fn = None
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
