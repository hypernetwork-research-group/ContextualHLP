import torch
import pickle
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.hypergraph_data import HyperGraphData
from typing import Callable

def train_test_split(dataset: InMemoryDataset, test_size: float = 0.2):
    indices = torch.randperm(len(dataset), device=dataset.x.device)
    split = int(len(dataset) * (1 - test_size))
    train_indices = torch.sort(indices[:split]).values
    test_indices = torch.sort(indices[split:]).values
    train_dataset = dataset[train_indices]
    test_dataset = dataset[test_indices]
    train_mask = torch.zeros(len(dataset), dtype=torch.bool, device=dataset.x.device)
    train_mask[train_indices] = True
    test_mask = torch.zeros(len(dataset), dtype=torch.bool, device=dataset.x.device)
    test_mask[test_indices] = True
    return train_dataset, test_dataset, train_indices, test_indices, train_mask, test_mask

def collate_fn(batch):
    x = batch[0].x
    edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.empty((0, batch[0].edge_attr.shape[1]), dtype=torch.long)
    for i in range(len(batch)):
        b = batch[i]
        b_edge_index = b.edge_index
        b_edge_index[1] += i
        edge_index = torch.hstack((edge_index, b_edge_index))
        b_edge_attr = b.edge_attr
        edge_attr = torch.vstack((edge_attr, b_edge_attr))
    unique, edge_index[0] = edge_index[0].unique(return_inverse=True)
    result = HyperGraphData(
        x=x[unique],
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
    return result

from abc import ABC

class CHLPBaseDataset(InMemoryDataset, ABC):

    GDRIVE_ID = None
    DATASET_NAME = None

    def __init__(self,
                 root: str = 'data',
                 *args, **kwargs):
        super().__init__(root, *args, **kwargs)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return f"{self.root}/{self.DATASET_NAME}/raw"

    @property
    def processed_dir(self):
        return f"{self.root}/{self.DATASET_NAME}/processed"

    @property
    def raw_file_names(self):
        return ["hyperedges.txt", "node_features.txt", "hyperedge_features.txt", "hyperedge_embeddings.txt", "node_embeddings.txt"]

    @property
    def processed_file_names(self):
        return "processed.pt"

    def download(self):
        from os import listdir
        if len(listdir(self.raw_dir)) > 0:
            return
        from gdown import download
        archive_file_name = self.raw_dir + "/" + "raw.zip"
        download(id=self.GDRIVE_ID, output=archive_file_name)
        import zipfile
        with zipfile.ZipFile(archive_file_name, 'r') as zip_ref:
            zip_ref.extractall(self.raw_dir)
        from os import remove
        remove(archive_file_name)

    def process(self):
        edge_index = [[], []]
        with open(self.raw_dir + "/hyperedges.txt", "r") as f:
            for i, line in enumerate(f):
                for l in line.split():
                    edge_index[0].append(int(l))
                    edge_index[1].append(i)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        with open(self.raw_dir + "/node_embeddings.pkl", "rb") as f:
            node_embeddings = torch.tensor(pickle.load(f))
        with open(self.raw_dir + "/hyperedge_embeddings.pkl", "rb") as f:
            hyperedge_embeddings = torch.tensor(pickle.load(f))

        data_list = [HyperGraphData(
            x=node_embeddings,
            edge_index=edge_index,
            edge_attr=hyperedge_embeddings,
        )]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])

    def __getitem__(self, idx) -> HyperGraphData:
        if isinstance(idx, int):
            idx = torch.tensor([idx])
        if isinstance(idx, list):
            idx = torch.tensor(idx)
        edge_index_mask = torch.isin(self._data.edge_index[1], idx)
        edge_index = self._data.edge_index[:, edge_index_mask]
        _, edge_index[1] = edge_index[1].unique(return_inverse=True)
        return HyperGraphData(
            x=self._data.x,
            edge_index=edge_index,
            edge_attr=self._data.edge_attr[idx],
        )

    def __len__(self) -> int:
        return self._data.edge_index[1].max().item() + 1

class IMDBHypergraphDataset(CHLPBaseDataset):
    
    GDRIVE_ID = "1D-dqEmkOfOVy6w0ZfrLtJrPw-dibwJ3V"
    DATASET_NAME = "IMDB"

class ARXIVHypergraphDataset(CHLPBaseDataset):
    
    GDRIVE_ID = "1XI4428OfnHlNaGpJN1BFPf5gKSpEEawh"
    DATASET_NAME = "ARXIV"

torch.serialization.add_safe_globals([HyperGraphData])
