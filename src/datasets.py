import torch
import pickle
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.hypergraph_data import HyperGraphData
from typing import Callable

class SplitDataset(InMemoryDataset):

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
            x_struct=self._data.x_struct
        )

    def __len__(self) -> int:
        return self._data.edge_index[1].max().item() + 1

def train_test_split(dataset: InMemoryDataset, test_size: float = 0.2):
    indices = torch.randperm(len(dataset), device=dataset.x.device)
    split = int(len(dataset) * (1 - test_size))
    train_indices = torch.sort(indices[:split]).values
    test_indices = torch.sort(indices[split:]).values
    train_data = dataset[train_indices]
    class TrainDataset(SplitDataset):
        def __init__(self):
            super().__init__()
            self._data = train_data
    train_dataset = TrainDataset()
    test_data = dataset[test_indices]
    class TestDataset(SplitDataset):
        def __init__(self):
            super().__init__()
            self._data = test_data
    test_dataset = TestDataset()
    train_mask = torch.zeros(len(dataset), dtype=torch.bool, device=dataset.x.device)
    train_mask[train_indices] = True
    test_mask = torch.zeros(len(dataset), dtype=torch.bool, device=dataset.x.device)
    test_mask[test_indices] = True
    return train_dataset, test_dataset, train_indices, test_indices, train_mask, test_mask

def collate_fn(batch):
    x = batch[0].x
    x_struct = batch[0].x_struct
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
    result.x_struct = x_struct[unique]
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

        # with open(self.raw_dir + "/eigvec.pkl", "rb") as f:
        #     x_struct = torch.tensor(pickle.load(f))
        x_struct = torch.randn(node_embeddings.shape[0], 768)
        data = HyperGraphData(
            x=node_embeddings,
            edge_index=edge_index,
            edge_attr=hyperedge_embeddings,
            x_struct=x_struct
        )
        data_list = [data]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        if self.transform is not None:
            data_list = [self.transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])

    def __getitem__(self, idx) -> HyperGraphData:
        if isinstance(idx, int):
            idx = torch.tensor([idx])
        if isinstance(idx, list):
            idx = torch.tensor(idx)
        edge_index_mask = torch.isin(self._data.edge_index[1], idx)
        edge_index = self._data.edge_index[:, edge_index_mask]
        _, edge_index[1] = edge_index[1].unique(return_inverse=True)
        
        data = HyperGraphData(
            x=self._data.x,
            edge_index=edge_index,
            edge_attr=self._data.edge_attr[idx],
            x_struct=self._data.x_struct
        )

        return data

    def __len__(self) -> int:
        return self._data.edge_index[1].max().item() + 1

class ARXIVHypergraphDataset(CHLPBaseDataset):
    
    GDRIVE_ID = "6q0TarVsnFi3wCchLvvc9iNe8kILwhX6"
    DATASET_NAME = "ARXIV"

class COURSERAHypergraphDataset(CHLPBaseDataset):
    
    GDRIVE_ID = "1h_Pe3ATRlXBt2Zhy6Bffb7RcCkk3PgfR"
    DATASET_NAME = "COURSERA"

class IMDBHypergraphDataset(CHLPBaseDataset):
    GDRIVE_ID = "1ghUYiyNDvbSF4VKhgMS0akjSL0Wei0uU"
    DATASET_NAME = "IMDB"

class IMDBHypergraphDataset2(CHLPBaseDataset):
    GDRIVE_ID = "19vdcTseOsVUDJzANzqcblPFJnfEQpJUk"
    DATASET_NAME = "IMDB2"
class PATENTHypergraphDataset(CHLPBaseDataset):
    GDRIVE_ID = "17FZAsGdMQMRZCWF6Vjjbz4F_NObwR3Cr"
    DATASET_NAME = "PATENT"

class IMDBVillainHypergraphDataset(CHLPBaseDataset):
    # GDRIVE_ID = "1h_Pe3ATRlXBt2Zhy6Bffb7RcCkk3PgfR"
    DATASET_NAME = "IMDB_VILLAIN"

class CourseraVillainHypergraphDataset(CHLPBaseDataset):
    # GDRIVE_ID = "1h_Pe3ATRlXBt2Zhy6Bffb7RcCkk3PgfR"
    DATASET_NAME = "COURSERA_VILLAIN"

class ArxivVillainHypergraphDataset(CHLPBaseDataset):
    # GDRIVE_ID = "1h_Pe3ATRlXBt2Zhy6Bffb7RcCkk3PgfR"
    DATASET_NAME = "ARXIV_VILLAIN"

class PatentVillainHypergraphDataset(CHLPBaseDataset):
    # GDRIVE_ID = "1h_Pe3ATRlXBt2Zhy6Bffb7RcCkk3PgfR"
    DATASET_NAME = "PATENT_VILLAIN"
    
torch.serialization.add_safe_globals([HyperGraphData])
