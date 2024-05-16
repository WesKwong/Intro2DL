import os

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as transfomrs


class GraphDataset(object):
    _name = None
    _dataset = None

    def __init__(self, path):
        data_path = os.path.join(path, self._name)
        self.gen_data(data_path)

    def __str__(self) -> str:
        num_nodes = self._dataset._data.num_nodes
        # For num. edges see:
        # - https://github.com/pyg-team/pytorch_geometric/issues/343
        # - https://github.com/pyg-team/pytorch_geometric/issues/852
        num_edges = self._dataset._data.num_edges // 2
        train_len = self._dataset[0].train_mask.sum()
        val_len = self._dataset[0].val_mask.sum()
        test_len = self._dataset[0].test_mask.sum()
        other_len = num_nodes - train_len - val_len - test_len
        class_str = ""
        class_str += f"Dataset: {self._dataset.name}" + "\n"
        class_str += f"Num. nodes: {num_nodes} (train={train_len}, val={val_len}, test={test_len}, other={other_len})" + "\n"
        class_str += f"Num. edges: {num_edges}" + "\n"
        class_str += f"Num. node features: {self._dataset.num_node_features}" + "\n"
        class_str += f"Num. classes: {self._dataset.num_classes}" + "\n"
        class_str += f"Dataset len.: {self._dataset.len()}" + "\n"
        return class_str

    def __repr__(self) -> str:
        return self.__str__()

    def gen_data(self, path):
        self._dataset = Planetoid(path,
                                  self._name,
                                  transform=transfomrs.NormalizeFeatures())

    def get_dataset(self):
        return self._dataset

    def get_data(self):
        return self._dataset[0]

    def get_num_features(self):
        return self._dataset.num_node_features

    def get_num_classes(self):
        return self._dataset.num_classes

    def get_train_loader(self, batchsize):
        return self.get_data()

    def get_val_loader(self, batchsize):
        return self.get_dataset()

    def get_test_loader(self, batchsize):
        return None
