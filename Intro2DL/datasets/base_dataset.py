import os
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader


class BaseDataset(ABC):
    name = None
    train_set = None
    val_set = None
    test_set = None
    train_loader = None
    val_loader = None
    test_loader = None

    def __init__(self, path) -> None:
        raw_data_path = os.path.join(path, self.name, "raw")
        split_data_path = os.path.join(path, self.name, "split")
        self.gen_raw_data(raw_data_path)
        self.split_train_data(raw_data_path, split_data_path)
        self.load_split_data(split_data_path)

    @abstractmethod
    def gen_raw_data(self, raw_path):
        raise NotImplementedError

    @abstractmethod
    def split_train_data(self, raw_path, split_path):
        raise NotImplementedError

    @abstractmethod
    def load_split_data(self, split_path):
        raise NotImplementedError

    def get_train_set(self):
        return self.train_set

    def get_val_set(self):
        return self.val_set

    def get_test_set(self):
        return self.test_set

    def get_train_loader(self, batchsize):
        return DataLoader(self.train_set, batch_size=batchsize, shuffle=True)

    def get_val_loader(self, batchsize):
        return DataLoader(self.val_set, batch_size=batchsize, shuffle=False)

    def get_test_loader(self, batchsize):
        return DataLoader(self.test_set, batch_size=batchsize, shuffle=False)
