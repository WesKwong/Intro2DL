import os

import torch
from torch.utils.data import TensorDataset

from .base_dataset import BaseDataset
from configs import global_config as config


class Lab1(BaseDataset):

    def __init__(self, path, N: int):
        self.name = 'Lab1'
        self.N = N
        super().__init__(path)

    def get_transform(self):
        pass

    def gen_raw_data(self, raw_path):
        N = self.N
        data_path = os.path.join(raw_path, f'N{N}.pt')
        # judge if the dataset exists
        if not config.prepare_new_dataset and os.path.exists(data_path):
            return

        # --------------------- generate data -------------------- #
        # generate with unique elements
        x = torch.FloatTensor(int(1.1 * N)).uniform_(1, 16)
        x = torch.unique(x, sorted=False)
        while x.numel() < N:
            x = torch.cat(
                [x, torch.FloatTensor(N - x.numel()).uniform_(1, 16)])
            x = torch.unique(x, sorted=False)
        x = x[:N].reshape(-1, 1)
        # shuffle the data
        shuffle_idx = torch.randperm(x.numel())
        x[:] = x.reshape(-1)[shuffle_idx].reshape(x.shape)
        # get targets
        f = lambda x: torch.log2(x) + torch.cos(torch.pi * x / 2)
        data = TensorDataset(x, f(x))

        # ---------------------- save data ---------------------- #
        if not os.path.exists(raw_path):
            os.makedirs(raw_path)
        torch.save(data, data_path)

    def split_train_data(self, raw_path, split_path):
        N = self.N
        data_path = os.path.join(raw_path, f'N{N}.pt')
        split_data_path = os.path.join(split_path, f'N{N}.pt')
        # judge if the dataset exists
        if not config.prepare_new_dataset and os.path.exists(split_data_path):
            return

        # ----------------- calculate split index ---------------- #
        train_size = 8
        val_size = 1
        test_size = 1
        total_size = train_size + val_size + test_size
        train_proportion = train_size / total_size
        val_proportion = val_size / total_size
        test_proportion = test_size / total_size
        train_index = int(train_proportion * N)
        val_index = int(val_proportion * N) + train_index
        test_index = int(test_proportion * N) + val_index

        # ---------------------- split data ---------------------- #
        data = torch.load(data_path)
        x = data.tensors[0]
        y = data.tensors[1]
        train_data = TensorDataset(x[:train_index], y[:train_index])
        val_data = TensorDataset(x[train_index:val_index], y[train_index:val_index])
        test_data = TensorDataset(x[val_index:test_index], y[val_index:test_index])

        # ---------------------- save data ----------------------- #
        split_data = {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
        if not os.path.exists(split_path):
            os.makedirs(split_path)
        torch.save(split_data, split_data_path)

    def load_split_data(self, split_path):
        split_data_path = os.path.join(split_path, f'N{self.N}.pt')
        split_data = torch.load(split_data_path)
        self.train_set = split_data['train']
        self.val_set = split_data['val']
        self.test_set = split_data['test']
