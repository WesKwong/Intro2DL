import os

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from .base_dataset import BaseDataset
from configs import global_config as config


class CIFAR10(BaseDataset):

    def __init__(self, path):
        self.name = 'CIFAR10'
        super().__init__(path)

    def gen_raw_data(self, raw_path):
        if not config.prepare_new_dataset and os.path.exists(
                os.path.join(raw_path, 'data.pt')):
            return
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.49139968, 0.48215827, 0.44653124],
                                 std=[0.24703233, 0.24348505, 0.26158768],
                                 inplace=True)
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.49139968, 0.48215827, 0.44653124],
                                 std=[0.24703233, 0.24348505, 0.26158768],
                                 inplace=True)
        ])
        train_set = datasets.CIFAR10(root=raw_path,
                                     train=True,
                                     download=True,
                                     transform=train_transform)
        test_set = datasets.CIFAR10(root=raw_path,
                                    train=False,
                                    download=True,
                                    transform=test_transform)
        if not os.path.exists(raw_path):
            os.makedirs(raw_path)
        data = {'train': train_set, 'test': test_set}
        torch.save(data, os.path.join(raw_path, 'data.pt'))

    def split_train_data(self, raw_path, split_path):
        if not config.prepare_new_dataset and os.path.exists(
                os.path.join(split_path, 'data.pt')):
            return
        data = torch.load(os.path.join(raw_path, 'data.pt'))
        train_set = data['train']
        test_set = data['test']
        train_size = int(len(train_set) * 0.98)
        val_size = len(train_set) - train_size
        train_set, val_set = torch.utils.data.random_split(
            train_set, [train_size, val_size])
        if not os.path.exists(split_path):
            os.makedirs(split_path)
        data = {'train': train_set, 'val': val_set, 'test': test_set}
        torch.save(data, os.path.join(split_path, 'data.pt'))

    def load_split_data(self, split_path):
        data = torch.load(os.path.join(split_path, 'data.pt'))
        self.train_set = data['train']
        self.val_set = data['val']
        self.test_set = data['test']
