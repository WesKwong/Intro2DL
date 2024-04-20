import os
from copy import copy

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from .base_dataset import BaseDataset
from configs import global_config as config


class CIFAR10(BaseDataset):

    def __init__(self, path):
        self.name = 'CIFAR10'
        super().__init__(path)

    def get_transform(self):
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.49139968, 0.48215827, 0.44653124],
                                 std=[0.24703233, 0.24348505, 0.26158768],
                                 inplace=True)
        ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.49139968, 0.48215827, 0.44653124],
                                 std=[0.24703233, 0.24348505, 0.26158768],
                                 inplace=True)
        ])

    def gen_raw_data(self, raw_path):
        if not config.prepare_new_dataset and os.path.exists(
                os.path.join(raw_path, 'data.pt')):
            return
        train_set = datasets.CIFAR10(root=raw_path,
                                     train=True,
                                     download=True)
        test_set = datasets.CIFAR10(root=raw_path,
                                    train=False,
                                    download=True)
        if not os.path.exists(raw_path):
            os.makedirs(raw_path)
        data = {'train': train_set, 'test': test_set}
        torch.save(data, os.path.join(raw_path, 'data.pt'))

    def split_train_data(self, raw_path, split_path):
        if not config.prepare_new_dataset and os.path.exists(
                os.path.join(split_path, 'data.pt')):
            return
        # Load raw data
        data = torch.load(os.path.join(raw_path, 'data.pt'))
        data_set = data['train']
        test_set = data['test']
        # Split train set into train and val set
        train_size = int(len(data_set) * 0.98)
        val_size = len(data_set) - train_size
        train_set, val_set = torch.utils.data.random_split(
            data_set, [train_size, val_size])
        # Set transform
        self.get_transform()
        train_set.dataset = copy(data_set)
        train_set.dataset.transform = self.train_transform
        val_set.dataset = copy(data_set)
        val_set.dataset.transform = self.test_transform
        test_set.transform = self.test_transform
        # Save split data
        if not os.path.exists(split_path):
            os.makedirs(split_path)
        data = {'train': train_set, 'val': val_set, 'test': test_set}
        torch.save(data, os.path.join(split_path, 'data.pt'))

    def load_split_data(self, split_path):
        data = torch.load(os.path.join(split_path, 'data.pt'))
        self.train_set = data['train']
        self.val_set = data['val']
        self.test_set = data['test']
