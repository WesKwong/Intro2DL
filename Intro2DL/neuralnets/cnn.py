import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    r"""Convolutional Neural Network (CNN) class.
    Args:
        in_channels (int, optional): Number of channels in the input data. Defaults to 3.
        n_classes (int, optional): Number of classes in the dataset. Defaults to 10.
    """

    def __init__(self, in_channels: int = 3, n_classes: int = 10) -> None:
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
