import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    r"""Convolutional Neural Network: LeNet-5.
    Args:
        in_channels (int, optional): Number of channels in the input data. Defaults to 3.
        n_classes (int, optional): Number of classes in the dataset. Defaults to 10.
    """

    def __init__(self, in_channels: int = 3, n_classes: int = 10) -> None:
        super(LeNet5, self).__init__()
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


class BNormLeNet5(nn.Module):
    r"""Convolutional Neural Network: LeNet-5 with Batch Normalization.
    Args:
        in_channels (int, optional): Number of channels in the input data. Defaults to 3.
        n_classes (int, optional): Number of classes in the dataset. Defaults to 10.
    """

    def __init__(self, in_channels: int = 3, n_classes: int = 10) -> None:
        super(BNormLeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LNormLeNet5(nn.Module):
    r"""Convolutional Neural Network: LeNet-5 with Layer Normalization.
    Args:
        in_channels (int, optional): Number of channels in the input data. Defaults to 3.
        n_classes (int, optional): Number of classes in the dataset. Defaults to 10.
    """

    def __init__(self, in_channels: int = 3, n_classes: int = 10) -> None:
        super(LNormLeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.ln1 = nn.LayerNorm([6, 28, 28])
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.ln2 = nn.LayerNorm([16, 10, 10])
        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.ln1(self.conv1(x))))
        x = self.pool(F.relu(self.ln2(self.conv2(x))))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class INormLeNet5(nn.Module):
    r"""Convolutional Neural Network: LeNet-5 with Instance Normalization.
    Args:
        in_channels (int, optional): Number of channels in the input data. Defaults to 3.
        n_classes (int, optional): Number of classes in the dataset. Defaults to 10.
    """

    def __init__(self, in_channels: int = 3, n_classes: int = 10) -> None:
        super(INormLeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.in1 = nn.InstanceNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.in2 = nn.InstanceNorm2d(16)
        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.in1(self.conv1(x))))
        x = self.pool(F.relu(self.in2(self.conv2(x))))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PadBNLeNet5(nn.Module):
    r"""Convolutional Neural Network: LeNet-5 with Padding and Batch Normalization.
    Args:
        in_channels (int, optional): Number of channels in the input data. Defaults to 3.
        n_classes (int, optional): Number of classes in the dataset. Defaults to 10.
    """

    def __init__(self, in_channels: int = 3, n_classes: int = 10) -> None:
        super(PadBNLeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, 5, padding=2)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 5, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class PadBNKernel3LeNet5(nn.Module):
    r"""Convolutional Neural Network: LeNet-5 with Padding and Batch Normalization and Kernel size 3.
    Args:
        in_channels (int, optional): Number of channels in the input data. Defaults to 3.
        n_classes (int, optional): Number of classes in the dataset. Defaults to 10.
    """

    def __init__(self, in_channels: int = 3, n_classes: int = 10) -> None:
        super(PadBNKernel3LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
