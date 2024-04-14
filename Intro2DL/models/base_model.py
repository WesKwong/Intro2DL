import torch

from neuralnets import get_nn
from tools.cuda_utils import get_device

device = get_device()


def get_loss_fn(hp):
    # name
    if 'name' in hp['dataset']:
        name = hp['dataset']['name']
    else:
        name = hp['dataset']
    # get loss function
    if name == 'Lab1':
        return torch.nn.MSELoss()
    elif name == 'CIFAR10':
        return torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Invalid dataset: {name}")

class BaseModel(object):

    def __init__(self, train_loader, val_loader, hyperparameters, experiment):
        self.train_loader = train_loader
        self.iter_train_loader = iter(train_loader)
        self.val_loader = val_loader
        self.hp = hyperparameters
        self.expt = experiment

        # log
        self.epoch = 0
        self.iteration = 0
        self.train_loss = 0.0
        self.val_loss = 0.0
        self.test_loss = 0.0
        self.schedule_flag = False
        self.current_lr = self.hp['lr']
        self.get_nn()
        self.get_optim()
        self.get_loss_fn()

        # init validation loss
        self.validate()

    def __str__(self) -> str:
        str = f"\nNet: {self.hp['net']}\n"
        str += self.net.__str__()
        str += f"\n\nOptimizer:\n"
        str += f"{self.optimizer}\n"
        str += f"\nLoss Function:\n"
        str += f"{self.loss_fn}\n"
        return str

    def __repr__(self) -> str:
        return self.__str__()

    def get_nn(self):
        self.net = get_nn(self.hp['net']).to(device)

    def get_optim(self):
        # optimizer
        optimizer = getattr(torch.optim, self.hp['optim'])
        self.optimizer = optimizer(self.net.parameters(), lr=self.hp['lr'])
        # learning rate scheduler
        scheduler = getattr(torch.optim.lr_scheduler,
                            self.hp['scheduler']['name'])
        self.scheduler = scheduler(self.optimizer,
                                   **self.hp['scheduler']['param'])
        self.min_lr = self.hp['scheduler']['min_lr']

    def get_loss_fn(self):
        # loss function
        self.loss_fn = get_loss_fn(self.hp)

    def train(self, iteration):
        train_loss = 0.0
        self.net.train()
        for i in range(iteration):
            # get data
            data, label = next(self.iter_train_loader, (None, None))
            if data is None:
                self.epoch += 1
                self.schedule_flag = True
                self.validate()
                self.iter_train_loader = iter(self.train_loader)
                data, label = next(self.iter_train_loader)

            # move to device
            self.net = self.net.to(device)
            data, label = data.to(device), label.to(device)

            # forward pass
            self.optimizer.zero_grad()
            pred = self.net(data)
            loss = self.loss_fn(pred, label)

            # backward pass
            loss.backward()
            self.optimizer.step()

            # log loss
            train_loss += loss.item()
            self.iteration += 1

            # learning rate scheduler
            if self.schedule_flag and self.current_lr > self.min_lr:
                self.scheduler.step()
                self.current_lr = self.optimizer.param_groups[0]['lr']
                if self.current_lr <= self.min_lr:
                    self.optimizer.param_groups[0]['lr'] = self.min_lr
                    self.current_lr = self.min_lr
                self.schedule_flag = False

        self.train_loss = train_loss / iteration

    def validate(self):
        val_loss = 0.0
        val_steps = 0
        n_correct = 0
        n_total = 0
        self.net.eval()
        with torch.no_grad():
            for data, label in self.val_loader:
                data, label = data.to(device), label.to(device)
                pred = self.net(data)
                # correct
                _, predicted = torch.max(pred.data, 1)
                n_total += label.size(0)
                n_correct += (predicted == label).sum().item()
                # loss
                val_loss += self.loss_fn(pred, label).item()
                val_steps += 1
        self.val_loss = val_loss / len(self.val_loader)
        self.val_acc = n_correct / n_total
        result = {
            "loss": self.val_loss,
            "acc": self.val_acc
        }
        return result

    def test(self, test_loader):
        test_loss = 0.0
        test_steps = 0
        n_correct = 0
        n_total = 0
        self.net.eval()
        with torch.no_grad():
            for data, label in test_loader:
                data, label = data.to(device), label.to(device)
                pred = self.net(data)
                # correct
                _, predicted = torch.max(pred.data, 1)
                n_total += label.size(0)
                n_correct += (predicted == label).sum().item()
                # loss
                test_loss += self.loss_fn(pred, label).item()
                test_steps += 1
        self.test_loss = test_loss / len(test_loader)
        self.test_acc = n_correct / n_total
        result = {
            'loss': self.test_loss,
            'acc': self.test_acc
        }
        return result