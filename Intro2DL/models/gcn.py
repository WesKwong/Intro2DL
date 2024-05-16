from .base_model import *


class GCNModel(BaseModel):

    def __init__(self, data, dataset, hyperparameters, experiment):
        self.data = data
        self.dataset = dataset
        self.x = data.x.to(device)
        self.edge_index = data.edge_index.to(device)
        self.y = data.y.to(device)
        self.train_mask = data.train_mask.to(device)
        self.val_mask = data.val_mask.to(device)
        self.test_mask = data.test_mask.to(device)
        super().__init__(None, None, hyperparameters, experiment)

    def get_nn(self):
        nfeat = self.dataset.num_node_features
        nclass = self.dataset.num_classes
        nhid = self.hp['nhid']
        activation = self.hp['activation']
        pairnorm = self.hp['pairnorm']
        dropout = self.hp['dropout']
        param = dict(nfeat=nfeat,
                      nclass=nclass,
                      nhid=nhid,
                      activation=activation,
                      pairnorm=pairnorm,
                      dropout=dropout)
        net_hp = dict(name=self.hp['net'], param=param)
        hp = dict(net=net_hp)
        self.net = get_nn(hp).to(device)

    def train(self, iteration):
        train_loss = 0.0
        self.net.train()
        for i in range(iteration):
            # forward
            self.optimizer.zero_grad()
            pred = self.net(self.x, self.edge_index)
            loss = self.loss_fn(pred[self.train_mask], self.y[self.train_mask])

            # backward
            loss.backward()
            self.optimizer.step()

            # log loss
            train_loss += loss.item()

            self.epoch += 1
            self.schedule_flag = True

            # learning rate scheduler
            if self.schedule_flag and self.current_lr > self.min_lr:
                self.scheduler.step()
                self.current_lr = self.optimizer.param_groups[0]['lr']
                if self.current_lr <= self.min_lr:
                    self.optimizer.param_groups[0]['lr'] = self.min_lr
                    self.current_lr = self.min_lr
                self.schedule_flag = False

        self.train_loss = train_loss / iteration
        return self.train_loss

    def validate(self):
        val_loss = 0.0
        val_steps = 0
        n_correct = 0
        n_total = 0
        self.net.eval()
        with torch.no_grad():
            pred = self.net(self.x, self.edge_index)[self.val_mask]
            label = self.y[self.val_mask]
            # correct
            _, predicted = torch.max(pred.data, 1)
            n_total += label.size(0)
            n_correct += (predicted == label).sum().item()
            # loss
            val_loss += self.loss_fn(pred, label).item()
            val_steps += 1
        self.val_loss = val_loss / val_steps
        self.val_acc = n_correct / n_total
        result = {
            "loss": self.val_loss,
            "acc": self.val_acc
        }
        return result

    def test(self, testloader):
        test_loss = 0.0
        test_steps = 0
        n_correct = 0
        n_total = 0
        self.net.eval()
        with torch.no_grad():
            pred = self.net(self.x, self.edge_index)[self.test_mask]
            label = self.y[self.test_mask]
            # correct
            _, predicted = torch.max(pred.data, 1)
            n_total += label.size(0)
            n_correct += (predicted == label).sum().item()
            # loss
            test_loss += self.loss_fn(pred, label).item()
            test_steps += 1
        self.test_loss = test_loss / test_steps
        self.test_acc = n_correct / n_total
        result = {
            "loss": self.test_loss,
            "acc": self.test_acc
        }
        return result
