from torch_geometric.utils import train_test_split_edges, negative_sampling
from sklearn.metrics import roc_auc_score

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
        # features & class
        nfeat = self.dataset.num_node_features
        nclass = self.dataset.num_classes
        # hyperparameters
        add_self_loops = self.hp['add_self_loops']
        nhid = self.hp['nhid']
        dropedge = self.hp['dropedge']
        pairnorm = self.hp['pairnorm']
        activation = self.hp['activation']
        # get net hp
        param = dict(nfeat=nfeat,
                     nclass=nclass,
                     add_self_loops=add_self_loops,
                     nhid=nhid,
                     activation=activation,
                     dropedge=dropedge,
                     pairnorm=pairnorm)
        net_hp = dict(name=self.hp['net'], param=param)
        # get hp
        hp = dict(net=net_hp)
        # get net
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
        result = {"loss": self.val_loss, "acc": self.val_acc}
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
        result = {"loss": self.test_loss, "acc": self.test_acc}
        return result


class LinkGCNModel(BaseModel):

    def __init__(self, data, dataset, hyperparameters, experiment):
        self.data = train_test_split_edges(data).to(device)
        self.dataset = dataset
        super().__init__(data, dataset, hyperparameters, experiment)

    def get_nn(self):
        # features & class
        nfeat = self.dataset.num_node_features
        nclass = 64
        # hyperparameters
        add_self_loops = self.hp['add_self_loops']
        nhid = self.hp['nhid']
        dropedge = self.hp['dropedge']
        pairnorm = self.hp['pairnorm']
        activation = self.hp['activation']
        # get net hp
        param = dict(nfeat=nfeat,
                     nclass=nclass,
                     add_self_loops=add_self_loops,
                     nhid=nhid,
                     activation=activation,
                     dropedge=dropedge,
                     pairnorm=pairnorm)
        net_hp = dict(name=self.hp['net'], param=param)
        # get hp
        hp = dict(net=net_hp)
        # get net
        self.net = get_nn(hp).to(device)

    def get_loss_fn(self):
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    @staticmethod
    def get_link_labels(pos_edge_index, neg_edge_index):
        pos_label = torch.ones(pos_edge_index.size(1), dtype=torch.float, device=pos_edge_index.device)
        neg_label = torch.zeros(neg_edge_index.size(1), dtype=torch.float, device=neg_edge_index.device)
        link_labels = torch.cat([pos_label, neg_label])
        return link_labels

    def train(self, iteration):
        train_loss = 0.0
        self.net.train()
        for i in range(iteration):
            # forward
            self.optimizer.zero_grad()

            neg_edge_index = negative_sampling(
                edge_index=self.data.train_pos_edge_index,
                num_nodes=self.data.num_nodes,
                num_neg_samples=self.data.train_pos_edge_index.size(1))

            z = self.net(self.data.x, self.data.train_pos_edge_index)
            link_logits = self.net.decode(z, self.data.train_pos_edge_index,
                                          neg_edge_index)
            link_labels = self.get_link_labels(self.data.train_pos_edge_index,
                                               neg_edge_index)
            loss = self.loss_fn(link_logits, link_labels)

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
        self.net.eval()
        with torch.no_grad():
            z = self.net(self.data.x, self.data.train_pos_edge_index)
            pos_edge_index = self.data.val_pos_edge_index
            neg_edge_index = self.data.val_neg_edge_index
            link_logits = self.net.decode(z, pos_edge_index, neg_edge_index)
            link_probs = link_logits.sigmoid()
            link_labels = self.get_link_labels(pos_edge_index, neg_edge_index)
            self.val_loss = self.loss_fn(link_logits, link_labels).item()
            self.val_acc = roc_auc_score(link_labels.cpu().numpy(),
                                         link_probs.cpu().numpy())
        result = {"loss": self.val_loss, "acc": self.val_acc}
        return result

    def test(self, testloader):
        self.net.eval()
        with torch.no_grad():
            z = self.net(self.data.x, self.data.train_pos_edge_index)
            pos_edge_index = self.data.test_pos_edge_index
            neg_edge_index = self.data.test_neg_edge_index
            link_logits = self.net.decode(z, pos_edge_index, neg_edge_index)
            link_probs = link_logits.sigmoid()
            link_labels = self.get_link_labels(pos_edge_index, neg_edge_index)
            self.test_loss = self.loss_fn(link_logits, link_labels).item()
            self.test_acc = roc_auc_score(link_labels.cpu().numpy(),
                                    link_probs.cpu().numpy())
        result = {"loss": self.test_loss, "acc": self.test_acc}
        return result
