import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.cuda_utils import get_device

device = get_device()


class MyGraphConv(nn.Module):

    def __init__(self,
                 in_features,
                 out_features,
                 add_self_loops=True,
                 laplace_normalize=True,
                 bias=True):
        super(MyGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.add_self_loops = add_self_loops
        self.laplace_normalize = laplace_normalize

        self.weight = nn.Parameter(torch.FloatTensor(in_features,
                                                     out_features))
        self.cached_adj = None
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input, adj):
        if self.cached_adj is None:
            # transform adj to sparse tensor
            num_nodes = input.size(0)
            self.cached_adj = torch.sparse_coo_tensor(
                indices=adj,
                values=torch.ones(adj.size(1)).to(device),
                size=(num_nodes, num_nodes))
            self.cached_adj = self.cached_adj.to_dense()
            # add self loops
            if self.add_self_loops:
                self.cached_adj = self.cached_adj.fill_diagonal_(1)
            else:
                self.cached_adj = self.cached_adj.fill_diagonal_(0)
            # normalize adj
            if self.laplace_normalize:
                deg = self.cached_adj.sum(dim=1)
                # A = D^-1/2 * A * D^-1/2
                deg_inv_sqrt = deg.pow(-0.5)
                deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
                deg_inv_sqrt = deg_inv_sqrt.diag()
                self.cached_adj = torch.mm(deg_inv_sqrt, self.cached_adj)
                self.cached_adj = torch.mm(self.cached_adj, deg_inv_sqrt)
            self.cached_adj = self.cached_adj.to_sparse()
        adj = self.cached_adj

        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + str(self.in_features) + ', ' \
               + str(self.out_features) + ')'

class MyPairNorm(nn.Module):

        def __init__(self, mode='PN', scale=1):
            super(MyPairNorm, self).__init__()
            self.mode = mode
            self.scale = scale

        def forward(self, x):
            if self.mode == 'PN':
                x = x / x.norm(dim=1)[:, None]
            elif self.mode == 'PN-SI':
                x = x / x.norm(dim=1)[:, None]
                x = self.scale * x
            elif self.mode == 'PN-TI':
                x = x / x.norm(dim=1)[:, None]
                x = x / x.norm(dim=1)[:, None]
            elif self.mode == 'None':
                pass
            else:
                raise ValueError('PairNorm mode must be PN, PN-SI, PN-TI, or None')
            return x

class MyGCN(nn.Module):

    def __init__(self, nfeat, nclass, nhid=[16], activation='ReLU', pairnorm=False, dropout=0.5):
        super(MyGCN, self).__init__()
        self.activation = getattr(nn, activation)()
        self.dropout = nn.Dropout(dropout)
        self.pairnorm = pairnorm

        # construct input layer
        self.layers = nn.ModuleList([MyGraphConv(nfeat, nhid[0])])

        # construct hidden layers
        layer_sizes = zip(nhid[:-1], nhid[1:])
        self.layers.extend([MyGraphConv(h1, h2) for h1, h2 in layer_sizes])

        # construct output layer
        self.layers.append(MyGraphConv(nhid[-1], nclass))

    # @staticmethod
    # def PairNorm(x_feature):
    #     mode = 'PN'
    #     scale = 5
    #     col_mean = x_feature.mean(dim=0)
    #     if mode == 'PN':
    #         x_feature = x_feature - col_mean
    #         row_norm_mean = (1e-6 + x_feature.pow(2).sum(dim=1).mean()).sqrt()
    #         x_feature = scale * x_feature / row_norm_mean

    #     if mode == 'PN-SI':
    #         x_feature = x_feature - col_mean
    #         row_norm_individual = (1e-6 + x_feature.pow(2).sum(dim=1, keepdim=True)).sqrt()
    #         x_feature = scale * x_feature / row_norm_individual

    #     if mode == 'PN-SCS':
    #         row_norm_individual = (1e-6 + x_feature.pow(2).sum(dim=1, keepdim=True)).sqrt()
    #         x_feature = scale * x_feature / row_norm_individual - col_mean

    #     return x_feature

    @staticmethod
    def PairNorm(x):
        row_norms = x.norm(dim=-1, keepdim=True)
        col_mean = row_norms.mean(dim=-2, keepdim=True)
        x = x / (row_norms * col_mean).clamp(min=1e-6)
        return x


    def forward(self, x, adj):
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation(layer(x, adj))
            if self.pairnorm:
                x = self.PairNorm(x)
            x = self.dropout(x)
        x = self.layers[-1](x, adj)
        return x
