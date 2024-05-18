import torch
import torch.nn as nn
from torch_geometric.utils.dropout import dropout_edge
from torch_geometric.nn.norm import PairNorm


class MyGraphConv(nn.Module):

    def __init__(self,
                 in_features,
                 out_features,
                 add_self_loops=True,
                 use_cached_adj=True,
                 laplace_normalize=True,
                 bias=True):
        super(MyGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_cached_adj = use_cached_adj
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
        if not self.use_cached_adj or self.cached_adj is None:
            # transform adj to sparse tensor
            num_nodes = input.size(0)
            self.cached_adj = torch.sparse_coo_tensor(
                indices=adj,
                values=torch.ones(adj.size(1), device=adj.device),
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

class DropEdge(nn.Module):

    def __init__(self, p):
        super(DropEdge, self).__init__()
        self.p = p

    def forward(self, adj):
        return dropout_edge(adj, p=self.p, training=self.training)


class MyGCN(nn.Module):

    def __init__(self,
                 nfeat,
                 nclass,
                 nhid=[16],
                 add_self_loops=False,
                 dropedge=0.0,
                 pairnorm=False,
                 activation='ReLU',
                 dropout=0.5):
        super(MyGCN, self).__init__()
        # dropedge
        self.use_dropedge = dropedge > 0.0
        self.dropedge = DropEdge(dropedge)
        # pairnorm
        self.use_pairnorm = pairnorm
        self.pairnorm = PairNorm()
        # activation function
        self.activation = getattr(nn, activation)()
        # dropout
        self.dropout = nn.Dropout(dropout)
        # GraphConv layers
        kwargs = dict(use_cached_adj=False if dropedge > 0.0 else True,
                      add_self_loops=add_self_loops)

        # construct input layer
        self.layers = nn.ModuleList([MyGraphConv(nfeat, nhid[0], **kwargs)])

        # construct hidden layers
        layer_sizes = zip(nhid[:-1], nhid[1:])
        self.layers.extend(
            [MyGraphConv(h1, h2, **kwargs) for h1, h2 in layer_sizes])

        # construct output layer
        self.layers.append(MyGraphConv(nhid[-1], nclass, **kwargs))

    def forward(self, x, adj):
        if self.use_dropedge:
            adj_mask = adj[0] < adj[1]
            adj = adj[:, adj_mask]
            adj, _ = self.dropedge(adj)
            adj = torch.cat([adj, adj.flip(0)], dim=1)
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation(layer(x, adj))
            if self.use_pairnorm:
                x = self.pairnorm(x)
            x = self.dropout(x)
        x = self.layers[-1](x, adj)
        return x
