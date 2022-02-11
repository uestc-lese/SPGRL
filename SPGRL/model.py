from torch_geometric.nn import GCNConv, GAE
import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, nhid1, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, nhid1, cached=True)
        self.conv2 = GCNConv(nhid1, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class GraphConvolution(torch.nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.acti = nn.ReLU(inplace=True)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return self.acti(output + self.bias)
        else:
            return self.acti(output)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(torch.nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, out)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training = self.training)
        x = self.gc2(x, adj)
        return x

class GAE(torch.nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, nclass, dropout=0.5):
        super(GAE, self).__init__()
        self.gcn = GCN(nfeat, nhid1, nhid2, dropout)
        # self.MLP = nn.Sequential(
        #     nn.Linear(nhid2, nclass),
        #     nn.LogSoftmax(dim=1)
        # )
    def dot_product_decode(self, Z):
        A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
        return A_pred

    def forward(self, x, adj):
        h = self.gcn(x, adj)
        A_pred = self.dot_product_decode(h)
        # z = self.MLP(h)
        return h, A_pred

class GClip(torch.nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, nclass, dropout=0.5):
        super(GClip, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid1, dropout)
        self.gc2 = GraphConvolution(nhid1, nhid2, dropout)
        self.gc3 = GraphConvolution(nhid1, nhid2, dropout)
        self.gae1 = GAE(nfeat, nhid1, nhid2, nclass, dropout)
        self.gae2 = GAE(nfeat, nhid1, nhid2, nclass, dropout)
        self.MLP = nn.Sequential(
            nn.Linear(nhid2 * 2, nhid2 * 2, bias=False),
            #nn.BatchNorm1d(nhid2 * 2),
            #nn.ReLU(inplace=True), # second layer
            #nn.BatchNorm1d(nhid2* 2, affine=False),# for acm dataset BatchNorm1d*3
            nn.Linear(nhid2 * 2, nhid2),
            nn.Linear(nhid2, nclass),
            nn.LogSoftmax(dim=1)
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    def encode(self, x, sadj , fadj):
        shidden1 = self.gc1(x, sadj)
        fhidden1 = self.gc1(x, fadj)
        return self.gc2(shidden1, sadj), self.gc3(shidden1, sadj), self.gc2(fhidden1, sadj), self.gc3(fhidden1, sadj)
    '''
    def reparameterize(self, smu, slogvar, fmu, flogvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
    '''

    def forward(self, x, sadj, fadj):
        smu, slogvar, fmu, flogvar  = self.encode(x, sadj, fadj)
        h1, A_pred1 = self.gae1(x, sadj)
        h2, A_pred2 = self.gae2(x, fadj)
        emb1 = h1 / h1.norm(dim=-1, keepdim=True)
        emb2 = h2 / h2.norm(dim=-1, keepdim=True)
        z = torch.cat((h1, h2), dim=1)
        out = self.MLP(z)
        return out, A_pred1, A_pred2, emb1, emb2, self.logit_scale.exp(), smu, slogvar, fmu, flogvar