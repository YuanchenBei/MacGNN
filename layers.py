#!/usr/bin/env python
# coding: utf-8

import torch
import pandas as pd
import numpy as np
import pickle
import math
from torch.utils.data import DataLoader
import torch.nn as nn
import sklearn.metrics as metrics
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence


class MultiLayerPerceptron(nn.Module):

    def __init__(self, layer, batch_norm=True, dropout=0.5):
        super(MultiLayerPerceptron, self).__init__()
        layers = []
        input_size = layer[0]
        for output_size in layer[1: -1]:
            layers.append(nn.Linear(input_size, output_size))
            if batch_norm:
                layers.append(nn.BatchNorm1d(output_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            input_size = output_size
        layers.append(nn.Linear(input_size, layer[-1]))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class Dice(nn.Module):
    
    def __init__(self):
        super(Dice, self).__init__()
        self.alpha = nn.Parameter(torch.zeros((1, )))
        
    def forward(self, x):
        avg = x.mean(dim=0)
        std = x.std(dim=0)
        norm_x = (x - avg) / std
        p = torch.sigmoid(norm_x+1e-8)

        return x.mul(p) + self.alpha * x.mul(1 - p)


class GCNlayer(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GCNlayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
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
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

