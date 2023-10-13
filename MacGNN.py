import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from layers import Dice
import torch.nn.functional as F
import math


class NeighborAggregation(nn.Module):

    def __init__(self, embed_dim=8, hidden_dim=8):
        super(NeighborAggregation, self).__init__()
        self.Q_w = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.K_w = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.V_w = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.trans_d = math.sqrt(hidden_dim)
        self.get_score = nn.Softmax(dim=-1)

    def forward(self, query, key):
        trans_Q = self.Q_w(query)
        trans_K = self.K_w(key)
        trans_V = self.V_w(query)
        score = self.get_score(torch.bmm(trans_Q, torch.transpose(trans_K,1,2))/(self.trans_d))
        answer = torch.mul(trans_V, score)
        return answer


class MacGNN(nn.Module):

    def __init__(self, field_dims, u_group_num, i_group_num, embed_dim, recent_len, tau=0.8, device='cpu'):
        super(MacGNN, self).__init__()
        self.user_embed = nn.Embedding(field_dims[0], embed_dim)
        self.item_embed = nn.Embedding(field_dims[1], embed_dim)
        self.cate_embed = nn.Embedding(field_dims[2], embed_dim)
        self.u_macro_embed = nn.Embedding(u_group_num + 1, embed_dim)
        self.i_macro_embed = nn.Embedding(i_group_num + 1, embed_dim)
        torch.nn.init.xavier_uniform_(self.user_embed.weight.data)
        torch.nn.init.xavier_uniform_(self.item_embed.weight.data)
        torch.nn.init.xavier_uniform_(self.cate_embed.weight.data)
        torch.nn.init.xavier_uniform_(self.u_macro_embed.weight.data)
        torch.nn.init.xavier_uniform_(self.i_macro_embed.weight.data)
        self.tau = tau
        self.u_shared_aggregator = NeighborAggregation(embed_dim, 2 * embed_dim)
        self.i_shared_aggregator = NeighborAggregation(embed_dim, 2 * embed_dim)
        self.u_group_num = u_group_num + 1
        self.i_group_num = i_group_num + 1
        self.recent_len = recent_len
        self.macro_weight_func = nn.Softmax(dim=1)
        self.u_gruop_slice = torch.arange(self.u_group_num, requires_grad=False).to(device)
        self.i_gruop_slice = torch.arange(self.i_group_num, requires_grad=False).to(device)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 14, 200),
            Dice(),
            nn.Linear(200, 80),
            Dice(),
            nn.Linear(80, 1)
        )

    def forward(self, x):
        # print(x.shape)
        user_embedding = self.user_embed(x[:, 0])
        user_1ord_neighbor = x[:, 1: self.i_group_num + 1]
        user_2ord_neighbor = x[:, self.i_group_num + 1: self.i_group_num + self.u_group_num + 1]
        user_recent = x[:, self.i_group_num + self.u_group_num + 1: self.i_group_num + self.u_group_num + self.recent_len + 1]
        item_embedding = self.item_embed(x[:, self.i_group_num + self.u_group_num + self.recent_len + 1])
        item_1ord_neighbor = x[:, self.i_group_num + self.u_group_num + self.recent_len + 2: self.i_group_num + 2 * self.u_group_num + self.recent_len + 2]
        item_2ord_neighbor = x[:, self.i_group_num + 2 * self.u_group_num + self.recent_len + 2: 2 * self.i_group_num + 2 * self.u_group_num + self.recent_len + 2]
        item_recent = x[:, 2 * self.i_group_num + 2 * self.u_group_num + self.recent_len + 2:]

        batch_u_gruop_slice = self.u_gruop_slice.expand(x.shape[0], self.u_group_num)
        batch_i_gruop_slice = self.i_gruop_slice.expand(x.shape[0], self.i_group_num)

        user_recent_mask = (user_recent > 0).float().unsqueeze(-1)
        item_recent_mask = (item_recent > 0).float().unsqueeze(-1)
        
        user_1ord_weight = self.macro_weight_func(torch.log(user_1ord_neighbor.float()+1) / self.tau).unsqueeze(-1)
        user_2ord_weight = self.macro_weight_func(torch.log(user_2ord_neighbor.float()+1) / self.tau).unsqueeze(-1)
        item_1ord_weight = self.macro_weight_func(torch.log(item_1ord_neighbor.float()+1) / self.tau).unsqueeze(-1)
        item_2ord_weight = self.macro_weight_func(torch.log(item_2ord_neighbor.float()+1) / self.tau).unsqueeze(-1)

        user_1ord_embedding = self.i_macro_embed(batch_i_gruop_slice)
        user_2ord_embedding = self.u_macro_embed(batch_u_gruop_slice)
        item_1ord_embedding = self.u_macro_embed(batch_u_gruop_slice)
        item_2ord_embedding = self.i_macro_embed(batch_i_gruop_slice)
        user_recent_embedding = self.item_embed(user_recent)
        item_recent_embedding = self.user_embed(item_recent)

        u_1ord_trans_emb = self.i_shared_aggregator(user_1ord_embedding, item_embedding.unsqueeze(1))
        u_2ord_trans_emb = self.u_shared_aggregator(user_2ord_embedding, user_embedding.unsqueeze(1))
        i_1ord_trans_emb = self.u_shared_aggregator(item_1ord_embedding, user_embedding.unsqueeze(1))
        i_2ord_trans_emb = self.i_shared_aggregator(item_2ord_embedding, item_embedding.unsqueeze(1))
        user_recent_trans_emb = self.i_shared_aggregator(user_recent_embedding, item_embedding.unsqueeze(1))
        item_recent_trans_emb = self.u_shared_aggregator(item_recent_embedding, user_embedding.unsqueeze(1))

        user_1ord_ws = torch.mul(u_1ord_trans_emb, user_1ord_weight).sum(dim=1)
        user_2ord_ws = torch.mul(u_2ord_trans_emb, user_2ord_weight).sum(dim=1)
        item_1ord_ws = torch.mul(i_1ord_trans_emb, item_1ord_weight).sum(dim=1)
        item_2ord_ws = torch.mul(i_2ord_trans_emb, item_2ord_weight).sum(dim=1)
        user_recent_ws = torch.mul(user_recent_trans_emb, user_recent_mask).sum(dim=1)
        item_recent_ws = torch.mul(item_recent_trans_emb, item_recent_mask).sum(dim=1)

        concated = torch.hstack([user_embedding, user_1ord_ws, user_2ord_ws, user_recent_ws, item_embedding, item_1ord_ws, item_2ord_ws, item_recent_ws])
        output = self.mlp(concated)
        output = torch.sigmoid(output)
        return output
