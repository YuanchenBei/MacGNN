#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import torch
import pandas as pd
import numpy as np
import pickle
from torch.utils.data import DataLoader
import torch.nn as nn
import sklearn.metrics as metrics
from gauc_metric import cal_group_auc
import tqdm
import matplotlib.pyplot as plt
from utils import DatasetBuilder
from macgnn import MacGNN
import argparse
import random
import os


############################get-args####################################
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', default='ml-10m')
parser.add_argument('--model_name', default='cgi')
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--early_epoch', type=int, default=5)
parser.add_argument('--learning_rate', type=float, default=1e-2)
parser.add_argument('--weight_decay', type=float, default=5e-5)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--sample_size', type=int, default=2)
parser.add_argument('--embed_dim', type=int, default=10)
parser.add_argument('--save_path', default='chkpt')
parser.add_argument('--record_path', default='record')
parser.add_argument('--use_gpu', default=True, help='Whether to use CUDA')
parser.add_argument('--cuda_id', type=int, default=0, help='CUDA id')
parser.add_argument('--seq_len', type=int, default=100, help='user hist len')
parser.add_argument('--short_len', type=int, default=20, help='user hist len')
parser.add_argument('--recent_len', type=int, default=20, help='user hist len')
parser.add_argument('--runs', type=int, default=1, help='model runs')
parser.add_argument('--seed', type=int, default=2023)
parser.add_argument('--tau', type=float, default=0.8)
parser.add_argument('--test_iter', type=int, default=50)


args = parser.parse_args()

print(args)


def set_seed(seed, cuda):
    print('Set Seed: ', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


embed_dim = args.embed_dim
learning_rate = args.learning_rate
weight_decay = args.weight_decay
epoch = args.epoch
trials = args.early_epoch
batch_size = args.batch_size
device = torch.device("cuda:%d" % (args.cuda_id) if (torch.cuda.is_available() and args.use_gpu) else "cpu")
save_path = args.save_path
record_path = args.record_path
model_name = args.model_name
dataset_name = args.dataset_name
seq_len = args.seq_len
short_len = args.short_len
recent_len = args.recent_len
sample_size = args.sample_size

set_seed(args.seed, args.use_gpu)


if args.dataset_name == 'ml-10m':
    with open(f'../data/{dataset_name}.pkl', 'rb') as f:
        print(f)
        train_set = np.array(pickle.load(f, encoding='latin1')) 
        test_set = np.array(pickle.load(f, encoding='latin1')) 
        cate_list = torch.tensor(pickle.load(f, encoding='latin1')).to(device)
        u_cluster_list = torch.tensor(pickle.load(f, encoding='latin1')).to(device)
        i_cluster_list = torch.tensor(pickle.load(f, encoding='latin1')).to(device)
        user_count, item_count, cate_count, u_cluster_num, i_cluster_num = pickle.load(f)


if args.dataset_name == 'elec':
    with open(f'../data/{dataset_name}.pkl', 'rb') as f:
        print(f)
        train_set = np.array(pickle.load(f, encoding='latin1'))  
        test_set = np.array(pickle.load(f, encoding='latin1')) 
        cate_list = torch.tensor(pickle.load(f, encoding='latin1')).to(device)  
        u_cluster_list = torch.tensor(pickle.load(f, encoding='latin1')).to(device)
        i_cluster_list = torch.tensor(pickle.load(f, encoding='latin1')).to(device)
        user_count, item_count, cate_count, u_cluster_num, i_cluster_num = pickle.load(f)

if args.dataset_name == 'kuairec':
    with open(f'../data/{dataset_name}.pkl', 'rb') as f:
        print(f)
        train_set = np.array(pickle.load(f, encoding='latin1'))  
        test_set = np.array(pickle.load(f, encoding='latin1'))  
        cate_list = torch.tensor(pickle.load(f, encoding='latin1')).to(device) 
        u_cluster_list = torch.tensor(pickle.load(f, encoding='latin1')).to(device)
        i_cluster_list = torch.tensor(pickle.load(f, encoding='latin1')).to(device)
        user_count, item_count, cate_count, u_cluster_num, i_cluster_num = pickle.load(f)


train_size = (u_cluster_num+i_cluster_num+recent_len+1+1)*2
test_size = (u_cluster_num+i_cluster_num+recent_len+1+1)*2
u_cluster_num -= 1

field_dims = [user_count + 1, item_count + 1, cate_count + 1]  # idx-0 for padding

train_data = DatasetBuilder(data=train_set, user_count=user_count, item_count=item_count)
test_data = DatasetBuilder(data=test_set, user_count=user_count, item_count=item_count)

train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=batch_size)


class EarlyStopper(object):

    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_auc = 0.0
        self.best_logloss = 1000000
        self.save_path = save_path

    def is_continuable(self, model, auc, log_loss):
        if auc > self.best_auc:
            self.best_logloss = log_loss
            self.best_auc = auc
            self.trial_counter = 0
            torch.save(model, self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False


# model training
def train(model, optimizer, train_data_loader, test_data_loader, criterion, device, early_stopper, epochs=10, test_iter=50, log_interval=20):
    
    total_loss = 0.0
    tk0 = tqdm.tqdm(train_data_loader, smoothing=0, mininterval=1.0)
    now_iter = 0
    break_flag = False
    for epo in range(epochs):
        for i, (fields, target) in enumerate(tk0):
            model.train()
            fields, target = fields.to(device), target.to(device)
            y = model(fields)

            loss = criterion(y, target.float())

            model.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            now_iter += 1
            
            if now_iter % test_iter == 0:
                # testing
                auc, log_losses, _ = evaluation(model, test_data_loader, device, use_gauc=False)
                if not early_stopper.is_continuable(model, auc, log_losses):
                    print(f'validation: best auc: {early_stopper.best_auc}, best logloss: {early_stopper.best_logloss}')
                    break_flag = True
                    break
            
            if (i+1) % log_interval == 0:
                # display
                tk0.set_postfix(loss=total_loss / log_interval)
                total_loss = 0
                
        if break_flag:
            break             


# model testing
def evaluation(model, data_loader, device, use_gauc=False):
    model.eval()
    targets, predicts, user_id_list = list(), list(), list()
    with torch.no_grad():
        for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, target = fields.to(device), target.to(device)
            y = model(fields)
            user_id_list.extend(fields[:, 0].tolist())
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
    gauc = None
    if use_gauc:
        gauc = cal_group_auc(targets, predicts, user_id_list)
    targets = np.array(targets)
    predicts = np.array(predicts)
    return metrics.roc_auc_score(targets, predicts), metrics.log_loss(targets, predicts), gauc
    
    
auc_runs = []
logloss_runs = []
gauc_runs = []

for now_run in range(args.runs):
    if args.runs != 1:
        set_seed(now_run, args.use_gpu)

    print("###########now run: %d##############" % now_run)
    print("use dataset: " + dataset_name)

    print("now model: " + model_name)
    if model_name == 'macgnn':
        model = MacGNN(field_dims=field_dims, u_group_num=u_cluster_num, i_group_num=i_cluster_num,
                       embed_dim=embed_dim, recent_len=recent_len, tau=args.tau, device=device).to(device)
    else:
        raise Exception("no model selected!")

    criterion = torch.nn.BCELoss()
    
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    early_stopper = EarlyStopper(num_trials=trials, save_path=f'{model_name}_{dataset_name}.pt')

    
    train(model, optimizer, train_data_loader, test_data_loader, criterion, device, early_stopper, epochs=args.epoch, test_iter=args.test_iter, log_interval=20)
    
    model = torch.load(f'{model_name}_{dataset_name}.pt').to(device)
    auc, log_losses, gauc = evaluation(model, test_data_loader, device, use_gauc=True)
    
    print(f'test auc: {auc}, test logloss: {log_losses}, test gauc: {gauc}')
    auc_runs.append(auc)
    logloss_runs.append(log_losses)
    gauc_runs.append(gauc)


auc_mean, auc_std = np.mean(np.array(auc_runs), axis=0), np.std(np.array(auc_runs), axis=0)
logloss_mean, logloss_std = np.mean(np.array(logloss_runs), axis=0), np.std(np.array(logloss_runs), axis=0)
gauc_mean, gauc_std = np.mean(np.array(gauc_runs), axis=0), np.std(np.array(gauc_runs), axis=0)

print("Test AUC: "+str(auc_mean)+" ± "+str(auc_std))
print("Test GAUC: "+str(gauc_mean)+" ± "+str(gauc_std))
print("Test Logloss: "+str(logloss_mean)+" ± "+str(logloss_std))
