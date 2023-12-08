#!/usr/bin/env python
# coding: utf-8
"""
This is a example of MAG generation and data processing on the MovieLens dataset.
"""

import numpy as np
import csv
import pandas as pd
import random
import pickle
import copy
from scipy.sparse import csr_matrix

np.random.seed(2023)
random.seed(2023)

path = './ratings.dat'
reviews_df  = pd.read_csv(path,sep='::',header=None)
reviews_df.columns = ['user_id','item_id','rating','timestamp']
reviews_df.loc[:,'rating'] = reviews_df['rating'].map(lambda x: 1 if x >= 4 else 0)

# reindex of the IDs
def build_map(df, col_name):
    key = sorted(df[col_name].unique().tolist())
    m = dict(zip(key, range(1,len(key)+1)))
    df.loc[:,col_name] = df[col_name].map(lambda x: m[x])
    return m, key


uid_map, uid_key = build_map(reviews_df, 'user_id')


path = './movies.dat'
meta_df  = pd.read_csv(path,sep='::',header=None)
meta_df.columns = ['item_id','title','genres']
meta_df = meta_df[['item_id', 'genres']]
meta_df.loc[:,'genres'] = meta_df['genres'].map(lambda x: x.split('|')[0])

vid_map, vid_key = build_map(meta_df, 'item_id')
cat_map, cat_key = build_map(meta_df, 'genres')

user_count, item_count, cate_count, example_count =    len(uid_map), len(vid_map), len(cat_map), reviews_df.shape[0]
print('user_count: %d\titem_count: %d\tcate_count: %d\texample_count: %d' %
      (user_count, item_count, cate_count, example_count))

meta_df = meta_df.sort_values('item_id')
meta_df = meta_df.reset_index(drop=True)

reviews_df['item_id'] = reviews_df['item_id'].map(lambda x: vid_map[x])
reviews_df = reviews_df.sort_values(['user_id', 'timestamp'])
reviews_df = reviews_df.reset_index(drop=True)

cate_list = [meta_df['genres'][i] for i in range(len(vid_map))]
cate_list = np.array(cate_list, dtype=np.int32)

cate_list = np.insert(cate_list, 0, 0)

with open('remap.pkl', 'wb') as f:
    pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL) # uid, iid, time(sorted)
    pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL) # cid of iid line
    pickle.dump((user_count, item_count, cate_count, example_count),
              f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((vid_key, cat_key, uid_key), f, pickle.HIGHEST_PROTOCOL)

pos_cnt, neg_cnt = 0, 0
for userId, hist in reviews_df.groupby('user_id'):
    movie_list = hist['item_id'].tolist()
    label_list = hist['rating'].tolist()

    pos_cnt += sum(label_list)
    neg_cnt += len(label_list) - sum(label_list)
    
train_hist_time, test_time = list(np.quantile(reviews_df.timestamp, [0.80, 0.90]))
train_df = reviews_df[reviews_df.timestamp <= test_time]
test_df = reviews_df[reviews_df.timestamp > test_time]

if train_df.shape[0]+test_df.shape[0] == reviews_df.shape[0]:
    print("Split Correct!")
else:
    print("Split Error!")

user_train_df = train_df
user_train_df = user_train_df.reset_index(drop=True)

item_train_df = train_df.sort_values(['item_id', 'timestamp'])
item_train_df = item_train_df.reset_index(drop=True)

train_hist_df = reviews_df[reviews_df.timestamp <= train_hist_time]

pos_train_hist_df = train_hist_df.drop(train_hist_df[train_hist_df['rating']==0].index)
pos_train_df = train_df.drop(train_df[train_df['rating']==0].index)

recent_len = 20

pos_user_train_hist_dict = {}
recent_user_train_hist_dict = {}
for user_id, hist in pos_train_hist_df.groupby('user_id'):
    item_list = hist['item_id'].tolist()
    pos_user_train_hist_dict[user_id] = item_list
    recent_user_train_hist_dict[user_id] = item_list[-recent_len:]


pos_user_train_dict = {}
recent_user_train_dict = {}
for user_id, hist in pos_train_df.groupby('user_id'):
    item_list = hist['item_id'].tolist()
    pos_user_train_dict[user_id] = item_list
    recent_user_train_dict[user_id] = item_list[-recent_len:]


pos_item_train_hist_dict = {}
recent_item_train_hist_dict = {}
for item_id, hist in pos_train_hist_df.groupby('item_id'):
    user_list = hist['user_id'].tolist()
    pos_item_train_hist_dict[item_id] = user_list
    recent_item_train_hist_dict[item_id] = user_list[-recent_len:]


pos_item_train_dict = {}
recent_item_train_dict = {}
for item_id, hist in pos_train_df.groupby('item_id'):
    user_list = hist['user_id'].tolist()
    pos_item_train_dict[item_id] = user_list
    recent_item_train_dict[item_id] = user_list[-recent_len:]


train_eval_df = reviews_df[(reviews_df.timestamp > train_hist_time) & (reviews_df.timestamp <= test_time)]

train_hist_row = []
train_hist_col = []
for user in list(pos_user_train_hist_dict.keys()):
    for item in pos_user_train_hist_dict[user]:
        train_hist_row.append(user)
        train_hist_col.append(item)

train_hist_edge = np.ones(len(train_hist_row))
train_hist_row = np.array(train_hist_row)
train_hist_col = np.array(train_hist_col)
train_hist_mat = csr_matrix((train_hist_edge, (train_hist_row, train_hist_col)), shape=(user_count+1, item_count+1))

i_cluster_list = cate_list

train_hist_ic_row = []
train_hist_ic_col = []

for item in range(len(i_cluster_list)):
    train_hist_ic_row.append(item)
    train_hist_ic_col.append(i_cluster_list[item])

train_hist_ic_edge = np.ones(len(train_hist_ic_row))
train_hist_ic_row = np.array(train_hist_ic_row)
train_hist_ic_col = np.array(train_hist_ic_col)
train_hist_ic_mat = csr_matrix((train_hist_ic_edge, (train_hist_ic_row, train_hist_ic_col)), shape=(item_count+1, len(cat_map)+1))

train_hist_u_1ord_mat = train_hist_mat*train_hist_ic_mat
train_hist_u_1ord_mat_dense = train_hist_u_1ord_mat.todense()


from sklearn import preprocessing
from sklearn.cluster import KMeans
train_hist_u_1ord_mat_normalized = preprocessing.normalize(train_hist_u_1ord_mat_dense, norm='l2')
cluster_fit = KMeans(n_clusters=20, random_state=0).fit(train_hist_u_1ord_mat_normalized)

with open('uc_cluster_kmeans.pkl', 'wb') as f:
    pickle.dump(cluster_fit.labels_, f, pickle.HIGHEST_PROTOCOL) # uid, iid, time(sorted)

with open('./uc_cluster_kmeans.pkl', 'rb') as f:
    u_cluster_list = pickle.load(f, encoding='latin1')


train_hist_uc_row = []
train_hist_uc_col = []

for user in range(len(u_cluster_list)):
    train_hist_uc_row.append(user)
    train_hist_uc_col.append(u_cluster_list[user])

train_hist_uc_edge = np.ones(len(train_hist_uc_row))
train_hist_uc_row = np.array(train_hist_uc_row)
train_hist_uc_col = np.array(train_hist_uc_col)
train_hist_uc_mat = csr_matrix((train_hist_uc_edge, (train_hist_uc_row, train_hist_uc_col)), shape=(user_count+1, len(set(u_cluster_list))))

train_hist_u_2ord_mat = train_hist_mat*(train_hist_mat.T*train_hist_uc_mat)
train_hist_i_2ord_mat = train_hist_mat.T*(train_hist_mat*train_hist_ic_mat)

train_hist_u_2ord_mat_dense = train_hist_u_2ord_mat.todense()
train_hist_i_2ord_mat_dense = train_hist_i_2ord_mat.todense()

train_hist_u_1ord_mat = train_hist_mat*train_hist_ic_mat
train_hist_i_1ord_mat = train_hist_mat.T*train_hist_uc_mat

train_hist_u_1ord_mat_dense = train_hist_u_1ord_mat.todense()
train_hist_i_1ord_mat_dense = train_hist_i_1ord_mat.todense()

train_hist_u_1ord_mat_dense_arr = train_hist_u_1ord_mat_dense.A
train_hist_u_2ord_mat_dense_arr = train_hist_u_2ord_mat_dense.A
train_hist_i_1ord_mat_dense_arr = train_hist_i_1ord_mat_dense.A
train_hist_i_2ord_mat_dense_arr = train_hist_i_2ord_mat_dense.A

train_eval_df = train_eval_df.reset_index(drop=True)

train_data = []
for idx, row in train_eval_df.iterrows():
    if idx % 100000 == 0:
        print("now have processed %d"%idx)
    now_user = row[0]
    now_item = row[1]
    if (now_user not in pos_user_train_hist_dict.keys()) or (now_item not in pos_item_train_hist_dict.keys()):
        continue
    now_label = row[2]
    now_user_1hop = train_hist_u_1ord_mat_dense_arr[now_user]
    now_user_2hop = train_hist_u_2ord_mat_dense_arr[now_user]
    now_item_1hop = train_hist_i_1ord_mat_dense_arr[now_item]
    now_item_2hop = train_hist_i_2ord_mat_dense_arr[now_item]
    user_recent = []
    user_recent.extend(recent_user_train_hist_dict[now_user])
    if len(user_recent) < recent_len:
        pad = [0 for i in range(recent_len-len(user_recent))]
        user_recent.extend(pad)
    item_recent = []
    item_recent.extend(recent_item_train_hist_dict[now_item])
    if len(item_recent) < recent_len:
        pad = [0 for i in range(recent_len-len(item_recent))]
        item_recent.extend(pad)
    now_train_seq = np.concatenate([np.array([now_user]), now_user_1hop, now_user_2hop, user_recent, np.array([now_item]), now_item_1hop, now_item_2hop, item_recent, np.array([now_label])], axis=0)
    train_data.append(now_train_seq)


train_data = np.array(train_data)
train_row = []
train_col = []
for user in list(pos_user_train_dict.keys()):
    for item in pos_user_train_dict[user]:
        train_row.append(user)
        train_col.append(item)

train_edge = np.ones(len(train_row))
train_row = np.array(train_row)
train_col = np.array(train_col)
train_mat = csr_matrix((train_edge, (train_row, train_col)), shape=(user_count+1, item_count+1))


train_uc_row = []
train_uc_col = []

train_ic_row = []
train_ic_col = []

for user in range(len(u_cluster_list)):
    train_uc_row.append(user)
    train_uc_col.append(u_cluster_list[user])

train_uc_edge = np.ones(len(train_uc_row))
train_uc_row = np.array(train_uc_row)
train_uc_col = np.array(train_uc_col)

for item in range(len(i_cluster_list)):
    train_ic_row.append(item)
    train_ic_col.append(i_cluster_list[item])

train_ic_edge = np.ones(len(train_ic_row))
train_ic_row = np.array(train_ic_row)
train_ic_col = np.array(train_ic_col)

train_uc_mat = csr_matrix((train_uc_edge, (train_uc_row, train_uc_col)), shape=(user_count+1, len(set(u_cluster_list))))
train_ic_mat = csr_matrix((train_ic_edge, (train_ic_row, train_ic_col)), shape=(item_count+1, len(cat_map)+1))

train_u_2ord_mat = train_mat*(train_mat.T*train_uc_mat)
train_i_2ord_mat = train_mat.T*(train_mat*train_ic_mat)

train_u_2ord_mat_dense = train_u_2ord_mat.todense()
train_i_2ord_mat_dense = train_i_2ord_mat.todense()

train_u_1ord_mat = train_mat*train_ic_mat
train_i_1ord_mat = train_mat.T*train_uc_mat

train_u_1ord_mat_dense = train_u_1ord_mat.todense()
train_i_1ord_mat_dense = train_i_1ord_mat.todense()

train_u_1ord_mat_dense_arr = train_u_1ord_mat_dense.A
train_u_2ord_mat_dense_arr = train_u_2ord_mat_dense.A
train_i_1ord_mat_dense_arr = train_i_1ord_mat_dense.A
train_i_2ord_mat_dense_arr = train_i_2ord_mat_dense.A


test_df = test_df.reset_index(drop=True)

test_data = []
for idx, row in test_df.iterrows():
    if idx % 100000 == 0:
        print("now have processed %d"%idx)
    now_user = row[0]
    now_item = row[1]
    if (now_user not in pos_user_train_dict.keys()) or (now_item not in pos_item_train_dict.keys()):
        continue
    now_label = row[2]
    now_user_1hop = train_u_1ord_mat_dense_arr[now_user]
    now_user_2hop = train_u_2ord_mat_dense_arr[now_user]
    now_item_1hop = train_i_1ord_mat_dense_arr[now_item]
    now_item_2hop = train_i_2ord_mat_dense_arr[now_item]
    user_recent = []
    user_recent.extend(recent_user_train_dict[now_user])
    if len(user_recent) < recent_len:
        pad = [0 for i in range(recent_len-len(user_recent))]
        user_recent.extend(pad)
    item_recent = []
    item_recent.extend(recent_item_train_dict[now_item])
    if len(item_recent) < recent_len:
        pad = [0 for i in range(recent_len-len(item_recent))]
        item_recent.extend(pad)
    now_test_seq = np.concatenate([np.array([now_user]), now_user_1hop, now_user_2hop, user_recent, np.array([now_item]), now_item_1hop, now_item_2hop, item_recent, np.array([now_label])], axis=0)
    test_data.append(now_test_seq)


test_data = np.array(test_data)
test_data.shape

u_cluster_num = len(set(u_cluster_list))
i_cluster_num = len(cat_map)

with open('../ml-10m.pkl', 'wb') as f:
    pickle.dump(train_data, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_data, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(u_cluster_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(i_cluster_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((user_count, item_count, cate_count, u_cluster_num, i_cluster_num), f, pickle.HIGHEST_PROTOCOL)

