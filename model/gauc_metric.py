#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from collections import defaultdict
from sklearn.metrics import roc_auc_score
import numpy as np

'''
calculate group_auc and cross_entropy_loss(log loss for binary classification)
@author: Qiao
'''


def cal_group_auc(labels, preds, user_id_list):
    """Calculate group auc"""

    print('*' * 50)
    if len(user_id_list) != len(labels):
        raise ValueError(
            "impression id num should equal to the sample num," \
            "impression id num is {0}".format(len(user_id_list)))
    group_score = defaultdict(lambda: [])
    group_truth = defaultdict(lambda: [])
    for idx, truth in enumerate(labels):
        user_id = user_id_list[idx]
        score = preds[idx]
        truth = labels[idx]
        group_score[user_id].append(score)
        group_truth[user_id].append(truth)

    group_flag = defaultdict(lambda: False)
    for user_id in set(user_id_list):
        truths = group_truth[user_id]
        flag = False
        for i in range(len(truths) - 1):
            if truths[i] != truths[i + 1]:
                flag = True
                break
        group_flag[user_id] = flag

    impression_total = 0
    total_auc = 0.0
    #
    for user_id in group_flag:
        if group_flag[user_id]:
            auc = roc_auc_score(np.asarray(group_truth[user_id]), np.asarray(group_score[user_id]))
            total_auc += auc * len(group_truth[user_id])
            impression_total += len(group_truth[user_id])
    group_auc = float(total_auc) / impression_total
    group_auc = round(group_auc, 5)
    return group_auc


def eval_by_freq(labels, preds, user_id_list, user2freq):
    """Calculate auc grouped by user frequency"""
    print('*' * 50)
    group_score = defaultdict(lambda: [])
    group_truth = defaultdict(lambda: [])
    #freq_labels = ['<10', '10~20', '20~30', '30~40', '40~50', '50~60', '60~70', '70~80', '80~90', '90~100', '>100']
    freq_labels = ['<20', '20~40', '40~60', '60~80', '80~100', '>100']
    for idx, truth in enumerate(labels):
        user_id = user_id_list[idx]
        score = preds[idx]
        truth = labels[idx]
        group_score[user2freq[user_id]].append(score)
        group_truth[user2freq[user_id]].append(truth)
    
    group_auc = defaultdict(lambda: [])
    for freq in freq_labels:
        freq_auc = roc_auc_score(np.asarray(group_truth[freq]), np.asarray(group_score[freq]))
        group_auc[freq].append(freq_auc)
    return group_auc