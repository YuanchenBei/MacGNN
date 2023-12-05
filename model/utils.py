#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import pandas as pd
import numpy as np
import pickle
from torch.utils.data import DataLoader


# In[2]:


class DatasetBuilder(torch.utils.data.Dataset):

    def __init__(self, data, user_count, item_count):
        self.x = torch.tensor(data[:, :-1], dtype=torch.long)
        self.y = torch.tensor(data[:, -1], dtype=torch.float).unsqueeze(1)
        self.field_dims = [user_count, item_count]
    
    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]


# In[ ]:




