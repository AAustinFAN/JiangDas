import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn.functional as F
import os
from torch.utils.data import Dataset, TensorDataset, DataLoader
from string import punctuation
import re


allfile = os.listdir('./')
csvname = list(filter(lambda x: (x[0] == 'L' and x[-4:] == '.csv'), allfile))

remove = '0123456789-'


def changeColumnName(column_list):
    temp = [re.sub('[a-zA-Z]', '', column) for column in column_list]
    temp = [re.sub('[\d]', '', column) for column in temp]
    temp = [re.sub('[-#]', '', column) for column in temp]
    temp[9] = temp[9] + 'Ua'
    temp[10] = temp[10] + 'Ub'
    temp[11] = temp[11] + 'Uc'
    temp[12] = temp[12] + 'Ia'
    temp[13] = temp[13] + 'Ib'
    temp[14] = temp[14] + 'Ic'
    return temp

df_list = []
for filename in csvname:
    df = pd.read_csv('./' + filename)
    df['cop'].replace(0.0, np.nan, inplace=True)
    df = df.dropna(axis=0)
    column_values = df.columns.values
    new_column_values = changeColumnName(column_values[1:-1])
    new_column_values.insert(0, '时间')
    new_column_values.append('cop')
    df.columns = new_column_values
    df_list.append(df)


total_df = pd.concat(df_list, axis=0, ignore_index=True)
total_df = total_df.set_index('时间')
total_df = total_df.sort_index()
total_df = total_df.reset_index()
total_df.to_csv('./all_chiller.csv', index=False)

