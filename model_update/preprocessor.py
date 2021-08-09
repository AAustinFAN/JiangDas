import pandas as pd
import numpy as np

def read():
    data = pd.read_csv('C:/Users/Administrator/Documents/WeChat Files/wxid_tstamvlvsn8411/FileStorage/File/2021-08/数据/数据/all_chiller.csv',usecols=['冷水机组冷冻供水温度', '冷水机组冷冻回水温度', '冷水机组负荷比','cop'])
    # datay = pd.read_csv('C:/Users/Administrator/Documents/WeChat Files/wxid_tstamvlvsn8411/FileStorage/File/2021-08/数据/数据/all_chiller.csv',usecols=['cop'])
    data = data.drop(data[data['cop'] > 15].index)

    datax = data[['冷水机组冷冻供水温度', '冷水机组冷冻回水温度', '冷水机组负荷比']]
    datay = data['cop']
    datax= datax.values
    datay= datay.values
    print(datax.shape,datay.shape)
    return datax,datay

read()