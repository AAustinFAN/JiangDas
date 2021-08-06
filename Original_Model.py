import torch.nn.functional as F
import os
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PATH = './'

allfile = os.listdir(PATH)
csvname = list(filter(lambda x: (x == 'all_chiller.csv'), allfile))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(16, 24)
        # self.layer2 = nn.Linear(24, 40)
        self.layer3 = nn.Linear(24, 12)
        self.out = nn.Linear(12, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        #x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = self.out(x)
        return x

for filename in csvname:
    df = pd.read_csv(PATH + '\\' + filename)
    train_data = df.head(30)
    test_data = df.drop(range(0, 30))
    train_x = np.array(train_data[train_data.columns.difference(['时间', 'cop', '冷水机组(电表)有功电度'])])
    train_x = torch.FloatTensor(train_x.astype(float))
    train_y = torch.FloatTensor(np.array(train_data['cop']).astype(float))
    #test_x = np.array(test_data[test_data.columns.difference(['record_timestamp', 'COP', 'COPD'])])
    #test_x = torch.FloatTensor(test_x.astype(float))
    #test_y = torch.FloatTensor(np.array(test_data['COP']).astype(float))

    net = Net()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    loss_func = nn.L1Loss()

    for epoch in range(1, 1001):
        output = net(train_x)
        output = output.squeeze()
        loss = loss_func(output, train_y)
        print(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(net.state_dict(), './' + filename + '_30h.pt')