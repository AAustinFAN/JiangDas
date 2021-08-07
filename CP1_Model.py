import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn.functional as F


train_data = np.array(pd.read_csv('./CP1.csv'))
train_unknown_list = []

for x in range(train_data.shape[0]):
    if train_data[x][-1] == 0.0:
        train_unknown_list.append(x)

train_data = np.delete(train_data, train_unknown_list, axis=0)
train_x = torch.FloatTensor(train_data[:, (6, 5, 3)].astype(float))
train_y = torch.FloatTensor(train_data[:, -1].astype(float))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(3, 24)
        self.layer2 = nn.Linear(24, 40)
        self.layer3 = nn.Linear(40, 12)
        self.out = nn.Linear(12, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = self.out(x)
        return x


net = Net()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
loss_func = nn.L1Loss()

for epoch in range(1, 1001):
    output = net(train_x)
    output = output.squeeze()
    loss = loss_func(output, train_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('Epoch ', epoch, ': ', loss.item())

torch.save(net.state_dict(), './CP1Model.pt')