import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn.functional as F
import os
from torch.utils.data import Dataset, TensorDataset, DataLoader


PATH = './'

allfile = os.listdir(PATH)
csvname = list(filter(lambda x: (x == 'all_chiller.csv'), allfile))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(16, 24)
        self.layer2 = nn.Linear(24, 40)
        self.layer3 = nn.Linear(40, 12)
        self.out = nn.Linear(12, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = self.out(x)
        return x


batch_size = 1
# accumulation_steps = 7

loss_func = nn.L1Loss()

for filename in csvname:
    cp1_loss = []
    cp1_update_loss = []
    direct_loss = []
    update_loss = []
    df = pd.read_csv(PATH + '\\' + filename)
    df = df.drop(labels=417)
    #df = df.drop(labels=30)
    test_data = df.drop(range(0, 30))
    test_x = np.array(test_data[test_data.columns.difference(['时间', 'cop', '冷水机组(电表)有功电度'])])
    test_x = torch.FloatTensor(test_x.astype(float))
    test_y = torch.FloatTensor(np.array(test_data['cop']).astype(float))

    dataset = TensorDataset(test_x, test_y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    net2 = Net()
    net2.load_state_dict(torch.load('./' + filename + '_30h.pt'))
    optimizer = torch.optim.Adam(net2.parameters(), lr=0.01)


    for i, (inputs, labels) in enumerate(dataloader):
        output2 = net2(inputs)

        loss2 = loss_func(output2, labels)
        update_loss.append(loss2.item())
        optimizer.zero_grad()
        loss2.backward()
        optimizer.step()

        '''
        update_loss.append(loss_func(output2, labels))
        loss2 = loss_func(output2, labels) / accumulation_steps
        loss2.backward()


        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        '''

    plt.title(filename[:-4])
    plt.xlabel('Hour')
    plt.ylabel('MAELoss')
    #plt.ylim((0, 1))

    plt.plot(update_loss, label='Updated_Model')
    plt.legend(['Updated_Model'])
    plt.show()

    #cp1_loss_npy = np.array(cp1_loss)
    #update_loss_npy = np.array(update_loss)
    #update_loss_npy = np.array(update_loss)

    #np.save('./CP1_Model_Loss.npy', cp1_loss_npy)
    #np.save('./Update_Model_MAELoss_1000h.npy', update_loss_npy)
    #np.save('./Updated_Model_Loss.npy', update_loss_npy)



