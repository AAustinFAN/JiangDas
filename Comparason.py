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


class CP1_Net(nn.Module):
    def __init__(self):
        super(CP1_Net, self).__init__()
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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(16, 24)
        # self.layer2 = nn.Linear(24, 40)
        self.layer3 = nn.Linear(24, 12)
        self.out = nn.Linear(12, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        # x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = self.out(x)
        return x


batch_size = 1
# accumulation_steps = 7
cp1_net = CP1_Net()
cp1_net.load_state_dict(torch.load('./CP1Model.pt'))
cp1_net.eval()

cp1_update_net = CP1_Net()
cp1_update_net.load_state_dict(torch.load('./CP1Model.pt'))
cp1_optimizer = torch.optim.Adam(cp1_update_net.parameters(), lr=0.01)

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
    test_cp1_x = torch.FloatTensor(np.array(test_data[['冷水机组冷冻供水温度', '冷水机组冷冻回水温度', '冷水机组负荷比']]).astype(float))
    test_x = np.array(test_data[test_data.columns.difference(['时间', 'cop', '冷水机组(电表)有功电度'])])
    test_x = torch.FloatTensor(test_x.astype(float))
    test_y = torch.FloatTensor(np.array(test_data['cop']).astype(float))

    cp1_dataset = TensorDataset(test_cp1_x, test_y)
    cp1_loader = DataLoader(cp1_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    dataset = TensorDataset(test_x, test_y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    net1 = Net()
    net1.load_state_dict(torch.load('./' + filename + '_30h.pt'))
    net1.eval()

    net2 = Net()
    net2.load_state_dict(torch.load('./' + filename + '_30h.pt'))
    optimizer = torch.optim.Adam(net2.parameters(), lr=0.01)

    for i, (inputs, labels) in enumerate(cp1_loader):
        outputs = cp1_net(inputs)
        outputs = outputs.view(-1,1)
        labels = labels.view(-1,1)
        loss = loss_func(outputs, labels)
        cp1_loss.append(loss.item())

        outputs_update = cp1_update_net(inputs)
        loss_update = loss_func(outputs_update, labels)
        cp1_optimizer.zero_grad()
        loss_update.backward()
        cp1_optimizer.step()
        cp1_update_loss.append((loss_update.item()))


    for i, (inputs, labels) in enumerate(dataloader):
        output1 = net1(inputs)
        output2 = net2(inputs)

        output1 = output1.view(-1,1)
        labels = labels.view(-1,1)
        loss1 = loss_func(output1, labels)

        direct_loss.append(loss1.item())

        output2 = output2.view(-1,1)
        labels = labels.view(-1,1)
        loss2 = loss_func(output2, labels)

        optimizer.zero_grad()
        loss2.backward()
        optimizer.step()
        update_loss.append(loss2.item())

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

    plt.plot(cp1_loss, label='CP1_Model')
    plt.plot(cp1_update_loss, label='CP1_Update_Model')
    plt.plot(direct_loss, label='Original_Model')
    plt.plot(update_loss, label='Updated_Model')
    plt.legend(['CP1_Model', 'CP1_Update_Model', 'Original_Model', 'Updated_Model'])
    plt.show()

    plt.plot(direct_loss, label='Original_Model', color='red')
    plt.plot(update_loss, label='Updated_Model')
    #plt.plot(update_loss[0:15], label='Updated_Model')
    plt.legend(['Original_Model', 'Updated_Model'])
    plt.show()
    #print(direct_loss[0:20])
    #print(update_loss[0:20])
    #cp1_loss_npy = np.array(cp1_loss)
    #update_loss_npy = np.array(update_loss)
    #update_loss_npy = np.array(update_loss)

    #np.save('./CP1_Model_Loss.npy', cp1_loss_npy)
    #np.save('./Update_Model_MAELoss_1000h.npy', update_loss_npy)
    #np.save('./Updated_Model_Loss.npy', update_loss_npy)



