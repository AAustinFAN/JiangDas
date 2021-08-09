import pickle

import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
import preprocessor


input_dim = config.input_dim
hidden_size = config.hidden_size
batch_size = config.batch_size
train_ratio = config.train_ratio
learningRate = config.learningRate
epoch = config.epoch

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden, bias=1)  # hidden layer
        self.predict = torch.nn.Linear(n_hidden, 1, bias=1)  # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))  # activation function for hidden layer
        x = self.predict(x)  # linear output
        return x

class TrainSet(Dataset):
    def __init__(self, datax, datay):
        self.data, self.label = datax, datay

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


datax,datay =preprocessor.read()
print('data size is ', datax.shape, datay.shape)


trainset = TrainSet(datax[0:200], datay[0:200])
testset = TrainSet(datax[200:477], datay[200:477])

ITER = 20
test_batch = 5

loss_result_all = np.zeros((ITER, testset.data.shape[0]//test_batch+1))
for II in range(ITER):
    print(II)


    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False)

    Seq_model = Net(n_feature=input_dim, n_hidden=hidden_size)  # define the network
    optimizer = torch.optim.SGD(Seq_model.parameters(), lr=learningRate)
    loss_func = nn.L1Loss()

    for step in range(epoch):
        Loss_list = []
        prelist = []
        for x, y in trainloader:
            x = x.to(torch.float32)
            y = y.to(torch.float32)

            prediction = Seq_model(x)  # input x and predict based on x
            for x in prediction:
                prelist.append(x.detach().numpy())
            # print(prediction,y)
            loss = loss_func(prediction, y)  # must be (1. nn output, 2. target)
            Loss_list.append(loss.item())
            optimizer.zero_grad()  # clear gradients for next train
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

    print('train ending')
    testloader = DataLoader(testset, batch_size=test_batch, shuffle=False)

    loss_func = nn.L1Loss()
    Loss_list = []
    prelist = []

    for x, y in testloader:
        x = x.to(torch.float32)
        y = y.to(torch.float32)

        prediction = Seq_model(x)  # input x and predict based on x
        for pre in prediction:
            prelist.append(pre.detach().numpy())
        # print(prediction,y)
        loss = loss_func(prediction, y)  # must be (1. nn output, 2. target)
        Loss_list.append(loss.item())
        # print('Epoch:{}, Loss:{:.5f}'.format(1, np.mean(Loss_list)))

        # # todo update
        optimizer.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

    loss_result_all[II, :] = np.array(Loss_list)


mean_iter_loss = np.mean(loss_result_all, axis=0)
print(mean_iter_loss)
plt.plot(range(mean_iter_loss.__len__()), mean_iter_loss)
plt.title("inference round= %s, var=%s, std=%s" % (ITER, loss_result_all.var(), loss_result_all.std()))
plt.show()

# with open('mean_loss.pkl', 'wb') as f:
#     pickle.dump(mean_iter_loss, f)
#     f.close()

with open('mean_loss_update.pkl', 'wb') as f:
    pickle.dump(mean_iter_loss, f)
    f.close()