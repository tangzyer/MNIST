import torch
import torch.nn as nn
import pandas as pd
import itertools
from subset import Subset
import torch.utils.data as Data
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from newloss import LogEntropyLoss
from utils import load_mnist


sub = Subset(10)
batch_size = 100
learning_rate = 0.001
launchtime = '06271644'


#a = torchvision.datasets.FashionMNIST(root='.', download=True)



# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 784, 300, 10


class Activation_Net(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Activation_Net, self).__init__()
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(in_dim,n_hidden_1),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden_1, n_hidden_2),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden_2,out_dim),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.dense(x)
        return x





#model = Activation_Net(28 * 28, 130, 26, 10).float()



# 定义损失函数和优化器
#criterion = nn.MSELoss()
criterion = LogEntropyLoss()


epochs = 16

class nnclassifier(object):

    def __init__(self, max_iters=32):
        self.model = Activation_Net(28 * 28, 130, 26, 10).float()
        self.max_iters = max_iters
        self.loss = []

    def predict(self, X):
        X = torch.FloatTensor(X)
        output = self.model(X)
        pred = output.argmax(dim=1)
        return pred.detach().cpu().numpy()

    def fit(self, train_loader, index):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
        criterion = LogEntropyLoss()
        for epoch in range(self.max_iters):
            loss = self.train(train_loader, optimizer, criterion)
            self.loss.append(loss)
        pass
        df = pd.DataFrame({'loss': self.loss})
        df.to_csv(launchtime+'N:'+str(len(train_loader))+'_'+str(index)+'.csv')

    def train(self, train_loader, optimizer, train_criterion):
        #model.train()
        sum_loss = 0.0
        with torch.autograd.set_detect_anomaly(True):
            for data in train_loader:
                img, label = data
                out = self.model(img)
                loss = train_criterion(out, label)
                print_loss = loss.data.item()
                sum_loss += print_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return sum_loss







