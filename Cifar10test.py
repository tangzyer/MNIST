import torch.nn as nn
import numpy as np
import math
from sklearn.metrics import accuracy_score
from subset import Subset
from torchvision import datasets
import torch

sub = Subset(6)

train_data = datasets.CIFAR10(root='./', train=True, download=True)
test_data = datasets.CIFAR10(root='./', train=False, download=True)

x_train = train_data.data
x_test = test_data.data
y_train = train_data.targets
y_test = test_data.targets


X_train_true = np.zeros(shape=(50000, 32, 32, 3), dtype=float)
X_test_true = np.zeros(shape=(10000, 32, 32, 3), dtype=float)

# 数据预处理
for index, sample in enumerate(x_train):
    X_train_true[index] = sample.astype(float) / 255
for index, sample in enumerate(x_test):
    X_test_true[index] = sample.astype(float) / 255

from sklearn.model_selection import train_test_split

train_X, _, train_y, _ = train_test_split(X_train_true, y_train, test_size=0.8, random_state=42)
test_X, _, test_y, _ = train_test_split(X_test_true, y_test, test_size=0.8, random_state=42)

test_labels = []
train_labels = []
X_train = []
X_test = []

for index, y in enumerate(train_y):
    if y < 6:
        X_train.append(train_X[index])
        train_labels.append(sub.index_to_obfuscated(y))

for index, y in enumerate(test_y):
    if y < 6:
        X_test.append(test_X[index])
        test_labels.append(y)


batch_size = 100
learning_rate = 0.001

from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.utils.data as Data

train_loss = []

# 将数据转换为torch的dataset格式
input = torch.from_numpy(np.array(X_train)).to(torch.float32)
label = torch.from_numpy(np.array(train_labels)).to(torch.float32)
torch_dataset = Data.TensorDataset(input, label)
train_loader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=True)

epochs = 2

from torchvision import models

def test():
    model = models.resnet18(pretrained=False)
    fc_in = model.fc.in_features  # 获取全连接层的输入特征维度
    model.fc = nn.Linear(fc_in, 6)
    if torch.cuda.is_available():
        model = model.cuda()
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        sum_loss = 0.0
        for data in train_loader:
            data_size = data[0].shape[0]
            img, label = data
            img = img.reshape(data_size, 3, 32, 32)
            if torch.cuda.is_available():
                img = Variable(img).cuda()
                label = Variable(label).cuda()
            else:
                img = Variable(img)
                label = Variable(label)
            out = model(img)
            loss = criterion(out, label)
            print_loss = loss.data.item()
            sum_loss += print_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Epoch', epoch, 'Loss:', sum_loss)
    model.eval()
    with torch.no_grad():
        test_input = torch.from_numpy(np.array(x_test)).to(torch.float32)
        test_inputs = test_input.reshape(1207, 3, 32, 32)
        if torch.cuda.is_available():
            test_inputs = Variable(test_inputs).cuda()
        test_outputs = model(test_inputs)
        test_preds = np.argmax(test_outputs.cpu().detach().numpy(), axis=1)
    print((accuracy_score(test_preds, test_labels)))