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

# 分割
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
learning_rate = 0.3
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.utils.data as Data

train_loss = []

# 将数据转换为torch的dataset格式
input = torch.from_numpy(np.array(X_train)).to(torch.float32)
label = torch.from_numpy(np.array(train_labels)).to(torch.float32)
torch_dataset = Data.TensorDataset(input, label)
train_loader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=True)


class Cifar_Net(torch.nn.Module):
    def __init__(self, mode="hidden0",
                 hidden_activation=torch.nn.ReLU):
        super(Cifar_Net, self).__init__()
        self.mode = mode
        if mode == "hidden0":
            self.net = torch.nn.Sequential(
                torch.nn.Linear(3072, 3),
                torch.nn.Softmax(dim=1)
            )
        elif mode == "hidden1":
            self.net = torch.nn.Sequential(
                torch.nn.Linear(3072, 512),
                hidden_activation(),
                torch.nn.Linear(512, 3),
                torch.nn.Softmax(dim=1)
            )
        elif mode == "hidden2":
            self.net = torch.nn.Sequential(
                torch.nn.Linear(3072, 512),
                hidden_activation(),
                torch.nn.Linear(512, 512),
                hidden_activation(),
                torch.nn.Linear(512, 3),
                torch.nn.Softmax(dim=1)
            )
        elif mode == "hidden3":
            self.net = torch.nn.Sequential(
                torch.nn.Linear(3072, 512),
                hidden_activation(),
                torch.nn.Linear(512, 512),
                hidden_activation(),
                torch.nn.Linear(512, 512),
                hidden_activation(),
                torch.nn.Linear(512, 3),
                torch.nn.Softmax(dim=1)
            )
        elif mode == "cnn":
            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0),
                hidden_activation(),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                torch.nn.Dropout(p=0.25))
            self.fc = torch.nn.Sequential(
                torch.nn.Linear(14400, 512),
                hidden_activation(),
                torch.nn.Dropout(p=0.5),
                torch.nn.Linear(512, 6),
                torch.nn.Softmax(dim=1)
            )
        elif mode == "cnntest":
            self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
            self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
            self.fc1 = torch.nn.Linear(in_features=16 * 5 * 5, out_features=120)
            self.fc2 = torch.nn.Linear(in_features=120, out_features=84)
            self.fc3 = torch.nn.Linear(in_features=84, out_features=6)
            self.sm = torch.nn.Softmax(dim=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.001)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.1)

    def forward(self, x):
        if self.mode == "cnn":
            x = self.conv(x)
            x = x.flatten(start_dim=1)
            x = self.fc(x)
            return x
        elif self.mode == "cnntest":
            x = self.pool1(torch.nn.functional.relu(self.conv1(x)))
            x = self.pool1(torch.nn.functional.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)  # reshape tensor
            x = torch.nn.functional.relu(self.fc1(x))
            x = torch.nn.functional.relu(self.fc2(x))
            x = self.sm(self.fc3(x))
            return x
        else:
            x = self.net(x)
            return x


epochs = 2

# modes = ["hidden2","hidden3"]
modes = ["cnntest"]
import matplotlib.pyplot as plt


def testMode(mode):
    model = Cifar_Net(mode)
    if torch.cuda.is_available():
        model = model.cuda()
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    validation_loss = []
    validation_accuracy = []
    for epoch in range(epochs):
        sum_loss = 0.0
        for data in train_loader:
            data_size = data[0].shape[0]
            img, label = data
            if mode == "cnntest":
                img = img.reshape(data_size, 3, 32, 32)
            elif mode != "cnn":
                img = img.reshape(-1, 3072)
            else:
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
        # 监视验证集
        print('Epoch', epoch, 'Loss:', sum_loss)
    model.eval()
    with torch.no_grad():
        test_input = torch.from_numpy(np.array(x_test)).to(torch.float32)
        if mode != "cnn":  # 得到损失函数
            test_inputs = test_input.view(-1, 3072)
        else:
            test_inputs = test_input.reshape(3000, 3, 32, 32)
        if torch.cuda.is_available():
            test_inputs = Variable(test_inputs).cuda()
        test_outputs = model(test_inputs)
        test_preds = np.argmax(test_outputs.cpu().detach().numpy(), axis=1)
    print(mode, (accuracy_score(test_preds, test_labels)))
    epoch_num = epochs
    x1 = range(0, epoch_num)
    x2 = range(0, epoch_num)
    y1 = validation_accuracy
    y2 = train_loss
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'o-')
    plt.title('Validation indicators vs. epoches')
    plt.ylabel('Validation accuracy')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, '.-')
    plt.xlabel('epochs')
    plt.ylabel('Train loss')
    # plt.savefig('Validation indicators '+mode+".png")
    plt.show()
    # 测试集正确率




for mode in modes:
    testMode(mode)
