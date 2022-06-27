import torch
import torch.nn as nn
import torchvision
import numpy as np
import itertools
from subset import Subset
import torch.utils.data as Data
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from newloss import LogEntropyLoss
from torchvision import transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

sub = Subset(10)
batch_size = 100
learning_rate = 0.000001


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


train_data = torchvision.datasets.FashionMNIST(root='.', train=True, download=True)
test_data = torchvision.datasets.FashionMNIST(root='.', train=False, download=True)

X_train = train_data.data
X_test = test_data.data
y_train = train_data.targets.numpy()
y_test = test_data.targets.numpy()

X_train_true = np.zeros(shape=(60000, 784), dtype=float)
X_test_true = np.zeros(shape=(10000, 784), dtype=float)

for index, sample in enumerate(X_train):
    X_train_true[index] = ((sample.numpy().reshape(-1)).astype(float)/255)
for index, sample in enumerate(X_test):
    X_test_true[index] = sample.numpy().reshape(-1).astype(float)/255

X_train_true, validation_X, y_train, validation_y = train_test_split(X_train_true, y_train, test_size=0.1, random_state=42)


#train = X_train_true
test_labels = []

y_train_obfuscated = []

for y in y_train:
    index = y.argmax()
    y_train_obfuscated.append(sub.index_to_subset_multi_hot(index))

y_train_obfuscated = np.array(y_train_obfuscated)
input = torch.from_numpy(np.array(X_train_true)).to(torch.float32)
label = torch.from_numpy(np.array(y_train_obfuscated)).to(torch.int64)
val_input = torch.from_numpy(np.array(validation_X)).to(torch.float32)
val_labels = []
torch_dataset = Data.TensorDataset(input, label)
train_loader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=True)

for y in y_test:
    test_labels.append(y.argmax())

for y in validation_y:
    val_labels.append(y.argmax())
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


model = Activation_Net(28 * 28, 130, 26, 10).float()

criterion = LogEntropyLoss()
#criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
epochs = 128

if __name__ == '__main__':
    train_loss = []
    val_acc = []
    test_input = torch.from_numpy(np.array(X_test_true)).to(torch.float32)
    test_output = model(test_input)
    test_preds = []
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    # cm = confusion_matrix(test_preds, test_labels, conf_matrix)
    for epoch in range(epochs):
        print("Epoch:", epoch)
        sum_loss = 0.0
        with torch.autograd.set_detect_anomaly(True):
            for data in train_loader:
                img, label = data
                out = model(img)
                loss = criterion(out, label)
                print_loss = loss.data.item()
                sum_loss += print_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print("Loss:", sum_loss)
            # 监视验证集accuracy
            train_loss.append(sum_loss)
            val_output = model(val_input)
            val_preds = []
            val_output = val_output.cpu().detach().numpy()
            for y in val_output:
                val_preds.append(y.argmax())
            acc = accuracy_score(val_labels, val_preds)
            val_acc.append(acc)
            print("Accuracy:", acc)
    #绘制错误率曲线
    epoch_num = epochs
    x1 = range(0, epoch_num)
    x2 = range(0, epoch_num)
    y1 = val_acc
    y2 = train_loss
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'o-')
    plt.title('Indicators vs. Epoches')
    plt.ylabel('Validation Accuracy')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, '.-')
    plt.xlabel('epochs')
    plt.ylabel('Train Loss')
    plt.show()
    # plt.plot(x1, y1, 'o-')
    # plt.title('Train loss vs. epoches')
    # plt.ylabel('Train loss')
    # plt.show()
    # plt.savefig("both_loss.jpg")
    #测试集正确率,混淆矩阵
    test_input = torch.from_numpy(np.array(X_test_true)).to(torch.float32)
    test_output = model(test_input)
    test_preds = []
    test_output = test_output.cpu().detach().numpy()
    for y in test_output:
        test_preds.append(y.argmax())
    print("Accuracy:", accuracy_score(test_labels, test_preds))
    # classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    # plot_confusion_matrix(cm, classes)





