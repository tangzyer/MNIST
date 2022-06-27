import torch.nn as nn
import torch
from torch.nn import Module
import torch.nn.functional as F
from torch.autograd import Variable

class mysoftmax(Module):

    def __init__(self, n_features, n_classes):
        super().__init__()
        self.fc = nn.Linear(n_features, n_classes)

    def __call__(self, x):
        return self.fc(x)


class entropyloss(Module):

    def __init__(self):
        super(entropyloss, self).__init__()

    def forward(self, logits, target):
        # logits format:[[a,b,c,d],[e,f,g,h]]
        # target format:[[1,1,1,0],[0,1,1,0]]
        logits = logits.contiguous()  # [NHW, C]
        logits = F.softmax(logits, 1)
        mask = target.bool()
        logits = logits.masked_fill(~mask, 0)
        loss = torch.sum(logits, dim=1)
        loss = torch.log(loss)
        loss = -1 * loss
        return (loss.mean())

class nnclassifier(object):

    def __init__(self, n_features, n_classes, max_iters = 999, threshold=0.01, lr=0.03):
        self.model = mysoftmax(n_features, n_classes)
        self.max_iters = max_iters
        self.threshold = threshold
        self.lr = lr

    def predict(self, X):
        X = torch.FloatTensor(X)
        output = self.model(X)
        pred = output.argmax(dim=1)
        return pred.detach().cpu().numpy()

    def fit(self, X, Y):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4)
        criterion = entropyloss()
        X = torch.FloatTensor(X)
        Y = torch.FloatTensor(Y)
        for epoch in range(self.max_iters):
            loss = self.train(X, Y, optimizer, criterion)
            if loss < self.threshold:
                break
        pass
    def train(self, X, Y, optimizer, train_criterion):
        #model.train()
        X = Variable(X)
        Y = Variable(Y)
        out = self.model(X)
        loss = train_criterion(out, Y)
        print_loss = loss.data.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return print_loss