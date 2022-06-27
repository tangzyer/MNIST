import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from subset import Subset
import torch.utils.data as Data
from torch.utils.data import DataLoader
import torch
from sklearn.model_selection import train_test_split

batch_size = 100

def generate_gaussian_data(k_cluster=4, dimension=2, n=100, shared_cov=False):
    data_X = []
    data_Y = []
    cov = np.diag(np.random.uniform(0, 3, dimension))
    for i in range(k_cluster):
        center = np.random.uniform(0, 10, dimension)
        print('center', center)
        if not shared_cov:
            cov = np.random.uniform(0, 3, dimension)
            cov = np.diag(cov)
        print('cov', cov)
        data_i = np.random.multivariate_normal(center, cov, n)
        plt.scatter(data_i[:, 0], data_i[:, 1], marker='o')
        data_X = data_X + data_i.tolist()
        data_Y = data_Y + [i]*n
    #plt.show()
    return data_X, data_Y


def generate_subset_test_data(X, Y, k_cluster, test_rate=0.2):
    sub = Subset(k_cluster)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_rate)
    y_train_sub = []
    for y in y_train:
        y_train_sub.append(sub.index_to_subset_multi_hot(y))
    return X_train, X_test, y_train_sub, y_test


def generate_centers_covs(k_cluster=4, dimension=2, shared_cov=False):
    centers = []
    covs= []
    cov = np.diag(np.random.uniform(0, 3, dimension))
    for i in range(k_cluster):
        center = np.random.uniform(0, 10, dimension)
        print('center', center)
        if not shared_cov:
            cov = np.random.uniform(0, 3, dimension)
            cov = np.diag(cov)
        print('cov', cov)
        centers.append(center)
        covs.append(cov)
    return centers, covs


def generate_gaussian_from_centers_covs(centers, covs, n):
    data_X = []
    data_Y = []
    for index, center in enumerate(centers):
        cov = covs[index]
        data_i = np.random.multivariate_normal(center, cov, n)
        plt.scatter(data_i[:, 0], data_i[:, 1], marker='o', s=0.5)
        data_X = data_X + data_i.tolist()
        data_Y = data_Y + [index] * n
    plt.savefig('2dsamples.png')
    return data_X, data_Y


def load_mnist(data_size = 0.1):
    sub = Subset(10)
    npzfile = np.load('mnist.npz')
    X_train = npzfile['X_train']
    X_test = npzfile['X_test']
    y_train = npzfile['y_train']
    y_test = npzfile['y_test']
    _, X_train, _,  y_train = train_test_split(X_train, y_train, test_size=data_size)
    _, X_test, _, y_test = train_test_split(X_test, y_test, test_size=data_size)
    X_train_true = np.zeros(shape=(len(X_train), 784), dtype=float)
    X_test_true = np.zeros(shape=(len(X_test), 784), dtype=float)
    for index, sample in enumerate(X_train):
        X_train_true[index] = ((sample.reshape(-1)).astype(float) / 255)
    for index, sample in enumerate(X_test):
        X_test_true[index] = sample.reshape(-1).astype(float) / 255
    test_labels = []
    y_train_obfuscated = []
    for y in y_train:
        index = y.argmax()
        y_train_obfuscated.append(sub.index_to_subset_multi_hot(index))
    y_train_obfuscated = np.array(y_train_obfuscated)
    for y in y_test:
        test_labels.append(y.argmax())
    return X_train_true, y_train_obfuscated , X_test_true, test_labels


def load_fromXY(X_train_true, y_train_obfuscated):
    input = torch.from_numpy(np.array(X_train_true)).to(torch.float32)
    label = torch.from_numpy(np.array(y_train_obfuscated)).to(torch.int64)
    torch_dataset = Data.TensorDataset(input, label)
    train_loader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=True)
    return train_loader