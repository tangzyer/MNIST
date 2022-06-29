from utils import *
from gmm import MyGMM
from gmm import bayesian
from LDA import MyLDA
from knn import knn_subsetlabels
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import warnings
import math
import pandas as pd
from test import nnclassifier
import numpy as np
from test import launchtime
import os

k_cluster = 10
acc_knn = []
acc_lda_gmm = []
#acc_bayes = []
acc_lda_mom = []
acc_nn = []
iters = [0.1, 0.3, 0.5, 0.7, 0.9]
a = np.exp(200)*np.ones(shape=(1,1))

for i in iters:
    acc_nn.append([])
    n = 50000 * i
    for j in range(3):
        X_train, y_train_sub, X_test, y_test = load_mnist(i)
        data_loader = load_fromXY(X_train, y_train_sub)
        model = nnclassifier()
        model.fit(data_loader, j)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_pred, y_test)
        acc_nn[-1].append(acc)
        print('NN:', acc, 'n:', n)

acc_nn_avg = np.mean(np.array(acc_nn), axis=1)
acc_nn_std = np.log(np.std(np.array(acc_nn), axis=1))

df = pd.DataFrame({ 'acc_nn_avg':acc_nn_avg,
                   'acc_nn_std':acc_nn_std})

df.to_csv(str(launchtime)+'nn.csv')

for i in iters:
    acc_knn.append([])
    n = 50000 * i
    for j in range(3):
        X_train, y_train_sub, X_test, y_test = load_mnist(i)
        model = knn_subsetlabels(int(2 * np.sqrt(n)))
        model.fit(X_train, y_train_sub)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_pred, y_test)
        acc_knn[-1].append(acc)
        print('knn:', acc, 'n:', n)


acc_knn_avg = np.mean(np.array(acc_knn), axis=1)
acc_knn_std = np.log(np.std(np.array(acc_knn), axis=1))

df = pd.DataFrame({'acc_knn_avg':acc_knn_avg, 'acc_knn_std':acc_knn_std})
df.to_csv(str(launchtime)+'knn.csv')


for i in iters:
    acc_lda_gmm.append([])
    acc_lda_mom.append([])
    n = 50000 * i
    for j in range(3):
        X_train, y_train_sub, X_test, y_test = load_mnist(i)
        gmm_model = MyGMM(k_cluster)
        gmm_model.fit(X_train, y_train_sub)
        y_pred = gmm_model.predict(X_test)
        acc = accuracy_score(y_pred, y_test)
        acc_lda_gmm[-1].append(acc)
        print('LDA with GMM:', acc, 'n:', n)
        LDA_sub = MyLDA(k_cluster)
        LDA_sub.fit(X_train, y_train_sub)
        y_pred = LDA_sub.predict(X_test)
        acc = accuracy_score(y_pred, y_test)
        acc_lda_mom[-1].append(acc)
        print('LDA with MoM:', acc, 'n:', n)
        
        


acc_lda_gmm_avg = np.mean(np.array(acc_lda_gmm), axis=1)
acc_lda_gmm_std = np.log(np.std(np.array(acc_lda_gmm), axis=1))
acc_lda_mom_avg = np.mean(np.array(acc_lda_mom), axis=1)
acc_lda_mom_std = np.log(np.std(np.array(acc_lda_mom), axis=1))

df = pd.DataFrame({'acc_lda_gmm_avg':acc_lda_gmm_avg, 'acc_lda_mom_avg':acc_lda_mom_avg, 
'acc_lda_gmm_std':acc_lda_gmm_std, 'acc_lda_mom_std':acc_lda_mom_std})
df.to_csv(str(launchtime)+'MNIST.csv')
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
