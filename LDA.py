import numpy as np
from numpy import *
from scipy.special import logsumexp

class MyLDA(object):
    def __init__(self, k_cluster=4):
        self.K = k_cluster
        self.w = np.ones(self.K).astype(np.float16)
        self.w = self.w / np.sum(self.w)

    def fit(self, x, label):
        A = []
        B = []
        subset_dict = {}
        for i in range(len(x)):
            if tuple(label[i]) in subset_dict.keys():
                sub_i = subset_dict[tuple(label[i])]
                B[sub_i].append(x[i])
                #C[sub_i] = np.mean(B[sub_i])
            else:
                subset_dict[tuple(label[i])] = len(A)
                A.append(np.multiply(label[i], self.w)/np.dot(label[i], self.w))
                B.append([x[i]])
                #C.append(x[i])` `
        C = [[]] * len(B)
        for i in range(len(B)):
            C[i] = np.mean(B[i], axis=0, keepdims=True)[0]
        A = np.array(A)
        C = np.array(C)
        self.mu = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A).astype(float64)), A.T), C)
        C = np.dot(A, self.mu)
        A = [[] for a in A]
        #B = [[]]*len(A)
        # for sub in subset_dict.keys():
        #     i = subset_dict[sub]
        #     e_i = C[i]
        #     sigma_i = self.mu-e_i
        #     cov_list = [sub[j]*self.w[j]*(np.array([sigma_i[j]]).T.dot(np.array([sigma_i[j]]))) for j in range(self.K)]
        #     a_i = np.sum(cov_list, axis=0, keepdims=True)/np.dot(np.asarray(sub), self.w)
        #     A[i] = a_i[0]
        for i in range(len(x)):
            sub_i = subset_dict[tuple(label[i])]
            a = np.dot(np.array([x[i]]).T, np.array([x[i]]))
            A[sub_i].append(a)
            #B[sub_i].append(x[i])
        A = [np.mean(a, axis=-0, keepdims=True)[0] for a in A]
        D = [0]*len(B) # means for each subset
        for key in subset_dict.keys():
            sub = np.asarray(key)
            sub_i = subset_dict[key]
            multi_list = []
            # multi_list = np.sum([self.w[j]*sub[j]/np.dot(sub,self.w)*self.mu[j] for j in np.nonzero(sub)[0]],
            #                     axis=0, keepdims=True)[0]
            for j in np.nonzero(sub)[0]:
                mu = np.dot(np.array([self.mu[j]]).T, np.array([self.mu[j]]))
                mu = self.w[j]*sub[j]/np.dot(sub,self.w)*mu
                multi_list.append(mu)
            multi_list = [self.w[j]*sub[j]/np.dot(sub,self.w)*(np.dot(np.array([self.mu[j]]).T,np.array([self.mu[j]]))) for j in np.nonzero(sub)[0]]
            D[sub_i] = np.sum(multi_list, axis=0, keepdims=True)[0]
        A = np.array(A)
        D = np.array(D)
        self.var = np.mean(A-D, axis=0, keepdims=True)[0]
        a, _ = np.linalg.eig(self.var)
        min_eig = np.min(np.real(a))
        if min_eig < 0:
            self.var -= 2* min_eig * np.eye(*self.var.shape)
        pass
        # print("mean:", self.mu)
        # print("var:", self.var)


    def predict(self, X):
        preds = []
        for x in X:
            x = np.array(x)
            probs = []
            for i in range(self.K):
                prob = x.dot(np.linalg.inv(self.var)).dot(self.mu[i].T)-0.5*self.mu[i].dot(np.linalg.inv(self.var)).dot(self.mu[i].T)+np.log(self.w[i])
                probs.append(prob)
            preds.append(np.array(probs).argmax())
        return preds




