import numpy as np
from collections import Counter


class knn_subsetlabels():
    def __init__(self, neighbors=3):
        self.neighbors = neighbors

    def fit(self, x, Y):
        self.X = np.array(x)
        self.Y = []
        for y in Y:
            self.Y.append(np.nonzero(y)[0])

    def dist(self, a, b):
        return np.linalg.norm(a - b)

    def k_nearest(self, x):
        distances = [self.dist(x, data) for data in self.X]
        top_k = np.array(self.Y)[np.argsort(distances)[:self.neighbors]]
        return top_k.tolist()

    def predict(self, X):
        y_pred = []
        for x in X:
            top_k_y = self.k_nearest(x)
            top_k_flatten = [i for item in top_k_y for i in item]
            votes = Counter(top_k_flatten).most_common(1)[0][0]
            y_pred.append(votes)
        return y_pred