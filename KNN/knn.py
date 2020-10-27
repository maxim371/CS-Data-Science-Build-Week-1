import math
import numpy as np
from scipy import stats
from sklearn import datasets
from pandas import DataFrame


class Euclidean:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def euclidean_distance(self, x, y):
        return np.sqrt(np.sum(self.x - self.y)**2)

class K_Nearest_Neighbor:

    def __init__(self, k):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        distances = self.get_distance(X_test)
        return self.labels(distances)

    def euclidean_distance(self, point1, point2, length):

        distance = 0
        for x in range(length):
            distance += (point1[x] - point2[x])**2
        return np.sqrt(distance)

    def get_distance(self, X_test):

        length = X_test.shape[1]                          
        distances = []
        for i in range(len(X_test)):
            distances.append([X_test[i], [] ])
            for a in self.X_train:
                dist = self.euclidean_distance(a, X_test[i], length)
                distances[i][1].append(dist)

        return distances

    def labels(self, distances):

        y_pred = []
        for i in range(len(distances)):
            distant = distances[i]
            y_indices = np.argsort(distant[1])[:self.k]
            k_nearest_class = [self.y_train[i%len(self.y_train)] for i in y_indices]
            label = [stats.mode(k_nearest_class)][0][0][0]
            y_pred.append(label)
        return y_pred

