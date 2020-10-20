import math
import numpy as np
import scipy
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

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test, target):
        distances = self.distance(x_test)
        return self.labels(distances)

    def euclidean_distance(self, vectorA, vectorB, length):

        distance = 0
        for x in range(length):
            distance += (point1[x] - point2[x])**2
        return np.sqrt(distance)

    def distance(self, x_test):

        length = x_test.shape[1]
        distance = []
        for i in range(len(x_test)):
            for row in range(len(self.x_train)):
                dist = self.euclidean_distance(self.x_train.iloc[row], x_test.iloc[i], length)
                distances.append(dist)

        return distances

    def labels(self, distances):

        y_indices = np.argsort(distances)[:self.k]
        k_nearest_class = [self.y_train[i] for i in y_indices]
        y_pred = [scipy.stats.mode(k_nearest_class)][0][0][0]
        return y_pred

