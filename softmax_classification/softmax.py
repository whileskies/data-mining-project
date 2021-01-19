import numpy as np


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


class Softmax:
    # features = [[X1], [X2], ..., [Xn]], n is num of samples
    # x1([x1, x2, ..., xm]) is bow vec of sample #1, m is num of features
    # labels = [0, 0, ..., 1, 1, ..., 5, 5, 5], size(labels) = n, c is num of classes
    # weights = [[w1], [w2], ..., [wm]]
    def __init__(self, features, labels, num_of_classes, alpha=0.01, iterations=50):
        self.alpha = alpha
        self.iterations = iterations
        self.features = np.array(features)
        self.labels = np.array(labels)

        self.n, self.m = np.shape(self.features)
        self.c = num_of_classes

        self.weights = np.zeros((self.c, self.m), dtype=np.float)

    @staticmethod
    def soft(X):
        x_exp = np.exp(X)
        x_sum = np.sum(x_exp, axis=1, keepdims=True)  # 保持二维特性
        return x_exp / x_sum

    def gradient(self, k, s):
        sum_grad = np.zeros(self.m)
        for i in range(self.n):
            if self.labels[i] == k:
                sum_grad += ((1 - s[i][k]) * self.features[i])
            else:
                sum_grad += ((0 - s[i][k]) * self.features[i])
        return sum_grad

    def train(self):
        for i in range(self.iterations):
            print('training, ' + str(i) + 'th')
            tmp_weights = np.zeros((self.c, self.m), dtype=np.float)
            for k in range(self.c):
                X = np.dot(self.features, self.weights.T)
                tmp_weights[k] = self.alpha * self.gradient(k, self.soft(X))
            self.weights += tmp_weights

    def predict(self, test_features):
        predict_soft = test_features @ self.weights.T
        return predict_soft.argmax(axis=1)
