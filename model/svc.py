import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.datasets import make_circles
from sklearn import preprocessing

x, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=1,
                           n_clusters_per_class=1, random_state=14)
x = preprocessing.scale(x)

x_test = x[:500]
y_test = y[:500]
x = x[500:]
y = y[500:]

y = np.where(y == 0, -1, 1)
y_test = np.where(y_test == 0, -1, 1)

sns.scatterplot(x=x[:, 0], y=x[:, 1], hue=y.reshape(-1))


class support_vector_machine:
    def __init__(self, C=10, features=2, sigma_sq=0.1, kernel="None"):
        self.C = C
        self.features = features
        self.sigma_sq = sigma_sq
        self.kernel = kernel
        self.weights = np.zeros(features)
        self.bias = 0.

    def __similarity(self, x, l):
        return np.exp(-sum((x - l) ** 2) / (2 * self.sigma_sq))

    def gaussian_kernel(self, x1, x):
        m = x.shape[0]
        n = x1.shape[0]
        op = [[self.__similarity(x1[x_index], x[l_index]) for l_index in range(m)] for x_index in range(n)]
        return np.array(op)

    def loss_function(self, y, y_hat):
        sum_terms = 1 - y * y_hat
        sum_terms = np.where(sum_terms < 0, 0, sum_terms)
        return self.C * np.sum(sum_terms) / len(y) + sum(self.weights ** 2) / 2

    def fit(self, x_train, y_train, epochs=1000, print_every_nth_epoch=100, learning_rate=0.01):
        y = y_train.copy()
        x = x_train.copy()
        self.initial = x.copy()

        assert x.shape[0] == y.shape[0], "Samples of x and y don't match."
        assert x.shape[1] == self.features, "Number of Features don't match"

        if self.kernel == "gaussian":
            x = self.gaussian_kernel(x, x)
            m = x.shape[0]
            self.weights = np.zeros(m)

        n = x.shape[0]

        for epoch in range(epochs):
            y_hat = np.dot(x, self.weights) + self.bias
            grad_weights = (-self.C * np.multiply(y, x.T).T + self.weights).T

            for weight in range(self.weights.shape[0]):
                grad_weights[weight] = np.where(1 - y_hat <= 0, self.weights[weight], grad_weights[weight])

            grad_weights = np.sum(grad_weights, axis=1)
            self.weights -= learning_rate * grad_weights / n
            grad_bias = -y * self.bias
            grad_bias = np.where(1 - y_hat <= 0, 0, grad_bias)
            grad_bias = sum(grad_bias)
            self.bias -= grad_bias * learning_rate / n
            if (epoch + 1) % print_every_nth_epoch == 0:
                print("--------------- Epoch {} --> Loss = {} ---------------".format(epoch + 1,
                                                                                      self.loss_function(y, y_hat)))

    def evaluate(self, x, y):
        pred = self.predict(x)
        pred = np.where(pred == -1, 0, 1)
        diff = np.abs(np.where(y == -1, 0, 1) - pred)
        return (len(diff) - sum(diff)) / len(diff)

    def predict(self, x):
        if self.kernel == "gaussian":
            x = self.gaussian_kernel(x, self.initial)
        return np.where(np.dot(x, self.weights) + self.bias > 0, 1, -1)


model = support_vector_machine(C=20, sigma_sq=0.01)
model.fit(x, y, epochs=20, print_every_nth_epoch=2, learning_rate=0.01)
print("Training Accuracy = {}".format(model.evaluate(x, y)))
