import numpy as np


class SVM:
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


x = np.array([[1, 2, 3], [2, 3, 3], [3, 1, 4], [6, 7, 4], [7, 5, 4], [8, 6, 4]])
y = np.array([-1, -1, -1, 1, 1, 1])

model = SVM(C=10, kernel="gaussian", sigma_sq=0.01, features=3)
model.fit(x, y, epochs=20, print_every_nth_epoch=2, learning_rate=0.01)

x_test = np.array([[4, 3, 2], [5, 5, 4], [1, 2, 3]])

p = model.predict(x)
print(p)
p = model.predict(x_test)
print(p)