import numpy as np


class SoftMarginSVM:
    def __init__(self,
                 learning_rate=0.01,
                 lambda_param=0.01,
                 num_iterations=1000,
                 kernel='linear',
                 degree=3,
                 gamma=0.1,
                 ):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.num_iterations = num_iterations
        self.kernel = kernel
        self.weights = None
        self.bias = None
        self.degree = degree
        self.gamma = gamma

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize the weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Apply gradient descent
        for _ in range(self.num_iterations):
            if self.kernel == 'linear':
                scores = self.linear_function(X)
            else:
                scores = self.nonlinear_function(X)

            # Calculate hinge loss
            loss = self.hinge_loss(scores, y)

            # Calculate gradients
            dw, db = self.calculate_gradients(X, y, loss)

            # Update weights and bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def linear_function(self, X):
        return np.dot(X, self.weights) + self.bias

    def nonlinear_function(self, X):
        if self.kernel == 'poly':
            return (np.dot(X, self.weights) + self.bias) ** self.degree  # Polynomial kernel of degree 3
        elif self.kernel == 'rbf':
            gamma = 0.1  # Radial basis function (RBF) kernel parameter
            return np.exp(-gamma * np.linalg.norm(X - self.weights, axis=1) ** 2)  # RBF kernel
        elif self.kernel == 'sigmoid':
            gamma = 0.1
            return np.tanh(gamma * np.dot(X, self.weights) + self.bias)
        else:
            raise Exception(f"Not support kernel {self.kernel}")

    def hinge_loss(self, scores, y):
        return np.maximum(0, 1 - y * scores)

    def calculate_gradients(self, X, y, loss):
        mask = loss > 0
        dw = self.lambda_param * self.weights - np.dot(X.T, y * mask)  # Gradient for weights
        db = -np.sum(y * mask)  # Gradient for bias
        return dw, db

    def predict(self, X):
        if self.kernel == 'linear':
            scores = self.linear_function(X)
        else:
            scores = self.nonlinear_function(X)
        return np.sign(scores)


if __name__ == "__main__":
    X_train = np.array([[1, 2, 3], [2, 1, 4], [2, 3, 5], [3, 2, 6]])
    y_train = np.array([-1, -1, 1, 1])

    svm = SoftMarginSVM(learning_rate=0.001, lambda_param=0.001, num_iterations=5000, kernel='poly')
    svm.fit(X_train, y_train)

    X_test = np.array([[1, 2, 3], [4, 0, 3], [3, 3, 5]])
    y_test = svm.predict(X_train)
    print(y_test)
