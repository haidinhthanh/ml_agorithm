import numpy as np
import matplotlib.pyplot as plt


class SVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_samples = X.shape[0]

        # Initialize weights and bias
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        # Gradient descent
        for _ in range(self.num_iterations):
            # SVM hinge loss function
            scores = np.dot(X, self.weights) + self.bias
            margins = y * scores
            svm_loss = np.maximum(0, 1 - margins)

            # Calculate gradients
            dW = self.lambda_param * self.weights - np.dot(X.T, (svm_loss > 0) * y)
            dB = -np.sum((svm_loss > 0) * y)

            # Update weights and bias
            self.weights -= self.learning_rate * dW
            self.bias -= self.learning_rate * dB

    def predict(self, X):
        scores = np.dot(X, self.weights) + self.bias
        return np.sign(scores)


# Example usage
X = np.array([[1, 2, ], [2, 3], [3, 1], [6, 7], [7, 5], [8, 6]])
y = np.array([-1, -1, -1, 1, 1, 1])

svm = SVM()
svm.fit(X, y)

# Plotting the decision boundary
plt.scatter(X[:, 0], X[:, 1], c=y)
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx = np.linspace(xlim[0], xlim[1], 100)
yy = np.linspace(ylim[0], ylim[1], 100)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = svm.predict(xy).reshape(XX.shape)
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
plt.show()
# X_test = np.array([[4, 3, 2], [5, 5, 4]])
#
# predictions = svm.predict(X_test)
# print(predictions)