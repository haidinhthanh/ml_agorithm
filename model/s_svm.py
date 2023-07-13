import numpy as np


class SoftMarginSVM:
    def __init__(self, C=1.0, kernel='linear', degree=3, gamma='scale'):
        self.C = C  # Regularization parameter
        self.kernel = kernel  # Kernel function
        self.degree = degree  # Degree for polynomial kernel
        self.gamma = gamma  # Gamma parameter for polynomial and Gaussian kernels
        self.weights = None
        self.bias = None

    def _linear_kernel(self, X1, X2):
        return np.dot(X1, X2.T)

    def _polynomial_kernel(self, X1, X2):
        return (np.dot(X1, X2.T) + 1) ** self.degree

    def _gaussian_kernel(self, X1, X2):
        return np.exp(-self.gamma * np.linalg.norm(X1 - X2) ** 2)

    def _kernel_function(self, X1, X2):
        if self.kernel == 'linear':
            return self._linear_kernel(X1, X2)
        elif self.kernel == 'poly':
            return self._polynomial_kernel(X1, X2)
        elif self.kernel == 'gaussian':
            return self._gaussian_kernel(X1, X2)
        else:
            raise ValueError('Invalid kernel function')

    def _hinge_loss(self, X, y):
        scores = np.dot(X, self.weights) + self.bias
        margins = 1 - y * scores
        loss = np.maximum(0, margins)
        return np.mean(loss) + 0.5 * self.C * np.dot(self.weights, self.weights.T)

    def _gradient(self, X, y):
        scores = np.dot(X, self.weights) + self.bias
        margins = 1 - y * scores
        margins[margins <= 0] = 0
        dW = np.dot(X.T, -y * margins) / X.shape[0] + self.C * self.weights
        dB = np.mean(-y * margins)
        return dW, dB

    def train(self, X, y, learning_rate=0.001, num_iterations=8000):
        X = np.array(X)
        y = np.array(y)
        num_samples, num_features = X.shape

        self.weights = np.random.randn(num_features)
        self.bias = 0

        for i in range(num_iterations):
            dW, dB = self._gradient(X, y)
            self.weights -= learning_rate * dW
            self.bias -= learning_rate * dB

            if (i + 1) % 100 == 0:
                loss = self._hinge_loss(X, y)
                print(f'Iteration {i+1}/{num_iterations}, Loss: {loss:.4f}')

    def predict(self, X):
        X = np.array(X)
        scores = np.dot(X, self.weights) + self.bias
        return np.sign(scores)


svm = SoftMarginSVM(kernel='gaussian')

# Generate some synthetic data
X = np.array([[1, 2], [2, 1], [2, 3], [3, 2]])
y = np.array([-1, -1, 1, 1])

# Train the SVM
svm.train(X, y)

# Predict on new data
X_test = np.array([[0, 0], [4, 4]])
y_pred = svm.predict(X)

print("Predictions:", y_pred)
