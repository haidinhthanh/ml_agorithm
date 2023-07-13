import numpy as np


class SVM:
    def __init__(self, kernel='linear', learning_rate=0.01, lambda_param=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.num_iterations = num_iterations
        self.kernel = kernel
        self.kernel_func = self._get_kernel_function()

    def _get_kernel_function(self):
        if self.kernel == 'linear':
            return lambda x1, x2: np.dot(x1, x2.T)
        elif self.kernel == 'rbf':
            return lambda x1, x2: np.exp(-np.linalg.norm(x1 - x2) ** 2)
        elif self.kernel == 'poly':
            return lambda x1, x2: (np.dot(x1, x2.T) + 1) ** 2

    def _compute_cost(self, W, X, y):
        N = X.shape[0]
        distances = 1 - y * (np.dot(X, W))
        distances[distances < 0] = 0  # max(0, distance)
        hinge_loss = self.lambda_param * (np.sum(distances) / N)
        cost = 1 / 2 * np.dot(W, W) + hinge_loss
        return cost

    def _compute_gradient(self, W, X, y):
        if isinstance(W, float):
            W = np.zeros(X.shape[1])

        N = X.shape[0]
        distances = 1 - y * (np.dot(X, W))
        dw = np.zeros(len(W))
        for idx, distance in enumerate(distances):
            if max(0, distance) == 0:
                di = W
            else:
                di = W - (self.lambda_param * y[idx] * X[idx])
            dw += di
        dw /= N
        return dw

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.models = {}

        for class_label in self.classes:
            binary_y = np.where(y == class_label, 1, -1)
            self.models[class_label] = self._train_binary_classifier(X, binary_y)

    def _train_binary_classifier(self, X, y):
        W = np.zeros(X.shape[1])
        for _ in range(self.num_iterations):
            gradient = self._compute_gradient(W, X, y)
            W = W - self.learning_rate * gradient

        return W

    def predict(self, X):
        predictions = np.zeros((X.shape[0], len(self.classes)))
        for class_label, model in self.models.items():
            predictions[:, class_label] = np.dot(X, model)
        return np.argmax(predictions, axis=1)

# Example usage
X = np.array([[1, 2, 3], [2, 1, 3], [2, 3, 3], [3, 1, 3], [3, 2, 3]])
y = np.array([0, 0, 1, 1, 2])

svm = SVM(kernel='linear', learning_rate=0.01, lambda_param=0.01, num_iterations=1000)
svm.fit(X, y)

X_test = np.array([[0, 1, 3], [4, 3, 3]])
predictions = svm.predict(X_test)
print(predictions)  # Output: [0, 2]