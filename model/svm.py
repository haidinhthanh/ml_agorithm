import numpy as np
from tqdm import tqdm

class SoftMarginSVM:
    def __init__(self,
                 C=1.0,
                 kernel='linear',
                 degree=3,
                 gamma=0.1,
                 learning_rate=0.01,
                 max_iterations=5000):
        self.C = C  # Regularization parameter
        self.kernel = kernel  # Kernel function
        self.degree = degree  # Degree parameter for polynomial kernel
        self.gamma = gamma  # Gamma parameter for polynomial/rbf kernels
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations

    def fit(self, X, y):
        self.X = X
        self.y = y

        # Convert class labels to -1 and 1
        self.y[self.y == 0] = -1

        # Initialize parameters
        self.alpha = np.zeros(len(X), dtype='float32')
        self.b = 0

        # Kernel matrix
        self.K = self.kernel_matrix(X)

        # Training using gradient descent
        for iteration in tqdm(range(self.max_iterations)):
            # Compute gradients
            gradients = np.ones(len(X)) - self.alpha * y * (self.K.dot(self.alpha * y))

            # Update parameters
            self.alpha = self.alpha + self.learning_rate * gradients

            # Project alpha back to feasible space
            self.alpha = np.clip(self.alpha, 0, self.C)

        # Find support vectors
        support_vector_indices = self.alpha > 0

        # Compute the intercept
        support_vector_labels = y[support_vector_indices]
        support_vectors = X[support_vector_indices]
        self.b = np.mean(support_vector_labels - self.decision_function(support_vectors))

    def predict(self, X):
        # Compute the decision function
        decision_values = self.decision_function(X)

        # Classify based on the sign of the decision values
        y_pred = np.sign(decision_values)

        return y_pred

    def decision_function(self, X):
        # Compute the decision function
        K = self.kernel_matrix(X, self.X)  # Compute kernel between new data and support vectors
        decision_values = (self.alpha * self.y).dot(K.T) + self.b  # Use K.T for dot product
        return decision_values

    def kernel_matrix(self, X1, X2=None):
        if self.kernel == 'linear':
            K = np.dot(X1, X2.T) if X2 is not None else np.dot(X1, X1.T)
        elif self.kernel == 'poly':
            K = (np.dot(X1, X2.T) if X2 is not None else np.dot(X1, X1.T) + 1) ** self.degree
        elif self.kernel == 'rbf':
            if X2 is not None:
                sq_norms_X1 = np.sum(X1 ** 2, axis=1, keepdims=True)
                sq_norms_X2 = np.sum(X2 ** 2, axis=1, keepdims=True)
                K = np.exp(-self.gamma * (sq_norms_X1 - 2 * np.dot(X1, X2.T) + sq_norms_X2.T))
            else:
                sq_norms = np.sum(X1 ** 2, axis=1, keepdims=True)
                K = np.exp(-self.gamma * (sq_norms - 2 * np.dot(X1, X1.T) + sq_norms.T))
        else:
            raise ValueError("Invalid kernel type. Available options are 'linear', 'poly', and 'rbf'.")

        return K


if __name__ == "__main__":
    X_train = np.array([[1, 2, 3], [2, 1, 4], [2, 3, 5], [3, 2, 6], [2, 5, 5], [3, 12, 6], [2, 13, 5], [3, 32, 6]])
    y_train = np.array([-1, -1, 1, 1, 1, -1, 1, -1])

    svm = SoftMarginSVM(C=1.0, kernel='rbf', gamma=0.1)
    svm.fit(X_train, y_train)

    y_test = svm.predict(X_train)
    print(y_test)

    X_test = np.array([[1, 2, 3], [4, 0, 3], [3, 3, 5]])
    y_test = svm.predict(X_test)
    print(y_test)
