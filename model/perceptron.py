import numpy as np


class Perceptron:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def train(self, X, y):
        num_samples, num_features = X.shape

        # Initialize weights and bias
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.num_iterations):
            for i in range(num_samples):
                # Compute activation
                activation = np.dot(self.weights, X[i]) + self.bias

                # Apply step function
                if activation >= 0:
                    y_hat = 1
                else:
                    y_hat = 0

                # Update weights and bias
                self.weights += self.learning_rate * (y[i] - y_hat) * X[i]
                self.bias += self.learning_rate * (y[i] - y_hat)

    def predict(self, X):
        num_samples = X.shape[0]
        predictions = np.zeros(num_samples)

        for i in range(num_samples):
            # Compute activation
            activation = np.dot(self.weights, X[i]) + self.bias

            # Apply step function
            if activation >= 0:
                predictions[i] = 1
            else:
                predictions[i] = 0

        return predictions


# Example usage
X = np.array([[2, 3], [1, 1], [5, 2], [3, 1]])
y = np.array([1, 0, 1, 0])

# Create a Perceptron instance
perceptron = Perceptron(learning_rate=0.1, num_iterations=100)

# Train the model
perceptron.train(X, y)

# Make predictions
test_samples = np.array([[4, 3], [1, 2]])
predictions = perceptron.predict(test_samples)

print(predictions)  # Output: [1, 0]
