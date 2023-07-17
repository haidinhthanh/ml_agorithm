import numpy as np


class Perceptron:
    def __init__(self, input_size):
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, inputs):
        z = np.dot(inputs, self.weights) + self.bias
        return self.sigmoid(z)

    def train(self, inputs, targets, learning_rate=0.01, epochs=1000):
        for epoch in range(epochs):
            for x, y in zip(inputs, targets):
                predicted = self.predict(x)
                error = y - predicted
                self.weights += learning_rate * error * predicted * (1 - predicted) * x
                self.bias += learning_rate * error * predicted * (1 - predicted)

    def evaluate(self, inputs, targets):
        correct_predictions = 0
        for x, y in zip(inputs, targets):
            predicted = 1 if self.predict(x) >= 0.5 else 0
            if predicted == y:
                correct_predictions += 1
        accuracy = correct_predictions / len(targets)
        return accuracy

# Sample training data and targets
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([0, 1, 1, 1])

# Create a Perceptron instance with input_size = 2 (since we have two features)
perceptron = Perceptron(input_size=2)

# Train the perceptron on the sample data
perceptron.train(inputs, targets, learning_rate=0.1, epochs=10000)

# Evaluate the perceptron on the same data
accuracy = perceptron.evaluate(inputs, targets)
print("Accuracy:", accuracy)

# Make predictions
new_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
predictions = [perceptron.predict(x) for x in new_data]
print("Predictions:", predictions)
