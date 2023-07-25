import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameter search space
param_space = {
    'C': np.linspace(0.1, 2.0, 20),  # Regularization parameter
    'penalty': ['l1', 'l2']  # Regularization type
}

# Number of random combinations to try
num_random_combinations = 20

best_accuracy = 0.0
best_params = None

for _ in range(num_random_combinations):
    # Randomly sample hyperparameters from the search space
    params = {
        'C': np.random.choice(param_space['C']),
        'penalty': np.random.choice(param_space['penalty'])
    }

    # Create and train the model
    model = LogisticRegression(solver='liblinear', **params)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Update best accuracy and parameters if necessary
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = params

# Train the final model using the best parameters
final_model = LogisticRegression(solver='liblinear', **best_params)
final_model.fit(X_train, y_train)

# Test the final model
y_pred_final = final_model.predict(X_test)
final_accuracy = accuracy_score(y_test, y_pred_final)

print("Best Hyperparameters:", best_params)
print("Best Accuracy:", best_accuracy)
print("Final Accuracy:", final_accuracy)
