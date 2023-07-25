from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from model.svm import SoftMarginSVM
from mertric.metrics import accuracy_score

data = load_diabetes()
X, y = data.data, data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

y_train[y_train == 0] = 1
y_train[y_train != 0] = -1
y_test[y_test == 0] = 1
y_test[y_test != 0] = -1

svm = SoftMarginSVM(C=1.0, kernel='linear', gamma=0.1, max_iterations=1000)
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)
print(accuracy_score(y_pred, y_test))