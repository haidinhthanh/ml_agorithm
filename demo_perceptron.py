from data.dataset import Dataset
import numpy as np
from model.perceptron import Perceptron
from mertric.metrics import accuracy_score

ds = Dataset.load('iris')
# print(ds)

classes = ds['class'].unique()
# print(classes)

x_is = ds[ds['class'] == 'Iris-setosa'][["sepallength", "sepalwidth", "petallength", "petalwidth"]].to_numpy()
y_is = ds[ds['class'] == 'Iris-setosa'][['class']]

x_iv = ds[ds['class'] != 'Iris-setosa'][["sepallength", "sepalwidth", "petallength", "petalwidth"]].to_numpy()
y_iv = ds[ds['class'] != 'Iris-setosa'][['class']]


y_is.loc[y_is['class'] == 'Iris-setosa'] = 1
y_iv.loc[y_iv['class'] != 'Iris-setosa'] = 0

X = np.concatenate([x_is, x_iv], dtype='float32')
y = np.concatenate([y_is.to_numpy(), y_iv.to_numpy()]).squeeze()

idx = np.random.permutation(range(len(y)))

X, y = X[idx, :], y[idx]

split = int(0.6 * len(y))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

perceptron = Perceptron(input_size=4)

# Train the perceptron on the sample data
perceptron.fit(X_train, y_train, learning_rate=0.1, epochs=1000)

y_pred = perceptron.predict(X_test)
y_pred[y_pred >= 0.5] = 1
y_pred[y_pred < 0.5] = 0
print(accuracy_score(y_test, y_pred))