from data.dataset import Dataset
import numpy as np
from model.svm import SoftMarginSVM
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
y_iv.loc[y_iv['class'] != 'Iris-setosa'] = -1

X = np.concatenate([x_is, x_iv], dtype='float32')
y = np.concatenate([y_is.to_numpy(), y_iv.to_numpy()]).squeeze()

idx = np.random.permutation(range(len(y)))

X, y = X[idx, :], y[idx]

split = int(0.6 * len(y))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

svm = SoftMarginSVM(C=1.0, kernel='rbf', gamma=0.1, max_iterations=500)
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)
print(accuracy_score(y_pred, y_test))