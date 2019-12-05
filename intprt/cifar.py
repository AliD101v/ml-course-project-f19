import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier
from data.CIFAR10 import *
from sklearn import metrics
import matplotlib
import matplotlib.pyplot as plt
from sklearn import multiclass
train, test = load_CIFAR10()
n = 5

print(f'First {n} records in...')
X = train[0][b'data']
y = train[0][b'labels']

X_test =  test[b'data']
y_test = test [b'labels']

y_test = np.asarray(y_test).reshape((-1,1))


for i in range (1,n):
    X = np.vstack((X, train[i][b'data']))
    y = np.hstack((y, train[i][b'labels']))

y = np.asarray(y).reshape((-1,1))


print(" Check X shapes  : ")
print(X.shape)
print(X_test.shape)

cifar_decision_tree = DecisionTreeClassifier(max_depth=3, random_state=0)
cifar_decision_tree.fit(X, y)
cifar_results = cifar_decision_tree.predict(X_test)

print(cifar_decision_tree.score(X_test, y_test))
print(metrics.accuracy_score(y_test,cifar_results))


plt.figure()
sklearn.tree.plot_tree(cifar_decision_tree, filled= True, max_depth=2, label='root', fontsize=6)
plt.show()