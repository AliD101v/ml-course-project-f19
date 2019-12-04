import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier
from data.CIFAR10 import *



# 5 batches 
# training_data = np.empty((1,3072))
# training_data_classes = np.empty((0))
# testing_data = np.empty((1,3072))
# dictionary = load_CIFAR10()

train, test = load_CIFAR10()
n = 5

print(f'First {n} records in...')
X = train[0][b'data']
y = train[0][b'labels']
print(type (y))
y = np.asarray(y)
print(type (y))

X_test =  test[b'data']
y_test = test [b'labels']
print(type (y_test))
y_test = np.asarray(y_test)
print(type (y_test))
# print()

for i in range (1,n):
    X = np.vstack((X, train[i][b'data']))
    y = np.hstack((y, train[i][b'labels']))


cifar_decision_tree = DecisionTreeClassifier(max_depth=10, random_state=0)
print(cifar_decision_tree)
cifar_decision_tree.fit(X, y)
cifar_results = cifar_decision_tree.predict(X_test)
print(cifar_decision_tree.score(cifar_results, y_test))

