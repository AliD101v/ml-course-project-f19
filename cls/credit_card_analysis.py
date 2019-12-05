import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn import metrics
from sklearn import svm
from sklearn import tree
from sklearn import ensemble
from sklearn import datasets, linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, r2_score

# data reading
filePath = "C:/Users/sidha/OneDrive/Documents/ml-course-project-f19/ml-course-project-f19/data/Default of credit card clients/default_of_credit_card_clients.xls"
data_excel = pd.read_excel(filePath, sheet_name='Data')

X = data_excel.iloc[:,:23]
y = data_excel.iloc[:,23]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =0.2)


# K-neighbors classifier
weight_options = ['distance','uniform']
for option in weight_options :
    neighbors_classifier = neighbors.KNeighborsClassifier(15, weights=option)
    neighbors_classifier.fit(X_train, y_train)
    knn_predict = neighbors_classifier.predict(X_test)
    knn_accuracy = metrics.accuracy_score(y_test,knn_predict)
    print("Option selected : ", option)
    print("Accuracy score : ", knn_accuracy)

# SVM - support vector classifier

svm_classifier = svm.SVC(kernel='poly',degree=3,coef0=0.1, gamma='scale')
svm_classifier.fit(X_train,y_train)

svm_results = svm_classifier.predict(X_test)

svm_score = svm_classifier.score(X_test, y_test)
print(svm_score)

svm_classifier_rbf = svm.SVC(kernel='rbf', gamma='scale')
svm_classifier_rbf.fit(X_train,y_train)
svm_rbf_results = svm_classifier_rbf.predict(X_test)

print("svm rbf results")
print(svm_classifier_rbf.score(X_test,y_test))

#decision tree

# purpose of random state ??
# try different values for random state
# try different values for max_depth
descision_tree_classifier = tree.DecisionTreeClassifier(random_state=0)
descision_tree_classifier.fit(X_train,y_train)

descision_tree_classifier_results = descision_tree_classifier.predict(X_test)
print("Decision tree results : ")
print(metrics.accuracy_score(y_test, descision_tree_classifier_results))
print(descision_tree_classifier.feature_importances_)

random_forest = ensemble.RandomForestClassifier(n_estimators=100, random_state= 0)
random_forest.fit(X_train, y_train)

random_forest_result = random_forest.predict(X_test)
print("Random forest result ")
print(metrics.accuracy_score(y_test,random_forest_result))


#Adaboost classification

ada_boost_classifier = ensemble.AdaBoostClassifier(n_estimators=50, algorithm='SAMME.R', random_state=0,learning_rate=1.0)
ada_boost_classifier.fit(X_train,y_train)

ada_boost_classifier_results = ada_boost_classifier.predict(X_test)
print("Adaboost accuracy score")
print(metrics.accuracy_score(y_test,ada_boost_classifier_results))

print("AdaBoost test score")
print(ada_boost_classifier.score(X_test,y_test))

print("AdaBoost train score")
print(ada_boost_classifier.score(X_train,y_train))


#Logistic regression

regression = linear_model.LogisticRegression(random_state=0,solver='lbfgs',max_iter=1000)
regression.fit(X_train, y_train)

regression_predict = regression.predict(X_test)

print("Coefficients :\n ", regression.coef_)

print("Mean squared error: %.2f" % mean_squared_error(y_test, regression_predict))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, regression_predict))
print('Accuracy score : %.2f' % metrics.accuracy_score(y_test, regression_predict))

# Gaussian naive bayes classification

gnb_classifier = GaussianNB()
gnb_classifier.fit(X_train, y_train)
gnb_classifier_results = gnb_classifier.predict(X_test)

print("Gaussian Naive Bayes result ")
print(metrics.accuracy_score(y_test,gnb_classifier_results))

print("Gaussian Naive Bayes test score")
print(gnb_classifier.score(X_test,y_test))

print("Gaussian Naive Bayes train score")
print(gnb_classifier.score(X_train,y_train))

#  neural nets