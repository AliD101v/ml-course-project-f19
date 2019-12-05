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
from sklearn import preprocessing

filePath_red = "C:/Users/sidha/OneDrive/Documents/ml-course-project-f19/ml-course-project-f19/data/Wine Quality/winequality-red.csv"
filePath_white = "C:/Users/sidha/OneDrive/Documents/ml-course-project-f19/ml-course-project-f19/data/Wine Quality/winequality-white.csv"

data_csv_red = pd.read_csv(filePath_red, delimiter=";")
data_csv_white = pd.read_csv(filePath_white, delimiter=";")

X_red = data_csv_red.iloc[:,:11]
y_red = data_csv_red.iloc[:,11]

X_white = data_csv_white.iloc[:,:11]
y_white = data_csv_white.iloc[:,11]

# X_red.append(X_white)
# y_red.append(y_white)

X_train, X_test, y_train, y_test = train_test_split(X_white,y_white, test_size= 0.2)

scaler = preprocessing.StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# Support vector regressor

support_vector_regressor = svm.SVR()
print(support_vector_regressor.fit(X_train,y_train))

support_vector_results = support_vector_regressor.predict(X_test)

print("SVR score")
print(support_vector_regressor.score(X_test,y_test))
print(metrics.max_error(y_test,support_vector_results))

# Decision tree regressor

tree_regressor = tree.DecisionTreeRegressor(random_state=0)
print(tree_regressor.fit(X_train,y_train))
tree_regressor_results = tree_regressor.predict(X_test)
print("Decision Tree results : " , tree_regressor.score(X_test,y_test))
print(metrics.max_error(y_test,tree_regressor_results))

# Random Forest Regression

random_forest_regressor = ensemble.RandomForestRegressor(n_estimators=100, random_state=0)
print(random_forest_regressor.fit(X_train,y_train))
random_forest_results = random_forest_regressor.predict(X_test)
print("Random forest regressor results : " ,random_forest_regressor.score(X_test,y_test))
print(metrics.max_error(y_test,random_forest_results))

# Adaboost regression

adaboost_regressor = ensemble.AdaBoostRegressor(n_estimators=100,random_state=0)
print(adaboost_regressor.fit(X_train, y_train))

adaboost_results = adaboost_regressor.predict(X_test)
print("Adaboost results : ", adaboost_regressor.score(X_test,y_test))
print(metrics.max_error(y_test,adaboost_results))

# Linear regression
linear_regressor = linear_model.LinearRegression()
print(linear_regressor.fit(X_train,y_train))
linear_regressor_results = linear_regressor.predict(X_test)
print("Linear Regression results : " , linear_regressor.score(X_test,y_test))
print(metrics.max_error(y_test,linear_regressor_results))
