#!/usr/bin/env python
# coding: utf-8

# In[7]:


# suppress sklearn warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# imports
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import sklearn
from sklearn.pipeline import Pipeline

# preprocessing
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import normalize
from sklearn.feature_selection import SelectFromModel
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# model selection
from sklearn.model_selection import train_test_split, GridSearchCV

# metrics
from sklearn import metrics
from sklearn.metrics import accuracy_score,                            precision_recall_fscore_support
# estimators
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# scipy
import scipy
from scipy.sparse import csr_matrix
from IPython.core.display import display

# plotting
import plotly
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt

# misc
import timeit
from datetime import datetime


from data.austrailian_data import *

# global configs and params
random_seed = 0
test_size = 0.2
fig_label_font = 'Libertinus Sans'
fig_legend_font = 'Libertinus Sans'
np.random.seed(random_seed)



# In[11]:



X,y=load_austrailian()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train)


classifiers = [
    LogisticRegression(random_state=random_seed),
    KNeighborsClassifier(),
    GaussianNB(),
#     SVC(probability=True, random_state=random_seed),
    DecisionTreeClassifier(random_state=random_seed),
    RandomForestClassifier(random_state=random_seed),
    AdaBoostClassifier(random_state=random_seed),
#     MLPClassifier(random_state=random_seed)
    ]


# In[ ]:


grid_params = {
        'LogisticRegression':{ 
        'LogisticRegression__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        },
        'KNeighborsClassifier':
        {
            'KNeighborsClassifier__n_neighbors': list(np.linspace(3,22,20))
        },
        'GaussianNB':{
            'GaussianNB__var_smoothing': list(np.logspace(-10, 0, num=11, base=10)),
        },
#         'SVC':
#         {
#             'SVC__kernel': ['linear', 'poly', 'rbf'],
#             'SVC__C': list(np.logspace(-5, 15, num=11, base=2)),
#             'SVC__gamma': list(np.logspace(-15, 3, num=10, base=2)),
#         },
        'DecisionTreeClassifier':
        {
            'DecisionTreeClassifier__criterion': ['gini', 'entropy'],
            'DecisionTreeClassifier__max_depth': list(np.linspace(1, 32, 32, endpoint=True)),
            # 'DecisionTreeClassifier__splitter': ['best', 'random'],
            # 'DecisionTreeClassifier__min_samples_split': list(np.linspace(0.1, 1.0, 10, endpoint=True)),
            # 'DecisionTreeClassifier__min_samples_leaf': list(np.linspace(0.1, 0.5, 5, endpoint=True)),
            # 'DecisionTreeClassifier__max_features': list(np.linspace(0.1, 0.5, 5, endpoint=True)),
        },
        'RandomForestClassifier':
        {
            'RandomForestClassifier__n_estimators': list(np.arange(10, 101)),
            'RandomForestClassifier__criterion': ['gini', 'entropy'],
            'RandomForestClassifier__max_depth': list(np.linspace(1, 32, 32, endpoint=True)),
            # 'RandomForestClassifier__splitter': ['best', 'random'],
            # 'RandomForestClassifier__min_samples_split': list(np.linspace(0.1, 1.0, 10, endpoint=True)),
            # 'RandomForestClassifier__min_samples_leaf': list(np.linspace(0.1, 0.5, 5, endpoint=True)),
            # 'RandomForestClassifier__max_features': list(np.linspace(0.1, 0.5, 5, endpoint=True)),
        },
        'AdaBoostClassifier':
        {
            'AdaBoostClassifier__n_estimators': list(np.arange(10, 51)),
            'AdaBoostClassifier__learning_rate': list(np.linspace(0.1, 1, 10, endpoint=True)),
        },
        'MLPClassifier':
        {
            'MLPClassifier__activation': ['logistic', 'tanh'],
            'MLPClassifier__solver': ['lbfgs', 'sgd', 'adam'],
            'MLPClassifier__hidden_layer_sizes': [(1,)] + [(i,) for i in np.arange(10, 101, 10)],
            'MLPClassifier__learning_rate': ['invscaling', 'adaptive'],
            'MLPClassifier__max_iter': list(np.arange(100, 501, 50)),
            
        }
    }

results = []

for classifier in classifiers:
    pipe = Pipeline(steps=[(classifier.__class__.__name__, classifier)])

    # Perform a grid search on the entire pipeline of the current classifier
    # Note: to disable the grid search, comment the following three lines,
    # and call fit() and predict() directly on the pipe object
    grid_clf = GridSearchCV(pipe, grid_params[classifier.__class__.__name__], n_jobs=8)
    grid_clf.fit(X_train, y_train)

    # best params are stored in the grid_clf.best_params_ object:
    ## print(grid_clf.best_params_)
    
    # store the best classifier for each classifier
    best_pipe = grid_clf.best_estimator_

    # just a piece of code in case we need access to the classifier in the pipe
    ## print(best_pipe[classifier.__class__.__name__])

    y_pred = best_pipe.predict(X_test)
    precision, recall, f1, _ =         precision_recall_fscore_support(y_test, y_pred, average='micro')

    result = {
                'Classifier': classifier.__class__.__name__,
                'Score': best_pipe.score(X_test, y_test),
                'Accuracy': accuracy_score(y_test, y_pred),
                'f1 score': f1,
                'Precision': precision,
                'Recall': recall
            }
    results.append(result)

results_df = pd.DataFrame(data=results, index=None,
                        columns=['Classifier', 'Score', 'Accuracy',
                        'f1 score', 'Precision', 'Recall'])
results_df.index = [''] * len(results_df)

display(results_df.sort_values(by=['Score'], ascending=False))

