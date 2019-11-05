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
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# model selection
from sklearn.model_selection import train_test_split, GridSearchCV
# metrics
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
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

from data.WDBC import *

# global configs and params
random_seed = 0
test_size = 0.2
fig_label_font = 'Libertinus Sans'
fig_legend_font = 'Libertinus Sans'
np.random.seed(random_seed)


# ────────────────────────────────────────────────────────────────────────────────
# # 1. Load the dataset(s)
# todo perform some exploratory data analysis
# todo check for missing/NA values
df = load_WDBC()
# print(df.describe())

# ────────────────────────────────────────────────────────────────────────────────
# # 2. Split the dataset(s) into training and test
X = df[df.columns[1:]]
y = df[df.columns[0]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                    random_state=random_seed)

# ────────────────────────────────────────────────────────────────────────────────
# # 3. Pipeline

# ## 3.1 Transformers
# ### 3.1.1 Continuous (quantitative or numeric) transformer
# Example:
# ```python
# numeric_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='median')),
#     ('scaler', StandardScaler())])
# ```
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])
# 
# ### 3.1.2 Categorical (qualitative) transformer
# Example:
# ```python
# categorical_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))])
# ```

# ### 3.1.3 Column transformer
# Example:
# ```python
# numeric_features = train.select_dtypes(include=['int64', 'float64']).columns
# categorical_features = train.select_dtypes(include=['object']).drop(['Loan_Status'], axis=1).columns
# 
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numeric_transformer, numeric_features),
#         ('cat', categorical_transformer, categorical_features)])
# ```

numeric_features = df.select_dtypes(include=['int64', 'float64']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)])

# ## 3.2 Classifier
# Example:
# ```python
# rf = Pipeline(steps=[('preprocessor', preprocessor),
#                       ('classifier', RandomForestClassifier())])
# rf.fit(X_train, y_train)
# y_pred = rf.predict(X_test)
# ```


# Example for model selection:
# ```python
# classifiers = [
#     KNeighborsClassifier(3),
#     SVC(kernel="rbf", C=0.025, probability=True),
#     NuSVC(probability=True),
#     DecisionTreeClassifier(),
#     RandomForestClassifier(),
#     AdaBoostClassifier(),
#     GradientBoostingClassifier()
#     ]for classifier in classifiers:
#     pipe = Pipeline(steps=[('preprocessor', preprocessor),
#                       ('classifier', classifier)])
#     pipe.fit(X_train, y_train)   
#     print(classifier)
#     print("model score: %.3f" % pipe.score(X_test, y_test))
# ```

classifiers = [
    LogisticRegression(),
    KNeighborsClassifier(3),
    GaussianNB(),
    SVC(kernel="rbf", C=0.025, probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    MLPClassifier()]

for classifier in classifiers:
    pipe = Pipeline(steps=[('preprocessor', preprocessor),
                    ('classifier', classifier)])
    pipe.fit(X_train, y_train)   
    print(classifier)
    print("model score: %.3f" % pipe.score(X_test, y_test))
    print('==================================================')

# ## 3.3 Hyperparameter tuning


# Exampel:
# ```python
# param_grid = { 
#     'classifier__n_estimators': [200, 500],
#     'classifier__max_features': ['auto', 'sqrt', 'log2'],
#     'classifier__max_depth' : [4,5,6,7,8],
#     'classifier__criterion' :['gini', 'entropy']}from sklearn.model_selection import GridSearchCVCV = GridSearchCV(rf, param_grid, n_jobs= 1)
                  
# CV.fit(X_train, y_train)  
# print(CV.best_params_)    
# print(CV.best_score_)
# ```

# # 4. Output
# ## 4.1 Results
# ## 4.1 Figures