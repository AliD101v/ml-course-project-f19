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
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn import svm
from sklearn.datasets import load_svmlight_file
import scipy
from scipy.sparse import csr_matrix
from IPython.core.display import display
import plotly
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import timeit
from datetime import datetime

# global configs and params
random_seed = 0
test_size = 0.2
fig_label_font = 'Libertinus Sans'
fig_legend_font = 'Libertinus Sans'
np.random.seed(random_seed)


# ────────────────────────────────────────────────────────────────────────────────
# # 1. Load the dataset(s)
# X, y = load_data()


# ────────────────────────────────────────────────────────────────────────────────
# # 2. Split the dataset(s) into training and test
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)


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