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
import pickle
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
from sklearn.metrics import explained_variance_score,\
                            max_error,\
                            mean_absolute_error,\
                            mean_squared_error,\
                            mean_squared_log_error,\
                            median_absolute_error,\
                            r2_score
# estimators
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
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

from data.wine_quality import *

# global configs and params
random_seed = 0
test_size = 0.2
fig_label_font = 'Libertinus Sans'
fig_legend_font = 'Libertinus Sans'
np.random.seed(random_seed)
grid_search = False

dataset_name = 'Wine_quality'
results_path = 'rgs/results/'
# results_name = f'cnn_{time.strftime("%Y%m%d-%H%M%S")}.pt'
results_name = f'{dataset_name}_20191206'
gridsearch_name = f'{dataset_name}_20191206'

# ────────────────────────────────────────────────────────────────────────────────
# # 1. Load the dataset(s)
# todo perform some exploratory data analysis
# todo check for missing/NA values
X,y =  load_wine_quality()
# print(df.describe())

# ────────────────────────────────────────────────────────────────────────────────
# # 2. Split the dataset(s) into training and test
# X = df[df.columns[1:-3]]
# y = df[df.columns[-1]]
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
# ```
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))])
# 
# categorical_transformer = Pipeline(steps=[
    # ('onehot', OneHotEncoder(handle_unknown='ignore'))])

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
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
# categorical_features = df.select_dtypes(include=['object']).columns

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
    SVR(),
    DecisionTreeRegressor(random_state=random_seed),
    RandomForestRegressor(random_state=random_seed),
    AdaBoostRegressor(random_state=random_seed),
    GaussianProcessRegressor(random_state=random_seed),
    LinearRegression(),
    MLPRegressor(random_state=random_seed)
    ]

grid_params = {
        'SVR':
        {
            'SVR__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'SVR__C': list(np.logspace(-5, 15, num=11, base=2)),
            'SVR__gamma': list(np.logspace(-15, 3, num=10, base=2)),
        },
        'DecisionTreeRegressor':
        {
            'DecisionTreeRegressor__criterion': ['mse', 'friedman_mse', 'mae'],
            'DecisionTreeRegressor__max_depth': list(np.linspace(1, 32, 32, endpoint=True)),
            # 'DecisionTreeRegressor__splitter': ['best', 'random'],
            # 'DecisionTreeRegressor__min_samples_split': list(np.linspace(0.1, 1.0, 10, endpoint=True)),
            # 'DecisionTreeRegressor__min_samples_leaf': list(np.linspace(0.1, 0.5, 5, endpoint=True)),
            # 'DecisionTreeRegressor__max_features': list(np.linspace(0.1, 0.5, 5, endpoint=True)),
        },
        'RandomForestRegressor':
        {
            'RandomForestRegressor__n_estimators': list(np.arange(10, 101)),
            'RandomForestRegressor__criterion': ['mse', 'mae'],
            'RandomForestRegressor__max_depth': list(np.linspace(1, 32, 32, endpoint=True)),
            # 'RandomForestRegressor__splitter': ['best', 'random'],
            # 'RandomForestRegressor__min_samples_split': list(np.linspace(0.1, 1.0, 10, endpoint=True)),
            # 'RandomForestRegressor__min_samples_leaf': list(np.linspace(0.1, 0.5, 5, endpoint=True)),
            # 'RandomForestRegressor__max_features': list(np.linspace(0.1, 0.5, 5, endpoint=True)),
        },
        'AdaBoostRegressor':
        {
            'AdaBoostRegressor__n_estimators': list(np.arange(10, 51)),
            'AdaBoostRegressor__learning_rate': list(np.linspace(0.1, 1, 10, endpoint=True)),
        },
        'GaussianProcessRegressor':{
            'GaussianProcessRegressor__kernel': [1.0*RBF(1.0), DotProduct() + WhiteKernel()],
            'GaussianProcessRegressor__alpha': list(np.linspace(1e-20, 1, 20, endpoint=True)),
            'GaussianProcessRegressor__normalize_y': [True, False],
        },
        'LinearRegression':{ 
        },
        'MLPRegressor':
        {
            'MLPRegressor__activation': ['identity', 'logistic', 'tanh', 'relu'],
            'MLPRegressor__solver': ['lbfgs', 'sgd', 'adam'],
            'MLPRegressor__hidden_layer_sizes': [(1,)] + [(i,) for i in np.arange(10, 101, 10)],
            'MLPRegressor__learning_rate': ['constant', 'invscaling', 'adaptive'],
            'MLPRegressor__max_iter': list(np.arange(100, 501, 50)),            
        }
    }

results = []

for classifier in classifiers:
    pipe = Pipeline(steps=[('preprocessor', preprocessor),
                    (classifier.__class__.__name__, classifier)])

    # Perform a grid search on the entire pipeline of the current classifier
    # Note: to disable the grid search, comment the following three lines,
    # and call fit() and predict() directly on the pipe object
    if (grid_search):
        grid_clf = GridSearchCV(pipe, grid_params[classifier.__class__.__name__], n_jobs=8)
        grid_clf.fit(X_train, y_train)
        # pipe.fit(X_train, y_train)

        # best params are stored in the grid_clf.best_params_ object:
        ## print(grid_clf.best_params_)
        
        # store the best classifier for each classifier
        pipe = grid_clf.best_estimator_
    else:
        pipe.fit(X_train, y_train)

    # just a piece of code in case we need access to the classifier in the pipe
    ## print(pipe[classifier.__class__.__name__])

    y_pred = pipe.predict(X_test)

    result = {
                'Classifier': classifier.__class__.__name__,
                'Score': pipe.score(X_test, y_test),
                'Explained variance score': explained_variance_score(y_test, y_pred),
                'Max error': max_error(y_test, y_pred),
                'Mean absolute error': mean_absolute_error(y_test, y_pred),
                'Mean squared error': mean_squared_error(y_test, y_pred),
                # 'Mean squared logarithmic error': mean_squared_log_error(y_test, y_pred),
                'Median absolute error': median_absolute_error(y_test, y_pred),
                'R^2 score': r2_score(y_test, y_pred)
            }
    results.append(result)

results_df = pd.DataFrame(data=results, index=None,
                        columns=['Classifier', 'Score', 'Explained variance score', 'Max error', 'Mean absolute error', 'Mean squared error', 'Median absolute error', 'R^2 score'])
results_df.index = [''] * len(results_df)


# ## 3.3 Hyperparameter tuning


# Exampel:
# ```python
# grid_param = { 
#     'classifier__n_estimators': [200, 500],
#     'classifier__max_features': ['auto', 'sqrt', 'log2'],
#     'classifier__max_depth' : [4,5,6,7,8],
#     'classifier__criterion' :['gini', 'entropy']}
# from sklearn.model_selection import GridSearchCV
# CV = GridSearchCV(rf, grid_param, n_jobs= 1)
                  
# CV.fit(X_train, y_train)  
# print(CV.best_params_)    
# print(CV.best_score_)
# ```

# # 4. Output
# ## 4.1 Results
# Jupyter Notebook
results_df = results_df.sort_values(by=['Score'], ascending=False)
display(results_df)
# Save the dataframe
results_df.to_pickle(results_path + results_name)
results_df.to_csv(results_path + results_name + '.csv')
# ## 4.1 Figures