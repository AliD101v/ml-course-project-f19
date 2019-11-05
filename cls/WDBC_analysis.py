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

