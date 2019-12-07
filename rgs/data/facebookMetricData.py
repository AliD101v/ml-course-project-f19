import pandas as pd
from sklearn import preprocessing


def load_facebookMetric():
    
    df = pd.read_csv("C:/Users/sidha/OneDrive/Documents/ml-course-project-f19/ml-course-project-f19/data/Facebook/dataset_Facebook.csv",delimiter=';')
    categorical_columns = df.select_dtypes(['category','object']).columns

# convert categorical data to numeric values
    df[categorical_columns] = df[categorical_columns].apply( lambda x:x.astype('category') )
    df[categorical_columns] = df[categorical_columns].apply( lambda x:x.cat.codes )

    scaler = preprocessing.StandardScaler()
    X=df[df.columns[1:6]]
    y=df[df.columns[7]]
    return X,y

