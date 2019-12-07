import numpy as np 
import pandas as pd 

def load_crime_data():
    df =pd.read_table('C:/Users/sidha/OneDrive/Documents/ml-course-project-f19/ml-course-project-f19/data/Communities and Crime/communities.data',delimiter=',',header=None)
    df = df.replace('?', None) 
    # df = df.where(df =="\?",None)
    df = df.drop()
    df = df.dropna(axis="columns")
    X=df[df.columns[0:127]]
    y=df[df.columns[127]]
    return X,y

X,y = load_crime_data()

print(X)
print(y)