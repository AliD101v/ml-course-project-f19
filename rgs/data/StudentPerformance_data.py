#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
from sklearn import preprocessing

def load_studentPerformance():
    df=pd.read_csv("student-por.csv",delimiter=';')
    categorical_columns = df.select_dtypes(['category','object']).columns

# convert categorical data to numeric values
    df[categorical_columns] = df[categorical_columns].apply( lambda x:x.astype('category') )
    df[categorical_columns] = df[categorical_columns].apply( lambda x:x.cat.codes )
    
    df=load_studentPerformance()
    X=df[df.columns[0:32]]
    y=df[df.columns[32]]
    return X,y


