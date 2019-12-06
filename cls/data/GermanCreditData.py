#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
from sklearn.model_selection import train_test_split
def load_GermanCredit():
    df=pd.read_table('german.data-numeric',sep='\s+',header=None)
    X = df[df.columns[:-1]]
    y = df[df.columns[-1]]

    return X,y


# df=load_GermanCredit()

