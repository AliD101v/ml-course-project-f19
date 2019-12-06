#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
from sklearn import preprocessing
def load_concrete():
    df = pd.read_excel("Concrete_Data.xls",)
    X=df[df.columns[:8]]
    y=df[df.columns[-1]]
    return X,y




