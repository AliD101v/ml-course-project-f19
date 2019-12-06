#!/usr/bin/env python
# coding: utf-8

# In[29]:


#### import pandas as pd
from sklearn import preprocessing

def load_SGEMM():
    df=pd.read_csv("sgemm_product.csv",delimiter=',')
    
    
    y=(df['Run1 (ms)']+df['Run1 (ms)']+df['Run1 (ms)']+df['Run1 (ms)'])/4
    X=df[df.columns[0:14]]
    return X,y

# df=load_SGEMM()



