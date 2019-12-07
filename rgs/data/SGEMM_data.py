import numpy as np 
import pandas as pd

#### import pandas as pd
from sklearn import preprocessing

def load_SGEMM():
    df=pd.read_csv("C:/Users/sidha/OneDrive/Documents/ml-course-project-f19/ml-course-project-f19/data/SGEMM/sgemm_product.csv",delimiter=',')
    y=(df['Run1 (ms)']+df['Run1 (ms)']+df['Run1 (ms)']+df['Run1 (ms)'])/4
    X=df[df.columns[0:14]]
    return X,y

# df=load_SGEMM()



