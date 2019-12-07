import pandas as pd
from sklearn.model_selection import train_test_split

def load_GermanCredit():
    df=pd.read_table('C:/Users/sidha/OneDrive/Documents/ml-course-project-f19/ml-course-project-f19/data/German credit/german.data',sep='\s+',header=None)
    
    X = df[df.columns[:-1]]
    y = df[df.columns[-1]]

    return X,y


# df=load_GermanCredit()

