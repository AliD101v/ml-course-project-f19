import pandas as pd

def load_yeastData():
    df=df=pd.read_table('C:/Users/sidha/OneDrive/Documents/ml-course-project-f19/ml-course-project-f19/data/Yeast/yeast.data',sep='\s+',header=None)
    categorical_columns = df.select_dtypes(['category','object']).columns

# convert categorical data to numeric values
    df[categorical_columns] = df[categorical_columns].apply( lambda x:x.astype('category') )
    df[categorical_columns] = df[categorical_columns].apply( lambda x:x.cat.codes )
    
    X=df[df.columns[1:9]]
    y=df[df.columns[9]]
    return X,y

