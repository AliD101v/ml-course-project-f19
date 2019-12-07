import pandas as pd
from sklearn.model_selection import train_test_split

def load_austrailian():
    df=pd.read_table('C:/Users/sidha/OneDrive/Documents/ml-course-project-f19/ml-course-project-f19/data/Australian credit/australian.dat',sep='\s+',header=None)
    df.columns=['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11','X12','X13','X14','Y']
    # df.head()
    df_cat=df[['X1','X4','X5','X6','X8','X9','X11','X12','Y']]
    df_cont=df[['X2','X3','X7','X10','X13','X14','Y']]
    # df_cat.head()
    X=df.drop('Y',axis=1)
    y=df['Y']
    return X,y
# X,y=load_austrailian()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

