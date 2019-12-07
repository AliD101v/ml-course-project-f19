import numpy as np
import pandas as pd

#This dataset takes a lot of time to run due to it's size.
def load_set_2_data():

    with open("C:/Users/sidha/OneDrive/Documents/ml-course-project-f19/ml-course-project-f19/data/Merck/ACT2_competition_training.csv") as f:
        cols = f.readline().rstrip('\n').split(',')
    X = np.loadtxt("C:/Users/sidha/OneDrive/Documents/ml-course-project-f19/ml-course-project-f19/data/Merck/ACT2_competition_training.csv", delimiter=',', usecols=range(2, len(cols)), skiprows=1, dtype=np.uint8)
    y = np.loadtxt("C:/Users/sidha/OneDrive/Documents/ml-course-project-f19/ml-course-project-f19/data/Merck/ACT2_competition_training.csv", delimiter=',', usecols=[1], skiprows=1)
#     outfile = TemporaryFile()
#     np.savez(outfile, X, y)
#     return outfile
    
    X= pd.DataFrame(X)
    y= pd.DataFrame(y)
    return X,y

def  load_set_4_data():
    #Read file to df with delimeter and header from 0-n
    with open("C:/Users/sidha/OneDrive/Documents/ml-course-project-f19/ml-course-project-f19/data/Merck/ACT4_competition_training.csv") as f:
        cols_4 = f.readline().rstrip('\n').split(',')
    X = np.loadtxt("C:/Users/sidha/OneDrive/Documents/ml-course-project-f19/ml-course-project-f19/data/Merck/ACT4_competition_training.csv", delimiter=',', usecols=range(2, len(cols_4)), skiprows=1, dtype=np.uint8)
    y = np.loadtxt("C:/Users/sidha/OneDrive/Documents/ml-course-project-f19/ml-course-project-f19/data/Merck/ACT4_competition_training.csv", delimiter=',', usecols=[1], skiprows=1)
    
    X= np.asarray(X)
    y= np.asarray(y)
    return X,y
    #     outfile_4 = TemporaryFile()
#     np.savez(outfile_4, X_4, y_4)
#     return outfile