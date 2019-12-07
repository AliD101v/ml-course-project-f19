import numpy as np
import pandas as pd

def load_wine_quality() :
    filePath_red = "C:/Users/sidha/OneDrive/Documents/ml-course-project-f19/ml-course-project-f19/data/Wine Quality/winequality-red.csv"
    filePath_white = "C:/Users/sidha/OneDrive/Documents/ml-course-project-f19/ml-course-project-f19/data/Wine Quality/winequality-white.csv"

    data_csv_red = pd.read_csv(filePath_red, delimiter=";")
    data_csv_white = pd.read_csv(filePath_white, delimiter=";")

    X_red = data_csv_red.iloc[:,:11]
    y_red = data_csv_red.iloc[:,11]

    X_white = data_csv_white.iloc[:,:11]
    y_white = data_csv_white.iloc[:,11]

    X_red.append(X_white)
    y_red.append(y_white)

    return X_red,y_red