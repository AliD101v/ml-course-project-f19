import pandas as pd
import numpy as np
from scipy.io.arff import loadarff

def load_seismic_data():

    file_path = filePath = "C:/Users/sidha/OneDrive/Documents/ml-course-project-f19/ml-course-project-f19/data/Seismic Bumps/seismic-bumps.arff"

    raw_data = loadarff(filePath) # data type tuple 

    seismic_df = pd.DataFrame(raw_data[0])
    categorical_columns = seismic_df.select_dtypes(['category','object']).columns

    # convert categorical data to numeric values
    seismic_df[categorical_columns] = seismic_df[categorical_columns].apply( lambda x:x.astype('category') )
    seismic_df[categorical_columns] = seismic_df[categorical_columns].apply( lambda x:x.cat.codes )

    X = seismic_df.iloc[:,:18]
    y = seismic_df['class']

    return X,y

