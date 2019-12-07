import numpy as np
import pandas as pd

def load_steel_faults_data() :
    filePath = "C:/Users/sidha/OneDrive/Documents/ml-course-project-f19/ml-course-project-f19/data/Steel plate faults/Faults.NNA"
    data_csv = pd.read_csv(filePath, delim_whitespace=True, header=None)
    X= data_csv.iloc[:,:27]
    y= data_csv.iloc[:,27:]

    y = np.dot(y.to_numpy(),[1,2,3,4,5,6,7])
    return X,y