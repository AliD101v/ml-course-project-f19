import pandas as pd
from scipy.io.arff import loadarff

def load_diabetic_data():
    file_path = "C:/Users/sidha/OneDrive/Documents/ml-course-project-f19/ml-course-project-f19/data/Diabetic Retinopathy/messidor_features.arff"
    # path = "ml-course-project-f19\data\Diabetic Retinopathy\messidor_features.arff"
    raw_data = loadarff(file_path) # data type tuple 

    messidor_columns = ['quality','pre_screening','ma1','ma2','ma3','ma4','ma5','ma6','ex1','ex2','ex3','ex4','ex5','ex6','ex7','ex8','ex9','dist_macula_optic_center','result_am_fm','class']
    messidor_df = pd.DataFrame(raw_data[0])
    messidor_df.columns = messidor_columns

    X = messidor_df.iloc[:,0:19]
    y = messidor_df.iloc[:,19]

    return X,y

