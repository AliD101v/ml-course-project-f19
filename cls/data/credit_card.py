import numpy as np
import pandas as pd

def load_credit_card_data():
    # data reading
    filePath = "C:/Users/sidha/OneDrive/Documents/ml-course-project-f19/ml-course-project-f19/data/Default of credit card clients/default_of_credit_card_clients.xls"
    data_excel = pd.read_excel(filePath, sheet_name='Data')

    X = data_excel.iloc[:,:23]
    y = data_excel.iloc[:,23]
    return X,y
