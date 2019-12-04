import numpy as np
import pandas as pd

def load_NOAAWeatherBellevue():
    featureNames = ['Temperature','Dew Point','Sea Level Pressure','Visibility','Average Wind Speed','Maximum Sustained Wind Speed','Maximum Temperature','Minimum Temperature']
    # Read the dataset
    df_X = pd.read_csv('data/NOAA Weather Bellevue/NEweather_data.csv',
        delimiter=',', header=None, names=featureNames)

    df_y = pd.read_csv('data/NOAA Weather Bellevue/NEweather_class.csv',
        delimiter=',', header=None, names=['Rain'])

    return df_X, df_y

# df_X,df_y = load_NOAAWeatherBellevue()
# print('Features:')
# print(df_X.head())
# df_X.info()
# df_X.describe()
# print('Targets:')
# print(df_y.head())
# df_y.info()
# df_y.describe()