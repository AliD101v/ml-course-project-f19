import pandas as pd

def load_BikeSharing():
    # Read the dataset 

    df = pd.read_csv('data/Bike Sharing/hour.csv',
        delimiter=',', header=0, index_col='instant')

    return df

# df = load_BikeSharing()
# print(df[df.columns[1:-3]].head())
# df.info()
# df.describe()