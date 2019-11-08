import pandas as pd

def load_WDBC():
    # Read the dataset 
    # Prepare the feature names
    featureNames_org = ['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness', 'Concavity', 'Concave points', 'Symmetry', 'Fractal dimension']
    featureNames = [f'{s} (M)' for s in featureNames_org]
    featureNames += [f'{s} (SE)' for s in featureNames_org]
    featureNames += [f'{s} (W)' for s in featureNames_org]
    featureNames = ['ID', 'Diagnosis'] + featureNames

    df = pd.read_csv('data/Breast Cancer Wisconsin/wdbc.data',
        delimiter=',', header=None, names=featureNames)
    # drop the ID column
    df = df.drop(columns='ID')
    # Map the targets from categorical (string) values to 0/1
    prediction_mapping = {'B':0, 'M':1}
    df['Diagnosis'] = df['Diagnosis'].map(prediction_mapping).astype(int)
    return df
# df = load_WDBC()
# print(df.head())
# df.info()
# df.describe()