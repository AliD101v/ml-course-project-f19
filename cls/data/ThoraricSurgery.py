import pandas as pd
from scipy.io.arff import loadarff 

def cat_to_num(df):
    """
    Convert a Pandas column (Series) from categorical/qualitative values to continuous/quantitative values.
    """
    feature_vals = sorted(df.unique())
    feature_vals_mapping = dict(zip(feature_vals, range(0, len(feature_vals) +
                             1)))
    return df.map(feature_vals_mapping).astype(int)

def load_ThoraricSurgery():
    # Read the dataset 
    # Prepare the feature names
    featureNames = ['DGN', 'PRE4', 'PRE5', 'PRE6', 'PRE7', 'PRE8', 'PRE9', 'PRE10', 'PRE11', 'PRE14', 'PRE17', 'PRE19', 'PRE25', 'PRE30', 'PRE32', 'AGE', 'Risk1Y']

    raw_data = loadarff('data/Thoracic Surgery/ThoraricSurgery.arff')
    df = pd.DataFrame(raw_data[0])
    df.columns = featureNames

    # # Map the targets from categorical (byte literal) values to integers.
    df[df.columns[0]] = df[df.columns[0]].apply(lambda x: x[3:])
    categorical_features = df.select_dtypes(include=['object']).columns
    for c in categorical_features:
        df[c] = df[c].apply(lambda x: x.decode())
        df[c] = cat_to_num(df[c])

    return df
# df = load_ThoraricSurgery()
# print(df.head())
# df.info()
# df.describe()
# print(df.isnull().values.any())
# print(df.to_string())