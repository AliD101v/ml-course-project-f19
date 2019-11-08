import pandas as pd

def load_Adult():
    # Read the dataset 
    # Prepare the feature names
    featureNames = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'Prediction']

    df_train = pd.read_csv('data/Adult Data Set/adult.data',
        delimiter=',', skipinitialspace=True, header=None, names=featureNames, na_values='?')

    df_test = pd.read_csv('data/Adult Data Set/adult.test',
        delimiter=',', skipinitialspace=True, header=None, skiprows=1, names=featureNames, na_values='?')
    # Map the targets from categorical (string) values to 0/1.
    prediction_mapping = {'<=50K':0, '>50K':1}
    df_train['Prediction'] = df_train['Prediction'].map(prediction_mapping).astype(int)

    prediction_mapping = {'<=50K.':0, '>50K.':1}
    df_test['Prediction'] = df_test['Prediction'].map(prediction_mapping).astype(int)
    return df_train, df_test
# df,_ = load_Adult()
# print(df[df.columns[:-1]].head())
# df.info()
# df.describe()