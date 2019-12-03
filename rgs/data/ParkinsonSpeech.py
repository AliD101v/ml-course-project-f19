import pandas as pd

def load_Parkinson_Speech():
    # Read the dataset 
    featureNames = ['Subject id', 'Jitter (local)','Jitter (local, absolute)','Jitter (rap)','Jitter (ppq5)','Jitter (ddp)','Shimmer (local)','Shimmer (local, dB)','Shimmer (apq3)','Shimmer (apq5)',' Shimmer (apq11)','Shimmer (dda)','AC','NTH','HTN','Median pitch','Mean pitch','Standard deviation','Minimum pitch','Maximum pitch','Number of pulses','Number of periods','Mean period','Standard deviation of period','Fraction of locally unvoiced frames','Number of voice breaks','Degree of voice breaks']

    df = pd.read_csv('data/Parkinson Speech/train_data.txt',
        delimiter=',', header=None, names=featureNames + ['UPDRS','class'])

    # Not currently using the test dataset, becuase it does not contain the UPDRS ground truth labels
    # df_test = pd.read_csv('data/Parkinson Speech/test_data.txt',
    #     delimiter=',', header=None, names=featureNames + ['class'])

    return df

# df = load_Parkinson_Speech()
# print('training data:')
# print(df.head())
# df.info()
# df.describe()
# print()
# df_test = df_test[df_test.columns[-2]]
# print('test data:')
# print(df_test.head())
# df_test.info()
# df_test.describe()