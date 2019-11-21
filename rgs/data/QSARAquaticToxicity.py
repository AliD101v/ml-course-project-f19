import pandas as pd

def load_QSAR_aquatic_toxicity():
    # Read the dataset 
    featureNames = ['TPSA(Tot)', 'SAacc', 'H-050', 'MLOGP', 'RDCHI', 'GATS1p', 'nN', 'C-040', 'LC50 [-LOG(mol/L)]']

    df = pd.read_csv('data/QSAR aquatic toxicity/qsar_aquatic_toxicity.csv',
        delimiter=';', header=None, names=featureNames)

    return df

# df = load_QSAR_aquatic_toxicity()
# print(df.head())
# df.info()
# df.describe()