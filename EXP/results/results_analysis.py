import pandas as pd

res = pd.read_csv('rna_models.csv')
models = res['Models'].unique()
for mod in models:
    mean = res.loc[res['Models'] == mod, 'R-Squared'].mean()
    print(mod, mean)
