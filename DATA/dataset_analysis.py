from Utils.data_utils import load_sequences_and_targets
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def unpivot(frame):
    N, K = frame.shape
    data = {
        "Translation initiation rate": frame.to_numpy().ravel("F"),
        "Switch": np.asarray(frame.columns).repeat(N)
    }
    return pd.DataFrame(data, columns=["Switch", "Translation initiation rate"])


# data_path = '/rds/general/user/hf721/home/RNA_FOLDING/DATA/Toehold_Dataset_Final_2019-10-23.csv'
# data_in, data_out = load_sequences_and_targets(data_path=data_path, in_cols=['seq_SwitchON_GFP'],
#                                                out_cols=['ON', 'OFF'], qc_level=1.1, analysis=True)
#
# df1 = data_in.duplicated()
# df2 = data_in.duplicated(subset=['seq_SwitchON_GFP'])

# data_out = unpivot(data_out)
# g = sns.displot(data_out, x="Translation initiation rate", hue="Switch", kde=True)
# plt.savefig('ImbalancedData.png')

# data_in['source_sequence'] = data_in['source_sequence'].str.replace(': ', '_')
# data_in['source_sequence'] = data_in['source_sequence'].str.replace('human ', 'human_')
# data_in['source_sequence'] = data_in['source_sequence'].str.replace('influenza_h1n1', 'H1N1')
# data_in['source_sequence'] = data_in['source_sequence'].str.replace('influenza_h3n2', 'H3N2')
# data_in['source_sequence'] = data_in['source_sequence'].str.replace('papilloma', 'HPV')
# data_in['seq_type'] = data_in['source_sequence'].str.split('_').str[0]
#
# var_map = {'random': 'Random sequence', 'human': 'Human genes (906x)', 'cardiovirus': 'Picornavirus',
#            'cosavirus': 'Picornavirus', 'coxsackie': 'Picornavirus', 'poliovirus': 'Picornavirus',
#            'dengue': 'Flavivirus', 'west nile': 'Flavivirus', 'yellow fever': 'Flavivirus', 'zika': 'Flavivirus',
#            'H1N1': 'Influenza', 'H3N2': 'Influenza', 'marburg': 'Filovirus', 'ebola': 'Filovirus', 'astrovirus': 'Others',
#            'chikungunya': 'Others', 'hantavirus': 'Others', 'lassa': 'Others', 'leishmania': 'Others',
#            'HPV': 'Others', 'rabies': 'Others', 'smallpox': 'Others'}
#
# var_sort = ['Random sequence', 'Human genes (906x)', 'Picornavirus', 'Flavivirus', 'Influenza', 'Filovirus', 'Others']
#
# variants_c = data_in['seq_type'].value_counts(ascending=True).to_frame()
# variants_c['Families'] = variants_c.index.map(mapper=var_map)
# variants_c['Variants'] = variants_c.index
# variants_c.Families = variants_c.Families.astype("category")
# variants_c.Families.cat.set_categories(var_sort, inplace=True)
# variants_c.sort_values(["Families"])
# p = sns.barplot(data=variants_c, x='Variants', y='seq_type', hue='Families', order=var_map, hue_order=var_sort, palette='deep', dodge=False)
# p.set_yscale("log")
# p.set(ylabel='Number of variants')
# plt.setp(p.get_xticklabels(), rotation=30, ha='right')
# plt.legend(loc='upper center', ncol=3)
# plt.tight_layout()
# plt.savefig('data_variants.png')

cols = ['trigger', 'loop1', 'switch', 'loop2', 'stem1', 'atg', 'stem2', 'linker', 'post_linker', 'ON', 'QC_ON_OFF']
data = pd.read_csv('/home/hfaure/RNA_FOLDING/DATA/Toehold_Dataset_Final_2019-10-23.csv', usecols=cols)
data = data[data.QC_ON_OFF >= 1.1].dropna(subset=['ON'])
# data.drop_duplicates(subset=in_cols, inplace=True)
for col in cols:
    print(col, data[col].nunique())

print('table shape:', data.shape)
print('done')
