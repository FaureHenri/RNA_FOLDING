import pandas as pd
from tqdm import tqdm
import RNA
from os.path import join


toehold_data = '/rds/general/user/hf721/home/RNA_Kinetics/00_data/Toehold_Dataset_Final_2019-10-23.csv'
riboset_path = '/rds/general/user/hf721/ephemeral/riboset'
switch_off = 'switchOFF'
switch_on = 'switchON'

in_cols = ['seq_SwitchOFF_GFP', 'seq_SwitchON_GFP']
out_cols = ['ON', 'OFF']
qc_level = 1.1

data = pd.read_csv(toehold_data, usecols=['sequence_id'] + in_cols + out_cols + ['QC_ON_OFF'])
data.set_index('sequence_id', inplace=True)
data = data[data.QC_ON_OFF >= qc_level].dropna(subset=out_cols)
data.drop_duplicates(inplace=True)
data.replace('T', 'U', regex=True, inplace=True)

for idx, sequence in tqdm(data.iterrows()):
    off_val, on_val = sequence['OFF'], sequence['ON']
    for k, in_col in enumerate(in_cols):
        seq = sequence[in_col]
        (ss, _) = RNA.fold(seq)
        if in_col == 'seq_SwitchOFF_GFP':
            filename = idx + '.txt'
            filepath = join(riboset_path, switch_off, filename)
        else:
            filename = idx + '.txt'
            filepath = join(riboset_path, switch_on, filename)

        with open(filepath, 'w') as outfile:
            outfile.write('>' + idx + '\n')
            outfile.write(seq + '\n')
            outfile.write(ss + '\n')
            # outfile.write('\n')
            # outfile.write(str(off_val) + ',' + str(on_val))
