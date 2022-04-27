import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import math


# Helper function one hot encode rna sequences
def rna2onehot(seq):
    onehot_map = {'A': [1, 0, 0, 0],
                  'U': [0, 1, 0, 0],
                  'C': [0, 0, 1, 0],
                  'G': [0, 0, 0, 1]}

    onehot_encoded_seq = np.stack([onehot_map[el] for el in ''.join([seq])])

    return onehot_encoded_seq


# Helper function to load rna sequences and translation initiation rates
def load_sequences_and_targets(data_path, in_cols, out_cols, qc_level=1.1):
    data = pd.read_csv(data_path)

    # Perform QC checks and drop rows with NaN outputs
    data = data[data.QC_ON_OFF >= qc_level].dropna(subset=out_cols)
    data.drop_duplicates(inplace=True)

    data.replace('T', 'U', regex=True, inplace=True)

    print(f'Data encoding in process...')
    time.sleep(1)
    tqdm.pandas()
    df_data_input = None
    for col in in_cols:
        df = data[col]
        encoded = df.progress_apply(rna2onehot).values
        encoded_arr = np.array(list(encoded))
        len_seqs = len(encoded_arr[0])
        num_nucleotides = len(encoded_arr[0][0])
        encoded_arr = encoded_arr.reshape(-1, len_seqs * num_nucleotides)
        df_tmp = pd.DataFrame(encoded_arr)

        if df_data_input is None:
            df_data_input = df_tmp
        else:
            df_data_input = pd.concat([df_data_input, df_tmp], axis=1)
    df_data_input.reset_index(drop=True, inplace=True)
    num_samples = df_data_input.shape[1]
    df_data_input.columns = list(range(num_samples))
    df_data_output = data[out_cols]

    return df_data_input, df_data_output


def paired(x, y):
    # A-U
    if x == [1, 0, 0, 0] and y == [0, 1, 0, 0]:
        return 2
    # G-C
    elif x == [0, 0, 0, 1] and y == [0, 0, 1, 0]:
        return 3
    # G-U
    elif x == [0, 0, 0, 1] and y == [0, 1, 0, 0]:
        return 0.8
    # U-A
    elif x == [0, 1, 0, 0] and y == [1, 0, 0, 0]:
        return 2
    # C-G
    elif x == [0, 0, 1, 0] and y == [0, 0, 0, 1]:
        return 3
    # U-G
    elif x == [0, 1, 0, 0] and y == [0, 0, 0, 1]:
        return 0.8
    else:
        return 0


def gaussian(x):
    return math.exp(-0.5*(x*x))


def creatmat(data):
    mat = np.zeros([len(data), len(data)])
    for i in range(len(data)):
        for j in range(len(data)):
            coefficient = 0
            for add in range(30):
                if i - add >= 0 and j + add < len(data):
                    score = paired(list(data[i - add]), list(data[j + add]))
                    if score == 0:
                        break
                    else:
                        coefficient = coefficient + score * gaussian(add)
                else:
                    break
            if coefficient > 0:
                for add in range(1, 30):
                    if i + add < len(data) and j - add >= 0:
                        score = paired(list(data[i + add]), list(data[j - add]))
                        if score == 0:
                            break
                        else:
                            coefficient = coefficient + score * gaussian(add)
                    else:
                        break
            mat[[i], [j]] = coefficient
    return mat
