import os
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# Helper function to pass string DNA/RNA sequence to one-hot
def rna2onehot(seq):
    seq = 'GGG' + seq
    onehot_map = {'A': [1, 0, 0, 0],
                  'U': [0, 1, 0, 0],
                  'C': [0, 0, 1, 0],
                  'G': [0, 0, 0, 1]}

    onehot_encoded_seq = np.stack([onehot_map[el] for el in ''.join([seq])])

    return onehot_encoded_seq


def load_sequences_and_targets(in_cols, out_cols, qc_level=1.1):
    data = pd.read_csv('../00_data/Toehold_Dataset_Final_2019-10-23.csv')

    # Perform QC checks and drop rows with NaN outputs
    data = data[data.QC_ON_OFF >= qc_level].dropna(subset=out_cols)
    data.drop_duplicates(inplace=True)

    if not os.path.isfile('../00_data/rna_seqs.csv'):
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
        df_data_input.to_csv('../00_data/rna_seqs.csv', index=False)
    else:
        df_data_input = pd.read_csv('../00_data/rna_seqs.csv')

    df_data_output = data[out_cols]

    return df_data_input, df_data_output


def train_main(config):
    in_columns = config['in_cols']
    out_columns = config['out_cols']

    data_in, data_out = load_sequences_and_targets(in_cols=in_columns, out_cols=out_columns, qc_level=config['qc_level'])

    r2_list = []
    for i in range(10):
        np.random.seed(i)

        X_train, X_test, y_train, y_test = train_test_split(data_in, data_out, train_size=0.75, random_state=i)

        reg = MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam', alpha=0.0001, batch_size=64,
                           learning_rate_init=0.0005, learning_rate='adaptive', max_iter=50, early_stopping=True,
                           n_iter_no_change=2, verbose=True)
        reg.fit(X_train.values, y_train.values.ravel())
        preds = reg.predict(X_test.values)

        r2 = r2_score(y_test.values.ravel(), preds, multioutput='raw_values')
        r2_list.append(r2)
        print("R2 score:", r2)

        mse = mean_squared_error(y_test.values.ravel(), preds, multioutput='raw_values')
        print("MSE score:", mse)

        mae = mean_absolute_error(y_test.values.ravel(), preds, multioutput='raw_values')
        print("MAE score:", mae)

    return r2_list


if __name__ == '__main__':
    hyperparameter_defaults = dict(
        in_cols=['switch'],
        out_cols=['ON'],
        qc_level=1.1,
    )

    r2_scores = train_main(hyperparameter_defaults)
    print(r2_scores)
