import pandas as pd
import numpy as np
import os
import math
import time
import random
from datetime import date
from tqdm import tqdm
from itertools import product
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from network import U_Net as FCNNet
from utils import get_config


# Helper function to pass string DNA/RNA sequence to one-hot
def rna2onehot(seq):
    # get sequence into an array
    seq_array = np.array(list(seq))

    # integer encode the sequence
    label_encoder = LabelEncoder()
    integer_encoded_seq = label_encoder.fit_transform(seq_array)

    # one hot the sequence
    onehot_encoder = OneHotEncoder(sparse=False, categories='auto')

    # reshape because that's what OneHotEncoder likes
    integer_encoded_seq = integer_encoded_seq.reshape(len(integer_encoded_seq), 1)
    onehot_encoded_seq = onehot_encoder.fit_transform(integer_encoded_seq)

    return onehot_encoded_seq


def load_sequences_and_targets(in_cols, out_cols, qc_level=1.1):
    data = pd.read_csv('../../00_data/Toehold_Dataset_Final_2019-10-23.csv')
    tqdm.pandas()

    input_cols = in_cols
    output_cols = out_cols

    # Perform QC checks and drop rows with NaN outputs
    data = data[data.QC_ON_OFF >= qc_level].dropna(subset=output_cols)

    # Show sample of dataframe structure
    print(data.head())

    print(f'Data encoding in process...')
    time.sleep(1)
    data_tmp = data[input_cols]
    df_data_input_tmp = data_tmp.progress_apply(rna2onehot)
    data_input = np.array(list(df_data_input_tmp.values))

    # Data Output selection (QC filtered, OutColumns Only & Drop NaNs)
    df_data_output = data[output_cols]
    data_output = df_data_output.values.astype('float32')

    return data_input, data_output


def paired(x, y):
    # A-U
    if x == [1, 0, 0, 0] and y == [0, 0, 0, 1]:
        return 2
    # G-C
    elif x == [0, 0, 1, 0] and y == [0, 1, 0, 0]:
        return 3
    # G-U
    elif x == [0, 0, 1, 0] and y == [0, 0, 0, 1]:
        return 0.8
    # U-A
    elif x == [0, 0, 0, 1] and y == [1, 0, 0, 0]:
        return 2
    # C-G
    elif x == [0, 1, 0, 0] and y == [0, 0, 1, 0]:
        return 3
    # U-G
    elif x == [0, 0, 0, 1] and y == [0, 0, 1, 0]:
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
            # Why 30??
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


class RNADataset(Dataset):
    def __init__(self, rna_seqs, rna_labs):
        self.rna_seqs = torch.from_numpy(rna_seqs)
        self.rna_labs = torch.from_numpy(rna_labs)

    def __len__(self):
        return len(self.rna_seqs)

    def __getitem__(self, idx):
        rna_seq = self.rna_seqs[idx]
        rna_len = rna_seq.shape[0]

        data_fcn = np.zeros((17, rna_len, rna_len))

        # all potential base pairs
        perm = list(product(np.arange(4), np.arange(4)))

        # all canonical base pairs + G-U  (A-U, U-A, C-G, G-C, G-u, U-G)
        # perm2 = [(0, 3), (3, 0), (1, 2), (2, 1), (2, 3), (3, 2)]
        for n, cord in enumerate(perm):
            i, j = cord
            data_fcn[n] = np.matmul(rna_seq[:, i].reshape(-1, 1), rna_seq[:, j].reshape(1, -1))

        data_fcn[16] = creatmat(rna_seq)

        return data_fcn, self.rna_labs[idx]


def train(contact_net, trainloader, testloader, epochs, log_epoch, patience_init, num_outputs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.L1Loss(reduction='mean')
    u_optimizer = optim.Adam(contact_net.parameters())
    best_r2 = -math.inf
    early_stopping = False
    no_improvement_counter = 0

    print('start training...')
    for epoch in range(epochs):
        losses = []
        total_loss = 0

        contact_net.train()
        batch_idx = None
        for batch_idx, data_batch in enumerate(trainloader):
            seq_embeddings, seq_lab = data_batch
            seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)
            seq_lab_batch = torch.Tensor(seq_lab.float()).to(device)

            pred_fn = contact_net(seq_embedding_batch)

            loss_u = criterion(pred_fn, seq_lab_batch)

            u_optimizer.zero_grad()
            loss_u.backward()
            u_optimizer.step()

            losses.append(loss_u.item())
            total_loss += loss_u.item()
            if batch_idx % 10 == 0 and batch_idx > 1:
                print("current GPU used", torch.cuda.memory_allocated())
                current_loss = np.mean(losses)
                message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}' \
                    .format((batch_idx + 1) * len(data_batch[0]), len(trainloader.dataset),
                            100. * batch_idx / len(trainloader), current_loss)
                print(message)
                losses = []

        total_loss /= (batch_idx + 1)
        print('MAE loss at epoch {}: {:.6f}'.format(epoch, total_loss))

        if epoch >= 0:
            today = date.today()
            date_today = today.strftime("%b_%d_%Y")
            opt_model = f'Ufold_{date_today}'
            torch.save(contact_net.state_dict(), f'./Opt_CNN/{opt_model}.pt')

        if (epoch + 1) % log_epoch == 0 or epoch + 1 == epochs:
            val_loss, val_r2, val_mse, val_mae = validate(testloader=testloader, model=contact_net, loss_fn=criterion,
                                                          epoch=epoch, out_features=num_outputs)

            if val_r2 >= best_r2:
                no_improvement_counter = 0
                print(f'\nR2 score did improve from {best_r2} to {val_r2}')
                best_r2 = val_r2
                today = date.today()
                date_today = today.strftime("%b_%d_%Y")
                opt_model = f'Unet_{date_today}'
                torch.save(contact_net.state_dict(), f'./Opt_CNN/{opt_model}.pt')
            else:
                no_improvement_counter += log_epoch
                if no_improvement_counter > patience_init:
                    early_stopping = True
                print(f'\nR2 score did not improve, is still {best_r2}')

            if epoch + 1 < epochs and not early_stopping:
                print('\nTraining in process ...')
            else:
                if early_stopping:
                    print(f'\nEarly Stopping at epoch {epoch + 1}')
                print('\nTraining process has finished.')
                break


def validate(testloader, model, loss_fn, epoch, out_features):
    print(f'\nValidation in process at epoch {epoch + 1} ...')
    with torch.no_grad():
        model.eval()
        final_loss = 0
        global_r2 = [0] * out_features
        global_mse = [0] * out_features
        global_mae = [0] * out_features

        for batch_idx, load_data in enumerate(testloader):
            inputs, targets = load_data
            inputs, targets = inputs.float(), targets.float()
            targets = targets.reshape((targets.shape[0], out_features))
            # optimizer.zero_grad()
            outputs = model(inputs)

            loss = loss_fn(outputs, targets)
            final_loss += loss.item()

            current_r2 = r2_score(outputs, targets, multioutput='raw_values')
            global_r2 = [sum(x) for x in zip(global_r2, current_r2)]

            current_mse = mean_squared_error(outputs, targets, multioutput='raw_values')
            global_mse = [sum(x) for x in zip(global_mse, current_mse)]

            current_mae = mean_absolute_error(outputs, targets, multioutput='raw_values')
            global_mae = [sum(x) for x in zip(global_mae, current_mae)]

        final_loss = final_loss / (batch_idx + 1)
        global_r2 = [x / (batch_idx + 1) for x in global_r2]
        global_mse = [x / (batch_idx + 1) for x in global_mse]
        global_mae = [x / (batch_idx + 1) for x in global_mae]

    print('Validation has achieved an R2 score of {:.6f}'.format(np.mean(global_r2)))

    labels = ['ON    |', 'OFF   |', 'ON_OFF|']
    labels = labels[:out_features]
    print('\n      | mae score | mse score | r2 score ')
    [print('{} {:.6f}  | {:.6f}  | {:.6f}'.format(*x)) for x in zip(labels, global_mae, global_mse, global_r2)]

    return final_loss, np.mean(global_r2), np.mean(global_mse), np.mean(global_mae)


if __name__ == "__main__":
    seed = 123
    print("Random Seed: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(device) == 'cuda':
        torch.cuda.manual_seed(seed)

    config = get_config('config_cnn.yaml')
    params = config['parameters']

    data_in, data_out = load_sequences_and_targets(in_cols=params['in_cols'], out_cols=params['out_cols'],
                                                   qc_level=params['qc_level'])

    X_train, X_test, y_train, y_test = train_test_split(data_in, data_out, train_size=0.75, random_state=seed)

    rna_train = RNADataset(X_train, y_train)
    rna_test = RNADataset(X_test, y_test)

    train_loader = DataLoader(rna_train, batch_size=64, shuffle=True, num_workers=8)
    test_loader = DataLoader(rna_test, batch_size=64, shuffle=True, num_workers=8)

    fccnet = FCNNet(img_ch=[17, 16, 32, 64, 128], output_ch=len(params['out_cols']))
    fccnet.to(device)

    train(contact_net=fccnet, trainloader=train_loader, testloader=test_loader, epochs=10, log_epoch=1, patience_init=5,
          num_outputs=len(params['out_cols']))

    print('used memory', torch.cuda.max_memory_allocated())
    print('Job done!')
