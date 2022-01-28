import pandas as pd
import numpy as np
import math
import time
import random
from datetime import date
from tqdm import tqdm
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim


# Helper function one hot encode rna sequences
def rna2onehot(seq):
    seq = 'GGG' + seq
    onehot_map = {'A': [1, 0, 0, 0],
                  'U': [0, 1, 0, 0],
                  'C': [0, 0, 1, 0],
                  'G': [0, 0, 0, 1]}

    onehot_encoded_seq = np.stack([onehot_map[el] for el in ''.join([seq])])

    return onehot_encoded_seq


# Helper function to load rna sequences and translation inition rates
def load_sequences_and_targets(in_cols, out_cols, qc_level=1.1):
    data = pd.read_csv('../00_data/Toehold_Dataset_Final_2019-10-23.csv')

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


class RNADataset(Dataset):
    def __init__(self, rna_seqs, rna_labs):
        self.rna_seqs = torch.from_numpy(rna_seqs)
        self.rna_labs = torch.from_numpy(rna_labs)

    def __len__(self):
        return len(self.rna_seqs)

    def __getitem__(self, idx):
        rna_seq = self.rna_seqs[idx]
        rna_seq = rna_seq.reshape(-1, 4)
        rna_len = rna_seq.shape[0]

        data_fcn = np.zeros((7, rna_len, rna_len))

        # all canonical base pairs + G-U  (A-U, U-A, C-G, G-C, G-U, U-G)
        perm = [(0, 1), (1, 0), (2, 3), (3, 2), (3, 1), (1, 3)]
        for n, cord in enumerate(perm):
            i, j = cord
            data_fcn[n] = np.matmul(rna_seq[:, i].reshape(-1, 1), rna_seq[:, j].reshape(1, -1))

        data_fcn[6] = 1 - data_fcn.sum(axis=0)
        # data_fcn[6] = creatmat(rna_seq)

        return data_fcn, self.rna_labs[idx]


class CNNet(nn.Module):
    def __init__(self, img_ch=[7, 32, 64, 128], output_ch=3, bloc_wd=148):
        super(CNNet, self).__init__()
        kernel = 2
        stride = 2
        # output_size = (input - kernel_size + 2 * padding) / stride + 1
        self.layers = nn.Sequential()
        for th, ch in enumerate(img_ch):
            if th < len(img_ch) - 1:
                self.layers.add_module(name='Conv' + str(th + 1),
                                       module=nn.Sequential(
                                           nn.Conv2d(ch, img_ch[th + 1], kernel_size=5, stride=1, padding=1, bias=True),
                                           nn.BatchNorm2d(img_ch[th + 1]),
                                           nn.ReLU(inplace=True)
                                       ))
                bloc_wd = int((bloc_wd - 5 + 2)/1 + 1)
                self.layers[th].add_module(name='Maxpool' + str(th + 1), module=nn.MaxPool2d(kernel_size=kernel, stride=stride))
                bloc_wd = int((bloc_wd - kernel) / stride + 1)

        self.fc = nn.Sequential(nn.Linear(img_ch[-1]*bloc_wd*bloc_wd, 16),
                                nn.BatchNorm1d(16),
                                nn.ReLU(inplace=True),
                                nn.Dropout(0.3),
                                nn.Linear(16, 16),
                                nn.ReLU(inplace=True),
                                nn.Linear(16, output_ch))

    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def train(contact_net, trainloader, testloader, epochs, log_epoch, patience_init, num_outputs, device):
    criterion = nn.L1Loss(reduction='mean')
    u_optimizer = optim.Adam(contact_net.parameters(), lr=0.001, eps=1e-8)
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
            if batch_idx % 50 == 0 and batch_idx > 1:
                current_loss = np.mean(losses)
                message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}' \
                    .format((batch_idx + 1) * len(data_batch[0]), len(trainloader.dataset),
                            100. * batch_idx / len(trainloader), current_loss)
                print(message)
                losses = []

        total_loss /= (batch_idx + 1)
        print('MAE loss at epoch {}: {:.6f}'.format(epoch, total_loss))

        if (epoch + 1) % log_epoch == 0 or epoch + 1 == epochs:
            val_loss, val_r2, val_mse, val_mae = validate(testloader=testloader, model=contact_net, loss_fn=criterion,
                                                          epoch=epoch, out_features=num_outputs, device=device)

            if val_r2 >= best_r2:
                no_improvement_counter = 0
                print(f'\nR2 score did improve from {best_r2} to {val_r2}')
                best_r2 = val_r2
                today = date.today()
                date_today = today.strftime("%b_%d_%Y")
                opt_model = f'Unet_{date_today}'
                # torch.save(contact_net.state_dict(), f'./Opt_CNN/{opt_model}.pt')
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

    return best_r2


def validate(testloader, model, loss_fn, epoch, out_features, device):
    print(f'\nValidation in process at epoch {epoch + 1} ...')
    with torch.no_grad():
        model.eval()
        final_loss = 0
        global_r2 = [0] * out_features
        global_mse = [0] * out_features
        global_mae = [0] * out_features

        for batch_idx, load_data in enumerate(testloader):
            inputs, targets = load_data
            inputs = torch.Tensor(inputs.float()).to(device)
            targets = torch.Tensor(targets.float()).to(device)

            outputs = model(inputs)

            loss = loss_fn(outputs, targets)
            final_loss += loss.item()

            current_r2 = r2_score(targets.cpu(), outputs.cpu(), multioutput='raw_values')
            global_r2 = [sum(x) for x in zip(global_r2, current_r2)]

            current_mse = mean_squared_error(targets.cpu(), outputs.cpu(), multioutput='raw_values')
            global_mse = [sum(x) for x in zip(global_mse, current_mse)]

            current_mae = mean_absolute_error(targets.cpu(), outputs.cpu(), multioutput='raw_values')
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


def train_main(config):
    seed = config['seed']
    print("Random Seed: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device used:', device)
    if str(device) == 'cuda':
        torch.cuda.manual_seed(seed)

    data_in, data_out = load_sequences_and_targets(in_cols=config['in_cols'], out_cols=config['out_cols'],
                                                   qc_level=config['qc_level'])

    dataix, dataiy = data_in.shape
    print(f'data_in shape: ({dataix}, {dataiy})')

    dataox, dataoy = data_out.shape
    print(f'data_out shape: ({dataox}, {dataoy})')

    scaler = QuantileTransformer()
    data_out = scaler.fit_transform(data_out)

    X_train, X_test, y_train, y_test = train_test_split(data_in, data_out, train_size=0.75, random_state=seed)
    rna_train = RNADataset(X_train.to_numpy(), y_train)
    rna_test = RNADataset(X_test.to_numpy(), y_test)
    print(f'length of train set: {len(rna_train)}\nlength of validation set: {len(rna_test)}')

    train_loader = DataLoader(rna_train, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    test_loader = DataLoader(rna_test, batch_size=config['batch_size'], shuffle=True, num_workers=0)

    fccnet = CNNet(img_ch=[7, 32, 64, 128], output_ch=len(config['out_cols']), bloc_wd=int(dataiy/4))
    fccnet.to(device)

    r2_cnn = train(contact_net=fccnet, trainloader=train_loader, testloader=test_loader, epochs=100, log_epoch=1,
                   patience_init=10, num_outputs=len(config['out_cols']), device=device)

    return r2_cnn


if __name__ == "__main__":
    params = dict(
        in_cols=['switch'],
        out_cols=['ON'],
        qc_level=1.1,
        batch_size=64,
        seed=123
    )

    r2_cnn = train_main(params)
    print("R-squared:", r2_cnn)
