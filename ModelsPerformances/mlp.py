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
from torch import nn
from torch.utils.data import DataLoader


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


class RNADataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)

        return x


class MLP(nn.Module):
    def __init__(self, filters, dropout):
        super().__init__()
        num_layers = len(filters)
        self.layers = nn.Sequential(Flatten())
        for i in range(num_layers):
            if i + 2 < num_layers:
                self.layers.add_module(name=str('layer' + str(i + 1)), module=self.dense_1d(filters[i], filters[i + 1], dropout))
            elif i + 1 < num_layers:
                self.layers.add_module(name=str('layer' + str(i + 1)), module=nn.Linear(filters[i], filters[i + 1]))
            else:
                break

    def forward(self, x):
        return self.layers(x)

    @staticmethod
    def dense_1d(in_x, out_x, dropout):
        return nn.Sequential(
            nn.Linear(in_x, out_x),
            nn.BatchNorm1d(out_x),
            nn.ReLU(True),
            nn.Dropout(dropout)
        )


def criterion_mse(predicted, ground_truth):
    loss = nn.MSELoss(reduction='mean')

    return loss(predicted, ground_truth)


def launch_train(device, num_epochs, model, trainloader, testloader, loss_fn, opt, out_features, patience_init, log_epoch):
    print('\nTraining in process ...')
    best_r2 = -math.inf
    early_stopping = False
    no_improvement_counter = 0
    for epoch in range(num_epochs):
        model.train()
        losses = []
        total_loss = 0

        batch_idx = None
        for batch_idx, load_data in enumerate(trainloader):
            inputs, targets = load_data
            inputs, targets = torch.Tensor(inputs.float()).to(device), torch.Tensor(targets.float()).to(device)

            # targets = targets.reshape((targets.shape[0], out_features))
            opt.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            total_loss += loss.item()
            if batch_idx % 10 == 0 and batch_idx > 1:
                current_loss = np.mean(losses)
                message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}' \
                    .format((batch_idx + 1) * len(load_data[0]), len(trainloader.dataset),
                            100. * batch_idx / len(trainloader), current_loss)
                # print(message)
                losses = []

        total_loss /= (batch_idx + 1)
        # train_loss = loss_name + '_train_loss'
        # wandb.log({'epoch': epoch + 1, train_loss: total_loss})
        print('MSE loss at epoch {}: {:.6f}'.format(epoch + 1, total_loss))
        if (epoch + 1) % log_epoch == 0 or epoch + 1 == num_epochs:
            val_loss, val_r2, val_mse, val_mae = launch_validation(device=device, testloader=testloader, model=model,
                                                                   loss_fn=loss_fn, epoch=epoch, out_features=out_features)
            # wandb.log({'epoch': epoch + 1, 'r2_score': val_r2})
            # wandb.log({'epoch': epoch + 1, 'mse_score': val_mse})
            # wandb.log({'epoch': epoch + 1, 'mae_score': val_mae})

            if val_r2 >= best_r2:
                no_improvement_counter = 0
                print(f'\nR2 score did improve from {best_r2} to {val_r2}')
                best_r2 = val_r2
                # wandb.log({'epoch': epoch + 1, 'R2_metric': best_r2})
                today = date.today()
                date_today = today.strftime("%b_%d_%Y")
                opt_model = f'mlp_1D_{date_today}'
                # torch.save(model.state_dict(), f'./Opt_MLP/{opt_model}.pt')
            else:
                no_improvement_counter += log_epoch
                if no_improvement_counter > patience_init:
                    early_stopping = True
                print(f'\nR2 score did not improve, is still {best_r2}')

            if epoch + 1 < num_epochs and not early_stopping:
                print('\nTraining in process ...')
            else:
                if early_stopping:
                    print(f'\nEarly Stopping at epoch {epoch + 1}')
                print('\nTraining process has finished.')
                break

    return best_r2


def launch_validation(device, testloader, model, loss_fn, epoch, out_features):
    print(f'\nValidation in process at epoch {epoch + 1} ...')
    with torch.no_grad():
        model.eval()
        final_loss = 0
        global_r2 = [0] * out_features
        global_mse = [0] * out_features
        global_mae = [0] * out_features

        for batch_idx, load_data in enumerate(testloader):
            inputs, targets = load_data
            inputs, targets = torch.Tensor(inputs.float()).to(device), torch.Tensor(targets.float()).to(device)
            targets = targets.reshape((targets.shape[0], out_features))
            # optimizer.zero_grad()
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

    in_columns = config['in_cols']
    out_columns = config['out_cols']
    num_outputs = len(out_columns)

    data_in, data_out = load_sequences_and_targets(in_cols=in_columns, out_cols=out_columns, qc_level=config['qc_level'])

    dataix, dataiy = data_in.shape
    print(f'data_in shape: ({dataix}, {dataiy})')

    dataox, dataoy = data_out.shape
    print(f'data_out shape: ({dataox}, {dataoy})')

    scaler_init = config['scaler_init']
    if scaler_init:
        scaler = QuantileTransformer()
        data_out = scaler.fit_transform(data_out)

    X_train, X_test, y_train, y_test = train_test_split(data_in, data_out, train_size=0.75, random_state=seed)
    rna_train = RNADataset(X_train.to_numpy(), y_train)
    rna_test = RNADataset(X_test.to_numpy(), y_test)
    print(f'length of train set: {len(rna_train)}\nlength of validation set: {len(rna_test)}')
    train_loader = DataLoader(rna_train, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    test_loader = DataLoader(rna_test, batch_size=config['batch_size'], shuffle=True, num_workers=0)

    filters = config['filters']
    filters.insert(0, dataiy)
    filters.insert(len(filters), num_outputs)
    mlp = MLP(filters=filters, dropout=config['dropout'])
    mlp.to(device)

    optimizer = torch.optim.Adam(mlp.parameters(), lr=config['learning_rate'], eps=config['epsilon'],
                                 weight_decay=config['weight_decay'])

    # Mean Squared Error
    criterion = criterion_mse

    r2_mlp = launch_train(device=device, num_epochs=config['epochs'], model=mlp, trainloader=train_loader, testloader=test_loader,
                          loss_fn=criterion, opt=optimizer, out_features=num_outputs, patience_init=30, log_epoch=5)

    return r2_mlp


if __name__ == '__main__':
    hyperparameter_defaults = dict(
        in_cols=['switch'],
        out_cols=['ON'],
        qc_level=1.1,
        scaler_init=True,
        epochs=200,
        filters=[128, 64, 32],
        optimizer='adam',
        loss_fn='mse',
        learning_rate=0.001,
        weight_decay=0.000005,
        epsilon=0.00000001,
        dropout=0.3,
        batch_size=64,
        seed=2
    )

    r2_mlp = train_main(hyperparameter_defaults)
    print("R-squared:", r2_mlp)
