import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import yaml
import math
import torch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)


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


# Training Loop
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
        print('MSE loss at epoch {}: {:.6f}'.format(epoch + 1, total_loss))
        if (epoch + 1) % log_epoch == 0 or epoch + 1 == num_epochs:
            val_loss, val_r2, val_mse, val_mae = launch_validation(device=device, testloader=testloader, model=model,
                                                                   loss_fn=loss_fn, epoch=epoch,
                                                                   out_features=out_features)
            if val_r2 >= best_r2:
                no_improvement_counter = 0
                print(f'\nR2 score did improve from {best_r2} to {val_r2}')
                best_r2 = val_r2
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


# Validation Loop
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
