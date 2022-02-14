import forgi
from itertools import permutations
import sys
import numpy as np
import pandas as pd
import math
import torch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def element_size(el, feats):
    if el in 'fthms':
        return feats[1] - feats[0] + 1
    elif el == 'i':
        return sum(map(lambda d: d[1] - d[0] + 1, zip(feats[::2], feats[1::2])))
    else:
        print('Unrecognized element ', el)
        sys.exit(1)


def get_graph_structure(df, file):
    elements_map = {'f': [1, 0, 0, 0, 0],  # dangling start
                    't': [0, 1, 0, 0, 0],  # dangling end
                    'h': [0, 0, 1, 0, 0],  # hairpin loop
                    'i': [0, 0, 0, 1, 0],  # internal loop
                    'm': [0, 0, 0, 0, 1],  # multi-loop
                    }
    f = open(file, 'r')
    seq_id = f.readlines()[0][1:-1]
    off_rate = df.loc[df['sequence_id'] == seq_id, 'OFF'].values[0]
    on_rate = df.loc[df['sequence_id'] == seq_id, 'ON'].values[0]

    cg = forgi.load_rna(file, allow_many=False)
    elements = cg.defines
    edges = cg.edges
    nodes = list(edges.keys())
    nodes_feats = {}  # 2 features: {node unique id: [one hot encoded element, num nucleotides]}
    edges_feats = [[], []]
    edges_att = []
    nodes_copy = [el for el in nodes if 's' not in el]
    nodes_mapping = {el: k + 1 for k, el in enumerate(nodes_copy)}
    print(nodes_mapping)
    for n in nodes:
        el, id = list(n)
        if el in ['f', 't', 'i', 'm', 'h']:
            nodes_feats[nodes_mapping[n]] = [elements_map[el], element_size(el, elements[n])]
        elif el == 's':
            stem_len = element_size(el, elements[n])
            neighbours = list(edges[n])
            covalent = permutations(neighbours, 2)
            for x, y in covalent:
                edges_feats[0].append(nodes_mapping[x])
                edges_feats[1].append(nodes_mapping[y])
                edges_att.append(stem_len)
        else:
            print('element not recognized ', n)
            sys.exit(2)

    return nodes_feats, edges_feats, edges_att, off_rate, on_rate


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


if __name__ == '__main__':
    print('Building graph ...')
    df = pd.read_csv('/rds/general/user/hf721/home/RNA_Kinetics/00_data/Toehold_Dataset_Final_2019-10-23.csv')
    nodes, edges, edges_att, off_val, on_val = get_graph_structure(df, 'smallpox_tile_23043.txt')
    print(nodes)
    print(edges[0])
    print(edges[1])
    print(edges_att)
    print(off_val, on_val)
