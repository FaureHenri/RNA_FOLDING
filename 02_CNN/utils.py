import numpy as np
import math
import torch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def hydrogen_bonds(sequence, structure):
    """
    For each closing parenthesis, I find the matching opening one and store their index in the base_pairs list.
    The assigned list is used to keep track of the assigned opening parenthesis
    """
    sequence = list(sequence)
    h_bonds = {'A': 2,
               'U': 2,
               'C': 3,
               'G': 3}

    opened = [idx for idx, i in enumerate(structure) if i == '(']
    closed = [idx for idx, i in enumerate(structure) if i == ')']

    assert len(opened) == len(closed)

    assigned = []
    base_pairs = []

    for close_idx in closed:
        for open_idx in opened:
            if open_idx < close_idx:
                if open_idx not in assigned:
                    candidate = open_idx
            else:
                break
        assigned.append(candidate)
        base_pairs.append([[candidate, close_idx], h_bonds[sequence[candidate]]])
        assigned.append(close_idx)
        base_pairs.append([[close_idx, candidate], h_bonds[sequence[close_idx]]])

    assert len(base_pairs) == 2 * len(opened)

    return base_pairs


def adjacency_mat(pairing, seq_len):
    mat = np.zeros((1, seq_len, seq_len))

    for i in range(seq_len):
        if i < seq_len - 1:
            mat[0, i, i + 1] = 5
        if i > 0:
            mat[0, i, i - 1] = 5

    # Adapt weight for non-canonical base pairing, or regarding the number of hydrogen bonds
    for [i, j], w in pairing:
        mat[0, i, j] = w

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
            targets = targets.unsqueeze(1)
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
            targets = targets.unsqueeze(1)
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
    seq = 'GGGAACCAAACACACAAACGCACAAAAAGCUACAGCAUCUGUGAGGAAUGCCUAACAGAGGAGAAGGCAUAUGUCACAGAUGAACCUGGCGGCAGCGCAAAAGAUGCGUAAAGGAGAA'
    ss = '............................(((.((((((((((((.(.((((((...........)))))).).)))))))))...)))..))).(((((.....))))).........'
    matrix = adjacency_mat(hydrogen_bonds(seq, ss), len(ss))
    print(matrix)
