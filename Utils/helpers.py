import numpy as np
import pandas as pd
import os
import os.path as osp
import math
import yaml
import torch
import random
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import wandb
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)


def set_seed(seed, device):
    print("Random Seed: ", seed)
    print('device used:', device)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if str(device) == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.manual_seed(seed)


def ifnotmkdir(dir_path):
    if not osp.isdir(dir_path):
        os.mkdir(dir_path)


def hist_residues(actual, predicted):
    actual = list(map(list, zip(*actual)))[0]
    predicted = list(map(list, zip(*predicted)))[0]
    residues = pd.DataFrame({'actual': actual, 'predicted': predicted})
    residues.to_csv('residues.csv')
    lower_b = np.linspace(0.0, 0.9, 10)
    upper_b = np.linspace(0.1, 1.0, 10)
    bins = [[round(l, 1), round(u, 1)] for (l, u) in zip(lower_b, upper_b)]

    actual_arr = np.array(actual)

    actual_mean = {}
    pred_mean = {}
    res_mean = {}
    res_q1 = {}
    res_q3 = {}
    for k, (l_b, u_b) in enumerate(bins):
        if k < len(bins) - 1:
            idx_bin = np.where((actual_arr >= l_b) & (actual_arr < u_b))[0].tolist()
        else:
            idx_bin = np.where((actual_arr >= l_b) & (actual_arr <= u_b))[0].tolist()
        a_bin = [actual[i] for i in idx_bin]
        p_bin = [predicted[i] for i in idx_bin]
        res_bin = [abs(a_bin[i] - p_bin[i]) for i in range(len(a_bin))]
        distrib = np.percentile(res_bin, [25, 50, 75], interpolation='midpoint')
        try:
            actual_mean[k] = sum(a_bin) / len(a_bin)
            pred_mean[k] = sum(p_bin) / len(p_bin)
            res_q1[k], res_mean[k], res_q3[k] = distrib[0], distrib[1], distrib[2]
        except:
            actual_mean[k], pred_mean[k] = 0, 0
            res_q1[k], res_mean[k], res_q3[k] = 0, 0, 0

    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.arange(1, len(bins) + 1)
    ax.bar(x=x, height=list(actual_mean.values()), width=0.25, align='center', label='Averaged True value')
    ax.bar(x=x, height=list(pred_mean.values()), width=0.07, align='center', label='Averaged Predicted value')
    xnew = np.linspace(1, len(bins), 10 * len(bins))

    spl_q1 = make_interp_spline(x, list(res_q1.values()), k=3)
    res_q1_smooth = spl_q1(xnew)
    ax.plot(xnew, res_q1_smooth, color='darkturquoise', linestyle='dashed', label='Residue 1st Quartile')

    spl_mean = make_interp_spline(x, list(res_mean.values()), k=3)
    res_mean_smooth = spl_mean(xnew)
    ax.plot(xnew, res_mean_smooth, color='teal', label='Residue Median')

    spl_q3 = make_interp_spline(x, list(res_q3.values()), k=3)
    res_q3_smooth = spl_q3(xnew)
    ax.plot(xnew, res_q3_smooth, color='darkslategrey', linestyle='dashed', label='Residue 3rd Quartile')

    ax.legend()
    plt.ylim([0, 1.0])
    plt.tight_layout()
    plt.title('Mean Absolute Error of the Translation Initiation rate over 10 bins equally distributed from 0 to 1')
    plt.savefig('resHist.png')
    print('Plot done')


def hist_families(actual, predicted, indexes):
    indexes = [el.tolist() for el in indexes]
    actual = list(map(list, zip(*actual)))[0]
    predicted = list(map(list, zip(*predicted)))[0]
    families = pd.DataFrame({'indices': indexes, 'actual': actual, 'predicted': predicted})
    families.to_csv('families.csv')

    print('Families'' plot done')


# Training Loop
def launch_train(device, num_epochs, model, trainloader, testloader, loss_fn, opt, out_features, patience_init,
                 log_epoch, graph=False, logs=False, msg=False):
    print('\nTraining in process ...')
    best_r2 = -math.inf
    early_stopping = False
    no_improvement_counter = 0
    actual_tir = None  # tir := translation initiation rate
    predicted_tir = None
    for epoch in range(num_epochs):
        model.train()
        losses = []
        total_loss = 0

        batch_idx = None
        for batch_idx, load_data in enumerate(trainloader):
            opt.zero_grad()

            if graph:
                load_data = load_data.to(device)
                outputs = model(load_data)
                loss = loss_fn(outputs, load_data.y.unsqueeze(dim=1))
            else:
                inputs, targets = load_data
                targets = targets.unsqueeze(1)
                inputs, targets = torch.Tensor(inputs.float()).to(device), torch.Tensor(targets.float()).to(device)
                targets = targets.reshape((targets.shape[0], out_features))
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)

            loss.backward()
            opt.step()
            losses.append(loss.item())
            total_loss += loss.item()
            if batch_idx % 50 == 0 and batch_idx > 1:
                current_loss = np.mean(losses)
                message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}' \
                    .format((batch_idx + 1) * len(load_data[0]), len(trainloader.dataset),
                            100. * batch_idx / len(trainloader), current_loss)
                if msg:
                    print(message)
                losses = []

        total_loss /= (batch_idx + 1)
        print('MSE loss at epoch {}: {:.6f}'.format(epoch + 1, total_loss))
        if (epoch + 1) % log_epoch == 0 or epoch + 1 == num_epochs:
            val_loss, val_r2, val_mse, val_mae, actual, predicted = launch_validation(device=device,
                                                                                      testloader=testloader,
                                                                                      model=model,
                                                                                      loss_fn=loss_fn,
                                                                                      epoch=epoch,
                                                                                      out_features=out_features,
                                                                                      graph=graph)

            if logs:
                wandb.log({'epoch': epoch + 1, 'r2_score': val_r2})
                wandb.log({'epoch': epoch + 1, 'mse_score': val_mse})
                wandb.log({'epoch': epoch + 1, 'mae_score': val_mae})

            if val_r2 >= best_r2:
                no_improvement_counter = 0
                print(f'\nR2 score did improve from {best_r2} to {val_r2}')
                best_r2 = val_r2
                actual_tir = actual
                predicted_tir = predicted
                if logs:
                    wandb.log({'epoch': epoch + 1, 'R2_metric': best_r2})

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
                print('\nPlotting bar chart ...')
                hist_residues(actual_tir, predicted_tir)
                # hist_families(actual_tir, predicted_tir, indexes)
                break

    return best_r2


# Validation Loop
def launch_validation(device, testloader, model, loss_fn, epoch, out_features, graph):
    print(f'\nValidation in process at epoch {epoch + 1} ...')
    with torch.no_grad():
        model.eval()
        final_loss = 0
        global_r2 = [0] * out_features
        global_mse = [0] * out_features
        global_mae = [0] * out_features
        predicted = []
        actual = []

        for batch_idx, load_data in enumerate(testloader):
            if graph:
                load_data = load_data.to(device)
                outputs = model(load_data)
                targets = load_data.y
                loss = loss_fn(outputs, targets.unsqueeze(dim=1))
            else:
                inputs, targets = load_data
                targets = targets.unsqueeze(1)
                inputs, targets = torch.Tensor(inputs.float()).to(device), torch.Tensor(targets.float()).to(device)
                targets = targets.reshape((targets.shape[0], out_features))
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)

            actual += targets.cpu().tolist()
            predicted += outputs.cpu().tolist()

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

    return final_loss, np.mean(global_r2), np.mean(global_mse), np.mean(global_mae), actual, predicted


if __name__ == '__main__':
    acts = [0.91, 0.81, 0.23, 0.45, 0.97, 0.20, 0.04, 0.56, 0.83, 0.77, 0.39, 0.10, 0.99, 0.23, 0.64, 0.68]
    preds = [0.87, 0.88, 0.44, 0.34, 0.98, 0.01, 0.41, 0.69, 0.77, 0.85, 0.54, 0.35, 1.0, 0.07, 0.49, 0.76]
    acts = [[x] for x in acts]
    preds = [[x] for x in preds]
    hist_residues(acts, preds)
