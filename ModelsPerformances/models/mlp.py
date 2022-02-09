import numpy as np
import random
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from utils import load_sequences_and_targets, launch_train


class RNADataset(Dataset):
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


def train_main(config):
    seed = config['seed']
    print("Random Seed: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = config['device']

    if str(device) == 'cuda':
        torch.cuda.manual_seed(seed)

    in_columns = config['in_cols']
    out_columns = config['out_cols']
    num_outputs = len(out_columns)

    data_in, data_out = load_sequences_and_targets(data_path=config['data'], in_cols=in_columns, out_cols=out_columns, qc_level=config['qc_level'])

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
    train_loader = DataLoader(rna_train, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    test_loader = DataLoader(rna_test, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])

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
                          loss_fn=criterion, opt=optimizer, out_features=num_outputs, patience_init=10, log_epoch=5)

    return r2_mlp


if __name__ == '__main__':
    hyperparameter_defaults = dict(
        device='cpu',
        data='../00_data/Toehold_Dataset_Final_2019-10-23.csv',
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
        num_workers=4,
        seed=123
    )

    r2_mlp = train_main(hyperparameter_defaults)
    print("R-squared:", r2_mlp)
