import numpy as np
from itertools import product
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PerfModels.models.mlp import MLP
from Utils.data_utils import load_sequences_and_targets, creatmat
from Utils.helpers import launch_train, set_seed
import wandb
from dotmap import DotMap


class RNADataset(Dataset):
    def __init__(self, rna_seqs, rna_labs, num_channels):
        self.rna_seqs = torch.from_numpy(rna_seqs)
        self.rna_labs = torch.from_numpy(rna_labs)
        self.num_channels = num_channels

    def __len__(self):
        return len(self.rna_seqs)

    def __getitem__(self, idx):
        rna_seq = self.rna_seqs[idx]
        rna_seq = rna_seq.reshape(-1, 4)
        rna_len = rna_seq.shape[0]

        data_fcn = np.zeros((self.num_channels, rna_len, rna_len))

        perm = None
        if self.num_channels == 17:
            # all potential base pairs
            perm = list(product(np.arange(4), np.arange(4)))
        elif self.num_channels == 7:
            # all canonical base pairs + G-U  (A-U, U-A, C-G, G-C, G-U, U-G)
            perm = [(0, 1), (1, 0), (2, 3), (3, 2), (3, 1), (1, 3)]

        for n, cord in enumerate(perm):
            i, j = cord
            data_fcn[n] = np.matmul(rna_seq[:, i].reshape(-1, 1), rna_seq[:, j].reshape(1, -1))

        if self.num_channels == 7:
            data_fcn[6] = 1 - data_fcn.sum(axis=0)
        elif self.num_channels == 17:
            data_fcn[16] = creatmat(rna_seq)

        return data_fcn, self.rna_labs[idx]


class CNNet(nn.Module):
    def __init__(self, img_ch=(7, 32, 64), output_ch=1, bloc_wd=148, dropout=0.3):
        super(CNNet, self).__init__()
        kernel = 2
        stride = 2
        # output_size = (input - kernel_size + 2 * padding) / stride + 1
        self.layers = nn.Sequential()
        for th, ch in enumerate(img_ch):
            if th < len(img_ch) - 1:
                self.layers.add_module(name='Conv' + str(th + 1),
                                       module=nn.Sequential(
                                           nn.Conv2d(ch, img_ch[th + 1], kernel_size=(5, 5), stride=(1, 1), padding=1,
                                                     bias=True),
                                           nn.BatchNorm2d(img_ch[th + 1]),
                                           nn.ReLU(inplace=True)
                                       ))
                bloc_wd = int((bloc_wd - 5 + 2*1)/1 + 1)
                self.layers[th].add_module(name='Maxpool' + str(th + 1), module=nn.MaxPool2d(kernel_size=kernel,
                                                                                             stride=stride))
                bloc_wd = int((bloc_wd - kernel + 2*0)/stride + 1)

        print("Tensor width: ", bloc_wd)
        print("input shape of MLP", img_ch[-1]*bloc_wd*bloc_wd)
        self.fc = MLP(filters=(img_ch[-1]*bloc_wd*bloc_wd, 64, 32, 16, output_ch), dropout=dropout)

    def forward(self, x):
        # for l in self.layers:
        #     print(x.size())
        #     x = l(x)
        x = self.layers(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def criterion_mse(predicted, ground_truth):
    loss = nn.L1Loss(reduction='mean')

    return loss(predicted, ground_truth)


def train_main(config, logs=False):
    seed = config.seed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(seed, device)

    num_outputs = len(config.out_cols)

    data_in, data_out = load_sequences_and_targets(data_path=config.data, in_cols=config.in_cols,
                                                   out_cols=config.out_cols, qc_level=config.qc_level)

    dataix, dataiy = data_in.shape
    print(f'data_in shape: ({dataix}, {dataiy})')

    dataox, dataoy = data_out.shape
    print(f'data_out shape: ({dataox}, {dataoy})')

    scaler = QuantileTransformer()
    data_out = scaler.fit_transform(data_out)

    X_train, X_test, y_train, y_test = train_test_split(data_in, data_out, train_size=0.75, random_state=seed)
    rna_train = RNADataset(X_train.to_numpy(), y_train, 7)
    rna_test = RNADataset(X_test.to_numpy(), y_test, 7)
    print(f'length of train set: {len(rna_train)}\nlength of validation set: {len(rna_test)}')

    train_loader = DataLoader(rna_train, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    test_loader = DataLoader(rna_test, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    fccnet = CNNet(img_ch=config.filters, output_ch=len(config.out_cols), bloc_wd=int(dataiy/4), dropout=config.dropout)
    fccnet.to(device)

    optimizer = torch.optim.Adam(fccnet.parameters(), lr=config.learning_rate, eps=config.epsilon,
                                 weight_decay=config.weight_decay)

    # Mean Squared Error
    criterion = criterion_mse

    if logs:
        wandb.watch(fccnet, criterion, log="all")

    r2_cnn = launch_train(device=device, num_epochs=config.epochs, model=fccnet, trainloader=train_loader,
                          testloader=test_loader, loss_fn=criterion, opt=optimizer, out_features=num_outputs,
                          patience_init=4, log_epoch=1, graph=False, logs=logs, msg=False)

    return r2_cnn


if __name__ == "__main__":
    hyperparameter_defaults = dict(
        device='cpu',
        data='/rds/general/user/hf721/home/RNA_FOLDING/DATA/Toehold_Dataset_Final_2019-10-23.csv',
        in_cols=['seq_SwitchON_GFP'],
        out_cols=['ON'],
        qc_level=1.1,
        scaler_init=True,
        epochs=200,
        filters=[7, 32, 64],
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

    # wandb.init()
    # config = wandb.config
    config = DotMap(hyperparameter_defaults)
    r2_cnn = train_main(config, logs=False)
    print("R-squared:", r2_cnn)
