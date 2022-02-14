import numpy as np
import os
import os.path as osp
import random
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from utils import adjacency_mat, hydrogen_bonds, launch_train


class RIBORNA(Dataset):
    def __init__(self, root, raw_files_data):
        self.root = root
        self.raw_files_data = raw_files_data

    def __len__(self):
        return len(self.raw_files_data)

    def __getitem__(self, idx):
        raw_file = self.raw_files_data[idx]
        raw_file_path = osp.join(self.root, raw_file)
        f = open(raw_file_path)
        lines = f.readlines()

        rna_seq = lines[4]
        rna_ss = lines[5]

        rna_mat = adjacency_mat(hydrogen_bonds(rna_seq, rna_ss), len(rna_seq))

        rna_func = float(lines[7].split(',')[1])

        return rna_mat, rna_func


class RIBONET(nn.Module):
    def __init__(self, img_ch=[1, 16, 32, 64], output_ch=1, bloc_wd=148):
        super(RIBONET, self).__init__()
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
        # for layer in self.layers:
        #     x = layer(x)
        #     print(x.size())
        x = self.layers(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def criterion_mse(predicted, ground_truth):
    loss = nn.L1Loss(reduction='mean')

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

    seed = 123
    train_size = 0.75
    data_path = '/rds/general/user/hf721/ephemeral/riboset'
    raw_files = os.listdir(data_path)
    random.Random(seed).shuffle(raw_files)
    train_len = int(train_size * len(raw_files))
    train_files = raw_files[:train_len]
    test_files = raw_files[train_len:]

    trainset = RIBORNA(config['data'], train_files)
    validset = RIBORNA(config['data'], test_files)

    print(f'length of train set: {len(trainset)}\nlength of validation set: {len(validset)}')

    train_loader = DataLoader(trainset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    test_loader = DataLoader(validset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])

    ribonet = RIBONET(img_ch=[1, 16, 32, 64], output_ch=1, bloc_wd=148)
    ribonet.to(device)

    optimizer = torch.optim.Adam(ribonet.parameters(), lr=config['learning_rate'], eps=config['epsilon'],
                                 weight_decay=config['weight_decay'])

    # Mean Squared Error
    criterion = criterion_mse

    r2_cnn = launch_train(device=device, num_epochs=config['epochs'], model=ribonet, trainloader=train_loader,
                          testloader=test_loader, loss_fn=criterion, opt=optimizer, out_features=1,
                          patience_init=10, log_epoch=5)

    return r2_cnn


if __name__ == "__main__":
    hyperparameter_defaults = dict(
        device='cuda',
        data='/rds/general/user/hf721/ephemeral/riboset',
        epochs=200,
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

    r2_cnn = train_main(hyperparameter_defaults)
    print("R-squared:", r2_cnn)
