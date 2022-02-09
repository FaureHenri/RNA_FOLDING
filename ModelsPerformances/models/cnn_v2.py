import numpy as np
from os import listdir
from os.path import join
import random
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from utils import launch_train
from models.cnn import CNNet, criterion_mse


class RNADataset(Dataset):
    def __init__(self, root, rna_seqs, rna_labs):
        self.root = root
        self.rna_seqs = rna_seqs
        self.rna_labs = rna_labs

    def __len__(self):
        return len(self.rna_seqs)

    def __getitem__(self, idx):
        rna_seq = torch.load(join(self.root, 'RNA_SEQS', self.rna_seqs[idx]))
        rna_lab = torch.load(join(self.root, 'RNA_FUNCS', self.rna_labs[idx]))

        return rna_seq, rna_lab


def train_main(config):
    seed = config['seed']
    print("Random Seed: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = config['device']

    if str(device) == 'cuda':
        torch.cuda.manual_seed(seed)

    num_outputs = len(config['out_cols'])

    data = config['data']
    data_in = listdir(join(data, 'RNA_SEQS'))
    data_out = listdir(join(data, 'RNA_FUNCS'))
    sample_seq0 = join(data, 'RNA_SEQS', data_in[0])
    dataix, dataiy, dataiz = torch.load(sample_seq0).size()
    print(f'data_in shape: ({dataix}, {dataiy}, {dataiz})')

    sample_lab0 = join(data, 'RNA_FUNCS', data_out[0])
    dataox = torch.load(sample_lab0).size()[0]
    print(f'data_out shape: ({dataox})')

    X_train, X_test, y_train, y_test = train_test_split(data_in, data_out, train_size=0.75, random_state=seed)
    rna_train = RNADataset(data, X_train, y_train)
    rna_test = RNADataset(data, X_test, y_test)
    print(f'length of train set: {len(rna_train)}\nlength of validation set: {len(rna_test)}')

    train_loader = DataLoader(rna_train, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    test_loader = DataLoader(rna_test, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])

    fccnet = CNNet(img_ch=[17, 32, 64, 128], output_ch=len(config['out_cols']), bloc_wd=dataiy)
    fccnet.to(device)

    optimizer = torch.optim.Adam(fccnet.parameters(), lr=config['learning_rate'], eps=config['epsilon'],
                                 weight_decay=config['weight_decay'])

    # Mean Squared Error
    criterion = criterion_mse

    r2_cnn = launch_train(device=device, num_epochs=config['epochs'], model=fccnet, trainloader=train_loader,
                          testloader=test_loader, loss_fn=criterion, opt=optimizer, out_features=num_outputs,
                          patience_init=10, log_epoch=5)

    return r2_cnn


if __name__ == "__main__":
    hyperparameter_defaults = dict(
        device='cpu',
        data='C:\\Users\\Henri\\PycharmProjects\\RNA_FOLDING\\00_data\\ContactRNA',
        in_cols=['seq_SwitchON_GFP'],
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
        batch_size=16,
        num_workers=0,
        seed=123
    )

    r2_cnn = train_main(hyperparameter_defaults)
    print("R-squared:", r2_cnn)
