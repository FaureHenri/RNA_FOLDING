import os
import os.path as osp
import numpy as np
from random import Random
from dotmap import DotMap
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from Utils.helpers import set_seed
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from sklearn.metrics import r2_score, mean_squared_error as mse, mean_absolute_error as mae
from Utils.models import *


class RNADataset(Dataset):
    def __init__(self, root, rna_seqs):
        self.root = root
        self.rna_seqs = rna_seqs

    def __len__(self):
        return len(self.rna_seqs)

    def __getitem__(self, idx):
        seq_id = osp.splitext(self.rna_seqs[idx])[0] + '.pt'
        seq_data = torch.load(osp.join(self.root, seq_id))
        return seq_data


class LitGCN(pl.LightningModule):
    """
    Pytorch Lightning module to build Neural Networks pipeline and facilitate training
    Access Logs using 'tensorboard --logdir=lightning_logs/' command

    Evaluation Metrics: MSE, MAE & R-Squared
    """
    def __init__(self, batch_size=32, readout='flattening'):
        super().__init__()
        self.batch_size = batch_size
        ch_mlp = None
        if readout == 'flattening':
            ch_mlp = [9472, 128, 64, 1]
        elif readout == 'pooling':
            ch_mlp = [64, 32, 1]

        self.lit_layers = gcnSeq('x, edge_index, batch', [
            (GCN(in_channels=4, readout=readout), 'x, edge_index, batch -> x_gcn'),
            (MLP(filters=ch_mlp, dropout=0.3), 'x_gcn -> x_out')
        ])
        self.best_r2 = -np.inf

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        return self.lit_layers(x, edge_index, batch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0005)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch, train_batch.y
        x_out = self.forward(x)
        x_out = x_out.view(-1)
        loss = F.mse_loss(x_out.float(), y.float())
        r2 = r2_score(y.cpu().detach().numpy(), x_out.cpu().detach().numpy())
        mse_s = mse(y.cpu().detach().numpy(), x_out.cpu().detach().numpy())
        mae_s = mae(y.cpu().detach().numpy(), x_out.cpu().detach().numpy())

        self.log('loss/train', loss, batch_size=self.batch_size)
        self.log('r2/train', r2, batch_size=self.batch_size)
        self.log('mse/train', mse_s, batch_size=self.batch_size)
        self.log('mae/train', mae_s, batch_size=self.batch_size)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch, val_batch.y
        x_out = self.forward(x)
        x_out = x_out.view(-1)
        loss = F.mse_loss(x_out.float(), y.float())
        r2 = r2_score(y.cpu().detach().numpy(), x_out.cpu().detach().numpy())
        mse_s = mse(y.cpu().detach().numpy(), x_out.cpu().detach().numpy())
        mae_s = mae(y.cpu().detach().numpy(), x_out.cpu().detach().numpy())

        self.log('loss/val', loss, batch_size=self.batch_size)
        self.log('r2/val', r2, batch_size=self.batch_size)
        self.log('mse/val', mse_s, batch_size=self.batch_size)
        self.log('mae/val', mae_s, batch_size=self.batch_size)
        pred = x_out.argmax(-1)
        # x_out, pred, y
        return r2

    def validation_epoch_end(self, outs):
        epoch_r2 = np.mean(outs)
        if epoch_r2 > self.best_r2:
            self.best_r2 = epoch_r2
            self.log('R2', round(float(self.best_r2), 4))


def train_main_gcn(params):
    seed = int(params.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed, device)

    train_size = 0.75
    raw_files = os.listdir(params.root)
    Random(seed).shuffle(raw_files)
    train_len = int(train_size * len(raw_files))
    train_files = raw_files[:train_len]
    test_files = raw_files[train_len:]

    print('Length of train set: ', len(train_files))
    print('Length of validation set: ', len(test_files))

    trainset = RNADataset(params.root, train_files)
    validset = RNADataset(params.root, test_files)

    train_loader = DataLoader(trainset, batch_size=params.batch_size, num_workers=params.num_workers)
    valid_loader = DataLoader(validset, batch_size=params.batch_size, num_workers=params.num_workers)
    ribonet = LitGCN(batch_size=params.batch_size, readout=params.readout)

    early_stop_callback = EarlyStopping(monitor="R2", min_delta=0.00, patience=5, verbose=True, mode="max")
    trainer = pl.Trainer(accelerator='gpu', gpus=1, max_epochs=params.epochs, callbacks=[early_stop_callback])
    trainer.fit(ribonet, train_loader, valid_loader)

    return ribonet.best_r2


if __name__ == '__main__':
    hyperparameter_defaults = dict(
        root='/media/sdd/hfaure/riboset/RiboCov',
        seed=65,
        num_workers=4,
        batch_size=64,
        epochs=50,
        readout='pooling'
    )

    config = DotMap(hyperparameter_defaults)

    r2 = train_main_gcn(config)
    print('Training finished successfully with R2 score:', r2)
