import numpy as np
from itertools import product
from dotmap import DotMap
from Utils.models import *
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.nn import functional as F
from sklearn.metrics import r2_score, mean_squared_error as mse, mean_absolute_error as mae
from Utils.helpers import set_seed
from Utils.data_utils import load_sequences_and_targets
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split


class RNADataset(Dataset):
    def __init__(self, rna_seqs, rna_labs):
        self.rna_seqs = torch.from_numpy(rna_seqs)
        self.rna_labs = torch.from_numpy(rna_labs)

    def __len__(self):
        return len(self.rna_seqs)

    def __getitem__(self, idx):
        rna_seq = self.rna_seqs[idx]
        rna_seq = rna_seq.reshape(-1, 4)
        rna_len = rna_seq.shape[0]

        data_fcn = np.zeros((16, rna_len, rna_len))

        perm = list(product(np.arange(4), np.arange(4)))

        for n, cord in enumerate(perm):
            i, j = cord
            data_fcn[n] = np.matmul(rna_seq[:, i].reshape(-1, 1), rna_seq[:, j].reshape(1, -1))

        return data_fcn, self.rna_labs[idx]



class LitCNN(pl.LightningModule):
    """
    Pytorch Lightning module to build Neural Networks pipeline and facilitate training
    Access Logs using 'tensorboard --logdir=lightning_logs/' command

    Evaluation Metrics: MSE, MAE & R-Squared
    """
    def __init__(self, batch):
        super().__init__()
        self.batch_size = batch
        self.lit_layers = Seq(CNN(in_ch=16, hidden_ch=(32, 64, 128)),
                              MLP(filters=[32768, 128, 64, 1], dropout=0.4))
        self.best_r2 = -np.inf

    def forward(self, x):
        return self.lit_layers(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x_out = self.forward(x)
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
        x, y = val_batch
        x_out = self.forward(x)
        loss = F.mse_loss(x_out.float(), y.float())
        r2 = r2_score(y.cpu().detach().numpy(), x_out.cpu().detach().numpy())
        mse_s = mse(y.cpu().detach().numpy(), x_out.cpu().detach().numpy())
        mae_s = mae(y.cpu().detach().numpy(), x_out.cpu().detach().numpy())

        self.log('loss/val', loss, batch_size=self.batch_size)
        self.log('r2/val', r2, batch_size=self.batch_size)
        self.log('mse/val', mse_s, batch_size=self.batch_size)
        self.log('mae/val', mae_s, batch_size=self.batch_size)
        pred = x_out.argmax(-1)

        # return x_out, pred, y
        return r2

    def validation_epoch_end(self, outs):
        epoch_r2 = np.mean(outs)
        if epoch_r2 > self.best_r2:
            self.best_r2 = epoch_r2
            self.log('R2', round(float(self.best_r2), 4))


def train_main_cnn(config):
    seed = config.seed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(seed, device)
    data_in, data_out = load_sequences_and_targets(data_path=config.data, in_cols=config.in_cols,
                                                   out_cols=config.out_cols, qc_level=config.qc_level)

    dataix, dataiy = data_in.shape
    print(f'data_in shape: ({dataix}, {dataiy})')

    dataox, dataoy = data_out.shape
    print(f'data_out shape: ({dataox}, {dataoy})')

    scaler = QuantileTransformer()
    data_out = scaler.fit_transform(data_out)

    X_train, X_val, y_train, y_val = train_test_split(data_in, data_out, train_size=0.75, random_state=seed)
    rna_train = RNADataset(X_train.to_numpy(), y_train)
    rna_val = RNADataset(X_val.to_numpy(), y_val)
    print(f'length of train set: {len(rna_train)}\nlength of validation set: {len(rna_val)}')

    train_loader = DataLoader(rna_train, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(rna_val, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    ribonet = LitCNN(batch=config.batch_size)
    early_stop_callback = EarlyStopping(monitor="R2", min_delta=0.00, patience=5, verbose=True, mode="max")
    trainer = pl.Trainer(accelerator='gpu', gpus=1, max_epochs=config.epochs, callbacks=[early_stop_callback])
    trainer.fit(ribonet, train_loader, val_loader)

    return ribonet.best_r2


if __name__ == '__main__':
    hyperparameter_defaults = dict(
        seed=65,
        num_workers=8,
        data='/home/hfaure/RNA_FOLDING/DATA/Toehold_Dataset_Final_2019-10-23.csv',
        in_cols=['seq_SwitchON_GFP'],
        out_cols=['ON'],
        qc_level=1.1,
        scaler_init=True,
        epochs=50,
        batch_size=64
    )

    config = DotMap(hyperparameter_defaults)

    r2 = train_main_cnn(config)
    print('Training finished successfully with R2 score:', r2)
