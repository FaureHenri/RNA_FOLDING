import os
import os.path as osp
from random import Random
import pandas as pd
import torch
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Dataset, DataLoader
from utils import get_graph_structure, launch_train


class RIBORNA(Dataset):
    def __init__(self, df, files):
        super().__init__(files)
        self.df = df
        self.files = files

    @property
    def raw_file_names(self):
        return self.files

    @property
    def processed_file_names(self):
        return os.listdir(self.processed_dir)

    def process(self):
        for file in self.raw_paths:
            self._process_one_step(file)

    def _process_one_step(self, txt_file):
        node_features, edge_index, edge_attr, _, label = get_graph_structure(self.df, txt_file)
        gph_data = Data(x=node_features,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        y=label
                        )

        pt_file = osp.splitext(txt_file)[0] + '.pt'
        out_path = osp.join(self.processed_dir, pt_file)
        torch.save(gph_data, out_path)
        return

    def __len__(self):
        return len(self.processed_file_names)

    def __getitem__(self, item):
        seq_id = osp.splitext(self.files[item])[0] + '.pt'
        seq_data = torch.load(osp.join(self.processed_dir, seq_id))
        return seq_data


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


def criterion_mse(predicted, ground_truth):
    loss = nn.L1Loss(reduction='mean')

    return loss(predicted, ground_truth)


def main(params):
    seed = params['seed']
    # device = params['device']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device used:', device)
    train_size = 0.8
    df_raw_data = pd.read_csv(params['raw_data'])
    raw_files = os.listdir(params['gph_data'])
    Random(seed).shuffle(raw_files)
    train_len = int(train_size * len(raw_files))
    train_files = raw_files[:train_len]
    test_files = raw_files[train_len:]

    trainset = RIBORNA(df_raw_data, train_files)
    validset = RIBORNA(df_raw_data, test_files)

    train_loader = DataLoader(trainset, batch_size=params['batch_size'], num_workers=params['num_workers'])
    valid_loader = DataLoader(validset, batch_size=params['batch_size'], num_workers=params['num_workers'])
    gcn = GCN(trainset.num_features, 16, 1)
    gcn.to(device)

    optimizer = torch.optim.Adam(gcn.parameters(), lr=params['learning_rate'], eps=params['epsilon'],
                                 weight_decay=params['weight_decay'])

    # Mean Squared Error
    criterion = criterion_mse

    r2_gcn = launch_train(device=params['device'], num_epochs=params['epochs'], model=gcn, trainloader=train_loader,
                          testloader=valid_loader, loss_fn=criterion, opt=optimizer, out_features=1, patience_init=10,
                          log_epoch=5)

    return r2_gcn


if __name__ == '__main__':
    configs = dict(raw_data='/rds/general/user/hf721/home/RNA_Kinetics/00_data/Toehold_Dataset_Final_2019-10-23.csv',
                   gph_data='/rds/general/user/hf721/ephemeral/riboset/switchON',
                   seed=123,
                   device='cpu',
                   num_workers=4,
                   batch_size=32,
                   epochs=200,
                   learning_rate=0.001,
                   epsilon=1e-8,
                   weight_decay=0.000005
                   )

    r2_score = main(configs)
    print('Training finished with R-squared: ', r2_score)
