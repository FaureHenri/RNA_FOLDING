from collections import OrderedDict
import torch
from torch.nn import Sequential as Seq, LSTM, Conv2d, BatchNorm1d, BatchNorm2d, MaxPool2d, ReLU, Dropout, Linear
from torch_geometric.nn import Sequential as gcnSeq, GCNConv, global_mean_pool
from torch_geometric.utils import to_dense_batch


class LSTM_NN(torch.nn.Module):
    def __init__(self, filters=(592, 128, 100, 1), dropout=0.3):
        super(LSTM_NN, self).__init__()
        input_size, hidden_size, hid2, output_size = filters
        self.lstm = LSTM(input_size, hidden_size, num_layers=1, dropout=0.4, bidirectional=True, batch_first=True)
        self.layers = Seq(Flatten(), self.dense_1d(2*hidden_size, hid2, dropout), Linear(hid2, output_size))

    def forward(self, x):
        x = x.float()
        lstm_out, _ = self.lstm(x.view(-1, 1, 592))
        return self.layers(lstm_out)

    @staticmethod
    def dense_1d(in_x, out_x, dropout):
        return Seq(Linear(in_x, out_x), BatchNorm1d(out_x), Dropout(dropout), ReLU(True))


class MLP(torch.nn.Module):
    """
    Multi-Layer Perceptron with modulable number of layers and neurons by layers
    """
    def __init__(self, filters, dropout):
        super().__init__()
        num_layers = len(filters)
        self.mlp_layers = Seq(Flatten())
        for i in range(num_layers):
            if i + 2 < num_layers:
                self.mlp_layers.add_module(name=str('layer' + str(i + 1)), module=self.dense_1d(filters[i], filters[i + 1], dropout))
            elif i + 1 < num_layers:
                self.mlp_layers.add_module(name=str('layer' + str(i + 1)), module=Linear(filters[i], filters[i + 1]))
            else:
                break

    def forward(self, x):
        return self.mlp_layers(x.float())

    @staticmethod
    def dense_1d(in_x, out_x, dropout):
        return Seq(Linear(in_x, out_x), BatchNorm1d(out_x), Dropout(dropout), ReLU(True))


class CNN(torch.nn.Module):
    """
    Convolutional Neural Network with input of shape [148, 148, 7]
    where 148 is the length of the ON state switch nucleotide sequence
    and 7 channels for Hydrogen bonds (AU, UA, GU, UG, CG, GC) and one channel summing 6 other channels
    """
    def __init__(self, in_ch=16, hidden_ch=(32, 64, 128)):
        super().__init__()
        self.cnn_layers = Seq(OrderedDict([
            ('conv1', self.dense_2d(in_ch, hidden_ch[0])),
            ('conv2', self.dense_2d(hidden_ch[0], hidden_ch[1])),
            ('conv3', self.dense_2d(hidden_ch[1], hidden_ch[2])),
            ('flatten', Flatten())
        ]))

    def forward(self, x):
        return self.cnn_layers(x.float())

    @staticmethod
    def dense_2d(x_in, x_out):
        return Seq(Conv2d(x_in, x_out, kernel_size=(5, 5), stride=(1, 1), padding=1, bias=True),
                   BatchNorm2d(x_out),
                   MaxPool2d(kernel_size=2, stride=2),
                   ReLU(inplace=True))


class GCN(torch.nn.Module):
    """
    Graph Convolutional Neural Network for nucleotide-based graphs
    """
    def __init__(self, in_channels=4, readout='flattening'):
        super().__init__()
        mod_readout = None
        if readout == 'flattening':
            mod_readout = (ReadOut(num_nodes=148), 'x3d, batch -> x3r')
        elif readout == 'pooling':
            mod_readout = (global_mean_pool, 'x3d, batch -> x3r')

        self.gcn_layers = gcnSeq('x, edge_index, batch', [
            (GCNConv(in_channels, 16), 'x, edge_index -> x1'),
            (ReLU(), 'x1 -> x1a'),
            (Dropout(p=0.1), "x1a -> x1d"),
            (GCNConv(16, 32), 'x1d, edge_index -> x2'),
            (ReLU(), 'x2 -> x2a'),
            (Dropout(p=0.1), "x2a -> x2d"),
            (GCNConv(32, 64), 'x2d, edge_index -> x3'),
            (ReLU(), 'x3 -> x3a'),
            (Dropout(p=0.1), "x3a -> x3d"),
            mod_readout,
            (Flatten(), 'x3r -> x_out')
        ])

    def forward(self, x, edge_index, batch):
        return self.gcn_layers(x, edge_index, batch)


class GCN2(torch.nn.Module):
    """
    Graph Convolutional Neural Network for element-based graphs
    """
    def __init__(self, in_channels=4, readout='flattening'):
        super().__init__()
        mod_readout = None
        if readout == 'flattening':
            mod_readout = (ReadOut(num_nodes=20), 'x3d, batch -> x3r')
        elif readout == 'pooling':
            mod_readout = (global_mean_pool, 'x3d, batch -> x3r')

        self.gcn2_layers = gcnSeq('x, edge_index, batch', [
            (GCNConv(in_channels, 64), 'x, edge_index -> x1'),
            (ReLU(), 'x1 -> x1a'),
            (Dropout(p=0.1), "x1a -> x1d"),
            (GCNConv(64, 32), 'x1d, edge_index -> x2'),
            (ReLU(), 'x2 -> x2a'),
            (Dropout(p=0.1), "x2a -> x2d"),
            (GCNConv(32, 16), 'x2d, edge_index -> x3'),
            (ReLU(), 'x3 -> x3a'),
            (Dropout(p=0.1), "x3a -> x3d"),
            mod_readout,
            (Flatten(), 'x3r -> x_out')
        ])

    def forward(self, x, edge_index, batch):
        return self.gcn2_layers(x, edge_index, batch)


class CatLayer(torch.nn.Module):
    """
    Concatenation Layer to concatenate CNN output and GCN output
    """
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        return torch.cat([x1, x2], dim=1)


class PrintLayer(torch.nn.Module):
    """
    Additional Layer to print Tensor shape during training
    """
    def __init__(self, msg=None):
        super(PrintLayer, self).__init__()
        self.msg = msg

    def forward(self, x):
        print(f'{self.msg}, tensor size: {x.size()}')
        return x


class ReadOut(torch.nn.Module):
    """
    Flattening module for graph neural networks
    """
    def __init__(self, num_nodes=148):
        super(ReadOut, self).__init__()
        self.num_nodes = num_nodes

    def forward(self, x, batch):
        return to_dense_batch(x, batch, 0.0, self.num_nodes)[0]


class Flatten(torch.nn.Module):
    """
    Flattening Layer
    """
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = torch.flatten(x, 1)
        return x
