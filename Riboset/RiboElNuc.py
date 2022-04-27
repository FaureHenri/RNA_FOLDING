"""
Build database with RNA encoded as GRAPH
nodes: Elements (126 features) one-hot encoded element + nucleotide
edges: unit between elements
"""

import os
import os.path as osp
import pandas as pd
from tqdm import tqdm
import torch
from torch import Tensor, LongTensor
from torch_geometric.data import Data, Dataset
from Utils.helpers import ifnotmkdir
from Utils.graph_encoder.el_nuc_utils import get_graph_structure


class RIBORNA(Dataset):
    def __init__(self, raw_data, processed_data, df, files):
        self.raw_data = raw_data
        self.df = df
        self.files = files
        super().__init__(processed_data)

    @property
    def raw_dir(self):
        return osp.join(self.raw_data)

    @property
    def raw_file_names(self):
        return self.files

    @property
    def processed_file_names(self):
        return [osp.splitext(f)[0] + '.pt' for f in self.raw_file_names]

    def process(self):
        for file in tqdm(self.files):
            node_features, edge_index, label = get_graph_structure(self.df, osp.join(self.raw_dir, file))
            gph_data = Data(x=Tensor(node_features),
                            edge_index=LongTensor(edge_index),
                            y=Tensor([label]))

            pt_file = osp.splitext(file)[0] + '.pt'
            out_path = osp.join(self.processed_dir, pt_file)
            torch.save(gph_data, out_path)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        seq_id = osp.splitext(self.files[idx])[0] + '.pt'
        seq_data = torch.load(osp.join(self.processed_dir, seq_id))
        return seq_data


if __name__ == "__main__":
    root = '/media/sdd/hfaure/riboset/switchON'
    root_processed = '/media/sdd/hfaure/riboset/RiboElNuc'
    ifnotmkdir(root_processed)
    raw_data = '/home/hfaure/RNA_FOLDING/DATA/Toehold_Dataset_Final_2019-10-23.csv'
    df_raw_data = pd.read_csv(raw_data)
    raw_files = os.listdir(root)
    riboset = RIBORNA(root, root_processed, df_raw_data, raw_files)
