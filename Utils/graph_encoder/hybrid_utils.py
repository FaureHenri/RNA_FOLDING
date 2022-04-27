import numpy as np
import pandas as pd
from itertools import product
from Utils.graph_encoder.nuc_utils import hydrogen_bonds, adjacency_mat
from Utils.data_utils import rna2onehot


def mat_cnn(rna_seq):
    encoded_rna = rna2onehot(rna_seq)
    encoded_rna = encoded_rna.reshape(-1, 4)
    rna_len = encoded_rna.shape[0]

    mat_conv = np.zeros((16, rna_len, rna_len))

    perm = list(product(np.arange(4), np.arange(4)))

    for n, cord in enumerate(perm):
        i, j = cord
        mat_conv[n] = np.matmul(encoded_rna[:, i].reshape(-1, 1), encoded_rna[:, j].reshape(1, -1))

    return mat_conv


def get_graph_structure(df, file):
    nucleotides_map = {'A': [1, 0, 0, 0],
                       'U': [0, 1, 0, 0],
                       'C': [0, 0, 1, 0],
                       'G': [0, 0, 0, 1]
                       }

    f = open(file, 'r')
    lines = f.readlines()
    seq_id = lines[0][1:-1]
    rna_seq = lines[1][:-1]
    rna_ss = lines[2]
    adj_mat = adjacency_mat(hydrogen_bonds(rna_seq, rna_ss), len(rna_seq))
    rate = df.loc[df['sequence_id'] == seq_id, 'ON'].values[0]
    nodes = list(rna_seq)
    nodes_feats = []  # 1 feature: [one hot encoded element]
    edges_feats = [[], []]
    for i, n in enumerate(nodes):
        nodes_feats.append(nucleotides_map[n])
        neighbours = np.where(adj_mat[i, :] == 1)[0].tolist()  # Only with value 2 for H-bonds
        # neighbours = np.nonzero(adj_mat[i, :])[0].tolist()  # for both covalent and H-bonds
        for ng in neighbours:
            edges_feats[0].append(i)
            edges_feats[1].append(ng)

    if len(rna_seq) == 118:
        nodes_feats += [[0, 0, 0, 0]]*(148 - 118)

    assert len(nodes_feats) == 148
    mat = mat_cnn(rna_seq)

    return nodes_feats, edges_feats, mat, rate


if __name__ == '__main__':
    print('Building graph ...')
    raw_df = pd.read_csv('/rds/general/user/hf721/home/RNA_FOLDING/DATA/Toehold_Dataset_Final_2019-10-23.csv')
    nodes, edges, mat, rate = get_graph_structure(raw_df, 'human_NR4A3_tile_96.txt')
    print(nodes)
    print(edges[0])
    print(edges[1])
    print('number of edges: ', len(edges[0]))
    print(mat)
    print(rate)
