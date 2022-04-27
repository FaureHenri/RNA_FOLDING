import numpy as np
import pandas as pd


def hydrogen_bonds(sequence, structure):
    """
    For each closing parenthesis and matching opening one, index is stored in the base_pairs list.
    The assigned list is used to keep track of the assigned opening parenthesis
    """
    sequence = list(sequence)
    h_bonds = {'A': 2,  # number of hydrogen bonds
               'U': 2,
               'C': 2,
               'G': 2}

    opened = [idx for idx, i in enumerate(structure) if i == '(']
    closed = [idx for idx, i in enumerate(structure) if i == ')']

    assert len(opened) == len(closed)

    assigned = []
    base_pairs = []

    for close_idx in closed:
        for open_idx in opened:
            if open_idx < close_idx:
                if open_idx not in assigned:
                    candidate = open_idx
            else:
                break
        assigned.append(candidate)
        base_pairs.append([[candidate, close_idx], h_bonds[sequence[candidate]]])
        assigned.append(close_idx)
        base_pairs.append([[close_idx, candidate], h_bonds[sequence[close_idx]]])

    assert len(base_pairs) == 2 * len(opened)

    return base_pairs


def adjacency_mat(pairing, seq_len):
    mat = np.zeros((seq_len, seq_len))

    for i in range(seq_len):
        if i < seq_len - 1:
            mat[i, i + 1] = 1
        if i > 0:
            mat[i, i - 1] = 1

    # Adapt weight for non-canonical base pairing, or regarding the number of hydrogen bonds
    for [i, j], w in pairing:
        mat[i, j] = w  # w if custom weighted bonds else 2 for H-bonds

    return mat


def get_graph_structure(df, file):
    nucleotides_map = {'A': [1, 0, 0, 0],
                       'U': [0, 1, 0, 0],
                       'C': [0, 0, 1, 0],
                       'G': [0, 0, 0, 1]
                       }

    bonds_map = {
        1: [1, 0, 0, 0],  # covalent bond  (bone network)
        2: [0, 1, 0, 0],  # H-bond  (AU / UA)
        3: [0, 0, 1, 0],  # H-bond  (CG / GC)
        4: [0, 0, 0, 1]   # H-bond  (GU / UG)
    }

    bonds_map_v2 = {
        1: 10,  # covalent bond  (bone network) (10 times stronger than H-bonds)
        2: 1,   # H-bond  (A-U)
        3: 1    # H-bond  (C-G)
    }

    f = open(file, 'r')
    lines = f.readlines()
    seq_id = lines[0][1:-1]
    rna_seq = lines[1][:-1]
    rna_ss = lines[2]
    adj_mat = adjacency_mat(hydrogen_bonds(rna_seq, rna_ss), len(rna_seq))
    off_rate = df.loc[df['sequence_id'] == seq_id, 'OFF'].values[0]
    on_rate = df.loc[df['sequence_id'] == seq_id, 'ON'].values[0]

    rate = None
    if len(rna_seq) == 148:
        rate = on_rate
    elif len(rna_seq) == 118:
        rate = off_rate
    else:
        print('switch len is ', len(seq_id))

    nodes = list(rna_seq)
    nodes_feats = []  # 1 feature: [one hot encoded element]
    edges_feats = [[], []]
    edges_weight = []
    for i, n in enumerate(nodes):
        nodes_feats.append(nucleotides_map[n])
        # neighbours = np.where(adj_mat[i, :] == 2)[0].tolist()  # Only with value 2 for H-bonds and 1 for Covalent bonds
        neighbours = np.nonzero(adj_mat[i, :])[0].tolist()  # for both covalent and H-bonds
        for ng in neighbours:
            bond_type_f = adj_mat[i, ng]
            edges_feats[0].append(i)
            edges_feats[1].append(ng)
            edges_weight.append(bonds_map_v2[bond_type_f])  # bonds_map[bond_type] for one hot encoded edge attributes

            # bond_type_b = adj_mat[ng, i]  # backward
            # edges_feats[0].append(ng)
            # edges_feats[1].append(i)
            # edges_weight.append(bonds_map_v2[bond_type_b])

    if len(rna_seq) == 118:
        nodes_feats += [[0, 0, 0, 0]]*(148 - 118)

    assert len(nodes_feats) == 148

    return nodes_feats, edges_feats, edges_weight, rate


if __name__ == '__main__':
    print('Building graph ...')
    raw_df = pd.read_csv('/rds/general/user/hf721/home/RNA_FOLDING/DATA/Toehold_Dataset_Final_2019-10-23.csv')
    nodes, edges, edges_weight, rate = get_graph_structure(raw_df, 'human_NR4A3_tile_96.txt')
    print(nodes)
    print(edges[0])
    print(edges[1])
    print('number of edges: ', len(edges[0]))
    print(edges_weight)
    print(rate)
