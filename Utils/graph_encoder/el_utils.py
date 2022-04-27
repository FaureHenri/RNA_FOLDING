import forgi
import pandas as pd
from GCN.utils.utils1 import element_size


def nucleotides_counter(seq, pos):
    sub_seq = ''
    for k, p in enumerate(pos[::2]):
        sub_seq += seq[p:pos[2*k + 1]]

    nucleo_feats = [sub_seq.count('A'), sub_seq.count('U'), sub_seq.count('C'), sub_seq.count('G')]

    return nucleo_feats


def get_graph_structure(df, file):
    elements_map = {'f': [1, 0, 0, 0, 0, 0],  # dangling start
                    't': [0, 1, 0, 0, 0, 0],  # dangling end
                    'h': [0, 0, 1, 0, 0, 0],  # hairpin loop
                    'i': [0, 0, 0, 1, 0, 0],  # internal loop
                    'm': [0, 0, 0, 0, 1, 0],  # multi-loop
                    's': [0, 0, 0, 0, 0, 1]   # stem
                    }
    f = open(file, 'r')
    lines = f.readlines()
    seq_id = lines[0][1:-1]
    rna_seq = lines[1][:-1]
    off_rate = df.loc[df['sequence_id'] == seq_id, 'OFF'].values[0]
    on_rate = df.loc[df['sequence_id'] == seq_id, 'ON'].values[0]

    rate = None
    if len(rna_seq) == 148:
        rate = on_rate
    elif len(rna_seq) == 118:
        rate = off_rate
    else:
        print('switch len is ', len(seq_id))

    cg = forgi.load_rna(file, allow_many=False)
    elements = cg.defines
    edges = cg.edges
    nodes = list(edges.keys())
    nodes_feats = []  # 10 features: one hot encoded element (1x6) * num nucleotides + num of each nucleotide (1x4)
    edges_feats = [[], []]
    nodes_mapping = {el: k for k, el in enumerate(nodes)}
    # print(nodes_mapping)
    for n in nodes:
        el, id = list(n)
        nodes_feats.append(list(map(lambda ft: ft * element_size(el, elements[n]), elements_map[el])) +
                           nucleotides_counter(rna_seq, elements[n]))
        neighbours = list(edges[n])
        for ng in neighbours:
            edges_feats[0].append(nodes_mapping[n])
            edges_feats[1].append(nodes_mapping[ng])

    return nodes_feats, edges_feats, rate


if __name__ == '__main__':
    print('Building graph ...')
    df = pd.read_csv('/rds/general/user/hf721/home/RNA_FOLDING/DATA/Toehold_Dataset_Final_2019-10-23.csv')
    nodes, edges, _, rate = get_graph_structure(df, 'human_NR4A3_tile_96.txt')
    print(nodes)
    print(edges[0])
    print(edges[1])
    print(rate)
