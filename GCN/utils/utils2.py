import forgi
import sys
import pandas as pd


def element_size(el, feats):
    if len(feats) == 0:
        return 0
    elif el in 'fthms':
        return feats[1] - feats[0] + 1
    elif el == 'i':
        return sum(map(lambda d: d[1] - d[0] + 1, zip(feats[::2], feats[1::2])))
    else:
        print('Unrecognized element ', el)
        sys.exit(1)


def get_graph_structure(df, file):
    elements_map = {'f': [1, 0, 0, 0, 0, 0],  # dangling start
                    't': [0, 1, 0, 0, 0, 0],  # dangling end
                    'h': [0, 0, 1, 0, 0, 0],  # hairpin loop
                    'i': [0, 0, 0, 1, 0, 0],  # internal loop
                    'm': [0, 0, 0, 0, 1, 0],  # multi-loop
                    's': [0, 0, 0, 0, 0, 1]   # stem
                    }
    f = open(file, 'r')
    seq_id = f.readlines()[0][1:-1]
    off_rate = df.loc[df['sequence_id'] == seq_id, 'OFF'].values[0]
    on_rate = df.loc[df['sequence_id'] == seq_id, 'ON'].values[0]

    cg = forgi.load_rna(file, allow_many=False)
    elements = cg.defines
    edges = cg.edges
    nodes = list(edges.keys())
    nodes_feats = []  # 5 features: one hot encoded element (1x5) * num nucleotides
    edges_feats = [[], []]
    num_nodes = len(nodes)
    nodes_mapping = {el: k for k, el in enumerate(nodes)}
    # print(nodes_mapping)
    for n in nodes:
        el, id = list(n)
        nodes_feats.append(list(map(lambda ft: ft * element_size(el, elements[n]), elements_map[el])))
        neighbours = list(edges[n])
        for ng in neighbours:
            edges_feats[0].append(nodes_mapping[n])
            edges_feats[1].append(nodes_mapping[ng])

    return nodes_feats, edges_feats, num_nodes, off_rate, on_rate


if __name__ == '__main__':
    print('Building graph ...')
    df = pd.read_csv('/rds/general/user/hf721/home/RNA_Kinetics/data/Toehold_Dataset_Final_2019-10-23.csv')
    nodes, edges, num_nodes, off_val, on_val = get_graph_structure(df, 'rna_struc.txt')
    print(nodes)
    print(edges[0])
    print(edges[1])
    print(num_nodes)
    print(off_val, on_val)
