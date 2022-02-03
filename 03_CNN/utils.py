import numpy as np
import math
import yaml


def pairs2map(pairs):
    seq_len = len(pairs)
    contact = np.zeros([seq_len, seq_len])
    for pair in pairs:
        contact[pair[0], pair[1]] = 1
    return contact


def gaussian(x):
    return math.exp(-0.5*(x*x))


def paired(x, y):
    if x == 'A' and y == 'U':
        return 2
    elif x == 'G' and y == 'C':
        return 3
    elif x == 'G' and y == 'U':
        return 0.8
    elif x == 'U' and y == 'A':
        return 2
    elif x == 'C' and y == 'G':
        return 3
    elif x == 'U' and y == 'G':
        return 0.8
    else:
        return 0


def creatmat(data):
    mat = np.zeros([len(data), len(data)])
    for i in range(len(data)):
        for j in range(len(data)):
            coefficient = 0
            for add in range(30):
                if i - add >= 0 and j + add < len(data):
                    score = paired(data[i - add], data[j + add])
                    if score == 0:
                        break
                    else:
                        coefficient = coefficient + score * gaussian(add)
                else:
                    break
            if coefficient > 0:
                for add in range(1, 30):
                    if i + add < len(data) and j - add >= 0:
                        score = paired(data[i + add], data[j - add])
                        if score == 0:
                            break
                        else:
                            coefficient = coefficient + score * gaussian(add)
                    else:
                        break
            mat[[i], [j]] = coefficient
    return mat


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)


if __name__ == '__main__':
    rna_1d = 'AUGCUC'
    rna_2d = pairs2map(rna_1d)
    print(rna_2d)
