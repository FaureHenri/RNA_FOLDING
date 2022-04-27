from dotmap import DotMap
import pandas as pd
import os.path as osp
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from EXP.mlp_ribo import train_main_mlp as r2_mlp
from EXP.lstm_ribo import train_main_lstm as r2_lstm
from EXP.cnn_ribo import train_main_cnn as r2_cnn
from EXP.gcn_riboNuc import train_main_gcn as r2_nucgcn
from EXP.gcn_riboElNuc import train_main_gcn as r2_elgcn
from EXP.hybrid_ribo import train_main_hybrid as r2_hybrid
import sys


def compare_main(params, resume=False):
    root = params.root
    regressors = [r2_mlp, r2_lstm, r2_cnn, r2_elgcn, r2_nucgcn, r2_hybrid]
    reg_names = ['MLP', 'LSTM', 'CNN', 'Elem-GCN', 'Nucleo-GCN', 'HYBRID']
    ribo_set = [None, None, None, 'RiboElNuc', 'RiboCov', 'RiboHybrid']

    device = torch.device("cuda" if torch.cuda.is_available() else sys.exit(1))
    print('device used:', device)

    models = []
    r2_scores = []
    seeds = []
    for s in range(1, 21):
        for k, reg in enumerate(regressors):
            print('\n', reg_names[k], s)
            params.seed = s
            if ribo_set[k]:
                params.root = osp.join(root, ribo_set[k])
            r2_score = reg(params)
            models.append(reg_names[k])
            r2_scores.append(r2_score)
            seeds.append(s)

    performance = pd.DataFrame({"Models": models, "R-Squared": r2_scores, "Seeds": seeds})

    if resume:
        old_perf = pd.read_csv('results/rna_models.csv')
        performance = pd.concat([old_perf, performance], axis=0)
        performance.reset_index(inplace=True, drop=True)

    performance.to_csv('rna_models.csv', index=False)
    plot_perfs(performance)


def plot_perfs(tab_perf):
    num_c = tab_perf['Models'].nunique()
    pal = sns.cubehelix_palette(start=2, rot=0, dark=0.15, light=.85, n_colors=num_c)
    sns.catplot(x="Models", y="R-Squared", data=tab_perf, kind='violin', palette=pal)
    plt.savefig('rna_models.png')



if __name__ == '__main__':
    hp = dict(
        seed=65,
        num_workers=4,
        data='/home/hfaure/RNA_FOLDING/DATA/Toehold_Dataset_Final_2019-10-23.csv',
        root='/media/sdd/hfaure/riboset',
        in_cols=['seq_SwitchON_GFP'],
        out_cols=['ON'],
        qc_level=1.1,
        scaler_init=True,
        epochs=50,
        batch_size=64,
        readout='flattening'
    )

    config = DotMap(hp)
    # compare_main(config, resume=False)

    tab = pd.read_csv('results/rna_models.csv')
    tab['Models'] = tab['Models'].str.replace('Elem-GCN', 'ElGCN')
    tab['Models'] = tab['Models'].str.replace('Nucleo-GCN', 'NuGCN')
    plot_perfs(tab)

    print('Performance comparison done')
