from dotmap import DotMap
import pandas as pd
import os.path as osp
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from EXP.gcn_riboNuc import train_main_gcn as r2_gcn
import sys


def compare_main(params, resume=False):
    root = params.root
    regressors = [r2_gcn, r2_gcn, r2_gcn]
    reg_names = ['GCN_cov', 'GCN_hyd', 'GCN_full']
    ribo_set = ['RiboCov', 'RiboHyd', 'RiboNuc']

    device = torch.device("cuda" if torch.cuda.is_available() else sys.exit(1))
    print('device used:', device)

    models = []
    r2_scores = []
    seeds = []
    for s in range(1, 11):
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
        old_perf = pd.read_csv('results/rna_gcns.csv')
        performance = pd.concat([old_perf, performance], axis=0)
        performance.reset_index(inplace=True, drop=True)

    performance.to_csv('rna_gcns.csv', index=False)
    plot_perfs(performance)


def plot_perfs(tab_perf):
    tab.rename(columns={"Models": "GCNs"}, inplace=True)
    num_c = tab_perf['GCNs'].nunique()
    pal = sns.cubehelix_palette(start=2, rot=0, dark=0.15, light=.85, n_colors=num_c)
    sns.catplot(x="GCNs", y="R-Squared", data=tab_perf, kind='violin', palette=pal)
    # plt.ylim([0.0, 0.75])
    plt.savefig('rna_gcns.png')



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
        batch_size=64
    )

    config = DotMap(hp)
    # compare_main(config, resume=False)

    tab = pd.read_csv('EXP/rna_gcns.csv')
    tab['Models'] = tab['Models'].str.replace('GCN_hyd', 'Hydrogen')
    tab['Models'] = tab['Models'].str.replace('GCN_cov', 'Covalent')
    tab['Models'] = tab['Models'].str.replace('GCN_full', 'Covalent + Hydrogen')
    plot_perfs(tab)

    print('Performance comparison done')
