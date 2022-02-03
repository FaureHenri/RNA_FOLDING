import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from ModelsPerformances.models.mlp import train_main as r2_mlp
from ModelsPerformances.models.lstm import train_main as r2_lstm
from ModelsPerformances.models.cnn import train_main as r2_cnn
from ModelsPerformances.models.cnn_v2 import train_main as r2_cnnv2
from utils import get_config


def call_config(num):
    config = None
    if num == 1:
        config = get_config('configs/config_mlp.yaml')
    elif num == 2:
        config = get_config('configs/config_lstm.yaml')
    elif num == 3:
        config = get_config('configs/config_cnn.yaml')
    elif num == 4:
        config = get_config('configs/config_cnnv2.yaml')

    return config['parameters']


regressors = [r2_mlp, r2_lstm, r2_cnn, r2_cnnv2]
reg_names = ['MLP', 'LSTM', 'CNN', 'CNN_v2']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device used:', device)

models = []
r2_scores = []
for s in range(1, 3):
    for k, reg in enumerate(regressors):
        print('\n', reg_names[k], s)
        parameters = call_config(k)
        parameters['device'] = device
        parameters['seed'] = s
        parameters['num_workers'] = 4
        r2_score = reg(parameters)
        models.append(reg_names[k])
        r2_scores.append(r2_score)

# old_perf = pd.read_csv('rna_models.csv')
performance = pd.DataFrame({"Models": models, "R-Squared": r2_scores})
# performance = pd.concat([old_perf, performance], axis=0)
# performance.reset_index(inplace=True, drop=True)
boxplot = sns.catplot(x="Models", y="R-Squared", data=performance, kind='violin')
plt.savefig('rna_models.png')

performance.to_csv('rna_models.csv', index=False)
