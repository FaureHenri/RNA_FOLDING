import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mlp import train_main as r2_mlp
from cnn import train_main as r2_cnn


mlp_params = dict(
    in_cols=['switch'],
    out_cols=['ON'],
    qc_level=1.1,
    scaler_init=True,
    epochs=200,
    filters=[128, 64, 32],
    optimizer='adam',
    loss_fn='mse',
    learning_rate=0.001,
    weight_decay=0.000005,
    epsilon=0.00000001,
    dropout=0.3,
    batch_size=64,
    seed=0
)

cnn_params = dict(
    in_cols=['seq_SwitchON_GFP'],
    out_cols=['ON'],
    qc_level=1.1,
    batch_size=64,
    seed=0
)

models = []
r2_scores = []
for s in range(1, 5):
    print('MLP', s)
    mlp_params['seed'] = s
    r2_mlp_score = r2_mlp(mlp_params)
    models.append('MLP')
    r2_scores.append(r2_mlp_score)

    print('CNN', s)
    cnn_params['seed'] = s
    r2_cnn_score = r2_cnn(cnn_params)
    models.append('CNN')
    r2_scores.append(r2_cnn_score)

performance = pd.DataFrame({"Models": models, "R-Squared": r2_scores})
boxplot = sns.catplot(x="Models", y="R-Squared", data=performance, kind='violin')
plt.savefig('rna_models.png')
performance.to_csv('rna_models.csv')
