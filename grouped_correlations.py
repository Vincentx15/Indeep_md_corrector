import os

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch
from torch.utils.data import DataLoader

from model import RMSDModel
from predict import validate
from loader import RMSDDataset


def get_grouped(csv_path):
    df = pd.read_csv(csv_path, index_col=0)
    df['uniprot'] = df['PL'].map(lambda x: str(x).split('-')[-1])
    uniprot_grouped = df.groupby('uniprot').groups
    return uniprot_grouped


parser = argparse.ArgumentParser(description='')
parser.add_argument("-m", "--model_name", default='mixed_4')
parser.add_argument("--save_name", default=None)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--nw", type=int, default=None)
args = parser.parse_args()

torch.manual_seed(0)
np.random.seed(0)
device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

# MODEL
model_name = args.model_name
model_path = os.path.join("saved_models", f'{model_name}.pth')
model = RMSDModel()
model.load_state_dict(torch.load(model_path))
eval_mode = True
if eval_mode:
    batch_size = 32
    model.eval()
else:
    batch_size = 1

# DATA
data_root = "data/fused"
mode = 'validation'
spacing = 1.
grid_size = 32
# num_workers = 0
num_workers = max(os.cpu_count() - 10, 4) if args.nw is None else args.nw

val_dataset = RMSDDataset(data_root=data_root, csv_to_read=f"df_rmsd_{mode}.csv", rotate=False,
                          spacing=spacing, grid_size=grid_size, get_pl_instead=0.)
val_loader = DataLoader(dataset=val_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False)
ground_truth, prediction, correlation, rmse = validate(model, device=device, loader=val_loader, return_pred=True)

print('Total correlation : ', correlation)

# print(correlation)
csv_path = os.path.join(data_root, 'data', f'df_rmsd_{mode}.csv')
plot = False
grouped = get_grouped(csv_path=csv_path)
grouped_result = dict()
if plot:
    fig, global_ax = plt.subplots(1, 1)
    # global_ax.set_xlim(0.5, 3)
    # global_ax.set_ylim(0.5, 3)
    fig.supxlabel('RMSD')
    fig.supylabel('Prediction')
for uniprot, keys in grouped.items():
    keys = keys.values
    keys = keys[keys < len(prediction)]
    group_gt, group_pred = ground_truth[keys], prediction[keys]
    correlation = scipy.stats.linregress(group_gt, group_pred).rvalue
    print(f'Correlation for {uniprot}: {correlation}')
    grouped_result[uniprot] = correlation
    if plot:
        group_gt = group_gt / np.max(group_gt)
        group_pred = group_pred / np.max(group_pred)
        global_ax.scatter(group_gt, group_pred, alpha=0.4)

print(f"Total system split correlation : ", np.mean(grouped_result.values()))

if plot:
    plt.show()
