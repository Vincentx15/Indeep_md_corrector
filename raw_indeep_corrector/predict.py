import numpy as np
import torch
import pandas as pd
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt

from models import Corrector
from loader import process_df
from learn import clonumpy

# model = torch.load('md_corrector.pt')
# model = torch.load('metad_corrector.pt')

# model = torch.load('md_1_correct_corrector.pt')
# model = torch.load('md_1_direct_corrector.pt')
# model = torch.load('md_2_correct_corrector.pt')
# model = torch.load('md_2_direct_corrector.pt')
model = torch.load('double_corrector.pt')


csv = 'train_indeep/Test/MD_df_rmsd_lig_KEAP1.csv'
# csv = 'train_indeep/Test/MD_df_rmsd_lig_IL2.csv'
# csv = 'train_indeep/Training/MD_df_rmsd_lig_BCL2.csv'
# csv = 'train_indeep/Training/MD_df_rmsd_lig_XIAP1nw9HD.csv'
# csv = 'train_indeep/Test/MetaD_df_rmsd_lig_IL2.csv'
df = pd.read_csv(csv)
data_in, data_out = process_df(df)
out = model(data_in)
np_pred, np_true = clonumpy(out), clonumpy(data_out)
original = data_in[0, 0, :]
giration = data_in[0, 1, :]

print(f'original ligandability : {pearsonr(original, np_true)[0]}')
print(f'original giration : {pearsonr(giration, np_true)[0]}')
print(f'corrected ligandability : {pearsonr(np_pred, np_true)[0]}')


def get_line_arrays(x, y):
    a, b = np.polyfit(x, y, deg=1)
    xseq = np.linspace(x.min(), x.max(), num=100)
    return a, b, xseq


fig, axs = plt.subplots(1, 2)
axs[0].scatter(original, np_true, alpha=0.5)
a, b, xseq = get_line_arrays(original, np_true)
axs[0].plot(xseq, a * xseq + b, color="k", lw=2.5)
axs[0].title.set_text(f'Old : {a:.2f}*x + {b:.2f}')

axs[1].scatter(np_pred, np_true, alpha=0.5)
a, b, xseq = get_line_arrays(np_pred, np_true)
axs[1].plot(xseq, a * xseq + b, color="k", lw=2.5)
axs[1].title.set_text(f'New : {a:.2f}*x + {b:.2f}')
plt.show()
