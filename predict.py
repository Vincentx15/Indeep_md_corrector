import torch
import pandas as pd
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt


from models import Corrector
from loader import process_df
from learn import clonumpy

model = torch.load('md_corrector.pt')
# csv = 'train_indeep/Test/MD_df_rmsd_lig_KEAP1.csv'
csv = 'train_indeep/Test/MD_df_rmsd_lig_IL2.csv'
df = pd.read_csv(csv)
data_in, data_out = process_df(df)
out = model(data_in)
np_pred, np_true = clonumpy(out), clonumpy(data_out)
print(pearsonr(np_pred, np_true))

original = data_in[0,0,:]
# plt.scatter(np_pred, np_true, alpha=0.5, label='new')
plt.scatter(original, np_true, alpha=0.5, label='old')
plt.legend()
plt.show()
