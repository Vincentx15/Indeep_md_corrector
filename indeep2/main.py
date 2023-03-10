import sys, os

import time
import torch
from torch.utils.data import Subset, DataLoader

import pymol
import pymol2

pymol.invocation.parse_args(['pymol', '-q'])  # optional, for quiet flag
pymol2.SingletonPyMOL().start()

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..'))

from indeep2.loader import RMSDDataset
from indeep2.model import RMSDModel

gpu_number = 1
device = f'cuda:{gpu_number}' if torch.cuda.is_available() else 'cpu'
model = RMSDModel().to(device)
optimizer = torch.optim.Adam(model.parameters())
mse_fn = torch.nn.MSELoss()

dataset = RMSDDataset("../data/data/df_rmsd_train.csv", spacing=0.7)
loader = DataLoader(dataset=dataset, shuffle=True, num_workers=os.cpu_count() - 1, batch_size=30)
# loader = DataLoader(dataset=dataset, shuffle=True, num_workers=0, batch_size=5)
time_init = time.time()
for epoch in range(10):
    for step, (grids, rmsds) in enumerate(loader):
        grids = grids.to(device)
        rmsds = rmsds.to(device)[:, None]

        model.zero_grad()
        out = model(grids)
        loss = mse_fn(out, rmsds)
        loss.backward()
        optimizer.step()

        if not step % 20:
            error_norm = torch.sqrt(loss).item()
            rmsds_std = torch.std(rmsds).item()
            print(
                f"Epoch : {epoch} ; step : {step} ; loss : {loss.item():.5f} ; error norm : {error_norm:.5f} ;"
                f" relative : {error_norm / rmsds_std:.5f} ; time : {time.time() - time_init:.1f}")
