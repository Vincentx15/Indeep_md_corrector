import sys, os

import time
import scipy

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


def train(model, device, optimizer, mse_fn, loader):
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


def validate(model, device, mse_fn, loader):
    time_init = time.time()
    predictions, ground_truth = [], []
    with torch.no_grad():
        for step, (grids, rmsds) in enumerate(loader):
            grids = grids.to(device)
            rmsds = rmsds.to(device)[:, None]
            out = model(grids)
            loss = mse_fn(out, rmsds)

            predictions.append(out)
            ground_truth.append(rmsds)
            if not step % 20:
                error_norm = torch.sqrt(loss).item()
                rmsds_std = torch.std(rmsds).item()
                print(f"step : {step} ; loss : {loss.item():.5f} ; error norm : {error_norm:.5f} ;"
                      f" relative : {error_norm / rmsds_std:.5f} ; time : {time.time() - time_init:.1f}")
    ground_truth = torch.squeeze(torch.cat(ground_truth, dim=0)).cpu().numpy()
    predictions = torch.squeeze(torch.cat(predictions, dim=0)).cpu().numpy()
    return ground_truth, predictions


if __name__ == '__main__':
    # from utils import correct_df
    # in_csv = "../InDeep_holo-like_pred_toVincent/data/df_rmsd_validation.csv"
    # in_csv = "../data/data/df_rmsd_test.csv"
    # correct_df(in_csv)
    # sys.exit()

    gpu_number = 0
    device = f'cuda:{gpu_number}' if torch.cuda.is_available() else 'cpu'
    model = RMSDModel().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    mse_fn = torch.nn.MSELoss()

    spacing = 1

    train_dataset = RMSDDataset("../data/data/df_rmsd_train.csv", spacing=spacing)
    val_dataset = RMSDDataset("../data/data/df_rmsd_validation.csv", spacing=spacing)
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, num_workers=os.cpu_count() - 1, batch_size=30)
    val_loader = DataLoader(dataset=val_dataset, num_workers=os.cpu_count() - 1, batch_size=30)

    train(model=model, device=device, mse_fn=mse_fn, loader=train_loader, optimizer=optimizer)
    torch.save(model.state_dict(), 'first_model.pth')
    ground_truth, predicition = validate(model=model, device=device, mse_fn=mse_fn, loader=val_loader)
    correlation = scipy.stats.linregress(ground_truth, predicition)
    print(correlation)
