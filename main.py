import sys, os

import numpy as np
import scipy
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import pymol
import pymol2

pymol.invocation.parse_args(['pymol', '-q'])  # optional, for quiet flag
pymol2.SingletonPyMOL().start()

from loader import RMSDDataset
from model import RMSDModel
from learning_utils import RbfLoss, categorical_loss
from predict import evaluate_all, validate


def train(model, device, optimizer, loss_fn, loader, writer, n_epochs=10, val_loader=None):
    time_init = time.time()

    for epoch in range(n_epochs):
        for step, (names, grids, rmsds) in enumerate(loader):
            grids = grids.to(device)
            rmsds = rmsds.to(device)[:, None]

            model.zero_grad()
            out = model(grids)

            loss = loss_fn(out, rmsds)
            loss.backward()
            optimizer.step()

            if not step % 20:
                step_total = len(loader) * epoch + step
                error_norm = torch.sqrt(loss).item()
                rmsds_std = torch.std(rmsds).item()
                print(
                    f"Epoch : {epoch} ; step : {step} ; loss : {loss.item():.5f} ; error norm : {error_norm:.5f} ;"
                    f" relative : {error_norm / rmsds_std:.5f} ; time : {time.time() - time_init:.1f}")
                writer.add_scalar('train_loss', error_norm, step_total)
        if val_loader is not None:
            print("Validation")
            validate(model=model, device=device, loader=val_loader, writer=writer, epoch=epoch)

        if not epoch % 50:
            # if epoch > 10 and not epoch % 50:
            all_res = evaluate_all(model)
            mean_md_corr = np.mean([v for v in all_res.values()])
            writer.add_scalar('MD_validation', mean_md_corr, epoch)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-m", "--model_name", default='long_train')
    parser.add_argument("--nw", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    model_name = args.model_name
    data_root = "data/low_rmsd"

    # Setup learning
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    writer = SummaryWriter(log_dir=f"logs/{model_name}")
    model_path = os.path.join("saved_models", f'{model_name}.pth')
    gpu_number = args.gpu
    device = f'cuda:{gpu_number}' if torch.cuda.is_available() else 'cpu'

    # Learning hyperparameters
    n_epochs = 500
    loss_fn = torch.nn.MSELoss()
    model = RMSDModel().to(device)
    optimizer = torch.optim.Adam(model.parameters())

    # Setup data
    spacing = 0.65
    grid_size = 32
    batch_size = 32
    num_workers = max(os.cpu_count() - 10, 4) if args.nw is None else args.nw

    train_dataset_1 = RMSDDataset(data_root=data_root, csv_to_read="df_rmsd_train.csv",
                                  spacing=spacing, grid_size=grid_size, get_pl_instead=0.1)
    train_dataset_2 = RMSDDataset(data_root="data/high_rmsd", csv_to_read="df_rmsd_train.csv",
                                  spacing=spacing,grid_size=grid_size, get_pl_instead=0.1)
    train_dataset_3 = RMSDDataset(data_root="data/double_rmsd", csv_to_read="df_rmsd_train.csv",
                                  spacing=spacing,grid_size=grid_size, get_pl_instead=0.1)
    train_dataset = torch.utils.data.ConcatDataset([train_dataset_1, train_dataset_2, train_dataset_3])
    val_dataset = RMSDDataset(data_root=data_root, csv_to_read="df_rmsd_validation.csv", rotate=False,
                              spacing=spacing, grid_size=grid_size,get_pl_instead=0.)
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, num_workers=num_workers, batch_size=batch_size)
    val_loader = DataLoader(dataset=val_dataset, num_workers=num_workers, batch_size=batch_size)

    # Train
    train(model=model, device=device, loss_fn=loss_fn, loader=train_loader,
          optimizer=optimizer, writer=writer, n_epochs=n_epochs, val_loader=val_loader)
    model.cpu()
    torch.save(model.state_dict(), model_path)

    # Validate with and without PL
    model = RMSDModel()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    print("Validation without PL")
    _ = validate(model=model, device=device, loader=val_loader)
    print("Validation with PL")
    pl_dataset = RMSDDataset(data_root="data/low_rmsd",
                             csv_to_read="df_pl_validation.csv",
                             spacing=spacing,
                             grid_size=grid_size,
                             rotate=False,
                             get_pl_instead=0)
    val_dataset_pl = torch.utils.data.ConcatDataset([val_dataset, pl_dataset])
    val_loader_pl = DataLoader(dataset=val_dataset_pl, num_workers=num_workers, batch_size=batch_size)
    _ = validate(model=model, device=device, loader=val_loader_pl)

    print("Validation without PL bs 1")
    batch_size = 1
    val_loader = DataLoader(dataset=val_dataset, num_workers=num_workers, batch_size=batch_size)
    _ = validate(model=model, device=device, loader=val_loader)
    print("Validation with PL bs 1")
    val_dataset_pl = torch.utils.data.ConcatDataset([val_dataset, pl_dataset])
    val_loader_pl = DataLoader(dataset=val_dataset_pl, num_workers=num_workers, batch_size=batch_size)
    _ = validate(model=model, device=device, loader=val_loader_pl)

    evaluate_all(model, batch_size=1, save_name=model_name)
