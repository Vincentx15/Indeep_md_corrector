import sys, os

import time

import numpy as np
import scipy

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import pymol
import pymol2

pymol.invocation.parse_args(['pymol', '-q'])  # optional, for quiet flag
pymol2.SingletonPyMOL().start()

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, ''))

from loader import RMSDDataset
from model import RMSDModel
from learning_utils import RbfLoss, categorical_loss
from predict import evaluate_all


def train(model, device, optimizer, loss_fn, loader, writer, n_epochs=10, val_loader=None):
    time_init = time.time()

    for epoch in range(n_epochs):
        for step, (grids, rmsds) in enumerate(loader):
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
            print("validation")
            ground_truth, prediction = validate(model=model, device=device, loss_fn=loss_fn, loader=val_loader)
            correlation = scipy.stats.linregress(ground_truth, prediction).rvalue
            rmse = np.mean(np.sqrt((ground_truth - prediction) ** 2))
            print(f'Validation mse={rmse}, correlation={correlation}')
            writer.add_scalar('rmse_val', rmse, epoch)
            writer.add_scalar('corr_val', correlation, epoch)

        if not epoch % 50:
            # if epoch > 10 and not epoch % 50:
            all_res = evaluate_all(model)
            mean_md_corr = np.mean([v for v in all_res.values()])
            writer.add_scalar('MD_validation', mean_md_corr, epoch)


def validate(model, device, loss_fn, loader):
    time_init = time.time()
    predictions, ground_truth = [], []
    with torch.no_grad():
        for step, (grids, rmsds) in enumerate(loader):
            grids = grids.to(device)
            rmsds = rmsds.to(device)[:, None]
            out = model(grids)
            loss = loss_fn(out, rmsds)

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
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-m", "--model_name", default='default')
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
    n_epochs = 250
    loss_fn = torch.nn.MSELoss()
    model = RMSDModel().to(device)
    optimizer = torch.optim.Adam(model.parameters())

    # Setup data
    spacing = 0.63
    batch_size = 32
    num_workers = max(os.cpu_count() - 10, 4) if args.nw is None else args.nw

    train_dataset_1 = RMSDDataset(data_root=data_root, csv_to_read="df_rmsd_train.csv",
                                  spacing=spacing, get_pl_instead=0.1)
    train_dataset_2 = RMSDDataset(data_root="data/high_rmsd", csv_to_read="df_rmsd_train.csv",
                                  spacing=spacing, get_pl_instead=0.1)
    train_dataset_3 = RMSDDataset(data_root="data/double_rmsd", csv_to_read="df_rmsd_train.csv",
                                  spacing=spacing, get_pl_instead=0.1)
    train_dataset = torch.utils.data.ConcatDataset([train_dataset_1, train_dataset_2, train_dataset_3])
    val_dataset = RMSDDataset(data_root=data_root, csv_to_read="df_rmsd_validation.csv",
                              spacing=spacing, get_pl_instead=0.)
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, num_workers=num_workers, batch_size=batch_size)
    val_loader = DataLoader(dataset=val_dataset, num_workers=num_workers, batch_size=batch_size)

    # Train
    train(model=model, device=device, loss_fn=loss_fn, loader=train_loader,
          optimizer=optimizer, writer=writer, n_epochs=n_epochs, val_loader=val_loader)
    model.cpu()
    torch.save(model.state_dict(), model_path)

    # Validate
    model = RMSDModel()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    ground_truth, prediction = validate(model=model, device=device, loss_fn=loss_fn, loader=val_loader)
    correlation = scipy.stats.linregress(ground_truth, prediction)
    print(correlation)

    evaluate_all(model, batch_size=1, save_name=model_name)
