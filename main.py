import sys, os

import time
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
from learning_utils import RbfLoss


def train(model, device, optimizer, loss_fn, loader, writer, n_epochs=10):
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

    model_name = 'default'
    data_root = "data/low_rmsd"

    # Setup learning
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    writer = SummaryWriter(log_dir=f"logs/{model_name}")
    model_path = os.path.join("saved_models", f'{model_name}.pth')
    gpu_number = 0
    device = f'cuda:{gpu_number}' if torch.cuda.is_available() else 'cpu'

    # Learning hyperparameters
    n_epochs = 10
    loss_fn = RbfLoss(min_value=0, max_value=4, nbins=10).to(device)
    # loss_fn = torch.nn.MSELoss()
    model = RMSDModel().to(device)
    optimizer = torch.optim.Adam(model.parameters())

    # Setup data
    spacing = 1.
    batch_size = 30
    train_dataset = RMSDDataset(data_root=data_root, csv_to_read="df_rmsd_train.csv", spacing=spacing)
    val_dataset = RMSDDataset(data_root=data_root, csv_to_read="df_rmsd_validation.csv", spacing=spacing)
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, num_workers=os.cpu_count() - 1,
                              batch_size=batch_size)
    val_loader = DataLoader(dataset=val_dataset, num_workers=os.cpu_count() - 1, batch_size=batch_size)

    # Train
    train(model=model, device=device, loss_fn=loss_fn, loader=train_loader,
          optimizer=optimizer, writer=writer, n_epochs=n_epochs)
    model.cpu()
    torch.save(model.state_dict(), model_path)

    # Validate
    model = RMSDModel()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    ground_truth, prediction = validate(model=model, device=device, mse_fn=loss_fn, loader=val_loader)
    correlation = scipy.stats.linregress(ground_truth, prediction)
    print(correlation)
