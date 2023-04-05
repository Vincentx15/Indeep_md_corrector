import os
import sys

import matplotlib.pyplot as plt
import torch
import numpy as np
from pymol import cmd
import psico.fullinit
import psico.helping
import scipy.ndimage
import scipy.spatial.distance
import pandas as pd

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from loader import GridComputer, selection_to_split_coords, RMSDDataset
from model import RMSDModel
import utils
import time


def predict(model, device, loader):
    predictions, ground_truth = [], []
    with torch.no_grad():
        for step, (grids, rmsds) in enumerate(loader):
            grids = grids.to(device)
            rmsds = rmsds.to(device)[:, None]
            out = model(grids)
            predictions.append(out)
            ground_truth.append(rmsds)
    ground_truth = torch.squeeze(torch.cat(ground_truth, dim=0)).cpu().numpy()
    predictions = torch.squeeze(torch.cat(predictions, dim=0)).cpu().numpy()
    return ground_truth, predictions


def build_pl_csv(data_root="data/low_rmsd",
                 csv_to_read="df_rmsd_train.csv"):
    """
    Build a csv in a similar format that the one for rosetta's files, for the PL files
    One needs to do it in a non redundant way
    """
    csv_file = os.path.join(data_root, "data/", csv_to_read)
    csv_to_dump = os.path.join(data_root, "data/", csv_to_read.replace('rmsd', 'pl'))

    df = pd.read_csv(csv_file)[['Path_PDB_Ros', 'Path_resis', 'RMSD']]
    existing_rows = set()
    new_df = pd.DataFrame(columns=df.columns)
    for i, row in df.iterrows():
        path_pdb, path_resi, rmsd = row.values
        pl_dir, decoy_file = os.path.split(path_pdb)
        pl_file = decoy_file.split('_')[0] + '.pdb'
        if not pl_file in existing_rows:
            existing_rows.add(pl_file)
            pdb_filename = os.path.join(pl_dir, pl_file)
            row = pdb_filename, path_resi, 0
            new_df.loc[len(new_df)] = row
    new_df.to_csv(csv_to_dump)


def validate(model, device, loader, writer=None, epoch=0):
    ground_truth, prediction = predict(model=model, device=device, loader=loader)
    correlation = scipy.stats.linregress(ground_truth, prediction).rvalue
    rmse = np.mean(np.sqrt((ground_truth - prediction) ** 2))
    print(f'Validation mse={rmse}, correlation={correlation}')
    if writer is not None:
        writer.add_scalar('rmse_val', rmse, epoch)
        writer.add_scalar('corr_val', correlation, epoch)
    return correlation, rmse


def predict_frame(model, grid, device=None):
    """
    Predict on a given grid with a given model
    :param model: torch model with 'inference_call' method
    :param grid: numpy grid
    :param device: to force putting the model on a specific device
    :return:
    """
    # Make the prediction
    if isinstance(grid, np.ndarray):
        torch_grid = torch.from_numpy(grid)
        torch_grid = torch_grid.float()[None, ...]
    else:
        torch_grid = grid
    if device is not None:
        model.to(device)
        torch_grid = torch_grid.to(device)
    else:
        if torch.cuda.is_available():
            model.cuda()
            torch_grid = torch_grid.cuda()

    # For c++ usage : change the model name as well as
    # the forward used (comment/uncomment for choosing the right branch)
    # model_dump_name = 'apo_hd.pt'
    # model.eval()
    # traced_script_module = torch.jit.trace(model.forward, torch_grid)
    # traced_script_module.save(model_dump_name)
    # sys.exit()

    out = model(torch_grid)

    # categorical setting :
    if out.shape[1] > 1:
        out = torch.argmax(out, dim=1)
    out = out.detach().squeeze().cpu().numpy()
    return out


def predict_pdb(pdbfilename,
                model,
                selection=None,
                spacing=1.,
                size=20):
    """
    Inference for validation on the small traj files.
    Assumption is clean PDB with just one chain.
    Give me the MD PDB, traj and selection for this system and I give you a prediction.
    """

    cmd.feedback('disable', 'all', 'everything')
    # chain = 'polymer.protein'
    if selection is None:
        box = f'polymer.protein'
    else:
        box = f'polymer.protein and {selection}'

    # Align the trajectory on the selected atoms on the first frame
    cmd.load(pdbfilename, 'prot')
    cmd.remove('hydrogens')
    print(cmd.get_object_list())
    print('Number of atoms in prot', cmd.select('prot'))
    coords = selection_to_split_coords(selection=f'prot and {box}')
    # Use to define center and size, and then a complex
    center = tuple(coords.mean(axis=0)[:3])
    grid = GridComputer(points=coords, center=center, size=size, spacing=spacing).grid
    score = predict_frame(grid=grid, model=model)
    print(score)
    return score


class MDDataset(Dataset):

    def __init__(self,
                 pdbfilename,
                 trajfilename,
                 selection=None,
                 spacing=1.,
                 size=20,
                 max_frames=None
                 ):
        cmd.feedback('disable', 'all', 'everything')
        # chain = 'polymer.protein'
        if selection is None:
            self.box = f'polymer.protein'
        else:
            self.box = f'polymer.protein and {selection}'

        cmd.reinitialize()
        cmd.load(pdbfilename, 'ref_pdb')
        cmd.load(pdbfilename, 'traj')
        cmd.load_traj(trajfilename, 'traj', state=1)
        # Align the coordinates of the trajectory onto the reference pdb
        cmd.align(mobile='traj', target='ref_pdb', mobile_state=1)
        # cmd.delete('ref_pdb')
        # Align the trajectory on the selected atoms on the first frame
        cmd.intra_fit(f'traj and name CA and {self.box}', state=1)
        cmd.remove('hydrogens')
        # print(cmd.get_object_list())
        # print('Number of atoms in prot', cmd.select('ref_pdb'))
        # print('Number of atoms in traj', cmd.select('traj'))
        nstates = cmd.count_states('traj')
        self.frames_to_use = nstates if max_frames is None else min(nstates, max_frames)
        self.size = size
        self.spacing = spacing

    def __len__(self):
        return self.frames_to_use

    def __getitem__(self, item):

        coords = selection_to_split_coords(selection=f'traj and {self.box}',
                                           state=item + 1)
        # Use to define center and size, and then a complex
        center = tuple(coords.mean(axis=0)[:3])
        grid = GridComputer(points=coords, center=center, rotate=False, size=self.size, spacing=self.spacing).grid
        return item, grid


def predict_traj(pdbfilename,
                 trajfilename,
                 model,
                 outfilename=None,
                 selection=None,
                 spacing=1.,
                 size=20,
                 max_frames=None,
                 batch_size=30):
    """
    Inference for validation on the small traj files.
    Assumption is clean PDB with just one chain.
    Give me the MD PDB, traj and selection for this system and I give you a prediction.
    """

    gpu_number = 0
    device = f'cuda:{gpu_number}' if torch.cuda.is_available() else 'cpu'

    if outfilename is None:
        outfilename = f'rmsd.txt'
    torch_dataset = MDDataset(pdbfilename=pdbfilename,
                              trajfilename=trajfilename,
                              selection=selection,
                              spacing=spacing,
                              size=size,
                              max_frames=max_frames)
    # batch_size = 1
    # num_workers=0
    num_workers = max(os.cpu_count() - 10, 1)
    torch_loader = DataLoader(dataset=torch_dataset, num_workers=num_workers, batch_size=batch_size)
    print_every = len(torch_loader) // 10

    predictions = list()
    with open(outfilename, 'w') as outfile:
        outfile.write(f'''# Topology file: {pdbfilename}
# Trajectory file: {trajfilename}
# Number of frames: {len(torch_loader)}
# Box selection: {torch_dataset.box}\n''')
        for i, (items, grids) in enumerate(torch_loader):
            # Get the coords of this specific frame from the traj cmd object.
            # Then from this object find the right box based on the box selection
            items = list(items.cpu().numpy())
            scores = predict_frame(grid=grids, model=model, device=device)
            if batch_size == 1:
                scores = scores.reshape((1,))
            predictions.append(scores)
            for item, score in zip(items, scores):
                outfile.write(f"{item},{score}\n")
            if not i % print_every:
                print("Done", i * batch_size, 'score : ', score)
    predictions = np.concatenate(predictions, axis=0)
    return predictions


def evaluate_one(model, directory="data/md/XIAP1nw9HD/", max_frames=None, batch_size=1):
    pdbfilename = os.path.join(directory, "step1_pdbreader_HIS.pdb")
    trajfilename = os.path.join(directory, "traj_comp_pbc.xtc")
    selection_filename = os.path.join(directory, "resis_ASA_thr_20.0.txt")
    with open(selection_filename, 'r') as f:
        sel = f.readline()
    predictions = predict_traj(model=model,
                               pdbfilename=pdbfilename,
                               trajfilename=trajfilename,
                               selection=sel,
                               max_frames=max_frames, batch_size=batch_size)
    rmsd_gt_csv = pd.read_csv(os.path.join(directory, "rmsd-min_traj_PLs.csv"))
    ground_truth = rmsd_gt_csv['RMSD'].values[:max_frames]
    return ground_truth, predictions


def evaluate_all(model, parent_directory="data/md/", max_frames=None, save_name=None, batch_size=1):
    if isinstance(model, str):
        model_path = os.path.join("saved_models", f'{model}.pth')
        model = RMSDModel()
        model.load_state_dict(torch.load(model_path))
    all_res = dict()
    for system in sorted(os.listdir(parent_directory)):
        system_directory = os.path.join(parent_directory, system)
        ground_truth, predictions = evaluate_one(model=model, directory=system_directory, max_frames=max_frames,
                                                 batch_size=batch_size)
        correlation = scipy.stats.linregress(ground_truth, predictions)
        print(system, correlation)
        rvalue = correlation.rvalue
        all_res[system] = rvalue
        if save_name is not None:
            dir_path = os.path.join("dumps", save_name)
            os.makedirs(dir_path, exist_ok=True)
            dump_name = os.path.join(dir_path, f'{system}.npz')
            np.savez_compressed(dump_name, ground_truth=ground_truth, predictions=predictions)
    print(all_res)
    for k, v in all_res.items():
        print(v)
    return all_res


def plot_all(save_name='first_model'):
    fig, ax = plt.subplots(3, 3)
    dumps_dir = os.path.join('dumps', save_name)
    for i, system in enumerate(os.listdir(dumps_dir)):
        system_dump = os.path.join(dumps_dir, system)
        archive = np.load(system_dump)
        ground_truth = archive['ground_truth']
        predictions = archive['predictions']

        current_ax = ax[i % 3, i // 3]
        current_ax.scatter(ground_truth, predictions, alpha=0.03)
        current_ax.set_title(system.split('.npz')[0])
    fig.supxlabel('RMSD')
    fig.supylabel('Prediction')
    plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-m", "--model_name")
    args = parser.parse_args()

    model_name = args.model_name

    # build_pl_csv(csv_to_read="df_rmsd_train.csv")
    # build_pl_csv(csv_to_read="df_rmsd_validation.csv")
    # build_pl_csv(csv_to_read="df_rmsd_test.csv")

    # path_pdb = "data/low_rmsd/data/Pockets/PL_test/P08254/1b8y-A-P08254/1b8y-A-P08254_0001_last.mmtf"
    # path_sel = "data/low_rmsd/Resis/P08254_resis_ASA_thr_20.txt"
    # with open(path_sel, 'r') as f:
    #     sel = f.readline()
    # predict_pdb(model=model, pdbfilename=path_pdb, selection=sel)

    # gt, pred = evaluate_one(model, max_frames=50)

    # import time
    #
    # batch_size = 1
    # all_res = evaluate_all(model_name, max_frames=None, save_name=model_name)
    # Unbatched time : 298
    # Batched time : 175
    plot_all(save_name=model_name)
