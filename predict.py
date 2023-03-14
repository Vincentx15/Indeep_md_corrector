#!/usr/bin/env python3
# -*- coding: UTF8 -*-

import os
import sys

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

from loader import GridComputer, selection_to_split_coords
from model import RMSDModel
import utils


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
        print(cmd.get_object_list())
        print('Number of atoms in prot', cmd.select('ref_pdb'))
        print('Number of atoms in traj', cmd.select('traj'))
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
                 max_frames=None):
    """
    Inference for validation on the small traj files.
    Assumption is clean PDB with just one chain.
    Give me the MD PDB, traj and selection for this system and I give you a prediction.
    """

    gpu_number = 0
    device = f'cuda:{gpu_number}' if torch.cuda.is_available() else 'cpu'

    if outfilename is None:
        outfilename = f'pred_rmsd.txt'
    torch_dataset = MDDataset(pdbfilename=pdbfilename,
                              trajfilename=trajfilename,
                              selection=selection,
                              spacing=spacing,
                              size=size,
                              max_frames=max_frames)
    # torch_loader = DataLoader(dataset=torch_dataset, num_workers=os.cpu_count() - 1)
    torch_loader = DataLoader(dataset=torch_dataset, num_workers=0)

    predictions = list()
    with open(outfilename, 'w') as outfile:
        outfile.write(f'''# Topology file: {pdbfilename}
# Trajectory file: {trajfilename}
# Number of frames: {len(torch_loader)}
# Box selection: {torch_dataset.box}\n''')
        for item, grid in torch_loader:
            # Get the coords of this specific frame from the traj cmd object.
            # Then from this object find the right box based on the box selection
            item = int(item)
            score = predict_frame(grid=grid, model=model, device=device)
            predictions.append(score)
            outfile.write(f"{item},{score}\n")
            if not item % 250:
                print("Done", item, 'score : ', score)
    return predictions


def evaluate_one(model, directory="data/md/XIAP1nw9HD/", max_frames=None):
    pdbfilename = os.path.join(directory, "step1_pdbreader_HIS.pdb")
    trajfilename = os.path.join(directory, "traj_comp_pbc.xtc")
    selection_filename = os.path.join(directory, "resis_ASA_thr_20.0.txt")
    with open(selection_filename, 'r') as f:
        sel = f.readline()
    predictions = predict_traj(model=model,
                               pdbfilename=pdbfilename,
                               trajfilename=trajfilename,
                               selection=sel,
                               max_frames=max_frames)
    rmsd_gt_csv = pd.read_csv(os.path.join(directory, "rmsd-min_traj_PLs.csv"))
    ground_truth = rmsd_gt_csv['RMSD'].values[:max_frames]
    correlation = scipy.stats.linregress(ground_truth, predictions)
    print(correlation)
    return correlation


def evaluate_all(model, parent_directory="data/md/", max_frames=None):
    all_res = dict()
    for system in os.listdir(parent_directory):
        system_directory = os.path.join(parent_directory, system)
        correlation = evaluate_one(model=model, directory=system_directory, max_frames=max_frames)
        rvalue = correlation.rvalue
        all_res[system] = rvalue
    return all_res


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='')
    args = parser.parse_args()

    model = RMSDModel()
    model_path = 'first_model.pth'
    model.load_state_dict(torch.load(model_path))

    # import torch.nn as nn
    # model.eval()
    # for m in model.modules():
    #     for child in m.children():
    #         if type(child) == nn.BatchNorm3d:
    #             child.track_running_stats = False
    #             child.running_mean = None
    #             child.running_var = None

    # path_pdb = "data/low_rmsd/data/Pockets/PL_test/P08254/1b8y-A-P08254/1b8y-A-P08254_0001_last.mmtf"
    # path_sel = "data/low_rmsd/Resis/P08254_resis_ASA_thr_20.txt"
    # with open(path_sel, 'r') as f:
    #     sel = f.readline()
    # predict_pdb(model=model, pdbfilename=path_pdb, selection=sel)

    # corr_one = evaluate_one(model, max_frames=50)
    # print(corr_one.rvalue)

    all_res = evaluate_all(model, max_frames=1000)
    print(all_res)
