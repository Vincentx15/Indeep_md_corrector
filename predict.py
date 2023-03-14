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
    torch_grid = torch.from_numpy(grid)
    torch_grid = torch_grid.float()[None, ...]
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


#
# def predict_pdb(model,
#                 pdbfile,
#                 selection=None,
#                 pdbfile_lig=None,
#                 selection_lig=None,
#                 spacing=1.,
#                 padding=8,
#                 xyz_min=None,
#                 xyz_max=None,
#                 device=None):
#     """
#
#     Makes a model prediction from a pdb.
#     Returns the prediction for the hd branch, the pl branch and the origin of the grid.
#     If there is a ligand, aligns the grid to include it and return a ligand mrc that looks like the prediction
#
#     :param model: A pytorch model
#     :param pdbfile: A pdb to be opened
#     :param selection: A pymol selection to run the prediction on, instead of the whole protein
#     :param pdbfile_lig: A pdb to use as a ligand/target for instance in the HDPL database
#     :param selection_lig: A selection to use for the ligand. Is pdbfile_lig is empty,
#     the ligand selection is done on 'pdbfile'
#     :param hetatm: A flag to indicate that the ligand is not a protein but a small molecule
#     :param spacing: The spacing of the grid
#     :param padding: ... and its padding
#     :param xyz_min: For usage with a fixed box, a tuple of the bottom corner of the box, the origin of an mrc
#     :param xyz_max: For usage with a fixed box, a tuple of the top corner of the box
#     :param enveloppe: A flag to compute enveloppe around the input and zero predictions outside of this enveloppe.
#     :return: out_hd, out_pl : two grids in the squeezed output format [1/6,x,y,z],
#                                 xyz_min, lig_grid if there is a ligand in the same [1/6,x,y,z] format
#     """
#     # We can instantiate the object either with 2 pdbs (Olivier and Karen setting)
#     # or with a single one with two different selections (usual setting)
#     if pdbfile_lig is None and selection_lig is not None:
#         pdbfile_lig = pdbfile
#     density = Density.Coords_channel(pdbfilename_p=pdbfile,
#                                      pdbfilename_l=pdbfile_lig)
#     coords = density.split_coords_by_channels(selection=selection)
#     assert coords.ndim == 2
#     assert coords.shape[-1] == 4
#
#     # If we also load a ligand, we then need to create coords_all to compute a common grid
#     coords_all = coords  # The size of the grid box is defined as the min max of the protein coordinates
#     if xyz_min is None:
#         xyz_min = coords_all[:, :3].min(axis=0)
#     if xyz_max is None:
#         xyz_max = coords_all[:, :3].max(axis=0)
#     grid = Complex.get_grid_channels(coords,
#                                      spacing=spacing,
#                                      padding=padding,
#                                      xyz_min=xyz_min,
#                                      xyz_max=xyz_max,
#                                      is_target=False)
#
#     # Make the prediction
#     score = predict_frame(model=model, grid=grid, device=device)
#     return score
#
#
# def pred_traj(pdbfilename,
#               trajfilename,
#               outfilename=None,
#               box=None,
#               chain=None,
#               hetatm=False,
#               spacing=1.,
#               vol=150,
#               margin=6,
#               ranchor=6,
#               anchor_xyz=None,
#               refpdb=None,
#               refsel=None):
#     """
#     Inference on trajectory file
#
#     Timing on GPU :
#     time spent in the loading step : 0.075
#     time spent predicting :          0.053
#     time spent post-processing :     0.023
#     ckp: checkpoint output dcd every ckp frame
#     """
#
#     cmd.feedback('disable', 'all', 'everything')
#
#     if chain is None:
#         chain = 'polymer.protein'
#
#     coords = build_coord_channel(pdbfilename_p=pdb_filename, selection=sel)
#     # Use to define center and size, and then a complex
#     center = coords.mean(axis=0)[:3]
#     center = tuple(center)
#     # print(coords.max(axis=0) - coords.min(axis=0))
#     size = 20
#     comp = Complex(points=coords, center=center, size=size)
#     grid = comp.grid
#
#
#     # density = Density.Coords_channel(pdbfilename, hetatm=hetatm)
#     cmd.load(pdbfilename, 'traj')
#     if box is None:
#         if anchor_xyz is None:
#             # The maximum box should be the chain
#             box = chain
#         else:
#             myspace = {'resids': []}
#             cmd.pseudoatom(object='PSD', pos=tuple(anchor_xyz))
#             cmd.iterate(f'{chain} within {ranchor} of PSD',
#                         'resids.append(resi)',
#                         space=myspace)
#             cmd.delete('PSD')
#             resids = myspace['resids']
#             box = ' or '.join([f'resid {r}' for r in np.unique(resids)])
#             box = f"({box}) and {chain}"
#     if trajfilename is not None:
#         cmd.load(pdbfilename, 'ref_pdb')
#         cmd.load_traj(trajfilename, 'traj', state=1)
#         # Align the coordinates of the trajectory onto the reference pdb
#         cmd.align('traj', 'ref_pdb', mobile_state=1)
#         cmd.delete('ref_pdb')
#         cmd.intra_fit(
#             f'traj and name CA and {box}', state=1
#         )  # Align the trajectory on the selected atoms on the first frame
#     cmd.remove('hydrogens')
#     print(cmd.get_object_list())
#     print('Number of atoms in prot', cmd.select('prot'))
#     print('Number of atoms in traj', cmd.select('traj'))
#     nstates = cmd.count_states('traj')
#
#     if anchor_xyz is None:
#         if refpdb is not None:
#             cmd.load(refpdb, 'refpdb')
#             cmd.align(f'refpdb and name CA and {box}',
#                       f'traj and name CA and {box}',
#                       target_state=1)
#             if refsel is None:
#                 refsel = 'hetatm'
#             anchor_xyz = cmd.get_coords(f'refpdb and {refsel}').mean(
#                 axis=0)  # COM of the reference anchor
#
#     if outfilename is None:
#         outfilename = f'{"ligandability" if hetatm else "interactibility"}_scores_radius-{margin}.txt'
#
#     with open(outfilename, 'w') as outfile:
#         outfile.write(f'''# Topology file: {pdbfilename}
# # Trajectory file: {trajfilename}
# # Number of frames: {nstates}
# # Prediction model: {"ligandability" if hetatm else "interactibility"}
# # Box selection: {box}
# # Chain: {chain}
# # Volume: {vol}
# # Radius: {margin}
# # Frame {"ligandability" if hetatm else "interactibility"} \
# {"ligandability_old" if hetatm else "interactibility_old"} \
# {"volume"}\n''')
#         for state in range(nstates):
#             # t0 = time.perf_counter()
#             # Get the coords of this specific frame from the traj cmd object.
#             # Then from this object find the right box based on the box selection
#             coords_frame = cmd.get_coords(selection=f'traj and {chain}',
#                                           state=state + 1)
#             coords_box = cmd.get_coords(
#                 selection=f'traj and {box} and {chain}', state=state + 1)
#             xyz_min = coords_box[:, :3].min(axis=0)
#             xyz_max = coords_box[:, :3].max(axis=0)
#
#             # Now we need to use the density object to iterate on the same atoms and use pymol
#             #  utils to split these atoms into our channels (for the ones for which we have a mapping)
#             # Using load coords, and the same chain selection,
#             #  we ensure the coordinates are the ones of the correct frame
#             other_coords = density.split_coords_by_channels(
#                 selection=chain, ligand=False, load_coords=coords_frame)
#
#             # Now take these (n,4) data and put them in the grid format using padding.
#             # Once the prediction is run, we remove the padding to focus on the relevant part of the output
#             #  and we run a local blobber on it to look only at the most relevant pixels.
#             # We zero out the rest of the grid and compute the mean over remaining pixels
#             grid = Complex.get_grid_channels(other_coords,
#                                              spacing=spacing,
#                                              padding=margin,
#                                              xyz_min=xyz_min,
#                                              xyz_max=xyz_max,
#                                              is_target=False)
#             # np.save('out/ingrid_%04d.npy' % state, grid)
#             out_hd, out_pl = predict_frame(grid=grid, model=model)
#             out_grid = 1 - out_hd[-1] if not hetatm else out_pl
#
#             # focus on the initial box selection
#             out_grid = out_grid[margin:-margin, margin:-margin,
#                        margin:-margin]
#
#             mask = grid.sum(axis=0)[margin:-margin, margin:-margin,
#                    margin:-margin] > 0.01
#             score_old = np.sort(out_grid.flatten())[::-1][:vol].mean()


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

    # cmd.feedback('disable', 'all', 'everything')
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

    # cmd.feedback('disable', 'all', 'everything')
    # chain = 'polymer.protein'
    gpu_number = 0
    device = f'cuda:{gpu_number}' if torch.cuda.is_available() else 'cpu'

    if selection is None:
        box = f'polymer.protein'
    else:
        box = f'polymer.protein and {selection}'

    cmd.reinitialize()
    cmd.load(pdbfilename, 'ref_pdb')
    cmd.load(pdbfilename, 'traj')
    cmd.load_traj(trajfilename, 'traj', state=1)
    # Align the coordinates of the trajectory onto the reference pdb
    cmd.align(mobile='traj', target='ref_pdb', mobile_state=1)
    # cmd.delete('ref_pdb')
    # Align the trajectory on the selected atoms on the first frame
    cmd.intra_fit(f'traj and name CA and {box}', state=1)
    cmd.remove('hydrogens')
    print(cmd.get_object_list())
    print('Number of atoms in prot', cmd.select('ref_pdb'))
    print('Number of atoms in traj', cmd.select('traj'))
    nstates = cmd.count_states('traj')
    frames_to_use = nstates if max_frames is None else min(nstates, max_frames)

    if outfilename is None:
        outfilename = f'pred_rmsd.txt'

    predictions = list()
    with open(outfilename, 'w') as outfile:
        outfile.write(f'''# Topology file: {pdbfilename}
# Trajectory file: {trajfilename}
# Number of frames: {nstates}
# Box selection: {box}\n''')
        for state in range(frames_to_use):
            # Get the coords of this specific frame from the traj cmd object.
            # Then from this object find the right box based on the box selection
            coords = selection_to_split_coords(selection=f'traj and {box}',
                                               state=state + 1)
            # Use to define center and size, and then a complex
            center = tuple(coords.mean(axis=0)[:3])
            grid = GridComputer(points=coords, center=center, size=size, spacing=spacing).grid
            score = predict_frame(grid=grid, model=model, device=device)
            predictions.append(score)
            outfile.write(f"{state},{score}\n")
            if not state % 500:
                print("Done", state, 'score : ', score)
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
    # model.eval()

    # path_pdb = "data/low_rmsd/data/Pockets/PL_test/P08254/1b8y-A-P08254/1b8y-A-P08254_0001_last.mmtf"
    # path_sel = "data/low_rmsd/Resis/P08254_resis_ASA_thr_20.txt"
    # with open(path_sel, 'r') as f:
    #     sel = f.readline()
    # predict_pdb(model=model, pdbfilename=path_pdb, selection=sel)

    # corr_one = evaluate_one(model, max_frames=500)
    # print(corr_one.rvalue)

    all_res=evaluate_all(model, max_frames=200)



