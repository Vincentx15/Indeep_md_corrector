import os
import random

import pymol.cmd as cmd
import numpy as np
from scipy import ndimage
from sklearn.gaussian_process.kernels import RBF
from scipy.spatial.transform import Rotation as R

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import sys

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, ''))

import utils


def build_selection(pdbfilename_p, selection=None):
    """
    Clean other pymol objects and extract coordinates
    :param pdbfilename_p:
    :param selection:
    :return:
    """
    # If cmd was initialized externally, remove conflicting names
    a = cmd.get_object_list('all')
    safe_sel = ['None', 'None'] + list({'prot', 'lig'}.intersection(set(a)))
    safe_sel = ' or '.join(safe_sel)
    cmd.delete(safe_sel)

    # Load the protein, prepare the general selection
    cmd.load(pdbfilename_p, 'prot')
    cmd.remove('hydrogens')
    prot_selection = 'prot and polymer.protein'
    selection = prot_selection if selection is None else prot_selection + ' and ' + selection
    return selection


def selection_to_split_coords(selection=None, state=0):
    # Store the coordinates in a (n, 4) array (coords_all)
    # with the last column corresponding to the channel id
    coords_all = None
    for cid, atomtype in enumerate(utils.ATOMTYPES):  # cid: channel id
        coords = cmd.get_coords(selection=f'{selection} and ({utils.ATOMTYPES[atomtype]})', state=state)
        if coords is not None:
            if coords_all is None:
                coords_all = np.c_[coords,
                np.repeat(cid, coords.shape[0])]
            else:
                coords_all = np.r_[coords_all,
                np.c_[coords,
                np.repeat(cid, coords.shape[0])]]
    coords_all = np.stack(coords_all)
    return coords_all


def get_split_coords(pdbfilename_p, selection=None, state=0):
    """
    The goal is to go from pdb files and optionnally some selections to the (n,4) or (n,1)
    format of coordinates annotated by their channel id
    """
    selection = build_selection(pdbfilename_p=pdbfilename_p, selection=selection)
    coords = selection_to_split_coords(selection=selection, state=state)
    return coords


"""
Make the conversion from (n,4) matrices to the grid format.
It also introduces the 'Complex' class that is fulling the Database object
"""


def just_one(coord, xi, yi, zi, sigma, feature, total_grid, use_multiprocessing=False):
    """

    :param coord: x,y,z
    :param grid:
    :param sigma:
    :return:
    """
    #  Find subgrid
    nx, ny, nz = xi.size, yi.size, zi.size

    bound = int(4 * sigma)
    x, y, z = coord
    binx = np.digitize(x, xi)
    biny = np.digitize(y, yi)
    binz = np.digitize(z, zi)
    min_bounds_x, max_bounds_x = max(0, binx - bound), min(nx, binx + bound)
    min_bounds_y, max_bounds_y = max(0, biny - bound), min(ny, biny + bound)
    min_bounds_z, max_bounds_z = max(0, binz - bound), min(nz, binz + bound)

    X, Y, Z = np.meshgrid(xi[min_bounds_x: max_bounds_x],
                          yi[min_bounds_y: max_bounds_y],
                          zi[min_bounds_z:max_bounds_z],
                          indexing='ij')
    X, Y, Z = X.flatten(), Y.flatten(), Z.flatten()

    #  Compute RBF
    rbf = RBF(sigma)
    subgrid = rbf(coord, np.c_[X, Y, Z])
    subgrid = subgrid.reshape((max_bounds_x - min_bounds_x,
                               max_bounds_y - min_bounds_y,
                               max_bounds_z - min_bounds_z))

    # Broadcast the feature throughout the local grid.
    subgrid = subgrid[None, ...]
    feature = feature[:, None, None, None]
    subgrid_feature = subgrid * feature

    #  Add on the first grid
    if not use_multiprocessing:
        total_grid[:, min_bounds_x: max_bounds_x, min_bounds_y: max_bounds_y,
        min_bounds_z:max_bounds_z] += subgrid_feature
    else:
        return min_bounds_x, max_bounds_x, min_bounds_y, max_bounds_y, min_bounds_z, max_bounds_z, subgrid_feature


def gaussian_blur(coords, xi, yi, zi, features=None, sigma=1., use_multiprocessing=False):
    """

    :param coords: (n_points, 3)
    :param xi:
    :param yi:
    :param zi:
    :param features: (n_points, dim) or None
    :param sigma:
    :param use_multiprocessing:
    :return:
    """

    nx, ny, nz = xi.size, yi.size, zi.size
    features = np.ones((len(coords), 1)) if features is None else features
    feature_len = features.shape[1]
    total_grid = np.zeros(shape=(feature_len, nx, ny, nz))

    if use_multiprocessing:
        import multiprocessing
        args = [(coord, xi, yi, zi, sigma, features[i], None, True) for i, coord in enumerate(coords)]
        pool = multiprocessing.Pool()
        grids_to_add = pool.starmap(just_one, args)
        for min_bounds_x, max_bounds_x, min_bounds_y, max_bounds_y, min_bounds_z, max_bounds_z, subgrid in grids_to_add:
            total_grid[:, min_bounds_x: max_bounds_x, min_bounds_y: max_bounds_y, min_bounds_z:max_bounds_z] += subgrid
    else:
        for i, coord in enumerate(coords):
            just_one(coord, feature=features[i], xi=xi, yi=yi, zi=zi, sigma=sigma, total_grid=total_grid)
    return total_grid


def get_grid(coords, features=None, spacing=2., padding=3, xyz_min=None, xyz_max=None, sigma=1.):
    """
    Generate a grid from the coordinates
    :param coords: (n,3) array
    :param features: (n,k) array
    :param spacing:
    :param padding:
    :param xyz_min:
    :param xyz_max:
    :param sigma:
    :return:
    """

    def get_bins(coords, spacing, padding, xyz_min=None, xyz_max=None):
        """
        Compute the 3D bins from the coordinates
        """
        if xyz_min is None:
            xm, ym, zm = coords.min(axis=0) - padding
        else:
            xm, ym, zm = xyz_min - padding
        if xyz_max is None:
            xM, yM, zM = coords.max(axis=0) + padding
        else:
            xM, yM, zM = xyz_max + padding

        # print(xm)
        # print(xM)
        # print(spacing)
        xi = np.arange(xm, xM, spacing)
        yi = np.arange(ym, yM, spacing)
        zi = np.arange(zm, zM, spacing)
        return xi, yi, zi

    xi, yi, zi = get_bins(coords, spacing, padding, xyz_min, xyz_max)
    grid = gaussian_blur(coords, xi, yi, zi, features=features, sigma=sigma)
    return grid


def get_enveloppe_grid(grid, gaussian_cutoff=0.2, iterations=6):
    """
    Returns a grid with a binary value corresponding to ones in the enveloppe of the input protein.
    :param grid:
    :param gaussian_cutoff:
    :param iterations:
    :return:
    """
    grid = grid.copy()
    summed_grid = np.sum(grid, axis=0)
    initial_mask = summed_grid > gaussian_cutoff
    mask = ndimage.binary_dilation(initial_mask, iterations=iterations)
    enveloppe = mask - np.int32(initial_mask)
    return enveloppe


class GridComputer(object):
    def __init__(self, points, center, name=None, size=20, spacing=1., rotate=True, compute_enveloppe=False):
        """
        - protein: a string identifying the protein
        - rotate: if True, randomly rotate the coordinates of the complex
        """
        self.name = name
        self.center = center
        self.spacing = spacing
        self.ids, coords = np.int32(points[:, -1]), points[:, :3]
        self.coords = self.random_rotate(coords) if rotate else coords
        self.xyz_min, self.xyz_max = np.asarray(center) - size // 2, np.asarray(center) + size // 2 + 0.0001

        self.grid = self.compute_grid()
        if compute_enveloppe:
            self.enveloppe = self.compute_enveloppe_grid()

    @staticmethod
    def random_rotate(coords, center=None):
        """
        Randomly rotate the coordinates of both proteins and ligands
        """
        alpha, beta, gamma = np.random.uniform(low=0., high=360., size=3)
        r = R.from_euler('zyx', [alpha, beta, gamma], degrees=True)
        # print(coords[:3])
        com = coords.mean(axis=0) if center is None else center
        coords_0 = coords - com
        rotated_shifted = r.apply(coords_0) + com
        # print(rotated_shifted[:3])
        return rotated_shifted

    def compute_grid(self):
        """
        Return the grid (with channels) for the protein
        with the corresponding channel ids (x, y, z, channel_id)
        """
        one_hot = np.eye(len(utils.ATOMTYPES))[self.ids]
        grid = get_grid(coords=self.coords,
                        features=one_hot,
                        spacing=self.spacing,
                        padding=0,
                        xyz_min=self.xyz_min,
                        xyz_max=self.xyz_max)
        return grid.astype(np.float32)

    def compute_enveloppe_grid(self):
        """
        Return the grid (with channels) for the protein
        with the corresponding channel ids (x, y, z, channel_id)
        """
        return get_enveloppe_grid(self.grid)

    def save_mrc_prot(self):
        """
        Save all the channels of the protein in separate mrc files
        """
        name = self.name if self.name is not None else 'protein'
        outbasename = os.path.splitext(name)[0]
        for channel_id, atomtype in enumerate(utils.ATOMTYPES):
            utils.save_density(self.grid[channel_id, ...],
                               '%s_%s.mrc' % (outbasename, atomtype),
                               self.spacing, self.xyz_min, padding=0)
        utils.save_density(self.grid.sum(axis=0),
                           '%s_ALL.mrc' % outbasename,
                           self.spacing, self.xyz_min, padding=0)


class RMSDDataset(Dataset):

    def __init__(self, csv_file="../data/data/df_rmsd_train.csv",
                 rotate=True,
                 get_pl_instead=0.5,
                 size=20,
                 spacing=1.0):
        self.df = pd.read_csv(csv_file)[['Path_PDB_Ros', 'Path_resis', 'RMSD']]
        # self.df = pd.read_csv(csv_file)
        self.rotate = rotate
        self.get_pl_instead = get_pl_instead
        self.size = size
        self.spacing = spacing

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        row = self.df.iloc[item, :]
        pdb, sel, rmsd = row
        pdb_filename = os.path.join('data', pdb)
        use_pl = self.get_pl_instead > random.random()
        if use_pl:
            pl_dir, decoy_file = os.path.split(pdb_filename)
            pl_file = decoy_file.split('_')[0] + '.pdb'
            pdb_filename = os.path.join(pl_dir, pl_file)
            rmsd = 0

        selection_filename = os.path.join('data', sel)
        with open(selection_filename, 'r') as f:
            sel = f.readline()

        # Based on that get coords
        coords = get_split_coords(pdbfilename_p=pdb_filename, selection=sel)

        # Use to define center and size, and then a complex
        center = coords.mean(axis=0)[:3]
        center = tuple(center)
        comp = GridComputer(points=coords, center=center, size=self.size, spacing=self.spacing)
        return comp.grid, np.asarray(rmsd).astype(np.float32)


if __name__ == '__main__':
    pass
    # pdbfilename_p = '../data/BCL2/pdbs/1gjh.pdb'
    # coords = build_coord_channel(pdbfilename_p)
    # comp = Complex(points=coords, name='1gjh', center=(3.4, -4.2, 16))
    # comp.save_mrc_prot()
    # print(comp.grid.shape)

    dataset = RMSDDataset()
    loader = DataLoader(dataset=dataset, num_workers=2, batch_size=2)
    for i, (grids, rmsds) in enumerate(loader):
        print(grids.shape)
