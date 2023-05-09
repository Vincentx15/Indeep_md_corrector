import os
import mrcfile
import numpy as np
import pandas as pd
import torch

from model import RMSDModel


def model_from_name(model, eval=True):
    if isinstance(model, str):
        model_path = os.path.join("saved_models", f'{model}.pth')
        model = RMSDModel()
        model.load_state_dict(torch.load(model_path))
    if eval:
        model.eval()
    return model


def build_pl_csv(data_root="data/low_rmsd",
                 csv_to_read="df_rmsd_train.csv"):
    """
    Build a csv in a similar format that the one for rosetta's files, for the PL files
    One needs to do it in a non-redundant way
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


def correct_df(in_csv):
    with open(in_csv, 'r') as f:
        lines = f.readlines()
        updated_lines = [line.replace('PL_train', 'PL_validation') for line in lines]
        # updated_lines = [line.replace('PL_train', 'PL_test') for line in lines]
    with open(in_csv, 'w') as f:
        f.writelines(updated_lines)


def read_atomtypes():
    """
    Read the atomtype_mapping.txt file and return the
    mapping as a python dictionary
    """
    atomtype_mapping = {}
    with open(os.path.join(os.path.dirname(__file__), 'atomtype_mapping.txt'), 'r') as atomtypefile:
        for line in atomtypefile:
            line = line.strip()
            resname, pdbatomtype, atomtype = line.split(" ")
            if atomtype in atomtype_mapping:
                atomtype_mapping[atomtype].append((resname, pdbatomtype))
            else:
                atomtype_mapping[atomtype] = [(resname, pdbatomtype), ]
    return atomtype_mapping


def get_selection(atomtype, atomtype_mapping):
    sel_list = atomtype_mapping[atomtype]
    selection = ''
    for i, ra in enumerate(sel_list):
        resname, atomname = ra
        if i > 0 and i < len(sel_list):
            selection += ' | '
        selection += '(resn %s & name %s)' % (resname, atomname)
    return selection


ATOMTYPES = read_atomtypes()
ATOMTYPES = {atomtype: get_selection(atomtype, ATOMTYPES) for atomtype in ATOMTYPES}


def save_density(density, outfilename, spacing, origin, padding):
    """
    Save the density file as mrc for the given atomname
    """
    density = density.astype('float32')
    with mrcfile.new(outfilename, overwrite=True) as mrc:
        mrc.set_data(density.T)
        mrc.voxel_size = spacing
        mrc.header['origin']['x'] = origin[0] - padding
        mrc.header['origin']['y'] = origin[1] - padding
        mrc.header['origin']['z'] = origin[2] - padding
        mrc.update_header_from_data()
        mrc.update_header_stats()


if __name__ == '__main__':
    # build_pl_csv(csv_to_read="df_rmsd_train.csv")
    # build_pl_csv(csv_to_read="df_rmsd_validation.csv")
    # build_pl_csv(csv_to_read="df_rmsd_test.csv")

    in_csv = "../InDeep_holo-like_pred_toVincent/data/df_rmsd_validation.csv"
    # in_csv = "../data/data/df_rmsd_test.csv"
    correct_df(in_csv)
