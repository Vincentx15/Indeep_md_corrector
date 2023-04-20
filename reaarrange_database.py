import os
import shutil
import glob

import pandas as pd


def merge(old_path, new_path, prefix):
    for mode in ['train', 'validation', 'test']:
        print(f"Doing {prefix} in mode : {mode}")
        # All old dataframes have a specificity, the names or columns are a bit different
        if prefix == 'hd':
            old_csv_path = os.path.join(old_path, 'data', f'df_rmsd_HD_{mode}.csv')
            df_old = pd.read_csv(old_csv_path)
        elif prefix == 'pl':
            old_csv_path = os.path.join(old_path, 'data', f'df_pl_{mode}.csv')
            df_old = pd.read_csv(old_csv_path, index_col=0)
        else:
            old_csv_path = os.path.join(old_path, 'data', f'df_rmsd_{mode}.csv')
            df_old = pd.read_csv(old_csv_path, index_col=0)

        new_csv_path = os.path.join(new_path, 'data', f'df_rmsd_{mode}.csv')
        if not os.path.exists(new_csv_path):
            # Now the new one can be infered, though we cannot start with pl that lacks a column
            assert prefix != 'pl'
            df_new = pd.DataFrame(data=None, columns=df_old.columns)
        else:
            df_new = pd.read_csv(new_csv_path, index_col=0)

        # df_new.drop(df_new[df_new['PDB_ros'].str.startswith('hd_')].index, inplace=True)
        # df_new.to_csv(new_csv_path)

        # Now let's go through our systems to copy the right files and the right lines
        for i, (_, row) in enumerate(df_old.iterrows()):
            if prefix == 'pl':
                # Path_PDB_Ros,Path_resis,RMSD
                path_pdb_ros, path_resi, rmsd = row.values
                pdb_ros = os.path.basename(path_pdb_ros).split('.')[0]
                pl = pdb_ros
            else:
                pdb_ros, pl, rmsd, path_pdb_ros, path_resi = row.values

            new_pdb_ros = f'{prefix}_{pdb_ros}'
            dirname = os.path.dirname(path_pdb_ros)
            new_basename = f'{prefix}_{os.path.basename(path_pdb_ros)}'
            new_path_pdb_ros = os.path.join(dirname, new_basename)
            if prefix == 'hd':
                new_path_pdb_ros = new_path_pdb_ros.replace('HD_', 'PL_')
            new_row = new_pdb_ros, pl, rmsd, new_path_pdb_ros, path_resi
            # df_new.loc[len(df_new)] = new_row
            df_new = df_new.append(pd.Series(new_row, index=df_new.columns), ignore_index=True)
            old_path_mmtf = os.path.join(old_path, path_pdb_ros)
            new_dir_mmtf = os.path.join(new_path, os.path.dirname(new_path_pdb_ros))
            new_path_mmtf = os.path.join(new_path, new_path_pdb_ros)
            os.makedirs(new_dir_mmtf, exist_ok=True)
            if not os.path.exists(new_path_mmtf):
                shutil.copy(old_path_mmtf, new_path_mmtf)
            if not i % 1000:
                print(i)
        df_new = df_new.drop_duplicates()
        df_new.to_csv(new_csv_path)


# merge('data/low_rmsd', 'data/fused', prefix='low_rmsd')
# merge('data/high_rmsd', 'data/fused', prefix='high_rmsd')
# merge('data/double_rmsd', 'data/fused', prefix='double_rmsd')
# merge('data/low_rmsd', 'data/fused', prefix='pl')
# merge('data/hd', 'data/fused', prefix='hd')


