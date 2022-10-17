import os
import pandas as pd
import torch


def load_all(train=True, meta=False):
    wdir = os.path.join('raw_indeep_corrector/train_indeep', 'Training' if train else 'Test')
    csv_files = []
    for file in os.listdir(wdir):
        if meta and 'Meta' in file:
            csv_files.append(file)
        if not meta and not 'Meta' in file:
            csv_files.append(file)
    csv_list = [pd.read_csv(os.path.join(wdir, csv_file)) for csv_file in csv_files]
    csv_list = [csv[['Ligandability', 'R_G', 'RMSD_pl_min']] for csv in csv_list]
    return csv_list


def df_to_torch(df):
    return torch.from_numpy(df.to_numpy().T)[None, ...].float()


def process_df(df):
    data_in, data_out = df[['Ligandability', 'R_G']], df[['RMSD_pl_min']]
    data_in, data_out = df_to_torch(data_in), df_to_torch(data_out)
    return data_in, data_out


def baseline_df(df):
    data_in, data_out = df[['Ligandability']], df[['RMSD_pl_min']]
    data_in, data_out = df_to_torch(data_in), df_to_torch(data_out)
    return data_in, data_out


if __name__ == '__main__':
    pass
    data = load_all(train=True, meta=True)
    print(data)
