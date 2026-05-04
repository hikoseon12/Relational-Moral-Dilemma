import pandas as pd


def open_csv(data_path):
    return pd.read_csv(data_path)


def save_csv(save_data_path, data):
    data.to_csv(save_data_path, index=False)
