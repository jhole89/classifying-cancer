import os
import shutil
import pandas as pd
import tensorflow as tf
from urllib.request import urlopen


def clean_run(model_dir='', train_data='', test_data=''):
    """Remove model and data files for a clean run"""

    if model_dir:
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
            print("\nCleaned: Model directory.\n")

    if train_data:
        if os.path.exists(train_data):
            os.remove(train_data)
            print("\nCleaned: Training data.\n")

    if test_data:
        if os.path.exists(test_data):
            os.remove(test_data)
            print("\nCleaned: Test data.\n")


def source_data(data_file, url):
    """Download data if not present on local FileSystem"""

    download_url = url + data_file

    if not os.path.exists(data_file):
        raw = urlopen(download_url).read()

        with open(data_file, 'wb') as f:
            f.write(raw)


if __name__ == '__main__':

    model_dir = 'nn_classifier'
    cancer_data = 'breast-cancer-wisconsin.data'
    data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/'

    clean_run(model_dir=model_dir, test_data=cancer_data)

    column_names = ['id', 'clump_thickness', 'unif_cell_size', 'unif_cell_shape',
                    'marg_adhesion', 'single_epith_cell_size', 'bare_nuclei',
                    'bland_chrom', 'norm_nucleoli', 'mitoses', 'class']

    source_data(cancer_data, data_url)

    dataframe = pd.read_csv(cancer_data, names=column_names)
