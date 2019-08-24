import os, shutil
import pandas as pd


def build_samples_folder(path):
    df = pd.read_csv(os.path.join(path, 'samples.csv'))
    samples_folder = os.path.join(path, 'samples')
    if os.path.exists(samples_folder):
        shutil.rmtree(samples_folder)
    os.makedirs(samples_folder)
    for idx in range(len(df.index)):
        file1 = os.path.join(path, df['filename'][idx])
        _, fname = os.path.split(file1)
        file2 = os.path.join(samples_folder, fname)
        shutil.copyfile(file1, file2)


path = os.path.join(os.environ.get('DATA_PATH'), 'retina')
build_samples_folder(path)