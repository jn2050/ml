import pandas as pd
import os
from shutil import copyfile


def copy_df_files(df):
    for idx, row in df.iterrows():
        fname1 = os.path.join(path, row.rawfilename)
        _, name = os.path.split(fname1)
        fname2 = os.path.join(path, os.path.join('images', name))
        copyfile(fname1, fname2)


basedir = 'retina'
path = os.path.join(os.environ.get('DATA_PATH'), basedir)

if os.path.exists(os.path.join(path, 'images')):
    print('images dir exist, aborting')
    exit(0)

os.makedirs(os.path.join(path, 'images'))

df = pd.read_csv(os.path.join(path, 'train.csv'))
copy_df_files(df)

df = pd.read_csv(os.path.join(path, 'valid.csv'))
copy_df_files(df)

print('Done')
