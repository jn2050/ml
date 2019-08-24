"""

    Build valid dir from train dir, moving part of the files from train to valid
    Builds train.csv and valid. csv from all.csv

    OpenCV images shape: [W, H, C], C is [Blue, Green, Red]

"""
# pylint: disable=E1101
import numpy as np
import pandas as pd
import cv2
import multiprocessing as mp
import glob, os, argparse, sys, random
sys.path.append('.')
from lib.transforms import *

P = 8

trf = Transforms([
    ScaleRadius(300.0),
    NormalizeColor(10.0),
    RemoveOuter(300.0),
    MinScale(512),
    CenterCrop(512),
])


def is_valid_file(fname):
    fname = fname.lower()
    return any(fname.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm'])


def mk_samples(path, csvfile, samplesfile, n=16, szs=None):
    cfile = os.path.join(path, csvfile)
    sfile = os.path.join(path, samplesfile)
    df = pd.read_csv(cfile)
    rows = []
    if szs is None:
        data = df.as_matrix(columns=df.columns)
        rand = np.random.choice(range(data.shape[0]), n, replace=False)
        rows = data[rand]
    else:
        for i, c in enumerate(sorted(df['class'].unique())):
            df1 = df[df['class'] == c]
            data = df1.as_matrix(columns=df.columns)
            rand = np.random.choice(range(data.shape[0]), szs[i], replace=False)
            rows += list(data[rand])
    df = pd.DataFrame.from_records(rows, columns=columns)
    df.to_csv(sfile, index=False)


def split_train_valid(path, fname, fname1, fname2, p=0.8):
    f = os.path.join(path, fname)
    f1 = os.path.join(path, fname1)
    f2 = os.path.join(path, fname2)
    df = pd.read_csv(f)
    n = len(df.index)
    n1 = int(p*n)
    idxs = np.random.permutation(n)[:n1]
    mask = np.zeros(n, dtype=bool)
    mask[idxs] = True
    df1 = df.loc[mask]
    df2 = df.loc[~mask]
    df1.to_csv(f1, index=False)
    df2.to_csv(f2, index=False)


def prep_one(dir1, dir2, df, i):
    rawfilename = dir1 + '/' + df.ix[i][0] + '.jpeg'
    filename = dir2 + '/' + df.ix[i][0] + '.jpeg'
    target = str(df.ix[i][1])
    if not os.path.isfile(os.path.join(path, rawfilename)):
        return None
    print(rawfilename)
    raw = open_image(os.path.join(path, rawfilename))
    if raw.std() < 0.10:
        print('Skiping low quality image')
        return None
    img = trf(raw)
    save_image(os.path.join(path, filename), img)
    var = np.var(np.swapaxes(img, 0, 2).reshape((3, -1)), axis=1)
    row = (filename, rawfilename, target, raw.shape[0], raw.shape[1], var[0], var[1], var[2])
    return row


def prep(path, csvfile, dir1, dir2):
    path2= os.path.join(path, dir2)
    df = pd.read_csv(csvfile)
    print(f'Building indexes from {csvfile}, found {len(df.index)} rows')
    if os.path.exists(path2):
        print(f'Destination path {path2} already exists. Aborted.')
        exit(0)
    else:
        os.makedirs(path2)
    rows1, rows2 = [], []

    #pool = mp.Pool(processes=P)
    #rows = [pool.apply(prep_one, args=(dir1, dir2, df, i,)) for i in range(len(df.index))]

    for i in range(len(df.index)):
        row = prep_one(dir1, dir2, df, i)
        if row is None:
            continue
        if random.random() > 0.2:
            rows1.append(row)
        else:
            rows2.append(row)

    df1 = pd.DataFrame.from_records(rows1, columns=columns)
    df1.to_csv(path + '/train.csv', index=False)
    df2 = pd.DataFrame.from_records(rows2, columns=columns)
    df2.to_csv(path + '/valid.csv', index=False)
    print(f'Processed: {len(df1.index)+len(df2.index)} total, {len(df1.index)} train, {len(df2.index)} valid')
    

def stats(path, csvfile):
    df = pd.read_csv(csvfile)
    m = np.zeros(3)
    s = np.zeros(3)
    for fname in df['filename']:
        img = open_image(os.path.join(path, fname))
        for i in range(3):
            m[i] += img[:,:,i].mean()
    m = m / len(df.index)
    print('mean = ', m)
    print('mean = ', m/255.0)
    for fname in df['filename']:
        img = open_image(os.path.join(path, fname))
        for i in range(3):
            s[i] += np.square((img[:,:,i] - m[i])).sum() / (img.shape[0]*img.shape[1])
    s = np.sqrt(s/len(df.index))
    print('std = ', s)
    print('std = ', s/255.0)


def stats_size(path, csvfile):
    df = pd.read_csv(csvfile)[['rows', 'cols']].drop_duplicates()
    print(df)


columns = ['filename', 'rawfilename', 'class', 'rows', 'cols', 'var1', 'var2', 'var3']
path = os.path.join(os.environ.get('DATA_PATH'), 'retina')

parser = argparse.ArgumentParser(description='Retina pre-processor')
parser.add_argument('--prep', help='Pre-process images',  action='store_true')
parser.add_argument('--samples', help='Build samples subset',  action='store_true')
parser.add_argument('--stats', help='Image stats',  action='store_true')
parser.add_argument('--sizes', help='Image sizes stats',  action='store_true')
args = parser.parse_args()

if args.sizes:
    stats_size(path, os.path.join(path, 'train.csv'))
    exit(0)
if args.stats:
    stats(path, os.path.join(path, 'train.csv'))
    exit(0)
if args.samples:
    mk_samples(path, 'train.csv', 'balanced.csv', 2, szs=[2048, 1969, 2048, 704, 557])
    split_train_valid(path, 'balanced.csv', 'train_balanced.csv', 'valid_balanced.csv')
    exit(0)
if args.prep:
    prep(path, os.path.join(path, 'retina.csv'), 'raw', 'images')
    exit(0)
