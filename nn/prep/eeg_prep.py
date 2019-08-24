
import scipy
from scipy import signal
import scipy.io as sio
import librosa
import numpy as np
import pandas as pd
import random
import os, glob, shutil
import subprocess
import re


path = os.path.join(os.environ.get('DATA_PATH'), 'eeg')
BUILD_NPY = False
MULTI_RUN = True
BUILD_DS = False
RACIO = 0.90
FR = 128
K 
F = K*FR
idx = 0

#L07090842 File missing
DROP =  ['L07090842', 'L07121609', 'L07161521', 'L07241524', 'L07251510']

# 46 clean EEGs
DS = [ 'L04181753', 'L04182149', 'L04192151', 'L04231615', 'L04232358', 'L04240848', 'L04260912', 'L04261620',
        'L04300858', 'L04301506', 'L05021532', 'L05070746', 'L05091456', 'L05100854', 'L05101539', 'L05170850',
        'L05171222', 'L05171518', 'L05210855', 'L05240858', 'L05280831', 'L05281331', 'L05281700', 'L05300839',
        'L06040831', 'L06051516', 'L06060902', 'L06061241', 'L06070900', 'L06071523', 'L06071842', 'L06141433',
        'L06141745', 'L06180840', 'L06191456', 'L06210858', 'L06211707', 'L07091230', 'L07120855', 'L07121419',
        'L07180853', 'L07190902', 'L07190921', 'L07191306', 'L07230915', 'L07250853' ]

def spectrum(x, fr=FR//2, maxf=None):
    n = len(x)
    d = 1 / fr
    hs = np.fft.rfft(x)
    s = np.absolute(hs)
    if maxf is not None:
        s = s[:maxf]
    fs = np.fft.rfftfreq(n, d)
    return s,fs


def load_eeg(fname):
    if not os.path.isfile(fname):
        print(f'Cant find file {fname}. Aborting.')
        exit(1)
    data = sio.loadmat(fname)
    data = np.array(data['raw_data']).astype(float)
    ss = []
    for s in data:
        ss.append(s.astype(float))
    return ss


def print_stats(x):
    print(x.min(), x.max(), x.mean(), x.std())


def build_index():
    file = os.path.join(path, 'EEG.xlsx')
    cols = ['File', 't0', 't5']
    df = pd.read_excel(file, 'Data')[cols]
    df = df.set_index('File', drop=False)
    return df


def mat_to_npy(df):
    print('Preparing files')
    for _,row in df.iterrows():
        fname1 = os.path.join(path, 'mat/'+row['File']+'.mat')
        fname2 = os.path.join(path, 'npy/'+row['File']+'.npy')
        x = np.array(load_eeg(fname1))
        np.save(fname2, x)
    print(f'Files transfered: {len(df.index)}')


def mat_to_npy_and_norm(df):
    print('Normalizing')
    data = []
    for _,row in df.iterrows():
        fname = 'mat/'+row['File']+'.mat'
        x = np.array(load_eeg(os.path.join(path, fname)))
        #x = np.clip(x, -150, 150)
        for e in x:
            for v in e:
                data.append(v)
    data = np.array(data)
    _mean, _std = data.mean(), data.std()
    print(f'mean={_mean}, std={_std}')
    for _,row in df.iterrows():
        fname1 = os.path.join(path, 'mat/'+row['File']+'.mat')
        fname2 = os.path.join(path, 'npy/'+row['File']+'.npy')
        x = np.array(load_eeg(fname1))
        #x = np.clip(x, -150, 150)
        x = (x-_mean)/_std
        np.save(fname2, x)
    print(f'Files normalized: {len(df.index)}')


def split_datasets(df, racio=RACIO):
    df = df.sample(frac=1)
    cut = int(racio*len(df))
    df1, df2 = df[:cut], df[cut:]
    s = 'train_set = ['
    for v in df1['File'].values:
        s += f"'{v}',"
    s += ']'
    print(s)
    s = 'valid_set = ['
    for v in df2['File'].values:
        s += f"'{v}',"
    s += ']'
    print(s)
    return df1, df2


def build_blocks(ss, cuts):
    #i1 = cuts[0]*FR+30*FR
    i1 = cuts[1]*FR-10*FR
    i2 = cuts[1]*FR-0*FR
    i3 = len(ss[0]) if len(ss[0])/FR<90*60 else len(ss[0])-30*60*FR
    n1, n2 = i1//F, (i3-i2)//F
    if n2>1000: n2=1000
    #if n2>n1: n2=n1
    x0, x1 = [], []
    if ss.shape[0]==2:
        pass
    for i in range(n1):
        x = [s[i*F:(i+1)*F] for s in ss]
        #x = [spectrum(s[i*F:(i+1)*F])[0] for s in ss]
        x = np.concatenate(x).reshape(len(x),-1)
        x0.append(x)
    for i in range(n2):
        x = [s[i2+i*F:i2+(i+1)*F] for s in ss]
        #x = [spectrum(s[i2+i*F:i2+(i+1)*F])[0] for s in ss]
        x = np.concatenate(x).reshape(len(x),-1)
        x1.append(x)
    return x0, x1


def build_dataset(df, csvname):
    global idx
    rows = []
    for _, row in df.iterrows():
        fname = os.path.join(path, 'npy/'+row['File']+'.npy')
        ss = np.load(fname)
        cuts = [row['t0'], row['t5']]
        x0, x1 = build_blocks(ss, cuts)
        #print(fname, len(x0), len(x1))
        for e in x0:
            f = f'files/file_{idx}.npy'
            rows.append((f, '1'))
            np.save(os.path.join(path, f), e)
            idx += 1
        for e in x1:
            f = f'files/file_{idx}.npy'
            rows.append((f, '0'))
            np.save(os.path.join(path, f), e)
            idx += 1
    df = pd.DataFrame.from_records(rows, columns=['filename', 'class'])
    df = df.sample(frac=1)
    df.to_csv(os.path.join(path, csvname), index=False)
    print(f'Dataset built: {csvname}: {df.shape}')
    return df


def reset_files_dirs():
    if os.path.exists(os.path.join(path, 'files')):
        shutil.rmtree(os.path.join(path, 'files'))
    os.makedirs(os.path.join(path, 'files'))
    if os.path.isfile(os.path.join(path, 'train.csv')):
        os.remove(os.path.join(path, 'train.csv'))
    if os.path.isfile(os.path.join(path, 'valid.csv')):
        os.remove(os.path.join(path, 'valid.csv'))


df = build_index()
df = df.drop(DROP)

if BUILD_NPY:
    if os.path.exists(os.path.join(path, 'npy')):
        shutil.rmtree(os.path.join(path, 'npy'))
    os.makedirs(os.path.join(path, 'npy'))
    #mat_to_npy_and_norm(df)
    mat_to_npy(df)
    exit(0)


if MULTI_RUN:
    fname_best = os.path.join(path, 'models/model_EEG_Net_best.pth.tar')
    fname_best_best = os.path.join(path, 'models/model_EEG_Net_best_best.pth.tar')
    VS = [9, 5, 2] # [23, 9, 5, 2, 1]
    NRUNS = 5
    EP = 100
    results = []
    best = 0.0
    for V in VS:
        df1, df2 = df.iloc[:len(DS)-V], df.iloc[len(DS)-V:]
        run = f'DATASET_{df1.shape[0]}_{df2.shape[0]}'
        reset_files_dirs()
        df_train = build_dataset(df1, 'train.csv')
        df_valid = build_dataset(df2, 'valid.csv')
        for k in range(NRUNS):
            print(run)
            fname = os.path.join(path, 'tmp_result.txt')
            f = open(fname, 'w')
            s = subprocess.call(['python', 'nn.py', 'eeg', '-e', str(EP), '--rm'], stdout=f)
            f.close()
            with open(fname, 'r') as f:
                s = f.read().replace('\n', '')
            v = float(re.search('(\d+\.\d+)$', s)[0])
            if v > best:
                best = v
                shutil.copyfile(fname_best, fname_best_best)
            print(f'Best:{v:.4f}')
            results.append((run,v))
            dfr = pd.DataFrame.from_records(results, columns=['run', 'best'])
    print(dfr)
    print(f'Best Best:{best:.4f}')
    dfr.to_csv(os.path.join(path, 'multi_results_e100_p04.csv'), index=False)
    exit(0)


if BUILD_DS:
    #df1, df2 = split_datasets(df)
    V = 9
    df1, df2 = df.iloc[:len(DS)-V], df.iloc[len(DS)-V:]
    df1.to_csv(os.path.join(path, 'df1'), index=False)
    df2.to_csv(os.path.join(path, 'df2'), index=False)
else:
    df1 = pd.read_csv(os.path.join(path, 'df1'))
    df2 = pd.read_csv(os.path.join(path, 'df2'))

reset_files_dirs()
df_train = build_dataset(df1, 'train.csv')
df_valid = build_dataset(df2, 'valid.csv')
