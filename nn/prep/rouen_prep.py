"""
    Nyquist-Shannon sampling theorem: max_freq = sample_rate / 2
    Number of bin:

"""
import numpy as np
import glob
import librosa
import scipy.signal
import os, shutil
import pandas as pd
from scipy.signal import spectrogram, hanning
import random
import re


def melgram(x, sr=22050):
    x = x[0*65536:9*65536]
    x = librosa.resample(x, 22050, 5880)
    D = librosa.logamplitude(librosa.feature.melspectrogram(x, sr=sr, n_mels=512), ref_power=1.0)
    D = D[:512, :256]
    return D


def datagram(x):
    sr1 = 22050
    sr2 = sr1/20
    x = x[0*65536:9*65536]
    x = librosa.resample(x, sr1, sr2)
    n_fft = 1024
    win_length = n_fft # Number of samples in each signal window
    overlap = 0.95
    hop_length = int(win_length * (1.0 - overlap)) # Number of samples advanced each time (overlap)
    D = librosa.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=scipy.signal.hamming)
    spect, phase = librosa.magphase(D)
    spect = np.log1p(spect)
    #print(spect.shape); exit(0)
    spect = spect[:512, :512]
    return spect


def datagram2(x):
    n_fft = 1024
    win_length = n_fft
    overlap = 0.99
    hop_length = int(win_length * (1.0 - overlap)) # Number of samples advanced each time (overlap)
    D = librosa.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=scipy.signal.hamming)
    spect, phase = librosa.magphase(D)
    spect = np.mean(np.log1p(spect), axis=1)
    spect = spect[:512]
    return spect


def features(x, sr=22050):
    stft = np.abs(librosa.stft(x))
    mfccs = np.mean(librosa.feature.mfcc(y=x, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(x, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(x), sr=sr).T, axis=0)
    fs = [mfccs, chroma, mel, contrast, tonnetz]
    v = np.array([])
    for f in fs:
        v = np.concatenate((v, np.asarray(f)))
    return v


def build_dict_from_fname_prefix(path, func):
    """
        Classes defined by file names prefix
        Example: car107.wav
    """
    rows1, rows2 = [], []
    g = glob.glob(path + '/raw/*.wav')
    cnames = [re.findall(r'\D+', os.path.basename(f)[:-4])[0] for f in g]
    cnames = np.unique(cnames)

    print('Found {:d} files'.format(len(g)))
    print('Found {:d} classes'.format(len(cnames)))
    print('Classes: ', cnames)

    xlist = []
    for i, f in enumerate(sorted(g)):
        print(os.path.basename(f))
        c = re.findall(r'\D+', os.path.basename(f)[:-4])[0]
        x, sr = librosa.load(f)
        x = func(x)
        xlist.append(x)
        filename = 'files/{:04d}.npy'.format(i)
        rawfilename = 'raw/{:s}'.format(os.path.basename(f))
        target = c
        row = (filename, rawfilename, target)
        np.save(os.path.join(path, filename), x)
        if random.random() > 0.2:
            rows1.append(row)
        else:
            rows2.append(row)

    data = np.vstack(xlist)
    mean, std = data.mean(), data.std()
    print(f'mean={mean}, std={std}')

    df1 = pd.DataFrame.from_records(rows1, columns=['filename', 'rawfilename', 'class'])
    df1.to_csv(path + '/train.csv', index=False)
    df2 = pd.DataFrame.from_records(rows2, columns=['filename', 'rawfilename', 'class'])
    df2.to_csv(path + '/valid.csv', index=False)
    print(f'Files processed: {len(df1.index) + len(df2.index)}, train:{len(df1.index)}, valid:{len(df2.index)}')


def prep(basedir, func):
    path = os.path.join(os.environ.get('DATA_PATH'), basedir)
    if os.path.exists(os.path.join(path, 'files')):
        shutil.rmtree(os.path.join(path, 'files'))
    if os.path.isfile(os.path.join(path, 'train.csv')):
        os.remove(os.path.join(path, 'train.csv'))
    if os.path.isfile(os.path.join(path, 'valid.csv')):
        os.remove(os.path.join(path, 'valid.csv'))
    os.makedirs(os.path.join(path, 'files'))
    build_dict_from_fname_prefix(path, func)


prep('rouen', datagram2)
