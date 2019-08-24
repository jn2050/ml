import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import numpy as np
import pandas as pd
import os
import random
from lib.prep import *
from lib.util import *
# pylint: disable=E1101


def np_split_by_idx(p, *a):
    n = a[0].shape[0]
    n1 = int(p*n)
    idxs = np.random.permutation(n)[:n1]
    mask = np.zeros(n, dtype=bool)
    mask[np.array(idxs)] = True
    return [(o[mask], o[~mask]) for o in a]


class Data:
    def __init__(self, cfg, ds1, ds2, sz=None, bs=None, workers=8, balanced=False):
        sampler = None
        if bs is None: bs = cfg.bs
        self.bs = bs
        self.path = cfg.path
        self.ds1 = ds1
        self.ds2 = ds2
        self.reg = False
        self.sz = sz
        pin = str(cfg.device) != 'cpu'
        self.dl1 = torch.utils.data.DataLoader(self.ds1, batch_size=bs, shuffle=True, num_workers=workers, pin_memory=pin, sampler=sampler)
        self.dl2 = torch.utils.data.DataLoader(self.ds2, batch_size=bs, shuffle=False, num_workers=workers, pin_memory=pin)
        self.classes = self.dl1.dataset.classes
        self.c = len(self.classes)
        cfg.logger.print_dl_stats(self)

    @classmethod
    def image_from_file(cls, cfg, sz=None, trf1=None, trf2=None, trfy=None, bs=None, workers=1, balanced=False, meta_cols=[]):
        ds1 = ImageFileDataset(cfg.path, 'train.csv', trf1, trfy, meta_cols)
        ds2 = ImageFileDataset(cfg.path, 'valid.csv', trf2, trfy, meta_cols)
        data = cls(cfg, ds1=ds1, ds2=ds2, sz=sz, bs=bs, workers=workers, balanced=balanced)
        return data

    @classmethod
    def numpy_from_file(cls, cfg, sz=None, trf1=None, trf2=None, trfy=None, bs=None, workers=1, balanced=False):
        ds1 = NumpyFileDataset(cfg.path, 'train.csv', trf1, trfy)
        ds2 = NumpyFileDataset(cfg.path, 'valid.csv', trf2, trfy)
        data = cls(cfg, ds1=ds1, ds2=ds2, sz=sz, bs=bs, workers=workers, balanced=balanced)
        return data

    @classmethod
    def numpy_from_memmory(cls, cfg, xy1, xy2, sz=None, trf1=None, trf2=None, trfy=None, bs=None, workers=1, balanced=False):
        ds1 = NumpyMemoryDataset(xy1, trf1, trfy)
        ds2 = NumpyMemoryDataset(xy2, trf2, trfy)
        data = cls(cfg, ds1=ds1, ds2=ds2, sz=sz, bs=bs, workers=workers, balanced=balanced)
        return data

    def load(self, fname):
        return self.ds1.load(fname)

    def pre(self, x):
        return self.ds1.pre(x)

    def pos(self, y):
        return self.ds1.pos(y)


class BaseDataset(Dataset):
    def __init__(self, x, y, reg=True, binary=False, switch_p=0.0, switch_k=0):
        self.bp = []

    def set_balance(self, b):
        if b>1.0: b=1.0
        if b<0.0: b=0.0
        bp1 = np.array([1/len(self.c_idxs)]*len(self.c_idxs))
        bp0 = np.array([len(c)/len(self) for c in self.c_idxs])
        self.bp = bp0+b*(bp1-bp0)
        print(f'Balanced: {b:1.2f} ->', self.bp)

    def _get_balanced_idx(self, idx):
        if not len(self.bp):
            return idx
        c = biased_roll(self.bp)
        return np.random.choice(self.c_idxs[c])


class NumpyMemoryDataset(Dataset):
    def __init__(self, xy, trf=None, trfy=None):
        self.x = torch.from_numpy(xy[0]).float()
        self.y = torch.from_numpy(xy[1]).long()
        self.trf = trf
        self.trfy = trfy
        if self.trfy:
            pass
        self.classes = np.unique(xy[1])
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.c = len(self.classes)
        labels = xy[1]
        self.c_idxs = np.array([np.where(labels==self.class_to_idx[c])[0] for c in self.classes])
        self.hist = [self.c_idxs[i].shape[0] for i in range(self.c_idxs.shape[0])]

    def __len__(self):
        return self.x.size()[0]

    def __getitem__(self, idx):
        if idx == len(self):
           raise StopIteration
        return self.x[idx], int(self.y[idx])


class FileDataset(Dataset):
    def __init__(self, path, csvfile, trf=None, trfy=None, meta_cols=[]):
        self.path = path
        self.trf = trf
        self.trfy = trfy
        self.balance = []
        self.meta_cols = meta_cols
        df = pd.read_csv(os.path.join(self.path, csvfile))[['filename',  'class'] + meta_cols]
        df['class'] = df['class'].astype(str)
        if self.trfy:
            df['class'] = df['class'].apply(self.trfy)
        self.classes = np.unique(df['class'])
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.c = len(self.classes)
        df['class'] = df['class'].apply(lambda y: self.class_to_idx[y])
        self.df = df
        #self.df, self.stats = df_process(df, [], meta_cols)
        labels = np.array(df['class'])
        self.c_idxs = np.array([np.where(labels==self.class_to_idx[c])[0] for c in self.classes])
        self.hist = [self.c_idxs[i].shape[0] for i in range(self.c_idxs.shape[0])]
        self.bp = []

    def set_balance(self, b):
        if b>1.0: b=1.0
        if b<0.0: b=0.0
        bp1 = np.array([1/len(self.c_idxs)]*len(self.c_idxs))
        bp0 = np.array([len(c)/len(self) for c in self.c_idxs])
        self.bp = bp0+b*(bp1-bp0)
        print(f'Balanced: {b:1.2f} ->', self.bp)

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        if idx == len(self):
            raise StopIteration
        if len(self.bp):
            c = biased_roll(self.bp)
            idx = np.random.choice(self.c_idxs[c])
        fname = self.df['filename'][idx]
        x = self.pre(fname)
        cols = ['rows', 'cols']
        if len(self.meta_cols):
            m = self.df.iloc[idx][cols].tolist()
            m = torch.FloatTensor(m).float()
            x = (x, m)
        y = int(self.df['class'][idx])
        return x, y

    def pre(self, fname):
        fname = os.path.join(self.path, fname)
        x = self.load(fname)
        return torch.from_numpy(x).float()

    def pos(self, y):
        c_idx = y.max(1, keepdim=True)[1].item()
        return self.classes[c_idx]


class ImageFileDataset(FileDataset):
    def load(self, fname):
        x = open_image(fname)
        if self.trf:
            x = self.trf(x)
        x = np.moveaxis(x, 2, 0)
        x = x.astype(np.float32)/255.0
        return x


class NumpyFileDataset(FileDataset):
    def load(self, fname):
        x = np.load(fname)
        if self.trf:
            x = self.trf(x)
        return x


class TabData:
    def __init__(self, cfg, df1, df2, cols_skip=[], bs=64, reg=True, binary=False, switch_p=0.0, switch_k=5, shuffle=False):
        """
            Assumes df1,df2 already processed with [cols_cat][cols_cont][cols_skip][col_dep] sequence, and
            cols_cat of type int32, cols_cont of type float32
        """
        self.reg = reg
        self.binary = binary
        cols = [c for c in df1.columns if c not in cols_skip]
        col_dep = cols[-1]
        self.df1, self.df2 = df1[cols], df2[cols]
        self.cols_cat = [c for c in cols if df1[c].dtype=='int32']
        self.ncats = len(self.cols_cat)
        self.nconts = len(cols)-self.ncats-1
        self.cat_sizes = {c:len(df1[c].unique()) for c in self.cols_cat}

        if reg:
            df1[col_dep] = df1[col_dep].astype('float32')
            df2[col_dep] = df2[col_dep].astype('float32')
            if not binary:
                df1[col_dep] = np.log(df1[col_dep])
                df2[col_dep] = np.log(df2[col_dep])
                self.y_range = (np.min(df1[col_dep])*0.8, np.max(df1[col_dep])*1.2)
        else:
            self.classes = np.sort(df1[self.col_dep].unique())
            self.c = len(self.classes)
            self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
            df1[col_dep] = [self.class_to_idx[e] for e in df1[col_dep]]
            df2[col_dep] = [self.class_to_idx[e] for e in df2[col_dep]]
            df1[col_dep] = df1[col_dep].astype('int32')
            df2[col_dep] = df2[col_dep].astype('int32')

        xy1 = df1.values
        xy2 = df2.values
        self.ds1 = SimpleDataset(xy1[:,:-1], xy1[:,-1], reg, binary, switch_p, switch_k)
        self.ds2 = SimpleDataset(xy2[:,:-1], xy2[:,-1], reg, binary)
        self.dl1 = torch.utils.data.DataLoader(self.ds1, batch_size=bs, shuffle=shuffle)
        self.dl2 = torch.utils.data.DataLoader(self.ds2, batch_size=bs, shuffle=False)
        cfg.logger.print_dl_stats(self)

    def pre(self, df):
        pass

    def pos(self, y):
        y = y.exp()
        return y


class SimpleDataset(Dataset):
    def __init__(self, x, y, reg=True, binary=False, switch_p=0.0, switch_k=0):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float() if reg else torch.from_numpy(y).long()
        if not reg:
            self.classes = np.unique(self.y)
            self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
            self.c = len(self.classes)
            labels = self.y.numpy()
            self.c_idxs = np.array([np.where(labels==self.class_to_idx[c])[0] for c in self.classes])
            self.hist = [self.c_idxs[i].shape[0] for i in range(self.c_idxs.shape[0])]
        if reg and binary:
            self.c_idxs = np.array([np.where(y==v)[0] for v in [0,1]])
        self.bp = []
        self.switch_p = switch_p
        self.switch_k = switch_k if switch_k>0 else self.x.size()[1]//10

    def set_balance(self, b):
        if b>1.0: b=1.0
        if b<0.0: b=0.0
        bp1 = np.array([1/len(self.c_idxs)]*len(self.c_idxs))
        bp0 = np.array([len(c)/self.x.shape[0] for c in self.c_idxs])
        self.bp = bp0+b*(bp1-bp0)
        print(f'Balanced: {b:1.2f} ->', self.bp)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if idx == len(self):
            raise StopIteration
        if len(self.bp):
            c = biased_roll(self.bp)
            idx = np.random.choice(self.c_idxs[c])
            x, y = self.x[idx], self.y[idx]
            idx2 = np.random.choice(self.c_idxs[c])
        else:
            x, y = self.x[idx], self.y[idx]
            idx2 = np.random.choice(len(self))
        if self.switch_p and random.random()>self.switch_p:
            x2 = self.x[idx2]
            for _ in range(self.switch_k):
                j = random.randint(0,self.x.size()[1]-1)
                x[j] = x2[j]
        return x, y


class ColabData:
    def __init__(self, df, bs=64):
        self.users, self.user2idx, user_x, self.n_users = self.proc_col(df.ix[:,0])
        self.items, self.item2idx, item_x, self.n_items = self.proc_col(df.ix[:,1])
        ratings_x = df.ix[:,2].values.astype(np.float32)
        self.n = len(ratings_x)
        self.min_rating, self.max_rating = min(ratings_x), max(ratings_x)

        user_xs, item_xs, ratings_xs = np_split_by_idx(0.8, user_x, item_x, ratings_x)
        train_x = np.stack((user_xs[0], item_xs[0]), axis=1)
        val_x = np.stack((user_xs[1], item_xs[1]), axis=1)

        self.ds1 = SimpleDataset(train_x, ratings_xs[0])
        self.ds2 = SimpleDataset(val_x, ratings_xs[1])
        self.dl1 = torch.utils.data.DataLoader(self.ds1, batch_size=bs, shuffle=True)
        self.dl2 = torch.utils.data.DataLoader(self.ds2, batch_size=bs, shuffle=False)
        print('Rows in train dataset: {:,}'.format(len(self.ds1)).replace(',', '.'))
        print('Rows in valid dataset: {:,}'.format(len(self.ds2)).replace(',', '.'))

    def proc_col(self, col):
        uniq = col.unique()
        name2idx = {o: i for i, o in enumerate(uniq)}
        col_data = np.array([name2idx[x] for x in col])
        return uniq, name2idx, col_data, len(uniq)
