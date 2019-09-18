import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype, is_numeric_dtype
import random, re


#dic.get(key, 0)
#dic = defaultdict()
#dic.__missing__ = lambda key: 0


DATE_PARTS = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear', 'Is_month_end', 'Is_month_start', 
            'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']


def scaller__norm(s, _mean, _std):
    return (s-_mean)/_std if _std>0 else s


def scaller__minmax(s, _min, _max):
    return ((s-_min)/(_max-_min)-0.5)*2.0 if (_max-_min)>0 else s


class Tab(object):
    def __init__(self, df, cols_cat, cols_cont, col_dep, cols_skip=[], maxhot=8):
        cols_onehot = []
        for c in cols_cat:
            if len(df[c].unique())<=maxhot:
                cols_onehot += [c]
        cols_cat2 = [c for c in cols_cat if c not in cols_onehot]
        nconts = len(cols_cont*2)
        for c in cols_onehot:
            nconts += len(df[df[c].isna()==False][c].unique())+1
        self.cols_cat = cols_cat
        self.cols_cat2 = cols_cat2
        self.cols_onehot = cols_onehot
        self.cols_cont = cols_cont
        self.col_dep = col_dep
        self.cols_skip = cols_skip
        self.ncats = len(self.cols_cat2)
        self.nconts = nconts
        self.mean = {c: df[c].mean() for c in cols_cont}
        self.std = {c: df[c].std() for c in cols_cont}
        self.median = {c: df[c].median() for c in cols_cont}
        self.min = {c: df[c].min() for c in cols_cont}
        self.max = {c: df[c].max() for c in cols_cont}
        self.d = {c: None for c in cols_cat}
        for c in cols_cat2+cols_onehot:
            self.d[c] = {e: i+1 for i,e in enumerate(list(np.sort(df[df[c].isna()==False][c].astype(str).unique())))}

    def proc(self, df):
        cols1 = self.cols_cat+self.cols_cont+self.cols_skip+[self.col_dep]
        cols1p = self.cols_cat+self.cols_cont+[self.col_dep]
        df = df[cols1].copy()
        for c in cols1p:
            un = len(df[c].unique())
            if un<2:
                print(f'Warning: column {c} with {un} unique values')
        for c in self.cols_cat2:
            df[c] = df[c].astype(str)
            df[c] = [self.d[c][e] if e in self.d[c] else 0 for e in df[c]]
        cols_dummies = []
        for c in self.cols_onehot:
            df[c] = df[c].astype(str)
            df[c] = [self.d[c][e] if e in self.d[c] else 0 for e in df[c]]
            df[c+'__NAN'] = (df[c]==0)
            cols_dummies.append(c+'__NAN')
            for e in self.d[c]:
                df[c+'__'+e] = (df[c]==self.d[c][e])
                cols_dummies.append(c+'__'+e)
        cols_na = []
        for c in self.cols_cont:
            cols_na.append(c+'_na')
            df[c+'_na'] = pd.isnull(c)
            df[c] = df[c].fillna(self.median[c])
            df[c] = scaller__norm(df[c], self.mean[c], self.std[c])
        for c in self.cols_cat2:
            df[c] = df[c].astype('int32')
        cols_cont2 = self.cols_cont+cols_na+cols_dummies
        for c in cols_cont2:
            df[c] = df[c].astype('float32')
        cols2 = self.cols_cat2+cols_cont2+self.cols_skip+[self.col_dep]
        df = df[cols2]
        print(f'df processed: rows={len(df)}, cols={len(df.columns)}, ncats={len(self.cols_cat2)}, nconts={len(cols_cont2)}, nskips={len(self.cols_skip)}')
        return df


def df_split(df, v=0.2):
    val_idx = [False if i < int(len(df)*(1.0-v)) else True for i in range(len(df))]
    return val_idx


def df_split_by_date(df, date, date_col='date'):
    df = df.set_index([date_col]).copy()
    df.sort_index(inplace=True)
    df1 = df[df.index<date].copy().reset_index()
    df2 = df[df.index>=date].copy().reset_index()
    #val_idx = np.flatnonzero((df.index>=date1) & (df.index<=date2))
    return df1, df2


def df_augment(df, cols, p=0.5, k=5):
    df = df.copy()
    n = int(len(df)*p/2)
    for _ in range(0, n):
        i1, i2 = random.randint(0,n-1), random.randint(0,n-1)
        print('ii, i2:', i1, i2)
        if i1==i2: continue
        for _ in range(k):
            j = random.randint(0,len(cols)-1)
            v1, v2 = df[cols[j]].iloc[i1], df[cols[j]].iloc[i2]
            df[cols[j]].iloc[i1], df[cols[j]].iloc[i2] = v2, v1
            print(cols[j], v1, v2)
    return df


def df_downsample_bin(df, col_dep, n=1):
    df0 = df[df[col_dep]==0]
    df1 = df[df[col_dep]==1]
    df0 = df0.sample(n=n*len(df1))
    df = pd.concat([df0, df1], axis=0).copy()
    df = df.sample(frac=1)
    print(f'Downsample: {len(df)} rows')
    print(f'Downsample: {100*len(df0)/len(df):2.1f}% False')
    print(f'Downsample: {100*len(df1)/len(df):2.1f}% True\n')
    return df


def df_upsample_bin(df, col_dep):
    df1 = df[df[col_dep]==1]
    n = (df.shape[0]-df1.shape[0]) // df1.shape[0]
    dfs = [df]+[df1]*(n-1)
    df = pd.concat(dfs, axis=0).copy()
    df = df.sample(frac=1).copy()
    print(f'Upsample: {len(df)} rows')
    return df


def df_downsample_val(df, col_dep, val=0.0, k=1):
    df0 = df[df[col_dep]<=val]
    df1 = df[df[col_dep]>val]
    df0 = df0.sample(n=k*len(df1))
    df = pd.concat([df0, df1], axis=0).copy()
    df = df.sample(frac=1)
    print(f'Downsample: {len(df)} rows')
    print(f'Downsample: {100*len(df0)/len(df):2.1f}% <= {val:.1f}')
    print(f'Downsample: {100*len(df1)/len(df):2.1f}% > {val:.1f}\n')
    return df


def df_sample(df, n):
    idxs = sorted(np.random.permutation(len(df))[:n])
    return df.iloc[idxs].copy()


def df_join(left, right, left_on, right_on=None, suffix='_y'):
    if right_on is None:
        right_on = left_on
    return left.merge(right, how='left', left_on=left_on, right_on=right_on, suffixes=("", suffix))


def df_fixdates_(df):
    for c in df.columns:
        if df[c].dtype == 'object':
            try:
                df[c] = pd.to_datetime(df[c], infer_datetime_format=True)
            except:
                pass
    return None


def df_join_by_date_range(df1, df2, key, date='date', date1='date1', date2='date2', link='id'):
    df2 = df2.copy().set_index([key], drop=False).sort_values([key, date1])
    data = []
    for i1, (_, row) in enumerate(df1.iterrows()):
        print(f'{100*i1/len(df1):2.1f}%', end='\r')
        if row[key] not in df2.index:
            continue
        rows2 = df2.loc[[row[key]]]
        for _,row2 in rows2.iterrows():
            if row[date] >= row2[date1] and row[date] < row2[date2]:
                row[link] = row2[link]
                data.append(row)
                break
    return pd.DataFrame(data)


def df_get_year_historic_cols(df, N, hist_cols):
    #N = (df[date_col].max()-df[date_col].min()).days//365 + 1
    cols = []
    for c in hist_cols:
        cols += [f'{c}{i+1}' for i in range(N-1)] + [f'{c}T'] + [f'{c}N{i+1}' for i in range(N-1)] + [f'{c}NT']
    return cols


def df_build_year_historic(df, key_col, date_col, cols_hist):
    N = (df[date_col].max()-df[date_col].min()).days//365 + 1
    cols, cols2 = list(df.columns.values), []
    for c in cols_hist:
        cols2 += [f'{c}{i+1}' for i in range(N-1)] + [f'{c}T'] + [f'{c}N{i+1}' for i in range(N-1)] + [f'{c}NT']
    df = df.copy().reset_index().set_index([key_col, date_col]).sort_index(axis=0, ascending=[True, True])
    data = []
    key_ = -1
    k = 0
    for i1, ((key, date), row) in enumerate(df.iterrows()):
        print(f'{100*i1/len(df):2.1f}%', end='\r')
        row2 = row
        row2[key_col] = key
        row2[date_col] = date
        if key != key_:
            key_ = key
            date0 = date
            vals, counts = {}, {}
            for c in cols_hist:
                vals[c], counts[c] = [0]*N, [0]*N
        idx = ((date-date0).days)//365
        for c in cols_hist:
            vals[c][idx], counts[c][idx] = row[f'{c}'], row[f'{c}N']
        for c in cols_hist:
            for i in range(N):
                row2[f'{c}{i+1}'], row2[f'{c}N{i+1}'] = 0, 0
            for i in range(idx):
                idx2 = idx-i-1
                row2[f'{c}{i+1}'], row2[f'{c}N{i+1}'] = vals[c][idx2], counts[c][idx2]
            row2[f'{c}T'], row2[f'{c}NT'] = sum(vals[c][:idx]), sum(counts[c][:idx])
        data.append(row2)
    df = pd.DataFrame(data)
    df[date_col] = pd.to_datetime(df[date_col], infer_datetime_format=True)
    df = df.reset_index().drop(columns=['level_0', 'index'])
    df = df[cols+cols2]
    return df, cols2


def df_split_by_cat_cont(df, cols_cat=[], cols_cont=[]):
    for c in df.columns:
        if c in cols_cat:
            continue
        if c in cols_cont:
            continue
        if np.issubdtype(df[c].dtype, np.datetime64):
            continue
        if not is_numeric_dtype(df[c]):
            cols_cat.append(c)
            continue
        if len(df[c].unique()) <= 50:
            cols_cat.append(c)
            continue
        if len(df[c].unique()) >= len(df)*0.25:
            continue
        cols_cont.append(c)
    return cols_cat, cols_cont


def df_datepart_(df, fldname, drop=True):
    fld = df[fldname]
    fld_dtype = fld.dtype
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        fld_dtype = np.datetime64
    if not np.issubdtype(fld_dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    fldname2_pre = re.sub('[Dd]ate$', '', fldname)
    for e in DATE_PARTS:
        df[fldname2_pre+e] = getattr(fld.dt,e.lower())
    df[fldname2_pre+'Elapsed'] = fld.astype(np.int64) // 10**9
    if drop: df.drop(fldname, axis=1, inplace=True)


def df_getelapsed(df, fld, pre):
    day1 = np.timedelta64(1, 'D')
    last_date = np.datetime64()
    last_store = 0
    res = []

    for s,v,d in zip(df.Store.values,df[fld].values, df.Date.values):
        if s != last_store:
            last_date = np.datetime64()
            last_store = s
        if v: last_date = d
        res.append(((d-last_date).astype('timedelta64[D]') / day1).astype(int))
    df[pre+fld] = res


def df_is_int(df, c):
    return np.array_equal(df[c].fillna(0), df[c].fillna(0).astype(int))


def df_fixmissing_(df, col, na_dict={}):
    if is_numeric_dtype(df[col]):
        if pd.isnull(df[col]).sum() or (col in na_dict):
            df[col+'_na'] = pd.isnull(df[col])
            filler = na_dict[col] if col in na_dict else df[col].median()
            df[col] = df[col].fillna(filler)
            na_dict[col] = filler
    return na_dict


def df_proc(df, col_dep=None, drop_cols=[], ignore_cols=[], na_dict={}, max_n_cat=None):
    df = df.copy()
    ignored_cols = df.loc[:, ignore_cols]
    df.drop(ignored_cols, axis=1, inplace=True)
    return df
