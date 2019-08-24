import pandas as pd
import numpy as np
import os, sys
sys.path.append('.')
from lib.prep import *
from lib.structured import *


calc_vars = ['ps_calc_04', 'ps_calc_05', 'ps_calc_06', 'ps_calc_07', 'ps_calc_08', 'ps_calc_09', 'ps_calc_10',
    'ps_calc_11', 'ps_calc_12', 'ps_calc_13', 'ps_calc_14', 'ps_calc_15_bin', 'ps_calc_16_bin', 'ps_calc_17_bin',
    'ps_calc_18_bin', 'ps_calc_19_bin', 'ps_calc_20_bin', 'ps_calc_01', 'ps_calc_02', 'ps_calc_03']
cat_vars = ['ps_ind_01', 'ps_ind_02_cat', 'ps_ind_03','ps_ind_04_cat', 'ps_ind_05_cat', 'ps_ind_14',
    'ps_ind_15', 'ps_car_01_cat', 'ps_car_02_cat', 'ps_car_03_cat', 'ps_car_04_cat', 'ps_car_05_cat', 'ps_car_06_cat',
    'ps_car_07_cat', 'ps_car_08_cat', 'ps_car_09_cat','ps_car_10_cat', 'ps_car_11_cat', 'ps_car_11']
cont_vars = ['ps_reg_01', 'ps_reg_02', 'ps_reg_03', 'ps_car_12', 'ps_car_13','ps_car_14', 'ps_car_15', 'ps_ind_06_bin', 'ps_ind_07_bin',
    'ps_ind_08_bin', 'ps_ind_09_bin', 'ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin', 'ps_calc_01',
    'ps_calc_02', 'ps_calc_03']
dep_var = 'target'


def downsample(df):
    df0 = df[df[dep_var]==0]
    df1 = df[df[dep_var]==1]
    df0 = df_sample(df0, len(df1))
    df = pd.concat([df0, df1], axis=0).copy()
    df = df.sample(frac=1)
    df = df.reset_index()
    df.to_feather(os.path.join(path,'auto_downsampled.feather'))


def upsample(df):
    df1 = df[df[dep_var]==1]
    n = (df.shape[0]-df1.shape[0]) // df1.shape[0]
    dfs = [df]+[df1]*(n-1)
    df = pd.concat(dfs, axis=0).copy()
    df = df.sample(frac=1)
    df = df.reset_index() 
    df.to_feather(os.path.join(path,'auto_upsampled.feather'))



def sample(df):
    df = df.sample(frac=1)
    df = df.reset_index() 
    df.to_feather(os.path.join(path,'auto.feather'))


path = os.path.join(os.environ.get('DATA_PATH'), 'auto')
#df = pd.read_csv(os.path.join(path, 'auto.csv'))
#df = df[cat_vars + cont_vars + [dep_var]]

cat_vars = ['Col1', 'Col2']
cont_vars = ['Col3']
example_data = {
    'Col1': ['A', 'B', 'A', 'D', 'D', 'C'],
    'Col2': [7, 8, 9, 10, 11, 12],
    'Col3': [2.0, 4.0, 2.0, 4.0, 2.0, 4.0]
}
df = pd.DataFrame.from_dict(example_data)
td = TabData(df, cat_vars, cont_vars, []).process()
#df = df.replace(-1, np.NaN)
#df = df_sample(df, 10000)

sample(df)