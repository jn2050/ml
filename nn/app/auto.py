import torch
import pandas as pd
from lib.prep import *
from lib.data import *
from lib.transforms import *
from lib.models import *
from lib.metrics import *
from lib.learner import *
from lib.util import *
# pylint: disable=E1101


"""
    Down sampled: 0.2706
    Upsampled:    0.2793 (0.2919)
"""


def auto(cfg):
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

    df = pd.read_feather(os.path.join(cfg.path, 'auto.feather'))  # 'auto_downsampled.feather', 'auto_upsampled.feather'
    df = df[cat_vars + cont_vars + [dep_var]]
    val_idx = df_split(df)

    data = StructData(cfg, df, val_idx, cat_vars, cont_vars, dep_var, bs=1024, reg=True, binary=True)
    model = StructNet(data, 1, [512,512,128], 0.1, [0.6,0.6,0.6], reg=True, binary=True)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=5e-5)
    crit = torch.nn.BCELoss()
    learner = Learner(cfg, data, model, opt=opt, crit=crit, metrics=[gini, accuracy2, precision2, recall2], name='auto')

    if cfg.lr_find:
        cfg.logger.print_lr_find(learner.lr_find())
        exit(0)

    if cfg.eval:
        y2, y, loss, metrics = learner.eval()
        print(metrics)
        exit(0)

    if cfg.predict is not None:
        y2 = learner.predict(cfg.predict)
        print(f'\nPredict: {cfg.predict} -> {y2}\n')
        exit(0)

    if cfg.rm:
        rm_model(learner)
        exit(0)

    learner.fit(cfg.epochs)
    learner.close()
