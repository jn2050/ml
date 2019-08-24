import torch
import pandas as pd
from lib.data import *
from lib.transforms import *
from lib.models import *
from lib.metrics import *
from lib.learner import *
from lib.util import *
from lib.prep import *
import pickle
# pylint: disable=E1101


class GM:
    def __init__(self, df1, df2):
        self.df1, self.df2 = df1, df2
        self.B1 = df1['premio1'].sum() - df1['D'].sum()
        self.B2 = df2['premio1'].sum() - df2['D'].sum()

    def __call__(self, y2, y):
        train_mode = True if y2.size()[0]==self.df1.shape[0] else False
        if train_mode: return 0.0
        pred = y2.cpu().numpy().copy()
        df = self.df1 if train_mode else self.df2
        B = self.B1 if train_mode else self.B2
        TT, gain = np.arange(0.0, 1.0, 0.01), []
        for T in TT:
            r = pred>T
            C, D = df[r==False]['premio1'].sum(), df[r==False]['D'].sum()
            if np.isnan(C) or np.isnan(D): C, D = 0.0, 0.0
            gain.append((C-D)-B)
        print(C-D)
        print(B)
        return np.array(gain).max()


def caravela(cfg):
    df1 = pd.read_feather(os.path.join(cfg.path, 'apolices_proc_train.feather'))
    df2 = pd.read_feather(os.path.join(cfg.path, 'apolices_proc_valid.feather'))
    cols_skip = pickle.load(open(os.path.join(cfg.path, 'cols_dict.pickle'), 'rb'))['cols_skip']
    gm = GM(df1, df2)

    data = TabData(cfg, df1, df2, cols_skip=cols_skip, bs=1024, reg=True, binary=True, switch_p=0.0, switch_k=20)
    model = TabNet(data, 1, [512,512,512], 0.05, [0.5,0.5,0.5], reg=True, binary=True)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=5e-5)
    crit = torch.nn.BCELoss() # torch.nn.MSELoss()
    learner = Learner(cfg, data, model, opt=opt, crit=crit, metrics=[gm, gini]) # accuracy2, precision2, recall2, f2

    if cfg.eval:
        x = df2.iloc[:,:-1].values
        x = torch.from_numpy(x).float()
        y2 = learner.predict(x, unsqueeze=False)
        df2['pred'] = y2.cpu().numpy()
        print(df2.shape)
        df2.reset_index().to_feather(os.path.join(cfg.path, 'apolices_results.feather'))
        exit(0)

    for b in [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]:
        data.ds1.set_balance(b)
        learner.fit(1)
    learner.fit(50)
    #learner.fit(cfg.epochs)

    learner.close()