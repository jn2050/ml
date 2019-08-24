import pandas as pd
import torch
from lib.data import *
from lib.models import *
from lib.learner import Learner
from lib.metrics import *


"""
    colab
    DF columns: userId, movieId, rating, timestamp

"""


def colab(cfg):
    df = pd.read_csv(os.path.join(cfg.path, 'ratings.csv'))
    df = df.sample(frac=1)
    if not cfg.use_cuda:
        df = df.sample(1024)

    data = ColabData(df, bs=1024)
    model = EmbeddingNet(data, 50)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-5)
    crit = nn.MSELoss()
    learner = Learner(cfg, data, model, opt=opt, crit=crit, metrics=[accuracy2], maximize=True, load=cfg.use_cuda)
    learner.fit(cfg.epochs)

    y, y2, _ = learner.epoch(0)
    a = accuracy2(y2, y)
    y2 = torch.round(y2 * 2) / 2
    for i in range(y.shape[0]):
        print('{:.1f} {:.1f} ->   {:d}'.format(y[i][0], y2[i][0], y[i][0]==y2[i][0]))
    print(a)
