import torch
from lib.learner import Learner
from lib.data import Data
from lib.transforms import Transforms, Normalize
from lib.models import FcNet
from lib.metrics import accuracy
# pylint: disable=E1101


"""
    Best 2017: 0.9928
    Best 2018: 0.9533
"""

def rouen(cfg):
    cfg.logger.text('Rouen Audio Classifier')

    trf = Transforms([
        Normalize(0.326709, 0.457423),
    ])

    data = Data.numpy_from_file(cfg, trf1=None, trf2=None)
    model = FcNet(512, [512, 512], data.c, [0.7, 0.7])
    opt = torch.optim.SGD(model.parameters(), cfg.lr, momentum=0.9, weight_decay=1e-4)
    crit = torch.nn.NLLLoss()
    learner = Learner(cfg, data, model, opt=opt, crit=crit, metrics=[accuracy])

    if cfg.lr_find:
        cfg.logger.print_lr_find(learner.lr_find())
        exit(0)

    if cfg.predict is not None:
        x = data.pre(cfg.predict)
        y2 = data.pos(learner.predict(x))
        print(f'\nPredict: {cfg.predict} -> {y2}\n')
        exit(0)

    learner.fit(cfg.epochs)
    learner.close()
