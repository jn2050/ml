import torch
from lib.prep import *
from lib.data import *
from lib.transforms import *
from lib.models import *
from lib.metrics import *
from lib.learner import Learner


def retina(cfg):
    cfg.logger.text('Retinopaty Classifier')
    sz = (3, 512, 512)
    mean = [143.10659137, 138.00140398, 134.93883852]
    std = [22.11333965, 23.73411752, 16.67617039]

    bias = [
        [0.20, 0.20, 0.20, 0.20, 0.20],
        [0.20, 0.20, 0.20, 0.20, 0.20],
        [0.20, 0.20, 0.20, 0.20, 0.20],
        [0.20, 0.20, 0.20, 0.20, 0.20],
        [0.20, 0.20, 0.20, 0.20, 0.20]
    ]

    trf1 = Transforms([
        #RandomCrop(sz[1]*0.8),
        #Scale(sz[1]),
        #RandomRotate(180.0, p=0.7),
        Normalize(mean, std),
    ])

    trf2 = Transforms([
        Normalize(mean, std),
    ])

    trfy = Transforms([
        lambda y: '0' if y in ['0'] else '1'
    ])

    data = Data.image_from_file(cfg, sz=sz, trf1=trf1, trf2=trf2, trfy=None, meta_cols=['rows', 'cols'])
    data.ds1.set_balance(bias[0])

    model = ConvClassifier(3, 16, [16, 32, 64, 64, 64, 64, 64, 64, 64], data.c, p=0.50, residual=True, sz=sz, msz=2)
    opt = torch.optim.SGD(model.parameters(), cfg.lr, momentum=0.9, weight_decay=1e-4)
    crit = nn.NLLLoss()  # weight=torch.FloatTensor([0.4, 0.6])
    learner = Learner(cfg, data, model, opt=opt, crit=crit, metrics=[accuracy], name='retina_conv')

    if cfg.lr_find:
        cfg.logger.print_lr_find(learner.lr_find())
        exit(0)

    if cfg.predict is not None:
        y2 = learner.predict(cfg.predict)
        print(f'\nPredict: {cfg.predict} -> {y2}\n')
        exit(0)

    learner.fit(cfg.epochs)
    learner.close()
