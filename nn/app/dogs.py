from torchvision import transforms
from lib.data import Data
from lib.models import ConvNet, ConvClassifier
from lib.metrics import accuracy
from lib.learner import Learner
import torch
import torch.nn as nn


def dogs(cfg):
    cfg.logger.text('Dogs and Cats Image Classifier')
    sz = (3, 224, 224)
    #mean = [0.485, 0.456, 0.406]
    #std = [0.229, 0.224, 0.225]

    trf1 = transforms.Compose([
        transforms.Resize((sz[1],sz[2])),
        transforms.RandomResizedCrop(sz[1], scale=(0.8,1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize(mean=mean, std=std)
    ])

    trf2 = transforms.Compose([
        transforms.Resize((sz[1],sz[2])),
        transforms.ToTensor(),
        #transforms.Normalize(mean=mean, std=std)
    ])

    data = Data.image_from_file(cfg, sz=sz, trf1=trf1, trf2=trf2)
    #model = ConvClassifier(resnet50, data.nclasses, sz, xfc=[512])
    model = ConvNet(3, 16, [16, 32, 64, 128, 256, 256, 512, 1024], data.c, p=0.4, residual=True, sz=sz)
    opt = torch.optim.SGD(model.parameters(), cfg.lr, momentum=0.9, weight_decay=1e-4)
    crit = nn.NLLLoss()
    learner = Learner(cfg, data, model, opt=opt, crit=crit, metrics=[accuracy], name='dogs_conv')
    learner.fit(cfg.epochs)
    learner.close()
