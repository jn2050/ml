import torch
import os, argparse
from lib.logger import Logger
from lib.util import *
# from lib.util import cudaDevices
# pylint: disable=E1101


class Object(object):
   pass


class Config():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='NN options')
        self.initialized = False
        self.cfg = None

    def initialize(self):
        self.parser.add_argument('dir', help='data directory relative to $DATA_PATH')
        self.parser.add_argument('-e', type=int, default=20, help='number of epochs')
        self.parser.add_argument('-b', type=int, default=64, help='input batch size')
        self.parser.add_argument('--gpu', type=int, default=0, help='gpu id to be used')
        self.parser.add_argument('--norm', help='get data norm',  action='store_true')
        self.parser.add_argument("--rm", help='rm best wights file', action='store_true')
        self.parser.add_argument("--eval", help='eval', action='store_true')
        self.parser.add_argument('--predict', type=str, help='predict file')
        self.parser.add_argument("--lr_find", help='learning rate finder', action='store_true')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2; use -1 for CPU')
        self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
        self.parser.add_argument('--imgSize', type=int, default=224, help='scale images to this size')
        self.parser.add_argument('--lr', type=float, default=0.01, help='lr')
        self.parser.add_argument('--lr1', type=float, default=0.001, help='lr low limit')
        self.parser.add_argument('--lr2', type=float, default=0.01, help='lr high limit')
        self.parser.add_argument('--cycle', type=int, default=0, help='lr cycle lenght')
        self.parser.add_argument('--cycle_mult', type=int, default=2, help='lr cycle multiple')
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.cfg = self.parser.parse_args()
        self.cfg.path = os.path.join(os.environ.get('DATA_PATH'), self.cfg.dir)
        self.cfg.bs = self.cfg.b
        self.cfg.epochs = self.cfg.e
        self.cfg.workers = 8 if torch.cuda.is_available() else 1
        self.cfg.device = torch.device('cuda:'+str(self.cfg.gpu) if torch.cuda.is_available() else 'cpu')
        self.cfg.logger = Logger(self.cfg.path)
        self.info()
        return self.cfg

    def init(self, path, bs=64):
        self.cfg = Object()
        self.cfg.path =  os.path.join(os.environ.get('DATA_PATH'), path)
        self.cfg.logger = Logger(path)
        self.cfg.epochs = 20
        self.cfg.bs = bs
        self.cfg.lr = 0.01
        self.cfg.lr1 = 0.01
        self.cfg.lr2 = 0.001
        self.cfg.cycle = 0
        self.cfg.cycle_mult = 2
        self.cfg.gpu = 0
        self.cfg.device = torch.device('cuda:'+str(self.cfg.gpu) if torch.cuda.is_available() else 'cpu')
        self.cfg.lr_find = False
        return self.cfg

    def info(self):
        sys_info()
        self.cfg.logger.text('Data path:', self.cfg.path)
        self.cfg.logger.text('Device:', str(self.cfg.device))
        print('')
