import torch
import numpy as np
import os, time
from lib.metrics import *
from lib.util import *
# pylint: disable=E1101


class Learner():

    def __init__(self, cfg, data, model, opt=None, crit=None, metrics=[], maximize=True, callbacks=[], load=True, name=None):
        self.cfg = cfg
        self.logger = cfg.logger
        self.data = data
        self.model = model.to(cfg.device)
        self.opt = opt
        self.crit = crit.to(cfg.device)
        self.metrics = [accuracy] if len(metrics) == 0 else metrics
        self.maximize = maximize
        self.callbacks = callbacks
        if not os.path.exists(os.path.join(cfg.path, 'models')):
            os.makedirs(os.path.join(cfg.path, 'models'))
        if name is None:
            name = str(model.__class__.__name__)
        self.file = cfg.path + '/models/model_' + name + '_best.pth.tar'
        self.lr = self.cfg.lr
        self.lr1 = self.cfg.lr1
        self.lr2 = self.cfg.lr2
        self.lrs = [cfg.lr1, cfg.lr2]
        self.cycle = self.cfg.cycle
        self.cycle_mult = self.cfg.cycle_mult
        self.cycle_factor = 1
        self.cycle_ep = 0
        self.scheduler = None
        if self.cfg.cycle:
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.opt, self.update_lr)
        self.best = 0.0 if self.maximize else 1e9
        if load and os.path.isfile(self.file):
            self.best = load_model(self.model, self.file)

    def set_lr(self, lr):
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=5e-5)
        #torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, lr1, last_epoch=-1)

    def set_lrs(self, lrs):
        if not isinstance(lrs, list):
            lrs = [lrs] * 2
        self.lrs = lrs
        if hasattr(self.model, 'blocks') and len(self.model.blocks) > 1:
            params = []
            for i, block in enumerate(self.model.blocks):
                params += [{'params': block.parameters(), 'lr': self.lrs[i]}]
            self.opt = torch.optim.SGD(params, self.lrs[0], momentum=0.9, weight_decay=1e-4)
        else:
            self.opt = torch.optim.SGD(self.model.parameters(), self.lrs[0], momentum=0.9, weight_decay=1e-4)

    def update_lr(self, n):
        b = len(self.data.ds1)
        ep = n//b
        it = n-ep*b
        cycle = self.cycle * self.cycle_factor
        ep1 = ep - self.cycle_ep
        if ep > 0 and ep1 % cycle == 0 and it == 0:
            self.cycle_ep = ep
            self.cycle_factor += 1
            if self.cycle_factor > self.cycle_mult:
                self.cycle_factor = 1
        total_it = cycle * b
        it1 = float((ep1*b + it) % total_it)
        r = (np.cos(np.pi * (it1/total_it)) + 1) / 2
        lr = r * self.lr
        #print('lr: {:d}:{:d} -> {:1.6f}\n'.format(ep, it, lr))
        return lr

    def lr_find(self, lr1=1e-10, lr2=10.0):
        self.model.train()
        lr = lr1
        results = []

        for x, y in self.data.dl1:
            if lr > lr2:
                break
            for param_group in self.opt.param_groups:
                param_group['lr'] = lr
            x = x.to(self.cfg.device).requires_grad_()
            y = y.to(self.cfg.device)
            y2 = self.model(x)
            loss = self.crit(y2, y)
            loss_val = loss.item()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            results.append((lr, loss_val))
            lr *= 2

        return results

    def fit(self, eps, verbose=True):
        if eps > 0 and verbose:
            self.logger.text('\nStarting train for {:d} epochs:'.format(eps))
        if len(self.data.dl1) == 0:
            print('No train data')
            return
        if len(self.data.dl2) == 0:
            print('No eval data')
            return
        for ep in range(eps):
            t = time.time()
            y, y2, loss1 = self.epoch(ep, train=True)
            ms_train = [metric(y2, y) for metric in self.metrics]
            with torch.no_grad():
                y, y2, loss2 = self.epoch(ep, train=False)
            ms_valid = [metric(y2, y) for metric in self.metrics]
            if (self.maximize and ms_valid[0] > self.best) or (not self.maximize and ms_valid[0] < self.best):
                self.best = ms_valid[0]
                save_model(self.model, self.file, self.best)
            t = time.time() - t
            self.logger.log(ep, t, loss1, ms_train, loss2, ms_valid, self.best, self.maximize)
        if not self.data.reg and verbose:
            self.logger.print_confusion(self)

    def epoch(self, ep, train=False):
        if train:
            self.model.train()
            dl = self.data.dl1
        else:
            self.model.eval()
            dl = self.data.dl2
        total_loss = 0.0

        for i, (x, y) in enumerate(dl):
            if(x.size()[0]<=1):
                continue
            if isinstance(x, list):
                x[0] = x[0].to(self.cfg.device).requires_grad_()
                x[1] = x[1].to(self.cfg.device).requires_grad_()
            else:
                x = x.to(self.cfg.device).requires_grad_()
            #y = torch.squeeze(y)
            y = y.to(self.cfg.device)

            if isinstance(x, list):
                y2 = self.model(x[0], x[1])
            else:
                y2 = self.model(x)
            loss = self.crit(y2, y)
            total_loss += loss.item()
            if train:
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
            if i == 0:
                y_ep = y.detach().clone()
                y2_ep = y2.detach().clone()
            else:
                y_ep = torch.cat((y_ep,  y.detach().clone()), 0)
                y2_ep = torch.cat((y2_ep,  y2.detach().clone()), 0)

            if self.cfg.cycle:
                self.scheduler.step()
            self.cfg.logger.it('train' if train else 'eval', ep, i, len(dl))
            for callback in self.callbacks:
                callback('train' if train else 'eval', ep, i, len(dl))

        return y_ep, y2_ep, total_loss/(i+1)

    def predict(self, x, unsqueeze=True):
        if unsqueeze:
            x.unsqueeze(dim=0)
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.cfg.device)
            y = self.model(x)
        return y
    
    def eval(self):
        with torch.no_grad():
            y, y2, loss = self.epoch(0, train=False)
        metrics = [metric(y2, y) for metric in self.metrics]
        return y2, y, loss, metrics

    def get_confusion(self):
        with torch.no_grad():
            y, y2, _ = self.epoch(0, train=False)
        y2 = y2.max(1, keepdim=True)[1]
        m = np.zeros((self.data.c, self.data.c), dtype=int)
        for i in range(y2.size()[0]):
            m[int(y[i])][int(y2[i].item())] += 1
        return m

    def close(self):
        self.logger.close()
