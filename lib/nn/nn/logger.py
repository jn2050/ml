from __future__ import print_function
import datetime
import os
from time import sleep
from lib.metrics import *
from lib.util import *


class Logger():
    def __init__(self, path=None):
        self.logf = None
        if path is None:
            return
        path = path + '/runs'
        if not os.path.exists(path):
            os.makedirs(path)
        d = datetime.datetime.now()
        s = 'run-{:04d}{:02d}{:02d}-{:02d}{:02d}{:02d}'.format(d.year, d.month, d.day, d.hour, d.minute, d.second)
        os.makedirs(path + '/' + s)
        self.path = path
        self.logf = open(path + '/' + s + '/log.csv', "w", 1)

    def text(self, s, s2=None):
        if s2:
            s = s + ' ' + s2
        print(s)

    def textArray(self, s_array):
        for s in s_array:
            self.text(s)

    def it(self, mode, ep, it, total):
        prefix = ' Epoch: {:d} {}:'.format(ep, mode)
        printProgressBar(it, total, prefix)

    def log(self, ep, t, loss1, ms_train, loss2, ms_valid, best, maximize=True):
        s1 = ' Epoch: {}\tTime: {:02d}:{:02d}\tLoss1: {:.4f}'\
                .format(ep, int(t)//60, int(t)%60, loss1)
        s1 += '\tM: [ '
        for m in ms_train:
            s1 += '{:.4f} '.format(m)
        s1 += ']'
        s1 += '\tLoss2: {:.4f}'.format(loss2)
        s1 += '\tM: [ '
        for m in ms_valid:
            s1 += '{:.4f} '.format(m)
        s1 += ']'
        if (maximize and ms_valid[0] > best) or (not maximize and ms_valid[0] < best):
            s1 = s1 + ' *'

        s2 = s1
        #s2 = '{},{:.3f},{:.4f},{:2.2f},{:.4f},{:2.2f}'\
        #        .format(ep, t1, loss1, 100*c1/m1, loss2, 100*c2/m2)
        if self.logf is not None:
            self.logf.write(s2 + '\n')
            self.logf.flush()
        print(s1)

    def print_dl_stats(self, data):
        print('Rows in train dataset: {:,}'.format(len(data.ds1)).replace(',', '.'))
        print('Rows in valid dataset: {:,}'.format(len(data.ds2)).replace(',', '.'))
        if not data.reg:
            print('Classes #{:d}: {}'.format(data.c, data.classes))
        if data.reg:
            return
        for ds, title in zip([data.ds1, data.ds2], ['Train', 'Eval']):
            print(f'\nHistogram {title}:')
            for i in range(len(ds.hist)):
                print('Class {}: {:d} {:2.2f}%:'.format(ds.classes[i], ds.hist[i], 100*ds.hist[i]/len(ds)))

    def print_lr_find(self, results):
        print('\nLearning Rate finder:\n')
        print('LR\tLoss')
        for r in results:
            print('{:2.6f}\t{:2.4f}'.format(r[0], r[1]))

    def print_confusion(self, learner):
        m = learner.get_confusion()
        print('\nConfusion matrix:\n')
        for i, c in enumerate(learner.data.classes):
            print(c, ':\t', m[i])
        print('\nBest:', learner.best)

    def print_predict(self, x, y):
        pass

    def close(self):
        self.text('')
        if self.logf is not None:
            self.logf.close()
