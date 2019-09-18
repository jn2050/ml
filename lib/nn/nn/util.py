import math
import random
import os, sys
import psutil
import torch
import numpy as np
# pylint: disable=E1101
#from pprint import pprint
#pprint(vars(ds1))

#ImageFile.LOAD_TRUNCATED_IMAGES = True


def _tuple(x):
    return x if isinstance(x, (list, tuple)) else (x,x)


def rand0(s):
    return random.random()*(s*2)-s


def open_np_image(fname):
    """ Opens a numpy representing an image.

    Arguments:
        fname: the file path of the image

    Returns:
        The numpy array of floats normalized to range between 0.0 - 1.0, in RGB format.
    """
    if not os.path.exists(fname):
        raise OSError('No such file or directory: {}'.format(fname))
    elif os.path.isdir(fname):
        raise OSError('Is a directory: {}'.format(fname))
    else:
        try:
            x = np.load(fname)
            return x
        except Exception as e:
            raise OSError('Error reading numpy image at: {}'.format(fname)) from e


def save_np_image(fname, x):
    """ Saves a numpy representing an image.

    Arguments:
        fname: the file path of the image
        x: the numpy array of floats normalized to range between 0.0 - 1.0, in RGB format

    Raises:
        OSError if the array can not be saved
    """
    if not os.path.exists(fname):
        raise OSError('No such file or directory: {}'.format(fname))
    elif os.path.isdir(fname):
        raise OSError('Is a directory: {}'.format(fname))
    else:
        try:
            np.save(fname, x)
        except Exception as e:
            raise OSError('Error reading numpy image at: {}'.format(fname)) from e


def get_n_params(model):
    total = 0
    for p in list(model.parameters()):
        n = 1
        for s in list(p.size()):
            n = n * s
        total += n
    return total


def get_cnn_size(net, sz):
    if sz is None:
        return 0
    C0, H0, W0 = sz
    C1, H1, W1 = C0, H0, W0
    for name, l in net.named_modules():
        classname = l.__class__.__name__
        # To avoid Resnet downsample blocks double effect
        if classname == 'Conv2d' and _tuple(l.kernel_size) == (1, 1) and 'downsample' in name:
            continue
        if classname == 'Conv2d':
            C1 = l.out_channels
        if classname == 'MaxPool2d':
            C1 = C0
        if classname == 'Conv2d' or classname == 'MaxPool2d':
            stride = _tuple(l.stride)
            padding = _tuple(l.padding)
            kernel_size = _tuple(l.kernel_size)
            dilation = _tuple(l.dilation)
            H1 = int(math.floor((H0 + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1))
            W1 = int(math.floor((W0 + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1))
            #print(C1, H1, W1)
        if classname == 'AvgPool2d':
            stride = _tuple(l.stride)
            padding = _tuple(l.padding)
            kernel_size = _tuple(l.kernel_size)
            C1 = C0
            H1 = int(math.floor((H0 + 2 * padding[0] - kernel_size[0]) / stride[0] + 1))
            W1 = int(math.floor((W0 + 2 * padding[1] - kernel_size[1]) / stride[1] + 1))
            #print(C1, H1, W1)
        C0, H0, W0 = C1, H1, W1
    return C1 * H1 * W1


def printModelStats(model, sz):
    if sz is not None:
        n = get_cnn_size(model, sz)
        print('Model features in last conv layer: {:,}'.format(n).replace(',', '.'))
    print('Model parameters: {:,}'.format(get_n_params(model)).replace(',', '.'))


def denorm(tensor):
    for i in range(tensor.size()[0]):
        for t, m, s in zip(tensor[i], [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]):
            t.mul_(s).add_(m).clamp_(0, 1)
    return tensor


def save_model(model, fname, best=0):
    state = {
        'state_dict': model.state_dict(),
        'accuracy': best
    }
    print('=> saving model {}, M:{:.4f}'.format(fname, best))
    torch.save(state, fname)


def load_model(model, fname):
    print('\n=> loading model {}'.format(fname))
    state = torch.load(fname,  map_location='cpu')
    model.load_state_dict(state['state_dict'])
    best = state['accuracy']
    print('Best so far: {:.4f}'.format(best))
    return best


def rm_model(learner):
    os.remove(learner.file)
    print('Best file removed')

#
# def accuracy_np(preds, targs):
#     preds = np.argmax(preds, 1)
#     return (preds==targs).mean()
#
#
# def accuracy(preds, targs):
#     preds = torch.max(preds, dim=1)[1]
#     return (preds==targs).float().mean()


def printProgressBar (iteration, total, prefix, length=80, fill='█'):
    percent = '{0:.1f}'.format(100 * (iteration / float(total)))
    filled = int(length * iteration // total)
    bar = fill * filled + '-' * (length - filled)
    if iteration < total - 1:
        print('\r%s |%s| %s%% ' % (prefix, bar, percent), end='\r')
    else:
        bar = ' ' * (length + len(prefix) + 12)
        print('\r%s' % bar, end='\r')


def biased_roll(dist):
    r = random.random()
    cum, result = 0, 0
    for d in dist:
        cum += d
        if r < cum:
            return result
        result += 1


def hStack(img1, img2):
    return np.hstack((img1, img2))


# Revert: x = np.moveaxis(x, 0, 2)


# class BiasedGetter:
#     def __init__(self, samples, n): # Called every epoch
#         samples = torch.LongTensor(samples)
#         samples = samples[torch.randperm(n)]
#         self.samples = samples
#         self.n = n
#         self.current = 0

#     def __iter__(self):
#         return self

#     def __next__(self):
#         if self.current == self.n:
#             raise StopIteration
#         else:
#             self.current += 1
#             return self.samples[self.current - 1]


# class wSampler(Sampler):

#     def __init__(self, samples, n):
#         super(wSampler, self).__init__(samples)
#         self.samples = samples
#         self.n = n

#     def __iter__(self):
#         return iter(BiasedGetter(self.samples, self.n))
#         #return iter(range(self.n))
#         #return iter(torch.randperm(len(self.data_source)).long())

#     def __len__(self):
#         return self.n

def cpuStats():
        print(sys.version)
        print(psutil.cpu_percent())
        print(psutil.virtual_memory())  # physical memory usage
        pid = os.getpid()
        py = psutil.Process(pid)
        memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
        print('memory GB:', memoryUse)


#trf = Transforms([lambda x: x.reshape(-1)])

#ImageFile.LOAD_TRUNCATED_IMAGES = True


def calcNorm(ds):
    x = [e[0] for e in list(ds)]
    x = torch.stack(x)
    mean = []
    std = []
    for i in range(3):
        pixels = x[:, i, :, :] #.ravel()
        mean.append(pixels.mean())
        std.append(pixels.std())
    print('Mean:', mean)
    print('STD:', std)


def sys_info():
    print('')
    print(f'Pytorch version: {str(torch.__version__)}')
    print(f'Pytorch cuDNN version: {str(torch.backends.cudnn.version())}')
    print('')