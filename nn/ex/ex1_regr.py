from itertools import count
import torch
import torch.autograd
import torch.nn.functional as F
from torch.autograd import Variable


POLY_DEGREE = 4
W_target = torch.randn(POLY_DEGREE, 1) * 5
b_target = torch.randn(1) * 5


def make_features(x):
    x = x.unsqueeze(1)
    x = torch.cat([x ** i for i in range(1, POLY_DEGREE+1)], 1)
    return x


def f(x):
    return x.mm(W_target) + b_target[0]


def poly_desc(W, b):
    result = 'y = '
    for i, w in enumerate(W):
        result += '{:+.2f} x^{} '.format(w, len(W) - i)
    result += '{:+.2f}'.format(b[0])
    return result


def get_batch(bs=32):
    random = torch.randn(bs)
    x = make_features(random)
    y = f(x)
    return Variable(x), Variable(y)


def regression():
    print('Regressison')

    fc = torch.nn.Linear(W_target.size(0), 1)

    for it in count(1):
        x, y = get_batch()
        fc.zero_grad()
        y2 = fc(x)

        loss_var = F.smooth_l1_loss(y2, y)
        #loss_var = F.mse_loss(y2, y)
        #loss_var = F.l1_loss_(y2, y)

        loss = loss_var.data[0]
        loss_var.backward()
        for param in fc.parameters():
            param.data.add_(-0.1 * param.grad.data)
        if loss < 1e-3:
            break

    print('Loss: {:.6f} after {} batches'.format(loss, it))
    print('==> Learned function:\t' + poly_desc(fc.weight.data.view(-1), fc.bias.data))
    print('==> Actual function:\t' + poly_desc(W_target.view(-1), b_target))


regression()

