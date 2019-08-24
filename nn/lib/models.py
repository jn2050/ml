import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, resnet50
from lib.util import *
# pylint: disable=E1101

NAMED_NETS = [resnet50, vgg16]

"""
    Initializers

"""

def children(m):
    return m if isinstance(m, (list, tuple)) else list(m.children())


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv2D') != -1:
        nn.init.xavier_normal_(m.weight.data, gain=1)
    elif classname == 'Linear':
        nn.init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.uniform_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

"""
    Blocks

"""


class ConvBnLayer(nn.Module):
    def __init__(self, n1, n2, kernel_size=3, stride=2, padding=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(n1, n2, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        weights_init_xavier(self.conv)
        self.batch = nn.BatchNorm2d(n2)
        weights_init_xavier(self.batch)

    def forward(self, x):
        return F.relu(self.batch(self.conv(x)))


class ConvBnLayer1D(nn.Module):
    def __init__(self, n1, n2, kernel_size=3, stride=2, padding=1, bias=True):
        super().__init__()
        self.conv = nn.Conv1d(n1, n2, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        weights_init_xavier(self.conv)
        self.batch = nn.BatchNorm1d(n2)
        weights_init_xavier(self.batch)

    def forward(self, x):
        return F.relu(self.batch(self.conv(x)))


class ConvBlock(nn.Module):
    def __init__(self, n, k=2, kernel_size=3, padding=1):
        super().__init__()
        self.layers = []
        for i in range(1, k):
            self.layers += [ConvBnLayer(n, n, kernel_size=kernel_size, stride=1, padding=padding, bias=True)]
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


class ConvBlockShift(nn.Module):
    def __init__(self, n1, n2, kernel_size=3, padding=1):
        super().__init__()
        self.layers = [ConvBnLayer(n1, n2, kernel_size=kernel_size, stride=2, padding=padding, bias=True)]
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


class ConvBlockResidual(ConvBlock):
    def forward(self, x):
        return x + super().forward(x)


class LinearBnLayer(nn.Module):
    def __init__(self, n1, n2, p=0.0, actn=nn.ReLU()):
        super().__init__()
        self.linear = nn.Linear(in_features=n1, out_features=n2)
        weights_init_xavier(self.linear)
        self.drop = nn.Dropout(p=p)
        self.batch = nn.BatchNorm1d(num_features=n2)
        weights_init_xavier(self.batch)
        self.actn = actn

    def forward(self, x):
        return self.batch(self.actn(self.drop(self.linear(x))))


class LinearBnFinalLayer(nn.Module):
    def __init__(self, n1, n2, actn=nn.LogSoftmax(dim=1)):
        super().__init__()
        self.linear = nn.Linear(in_features=n1, out_features=n2)
        weights_init_xavier(self.linear)
        self.actn = actn

    def forward(self, x):
        return self.actn(self.linear(x))


"""
    Nets

"""

class ConvNet(nn.Module):
    def __init__(self, n1, n2, layers, k=2, p=None, residual=False, sz=None):
        super(ConvNet, self).__init__()
        self.sz = sz
        batch = nn.BatchNorm2d(n1)
        weights_init_xavier(batch)
        net = [batch]
        net += [ConvBnLayer(n1, n2, kernel_size=5, stride=1, padding=2)]
        n1 = n2
        for i, n in enumerate(layers):
            net += [ConvBlockShift(n1, n)]
            net += [ConvBlockResidual(n, k=k)] if residual else [ConvBlock(n, k=k)]
            if p is not None:
                p1 = p if i > 1 else 0.0
                net += [nn.Dropout2d(p=p1)]
            n1 = n
        net += [nn.AdaptiveMaxPool2d(1)]
        net += [Flatten()]
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)

    def print(self):
        printModelStats(self.net, self.sz)


class FcNet(nn.Module):
    def __init__(self, n1, szs, c, ps=[]):
        super(FcNet, self).__init__()
        if len(ps) < len(szs):
            ps = [0.0] * len(szs)
        net = [LinearBnLayer(n1, szs[0], p=0.0)]
        for i in range(len(szs)-1):
            net += [LinearBnLayer(szs[i], szs[i+1], p=ps[i])]
        net += [LinearBnFinalLayer(szs[-1], c)]
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class ConvClassifier(nn.Module):
    def __init__(self, n1, n2, cnn_layers, c, k=2, p=None, residual=False, sz=None, msz=0):
        super(ConvClassifier, self).__init__()
        self.conv = ConvNet(n1, n2, cnn_layers, p=p, residual=residual, sz=sz)
        self.fc = LinearBnFinalLayer(cnn_layers[-1]+msz, c)
        self.msz = msz

    def forward(self, *x):
        if len(x)==1:
            return self.fc(self.conv(x))
        x, m = x
        x = self.conv(x)
        x = torch.cat((x, m), 1)
        return self.fc(x)


class ConvMetaClassifier(nn.Module):
    def __init__(self, n1, n2, cnn_layers, c, k=2, p=None, residual=False, sz=None):
        super(ConvMetaClassifier, self).__init__()
        self.conv = ConvNet(n1, n2, cnn_layers, p=p, residual=residual, sz=sz)
        self.fc = LinearBnFinalLayer(cnn_layers[-1]+2, c)

    def forward(self, x, m):
        x = self.conv(x)
        x = torch.cat((x, m), 1)
        x = self.fc(x)
        return x


class ConvNamedClassifier(nn.Module):
    def __init__(self, model, c, sz=None, xfc=[], ps=None, pretrained=True, multi=False):
        super(ConvNamedClassifier, self).__init__()
        if not isinstance(ps, list):
            ps = [ps] * (len(xfc) + 1)
        cnn = model(pretrained=pretrained) if model in NAMED_NETS else  model()
        if list(cnn.modules())[-1].__class__.__name__ == 'Linear':
            print('Poping last linear layer')
            cnn = nn.Sequential(*list((cnn.children()))[:-1])
        cnn = [cnn] + [Flatten()]  # AdaptiveConcatPool2d()
        self.cnn = nn.Sequential(*cnn)
        n = get_cnn_size(self.cnn, sz)
        fc = []
        n1 = n
        for i, n2 in enumerate(xfc):
            fc += [LinearBnLayer(n1, n2, p=ps[i], actn=nn.ReLU())]
            n1 = n2
        final_actn = nn.Sigmoid() if multi else nn.LogSoftmax(dim=1)
        fc += [LinearBnLayer(n1, c, p=ps[i], actn=final_actn)]
        self.fc = nn.Sequential(*fc)

        self.net = nn.Sequential(*(cnn + fc))
        printModelStats(self.net, sz)

    def forward(self, x):
        return self.net(x)


class TabNet(nn.Module):
    def __init__(self, data, out_sz, szs, emb_p, ps, reg=True, binary=False):
        super().__init__()
        self.data = data
        self.reg = reg
        self.binary = binary
        self.ncats = data.ncats
        self.nconts = data.nconts
        #cat_sz = [(c, len(data.tab.d[c])+1) for c in data.cols_cat]
        cat_sz = [(c, data.cat_sizes[c]+1) for c in data.cols_cat]
        emb_sz = [(c, min(50, (c+1)//2)) for _,c in cat_sz]
        self.embs = nn.ModuleList([nn.Embedding(c, s) for c,s in emb_sz])
        for emb in self.embs:
            x = emb.weight.data
            sc = 2 / (x.size(1) + 1)
            x.uniform_(-sc, sc)
        self.nemb = sum(e.embedding_dim for e in self.embs)
        self.emb_drop = nn.Dropout(emb_p)
        szs = [self.nemb+self.nconts] + szs
        self.lins = nn.ModuleList([nn.Linear(szs[i], szs[i+1]) for i in range(len(szs)-1)])
        for l in self.lins:
            nn.init.kaiming_normal_(l.weight.data)
        self.bns = nn.ModuleList([nn.BatchNorm1d(sz) for sz in szs[1:]])
        for b in self.bns:
            weights_init_xavier(b)
        if not reg:
            out_sz = self.data.c
        self.outp = nn.Linear(szs[-1], out_sz)
        nn.init.kaiming_normal_(self.outp.weight.data)
        self.emb_drop = nn.Dropout(emb_p)
        self.drops = nn.ModuleList([nn.Dropout(drop) for drop in ps])
        self.bn = nn.BatchNorm1d(self.nemb+self.nconts)
        weights_init_xavier(self.bn)
        p,i,c,e = sum([p.numel() for p in self.parameters()]), self.ncats+self.nconts, self.ncats, self.nemb
        print(f'\nTabNet with {p:,.0f} params: {i:,.0f}->{i-c+e:,.0f} inputs and {c:,.0f}->{e:,.0f} embeds'.replace(',', '.'))

    def forward(self, x):
        cats = x[:, :self.ncats].long() if self.ncats!=0 else []
        conts = x[:, self.ncats:self.ncats+self.nconts].contiguous()
        if self.nemb!=0:
            x = [e(cats[:,i]) for i,e in enumerate(self.embs)]
            x = torch.cat(x, 1)
            x = self.emb_drop(x)
        if self.nconts != 0:
            #x2 = self.bn(conts)
            x = torch.cat([x, conts], 1) if self.nemb!=0 else conts
        x = self.bn(x)
        for l,d,b in zip(self.lins, self.drops, self.bns):
            x = F.relu(l(x))
            x = d(x)
            x = b(x)
        x = self.outp(x)
        if self.reg:
            x = F.sigmoid(x)
            if not self.binary:
                x = x*(self.data.y_range[1]-self.data.y_range[0])+self.data.y_range[0]
            x = torch.squeeze(x)
        else:
            x = F.log_softmax(x, dim=1)
        return x


class EmbeddingNet(nn.Module):
    def __init__(self, data, n_factors, nh=10, p1=0.05, p2=0.5):
        super().__init__()
        self.min_rating, self.max_rating = data.min_rating, data.max_rating
        self.u = nn.Embedding(data.n_users, n_factors)
        self.u.weight.data.uniform_(-0.05, 0.05)
        self.i = nn.Embedding(data.n_items, n_factors)
        self.i.weight.data.uniform_(-0.05, 0.05)
        self.lin1 = nn.Linear(n_factors * 2, nh)
        self.lin2 = nn.Linear(nh, 1)
        self.drop1 = nn.Dropout(p1)
        self.drop2 = nn.Dropout(p2)

    def forward(self, x):
        users, items = x[:,0], x[:,1]
        x = torch.cat([self.u(users), self.i(items)], dim=1)
        x = self.drop1(x)
        x = self.drop2(F.relu(self.lin1(x)))
        x = F.sigmoid(self.lin2(x))
        x = x * float(self.max_rating - self.min_rating + 1) + float(self.min_rating - 0.5)
        return x


class EEG_Net(nn.Module):
    def __init__(self, n1, n2, layers, c, p=None):
        super(EEG_Net, self).__init__()
        batch = nn.BatchNorm1d(num_features=n1)
        weights_init_xavier(batch)
        net = [batch]
        #net += [nn.Dropout(p=p/10)] if p is not None else [nn.Dropout(p=0.0)]
        net += [ConvBnLayer1D(n1, n2, kernel_size=5, stride=2, padding=2)]
        n1 = n2
        for i, n in enumerate(layers):
            net += [ConvBnLayer1D(n1, n, kernel_size=5, stride=2, padding=2)]
            #net += [nn.Dropout(p=p/2)] if p is not None else [nn.Dropout(p=0.0)]
            n1 = n
        self.conv = nn.Sequential(*net)
        net += [nn.AdaptiveMaxPool1d(1)]
        net += [Flatten()]
        #net += [nn.Dropout(p=p)] if p is not None else [nn.Dropout(p=0.0)]
        net += [LinearBnLayer(layers[-1], 1024)]
        net += [nn.Dropout(p=p)] if p is not None else [nn.Dropout(p=0.0)]
        net += [LinearBnFinalLayer(1024, c)]
        self.net = nn.Sequential(*net)
        p = sum([p.numel() for p in self.parameters()])
        print(f'\nEEG_Net: {p:,.0f} params'.replace(',', '.'))

    def forward(self, x):
        #x = self.conv(x); print(x.size()); exit(0)
        return self.net(x)