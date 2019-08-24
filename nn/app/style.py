from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import scipy
import copy
import os, glob
import cv2


def img_load(fname, loader):
    img = Image.open(fname)
    img = Variable(loader(img))
    img = img.unsqueeze(0)
    return img


def img_save(tensor, fname, unloader, sz):
    img = tensor.clone().data.cpu()
    img = img.view(3, sz, sz)
    img = unloader(img)
    scipy.misc.imsave(fname, img)


def img_files_resize(files, sz):
    for f in files:
        print(f)
        img = cv2.imread(f)
        img = cv2.resize(img, (sz, sz))
        cv2.imwrite(f, img)


class GramMatrix(nn.Module):
    def forward(self, input):
        a, b, c, d = input.size()
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t())
        # 'normalize' the values of the gram matrix
        return G.div(a * b * c * d)


class ContentLoss(nn.Module):

    def __init__(self, target, weight):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        self.target = target.detach() * weight
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.weight = weight
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.loss = self.criterion(input * self.weight, self.target)
        self.output = input
        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss


class StyleTransfer(object):
    def __init__(self, cfg, content, style):
        super(StyleTransfer, self).__init__()
        self.cfg = cfg
        self.content = content
        self.style = style
        dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        #self.out = content_img.clone()
        self.out = Variable(torch.randn(content.data.size())).type(dtype)
        self.out = nn.Parameter(self.out.data)

        self.content_layers = ['conv_4']
        self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        self.content_weight = 1
        self.style_weight = 1000

        self.loss_network = models.vgg19(pretrained=True)

        self.gram = GramMatrix()
        self.loss = nn.MSELoss()
        self.optimizer = optim.LBFGS([self.out])

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.loss_network.cuda()
            self.gram.cuda()

    def epoch(self):
        def closure():
            self.optimizer.zero_grad()
            out = self.out.clone()
            out.data.clamp_(0, 1)
            content = self.content.clone()
            style = self.style.clone()
            content_loss = 0
            style_loss = 0
            i = 1
            not_inplace = lambda layer: nn.ReLU(inplace=False) if isinstance(layer, nn.ReLU) else layer
            for layer in list(self.loss_network.features):
                layer = not_inplace(layer)
                if self.use_cuda:
                    layer.cuda()
                out, content, style = layer.forward(out), layer.forward(content), layer.forward(style)
                if isinstance(layer, nn.Conv2d):
                    name = "conv_" + str(i)
                    if name in self.content_layers:
                        content_loss += self.loss(out * self.content_weight, content.detach() * self.content_weight)
                    if name in self.style_layers:
                        out_g, style_g = self.gram.forward(out), self.gram.forward(style)
                        style_loss += self.loss(out_g * self.style_weight, style_g.detach() * self.style_weight)
                if isinstance(layer, nn.ReLU):
                    i += 1
            total_loss = content_loss + style_loss
            total_loss.backward()
            return total_loss

        self.optimizer.step(closure)
        return self.out

    def transfer(self, eps):
        for i in range(eps):
            out = self.epoch()
            if i % 10 == 0:
                print("Epoch: %d" % (i))
                #fname = os.path.join(self.cfg.path, 'outputs/%d.png' % (i))
                #out.data.clamp_(0, 1)
                #img_save(out, fname)
        out.data.clamp_(0, 1)
        return out


def style(cfg):
    sz = 512 if cfg.use_cuda else 32
    eps = 30
    dtype = torch.cuda.FloatTensor if cfg.use_cuda else torch.FloatTensor
    path_c = os.path.join(cfg.path, 'content')
    path_s = os.path.join(cfg.path, 'style')
    path_o = os.path.join(cfg.path, 'output')
    files_c = np.asarray(glob.glob(path_c + '/*.jpg'))
    files_s = np.asarray(glob.glob(path_s + '/*.jpg'))

    #img_files_resize(files_c, 512)
    #img_files_resize(files_s, 512)
    loader = transforms.Compose([
        transforms.Resize((sz,sz)),
        transforms.ToTensor()])
    unloader = transforms.ToPILImage()

    for c in sorted(files_c):
        for s in sorted(files_s):
            img1 = img_load(c, loader).type(dtype)
            img2 = img_load(s, loader).type(dtype)
            assert img1.size() == img2.size(), "style and content images must have the same size"
            style_transfer = StyleTransfer(cfg, img1, img2)
            img3 = style_transfer.transfer(eps)
            f = os.path.join(path_o, os.path.basename(c)[:-4] + '_' + os.path.basename(s)[:-4] + '.jpg')
            img_save(img3, f, unloader, sz)
            print(f)
