import torch
import cv2
import os
import numpy as np
import math
import random
import threading
from enum import IntEnum
from lib.util import *
# pylint: disable=E1101


def open_image(fname):
    """ Opens an image using OpenCV given the file path.

    Arguments:
        fname: absolute path of the image file

    Returns:
        The image in RGB format as numpy array of floats normalized to range between 0.0 - 1.0
    """
    flags = cv2.IMREAD_UNCHANGED+cv2.IMREAD_ANYDEPTH+cv2.IMREAD_ANYCOLOR
    if not os.path.exists(fname):
        raise OSError('No such file or directory: {}'.format(fname))
    elif os.path.isdir(fname):
        raise OSError('Is a directory: {}'.format(fname))
    else:
        try:
            img = cv2.imread(str(fname), flags)
            #img = img.astype(np.float32)/255.0
            if img is None: raise OSError(f'File not recognized by opencv: {fname}')
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            raise OSError('Error handling image at: {}'.format(fname)) from e
            

def save_image(fname, img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #img = (img * 255.0).astype(np.uint8)
    cv2.imwrite(str(fname), img)


class Transforms():
    def __init__(self, tfms):
        self.tfms = tfms

    def __call__(self, x):
        for trf in  self.tfms:
            x = trf(x)
        return x

    def __repr__(self):
        return str(self.tfms)


class Transform():
    def __init__(self):
        self.store = threading.local()

    def set_state(self):
        pass

    def __call__(self, x):
        self.set_state()
        return self.do_transform(x)


class Normalize():
    def __init__(self, m, s, shift=0):
        self.m = np.array(m, dtype=np.float32)
        self.s = np.array(s, dtype=np.float32)
        self.shift = shift

    def __call__(self, x):
        x = (x-self.m)/self.s
        x = x - self.shift
        return x


class Denormalize():
    def __init__(self, m, s, shift=0):
        self.m = np.array(m, dtype=np.float32)
        self.s = np.array(s, dtype=np.float32)
        self.shift = shift
    def __call__(self, x):
        x = x + self.shift
        x = x*self.s + self.m
        return x


class Unit():
    def __init__(self, _min, _max):
        self.min = _min
        self.max = _max

    def __call__(self, x):
        return 2.0*((x-self.min)/(self.max-self.min)-0.5)


class CenterCrop(Transform):
    def __init__(self, sz=None):
        super().__init__()
        self.sz = int(sz)

    def do_transform(self, x):
        r,c,*_ = x.shape
        if min(r,c) < self.sz:
            return x
        sz = min(r,c) if self.sz is None else self.sz
        r1 = math.ceil((r-sz)/2)
        c1 = math.ceil((c-sz)/2)
        return x[r1:r1+sz, c1:c1+sz]


class RandomCrop(Transform):
    def __init__(self, sz):
        super().__init__()
        self.sz = int(sz)

    def set_state(self):
        self.store.r0 = np.random.uniform(0, 1)
        self.store.c0 = np.random.uniform(0, 1)

    def do_transform(self, x):
        r,c,*_ = x.shape
        r1 = np.floor(self.store.r0*(r-self.sz)).astype(int)
        c1 = np.floor(self.store.c0*(c-self.sz)).astype(int)
        return x[r1:r1+self.sz, c1:c1+self.sz]


def scale_to(x, ratio, targ): 
    return max(math.floor(x*ratio), targ)


class Scale(Transform):
    """ Scales min size to sz.
    """
    def __init__(self, sz):
        super().__init__()
        self.sz = sz

    def do_transform(self, x):
        r,c,*_ = x.shape
        ratio = self.sz/min(r,c)
        sz = (scale_to(c, ratio, self.sz), scale_to(r, ratio, self.sz))
        return cv2.resize(x, sz, interpolation=cv2.INTER_AREA)


class MinScale(Scale):
    """ Scales min size to sz if min size < sz
    """
    def __init__(self, sz):
        super().__init__(sz)

    def do_transform(self, x):
        r,c,*_ = x.shape
        if min(r,c) < self.sz:
            return super().do_transform(x)
        return x


def rotate_cv(im, deg, mode=cv2.BORDER_CONSTANT, interpolation=cv2.INTER_AREA):
    r,c,*_ = im.shape
    M = cv2.getRotationMatrix2D((c/2,r/2),deg,1)
    return cv2.warpAffine(im,M,(c,r), borderMode=mode, flags=cv2.WARP_FILL_OUTLIERS+interpolation)


class RandomRotate(Transform):
    """ Rotates images.

     Arguments:
        deg (float): degree to rotate.
        p (float): probability of rotation
        mode: type of border
    """
    def __init__(self, deg, p=0.75, mode=cv2.BORDER_REFLECT):
        super().__init__()
        self.deg,self.p = deg,p
        self.mode = mode

    def set_state(self):
        self.store.rdeg = rand0(self.deg)
        self.store.rp = random.random()<self.p

    def do_transform(self, x):
        if self.store.rp:
            x = rotate_cv(x, self.store.rdeg, mode= self.mode, interpolation=cv2.INTER_AREA)
        return x


class ScaleRadius(Transform):
    def __init__(self, scale=300.0):
        self.scale = scale

    def do_transform(self, x):
        v = x[int(x.shape[0]/2),:,:].sum(1)
        r = (v > v.mean()/10).sum() / 2
        if 0.01 < r < 100.0:
            print(r.shape)
            r = 1000.0
        s = self.scale * 1.0 / r
        return cv2.resize(x, (0,0), fx=s, fy=s)
        

class NormalizeColor(Transform):
    def __init__(self, range):
        self.range = range

    def do_transform(self, x):
        return cv2.addWeighted(x, 4.0, cv2.GaussianBlur(x, (0,0), self.range), -4.0, 128)


class RemoveOuter(Transform):
    def __init__(self, scale=300.0):
        self.scale = scale

    def do_transform(self, x):
        z = np.zeros(x.shape)
        cv2.circle(z, (int(x.shape[1]/2), int(x.shape[0]/2)), int(self.scale * 0.9), (1, 1, 1), -1, 8, 0)
        x = x * z + 128 * (1 - z)
        x = x.astype(np.uint8)
        return x



def toGray(img):
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img[:,:, None]
    return img


def resize(img, size):
    return cv2.resize(img, (size, size))


def resize_2(img, size):
    size1 = img.shape[:2]  # old_size is in (height, width) format
    r = float(size) / max(size1)
    size2 = tuple([int(x * r) for x in size1])
    img = cv2.resize(img, (size2[1], size2[0]))
    delta_w = size - size2[1]
    delta_h = size - size2[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    color = [128, 128, 128]
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img


def cropCircle(img):
    """
        Crops a circular like image from a dark background
    """
    x = img[int(img.shape[0]/2),:,:]
    #x = (x > x.mean()/5)
    x = (x != 128)
    x = np.where(x > 0)
    w1 = x[0][0]
    w2 = x[0][-1]
    x = img[:,int(img.shape[1]/2),:]
    #x = (x > x.mean()/5)
    x = (x != 128)
    x = np.where(x > 0)
    h1 = x[0][0]
    h2 = x[0][-1]
    img = img[h1:h2, w1:w2, :]
    return img


def equalizeHist(img):
    img = img.astype(np.uint8)
    for c in range(img.shape[2]):
        img[:, :, c] = cv2.equalizeHist(img[:, :, c])
    return img


def equalizeHistCLAHE(img):
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16,16))
    img = img.astype(np.uint8)
    for c in range(img.shape[2]):
        img[:, :, c] = clahe.apply(img[:, :, c])
    return img


def deNoise(img):
    return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)


def adjustGamma(img, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(img, table)