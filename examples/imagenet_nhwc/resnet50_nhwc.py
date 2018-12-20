# Original author: yasunorikudo
# (https://github.com/yasunorikudo/chainer-ResNet)

import numpy as np

import chainer
import chainer.functions as F
from chainer import initializers
import chainer.links as L

import cupy
from cupy import prof

from chainer import static_code
from chainer import static_graph


class BottleNeckA(chainer.Chain):

    def __init__(self, in_size, ch, out_size, stride=2):
        self.dtype = dtype = np.float16
        W = initializers.HeNormal(1 / np.sqrt(2), self.dtype)

        super(BottleNeckA, self).__init__()

        with self.init_scope():
            self.conv1 = L.NhwcConvolution2D(in_size, ch, 1, stride, 0, initialW=W, nobias=True)
            self.bn1 = L.BnormAddActivation(ch, activation="relu")
            self.conv2 = L.NhwcConvolution2D(ch, ch, 3, 1, 1, initialW=W, nobias=True)
            self.bn2 = L.BnormAddActivation(ch, activation="relu")
            self.conv3 = L.NhwcConvolution2D(ch, out_size, 1, 1, 0, initialW=W, nobias=True)
            self.bn3 = L.BnormAddActivation(out_size, activation="relu")

            self.conv4 = L.NhwcConvolution2D(in_size, out_size, 1, stride, 0, initialW=W, nobias=True)
            self.bn4 = L.BnormAddActivation(out_size)

    # @prof.TimeRangeDecorator('BottleNeckA', color_id=3)
    # @static_graph
    def __call__(self, x):
        h0 = self.bn4(self.conv4(x))

        h = self.bn1(self.conv1(x))
        h = self.bn2(self.conv2(h))
        y = self.bn3(self.conv3(h), h0)

        return y


class BottleNeckB(chainer.Chain):

    def __init__(self, in_size, ch):
        self.dtype = dtype = np.float16
        W = initializers.HeNormal(1 / np.sqrt(2), self.dtype)

        super(BottleNeckB, self).__init__()
        with self.init_scope():
            self.conv1 = L.NhwcConvolution2D(in_size, ch, 1, 1, 0, initialW=W, nobias=True)
            self.bn1 = L.BnormAddActivation(ch, activation='relu')
            self.conv2 = L.NhwcConvolution2D(ch, ch, 3, 1, 1, initialW=W, nobias=True)
            self.bn2 = L.BnormAddActivation(ch, activation='relu')
            self.conv3 = L.NhwcConvolution2D(ch, in_size, 1, 1, 0, initialW=W, nobias=True)
            self.bn3 = L.BnormAddActivation(in_size, activation='relu')

    # @prof.TimeRangeDecorator('BottleNeckB', color_id=2)
    # @static_graph
    def __call__(self, x):
        h = self.bn1(self.conv1(x))
        h = self.bn2(self.conv2(h))
        y = self.bn3(self.conv3(h), x)

        return y


class Block(chainer.ChainList):

    def __init__(self, layer, in_size, ch, out_size, stride=2):
        super(Block, self).__init__()
        self.add_link(BottleNeckA(in_size, ch, out_size, stride))
        for i in range(layer - 1):
            self.add_link(BottleNeckB(out_size, ch))

    # @prof.TimeRangeDecorator('Block', color_id=1)
    def __call__(self, x):
        for f in self.children():
            x = f(x)
        return x


class ResNet50(chainer.Chain):

    insize = 224

    def __init__(self):
        self.dtype = dtype = np.float16
        W = initializers.HeNormal(1 / np.sqrt(2), self.dtype)
        bias = initializers.Zero(self.dtype)

        super(ResNet50, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                3, 64, 7, 2, 3, initialW=W, initial_bias=bias)
            self.bn1 = L.BatchNormalization(64, dtype=dtype)
            self.res2 = Block(3, 64, 64, 256, 1)
            self.res3 = Block(4, 256, 128, 512)
            self.res4 = Block(6, 512, 256, 1024)
            self.res5 = Block(3, 1024, 512, 2048)
            self.fc = L.Linear(2048, 1000, initialW=W, initial_bias=bias)

    # @prof.TimeRangeDecorator('ResNet50', color_id=0)
    def __call__(self, x, t):
        h = F.cast(x, self.dtype)
        h = self.bn1(self.conv1(h))
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = F.transpose(h, (0, 2, 3, 1))
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = self.res5(h)
        h = F.transpose(h, (0, 3, 1, 2))
        h = F.average_pooling_2d(h, 7, stride=1)
        h = self.fc(h)

        loss = F.softmax_cross_entropy(h, t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
        return loss
