# Original author: yasunorikudo
# (https://github.com/yasunorikudo/chainer-ResNet)

import chainer
from chainer.configuration import using_config
import chainer.functions as F
from chainer import initializers
import chainer.links as L

# import cupy

_bn_eps = 2.5e-5
_bn_eps = 1e-4


class BottleNeckA(chainer.Chain):

    def __init__(self, in_size, ch, out_size, stride=2):
        super(BottleNeckA, self).__init__()
        initialW = initializers.HeNormal()
        self.stride = stride
        with self.init_scope():
            self.conv4 = L.Convolution2D(
                in_size, out_size, 1, stride, 0,
                initialW=initialW, nobias=True)
            self.bn4 = L.BatchNormalization(out_size)
            self.resnet_bottle_neck = L.ResNetBottleNeck(
                ch=(in_size, ch, out_size), stride=stride, bn_eps=_bn_eps)

    def forward(self, x):
        h = self.bn4(self.conv4(x))
        y = self.resnet_bottle_neck(x, h)
        return y


class BottleNeckB(chainer.Chain):

    def __init__(self, in_size, ch):
        super(BottleNeckB, self).__init__()
        with self.init_scope():
            self.resnet_bottle_neck = L.ResNetBottleNeck(
                ch=(in_size, ch, in_size), bn_eps=_bn_eps)

    def forward(self, x):
        y = self.resnet_bottle_neck(x)
        return y


class Block(chainer.ChainList):

    def __init__(self, layer, in_size, ch, out_size, stride=2):
        super(Block, self).__init__()
        self.add_link(BottleNeckA(in_size, ch, out_size, stride))
        for i in range(layer - 1):
            self.add_link(BottleNeckB(out_size, ch))

    def forward(self, x):
        for f in self.children():
            x = f(x)
        return x


class ResNet50(chainer.Chain):

    insize = 224

    def __init__(self):
        super(ResNet50, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                3, 64, 7, 2, 3, initialW=initializers.HeNormal(),
                nobias=True)
            self.bn1 = L.BatchNormalization(64)
            self.res2 = Block(3, 64, 64, 256, 1)
            self.res3 = Block(4, 256, 128, 512)
            self.res4 = Block(6, 512, 256, 1024)
            self.res5 = Block(3, 1024, 512, 2048)
            self.fc = L.Linear(2048, 1000)

    def forward(self, x, t):
        with using_config('tensor_layout', 'NHWC'):
            h = F.transpose_to_NHWC(x)
            h = self.bn1(self.conv1(h))
            h = F.max_pooling_2d(F.relu(h), 3, stride=2)
            h = self.res2(h)
            h = self.res3(h)
            h = self.res4(h)
            h = self.res5(h)
            h = F.average_pooling_2d(h, 7, stride=1)
            h = F.transpose_to_NCHW(h)
        h = self.fc(h)
        loss = F.softmax_cross_entropy(h, t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
        return loss
