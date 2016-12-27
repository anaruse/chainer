import numpy as np

import chainer
import chainer.functions as F
import chainer.functions.util.fused_function as FF
from chainer import initializers
import chainer.links as L


class Alexff(chainer.Chain):

    """Single-GPU AlexNet without partition toward the channel axis."""

    insize = 227

    def __init__(self):
        super(Alexff, self).__init__(
            conv1=L.Convolution2D(None,  96, 11, stride=4),
            conv2=L.Convolution2D(None, 256,  5, pad=2),
            conv3=L.Convolution2D(None, 384,  3, pad=1),
            conv4=L.Convolution2D(None, 384,  3, pad=1, pre_funcs=F.ReLU()),
            conv5=L.Convolution2D(None, 256,  3, pad=1, pre_funcs=F.ReLU()),
            fc6=L.Linear(None, 4096),
            fc7=L.Linear(None, 4096),
            fc8=L.Linear(None, 1000),
        )
        self.train = True

    def __call__(self, x, t):

        h = self.conv1(x)
        # h = F.relu(h)
        # h = F.local_response_normalization(h)
        # h = F.max_pooling_2d(h, 3, stride=2)
        ff = FF.FusedFunction(F.ReLU(),
                              F.LocalResponseNormalization(),
                              F.MaxPooling2D(3, 2))
        h = ff(h)

        h = self.conv2(h)
        # h = F.relu(h)
        # h = F.local_response_normalization(h)
        # h = F.max_pooling_2d(h, 3, stride=2)
        ff = FF.FusedFunction(F.ReLU(),
                              F.LocalResponseNormalization(),
                              F.MaxPooling2D(3, 2))
        h = ff(h)

        h = self.conv3(h)
        # h = F.relu(h)

        h = self.conv4(h)
        # h = F.relu(h)

        h = self.conv5(h)
        # h = F.relu(h)
        # h = F.max_pooling_2d(h, 3, stride=2)
        ff = FF.FusedFunction(F.ReLU(),
                              F.MaxPooling2D(3, 2))
        h = ff(h)

        h = self.fc6(h)
        # h = F.relu(h)
        # h = F.dropout(h, train=self.train)
        ff = FF.FusedFunction(F.ReLU(),
                              F.Dropout(0.5))
        h = ff(h)

        h = self.fc7(h)
        # h = F.relu(h)
        # h = F.dropout(h, train=self.train)
        ff = FF.FusedFunction(F.ReLU(),
                              F.Dropout(0.5))
        h = ff(h)

        h = self.fc8(h)

        loss = F.softmax_cross_entropy(h, t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
        return loss


class AlexFp16(Alexff):

    """Single-GPU AlexNet without partition toward the channel axis."""

    insize = 227

    def __init__(self):
        self.dtype = np.float16
        W = initializers.HeNormal(1 / np.sqrt(2), self.dtype)
        bias = initializers.Zero(self.dtype)
        chainer.Chain.__init__(
            self,
            conv1=L.Convolution2D(3,  96, 11, stride=4, initialW=W, bias=bias),
            conv2=L.Convolution2D(96, 256,  5, pad=2, initialW=W, bias=bias),
            conv3=L.Convolution2D(256, 384,  3, pad=1, initialW=W, bias=bias),
            conv4=L.Convolution2D(384, 384,  3, pad=1, initialW=W, bias=bias),
            conv5=L.Convolution2D(384, 256,  3, pad=1, initialW=W, bias=bias),
            fc6=L.Linear(9216, 4096, initialW=W, bias=bias),
            fc7=L.Linear(4096, 4096, initialW=W, bias=bias),
            fc8=L.Linear(4096, 1000, initialW=W, bias=bias),
        )
        self.train = True

    def __call__(self, x, t):
        return Alexff.__call__(self, F.cast(x, self.dtype), t)
