import chainer
import chainer.functions as F
import chainer.links as L

import cupy.cuda.stream as stream
import cupy.cuda.nvtx as nvtx
import cupy.cuda.runtime as runtime

import contextlib

_debug = False

@contextlib.contextmanager
def nvtx_range(name, cid=0):
    if _debug:
        nvtx.RangePush(name, cid)
    yield
    if _debug:
        runtime.deviceSynchronize()
        nvtx.RangePop()

class GoogLeNet_ooc(chainer.Chain):

    insize = 224

    def __init__(self):
        super(GoogLeNet_ooc, self).__init__(
            conv1=L.Convolution2D(None,  64, 7, stride=2, pad=3),
            conv2_reduce=L.Convolution2D(None,  64, 1),
            conv2=L.Convolution2D(None, 192, 3, stride=1, pad=1),
            inc3a=L.Inception(None,  64,  96, 128, 16,  32,  32),
            inc3b=L.Inception(None, 128, 128, 192, 32,  96,  64),
            inc4a=L.Inception(None, 192,  96, 208, 16,  48,  64),
            inc4b=L.Inception(None, 160, 112, 224, 24,  64,  64),
            inc4c=L.Inception(None, 128, 128, 256, 24,  64,  64),
            inc4d=L.Inception(None, 112, 144, 288, 32,  64,  64),
            inc4e=L.Inception(None, 256, 160, 320, 32, 128, 128),
            inc5a=L.Inception(None, 256, 160, 320, 32, 128, 128),
            inc5b=L.Inception(None, 384, 192, 384, 48, 128, 128),
            loss3_fc=L.Linear(None, 1000),

            loss1_conv=L.Convolution2D(None, 128, 1),
            loss1_fc1=L.Linear(None, 1024),
            loss1_fc2=L.Linear(None, 1000),

            loss2_conv=L.Convolution2D(None, 128, 1),
            loss2_fc1=L.Linear(None, 1024),
            loss2_fc2=L.Linear(None, 1000)
        )
        # self.stream = stream.Stream(non_blocking=True)
        self.stream = None
        self.train = True
        # self.disable_swapout_params()

    def __call__(self, x, t):
        with nvtx_range("F0", 0):
            h = F.relu(self.conv1(x))
        h.set_end_of_sub_graph(stream=self.stream)

        with nvtx_range("F0", 0):
            h = F.local_response_normalization(
                F.max_pooling_2d(h, 3, stride=2), n=5)
        h.set_end_of_sub_graph(stream=self.stream)

        with nvtx_range("F0", 0):
            h = F.relu(self.conv2_reduce(h))
        h.set_end_of_sub_graph(stream=self.stream)

        with nvtx_range("F0", 0):
            h = F.relu(self.conv2(h))
        h.set_end_of_sub_graph(stream=self.stream)

        with nvtx_range("F0", 0):
            h = F.max_pooling_2d(
                F.local_response_normalization(h, n=5), 3, stride=2)
        h.set_end_of_sub_graph(stream=self.stream)

        with nvtx_range("F1", 1):
            h = self.inc3a(h)
        h.set_end_of_sub_graph(stream=self.stream)

        with nvtx_range("F2", 2):
            h = self.inc3b(h)
            h = F.max_pooling_2d(h, 3, stride=2)
        h.set_end_of_sub_graph(stream=self.stream)

        with nvtx_range("F3", 3):
            h = self.inc4a(h)
        h.set_end_of_sub_graph(stream=self.stream)
        h.disable_swapout()

        with nvtx_range("L1", 4):
            l = F.average_pooling_2d(h, 5, stride=3)
            l = F.relu(self.loss1_conv(l))
            l = F.relu(self.loss1_fc1(l))
            l = self.loss1_fc2(l)
        l.set_end_of_sub_graph(stream=self.stream)
        loss1 = F.softmax_cross_entropy(l, t)

        with nvtx_range("F4", 5):
            h = self.inc4b(h)
        h.set_end_of_sub_graph(stream=self.stream)

        with nvtx_range("F4", 5):
            h = self.inc4c(h)
        h.set_end_of_sub_graph(stream=self.stream)

        with nvtx_range("F4", 5):
            h = self.inc4d(h)
        h.set_end_of_sub_graph(stream=self.stream)
        h.disable_swapout()

        with nvtx_range("L2", 6):
            l = F.average_pooling_2d(h, 5, stride=3)
            l = F.relu(self.loss2_conv(l))
            l = F.relu(self.loss2_fc1(l))
            l = self.loss2_fc2(l)
        l.set_end_of_sub_graph(stream=self.stream)
        loss2 = F.softmax_cross_entropy(l, t)

        with nvtx_range("F5", 7):
            h = self.inc4e(h)
            h = F.max_pooling_2d(h, 3, stride=2)
        h.set_end_of_sub_graph(stream=self.stream)

        with nvtx_range("F5", 7):
            h = self.inc5a(h)
        h.set_end_of_sub_graph(stream=self.stream)

        with nvtx_range("F5", 7):
            h = self.inc5b(h)
        h.set_end_of_sub_graph(stream=self.stream)

        h = F.average_pooling_2d(h, 7, stride=1)
        h = self.loss3_fc(F.dropout(h, 0.4, train=self.train))
        loss3 = F.softmax_cross_entropy(h, t)

        loss = 0.3 * (loss1 + loss2) + loss3
        accuracy = F.accuracy(h, t)

        chainer.report({
            'loss': loss,
            'loss1': loss1,
            'loss2': loss2,
            'loss3': loss3,
            'accuracy': accuracy
        }, self)

        if self.stream is None:
            self.stream = stream.Stream(non_blocking=True)

        return loss
