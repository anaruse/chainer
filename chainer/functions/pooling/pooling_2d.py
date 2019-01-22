from chainer.backends import cuda
from chainer import function_node
from chainer.utils import collections_abc
from chainer.utils import conv
from chainer.utils import type_check

if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cuda.cuda.cudnn


def _pair(x):
    if isinstance(x, collections_abc.Iterable):
        return x
    return x, x


class Pooling2D(function_node.FunctionNode):

    """Base class of pooling function over a set of 2d planes."""

    def __init__(self, ksize, stride=None, pad=0, cover_all=True,
                 return_indices=False, layout='NCHW'):
        if stride is None:
            stride = ksize

        self.kh, self.kw = _pair(ksize)
        self.sy, self.sx = _pair(stride)
        self.ph, self.pw = _pair(pad)

        self.cover_all = cover_all
        self.return_indices = return_indices
        self.layout = layout
        if layout not in ('NCHW', 'NHWC'):
            raise ValueError('unsupported layout: {}'.format(layout))

        self._used_cudnn = False
        self._cudnn_inputs = None
        self._cudnn_outputs = None

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype.kind == 'f',
            in_types[0].ndim == 4
        )

    def forward_gpu(self, x):
        self.retain_inputs((0,))
        self._used_cudnn = True

        # Implementation using cudnn
        x = x[0]

        if self.layout == 'NCHW':
            n, c, h, w = x.shape
        else:
            n, h, w, c = x.shape

        y_h = conv.get_conv_outsize(
            h, self.kh, self.sy, self.ph, self.cover_all)
        assert y_h > 0, 'Height in the output should be positive.'
        y_w = conv.get_conv_outsize(
            w, self.kw, self.sx, self.pw, self.cover_all)
        assert y_w > 0, 'Width in the output should be positive.'

        if self.layout == 'NCHW':
            y = cuda.cupy.empty((n, c, y_h, y_w), dtype=x.dtype)
            self.cudnn_layout = libcudnn.CUDNN_TENSOR_NCHW
        else:
            y = cuda.cupy.empty((n, y_h, y_w, c), dtype=x.dtype)
            self.cudnn_layout = libcudnn.CUDNN_TENSOR_NHWC

        cudnn.pooling_forward(
            x, y,
            (self.kh, self.kw), (self.sy, self.sx), (self.ph, self.pw),
            self._get_pool_mode(), self.cudnn_layout)
        self._cudnn_inputs = (x,)
        self._cudnn_outputs = (y,)
        self.retain_outputs((0,))
        return y,

    def backward_gpu(self, x, gy):
        # Implementation using cudnn
        x = x[0]
        y = self._cudnn_outputs[0]
        gx = cudnn.pooling_backward(
            x, y, gy[0],
            (self.kh, self.kw), (self.sy, self.sx), (self.ph, self.pw),
            self._get_pool_mode(), self.cudnn_layout)
        return gx,

    def _get_pool_mode(self):
        raise NotImplementedError()
