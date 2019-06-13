import numpy

from chainer import configuration
from chainer.functions.fused_ops import fused_ops
from chainer import initializers
from chainer import link
from chainer import variable


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x


class FusedScaleBiasActConvBn(link.Link):

    """__init__(self, in_channels, out_channels, ksize, stride=1, pad=0, \
bn_decay=0.9, bn_eps=2e-5):

    h = scale * x + bias
    h = act(h)
    y = conv(h, w)
    ysum, ysqsum = bn_stats(y)
    scale, bias = bn_finalize(gamma, beta, ysum, ysqsum, ...)

    """

    dilate = 1
    group = 1
    tensor_layout = 'NHWC'
    
    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 bn_decay=0.9, bn_eps=2e-5):
        super(FusedScaleBiasActConvBn, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ksize = _pair(ksize)
        self.stride = _pair(stride)
        self.pad = _pair(pad)
        self.bn_decay = bn_decay
        self.bn_eps = bn_eps

        with self.init_scope():
            W_initializer = initializers._get_initializer(None)
            W_initializer.dtype = numpy.float16
            self.W = variable.Parameter(W_initializer)

            gamma_initializer = initializers._get_initializer(1)
            gamma_initializer.dtype = numpy.float32
            self.gamma = variable.Parameter(gamma_initializer)

            beta_initializer = initializers._get_initializer(0)
            beta_initializer.dtype = numpy.float32
            self.beta = variable.Parameter(beta_initializer)

        W_shape = (self.out_channels, self.ksize[0], self.ksize[1],
                   self.in_channels)
        self.W.initialize(W_shape)

        self.gamma.initialize(self.out_channels)
        self.beta.initialize(self.out_channels)

        self.running_mean = self._init_array(None, 0, self.out_channels,
                                             numpy.float32)
        self.register_persistent('running_mean')
        self.running_var = self._init_array(None, 1, self.out_channels,
                                            numpy.float32)
        self.register_persistent('running_var')

    def _init_array(self, initializer, default_value, size, dtype):
        if initializer is None:
            initializer = default_value
        initializer = initializers._get_initializer(initializer)
        return initializers.generate_array(
            initializer, size, self.xp, dtype=dtype, device=self.device)

    def forward(self, x, scale=None, bias=None, **kwargs):
        if configuration.config.train:
            ret = fused_ops.fused_scale_bias_act_conv_bn(
                x, scale, bias, self.W, self.gamma, self.beta,
                stride=self.stride, pad=self.pad,
                bn_eps=self.bn_eps, bn_decay=self.bn_decay,
                running_mean=self.running_mean, running_var=self.running_var)
        else:
            ret = fused_ops.fused_scale_bias_act_conv_bn_inference(
                x, scale, bias, self.W, self.gamma, self.beta,
                stride=self.stride, pad=self.pad,
                bn_eps=self.bn_eps,
                running_mean=self.running_mean, running_var=self.running_var)
        
        return ret
