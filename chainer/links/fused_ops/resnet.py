import numpy

from chainer import configuration
from chainer.functions.fused_ops import resnet
from chainer import initializers
from chainer import link
from chainer import variable


class ResNetBottleNeck(link.Link):

    tensor_layout = 'NHWC'

    def __init__(self, ch, stride=1, bn_decay=0.9, bn_eps=2e-5):
        super(ResNetBottleNeck, self).__init__()

        self.ch = ch
        self.stride = stride
        self.bn_decay = bn_decay
        self.bn_eps = bn_eps

        with self.init_scope():
            W_initializer = initializers._get_initializer(None)
            W_initializer.dtype = numpy.float16
            self.W1 = variable.Parameter(W_initializer)
            self.W2 = variable.Parameter(W_initializer)
            self.W3 = variable.Parameter(W_initializer)
            gamma_initializer = initializers._get_initializer(1)
            gamma_initializer.dtype = numpy.float32
            self.gamma1 = variable.Parameter(gamma_initializer)
            self.gamma2 = variable.Parameter(gamma_initializer)
            self.gamma3 = variable.Parameter(gamma_initializer)
            beta_initializer = initializers._get_initializer(0)
            beta_initializer.dtype = numpy.float32
            self.beta1 = variable.Parameter(beta_initializer)
            self.beta2 = variable.Parameter(beta_initializer)
            self.beta3 = variable.Parameter(beta_initializer)
        
        self.W1.initialize((ch[1], 1, 1, ch[0]))
        self.W2.initialize((ch[1], 3, 3, ch[1]))
        self.W3.initialize((ch[2], 1, 1, ch[1]))
        self.gamma1.initialize(ch[1])
        self.gamma2.initialize(ch[1])
        self.gamma3.initialize(ch[2])
        self.beta1.initialize(ch[1])
        self.beta2.initialize(ch[1])
        self.beta3.initialize(ch[2])
        self.running_mean1 = self._init_array(None, 0, ch[1])
        self.running_mean2 = self._init_array(None, 0, ch[1])
        self.running_mean3 = self._init_array(None, 0, ch[2])
        self.running_var1 = self._init_array(None, 0, ch[1])
        self.running_var2 = self._init_array(None, 0, ch[1])
        self.running_var3 = self._init_array(None, 0, ch[2])
        self.register_persistent('running_mean1')
        self.register_persistent('running_mean2')
        self.register_persistent('running_mean3')
        self.register_persistent('running_var1')
        self.register_persistent('running_var2')
        self.register_persistent('running_var3')

    def _init_array(self, initializer, default_value, size,
                    dtype=numpy.float32):
        if initializer is None:
            initializer = default_value
        initializer = initializers._get_initializer(initializer)
        return initializers.generate_array(
            initializer, size, self.xp, dtype=dtype, device=self.device)

    def forward(self, x, h=None, **kwargs):
        ret = resnet.resnet_bottle_neck(
            x, h, (self.W1, self.W2, self.W3),
            (self.gamma1, self.gamma2, self.gamma3),
            (self.beta1, self.beta2, self.beta3),
            (self.running_mean1, self.running_mean2, self.running_mean3),
            (self.running_var1, self.running_var2, self.running_var3),
            self.stride, self.bn_eps, self.bn_decay)
        return ret
