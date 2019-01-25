import numpy

import cupy
from cupy import prof

import chainer
from chainer import configuration
from chainer import functions
from chainer import initializers
from chainer import link
from chainer import variable


class BnormAddActivation(link.Link):
    " y = activation( bnorm(x) + z ) "
    gamma = None
    beta = None
    avg_mean = None
    avg_var = None
    dtype = numpy.float16
    param_dtype = numpy.float32

    def __init__(self, size, decay=0.9, eps=2e-5,
                 use_gamma=True, use_beta=True,
                 initial_gamma=None, initial_beta=None,
                 initial_avg_mean=None, initial_avg_var=None,
                 activation=None):
        super(BnormAddActivation, self).__init__()

        self._initial_avg_mean = initial_avg_mean
        self._initial_avg_var = initial_avg_var
        self.decay = decay
        self.eps = eps
        if activation not in (None, 'relu'):
            msg = 'Unknown activation mode: {}'.format(activation)
            raise RuntimeError(msg)
        self.activation = activation

        with self.init_scope():
            if use_gamma:
                if initial_gamma is None:
                    initial_gamma = 1
                gamma_initializer = \
                    initializers._get_initializer(initial_gamma)
                gamma_initializer.dtype = self.param_dtype
                self.gamma = variable.Parameter(gamma_initializer)
            if use_beta:
                if initial_beta is None:
                    initial_beta = 0
                beta_initializer = initializers._get_initializer(initial_beta)
                beta_initializer.dtype = self.param_dtype
                self.beta = variable.Parameter(beta_initializer)

            self._initialize_params(size)

    def _initialize_params(self, shape):
        self.avg_mean = self._init_array(self._initial_avg_mean, 0, shape)
        self._initial_avg_mean = None
        self.register_persistent('avg_mean')
        self.avg_var = self._init_array(self._initial_avg_var, 1, shape)
        self._initial_avg_var = None
        self.register_persistent('avg_var')
        if self.gamma is not None:
            self.gamma.initialize(shape)
        if self.beta is not None:
            self.beta.initialize(shape)
        self._ones = None
        self._zeros = None
        self._dummy_mean = None
        self._dummy_var = None

    def _init_array(self, initializer, default_value, size):
        if initializer is None:
            initializer = default_value
        initializer = initializers._get_initializer(initializer)
        return initializers.generate_array(
            initializer, size, self.xp, dtype=self.param_dtype)

    # @prof.TimeRangeDecorator()
    def forward(self, x, z=None, **kwargs):
        gamma = self.gamma
        if gamma is None:
            print('# initialize gamma')
            with chainer.using_device(self.device):
                gamma = self.xp.ones(
                    self.avg_mean.shape, dtype=self.param_dtype)

        beta = self.beta
        if beta is None:
            print('# initialize beta')
            with chainer.using_device(self.device):
                beta = self.xp.zeros(
                    self.avg_mean.shape, dtype=self.param_dtype)

        if self._ones is None:
            with chainer.using_device(self.device):
                self._ones = self.xp.ones(self.avg_mean.shape, dtype=self.param_dtype)
                self._zeros = self.xp.zeros(self.avg_mean.shape, dtype=self.param_dtype)
                self._dummy_mean = self.xp.ones(self.avg_mean.shape, dtype=self.param_dtype)
                self._dummy_var = self.xp.ones(self.avg_mean.shape, dtype=self.param_dtype)

        if configuration.config.train:
            ret = functions.bnorm_add_activation(
                x, gamma, beta, z,
                eps=self.eps, running_mean=self.avg_mean,
                running_var=self.avg_var, decay=self.decay,
                activation=self.activation)
        else:
            # TODO(anaruse): improve performance
            ret = functions.fixed_bnorm_add_activation(
                x, gamma, beta, self.avg_mean, self.avg_var, z, self.eps, self.activation)
            # gamma = gamma.data.astype(self.dtype)
            # beta = beta.data.astype(self.dtype)
            # mean = self.avg_mean.astype(self.dtype)
            # var = self.avg_var.astype(self.dtype)
            # # print('# mean: {}'.format(mean))
            # # print('# var: {}'.format(var))
            # _x = x.transpose(0, 2, 3, 1)  # NCHW --> NHWC
            # h = functions.fixed_batch_normalization(
            #     _x, gamma, beta, mean, var, self.eps)
            # h = h.transpose(0, 3, 1, 2)  # NHWC --> NCHW
            # if z is not None:
            #     h = h + z
            # if self.activation is 'relu':
            #     h = functions.relu(h)
            # ret = h
        return ret
