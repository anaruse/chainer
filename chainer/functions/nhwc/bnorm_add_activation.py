import numpy

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer import function_node
from chainer.utils import argument
from chainer.utils import type_check


if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cuda.cuda.cudnn
    _cudnn_version = cuda.cuda.cudnn.getVersion()
    memory = cuda.cuda.memory


class BnormAddActivation(function_node.FunctionNode):
    " y = activation( bnorm(x) + z ) "
    mean = None
    inv_std = None
    cudnn_bn_mode = libcudnn.CUDNN_BATCHNORM_SPATIAL_PERSISTENT
    cudnn_tensor_layout = libcudnn.CUDNN_TENSOR_NHWC
    dtype = numpy.float16
    param_dtype = numpy.float32

    def __init__(self, eps=2e-5, mean=None, var=None, decay=0.9,
                 activation=None):
        self.running_mean = mean
        self.running_var = var
        self.eps = eps
        if eps < libcudnn.CUDNN_BN_MIN_EPSILON:
            raise RuntimeError(
                'cuDNN does not allow an eps value '
                'less than {}.'.format(libcudnn.CUDNN_BN_MIN_EPSILON))
        self.decay = decay
        if activation not in (None, 'sigmoid', 'relu', 'tanh', 'clipped_relu',
                              'elu'):
            msg = 'Unknown activation mode: {}'.format(activation)
            raise RuntimeError(msg)
        self.activation = activation

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(3 <= n_in, n_in <= 4)
        if n_in == 3:
            x_type, gamma_type, beta_type = in_types
            type_check.expect(
                x_type.dtype == self.dtype,
                x_type.ndim == 4,
                gamma_type.dtype == self.param_dtype,
                gamma_type.ndim == 1,
                gamma_type.dtype == beta_type.dtype,
                gamma_type.shape == beta_type.shape,
            )
        if n_in == 4:
            x_type, gamma_type, beta_type, z_type = in_types
            type_check.expect(
                x_type.dtype == self.dtype,
                x_type.ndim == 4,
                z_type.dtype == x_type.dtype,
                z_type.ndim == x_type.ndim,
                gamma_type.dtype == self.param_dtype,
                gamma_type.ndim == 1,
                gamma_type.dtype == beta_type.dtype,
                gamma_type.shape == beta_type.shape,
            )

    def forward(self, inputs):
        if len(inputs) == 3:
            self.retain_inputs((0, 1, 2))
            (x, gamma, beta), z = inputs, None
        else:
            self.retain_inputs((0, 1, 2, 3))
            x, gamma, beta, z = inputs

        xp = backend.get_array_module(x)
        if xp is numpy:
            raise RuntimeError('this function is available on GPU only')
        if not chainer.should_use_cudnn('>=auto', 7400):
            msg = 'this function is available with cuDNN 7.4 or later.'
            raise RuntimeError(msg)

        if self.running_mean is None:
            self.running_mean = xp.zeros_like(gamma)
            self.running_var = xp.zeros_like(gamma)

        x = cuda.cupy.ascontiguousarray(x)
        gamma = cuda.cupy.ascontiguousarray(gamma)
        beta = cuda.cupy.ascontiguousarray(beta)
        handle = cudnn.get_handle()
        x_desc = cudnn.create_tensor_descriptor(x, self.cudnn_tensor_layout)
        if self.activation is None:
            act_desc = cudnn.create_activation_descriptor(
                libcudnn.CUDNN_ACTIVATION_RELU)  # dummy
            cudnn_bn_ops = libcudnn.CUDNN_BATCHNORM_OPS_BN
        else:
            if self.activation is 'sigmoid':
                act_mode = libcudnn.CUDNN_ACTIVATION_SIGMOID
            elif self.activation is 'relu':
                act_mode = libcudnn.CUDNN_ACTIVATION_RELU
            elif self.activation is 'tanh':
                act_mode = libcudnn.CUDNN_ACTIVATION_TANH
            elif self.activation is 'clipped_relu':
                act_mode = libcudnn.CUDNN_ACTIVATION_CLIPPED_RELU
            elif self.activation is 'clipped_elu':
                act_mode = libcudnn.CUDNN_ACTIVATION_CLIPPED_ELU
            act_desc = cudnn.create_activation_descriptor(act_mode)
            if z is None:
                cudnn_bn_ops = libcudnn.CUDNN_BATCHNORM_OPS_BN_ACTIVATION
            else:
                cudnn_bn_ops = libcudnn.CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION
        self.cudnn_bn_ops = cudnn_bn_ops
        self.act_desc = act_desc
        if z is None:
            z = x  # dummy
        else:
            z = cuda.cupy.ascontiguousarray(z)

        derivedBnDesc = cudnn.create_uninitialized_tensor_descriptor()
        libcudnn.deriveBNTensorDescriptor(derivedBnDesc.value,
                                          x_desc.value, self.cudnn_bn_mode)
        running_mean = self.running_mean
        running_var = self.running_var

        one = numpy.array(1, dtype=self.param_dtype).ctypes
        zero = numpy.array(0, dtype=self.param_dtype).ctypes
        y = cuda.cupy.empty_like(x)
        # Factor used in the moving average
        factor = 1 - self.decay

        if self.mean is None:
            self.mean = xp.empty_like(gamma)
            self.inv_std = xp.empty_like(gamma)

        workspace_size = libcudnn.getBatchNormalizationForwardTrainingExWorkspaceSize(  # NOQA
            handle, self.cudnn_bn_mode, cudnn_bn_ops, x_desc.value,
            x_desc.value, x_desc.value, derivedBnDesc.value,
            act_desc.value)
        workspace = memory.alloc(workspace_size)

        self.reservespace_size = libcudnn.getBatchNormalizationTrainingExReserveSpaceSize(  # NOQA
            handle, self.cudnn_bn_mode, cudnn_bn_ops, act_desc.value,
            x_desc.value)
        self.reservespace = memory.alloc(self.reservespace_size)

        libcudnn.batchNormalizationForwardTrainingEx(
            handle, self.cudnn_bn_mode, cudnn_bn_ops,
            one.data, zero.data,
            x_desc.value, x.data.ptr,
            x_desc.value, z.data.ptr,
            x_desc.value, y.data.ptr,
            derivedBnDesc.value,
            gamma.data.ptr, beta.data.ptr,
            factor,
            running_mean.data.ptr, running_var.data.ptr,
            self.eps, self.mean.data.ptr, self.inv_std.data.ptr,
            act_desc.value,
            workspace.ptr, workspace_size,
            self.reservespace.ptr, self.reservespace_size)

        self.retain_outputs((0, ))
        return y,

    def backward(self, indexes, grad_outputs):
        if self.cudnn_bn_ops is libcudnn.CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION:
            x, gamma, beta, z = self.get_retained_inputs()
        else:
            x, gamma, beta = self.get_retained_inputs()
            z = None
        y = self.get_retained_outputs()[0]
        gy, = grad_outputs

        f = BnormAddActivationGrad(self.eps, self.mean, self.inv_std,
                                   self.cudnn_bn_ops, self.act_desc,
                                   self.reservespace, self.reservespace_size)
        return f.apply((x, gamma, beta, z, y, gy))


class BnormAddActivationGrad(function_node.FunctionNode):

    cudnn_bn_mode = libcudnn.CUDNN_BATCHNORM_SPATIAL_PERSISTENT
    cudnn_tensor_layout = libcudnn.CUDNN_TENSOR_NHWC
    dtype = numpy.float16
    param_dtype = numpy.float32

    def __init__(self, eps, mean, inv_std,
                 cudnn_bn_ops, act_desc, reservespace, reservespace_size):
        self.eps = eps
        self.mean = mean
        self.inv_std = inv_std
        self.cudnn_bn_ops = cudnn_bn_ops
        self.act_desc = act_desc
        self.reservespace = reservespace
        self.reservespace_size = reservespace_size

    def forward(self, inputs):
        x, gamma, beta, z, y, gy = inputs

        x = cuda.cupy.ascontiguousarray(x)
        gamma = cuda.cupy.ascontiguousarray(gamma)
        beta = cuda.cupy.ascontiguousarray(beta)
        if z is not None:
            z = cuda.cupy.ascontiguousarray(z)
        y = cuda.cupy.ascontiguousarray(y)
        gy = cuda.cupy.ascontiguousarray(gy)
        handle = cudnn.get_handle()
        x_desc = cudnn.create_tensor_descriptor(x, self.cudnn_tensor_layout)
        derivedBnDesc = cudnn.create_uninitialized_tensor_descriptor()
        libcudnn.deriveBNTensorDescriptor(derivedBnDesc.value,
                                          x_desc.value, self.cudnn_bn_mode)
        one = numpy.array(1, dtype=self.param_dtype).ctypes
        zero = numpy.array(0, dtype=self.param_dtype).ctypes

        gx = cuda.cupy.empty_like(x)
        if z is not None:
            gz = cuda.cupy.empty_like(x)
        else:
            gz = gx  # dummy
        ggamma = cuda.cupy.empty_like(gamma)
        gbeta = cuda.cupy.empty_like(gamma)

        workspace_size = libcudnn.getBatchNormalizationBackwardExWorkspaceSize(
            handle, self.cudnn_bn_mode, self.cudnn_bn_ops, x_desc.value,
            x_desc.value, x_desc.value, x_desc.value, x_desc.value,
            derivedBnDesc.value, self.act_desc.value)
        workspace = memory.alloc(workspace_size)

        libcudnn.batchNormalizationBackwardEx(
            handle, self.cudnn_bn_mode, self.cudnn_bn_ops,
            one.data, zero.data, one.data, zero.data,
            x_desc.value, x.data.ptr,
            x_desc.value, y.data.ptr,
            x_desc.value, gy.data.ptr,
            x_desc.value, gz.data.ptr,
            x_desc.value, gx.data.ptr,
            derivedBnDesc.value,
            gamma.data.ptr, beta.data.ptr,
            ggamma.data.ptr, gbeta.data.ptr,
            self.eps,
            self.mean.data.ptr, self.inv_std.data.ptr,
            self.act_desc.value,
            workspace.ptr, workspace_size,
            self.reservespace.ptr, self.reservespace_size)

        if z is None:
            return gx, ggamma, gbeta
        else:
            return gx, ggamma, gbeta, gz


def bnorm_add_activation(x, gamma, beta, z=None, **kwargs):

    eps, running_mean, running_var, decay, activation = argument.parse_kwargs(
        kwargs, ('eps', 2e-5), ('running_mean', None),
        ('running_var', None), ('decay', 0.9), ('activation', None))

    fnode = BnormAddActivation(eps, running_mean, running_var, decay,
                               activation)
    if z is None:
        args = x, gamma, beta
    else:
        args = x, gamma, beta, z
    y, = fnode.apply(args)
    return y
