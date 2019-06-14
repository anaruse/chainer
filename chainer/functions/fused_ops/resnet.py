import numpy

import cupy

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer import function_node
import chainer.functions
from chainer import utils
from chainer.utils import argument
from chainer.utils import conv
from chainer.utils import type_check

from cupy import prof


if cuda.cudnn_enabled:
    _cudnn_version = cuda.cuda.cudnn.getVersion()  # type: ignore
    cudnn = cuda.cudnn
    libcudnn = cuda.libcudnn
    memory = cuda.cuda.memory


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x


_num_SM = [None] * 16


def _get_num_SM():
    dev_id = cupy.cuda.runtime.getDevice()
    if _num_SM[dev_id] is None:
        _num_SM[dev_id] = cupy.cuda.runtime.deviceGetAttribute(
            cupy.cuda.runtime.cudaDevAttrMultiProcessorCount, dev_id)
    return _num_SM[dev_id]


class ResNetBottleNeckFunction(function_node.FunctionNode):

    _supports_nhwc_tensor_layout = True
    cover_all = False
    tensor_layout = 'NHWC'
    dilate = 1
    bn_mode = libcudnn.CUDNN_BATCHNORM_SPATIAL
    # bn_mode = libcudnn.CUDNN_BATCHNORM_SPATIAL_PERSISTENT
    ptr_ph = libcudnn.CUDNN_PTR_16B_ALIGNED

    def __init__(self, running_mean_set, running_var_set,
                 bn_eps=2.5e-5, bn_decay=0.9):
        self.bn_eps = bn_eps
        self.bn_decay = bn_decay
        self.running_mean_set = running_mean_set
        self.running_var_set = running_var_set
        self.stride = (1, 1, 1)
        self.pad = (0, 1, 0)

    def _get_out_size(self, x, W, stride, pad, dilate=1):
        _, h, w, _ = x.shape
        _, kh, kw, _ = W.shape
        out_h = conv.get_conv_outsize(
            h, kh, stride, pad, cover_all=self.cover_all, d=dilate)
        if out_h <= 0:
            raise RuntimeError('Height in the output should be positive.')
        out_w = conv.get_conv_outsize(
            w, kw, stride, pad, cover_all=self.cover_all, d=dilate)
        if out_w <= 0:
            raise RuntimeError('Width in the output should be positive.')
        return out_h, out_w
        
    def forward(self, inputs):
        self.len_inputs = len(inputs)
        x0 = inputs[0]
        W0, gamma0, beta0 = inputs[1:4]
        W1, gamma1, beta1 = inputs[4:7]
        W2, gamma2, beta2 = inputs[7:10]
        self.retain_inputs((0, 1, 2, 3, 4, 5, 6, 7, 8, 9))

        x0 = cuda.cupy.ascontiguousarray(x0)
        W0 = cuda.cupy.ascontiguousarray(W0)
        W1 = cuda.cupy.ascontiguousarray(W1)
        W2 = cuda.cupy.ascontiguousarray(W2)
        gamma0 = cuda.cupy.ascontiguousarray(gamma0)
        gamma1 = cuda.cupy.ascontiguousarray(gamma1)
        gamma2 = cuda.cupy.ascontiguousarray(gamma2)
        beta0 = cuda.cupy.ascontiguousarray(beta0)
        beta1 = cuda.cupy.ascontiguousarray(beta1)
        beta2 = cuda.cupy.ascontiguousarray(beta2)

        x1, scale1, bias1, mean1, invstd1 = self.fw_scale_bias_relu_conv_bnorm(
            0, x0, W0, gamma0, beta0)
        x2, scale2, bias2, mean2, invstd2 = self.fw_scale_bias_relu_conv_bnorm(
            1, x1, W1, gamma1, beta1, scale1, bias1)
        x3, scale3, bias3, mean3, invstd3 = self.fw_scale_bias_relu_conv_bnorm(
            2, x2, W2, gamma2, beta2, scale2, bias2)
        y = self.fw_scale_bias_add_relu(x3, scale3, bias3, x0)

        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.scale1 = scale1
        self.scale2 = scale2
        self.scale3 = scale3
        self.bias1 = bias1
        self.bias2 = bias2
        self.bias3 = bias3
        self.saved_mean = (mean1, mean2, mean3)
        self.saved_invstd = (invstd1, invstd2, invstd3)
        
        self.retain_outputs((0,))
        return y,

    def fw_scale_bias_relu_conv_bnorm(self, lid, x, W, gamma, beta,
                                      scale=None, bias=None):
        '''
        y = conv(relu(x * scale + bias), W)
        o_scale, o_bias = f(y, gamma, beta)
        '''
        n, x_h, x_w, x_c = x.shape
        y_c, _, _, _ = W.shape
        y_h, y_w = self._get_out_size(x, W, self.stride[lid], self.pad[lid])

        y = cuda.cupy.empty((n, y_h, y_w, y_c), dtype=x.dtype)
        o_scale = cuda.cupy.empty((y_c,), dtype=x.dtype)
        o_bias = cuda.cupy.empty_like(o_scale)
        mean = cuda.cupy.empty((y_c,), dtype=gamma.dtype)
        invstd = cuda.cupy.empty_like(mean)
        ysum = cuda.cupy.empty_like(mean)
        ysqsum = cuda.cupy.empty_like(mean)

        x_desc = cudnn.create_tensor_descriptor(
            x, libcudnn.CUDNN_TENSOR_NHWC)
        w_desc = cudnn.create_filter_descriptor(
            W, libcudnn.CUDNN_TENSOR_NHWC)
        gamma_desc = cudnn.create_tensor_descriptor(
            gamma.reshape(1, 1, 1, y_c), libcudnn.CUDNN_TENSOR_NHWC)
        y_desc = cudnn.create_tensor_descriptor(
            y, libcudnn.CUDNN_TENSOR_NHWC)
        o_scale_desc = cudnn.create_tensor_descriptor(
            o_scale.reshape(1, 1, 1, y_c), libcudnn.CUDNN_TENSOR_NHWC)
        ysum_desc = cudnn.create_tensor_descriptor(
            ysum.reshape(1, 1, 1, y_c), libcudnn.CUDNN_TENSOR_NHWC)

        conv_desc = cudnn.create_convolution_descriptor(
            (self.pad[lid], self.pad[lid]),
            (self.stride[lid], self.stride[lid]),
            W.dtype, use_tensor_core=True)
        bn_mode = self.bn_mode
        ptr_ph = self.ptr_ph

        #
        ops = libcudnn.CUDNN_FUSED_SCALE_BIAS_ACTIVATION_CONV_BNSTATS
        const_param = [
            (libcudnn.CUDNN_PARAM_XDESC, x_desc),
            (libcudnn.CUDNN_PARAM_XDATA_PLACEHOLDER, ptr_ph),
            (libcudnn.CUDNN_PARAM_BN_MODE, bn_mode),
            (libcudnn.CUDNN_PARAM_CONV_DESC, conv_desc),
            (libcudnn.CUDNN_PARAM_WDESC, w_desc),
            (libcudnn.CUDNN_PARAM_WDATA_PLACEHOLDER, ptr_ph),
            (libcudnn.CUDNN_PARAM_YDESC, y_desc),
            (libcudnn.CUDNN_PARAM_YDATA_PLACEHOLDER, ptr_ph),
            (libcudnn.CUDNN_PARAM_YSTATS_DESC, ysum_desc),
            (libcudnn.CUDNN_PARAM_YSUM_PLACEHOLDER, ptr_ph),
            (libcudnn.CUDNN_PARAM_YSQSUM_PLACEHOLDER, ptr_ph),
        ]
        var_param = [
            (libcudnn.CUDNN_PTR_XDATA, x),
            (libcudnn.CUDNN_PTR_WDATA, W),
            (libcudnn.CUDNN_PTR_YDATA, y),
            (libcudnn.CUDNN_PTR_YSUM, ysum),
            (libcudnn.CUDNN_PTR_YSQSUM, ysqsum),
        ]
        if scale is not None:
            scale = cuda.cupy.ascontiguousarray(scale)
            bias = cuda.cupy.ascontiguousarray(bias)
            scale_desc = cudnn.create_tensor_descriptor(
                scale.reshape(1, 1, 1, x_c), libcudnn.CUDNN_TENSOR_NHWC)
            act_desc = cudnn.create_activation_descriptor(
                libcudnn.CUDNN_ACTIVATION_RELU)
            const_param.extend([
                (libcudnn.CUDNN_PARAM_BN_EQSCALEBIAS_DESC, scale_desc),
                (libcudnn.CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER, ptr_ph),
                (libcudnn.CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER, ptr_ph),
                (libcudnn.CUDNN_PARAM_ACTIVATION_DESC, act_desc),
            ])
            var_param.extend([
                (libcudnn.CUDNN_PTR_BN_EQSCALE, scale),
                (libcudnn.CUDNN_PTR_BN_EQBIAS, bias),
            ])
        plan = cudnn.create_fused_ops_plan(ops)
        const_pack = cudnn.create_fused_ops_const_param_pack(
            ops, const_param)
        workspace_size = cudnn.make_fused_ops_plan(plan, const_pack)
        max_workspace_size = cudnn.get_max_workspace_size()
        if workspace_size > max_workspace_size:
            msg = ('required workspace size ({}) is larger than max workspace'
                   ' size ({})'.format(workspace_size, max_workspace_size))
            raise RuntimeError(msg)
        workspace = memory.alloc(workspace_size)
        var_param.extend([
            (libcudnn.CUDNN_PTR_WORKSPACE, workspace),
            (libcudnn.CUDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES,
             workspace_size),
        ])
        var_pack = cudnn.create_fused_ops_variant_param_pack(
            ops, var_param)
        cudnn.fused_ops_execute(plan, var_pack)
        del workspace

        #
        ops = libcudnn.CUDNN_FUSED_BN_FINALIZE_STATISTICS_TRAINING
        const_param = [
            (libcudnn.CUDNN_PARAM_BN_MODE, bn_mode),
            (libcudnn.CUDNN_PARAM_YSTATS_DESC, ysum_desc),
            (libcudnn.CUDNN_PARAM_YSUM_PLACEHOLDER, ptr_ph),
            (libcudnn.CUDNN_PARAM_YSQSUM_PLACEHOLDER, ptr_ph),
            (libcudnn.CUDNN_PARAM_BN_SCALEBIAS_MEANVAR_DESC, gamma_desc),
            (libcudnn.CUDNN_PARAM_BN_SCALE_PLACEHOLDER, ptr_ph),
            (libcudnn.CUDNN_PARAM_BN_BIAS_PLACEHOLDER, ptr_ph),
            (libcudnn.CUDNN_PARAM_BN_SAVED_MEAN_PLACEHOLDER, ptr_ph),
            (libcudnn.CUDNN_PARAM_BN_SAVED_INVSTD_PLACEHOLDER, ptr_ph),
            (libcudnn.CUDNN_PARAM_BN_RUNNING_MEAN_PLACEHOLDER, ptr_ph),
            (libcudnn.CUDNN_PARAM_BN_RUNNING_VAR_PLACEHOLDER, ptr_ph),
            (libcudnn.CUDNN_PARAM_BN_EQSCALEBIAS_DESC, o_scale_desc),
            (libcudnn.CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER, ptr_ph),
            (libcudnn.CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER, ptr_ph),
        ]
        acc_count = n * x_h * x_w
        factor = 1 - self.bn_decay
        var_param = [
            (libcudnn.CUDNN_PTR_YSUM, ysum),
            (libcudnn.CUDNN_PTR_YSQSUM, ysqsum),
            (libcudnn.CUDNN_PTR_BN_SCALE, gamma),
            (libcudnn.CUDNN_PTR_BN_BIAS, beta),
            (libcudnn.CUDNN_PTR_BN_SAVED_MEAN, mean),
            (libcudnn.CUDNN_PTR_BN_SAVED_INVSTD, invstd),
            (libcudnn.CUDNN_PTR_BN_RUNNING_MEAN, self.running_mean_set[lid]),
            (libcudnn.CUDNN_PTR_BN_RUNNING_VAR, self.running_var_set[lid]),
            (libcudnn.CUDNN_PTR_BN_EQSCALE, o_scale),
            (libcudnn.CUDNN_PTR_BN_EQBIAS, o_bias),
            (libcudnn.CUDNN_SCALAR_INT64_T_BN_ACCUMULATION_COUNT, acc_count),
            (libcudnn.CUDNN_SCALAR_DOUBLE_BN_EXP_AVG_FACTOR, factor),
            (libcudnn.CUDNN_SCALAR_DOUBLE_BN_EPSILON, self.bn_eps),
        ]
        plan = cudnn.create_fused_ops_plan(ops)
        const_pack = cudnn.create_fused_ops_const_param_pack(
            ops, const_param)
        workspace_size = cudnn.make_fused_ops_plan(plan, const_pack)
        max_workspace_size = cudnn.get_max_workspace_size()
        if workspace_size > max_workspace_size:
            msg = ('required workspace size ({}) is larger than max workspace'
                   ' size ({})'.format(workspace_size, max_workspace_size))
            raise RuntimeError(msg)
        workspace = memory.alloc(workspace_size)
        var_param.extend([
            (libcudnn.CUDNN_PTR_WORKSPACE, workspace),
            (libcudnn.CUDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES,
             workspace_size)
        ])
        var_pack = cudnn.create_fused_ops_variant_param_pack(
            ops, var_param)
        cudnn.fused_ops_execute(plan, var_pack)
        del workspace
        
        return y, o_scale, o_bias, mean, invstd

    def fw_scale_bias_add_relu(self, x, scale, bias, y):
        '''
        z = x * scale + bias + y
        '''
        z = x * scale + bias + y
        z[z <= 0] = 0
        return z

    def backward(self, indexes, grad_outputs):
        inputs = self.get_retained_inputs()
        x0 = inputs[0]
        W0, gamma0, beta0 = inputs[1:4]
        W1, gamma1, beta1 = inputs[4:7]
        W2, gamma2, beta2 = inputs[7:10]
        y, = self.get_retained_outputs()
        gy, = grad_outputs

        x0 = cuda.cupy.ascontiguousarray(x0.data)
        W0 = cuda.cupy.ascontiguousarray(W0.data)
        W1 = cuda.cupy.ascontiguousarray(W1.data)
        W2 = cuda.cupy.ascontiguousarray(W2.data)
        gamma0 = cuda.cupy.ascontiguousarray(gamma0.data)
        gamma1 = cuda.cupy.ascontiguousarray(gamma1.data)
        gamma2 = cuda.cupy.ascontiguousarray(gamma2.data)
        beta0 = cuda.cupy.ascontiguousarray(beta0.data)
        beta1 = cuda.cupy.ascontiguousarray(beta1.data)
        beta2 = cuda.cupy.ascontiguousarray(beta2.data)
        y = cuda.cupy.ascontiguousarray(y.data)
        gy = cuda.cupy.ascontiguousarray(gy.data)

        gx3, g_scale3, g_bias3, _gx0 = self.bw_scale_bias_add_relu(
            self.x3, self.scale3, self.bias3, y, gy)

        ret = self.bw_scale_bias_relu_conv_bnorm(
            2,
            self.x2, self.scale2, self.bias2, W2,
            self.x3, self.scale3, self.bias3,
            gx3, g_scale3, g_bias3)
        gx2, g_scale2, g_bias2, gW2, g_gamma2, g_beta2 = ret
        del gx3
        self.x3 = None

        ret = self.bw_scale_bias_relu_conv_bnorm(
            1,
            self.x1, self.scale1, self.bias1, W1,
            self.x2, self.scale2, self.bias2,
            gx2, g_scale2, g_bias2)
        gx1, g_scale1, g_bias1, gW1, g_gamma1, g_beta1 = ret
        del gx2
        self.x2 = None

        ret = self.bw_scale_bias_relu_conv_bnorm(
            0,
            x0, None, None, W0,
            self.x1, self.scale1, self.bias1,
            gx1, g_scale1, g_bias1)
        gx0, _, _, gW0, g_gamma0, g_beta0 = ret
        del gx1
        self.x1 = None

        gx0 = gx0 + _gx0

        gx0 = chainer.Variable(gx0)
        gW0 = chainer.Variable(gW0)
        gW1 = chainer.Variable(gW1)
        gW2 = chainer.Variable(gW2)
        g_gamma0 = chainer.Variable(g_gamma0)
        g_gamma1 = chainer.Variable(g_gamma1)
        g_gamma2 = chainer.Variable(g_gamma2)
        g_beta0 = chainer.Variable(g_beta0)
        g_beta1 = chainer.Variable(g_beta1)
        g_beta2 = chainer.Variable(g_beta2)
        ret = (gx0,
               gW0, g_gamma0, g_beta0,
               gW1, g_gamma1, g_beta1,
               gW2, g_gamma2, g_beta2)
        return ret
        
    def bw_scale_bias_add_relu(self, x, scale, bias, z, gz):
        '''
        z = x * scale + bias + y
        '''
        _n, _h, _w, _c = x.shape
        gz[z <= 0] = 0
        gy = gz
        g_bias = cupy.sum(gz, axis=(0, 1, 2)).reshape((_c,))
        g_scale = cupy.sum(gz * x, axis=(0, 1, 2)).reshape((_c,))
        # gx = gz * scale  # test(A)
        gx = gz  # test(A)
        return gx, g_scale, g_bias, gy

    def bw_scale_bias_relu_conv_bnorm(self, lid,
                                      x, i_scale, i_bias, W,
                                      y, o_scale, o_bias,
                                      gy, go_scale, go_bias):
        '''
        y = conv(relu(x * i_scale + i_bias), W)
        o_scale, o_bias = f(y, gamma, beta)
        '''
        n, x_h, x_w, x_c = x.shape
        _, y_h, y_w, y_c = y.shape

        gW = cupy.empty_like(W)

        x_desc = cudnn.create_tensor_descriptor(x, libcudnn.CUDNN_TENSOR_NHWC)
        w_desc = cudnn.create_filter_descriptor(W, libcudnn.CUDNN_TENSOR_NHWC)
        y_desc = cudnn.create_tensor_descriptor(y, libcudnn.CUDNN_TENSOR_NHWC)

        conv_desc = cudnn.create_convolution_descriptor(
            (self.pad[lid], self.pad[lid]),
            (self.stride[lid], self.stride[lid]),
            W.dtype, use_tensor_core=True)
        bn_mode = self.bn_mode
        ptr_ph = self.ptr_ph
        
        # compute g_gamma and g_beta
        g_gamma = go_scale * self.saved_invstd[lid]
        g_beta = go_bias.astype(g_gamma.dtype)

        # adjust gy
        _y = (y - self.saved_mean[lid]) * self.saved_invstd[lid]
        gy_dtype = gy.dtype
        gy = gy * o_scale  # test(A)
        gy = gy - (g_beta + g_gamma * _y) / (n * y_h * y_w)
        gy = gy.astype(gy_dtype)
        
        # compute gW
        ops = libcudnn.CUDNN_FUSED_SCALE_BIAS_ACTIVATION_WGRAD
        const_param = [
            (libcudnn.CUDNN_PARAM_XDESC, x_desc),
            (libcudnn.CUDNN_PARAM_XDATA_PLACEHOLDER, ptr_ph),
            (libcudnn.CUDNN_PARAM_BN_MODE, bn_mode),
            (libcudnn.CUDNN_PARAM_CONV_DESC, conv_desc),
            (libcudnn.CUDNN_PARAM_DWDESC, w_desc),
            (libcudnn.CUDNN_PARAM_DWDATA_PLACEHOLDER, ptr_ph),
            (libcudnn.CUDNN_PARAM_DYDESC, y_desc),
            (libcudnn.CUDNN_PARAM_DYDATA_PLACEHOLDER, ptr_ph),
        ]
        var_param = [
            (libcudnn.CUDNN_PTR_XDATA, x),
            (libcudnn.CUDNN_PTR_DWDATA, gW),
            (libcudnn.CUDNN_PTR_DYDATA, gy),
        ]
        if i_scale is not None:
            scale_desc = cudnn.create_tensor_descriptor(
                i_scale.reshape(1, 1, 1, x_c), libcudnn.CUDNN_TENSOR_NHWC)
            act_desc = cudnn.create_activation_descriptor(
                libcudnn.CUDNN_ACTIVATION_RELU)
            const_param.extend([
                (libcudnn.CUDNN_PARAM_BN_EQSCALEBIAS_DESC, scale_desc),
                (libcudnn.CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER, ptr_ph),
                (libcudnn.CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER, ptr_ph),
                (libcudnn.CUDNN_PARAM_ACTIVATION_DESC, act_desc),
            ])
            var_param.extend([
                (libcudnn.CUDNN_PTR_BN_EQSCALE, i_scale),
                (libcudnn.CUDNN_PTR_BN_EQBIAS, i_bias),
            ])
        plan = cudnn.create_fused_ops_plan(ops)
        const_pack = cudnn.create_fused_ops_const_param_pack(
            ops, const_param)
        workspace_size = cudnn.make_fused_ops_plan(plan, const_pack)
        max_workspace_size = cudnn.get_max_workspace_size()
        if workspace_size > max_workspace_size:
            msg = ('required workspace size ({}) is larger than max workspace'
                   ' size ({})'.format(workspace_size, max_workspace_size))
            raise RuntimeError(msg)
        workspace = memory.alloc(workspace_size)
        var_param.extend([
            (libcudnn.CUDNN_PTR_WORKSPACE, workspace),
            (libcudnn.CUDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES,
             workspace_size),
        ])
        var_pack = cudnn.create_fused_ops_variant_param_pack(
            ops, var_param)
        cudnn.fused_ops_execute(plan, var_pack)
        del workspace

        # compute gx
        gx = chainer.functions.deconvolution_2d(
            gy, W, stride=self.stride[lid], pad=self.pad[lid],
            outsize=(x_h, x_w),
            tensor_layout=self.tensor_layout).data

        # compute gx, gi_scale and gi_bias
        if i_scale is None:
            gi_bias = None
            gi_scale = None
        else:
            _x = x * i_scale + i_bias
            gx[_x <= 0] = 0
            gi_bias = cupy.sum(gx, axis=(0, 1, 2)).reshape((x_c,))
            gi_scale = cupy.sum(gx * x, axis=(0, 1, 2)).reshape((x_c,))
            # gx = gx * i_scale  # test(A)
        
        return gx, gi_scale, gi_bias, gW, g_gamma, g_beta


def resnet_bottle_neck(x, W_set, gamma_set, beta_set,
                       running_mean_set, running_var_set,
                       bn_eps=2.5e-5, bn_decay=0.9):
    fnode = ResNetBottleNeckFunction(running_mean_set, running_var_set,
                                     bn_eps, bn_decay)
    args = (x,
            W_set[0], gamma_set[0], beta_set[0],
            W_set[1], gamma_set[1], beta_set[1],
            W_set[2], gamma_set[2], beta_set[2])
    y, = fnode.apply(args)
    return y
