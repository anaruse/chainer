import numpy

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer import function_node
import chainer.functions
from chainer import utils
from chainer.utils import argument
from chainer.utils import conv
from chainer.utils import type_check


if cuda.cudnn_enabled:
    _cudnn_version = cuda.cuda.cudnn.getVersion()  # type: ignore
    cudnn = cuda.cudnn
    libcudnn = cuda.libcudnn
    memory = cuda.cuda.memory
    

def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x


class FusedScaleBiasActConvBnstatsFunction(function_node.FunctionNode):

    _supports_nhwc_tensor_layout = True
    cover_all = False
    dy = 1
    dx = 1
    
    def __init__(self, scale=None, bias=None, stride=1, pad=0,
                 bn_eps=2.5e-5, bn_decay=0.9,
                 running_mean=None, running_var=None):

        if isinstance(scale, chainer.Variable):
            scale = scale.data
        if isinstance(bias, chainer.Variable):
            bias = bias.data
        self.scale = scale
        self.bias = bias
        self.sy, self.sx = _pair(stride)
        self.ph, self.pw = _pair(pad)
        self.bn_eps = bn_eps
        self.bn_decay = bn_decay
        self.running_mean = running_mean
        self.running_var = running_var

    def _get_out_size(self, x, W):
        _, h, w, _ = x.shape
        _, kh, kw, _ = W.shape
        out_h = conv.get_conv_outsize(
            h, kh, self.sy, self.ph, cover_all=self.cover_all, d=self.dy)
        if out_h <= 0:
            raise RuntimeError('Height in the output should be positive.')
        out_w = conv.get_conv_outsize(
            w, kw, self.sx, self.pw, cover_all=self.cover_all, d=self.dx)
        if out_w <= 0:
            raise RuntimeError('Width in the output should be positive.')
        return out_h, out_w

    def forward_gpu(self, inputs):
        self.retain_inputs((0, 1, 2, 3))
        x, W, gamma, beta = inputs

        n, in_h, in_w, in_c = x.shape
        out_c, _, _, _ = W.shape
        out_h, out_w = self._get_out_size(x, W)
        
        x = cuda.cupy.ascontiguousarray(x)
        W = cuda.cupy.ascontiguousarray(W)
        gamma = cuda.cupy.ascontiguousarray(gamma)
        beta = cuda.cupy.ascontiguousarray(beta)

        x_desc = cudnn.create_tensor_descriptor(
            x, libcudnn.CUDNN_TENSOR_NHWC)
        w_desc = cudnn.create_filter_descriptor(
            W, libcudnn.CUDNN_TENSOR_NHWC)
        gamma_desc = cudnn.create_tensor_descriptor(
            gamma.reshape(1, 1, 1, out_c), libcudnn.CUDNN_TENSOR_NHWC)
        y = cuda.cupy.empty((n, out_h, out_w, out_c), dtype=x.dtype)
        y_desc = cudnn.create_tensor_descriptor(
            y, libcudnn.CUDNN_TENSOR_NHWC)
        ysum = cuda.cupy.empty((1, 1, 1, out_c), dtype=numpy.float32)
        ysqsum = cuda.cupy.empty_like(ysum)
        self.saved_mean = cuda.cupy.empty_like(ysum)
        self.saved_invstd = cuda.cupy.empty_like(ysum)
        ysum_desc = cudnn.create_tensor_descriptor(
            ysum, libcudnn.CUDNN_TENSOR_NHWC)
        out_scale = cuda.cupy.empty((1, 1, 1, out_c), dtype=numpy.float16)
        out_bias = cuda.cupy.empty_like(out_scale)
        out_scale_desc = cudnn.create_tensor_descriptor(
            out_scale, libcudnn.CUDNN_TENSOR_NHWC)

        conv_desc = cudnn.create_convolution_descriptor(
            (self.ph, self.pw), (self.sy, self.sx), W.dtype,
            use_tensor_core=True)
        bn_mode = libcudnn.CUDNN_BATCHNORM_SPATIAL_PERSISTENT
        ptr_ph = libcudnn.CUDNN_PTR_16B_ALIGNED

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
        if self.scale is not None:
            scale = cuda.cupy.ascontiguousarray(self.scale)
            bias = cuda.cupy.ascontiguousarray(self.bias)
            scale_desc = cudnn.create_tensor_descriptor(
                scale.reshape(1, 1, 1, in_c), libcudnn.CUDNN_TENSOR_NHWC)
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
            (libcudnn.CUDNN_PARAM_BN_EQSCALEBIAS_DESC, out_scale_desc),
            (libcudnn.CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER, ptr_ph),
            (libcudnn.CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER, ptr_ph),
        ]
        acc_count = n * in_h * in_w
        var_param = [
            (libcudnn.CUDNN_PTR_YSUM, ysum),
            (libcudnn.CUDNN_PTR_YSQSUM, ysqsum),
            (libcudnn.CUDNN_PTR_BN_SCALE, gamma),
            (libcudnn.CUDNN_PTR_BN_BIAS, beta),
            (libcudnn.CUDNN_PTR_BN_SAVED_MEAN, self.saved_mean),
            (libcudnn.CUDNN_PTR_BN_SAVED_INVSTD, self.saved_invstd),
            (libcudnn.CUDNN_PTR_BN_RUNNING_MEAN, self.running_mean),
            (libcudnn.CUDNN_PTR_BN_RUNNING_VAR, self.running_var),
            (libcudnn.CUDNN_PTR_BN_EQSCALE, out_scale),
            (libcudnn.CUDNN_PTR_BN_EQBIAS, out_bias),
            (libcudnn.CUDNN_SCALAR_INT64_T_BN_ACCUMULATION_COUNT, acc_count),
            (libcudnn.CUDNN_SCALAR_DOUBLE_BN_EXP_AVG_FACTOR, 0.1),
            (libcudnn.CUDNN_SCALAR_DOUBLE_BN_EPSILON, 2e-5),
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

        return y, out_scale, out_bias


def fused_scale_bias_act_conv_bnstats(x, W,
                                      gamma, beta,
                                      scale=None, bias=None,
                                      stride=1, pad=0,
                                      bn_eps=2e-5, bn_decay=0.9,
                                      running_mean=None, running_var=None):

    fnode = FusedScaleBiasActConvBnstatsFunction(
        scale, bias, stride, pad, bn_eps, bn_decay, running_mean, running_var)
    return fnode.apply((x, W, gamma, beta))
