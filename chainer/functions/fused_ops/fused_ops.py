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


class FusedScaleBiasActConvBnFunction(function_node.FunctionNode):

    _supports_nhwc_tensor_layout = True
    cover_all = False
    dy = 1
    dx = 1
    tensor_layout = 'NHWC'
    
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

        #print('# x.shape: {}'.format(x.shape))
        #print('# y.shape: {}'.format(y.shape))

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
        factor = 1 - self.bn_decay
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

        self.retain_outputs((0, 1))
        return y, out_scale, out_bias

    # @prof.TimeRangeDecorator(sync=True)
    def backward(self, indexes, grad_outputs):
        x, W, gamma, beta = self.get_retained_inputs()
        y, out_scale = self.get_retained_outputs()
        gy = grad_outputs[0]

        #print('# indexes: {}'.format(indexes))
        #print('# len(grad_outputs): {}'.format(len(grad_outputs)))

        n, in_h, in_w, in_c = x.shape
        _, out_h, out_w, out_c = gy.shape

        x = cuda.cupy.ascontiguousarray(x.data)
        W = cuda.cupy.ascontiguousarray(W.data)
        gamma = cuda.cupy.ascontiguousarray(gamma.data)
        beta = cuda.cupy.ascontiguousarray(beta.data)
        gy = cuda.cupy.ascontiguousarray(gy.data)
        y = cuda.cupy.ascontiguousarray(y.data)
        out_scale = cuda.cupy.ascontiguousarray(out_scale.data)

        gW = cuda.cupy.empty_like(W)

        x_desc = cudnn.create_tensor_descriptor(
            x, libcudnn.CUDNN_TENSOR_NHWC)
        w_desc = cudnn.create_filter_descriptor(
            W, libcudnn.CUDNN_TENSOR_NHWC)
        y_desc = cudnn.create_tensor_descriptor(
            gy, libcudnn.CUDNN_TENSOR_NHWC)

        conv_desc = cudnn.create_convolution_descriptor(
            (self.ph, self.pw), (self.sy, self.sx), W.dtype,
            use_tensor_core=True)
        bn_mode = libcudnn.CUDNN_BATCHNORM_SPATIAL_PERSISTENT
        ptr_ph = libcudnn.CUDNN_PTR_16B_ALIGNED

        # **** ggamma, gbeta ****
        ggamma, gbeta, gy = cupy_compute_grad_gamma_beta_y(
            gamma, beta, y, gy, out_scale, self.saved_mean, self.saved_invstd)

        # **** gW ****
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

        # **** gx ****
        gx = chainer.functions.deconvolution_2d(
            gy, W, stride=(self.sy, self.sx), pad=(self.ph, self.pw),
            outsize=(in_h, in_w), dilate=(self.dy, self.dx),
            tensor_layout=self.tensor_layout).data
        if self.scale is not None:
            gx = cupy_adjust_grad_x(x, self.scale, self.bias, gx)

        gx = chainer.Variable(gx)
        gW = chainer.Variable(gW)
        ggamma = chainer.Variable(ggamma)
        gbeta = chainer.Variable(gbeta)

        return gx, gW, ggamma, gbeta


def cupy_adjust_grad_x(x, scale, bias, gx):
    _n, _h, _w, _c = x.shape

    if False:
        # reference implementation
        gx = scale * gx
        _x = scale * x + bias
        gx[_x < 0] = 0
        return gx

    block_size = 512
    n_blocks = (_n * _h * _w * _c + block_size - 1) // block_size
    _cupy_adjust_grad_x()(
        (n_blocks,), (block_size,),
        (x, scale, bias, gx, _n, _h, _w, _c))
    return gx


def _cupy_adjust_grad_x():
    return cupy.RawKernel(r'''
#include <cuda_fp16.h>
    extern "C" __global__
    void cupy_adjust_grad_x(
        const half *x, const half *scale, const half *bias,
        half *gx,
        int _n, int _h, int _w, int _c)
    {
        int i = threadIdx.x + blockDim.x * blockIdx.x;
        int num_elements = _n * _h * _w * _c;
        if (i >= num_elements) return;
        int c = i % _c;

        float _x_ = (float)x[i] * (float)scale[c] + (float)bias[c];
        if (_x_ > 0.0) {
            gx[i] = (half)((float)gx[i] * (float)scale[c]);
        }
        else {
            gx[i] = (half)0;
        }
    }
    ''', 'cupy_adjust_grad_x')


def cupy_compute_grad_gamma_beta_y(gamma, beta, y, gy, scale, mean, invstd):
    _n, _h, _w, _c = y.shape

    if False:
        # reference implementation
        _scale = scale.astype(numpy.float32)

        ggamma = gy * y
        ggamma = ggamma.astype(numpy.float32)
        ggamma = cuda.cupy.sum(ggamma, axis=(0, 1, 2))
        ggamma = ggamma / _scale * invstd
        ggamma = ggamma * invstd
        ggamma = ggamma.reshape((_c,))

        gbeta = gy
        gbeta = cuda.cupy.sum(gbeta, axis=(0, 1, 2))
        gbeta = gbeta / _scale
        gbeta = gbeta.reshape((_c,))

        _y = (y - mean) * invstd
        _gy = gy / _scale * invstd * gamma
        _gy = _gy - (gbeta + ggamma * _y) / (_n * _h * _w)
        _gy = _gy.astype(numpy.float16)

        return ggamma, gbeta, _gy

    ggamma = cupy.zeros((_c,), dtype=numpy.float32)
    gbeta = cupy.zeros_like(ggamma)
    block_size = 1024
    n_blocks = _get_num_SM()
    _cupy_compute_grad_gamma_beta()(
        (n_blocks,), (block_size,),
        (y, gy, scale, invstd,
         ggamma, gbeta,
         _n, _h, _w, _c))

    _gy = cupy.empty_like(gy)
    block_size = 256
    n_blocks = (_n * _h * _w * _c + block_size - 1) // block_size
    _cupy_adjust_grad_y()(
        (n_blocks,), (block_size,),
        (gamma, ggamma, beta, gbeta, y, gy, mean, invstd, scale, _gy,
         _n, _h, _w, _c))

    return ggamma, gbeta, _gy


def _cupy_adjust_grad_y():
    return cupy.RawKernel(r'''
#include <cuda_fp16.h>
    extern "C" __global__
    void cupy_adjust_grad_y(
        const float *gamma, const float *ggamma,
        const float *beta, const float *gbeta,
        const half *y, const half *gy,
        const float *mean, const float *invstd,
        const half *scale,
        half *out_gy,
        int _n, int _h, int _w, int _c)
    {
        int tid = threadIdx.x + blockDim.x * blockIdx.x;
        int num_threads = blockDim.x * gridDim.x;
        int num_elements = _n * _h * _w * _c;

        for (int i = tid; i < num_elements; i += num_threads) {
            int c = i % _c;
            float _y = ((float)y[i] - mean[c]) * invstd[c];
            float _gy = (float)gy[i] / (float)scale[c] * invstd[c] * gamma[c]
                - (gbeta[c] + ggamma[c] * _y) / (_n * _h * _w);
            out_gy[i] = (half)_gy;
        }
    }
    ''', 'cupy_adjust_grad_y')


def _cupy_compute_grad_gamma_beta():
    return cupy.RawKernel(r'''
#include <cuda_fp16.h>
    extern "C" __global__
    void cupy_compute_grad_gamma_beta(
        const half *y, const half *gy, const half *scale, const float *invstd,
        float *ggamma, float *gbeta,
        int _n, int _h, int _w, int _c)
    {
        __shared__ float sm_ggamma[2048];
        __shared__ float sm_gbeta[2048];

        for (int i = threadIdx.x; i < _c; i += blockDim.x) {
            sm_ggamma[i] = 0.0;
            sm_gbeta[i] = 0.0;
        }
        __syncthreads();

        int tid = threadIdx.x + blockDim.x * blockIdx.x;
        int num_threads = blockDim.x * gridDim.x;
        int num_elements = _n * _h * _w * _c;
        int pre_c = -1;
        float my_ggamma = 0.0;
        float my_gbeta = 0.0;
        for (int i = tid; i < num_elements; i += num_threads) {
            int cur_c = i % _c;
            if (pre_c >= 0 && pre_c != cur_c) {
                my_ggamma /= (float)scale[pre_c];
                my_ggamma *= invstd[pre_c];
                my_gbeta /= (float)scale[pre_c];
                atomicAdd(&sm_ggamma[pre_c], my_ggamma);
                atomicAdd(&sm_gbeta[pre_c], my_gbeta);
                my_ggamma = 0.0;
                my_gbeta = 0.0;
            }
            my_ggamma += (float)y[i] * (float)gy[i];
            my_gbeta += (float)gy[i];
            pre_c = cur_c;
        }
        if (pre_c >= 0) {
            my_ggamma /= (float)scale[pre_c];
            my_ggamma *= invstd[pre_c];
            my_gbeta /= (float)scale[pre_c];
            atomicAdd(&sm_ggamma[pre_c], my_ggamma);
            atomicAdd(&sm_gbeta[pre_c], my_gbeta);
            my_ggamma = 0.0;
            my_gbeta = 0.0;
        }

        __syncthreads();
        for (int i = threadIdx.x; i < _c; i += blockDim.x) {
            atomicAdd(&ggamma[i], sm_ggamma[i]);
            atomicAdd(&gbeta[i], sm_gbeta[i]);
        }
    }
    ''', 'cupy_compute_grad_gamma_beta')


class FusedScaleBiasActConvBnInferenceFunction(function_node.FunctionNode):

    _supports_nhwc_tensor_layout = True
    cover_all = False
    dy = 1
    dx = 1
    tensor_layout = 'NHWC'

    def __init__(self, scale=None, bias=None, stride=1, pad=0, bn_eps=2e-5,
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
        ]
        var_param = [
            (libcudnn.CUDNN_PTR_XDATA, x),
            (libcudnn.CUDNN_PTR_WDATA, W),
            (libcudnn.CUDNN_PTR_YDATA, y),
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
        ops = libcudnn.CUDNN_FUSED_BN_FINALIZE_STATISTICS_INFERENCE
        const_param = [
            (libcudnn.CUDNN_PARAM_BN_MODE, bn_mode),
            (libcudnn.CUDNN_PARAM_BN_SCALEBIAS_MEANVAR_DESC, gamma_desc),
            (libcudnn.CUDNN_PARAM_BN_SCALE_PLACEHOLDER, ptr_ph),
            (libcudnn.CUDNN_PARAM_BN_BIAS_PLACEHOLDER, ptr_ph),
            (libcudnn.CUDNN_PARAM_BN_RUNNING_MEAN_PLACEHOLDER, ptr_ph),
            (libcudnn.CUDNN_PARAM_BN_RUNNING_VAR_PLACEHOLDER, ptr_ph),
            (libcudnn.CUDNN_PARAM_BN_EQSCALEBIAS_DESC, out_scale_desc),
            (libcudnn.CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER, ptr_ph),
            (libcudnn.CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER, ptr_ph),
        ]
        var_param = [
            (libcudnn.CUDNN_PTR_BN_SCALE, gamma),
            (libcudnn.CUDNN_PTR_BN_BIAS, beta),
            (libcudnn.CUDNN_PTR_BN_RUNNING_MEAN, self.running_mean),
            (libcudnn.CUDNN_PTR_BN_RUNNING_VAR, self.running_var),
            (libcudnn.CUDNN_PTR_BN_EQSCALE, out_scale),
            (libcudnn.CUDNN_PTR_BN_EQBIAS, out_bias),
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

        return y, out_scale, out_bias


def fused_scale_bias_act_conv_bn(
        x, W, gamma, beta, scale=None, bias=None, stride=1, pad=0,
        bn_eps=2e-5, bn_decay=0.9, running_mean=None, running_var=None):

    fnode = FusedScaleBiasActConvBnFunction(
        scale, bias, stride, pad, bn_eps, bn_decay, running_mean, running_var)
    return fnode.apply((x, W, gamma, beta))


def fused_scale_bias_act_conv_bn_inference(
        x, W, gamma, beta, scale=None, bias=None, stride=1, pad=0,
        bn_eps=2e-5, running_mean=None, running_var=None):

    fnode = FusedScaleBiasActConvBnInferenceFunction(
        scale, bias, stride, pad, bn_eps, running_mean, running_var)
    return fnode.apply((x, W, gamma, beta))
