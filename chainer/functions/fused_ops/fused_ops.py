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
    
    def __init__(self, stride=1, pad=0,
                 bn_eps=2.5e-5, bn_decay=0.9,
                 running_mean=None, running_var=None):
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
        self.len_inputs = len(inputs)
        if self.len_inputs == 4:
            self.retain_inputs((0, 1, 2, 3))
            x, W, gamma, beta = inputs
            scale, bias = None, None
        else:
            self.retain_inputs((0, 1, 2, 3, 4, 5))
            x, W, gamma, beta, scale, bias = inputs

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
        # bn_mode = libcudnn.CUDNN_BATCHNORM_SPATIAL_PERSISTENT
        bn_mode = libcudnn.CUDNN_BATCHNORM_SPATIAL
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
        if scale is not None:
            scale = cuda.cupy.ascontiguousarray(scale)
            bias = cuda.cupy.ascontiguousarray(bias)
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

        out_scale = out_scale.reshape((out_c,))
        out_bias = out_bias.reshape((out_c,))

        self.retain_outputs((0, 1, 2))
        return y, out_scale, out_bias

    # @prof.TimeRangeDecorator(sync=True)
    def backward(self, indexes, grad_outputs):
        if self.len_inputs == 4:
            x, W, gamma, beta = self.get_retained_inputs()
            scale, bias = None, None
        else:
            x, W, gamma, beta, scale, bias = self.get_retained_inputs()
        y, out_scale, out_bias = self.get_retained_outputs()
        gy, g_out_scale, g_out_bias = grad_outputs

        n, in_h, in_w, in_c = x.shape
        _, out_h, out_w, out_c = gy.shape

        x = cuda.cupy.ascontiguousarray(x.data)
        W = cuda.cupy.ascontiguousarray(W.data)
        gamma = cuda.cupy.ascontiguousarray(gamma.data)
        beta = cuda.cupy.ascontiguousarray(beta.data)
        y = cuda.cupy.ascontiguousarray(y.data)
        out_scale = cuda.cupy.ascontiguousarray(out_scale.data)
        out_bias = cuda.cupy.ascontiguousarray(out_bias.data)
        gy = cuda.cupy.ascontiguousarray(gy.data)
        g_out_scale = cuda.cupy.ascontiguousarray(g_out_scale.data)
        g_out_bias = cuda.cupy.ascontiguousarray(g_out_bias.data)

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
        # bn_mode = libcudnn.CUDNN_BATCHNORM_SPATIAL_PERSISTENT
        bn_mode = libcudnn.CUDNN_BATCHNORM_SPATIAL
        ptr_ph = libcudnn.CUDNN_PTR_16B_ALIGNED

        # **** gy, ggamma, gbeta ****
        gy, ggamma, gbeta = self._compute_grad_y_gamma_beta(
            y, gy, gamma, g_out_scale, g_out_bias)

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
        if scale is not None:
            scale = cuda.cupy.ascontiguousarray(scale.data)
            bias = cuda.cupy.ascontiguousarray(bias.data)
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
        if scale is None:
            gx = chainer.Variable(gx)
            gW = chainer.Variable(gW)
            ggamma = chainer.Variable(ggamma)
            gbeta = chainer.Variable(gbeta)
            return gx, gW, ggamma, gbeta

        # **** gx, g_scale, g_bias ****
        gx, g_scale, g_bias = self._compute_grad_x_scale_bias(
            x, gx, scale, bias)
        gx = chainer.Variable(gx)
        gW = chainer.Variable(gW)
        ggamma = chainer.Variable(ggamma)
        gbeta = chainer.Variable(gbeta)
        g_scale = chainer.Variable(g_scale)
        g_bias = chainer.Variable(g_bias)
        return gx, gW, ggamma, gbeta, g_scale, g_bias

    def _compute_grad_y_gamma_beta(self, y, gy, gamma, g_scale, g_bias):
        _n, _h, _w, _c = y.shape
        g_gamma = g_scale * self.saved_invstd
        g_gamma = g_gamma.reshape((_c,))
        g_beta = g_bias.astype(numpy.float32).reshape((_c,))
        if False:
            # reference
            _y = (y - self.saved_mean) * self.saved_invstd
            _gy = gy - (g_beta + g_gamma * _y) / (_n * _h * _w)
            _gy = _gy.astype(gy.dtype)
            return _gy, g_gamma, g_beta

        _gy = cupy.empty_like(gy)
        block_size = 256
        n_blocks = (_n * _h * _w * _c + block_size - 1) // block_size
        self._cupy_adjust_grad_y()(
            (n_blocks,), (block_size,),
            (y, gy, gamma, g_gamma, g_beta, self.saved_mean, self.saved_invstd,
             _gy,
             _n, _h, _w, _c))
        return _gy, g_gamma, g_beta

    def _compute_grad_x_scale_bias(self, x, gx, scale, bias):
        _n, _h, _w, _c = x.shape
        if False:
            # reference
            _x = scale * x + bias
            _gx = gx
            _gx[_x < 0] = 0
            g_bias = _gx
            g_bias = cupy.sum(g_bias, axis=(0, 1, 2)).reshape((_c,))
            g_scale = _gx * x
            g_scale = cupy.sum(g_scale, axis=(0, 1, 2)).reshape((_c,))
            _gx = scale * _gx
            return _gx, g_scale, g_bias

        _gx = cupy.empty_like(gx)
        g_scale = cupy.zeros((_c,), dtype=scale.dtype)
        g_bias = cupy.zeros_like(g_scale)

        block_size = 1024
        n_blocks = _get_num_SM()
        self._cupy_compute_grad_x_scale_bias()(
            (n_blocks,), (block_size,),
            (x, gx, scale, bias,
             _gx, g_scale, g_bias,
             _n, _h, _w, _c // 2))
        return _gx, g_scale, g_bias

    def _cupy_adjust_grad_y(self):
        return cupy.RawKernel(r'''
#include <cuda_fp16.h>
    extern "C" __global__
    void cupy_adjust_grad_y(
        const half *y, const half *gy,
        const float *gamma, const float *g_gamma, const float *g_beta,
        const float *mean, const float *invstd,
        half *o_gy,
        int _n, int _h, int _w, int _c)
        {
            int i = threadIdx.x + blockDim.x * blockIdx.x;
            int num_elements = _n * _h * _w * _c;
            if (i >= num_elements) return;
            int c = i % _c;
            float _y = ((float)y[i] - mean[c]) * invstd[c];
            float _gy = (float)gy[i]
                      - (g_beta[c] + g_gamma[c] * _y) / (_n * _h * _w);
            o_gy[i] = (half)_gy;
        }
        ''', 'cupy_adjust_grad_y')

    def _cupy_compute_grad_x_scale_bias(self):
        return cupy.RawKernel(r'''
#include <cuda_fp16.h>
    extern "C" __global__
    void cupy_compute_grad_x_scale_bias(
        const half2 *x, const half2 *gx, const half2 *scale, const half2 *bias,
        half2 *o_gx, half2 *g_scale, half2 *g_bias,
        int _n, int _h, int _w, int _c)
    {
        __shared__ float2 sm_g_bias[1024];
        __shared__ float2 sm_g_scale[1024];
        for (int i = threadIdx.x; i < _c; i += blockDim.x) {
            sm_g_bias[i].x = 0.0;
            sm_g_bias[i].y = 0.0;
            sm_g_scale[i].x = 0.0;
            sm_g_scale[i].y = 0.0;
        }
        __syncthreads();

        int tid = threadIdx.x + blockDim.x * blockIdx.x;
        int num_threads = blockDim.x * gridDim.x;
        int num_elements = _n * _h * _w * _c;
        int pre_c = -1;
        float2  my_g_bias;
        float2  my_g_scale;
        float2  my_scale;
        float2  my_bias;
        half2  _h2_tmp;
        for (int i = tid; i < num_elements; i += num_threads) {
            int cur_c = i % _c;
            if (pre_c != cur_c) {
                if (pre_c >= 0) {
                    atomicAdd(&sm_g_bias[pre_c].x,  my_g_bias.x);
                    atomicAdd(&sm_g_bias[pre_c].y,  my_g_bias.y);
                    atomicAdd(&sm_g_scale[pre_c].x, my_g_scale.x);
                    atomicAdd(&sm_g_scale[pre_c].y, my_g_scale.y);
                }
                my_g_bias.x  = 0.0;
                my_g_bias.y  = 0.0;
                my_g_scale.x = 0.0;
                my_g_scale.y = 0.0;
                _h2_tmp = scale[cur_c];
                my_scale.x = _h2_tmp.x;
                my_scale.y = _h2_tmp.y;
                _h2_tmp = bias[cur_c];
                my_bias.x =  _h2_tmp.x;
                my_bias.y =  _h2_tmp.y;
            }
            pre_c = cur_c;

            float2 _x;
            _x.x = my_scale.x * (float)x[i].x + my_bias.x;
            _x.y = my_scale.y * (float)x[i].y + my_bias.y;
            float2 _gx = {(float)gx[i].x, (float)gx[i].y};
            if (_x.x <= 0.0) _gx.x = 0.0;
            if (_x.y <= 0.0) _gx.y = 0.0;
            my_g_bias.x  += _gx.x;
            my_g_bias.y  += _gx.y;
            my_g_scale.x += _gx.x * _x.x;
            my_g_scale.y += _gx.y * _x.y;

            _h2_tmp.x = (half)(_gx.x * my_scale.x);
            _h2_tmp.y = (half)(_gx.y * my_scale.y);
            o_gx[i] = _h2_tmp;
        }
        if (pre_c >= 0) {
            atomicAdd(&sm_g_bias[pre_c].x,  my_g_bias.x);
            atomicAdd(&sm_g_bias[pre_c].y,  my_g_bias.y);
            atomicAdd(&sm_g_scale[pre_c].x, my_g_scale.x);
            atomicAdd(&sm_g_scale[pre_c].y, my_g_scale.y);
        }

        __syncthreads();
        for (int i = threadIdx.x; i < _c; i += blockDim.x) {
            _h2_tmp.x = (half)sm_g_scale[i].x;
            _h2_tmp.y = (half)sm_g_scale[i].y;
            atomicAdd(&g_scale[i], _h2_tmp);
            _h2_tmp.x = (half)sm_g_bias[i].x;
            _h2_tmp.y = (half)sm_g_bias[i].y;
            atomicAdd(&g_bias[i], _h2_tmp);
        }
    }
    ''', 'cupy_compute_grad_x_scale_bias')


class FusedScaleBiasActConvBnInferenceFunction(function_node.FunctionNode):

    _supports_nhwc_tensor_layout = True
    cover_all = False
    dy = 1
    dx = 1
    tensor_layout = 'NHWC'

    def __init__(self, stride=1, pad=0, bn_eps=2e-5,
                 running_mean=None, running_var=None):
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
        self.len_inputs = len(inputs)
        if self.len_inputs == 4:
            self.retain_inputs((0, 1, 2, 3))
            x, W, gamma, beta = inputs
            scale, bias = None, None
        else:
            self.retain_inputs((0, 1, 2, 3, 4, 5))
            x, W, gamma, beta, scale, bias = inputs

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
        # bn_mode = libcudnn.CUDNN_BATCHNORM_SPATIAL_PERSISTENT
        bn_mode = libcudnn.CUDNN_BATCHNORM_SPATIAL
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
        if scale is not None:
            scale = cuda.cupy.ascontiguousarray(scale)
            bias = cuda.cupy.ascontiguousarray(bias)
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
        x, scale, bias, W, gamma, beta, stride=1, pad=0,
        bn_eps=2e-5, bn_decay=0.9, running_mean=None, running_var=None):

    fnode = FusedScaleBiasActConvBnFunction(
        stride, pad, bn_eps, bn_decay, running_mean, running_var)
    if scale is None:
        args = (x, W, gamma, beta)
    else:
        args = (x, W, gamma, beta, scale, bias)
    return fnode.apply(args)


def fused_scale_bias_act_conv_bn_inference(
        x, scale, bias, W, gamma, beta, stride=1, pad=0,
        bn_eps=2e-5, running_mean=None, running_var=None):

    fnode = FusedScaleBiasActConvBnInferenceFunction(
        stride, pad, bn_eps, running_mean, running_var)
    if scale is None:
        args = (x, W, gamma, beta)
    else:
        args = (x, W, gamma, beta, scale, bias)
    return fnode.apply(args)


# ************************************************************


class FusedScaleBiasAddReluFunction(function_node.FunctionNode):

    _supports_nhwc_tensor_layout = True

    def __init__(self):
        pass

    def forward_gpu(self, inputs):
        x, scale, bias, y = inputs
        self.retain_inputs((0, 1))
        self.retain_outputs((0,))
        if False:
            # reference
            z = scale * x + bias + y
            z[z <= 0] = 0
            return z,

        _n, _h, _w, _c = x.shape
        x = cuda.cupy.ascontiguousarray(x)
        y = cuda.cupy.ascontiguousarray(y)
        bias = cuda.cupy.ascontiguousarray(bias)
        scale = cuda.cupy.ascontiguousarray(scale)
        z = cupy.empty_like(x)
        block_size = 256
        n_blocks = (_n * _h * _w * _c // 2 + block_size - 1) // block_size
        self._cupy_forward()(
            (n_blocks,), (block_size,),
            (x, scale, bias, y,
             z,
             _n, _h, _w, _c // 2))
        return z,

    def _cupy_forward(self):
        return cupy.RawKernel(r'''
#include <cuda_fp16.h>
    extern "C" __global__
    void cupy_FSBAR_forward(
        const half2 *x, const half2 *scale, const half2 *bias, const half2 *y,
        half2 *z,
        int _n, int _h, int _w, int _c)
    {
        int i = threadIdx.x + blockDim.x * blockIdx.x;
        int num_elements = _n * _h * _w * _c;
        if (i >= num_elements) return;
        int c = i % _c;
        half2 _x = x[i];
        half2 _y = y[i];
        half2 _bias = bias[c];
        half2 _scale = scale[c];
        float2 _z;
        _z.x = (float)_x.x * (float)_scale.x + (float)_bias.x + (float)_y.x;
        _z.y = (float)_x.y * (float)_scale.y + (float)_bias.y + (float)_y.y;
        if (_z.x < 0) _z.x = 0.0;
        if (_z.y < 0) _z.y = 0.0;
        half2 h2_z;
        h2_z.x = (half)_z.x;
        h2_z.y = (half)_z.y;
        z[i] = h2_z;
    }
    ''', 'cupy_FSBAR_forward')

    def backward(self, indexes, grad_outputs):
        gz, = grad_outputs
        x, scale = self.get_retained_inputs()
        z, = self.get_retained_outputs()
        _n, _h, _w, _c = x.shape

        x = cuda.cupy.ascontiguousarray(x.data)
        z = cuda.cupy.ascontiguousarray(z.data)
        gz = cuda.cupy.ascontiguousarray(gz.data)
        scale = cuda.cupy.ascontiguousarray(scale.data)

        if False:
            # reference
            _gz = gz
            _gz[z <= 0] = 0
            gy = _gz
            g_bias = _gz
            g_bias = cupy.sum(g_bias, axis=(0, 1, 2)).reshape((_c,))
            g_scale = _gz * x
            g_scale = cupy.sum(g_scale, axis=(0, 1, 2)).reshape((_c,))
            gx = scale * _gz

            gx = chainer.Variable(gx)
            gy = chainer.Variable(gy)
            g_scale = chainer.Variable(g_scale)
            g_bias = chainer.Variable(g_bias)
            return gx, g_scale, g_bias, gy

        gx = cupy.empty_like(x)
        gy = cupy.empty_like(gx)
        g_scale = cupy.zeros((_c,), dtype=scale.dtype)
        g_bias = cupy.zeros_like(g_scale)
        block_size = 1024
        n_blocks = _get_num_SM()
        assert _c <= 2048
        self._cupy_backward()(
            (n_blocks,), (block_size,),
            (x, z, gz, scale,
             gx, g_scale, g_bias, gy,
             _n, _h, _w, _c // 2))

        gx = chainer.Variable(gx)
        gy = chainer.Variable(gy)
        g_bias = chainer.Variable(g_bias)
        g_scale = chainer.Variable(g_scale)
        return gx, g_scale, g_bias, gy

    def _cupy_backward(self):
        return cupy.RawKernel(r'''
#include <cuda_fp16.h>
    extern "C" __global__
    void cupy_FSBAR_backward(
        const half2 *x, const half2 *z, const half2 *gz, const half2 *scale,
        half2 *gx, half2 *g_scale, half2 *g_bias, half2 *gy,
        int _n, int _h, int _w, int _c)
    {
        __shared__ float2 sm_g_bias[1024];
        __shared__ float2 sm_g_scale[1024];
        for (int i = threadIdx.x; i < _c; i += blockDim.x) {
            sm_g_bias[i].x = 0.0;
            sm_g_bias[i].y = 0.0;
            sm_g_scale[i].x = 0.0;
            sm_g_scale[i].y = 0.0;
        }
        __syncthreads();

        int tid = threadIdx.x + blockDim.x * blockIdx.x;
        int num_threads = blockDim.x * gridDim.x;
        int num_elements = _n * _h * _w * _c;
        int pre_c = -1;
        float2  my_g_bias;
        float2  my_g_scale;
        half2  _h2_tmp;
        for (int i = tid; i < num_elements; i += num_threads) {
            int cur_c = i % _c;
            if (pre_c != cur_c) {
                if (pre_c >= 0) {
                    atomicAdd(&sm_g_bias[pre_c].x,  my_g_bias.x);
                    atomicAdd(&sm_g_bias[pre_c].y,  my_g_bias.y);
                    atomicAdd(&sm_g_scale[pre_c].x, my_g_scale.x);
                    atomicAdd(&sm_g_scale[pre_c].y, my_g_scale.y);
                }
                my_g_bias.x  = 0.0;
                my_g_bias.y  = 0.0;
                my_g_scale.x = 0.0;
                my_g_scale.y = 0.0;
            }
            pre_c = cur_c;

            float2 _x = {(float)x[i].x, (float)x[i].y};
            float2 _z = {(float)z[i].x, (float)z[i].y};
            float2 _gz = {(float)gz[i].x, (float)gz[i].y};
            if (_z.x <= 0.0) _gz.x = 0.0;
            if (_z.y <= 0.0) _gz.y = 0.0;
            my_g_bias.x  += _gz.x;
            my_g_bias.y  += _gz.y;
            my_g_scale.x += _gz.x * _x.x;
            my_g_scale.y += _gz.y * _x.y;
            _h2_tmp.x = (half)(_gz.x * (float)scale[cur_c].x);
            _h2_tmp.y = (half)(_gz.y * (float)scale[cur_c].y);
            gx[i] = _h2_tmp;
            _h2_tmp.x = (half)_gz.x;
            _h2_tmp.y = (half)_gz.y;
            gy[i] = _h2_tmp;
        }
        if (pre_c >= 0) {
            atomicAdd(&sm_g_bias[pre_c].x,  my_g_bias.x);
            atomicAdd(&sm_g_bias[pre_c].y,  my_g_bias.y);
            atomicAdd(&sm_g_scale[pre_c].x, my_g_scale.x);
            atomicAdd(&sm_g_scale[pre_c].y, my_g_scale.y);
        }

        __syncthreads();
        for (int i = threadIdx.x; i < _c; i += blockDim.x) {
            _h2_tmp.x = (half)sm_g_bias[i].x;
            _h2_tmp.y = (half)sm_g_bias[i].y;
            atomicAdd(&g_bias[i], _h2_tmp);
            _h2_tmp.x = (half)sm_g_scale[i].x;
            _h2_tmp.y = (half)sm_g_scale[i].y;
            atomicAdd(&g_scale[i], _h2_tmp);
        }
    }
    ''', 'cupy_FSBAR_backward')


def fused_scale_bias_add_relu(x, scale, bias, y):
    fnode = FusedScaleBiasAddReluFunction()
    z, = fnode.apply((x, scale, bias, y))
    return z
