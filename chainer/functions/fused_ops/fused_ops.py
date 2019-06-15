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
_plans = {}


def _get_num_SM(device_id=None):
    if device_id is None:
        device_id = cupy.cuda.runtime.getDevice()
    if _num_SM[device_id] is None:
        _num_SM[device_id] = cupy.cuda.runtime.deviceGetAttribute(
            cupy.cuda.runtime.cudaDevAttrMultiProcessorCount, device_id)
    return _num_SM[device_id]


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

        y = cuda.cupy.empty((n, out_h, out_w, out_c), dtype=x.dtype)
        ysum = cuda.cupy.empty((1, 1, 1, out_c), dtype=numpy.float32)
        ysqsum = cuda.cupy.empty_like(ysum)
        self.saved_mean = cuda.cupy.empty_like(ysum)
        self.saved_invstd = cuda.cupy.empty_like(ysum)
        out_scale = cuda.cupy.empty((1, 1, 1, out_c), dtype=numpy.float16)
        out_bias = cuda.cupy.empty_like(out_scale)

        # bn_mode = libcudnn.CUDNN_BATCHNORM_SPATIAL_PERSISTENT
        bn_mode = libcudnn.CUDNN_BATCHNORM_SPATIAL
        ptr_ph = libcudnn.CUDNN_PTR_16B_ALIGNED

        #
        ops = libcudnn.CUDNN_FUSED_SCALE_BIAS_ACTIVATION_CONV_BNSTATS
        key = (ops, bn_mode, x.shape, x.dtype, W.shape, W.dtype,
               y.shape, y.dtype, scale is None)
        if key in _plans:
            plan, workspace_size = _plans[key]
        else:
            x_desc = cudnn.create_tensor_descriptor(
                x, libcudnn.CUDNN_TENSOR_NHWC)
            w_desc = cudnn.create_filter_descriptor(
                W, libcudnn.CUDNN_TENSOR_NHWC)
            y_desc = cudnn.create_tensor_descriptor(
                y, libcudnn.CUDNN_TENSOR_NHWC)
            ysum_desc = cudnn.create_tensor_descriptor(
                ysum, libcudnn.CUDNN_TENSOR_NHWC)
            conv_desc = cudnn.create_convolution_descriptor(
                (self.ph, self.pw), (self.sy, self.sx), W.dtype,
                use_tensor_core=True)
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
            if scale is not None:
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
            plan = cudnn.create_fused_ops_plan(ops)
            const_pack = cudnn.create_fused_ops_const_param_pack(
                ops, const_param)
            workspace_size = cudnn.make_fused_ops_plan(plan, const_pack)
            max_workspace_size = cudnn.get_max_workspace_size()
            if workspace_size > max_workspace_size:
                msg = ('required workspace size ({}) is larger than max workspace'
                       ' size ({})'.format(workspace_size, max_workspace_size))
                raise RuntimeError(msg)
            _plans[key] = (plan, workspace_size)
        workspace = memory.alloc(workspace_size)
        var_param = [
            (libcudnn.CUDNN_PTR_XDATA, x),
            (libcudnn.CUDNN_PTR_WDATA, W),
            (libcudnn.CUDNN_PTR_YDATA, y),
            (libcudnn.CUDNN_PTR_YSUM, ysum),
            (libcudnn.CUDNN_PTR_YSQSUM, ysqsum),
            (libcudnn.CUDNN_PTR_WORKSPACE, workspace),
            (libcudnn.CUDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES,
             workspace_size),
        ]
        if scale is not None:
            scale = cuda.cupy.ascontiguousarray(scale)
            bias = cuda.cupy.ascontiguousarray(bias)
            var_param.extend([
                (libcudnn.CUDNN_PTR_BN_EQSCALE, scale),
                (libcudnn.CUDNN_PTR_BN_EQBIAS, bias),
            ])
        var_pack = cudnn.create_fused_ops_variant_param_pack(
            ops, var_param)
        cudnn.fused_ops_execute(plan, var_pack)
        del workspace

        #
        ops = libcudnn.CUDNN_FUSED_BN_FINALIZE_STATISTICS_TRAINING
        key = (ops, bn_mode, ysum.shape, ysum.dtype, gamma.shape, gamma.dtype)
        if key in _plans:
            plan, workspace_size = _plans[key]
        else:
            gamma_desc = cudnn.create_tensor_descriptor(
                gamma.reshape(1, 1, 1, out_c), libcudnn.CUDNN_TENSOR_NHWC)
            out_scale_desc = cudnn.create_tensor_descriptor(
                out_scale, libcudnn.CUDNN_TENSOR_NHWC)
            ysum_desc = cudnn.create_tensor_descriptor(
                ysum, libcudnn.CUDNN_TENSOR_NHWC)
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
            plan = cudnn.create_fused_ops_plan(ops)
            const_pack = cudnn.create_fused_ops_const_param_pack(
                ops, const_param)
            workspace_size = cudnn.make_fused_ops_plan(plan, const_pack)
            max_workspace_size = cudnn.get_max_workspace_size()
            if workspace_size > max_workspace_size:
                msg = ('required workspace size ({}) is larger than max workspace'
                       ' size ({})'.format(workspace_size, max_workspace_size))
                raise RuntimeError(msg)
            _plans[key] = (plan, workspace_size)
        workspace = memory.alloc(workspace_size)
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
            (libcudnn.CUDNN_PTR_WORKSPACE, workspace),
            (libcudnn.CUDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES,
             workspace_size)
        ]
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
        if scale is not None:
            scale = cuda.cupy.ascontiguousarray(scale.data)
            bias = cuda.cupy.ascontiguousarray(bias.data)

        gW = cuda.cupy.empty_like(W)

        # bn_mode = libcudnn.CUDNN_BATCHNORM_SPATIAL_PERSISTENT
        bn_mode = libcudnn.CUDNN_BATCHNORM_SPATIAL
        ptr_ph = libcudnn.CUDNN_PTR_16B_ALIGNED

        # **** gy, ggamma, gbeta ****
        gy, ggamma, gbeta = self._compute_grad_y_gamma_beta(
            y, gy, gamma, g_out_scale, g_out_bias)

        # **** gW ****
        ops = libcudnn.CUDNN_FUSED_SCALE_BIAS_ACTIVATION_WGRAD
        key = (ops, bn_mode, x.shape, x.dtype, W.shape, W.dtype,
               y.shape, y.dtype, scale is None)
        if key in _plans:
            plan, workspace_size = _plans[key]
        else:
            x_desc = cudnn.create_tensor_descriptor(
                x, libcudnn.CUDNN_TENSOR_NHWC)
            w_desc = cudnn.create_filter_descriptor(
                W, libcudnn.CUDNN_TENSOR_NHWC)
            y_desc = cudnn.create_tensor_descriptor(
                gy, libcudnn.CUDNN_TENSOR_NHWC)
            conv_desc = cudnn.create_convolution_descriptor(
                (self.ph, self.pw), (self.sy, self.sx), W.dtype,
                use_tensor_core=True)
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
            if scale is not None:
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
            plan = cudnn.create_fused_ops_plan(ops)
            const_pack = cudnn.create_fused_ops_const_param_pack(
                ops, const_param)
            workspace_size = cudnn.make_fused_ops_plan(plan, const_pack)
            max_workspace_size = cudnn.get_max_workspace_size()
            if workspace_size > max_workspace_size:
                msg = ('required workspace size ({}) is larger than max workspace'
                       ' size ({})'.format(workspace_size, max_workspace_size))
                raise RuntimeError(msg)
            _plans[key] = (plan, workspace_size)
        workspace = memory.alloc(workspace_size)
        var_param = [
            (libcudnn.CUDNN_PTR_XDATA, x),
            (libcudnn.CUDNN_PTR_DWDATA, gW),
            (libcudnn.CUDNN_PTR_DYDATA, gy),
            (libcudnn.CUDNN_PTR_WORKSPACE, workspace),
            (libcudnn.CUDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES,
             workspace_size),
        ]
        if scale is not None:
            var_param.extend([
                (libcudnn.CUDNN_PTR_BN_EQSCALE, scale),
                (libcudnn.CUDNN_PTR_BN_EQBIAS, bias),
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
        n_blocks = _get_num_SM(y.data.device_id) * 4
        self._cupy_adjust_grad_y()(
            (n_blocks,), (block_size,),
            (y, gy, gamma, g_gamma, g_beta, self.saved_mean, self.saved_invstd,
             _gy,
             _n, _h, _w, _c // 2))
        return _gy, g_gamma, g_beta

    def _compute_grad_x_scale_bias(self, x, gx, scale, bias):
        _n, _h, _w, _c = x.shape
        if False:
            # reference
            _x = scale * x + bias
            _gx = gx
            _gx[_x <= 0] = 0
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
        n_blocks = _get_num_SM(x.data.device_id)
        assert _c <= 2048
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
        const half2 *y, const half2 *gy,
        const float2 *gamma, const float2 *g_gamma, const float2 *g_beta,
        const float2 *mean, const float2 *invstd,
        half2 *o_gy,
        int _n, int _h, int _w, int _c)
    {
        int tid = threadIdx.x + blockDim.x * blockIdx.x;
        int num_threads = blockDim.x * gridDim.x;
        int num_elements = _n * _h * _w * _c;
        float2 mean_c;
        float2 invstd_c;
        float2 g_beta_c;
        float2 g_gamma_c;
        int pre_c = -1;
        for (int i = tid; i < num_elements; i += num_threads) {
            int c = i % _c;
            if (pre_c != c) {
                mean_c = mean[c];
                invstd_c = invstd[c];
                g_beta_c = g_beta[c];
                g_gamma_c = g_gamma[c];
            }
            pre_c = c;
            half2 y_i = y[i];
            half2 gy_i = gy[i];
            float2 _y;
            _y.x = ((float)y_i.x - mean_c.x) * invstd_c.x;
            _y.y = ((float)y_i.y - mean_c.y) * invstd_c.y;
            float2 _gy;
            _gy.x = (float)gy_i.x;
            _gy.y = (float)gy_i.y;
            _gy.x -= (g_beta_c.x + g_gamma_c.x * _y.x) / (_n * _h * _w);
            _gy.y -= (g_beta_c.y + g_gamma_c.y * _y.y) / (_n * _h * _w);
            half2 o_gy_i;
            o_gy_i.x = (half)_gy.x;
            o_gy_i.y = (half)_gy.y;
            o_gy[i] = o_gy_i;
        }
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
        half2  my_scale;
        half2  my_bias;
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
                my_scale = scale[cur_c];
                my_bias  = bias[cur_c];
            }
            pre_c = cur_c;
            half2 x_i = x[i];
            half2 gx_i = gx[i];
            float2 _x;
            _x.x = (float)my_scale.x * (float)x_i.x + (float)my_bias.x;
            _x.y = (float)my_scale.y * (float)x_i.y + (float)my_bias.y;
            float2 _gx = {(float)gx_i.x, (float)gx_i.y};
            if (_x.x <= 0.0) _gx.x = 0.0;
            if (_x.y <= 0.0) _gx.y = 0.0;
            my_g_bias.x  += _gx.x;
            my_g_bias.y  += _gx.y;
            my_g_scale.x += _gx.x * _x.x;
            my_g_scale.y += _gx.y * _x.y;
            half2 o_gx_i;
            o_gx_i.x = (half)(_gx.x * (float)my_scale.x);
            o_gx_i.y = (half)(_gx.y * (float)my_scale.y);
            o_gx[i] = o_gx_i;
        }
        if (pre_c >= 0) {
            atomicAdd(&sm_g_bias[pre_c].x,  my_g_bias.x);
            atomicAdd(&sm_g_bias[pre_c].y,  my_g_bias.y);
            atomicAdd(&sm_g_scale[pre_c].x, my_g_scale.x);
            atomicAdd(&sm_g_scale[pre_c].y, my_g_scale.y);
        }

        __syncthreads();
        for (int i = threadIdx.x; i < _c; i += blockDim.x) {
            half2 g_scale_i;
            g_scale_i.x = (half)sm_g_scale[i].x;
            g_scale_i.y = (half)sm_g_scale[i].y;
            atomicAdd(&g_scale[i], g_scale_i);
            half2 g_bias_i;
            g_bias_i.x = (half)sm_g_bias[i].x;
            g_bias_i.y = (half)sm_g_bias[i].y;
            atomicAdd(&g_bias[i], g_bias_i);
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
        self.len_inputs = len(inputs)
        if self.len_inputs == 4:
            x1, scale1, bias1, x2 = inputs
            scale2, bias2 = None, None
            self.retain_inputs((0, 1, 2, 3))
        else:
            x1, scale1, bias1, x2, scale2, bias2 = inputs
            self.retain_inputs((0, 1, 2, 3, 4, 5))

        self.retain_outputs((0,))

        if False:
            if scale2 is None:
                y = scale1 * x1 + bias1 + x2
            else:
                y = scale1 * x1 + bias1 + scale2 * x2 + bias2
            y[y <= 0] = 0
            return y,

        _n, _h, _w, _c = x1.shape
        x1 = cuda.cupy.ascontiguousarray(x1)
        x2 = cuda.cupy.ascontiguousarray(x2)
        bias1 = cuda.cupy.ascontiguousarray(bias1)
        scale1 = cuda.cupy.ascontiguousarray(scale1)
        if scale2 is not None:
            bias2 = cuda.cupy.ascontiguousarray(bias2)
            scale2 = cuda.cupy.ascontiguousarray(scale2)
        else:
            bias2 = 0
            scale2 = 0
        y = cupy.empty_like(x1)

        block_size = 256
        n_blocks = _get_num_SM(y.data.device_id) * 4
        self._cupy_forward()(
            (n_blocks,), (block_size,),
            (x1, scale1, bias1, x2, scale2, bias2,
             y,
             _n, _h, _w, _c // 2))
        return y,

    def _cupy_forward(self):
        return cupy.RawKernel(r'''
#include <cuda_fp16.h>
    extern "C" __global__
    void cupy_FSBAR_forward(
        const half2 *x1, const half2 *scale1, const half2 *bias1,
        const half2 *x2, const half2 *scale2, const half2 *bias2,
        half2 *y,
        int _n, int _h, int _w, int _c)
    {
        int tid = threadIdx.x + blockDim.x * blockIdx.x;
        int num_threads = blockDim.x * gridDim.x;
        int num_elements = _n * _h * _w * _c;
        half2 bias1_c, scale1_c;
        half2 bias2_c, scale2_c;
        int pre_c = -1;
        for (int i = tid; i < num_elements; i += num_threads) {
            int c = i % _c;
            if (pre_c != c) {
                bias1_c = bias1[c];
                scale1_c = scale1[c];
                if (scale2 != NULL) {
                    bias2_c = bias2[c];
                    scale2_c = scale2[c];
                }
            }
            pre_c = c;
            half2 x1_i = x1[i];
            half2 x2_i = x2[i];
            float2 _y;
            _y.x = (float)x1_i.x * (float)scale1_c.x + (float)bias1_c.x;
            _y.y = (float)x1_i.y * (float)scale1_c.y + (float)bias1_c.y;
            if (scale2 != NULL) {
                _y.x += (float)x2_i.x * (float)scale2_c.x + (float)bias2_c.x;
                _y.y += (float)x2_i.y * (float)scale2_c.y + (float)bias2_c.y;
            }
            else {
                _y.x += (float)x2_i.x;
                _y.y += (float)x2_i.y;
            }
            if (_y.x <= 0) _y.x = 0.0;
            if (_y.y <= 0) _y.y = 0.0;
            half2 y_i;
            y_i.x = (half)_y.x;
            y_i.y = (half)_y.y;
            y[i] = y_i;
        }
    }
    ''', 'cupy_FSBAR_forward')

    def backward(self, indexes, grad_outputs):
        if self.len_inputs == 4:
            x1, scale1, bias1, x2 = self.get_retained_inputs()
            scale2, bias2 = None, None
        else:
            x1, scale1, bias1, x2, scale2, bias2 = self.get_retained_inputs()
        gy, = grad_outputs
        y, = self.get_retained_outputs()
        _n, _h, _w, _c = x1.shape

        x1 = cuda.cupy.ascontiguousarray(x1.data)
        x2 = cuda.cupy.ascontiguousarray(x2.data)
        y = cuda.cupy.ascontiguousarray(y.data)
        gy = cuda.cupy.ascontiguousarray(gy.data)
        scale1 = cuda.cupy.ascontiguousarray(scale1.data)
        if self.len_inputs == 6:
            scale2 = cuda.cupy.ascontiguousarray(scale2.data)

        if False:
            # reference
            _gy = gy
            _gy[y <= 0] = 0
            g_bias1 = cupy.sum(_gy, axis=(0, 1, 2)).reshape((_c,))
            g_scale1 = cupy.sum(_gy * x1, axis=(0, 1, 2)).reshape((_c,))
            gx1 = scale1 * _gy
            if self.len_inputs == 6:
                g_bias2 = g_bias1
                g_scale2 = cupy.sum(_gy * x2, axis=(0, 1, 2)).reshape((_c,))
                gx2 = scale2 * _gy
            else:
                gx2 = _gy

            gx1 = chainer.Variable(gx1)
            gx2 = chainer.Variable(gx2)
            g_scale1 = chainer.Variable(g_scale1)
            g_bias1 = chainer.Variable(g_bias1)
            if self.len_inputs == 6:
                g_scale2 = chainer.Variable(g_scale2)
                g_bias2 = chainer.Variable(g_bias2)
                return gx1, g_scale1, g_bias1, gx2, g_scale2, g_bias2
            else:
                return gx1, g_scale1, g_bias1, gx2

        gx1 = cupy.empty_like(x1)
        gx2 = cupy.empty_like(x2)
        g_scale1 = cupy.zeros_like(scale1)
        g_bias1 = cupy.zeros_like(scale1)
        if self.len_inputs == 6:
            g_scale2 = cupy.zeros_like(scale2)
        else:
            scale2 = 0
            g_scale2 = 0

        block_size = 512
        n_blocks = _get_num_SM(y.data.device_id) * 2
        assert _c <= 2048
        self._cupy_backward()(
            (n_blocks,), (block_size,),
            (x1, x2, scale1, scale2,
             y, gy,
             gx1, gx2, g_bias1, g_scale1, g_scale2,
             _n, _h, _w, _c // 2))

        gx1 = chainer.Variable(gx1)
        gx2 = chainer.Variable(gx2)
        g_scale1 = chainer.Variable(g_scale1)
        g_bias1 = chainer.Variable(g_bias1)
        if self.len_inputs == 6:
            g_scale2 = chainer.Variable(g_scale2)
            g_bias2 = g_bias1
            return gx1, g_scale1, g_bias1, gx2, g_scale2, g_bias2
        else:
            return gx1, g_scale1, g_bias1, gx2

    def _cupy_backward(self):
        return cupy.RawKernel(r'''
#include <cuda_fp16.h>
    extern "C" __global__
    void cupy_FSBAR_backward(
        const half2 *x1, const half2 *x2,
        const half2 *scale1, const half2 *scale2,
        const half2 *y, const half2 *gy,
        half2 *gx1, half2 *gx2,
        half2 *g_bias1, half2 *g_scale1, half2 *g_scale2,
        int _n, int _h, int _w, int _c)
    {
        __shared__ float2 sm_g_bias1[1024];
        __shared__ float2 sm_g_scale1[1024];
        __shared__ float2 sm_g_scale2[1024];
        for (int i = threadIdx.x; i < _c; i += blockDim.x) {
            sm_g_bias1[i].x = 0.0;
            sm_g_bias1[i].y = 0.0;
            sm_g_scale1[i].x = 0.0;
            sm_g_scale1[i].y = 0.0;
            if (scale2 != NULL) {
                sm_g_scale2[i].x = 0.0;
                sm_g_scale2[i].y = 0.0;
            }
        }
        __syncthreads();

        int tid = threadIdx.x + blockDim.x * blockIdx.x;
        int num_threads = blockDim.x * gridDim.x;
        int num_elements = _n * _h * _w * _c;
        int pre_c = -1;
        float2  my_g_bias1;
        float2  my_g_scale1;
        float2  my_g_scale2;
        float2 scale1_c;
        float2 scale2_c;
        for (int i = tid; i < num_elements; i += num_threads) {
            float2 x1_i = {(float)x1[i].x, (float)x1[i].y};
            float2 x2_i = {(float)x2[i].x, (float)x2[i].y};
            float2 y_i = {(float)y[i].x, (float)y[i].y};
            float2 gy_i = {(float)gy[i].x, (float)gy[i].y};
            if (y_i.x <= 0.0) gy_i.x = 0.0;
            if (y_i.y <= 0.0) gy_i.y = 0.0;
            int cur_c = i % _c;
            if (pre_c != cur_c) {
                if (pre_c >= 0) {
                    atomicAdd(&sm_g_bias1[pre_c].x,  my_g_bias1.x);
                    atomicAdd(&sm_g_bias1[pre_c].y,  my_g_bias1.y);
                    atomicAdd(&sm_g_scale1[pre_c].x, my_g_scale1.x);
                    atomicAdd(&sm_g_scale1[pre_c].y, my_g_scale1.y);
                    if (scale2 != NULL) {
                        atomicAdd(&sm_g_scale2[pre_c].x, my_g_scale2.x);
                        atomicAdd(&sm_g_scale2[pre_c].y, my_g_scale2.y);
                    }
                }
                pre_c = cur_c;
                scale1_c.x = (float)scale1[cur_c].x;
                scale1_c.y = (float)scale1[cur_c].y;
                if (scale2 != NULL) {
                    scale2_c.x = (float)scale2[cur_c].x;
                    scale2_c.y = (float)scale2[cur_c].y;
                }
                my_g_bias1.x  = gy_i.x;
                my_g_bias1.y  = gy_i.y;
                my_g_scale1.x = gy_i.x * x1_i.x;
                my_g_scale1.y = gy_i.y * x1_i.y;
                if (scale2 != NULL) {
                    my_g_scale2.x = gy_i.x * x2_i.x;
                    my_g_scale2.y = gy_i.y * x2_i.y;
                }
            }
            else {
                my_g_bias1.x  += gy_i.x;
                my_g_bias1.y  += gy_i.y;
                my_g_scale1.x += gy_i.x * x1_i.x;
                my_g_scale1.y += gy_i.y * x1_i.y;
                if (scale2 != NULL) {
                    my_g_scale2.x = gy_i.x * x2_i.x;
                    my_g_scale2.y = gy_i.y * x2_i.y;
                }
            }
            half2 gx1_i;
            gx1_i.x = (half)(gy_i.x * scale1_c.x);
            gx1_i.y = (half)(gy_i.y * scale1_c.y);
            gx1[i] = gx1_i;
            half2 gx2_i;
            if (scale2 != NULL) {
                gx2_i.x = (half)(gy_i.x * scale2_c.x);
                gx2_i.y = (half)(gy_i.y * scale2_c.y);
            }
            else {
                gx2_i.x = (half)gy_i.x;
                gx2_i.y = (half)gy_i.y;
            }
            gx2[i] = gx2_i;
        }
        if (pre_c >= 0) {
            atomicAdd(&sm_g_bias1[pre_c].x,  my_g_bias1.x);
            atomicAdd(&sm_g_bias1[pre_c].y,  my_g_bias1.y);
            atomicAdd(&sm_g_scale1[pre_c].x, my_g_scale1.x);
            atomicAdd(&sm_g_scale1[pre_c].y, my_g_scale1.y);
            if (scale2 != NULL) {
                atomicAdd(&sm_g_scale2[pre_c].x, my_g_scale2.x);
                atomicAdd(&sm_g_scale2[pre_c].y, my_g_scale2.y);
            }
        }

        __syncthreads();
        for (int i = threadIdx.x; i < _c; i += blockDim.x) {
            half2 g_bias1_i;
            g_bias1_i.x = (half)sm_g_bias1[i].x;
            g_bias1_i.y = (half)sm_g_bias1[i].y;
            atomicAdd(&g_bias1[i], g_bias1_i);
            half2 g_scale1_i;
            g_scale1_i.x = (half)sm_g_scale1[i].x;
            g_scale1_i.y = (half)sm_g_scale1[i].y;
            atomicAdd(&g_scale1[i], g_scale1_i);
            if (scale2 != NULL) {
                half2 g_scale2_i;
                g_scale2_i.x = (half)sm_g_scale2[i].x;
                g_scale2_i.y = (half)sm_g_scale2[i].y;
                atomicAdd(&g_scale2[i], g_scale2_i);
            }
        }
    }
    ''', 'cupy_FSBAR_backward')


def fused_scale_bias_add_relu(x1, scale1, bias1, x2, scale2=None, bias2=None):
    fnode = FusedScaleBiasAddReluFunction()
    if scale2 is None:
        y, = fnode.apply((x1, scale1, bias1, x2))
    else:
        y, = fnode.apply((x1, scale1, bias1, x2, scale2, bias2))
    return y
