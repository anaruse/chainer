import numpy

import cupy

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer import configuration
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


class ResNetBottleNeckFunction(function_node.FunctionNode):

    _supports_nhwc_tensor_layout = True
    cover_all = False
    tensor_layout = 'NHWC'
    dilate = 1
    # bn_mode = libcudnn.CUDNN_BATCHNORM_SPATIAL
    bn_mode = libcudnn.CUDNN_BATCHNORM_SPATIAL_PERSISTENT
    ptr_ph = libcudnn.CUDNN_PTR_16B_ALIGNED

    # reference = True
    reference = False
    
    def __init__(self, running_mean_set, running_var_set,
                 stride=1, bn_eps=2.5e-5, bn_decay=0.9):
        self.bn_eps = bn_eps
        self.bn_decay = bn_decay
        self.running_mean_set = running_mean_set
        self.running_var_set = running_var_set
        self.stride = (1, stride, 1)
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
        if self.len_inputs == 10:
            x4 = None
        else:
            x4 = inputs[10]
        if configuration.config.train:
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
        if x4 is not None:
            x4 = cuda.cupy.ascontiguousarray(x4)

        x1, scale1, bias1, mean1, invstd1 = self.fw_scale_bias_relu_conv_bnorm(
            0, x0, W0, gamma0, beta0)
        x2, scale2, bias2, mean2, invstd2 = self.fw_scale_bias_relu_conv_bnorm(
            1, x1, W1, gamma1, beta1, scale1, bias1)
        x3, scale3, bias3, mean3, invstd3 = self.fw_scale_bias_relu_conv_bnorm(
            2, x2, W2, gamma2, beta2, scale2, bias2)
        if x4 is not None:
            # print('# x.shape: {}'.format(x4.shape))
            y = self.fw_scale_bias_add_relu(x3, scale3, bias3, x4)
        else:
            y = self.fw_scale_bias_add_relu(x3, scale3, bias3, x0)

        if configuration.config.train:
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
            if False:  # debug
                _maxabs1 = cupy.max(cupy.abs(invstd1))
                _maxabs2 = cupy.max(cupy.abs(invstd2))
                _maxabs3 = cupy.max(cupy.abs(invstd3))
                if max(_maxabs1, _maxabs2, _maxabs3) > 100:
                    print('# saved_invstd: {}'.format(
                        (_maxabs1, _maxabs2, _maxabs3)))

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

        bn_mode = self.bn_mode
        ptr_ph = self.ptr_ph

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
                ysum.reshape(1, 1, 1, y_c), libcudnn.CUDNN_TENSOR_NHWC)
            conv_desc = cudnn.create_convolution_descriptor(
                (self.pad[lid], self.pad[lid]),
                (self.stride[lid], self.stride[lid]),
                W.dtype, use_tensor_core=True)
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
                    scale.reshape(1, 1, 1, x_c), libcudnn.CUDNN_TENSOR_NHWC)
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
        if configuration.config.train:
            ops = libcudnn.CUDNN_FUSED_BN_FINALIZE_STATISTICS_TRAINING
            key = (ops, bn_mode, ysum.shape, ysum.dtype,
                   gamma.shape, gamma.dtype)
            if key in _plans:
                plan, workspace_size = _plans[key]
            else:
                gamma_desc = cudnn.create_tensor_descriptor(
                    gamma.reshape(1, 1, 1, y_c), libcudnn.CUDNN_TENSOR_NHWC)
                o_scale_desc = cudnn.create_tensor_descriptor(
                    o_scale.reshape(1, 1, 1, y_c), libcudnn.CUDNN_TENSOR_NHWC)
                ysum_desc = cudnn.create_tensor_descriptor(
                    ysum.reshape(1, 1, 1, y_c), libcudnn.CUDNN_TENSOR_NHWC)
                const_param = (
                    (libcudnn.CUDNN_PARAM_BN_MODE, bn_mode),
                    (libcudnn.CUDNN_PARAM_YSTATS_DESC, ysum_desc),
                    (libcudnn.CUDNN_PARAM_YSUM_PLACEHOLDER, ptr_ph),
                    (libcudnn.CUDNN_PARAM_YSQSUM_PLACEHOLDER, ptr_ph),
                    (libcudnn.CUDNN_PARAM_BN_SCALEBIAS_MEANVAR_DESC,
                     gamma_desc),
                    (libcudnn.CUDNN_PARAM_BN_SCALE_PLACEHOLDER, ptr_ph),
                    (libcudnn.CUDNN_PARAM_BN_BIAS_PLACEHOLDER, ptr_ph),
                    (libcudnn.CUDNN_PARAM_BN_SAVED_MEAN_PLACEHOLDER, ptr_ph),
                    (libcudnn.CUDNN_PARAM_BN_SAVED_INVSTD_PLACEHOLDER, ptr_ph),
                    (libcudnn.CUDNN_PARAM_BN_RUNNING_MEAN_PLACEHOLDER, ptr_ph),
                    (libcudnn.CUDNN_PARAM_BN_RUNNING_VAR_PLACEHOLDER, ptr_ph),
                    (libcudnn.CUDNN_PARAM_BN_EQSCALEBIAS_DESC, o_scale_desc),
                    (libcudnn.CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER, ptr_ph),
                    (libcudnn.CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER, ptr_ph),
                )
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
            acc_count = n * x_h * x_w
            factor = 1 - self.bn_decay
            var_param = (
                (libcudnn.CUDNN_PTR_YSUM, ysum),
                (libcudnn.CUDNN_PTR_YSQSUM, ysqsum),
                (libcudnn.CUDNN_PTR_BN_SCALE, gamma),
                (libcudnn.CUDNN_PTR_BN_BIAS, beta),
                (libcudnn.CUDNN_PTR_BN_SAVED_MEAN, mean),
                (libcudnn.CUDNN_PTR_BN_SAVED_INVSTD, invstd),
                (libcudnn.CUDNN_PTR_BN_RUNNING_MEAN,
                 self.running_mean_set[lid]),
                (libcudnn.CUDNN_PTR_BN_RUNNING_VAR, self.running_var_set[lid]),
                (libcudnn.CUDNN_PTR_BN_EQSCALE, o_scale),
                (libcudnn.CUDNN_PTR_BN_EQBIAS, o_bias),
                (libcudnn.CUDNN_SCALAR_INT64_T_BN_ACCUMULATION_COUNT,
                 acc_count),
                (libcudnn.CUDNN_SCALAR_DOUBLE_BN_EXP_AVG_FACTOR, factor),
                (libcudnn.CUDNN_SCALAR_DOUBLE_BN_EPSILON, self.bn_eps),
                (libcudnn.CUDNN_PTR_WORKSPACE, workspace),
                (libcudnn.CUDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES,
                 workspace_size))
        else:
            ops = libcudnn.CUDNN_FUSED_BN_FINALIZE_STATISTICS_INFERENCE
            key = (ops, bn_mode, gamma.shape, gamma.dtype)
            if key in _plans:
                plan, workspace_size = _plans[key]
            else:
                gamma_desc = cudnn.create_tensor_descriptor(
                    gamma.reshape(1, 1, 1, y_c), libcudnn.CUDNN_TENSOR_NHWC)
                o_scale_desc = cudnn.create_tensor_descriptor(
                    o_scale.reshape(1, 1, 1, y_c), libcudnn.CUDNN_TENSOR_NHWC)
                const_param = (
                    (libcudnn.CUDNN_PARAM_BN_MODE, bn_mode),
                    (libcudnn.CUDNN_PARAM_BN_SCALEBIAS_MEANVAR_DESC,
                     gamma_desc),
                    (libcudnn.CUDNN_PARAM_BN_SCALE_PLACEHOLDER, ptr_ph),
                    (libcudnn.CUDNN_PARAM_BN_BIAS_PLACEHOLDER, ptr_ph),
                    (libcudnn.CUDNN_PARAM_BN_RUNNING_MEAN_PLACEHOLDER, ptr_ph),
                    (libcudnn.CUDNN_PARAM_BN_RUNNING_VAR_PLACEHOLDER, ptr_ph),
                    (libcudnn.CUDNN_PARAM_BN_EQSCALEBIAS_DESC, o_scale_desc),
                    (libcudnn.CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER, ptr_ph),
                    (libcudnn.CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER, ptr_ph),
                )
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
            var_param = (
                (libcudnn.CUDNN_PTR_BN_SCALE, gamma),
                (libcudnn.CUDNN_PTR_BN_BIAS, beta),
                (libcudnn.CUDNN_PTR_BN_RUNNING_MEAN,
                 self.running_mean_set[lid]),
                (libcudnn.CUDNN_PTR_BN_RUNNING_VAR, self.running_var_set[lid]),
                (libcudnn.CUDNN_PTR_BN_EQSCALE, o_scale),
                (libcudnn.CUDNN_PTR_BN_EQBIAS, o_bias),
                (libcudnn.CUDNN_SCALAR_DOUBLE_BN_EPSILON, self.bn_eps),
                (libcudnn.CUDNN_PTR_WORKSPACE, workspace),
                (libcudnn.CUDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES,
                 workspace_size))
        var_pack = cudnn.create_fused_ops_variant_param_pack(
            ops, var_param)
        cudnn.fused_ops_execute(plan, var_pack)
        del workspace
        
        return y, o_scale, o_bias, mean, invstd

    def fw_scale_bias_add_relu(self, x, scale, bias, y):
        '''
        z = x * scale + bias + y
        '''
        if self.reference:
            z = x * scale + bias + y
            z[z <= 0] = 0
        else:
            _n, _h, _w, _c = x.shape
            z = cupy.empty_like(x)
            block_size = 256
            n_blocks = _get_num_SM(x.data.device_id) * 4
            # self._fw_scale_bias_add_relu()(
            #     (n_blocks,), (block_size,),
            #     (x, scale, bias, y,
            #      z,
            #      _n, _h, _w, _c // 2))
            self._fw_scale_bias_add_relu_vec4()(
                (n_blocks,), (block_size,),
                (x, scale, bias, y,
                 z,
                 _n, _h, _w, _c // 4))
        return z

    def _fw_scale_bias_add_relu(self):
        return cupy.RawKernel(r'''
#include <cuda_fp16.h>
    extern "C" __global__
    void cupy_resnet_fw_SBAR(
        const half2 *x, const half2 *scale, const half2 *bias, const half2 *y,
        half2 *z,
        int _n, int _h, int _w, int _c)
    {
        int tid = threadIdx.x + blockDim.x * blockIdx.x;
        int num_threads = blockDim.x * gridDim.x;
        int num_elements = _n * _h * _w * _c;
        half2 bias_c;
        half2 scale_c;
        int pre_c = -1;
        for (int i = tid; i < num_elements; i += num_threads) {
            int c = i % _c;
            if (pre_c != c) {
                bias_c = bias[c];
                scale_c = scale[c];
            }
            pre_c = c;
            half2 x_i = x[i];
            half2 y_i = y[i];
            float2 _z;
            _z.x = (float)x_i.x * (float)scale_c.x + (float)bias_c.x + (float)y_i.x;
            _z.y = (float)x_i.y * (float)scale_c.y + (float)bias_c.y + (float)y_i.y;
            if (_z.x <= 0) _z.x = 0.0;
            if (_z.y <= 0) _z.y = 0.0;
            half2 z_i;
            z_i.x = (half)_z.x;
            z_i.y = (half)_z.y;
            z[i] = z_i;
        }
    }
    ''', 'cupy_resnet_fw_SBAR')
    
    def _fw_scale_bias_add_relu_vec4(self):
        return cupy.RawKernel(r'''
#include <cuda_fp16.h>
#define _US_TO_HALF(var) *(reinterpret_cast<half *>(&(var)))
#define _HALF_TO_US(var) *(reinterpret_cast<unsigned short *>(&(var)))
#define _XSBY(X,S,B,Y) ((float)_US_TO_HALF(X) * (float)_US_TO_HALF(S) + (float)_US_TO_HALF(B) + + (float)_US_TO_HALF(Y))
#define _FLOAT_TO_US(DST, SRC) do {half tmp = (half)(SRC); DST = _HALF_TO_US(tmp);} while (0)
    extern "C" __global__
    void cupy_resnet_fw_SBAR_vec4(
        const ushort4 *x, const ushort4 *scale, const ushort4 *bias,
        const ushort4 *y,
        ushort4 *z,
        int _n, int _h, int _w, int _c)
    {
        int tid = threadIdx.x + blockDim.x * blockIdx.x;
        int num_threads = blockDim.x * gridDim.x;
        int num_elements = _n * _h * _w * _c;
        ushort4 bias_c;
        ushort4 scale_c;
        int pre_c = -1;
        for (int i = tid; i < num_elements; i += num_threads) {
            int c = i % _c;
            if (pre_c != c) {
                bias_c = bias[c];
                scale_c = scale[c];
            }
            pre_c = c;
            ushort4 x_i = x[i];
            ushort4 y_i = y[i];
            float4 _z;
            _z.x = _XSBY(x_i.x, scale_c.x, bias_c.x, y_i.x);
            _z.y = _XSBY(x_i.y, scale_c.y, bias_c.y, y_i.y);
            _z.z = _XSBY(x_i.z, scale_c.z, bias_c.z, y_i.z);
            _z.w = _XSBY(x_i.w, scale_c.w, bias_c.w, y_i.w);
            if (_z.x <= 0) _z.x = 0.0;
            if (_z.y <= 0) _z.y = 0.0;
            if (_z.z <= 0) _z.z = 0.0;
            if (_z.w <= 0) _z.w = 0.0;
            ushort4 z_i;
            _FLOAT_TO_US(z_i.x, _z.x);
            _FLOAT_TO_US(z_i.y, _z.y);
            _FLOAT_TO_US(z_i.z, _z.z);
            _FLOAT_TO_US(z_i.w, _z.w);
            z[i] = z_i;
        }
    }
    ''', 'cupy_resnet_fw_SBAR_vec4')

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

        gx3, g_gamma2, g_beta2, gx4 = self.bw_scale_bias_add_relu(
            self.x3, self.scale3, self.bias3, y, gy)

        ret = self.bw_scale_bias_relu_conv_bnorm(
            2,
            self.x2, self.scale2, self.bias2, W2,
            self.x3, self.scale3, self.bias3,
            gx3, g_gamma2, g_beta2)
        gx2, g_gamma1, g_beta1, gW2 = ret
        del gx3
        self.x3 = None

        ret = self.bw_scale_bias_relu_conv_bnorm(
            1,
            self.x1, self.scale1, self.bias1, W1,
            self.x2, self.scale2, self.bias2,
            gx2, g_gamma1, g_beta1)
        gx1, g_gamma0, g_beta0, gW1 = ret
        del gx2
        self.x2 = None

        if self.len_inputs == 10:
            gx0 = gx4
            gx4 = None
        else:
            gx0 = None

        ret = self.bw_scale_bias_relu_conv_bnorm(
            0,
            x0, None, None, W0,
            self.x1, self.scale1, self.bias1,
            gx1, g_gamma0, g_beta0,
            gx0)
        gx0, _, _, gW0 = ret
        del gx1
        self.x1 = None

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
        if gx4 is not None:
            gx4 = chainer.Variable(gx4)
            ret = ret + (gx4,)
        return ret
        
    def bw_scale_bias_add_relu(self, x, scale, bias, z, gz):
        '''
        z = x * scale + bias + y
        '''
        _n, _h, _w, _c = x.shape
        if self.reference:
            gz[z <= 0] = 0
            gy = gz
            g_bias = cupy.sum(gz, axis=(0, 1, 2)).reshape((_c,))
            g_scale = cupy.sum(gz * x, axis=(0, 1, 2)).reshape((_c,))
            g_beta = g_bias.astype(numpy.float32)
            g_gamma = g_scale * self.saved_invstd[2]
            gx = gz * scale
        else:
            g_gamma = cupy.zeros((_c,), dtype=numpy.float32)
            g_beta = cupy.zeros((_c,), dtype=numpy.float32)
            gy = cupy.empty_like(x)
            gx = cupy.empty_like(x)
            block_size = 1024
            n_blocks = _get_num_SM(x.data.device_id)
            assert _c <= 2048
            # self._bw_scale_bias_add_relu()(
            #     (n_blocks,), (block_size,),
            #     (x, scale, bias, z, gz, self.saved_invstd[2],
            #      gx, g_gamma, g_beta, gy,
            #      _n, _h, _w, _c // 2))
            self._bw_scale_bias_add_relu_vec4()(
                (n_blocks,), (block_size,),
                (x, scale, bias, z, gz, self.saved_invstd[2],
                 gx, g_gamma, g_beta, gy,
                 _n, _h, _w, _c // 4))

        return gx, g_gamma, g_beta, gy

    def _bw_scale_bias_add_relu(self):
        return cupy.RawKernel(r'''
#include <cuda_fp16.h>
    extern "C" __global__
    void cupy_resnet_bw_SBAR_gX_gGamma(
        const half2 *x, const half2 *scale, const half2 *bias,
        const half2 *z, const half2 *gz, const float2 *invstd,
        half2 *gx, float2 *g_gamma, float2 *g_beta, half2 *gy,
        int _n, int _h, int _w, int _c)
    {
        __shared__ float2 sm_g_beta[1024];
        __shared__ float2 sm_g_gamma[1024];
        for (int i = threadIdx.x; i < _c; i += blockDim.x) {
            sm_g_beta[i].x = 0.0;
            sm_g_beta[i].y = 0.0;
            sm_g_gamma[i].x = 0.0;
            sm_g_gamma[i].y = 0.0;
        }
        __syncthreads();

        int tid = threadIdx.x + blockDim.x * blockIdx.x;
        int num_threads = blockDim.x * gridDim.x;
        int num_elements = _n * _h * _w * _c;
        int pre_c = -1;
        float2  my_g_beta;
        float2  my_g_gamma;
        float2  scale_c;
        for (int i = tid; i < num_elements; i += num_threads) {
            float2 x_i = {(float)x[i].x, (float)x[i].y};
            float2 z_i = {(float)z[i].x, (float)z[i].y};
            float2 gz_i = {(float)gz[i].x, (float)gz[i].y};
            if (z_i.x <= 0.0) gz_i.x = 0.0;
            if (z_i.y <= 0.0) gz_i.y = 0.0;
            int cur_c = i % _c;
            if (pre_c != cur_c) {
                if (pre_c >= 0) {
                    atomicAdd(&sm_g_beta[pre_c].x,  my_g_beta.x);
                    atomicAdd(&sm_g_beta[pre_c].y,  my_g_beta.y);
                    atomicAdd(&sm_g_gamma[pre_c].x, my_g_gamma.x);
                    atomicAdd(&sm_g_gamma[pre_c].y, my_g_gamma.y);
                }
                pre_c = cur_c;
                scale_c.x = (float)scale[cur_c].x;
                scale_c.y = (float)scale[cur_c].y;
                my_g_beta.x  = gz_i.x;
                my_g_beta.y  = gz_i.y;
                my_g_gamma.x = gz_i.x * x_i.x;
                my_g_gamma.y = gz_i.y * x_i.y;
            }
            else {
                my_g_beta.x  += gz_i.x;
                my_g_beta.y  += gz_i.y;
                my_g_gamma.x += gz_i.x * x_i.x;
                my_g_gamma.y += gz_i.y * x_i.y;
            }
            half2 gx_i;
            gx_i.x = (half)(gz_i.x * scale_c.x);
            gx_i.y = (half)(gz_i.y * scale_c.y);
            gx[i] = gx_i;
            half2 gy_i;
            gy_i.x = (half)gz_i.x;
            gy_i.y = (half)gz_i.y;
            gy[i] = gy_i;
        }
        if (pre_c >= 0) {
            atomicAdd(&sm_g_beta[pre_c].x,  my_g_beta.x);
            atomicAdd(&sm_g_beta[pre_c].y,  my_g_beta.y);
            atomicAdd(&sm_g_gamma[pre_c].x, my_g_gamma.x);
            atomicAdd(&sm_g_gamma[pre_c].y, my_g_gamma.y);
        }

        __syncthreads();
        for (int i = threadIdx.x; i < _c; i += blockDim.x) {
            atomicAdd(&g_beta[i].x, sm_g_beta[i].x);
            atomicAdd(&g_beta[i].y, sm_g_beta[i].y);
            atomicAdd(&g_gamma[i].x, sm_g_gamma[i].x * invstd[i].x);
            atomicAdd(&g_gamma[i].y, sm_g_gamma[i].y * invstd[i].y);
        }
    }
    ''', 'cupy_resnet_bw_SBAR_gX_gGamma')
    
    def _bw_scale_bias_add_relu_vec4(self):
        return cupy.RawKernel(r'''
#include <cuda_fp16.h>
#define _US_TO_HALF(var) *(reinterpret_cast<half *>(&(var)))
#define _HALF_TO_US(var) *(reinterpret_cast<unsigned short *>(&(var)))
#define _FLOAT_TO_US(DST, SRC) do {half tmp = (half)(SRC); DST = _HALF_TO_US(tmp);} while (0)
    extern "C" __global__
    void cupy_resnet_bw_SBAR_gX_gGamma_vec4(
        const ushort4 *x, const ushort4 *scale, const ushort4 *bias,
        const ushort4 *z, const ushort4 *gz, const float4 *invstd,
        ushort4 *gx, float4 *g_gamma, float4 *g_beta, ushort4 *gy,
        int _n, int _h, int _w, int _c)
    {
        __shared__ float4 sm_g_beta[512];
        __shared__ float4 sm_g_gamma[512];
        float *ptr_sm_g_beta = &(sm_g_beta[0].x);
        float *ptr_sm_g_gamma = &(sm_g_gamma[0].x);
        for (int i = threadIdx.x; i < 4*_c; i += blockDim.x) {
            ptr_sm_g_beta[i] = 0.0;
            ptr_sm_g_gamma[i] = 0.0;
        }
        __syncthreads();

        int tid = threadIdx.x + blockDim.x * blockIdx.x;
        int num_threads = blockDim.x * gridDim.x;
        int num_elements = _n * _h * _w * _c;
        int pre_c = -1;
        float4  my_g_beta;
        float4  my_g_gamma;
        ushort4  scale_c;
        for (int i = tid; i < num_elements; i += num_threads) {
            ushort4 z_i = z[i];
            float4 _z_i;
            _z_i.x = (float)_US_TO_HALF(z_i.x);
            _z_i.y = (float)_US_TO_HALF(z_i.y);
            _z_i.z = (float)_US_TO_HALF(z_i.z);
            _z_i.w = (float)_US_TO_HALF(z_i.w);
            ushort4 gz_i = gz[i];
            float4 _gz_i;
            _gz_i.x = (float)_US_TO_HALF(gz_i.x);
            _gz_i.y = (float)_US_TO_HALF(gz_i.y);
            _gz_i.z = (float)_US_TO_HALF(gz_i.z);
            _gz_i.w = (float)_US_TO_HALF(gz_i.w);
            if (_z_i.x <= 0.0) _gz_i.x = 0.0;
            if (_z_i.y <= 0.0) _gz_i.y = 0.0;
            if (_z_i.z <= 0.0) _gz_i.z = 0.0;
            if (_z_i.w <= 0.0) _gz_i.w = 0.0;
            int cur_c = i % _c;
            if (pre_c != cur_c) {
                if (pre_c >= 0) {
                    atomicAdd(&sm_g_beta[pre_c].x,  my_g_beta.x);
                    atomicAdd(&sm_g_beta[pre_c].y,  my_g_beta.y);
                    atomicAdd(&sm_g_beta[pre_c].z,  my_g_beta.z);
                    atomicAdd(&sm_g_beta[pre_c].w,  my_g_beta.w);
                    atomicAdd(&sm_g_gamma[pre_c].x, my_g_gamma.x);
                    atomicAdd(&sm_g_gamma[pre_c].y, my_g_gamma.y);
                    atomicAdd(&sm_g_gamma[pre_c].z, my_g_gamma.z);
                    atomicAdd(&sm_g_gamma[pre_c].w, my_g_gamma.w);
                }
                pre_c = cur_c;
                scale_c = scale[cur_c];
                my_g_beta.x  = 0.0;
                my_g_beta.y  = 0.0;
                my_g_beta.z  = 0.0;
                my_g_beta.w  = 0.0;
                my_g_gamma.x = 0.0;
                my_g_gamma.y = 0.0;
                my_g_gamma.z = 0.0;
                my_g_gamma.w = 0.0;
            }
            ushort4 x_i = x[i];
            float4 _x_i;
            _x_i.x = (float)_US_TO_HALF(x_i.x);
            _x_i.y = (float)_US_TO_HALF(x_i.y);
            _x_i.z = (float)_US_TO_HALF(x_i.z);
            _x_i.w = (float)_US_TO_HALF(x_i.w);
            my_g_beta.x += _gz_i.x;
            my_g_beta.y += _gz_i.y;
            my_g_beta.z += _gz_i.z;
            my_g_beta.w += _gz_i.w;
            my_g_gamma.x += _gz_i.x * _x_i.x;
            my_g_gamma.y += _gz_i.y * _x_i.y;
            my_g_gamma.z += _gz_i.z * _x_i.z;
            my_g_gamma.w += _gz_i.w * _x_i.w;
            ushort4 gx_i;
            _FLOAT_TO_US(gx_i.x, _gz_i.x * (float)_US_TO_HALF(scale_c.x));
            _FLOAT_TO_US(gx_i.y, _gz_i.y * (float)_US_TO_HALF(scale_c.y));
            _FLOAT_TO_US(gx_i.z, _gz_i.z * (float)_US_TO_HALF(scale_c.z));
            _FLOAT_TO_US(gx_i.w, _gz_i.w * (float)_US_TO_HALF(scale_c.w));
            gx[i] = gx_i;
            ushort4 gy_i;
            _FLOAT_TO_US(gy_i.x, _gz_i.x);
            _FLOAT_TO_US(gy_i.y, _gz_i.y);
            _FLOAT_TO_US(gy_i.z, _gz_i.z);
            _FLOAT_TO_US(gy_i.w, _gz_i.w);
            gy[i] = gy_i;
        }
        if (pre_c >= 0) {
            atomicAdd(&sm_g_beta[pre_c].x,  my_g_beta.x);
            atomicAdd(&sm_g_beta[pre_c].y,  my_g_beta.y);
            atomicAdd(&sm_g_beta[pre_c].z,  my_g_beta.z);
            atomicAdd(&sm_g_beta[pre_c].w,  my_g_beta.w);
            atomicAdd(&sm_g_gamma[pre_c].x, my_g_gamma.x);
            atomicAdd(&sm_g_gamma[pre_c].y, my_g_gamma.y);
            atomicAdd(&sm_g_gamma[pre_c].z, my_g_gamma.z);
            atomicAdd(&sm_g_gamma[pre_c].w, my_g_gamma.w);
        }

        __syncthreads();
        float *ptr_g_beta = &(g_beta[0].x);
        float *ptr_g_gamma = &(g_gamma[0].x);
        const float *ptr_invstd = &(invstd[0].x);
        for (int i = threadIdx.x; i < 4*_c; i += blockDim.x) {
            atomicAdd(&ptr_g_beta[i], ptr_sm_g_beta[i]);
            atomicAdd(&ptr_g_gamma[i], ptr_sm_g_gamma[i] * ptr_invstd[i]);
        }
    }
    ''', 'cupy_resnet_bw_SBAR_gX_gGamma_vec4')

    def bw_scale_bias_relu_conv_bnorm(self, lid,
                                      x, i_scale, i_bias, W,
                                      y, o_scale, o_bias,
                                      gy, g_gamma, g_beta,
                                      gx=None):
        '''
        y = conv(relu(x * i_scale + i_bias), W)
        o_scale, o_bias = f(y, gamma, beta)
        '''
        n, x_h, x_w, x_c = x.shape
        _, y_h, y_w, y_c = y.shape

        gW = cupy.empty_like(W)

        bn_mode = self.bn_mode
        ptr_ph = self.ptr_ph
        
        # adjust gy
        if self.reference:
            _y = (y - self.saved_mean[lid]) * self.saved_invstd[lid]
            gy_dtype = gy.dtype
            gy = gy - (g_beta + g_gamma * _y) / (n * y_h * y_w)
            gy = gy.astype(gy_dtype)
        else:
            _n, _h, _w, _c = y.shape
            block_size = 1024
            n_blocks = _get_num_SM(y.data.device_id)
            # self._bw_scale_bias_relu_conv_bnorm_gY()(
            #     (n_blocks,), (block_size,),
            #     (y, g_gamma, g_beta,
            #      self.saved_mean[lid], self.saved_invstd[lid],
            #      gy,
            #      _n, _h, _w, _c // 2))
            self._bw_scale_bias_relu_conv_bnorm_gY_vec4()(
                (n_blocks,), (block_size,),
                (y, g_gamma, g_beta,
                 self.saved_mean[lid], self.saved_invstd[lid],
                 gy,
                 _n, _h, _w, _c // 4))
        
        # compute gW
        ops = libcudnn.CUDNN_FUSED_SCALE_BIAS_ACTIVATION_WGRAD
        key = (ops, bn_mode, x.shape, x.dtype, W.shape, W.dtype,
               y.shape, y.dtype, i_scale is None)
        if key in _plans:
            plan, workspace_size = _plans[key]
        else:
            x_desc = cudnn.create_tensor_descriptor(
                x, libcudnn.CUDNN_TENSOR_NHWC)
            w_desc = cudnn.create_filter_descriptor(
                W, libcudnn.CUDNN_TENSOR_NHWC)
            y_desc = cudnn.create_tensor_descriptor(
                y, libcudnn.CUDNN_TENSOR_NHWC)
            conv_desc = cudnn.create_convolution_descriptor(
                (self.pad[lid], self.pad[lid]),
                (self.stride[lid], self.stride[lid]),
                W.dtype, use_tensor_core=True)
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
        if i_scale is not None:
            var_param.extend([
                (libcudnn.CUDNN_PTR_BN_EQSCALE, i_scale),
                (libcudnn.CUDNN_PTR_BN_EQBIAS, i_bias),
            ])
        var_pack = cudnn.create_fused_ops_variant_param_pack(
            ops, var_param)
        cudnn.fused_ops_execute(plan, var_pack)
        del workspace

        # compute gx
        if gx is None:
            gx = cupy.empty_like(x)
            conv_beta = 0.0
        else:
            conv_beta = 1.0
        pad = (self.pad[lid], self.pad[lid])
        stride = (self.stride[lid], self.stride[lid])
        dilation = (1, 1)
        groups = 1
        deterministic = configuration.config.cudnn_deterministic
        auto_tune = configuration.config.autotune
        tensor_core = configuration.config.use_cudnn_tensor_core
        cudnn_tensor_layout = utils.get_cudnn_tensor_layout(self.tensor_layout)
        cuda.cudnn.convolution_backward_data(
            W, gy, None, gx, pad, stride, dilation, groups,
            deterministic=deterministic, auto_tune=auto_tune,
            tensor_core=tensor_core,
            d_layout=cudnn_tensor_layout, w_layout=cudnn_tensor_layout,
            beta=conv_beta)

        # compute gx, gi_scale and gi_bias
        if i_scale is None:
            g_gamma = None
            g_beta = None
        else:
            if self.reference:
                _x = x * i_scale + i_bias
                gx[_x <= 0] = 0
                gi_bias = cupy.sum(gx, axis=(0, 1, 2)).reshape((x_c,))
                gi_scale = cupy.sum(gx * x, axis=(0, 1, 2)).reshape((x_c,))
                gx = gx * i_scale
                g_gamma = gi_scale * self.saved_invstd[lid-1]
                g_beta = gi_bias.astype(g_gamma.dtype)
            else:
                _n, _h, _w, _c = x.shape
                g_gamma = cupy.zeros((x_c,), dtype=numpy.float32)
                g_beta = cupy.zeros((x_c,), dtype=numpy.float32)
                block_size = 1024
                n_blocks = _get_num_SM(x.data.device_id)
                assert _c <= 2048
                # self._bw_scale_bias_relu_conv_bnorm_gX_gScale_gBias()(
                #     (n_blocks,), (block_size,),
                #     (x, i_scale, i_bias, self.saved_invstd[lid-1],
                #      gx, g_gamma, g_beta,
                #      _n, _h, _w, _c // 2))
                self._bw_scale_bias_relu_conv_bnorm_gX_gGamma_vec4()(
                    (n_blocks,), (block_size,),
                    (x, i_scale, i_bias, self.saved_invstd[lid-1],
                     gx, g_gamma, g_beta,
                     _n, _h, _w, _c // 4))

        return gx, g_gamma, g_beta, gW

    def _bw_scale_bias_relu_conv_bnorm_gY(self):
        return cupy.RawKernel(r'''
#include <cuda_fp16.h>
    extern "C" __global__
    void cupy_resnet_bw_SBRCB_gY(
        const half2 *y, const float2 *g_gamma, const float2 *g_beta,
        const float2 *mean, const float2 *invstd,
        half2 *gy,
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
            gy_i.x = (half)_gy.x;
            gy_i.y = (half)_gy.y;
            gy[i] = gy_i;
        }
    }
        ''', 'cupy_resnet_bw_SBRCB_gY')

    def _bw_scale_bias_relu_conv_bnorm_gY_vec4(self):
        return cupy.RawKernel(r'''
#include <cuda_fp16.h>
#define _US_TO_HALF(var) *(reinterpret_cast<half *>(&(var)))
#define _HALF_TO_US(var) *(reinterpret_cast<unsigned short *>(&(var)))
#define _YMI(Y,M,I) (((float)_US_TO_HALF(Y) - M) * I)
#define _GYBGY(GY,B,G,Y,num) ((float)_US_TO_HALF(GY) - (B + G * Y) / num)
#define _FLOAT_TO_US(DST, SRC) do {half tmp = (half)(SRC); DST = _HALF_TO_US(tmp);} while (0)
    extern "C" __global__
    void cupy_resnet_bw_SBRCB_gY_vec4(
        const ushort4 *y, const float4 *g_gamma, const float4 *g_beta,
        const float4 *mean, const float4 *invstd,
        ushort4 *gy,
        int _n, int _h, int _w, int _c)
    {
        int tid = threadIdx.x + blockDim.x * blockIdx.x;
        int num_threads = blockDim.x * gridDim.x;
        int num_elements = _n * _h * _w * _c;
        float4 mean_c;
        float4 invstd_c;
        float4 g_beta_c;
        float4 g_gamma_c;
        int num = _n * _h * _w;
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
            ushort4 y_i = y[i];
            ushort4 gy_i = gy[i];
            float4 _y;
            _y.x = _YMI(y_i.x, mean_c.x, invstd_c.x);
            _y.y = _YMI(y_i.y, mean_c.y, invstd_c.y);
            _y.z = _YMI(y_i.z, mean_c.z, invstd_c.z);
            _y.w = _YMI(y_i.w, mean_c.w, invstd_c.w);
            float4 _gy;
            _gy.x = _GYBGY(gy_i.x, g_beta_c.x, g_gamma_c.x, _y.x, num);
            _gy.y = _GYBGY(gy_i.y, g_beta_c.y, g_gamma_c.y, _y.y, num);
            _gy.z = _GYBGY(gy_i.z, g_beta_c.z, g_gamma_c.z, _y.z, num);
            _gy.w = _GYBGY(gy_i.w, g_beta_c.w, g_gamma_c.w, _y.w, num);
            _FLOAT_TO_US(gy_i.x, _gy.x);
            _FLOAT_TO_US(gy_i.y, _gy.y);
            _FLOAT_TO_US(gy_i.z, _gy.z);
            _FLOAT_TO_US(gy_i.w, _gy.w);
            gy[i] = gy_i;
        }
    }
        ''', 'cupy_resnet_bw_SBRCB_gY_vec4')

    def _bw_scale_bias_relu_conv_bnorm_gX_gScale_gBias(self):
        return cupy.RawKernel(r'''
#include <cuda_fp16.h>
    extern "C" __global__
    void cupy_resnet_bw_SBRCB_gX_gGamma(
        const half2 *x, const half2 *scale, const half2 *bias,
        const float2 *invstd,
        half2 *gx, float2 *g_gamma, float2 *g_beta,
        int _n, int _h, int _w, int _c)
    {
        __shared__ float2 sm_g_beta[1024];
        __shared__ float2 sm_g_gamma[1024];
        for (int i = threadIdx.x; i < _c; i += blockDim.x) {
            sm_g_beta[i].x = 0.0;
            sm_g_beta[i].y = 0.0;
            sm_g_gamma[i].x = 0.0;
            sm_g_gamma[i].y = 0.0;
        }
        __syncthreads();

        int tid = threadIdx.x + blockDim.x * blockIdx.x;
        int num_threads = blockDim.x * gridDim.x;
        int num_elements = _n * _h * _w * _c;
        int pre_c = -1;
        float2  my_g_beta;
        float2  my_g_gamma;
        half2  my_scale;
        half2  my_bias;
        for (int i = tid; i < num_elements; i += num_threads) {
            int cur_c = i % _c;
            if (pre_c != cur_c) {
                if (pre_c >= 0) {
                    atomicAdd(&sm_g_beta[pre_c].x,  my_g_beta.x);
                    atomicAdd(&sm_g_beta[pre_c].y,  my_g_beta.y);
                    atomicAdd(&sm_g_gamma[pre_c].x, my_g_gamma.x);
                    atomicAdd(&sm_g_gamma[pre_c].y, my_g_gamma.y);
                }
                my_g_beta.x  = 0.0;
                my_g_beta.y  = 0.0;
                my_g_gamma.x = 0.0;
                my_g_gamma.y = 0.0;
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
            my_g_beta.x  += _gx.x;
            my_g_beta.y  += _gx.y;
            my_g_gamma.x += _gx.x * _x.x;
            my_g_gamma.y += _gx.y * _x.y;
            gx_i.x = (half)(_gx.x * (float)my_scale.x);
            gx_i.y = (half)(_gx.y * (float)my_scale.y);
            gx[i] = gx_i;
        }
        if (pre_c >= 0) {
            atomicAdd(&sm_g_beta[pre_c].x,  my_g_beta.x);
            atomicAdd(&sm_g_beta[pre_c].y,  my_g_beta.y);
            atomicAdd(&sm_g_gamma[pre_c].x, my_g_gamma.x);
            atomicAdd(&sm_g_gamma[pre_c].y, my_g_gamma.y);
        }

        __syncthreads();
        for (int i = threadIdx.x; i < _c; i += blockDim.x) {
            atomicAdd(&g_beta[i].x, sm_g_beta[i].x);
            atomicAdd(&g_beta[i].y, sm_g_beta[i].y);
            atomicAdd(&g_gamma[i].x, sm_g_gamma[i].x * invstd[i].x);
            atomicAdd(&g_gamma[i].y, sm_g_gamma[i].y * invstd[i].y);
        }
    }
    ''', 'cupy_resnet_bw_SBRCB_gX_gGamma')

    def _bw_scale_bias_relu_conv_bnorm_gX_gGamma_vec4(self):
        return cupy.RawKernel(r'''
#include <cuda_fp16.h>
#define _US_TO_HALF(var) *(reinterpret_cast<half *>(&(var)))
#define _HALF_TO_US(var) *(reinterpret_cast<unsigned short *>(&(var)))
#define _XSB(X,S,B) ((float)_US_TO_HALF(X) * (float)_US_TO_HALF(S) + (float)_US_TO_HALF(B))
#define _GXS(D,GX,S) do {half tmp = (half)((GX) * (float)_US_TO_HALF(S)); D = _HALF_TO_US(tmp);} while (0)
    extern "C" __global__
    void cupy_resnet_bw_SBRCB_gX_gGamma_vec4(
        const ushort4 *x, const ushort4 *scale, const ushort4 *bias,
        const float4 *invstd,
        ushort4 *gx,
        float4 *g_gamma, float4 *g_beta,
        int _n, int _h, int _w, int _c)
    {
        __shared__ float4 sm_g_beta[512];
        __shared__ float4 sm_g_gamma[512];
        float *ptr_sm_g_beta = &(sm_g_beta[0].x);
        float *ptr_sm_g_gamma = &(sm_g_gamma[0].x);
        for (int i = threadIdx.x; i < 4*_c; i += blockDim.x) {
            ptr_sm_g_beta[i] = 0.0;
            ptr_sm_g_gamma[i] = 0.0;
        }
        __syncthreads();

        int tid = threadIdx.x + blockDim.x * blockIdx.x;
        int num_threads = blockDim.x * gridDim.x;
        int num_elements = _n * _h * _w * _c;
        int pre_c = -1;
        float4  my_g_beta;
        float4  my_g_gamma;
        ushort4  my_scale;
        ushort4  my_bias;
        for (int i = tid; i < num_elements; i += num_threads) {
            int cur_c = i % _c;
            if (pre_c != cur_c) {
                if (pre_c >= 0) {
                    atomicAdd(&sm_g_beta[pre_c].x,  my_g_beta.x);
                    atomicAdd(&sm_g_beta[pre_c].y,  my_g_beta.y);
                    atomicAdd(&sm_g_beta[pre_c].z,  my_g_beta.z);
                    atomicAdd(&sm_g_beta[pre_c].w,  my_g_beta.w);
                    atomicAdd(&sm_g_gamma[pre_c].x, my_g_gamma.x);
                    atomicAdd(&sm_g_gamma[pre_c].y, my_g_gamma.y);
                    atomicAdd(&sm_g_gamma[pre_c].z, my_g_gamma.z);
                    atomicAdd(&sm_g_gamma[pre_c].w, my_g_gamma.w);
                }
                my_g_beta.x  = 0.0;
                my_g_beta.y  = 0.0;
                my_g_beta.z  = 0.0;
                my_g_beta.w  = 0.0;
                my_g_gamma.x = 0.0;
                my_g_gamma.y = 0.0;
                my_g_gamma.z = 0.0;
                my_g_gamma.w = 0.0;
                my_scale = scale[cur_c];
                my_bias  = bias[cur_c];
            }
            pre_c = cur_c;
            ushort4 x_i = x[i];
            float4 _x;
            _x.x = _XSB(x_i.x, my_scale.x, my_bias.x);
            _x.y = _XSB(x_i.y, my_scale.y, my_bias.y);
            _x.z = _XSB(x_i.z, my_scale.z, my_bias.z);
            _x.w = _XSB(x_i.w, my_scale.w, my_bias.w);
            ushort4 gx_i = gx[i];
            float4 _gx;
            _gx.x = (float)_US_TO_HALF(gx_i.x);
            _gx.y = (float)_US_TO_HALF(gx_i.y);
            _gx.z = (float)_US_TO_HALF(gx_i.z);
            _gx.w = (float)_US_TO_HALF(gx_i.w);
            if (_x.x <= 0.0) _gx.x = 0.0;
            if (_x.y <= 0.0) _gx.y = 0.0;
            if (_x.z <= 0.0) _gx.z = 0.0;
            if (_x.w <= 0.0) _gx.w = 0.0;
            my_g_beta.x  += _gx.x;
            my_g_beta.y  += _gx.y;
            my_g_beta.z  += _gx.z;
            my_g_beta.w  += _gx.w;
            my_g_gamma.x += _gx.x * _x.x;
            my_g_gamma.y += _gx.y * _x.y;
            my_g_gamma.z += _gx.z * _x.z;
            my_g_gamma.w += _gx.w * _x.w;
            _GXS(gx_i.x, _gx.x, my_scale.x);
            _GXS(gx_i.y, _gx.y, my_scale.y);
            _GXS(gx_i.z, _gx.z, my_scale.z);
            _GXS(gx_i.w, _gx.w, my_scale.w);
            gx[i] = gx_i;
        }
        if (pre_c >= 0) {
            atomicAdd(&sm_g_beta[pre_c].x,  my_g_beta.x);
            atomicAdd(&sm_g_beta[pre_c].y,  my_g_beta.y);
            atomicAdd(&sm_g_beta[pre_c].z,  my_g_beta.z);
            atomicAdd(&sm_g_beta[pre_c].w,  my_g_beta.w);
            atomicAdd(&sm_g_gamma[pre_c].x, my_g_gamma.x);
            atomicAdd(&sm_g_gamma[pre_c].y, my_g_gamma.y);
            atomicAdd(&sm_g_gamma[pre_c].z, my_g_gamma.z);
            atomicAdd(&sm_g_gamma[pre_c].w, my_g_gamma.w);
        }

        __syncthreads();
        float *ptr_g_beta = &(g_beta[0].x);
        float *ptr_g_gamma = &(g_gamma[0].x);
        const float *ptr_invstd = &(invstd[0].x);
        for (int i = threadIdx.x; i < 4*_c; i += blockDim.x) {
            atomicAdd(&ptr_g_beta[i], ptr_sm_g_beta[i]);
            atomicAdd(&ptr_g_gamma[i], ptr_sm_g_gamma[i] * ptr_invstd[i]);
        }
    }
    ''', 'cupy_resnet_bw_SBRCB_gX_gGamma_vec4')


def resnet_bottle_neck(x, h, W_set, gamma_set, beta_set,
                       running_mean_set, running_var_set,
                       stride=1, bn_eps=2.5e-5, bn_decay=0.9):
    fnode = ResNetBottleNeckFunction(running_mean_set, running_var_set,
                                     stride, bn_eps, bn_decay)
    args = (x,
            W_set[0], gamma_set[0], beta_set[0],
            W_set[1], gamma_set[1], beta_set[1],
            W_set[2], gamma_set[2], beta_set[2])
    if h is not None:
        args = args + (h,)
    y, = fnode.apply(args)
    return y
