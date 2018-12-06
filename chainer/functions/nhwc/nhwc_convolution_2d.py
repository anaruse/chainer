import numpy

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer.backends import intel64
from chainer import configuration
from chainer import function_node
import chainer.functions
from chainer.utils import argument
from chainer.utils import conv
from chainer.utils import type_check
import chainerx

if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cuda.cuda.cudnn
    _cudnn_version = cuda.cuda.cudnn.getVersion()

import cupy
from cupy import prof


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x


# Used by deconvolution_2d.py.
# TODO(beam2d): Unify matmul implementations
def _matmul(a, b):
    xp = backend.get_array_module(a)
    if not hasattr(xp, 'matmul'):
        # NumPy 1.9 does not support matmul. We use einsum instead.
        return xp.einsum('ijl,ilk->ijk', a, b)
    return xp.matmul(a, b)


class NhwcConvolution2DFunction(function_node.FunctionNode):

    _use_ideep = False

    def __init__(self, stride=1, pad=0, cover_all=False, **kwargs):
        dilate, groups, d_layout, w_layout = argument.parse_kwargs(
            kwargs, ('dilate', 1), ('groups', 1), ('d_layout', 'NHWC'), ('w_layout', 'NHWC'),
            deterministic="deterministic argument is not supported anymore. "
            "Use chainer.using_config('cudnn_deterministic', value) context "
            "where value is either `True` or `False`.",
            requires_x_grad="requires_x_grad argument is not supported "
            "anymore. Just remove the argument. Note that whether to compute "
            "the gradient w.r.t. x is automatically decided during "
            "backpropagation.")

        self.sy, self.sx = _pair(stride)
        self.ph, self.pw = _pair(pad)
        self.cover_all = cover_all
        self.dy, self.dx = _pair(dilate)
        self.groups = groups
        if d_layout not in ('NCHW', 'NHWC'):
            raise ValueError('unsupported d_layout: {}'.format(d_layout))
        self.d_layout = d_layout
        if w_layout not in ('NCHW', 'NHWC'):
            raise ValueError('unsupported w_layout: {}'.format(w_layout))
        self.w_layout = w_layout

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(2 <= n_in, n_in <= 3)

        x_type = in_types[0]
        w_type = in_types[1]
        if self.d_layout == 'NCHW':
            xC = x_type.shape[1]
        else:
            xC = x_type.shape[3]
        if self.w_layout == 'NCHW':
            wC = w_type.shape[1]
        else:
            wC = w_type.shape[3]
        type_check.expect(
            x_type.dtype.kind == 'f',
            w_type.dtype.kind == 'f',
            x_type.ndim == 4,
            w_type.ndim == 4,
            xC == wC * self.groups,
        )

        if type_check.eval(n_in) == 3:
            b_type = in_types[2]
            type_check.expect(
                b_type.dtype == x_type.dtype,
                b_type.ndim == 1,
                b_type.shape[0] == w_type.shape[0],
            )

    def _get_out_size(self, inputs):
        x, W = inputs[:2]
        if self.w_layout == 'NCHW':
            _, _, kh, kw = W.shape
        else:
            _, kh, kw, _ = W.shape
        if self.d_layout == 'NCHW':
            _, _, h, w = x.shape
        else:
            _, h, w, _ = x.shape
        out_h = conv.get_conv_outsize(
            h, kh, self.sy, self.ph, cover_all=self.cover_all, d=self.dy)
        if out_h <= 0:
            raise RuntimeError('Height in the output should be positive.')
        out_w = conv.get_conv_outsize(
            w, kw, self.sx, self.pw, cover_all=self.cover_all, d=self.dx)
        if out_w <= 0:
            raise RuntimeError('Width in the output should be positive.')
        return out_h, out_w

    def forward_cpu(self, inputs):
        self.retain_inputs((0, 1))  # retain only x and W
        if len(inputs) == 2:
            (x, W), b = inputs, None
        else:
            x, W, b = inputs

        if (intel64.should_use_ideep('>=auto')
                and intel64.inputs_all_ready(inputs)):
            self._use_ideep = True

        if self.groups > 1:
            return self._forward_grouped_convolution(x, W, b)
        else:
            return self._forward_cpu_core(x, W, b)

    # @prof.TimeRangeDecorator('Conv.forward_gpu')
    def forward_gpu(self, inputs):
        self.retain_inputs((0, 1))  # retain only x and W
        if len(inputs) == 2:
            (x, W), b = inputs, None
        else:
            x, W, b = inputs

        if self.w_layout == 'NCHW':
            out_c, _, kh, kw = W.shape
        else:
            out_c, kh, kw, _ = W.shape
        if self.d_layout == 'NCHW':
            n, _, h, w = x.shape
        else:
            n, h, w, _ = x.shape

        out_h, out_w = self._get_out_size(inputs)

        if self.d_layout == 'NCHW':
            y = cuda.cupy.empty((n, out_c, out_h, out_w), dtype=x.dtype)
        else:
            y = cuda.cupy.empty((n, out_h, out_w, out_c), dtype=x.dtype)

        use_cudnn = (
            chainer.should_use_cudnn('>=auto')
            and not self.cover_all
            and x.dtype == W.dtype
            and ((self.dy == 1 and self.dx == 1) or _cudnn_version >= 6000)
            and (self.groups <= 1 or _cudnn_version >= 7000)
        )
        if not use_cudnn:
            raise RuntimeError('this functions is available with cuDNN.')
        
        return self._forward_cudnn(x, W, b, y)

    # @prof.TimeRangeDecorator('Conv.forward_cudnn')
    def _forward_cudnn(self, x, W, b, y):
        pad = (self.ph, self.pw)
        stride = (self.sy, self.sx)
        dilation = (self.dy, self.dx)
        auto_tune = configuration.config.autotune
        tensor_core = configuration.config.use_cudnn_tensor_core
        # print('# x.shape: {}'.format(x.shape))
        # print('# W.shape: {}'.format(W.shape))
        # print('# y.shape: {}'.format(y.shape))
        # print('# x.data.ptr: {}'.format(x.data.ptr))
        # print('# W.data.ptr: {}'.format(W.data.ptr))
        # print('# y.data.ptr: {}'.format(y.data.ptr))
        # print('# x.dtype: {}'.format(x.dtype))
        # print('# W.dtype: {}'.format(W.dtype))
        # print('# y.dtype: {}'.format(y.dtype))
        if self.d_layout == 'NCHW':
            cudnn_d_layout = libcudnn.CUDNN_TENSOR_NCHW
        else:
            cudnn_d_layout = libcudnn.CUDNN_TENSOR_NHWC
        if self.w_layout == 'NCHW':
            cudnn_w_layout = libcudnn.CUDNN_TENSOR_NCHW
        else:
            cudnn_w_layout = libcudnn.CUDNN_TENSOR_NHWC
        cuda.cudnn.convolution_forward(
            x, W, b, y, pad, stride, dilation, self.groups,
            auto_tune=auto_tune, tensor_core=tensor_core,
            d_layout=cudnn_d_layout, w_layout=cudnn_w_layout)
        return y,

    # @prof.TimeRangeDecorator('Conv.backward')
    def backward(self, indexes, grad_outputs):
        x, W = self.get_retained_inputs()
        gy, = grad_outputs

        ret = []
        if 0 in indexes:
            if self.d_layout == 'NCHW':
                _, _, xh, xw = x.shape
            else:
                _, xh, xw, _ = x.shape  # 'NHWC'
            gx = chainer.functions.nhwc_deconvolution_2d(
                gy, W, stride=(self.sy, self.sx), pad=(self.ph, self.pw),
                outsize=(xh, xw), dilate=(self.dy, self.dx),
                groups=self.groups, d_layout=self.d_layout, w_layout=self.w_layout)
            ret.append(gx)
        if 1 in indexes:
            gW, = NhwcConvolution2DGradW(self).apply((x, gy))
            ret.append(gW)
        if 2 in indexes:
            gb = chainer.functions.sum(gy, axis=(0, 2, 3))
            ret.append(gb)

        return ret


class NhwcConvolution2DGradW(function_node.FunctionNode):

    def __init__(self, conv2d):
        W_node = conv2d.inputs[1]
        if conv2d.w_layout == 'NCHW':
            _, _, self.kh, self.kw = W_node.shape
        else:
            _, self.kh, self.kw, _ = W_node.shape
        self.sy = conv2d.sy
        self.sx = conv2d.sx
        self.ph = conv2d.ph
        self.pw = conv2d.pw
        self.dy = conv2d.dy
        self.dx = conv2d.dx
        self.cover_all = conv2d.cover_all
        self.W_dtype = W_node.dtype
        self.groups = conv2d.groups
        self.d_layout = conv2d.d_layout
        self.w_layout = conv2d.w_layout

    def forward_cpu(self, inputs):
        self.retain_inputs((0, 1))
        x, gy = inputs

        if self.groups > 1:
            return self._forward_grouped_convolution(x, gy)
        else:
            return self._forward_cpu_core(x, gy)

    def _forward_cpu_core(self, x, gy):
        if self._use_ideep:
            return self._forward_ideep(x, gy)

        # NumPy raises an error when the array is not contiguous.
        # See: https://github.com/chainer/chainer/issues/2744
        # TODO(niboshi): Remove this code when NumPy is fixed.
        if (not (gy.flags.c_contiguous or gy.flags.f_contiguous) and
                1 in gy.shape):
            gy = numpy.ascontiguousarray(gy)

        col = conv.im2col_cpu(
            x, self.kh, self.kw, self.sy, self.sx, self.ph, self.pw,
            cover_all=self.cover_all, dy=self.dy, dx=self.dx)
        gW = numpy.tensordot(gy, col, ((0, 2, 3), (0, 4, 5))
                             ).astype(self.W_dtype, copy=False)
        return gW,

    # @prof.TimeRangeDecorator('GradW.forward_gpu')
    def forward_gpu(self, inputs):
        self.retain_inputs((0, 1))
        x, gy = inputs

        use_cudnn = (
            chainer.should_use_cudnn('>=auto')
            and not self.cover_all
            and x.dtype == self.W_dtype
            and ((self.dy == 1 and self.dx == 1)
                 or (_cudnn_version >= 6000
                     and not configuration.config.cudnn_deterministic))
            and (self.groups <= 1 or _cudnn_version >= 7000)
        )
        if not use_cudnn:
            raise RuntimeError('this functions is available with cuDNN.')

        return self._forward_cudnn(x, gy)

    # @prof.TimeRangeDecorator('GradW.forward_cudnn')
    def _forward_cudnn(self, x, gy):
        if self.d_layout == 'NCHW':
            _, out_c, out_h, out_w = gy.shape
            n, c, h, w = x.shape
        else:
            _, out_h, out_w, out_c = gy.shape
            n, h, w, c = x.shape

        iC = c
        iCg = int(iC / self.groups)
        if self.w_layout == 'NCHW':
            gW = cuda.cupy.empty((out_c, iCg, self.kh, self.kw),
                                 dtype=self.W_dtype)
        else:
            gW = cuda.cupy.empty((out_c, self.kh, self.kw, iCg),
                                 dtype=self.W_dtype)
        pad = (self.ph, self.pw)
        stride = (self.sy, self.sx)
        dilation = (self.dy, self.dx)
        deterministic = configuration.config.cudnn_deterministic
        auto_tune = configuration.config.autotune
        tensor_core = configuration.config.use_cudnn_tensor_core
        if self.d_layout == 'NCHW':
            cudnn_d_layout = libcudnn.CUDNN_TENSOR_NCHW
        else:
            cudnn_d_layout = libcudnn.CUDNN_TENSOR_NHWC
        if self.w_layout == 'NCHW':
            cudnn_w_layout = libcudnn.CUDNN_TENSOR_NCHW
        else:
            cudnn_w_layout = libcudnn.CUDNN_TENSOR_NHWC
        cuda.cudnn.convolution_backward_filter(
            x, gy, gW, pad, stride, dilation, self.groups,
            deterministic=deterministic, auto_tune=auto_tune,
            tensor_core=tensor_core,
            d_layout=cudnn_d_layout, w_layout=cudnn_w_layout)

        return gW,

    def backward(self, indexes, grad_outputs):
        x, gy = self.get_retained_inputs()
        ggW, = grad_outputs

        ret = []
        if 0 in indexes:
            if self.d_layout == 'NCHW':
                _, _, xh, xw = x.shape
            else:
                _, xh, xw, _ = x.shape
            gx = chainer.functions.nhwc_deconvolution_2d(
                gy, ggW, stride=(self.sy, self.sx), pad=(self.ph, self.pw),
                outsize=(xh, xw), dilate=(self.dy, self.dx),
                groups=self.groups, d_layout=self.d_layout, w_layout=self.w_layout)
            ret.append(gx)
        if 1 in indexes:
            ggy = convolution_2d(
                x, ggW, stride=(self.sy, self.sx), pad=(self.ph, self.pw),
                cover_all=self.cover_all, dilate=(self.dy, self.dx),
                groups=self.groups, d_layout=self.d_layout, w_layout=self.w_layout)
            ret.append(ggy)

        return ret


def nhwc_convolution_2d(x, W, b=None, stride=1, pad=0, cover_all=False, **kwargs):
    """convolution_2d(x, W, b=None, stride=1, pad=0, cover_all=False, *, dilate=1, groups=1)

    Two-dimensional convolution function.

    This is an implementation of two-dimensional convolution in ConvNets.
    It takes three variables: the input image ``x``, the filter weight ``W``,
    and the bias vector ``b``.

    Notation: here is a notation for dimensionalities.

    - :math:`n` is the batch size.
    - :math:`c_I` and :math:`c_O` are the number of the input and output
      channels, respectively.
    - :math:`h_I` and :math:`w_I` are the height and width of the input image,
      respectively.
    - :math:`h_K` and :math:`w_K` are the height and width of the filters,
      respectively.
    - :math:`h_P` and :math:`w_P` are the height and width of the spatial
      padding size, respectively.

    Then the ``Convolution2D`` function computes correlations between filters
    and patches of size :math:`(h_K, w_K)` in ``x``.
    Note that correlation here is equivalent to the inner product between
    expanded vectors.
    Patches are extracted at positions shifted by multiples of ``stride`` from
    the first position ``(-h_P, -w_P)`` for each spatial axis.
    The right-most (or bottom-most) patches do not run over the padded spatial
    size.

    Let :math:`(s_Y, s_X)` be the stride of filter application. Then, the
    output size :math:`(h_O, w_O)` is determined by the following equations:

    .. math::

       h_O &= (h_I + 2h_P - h_K) / s_Y + 1,\\\\
       w_O &= (w_I + 2w_P - w_K) / s_X + 1.

    If ``cover_all`` option is ``True``, the filter will cover the all
    spatial locations. So, if the last stride of filter does not cover the
    end of spatial locations, an addtional stride will be applied to the end
    part of spatial locations. In this case, the output size :math:`(h_O, w_O)`
    is determined by the following equations:

    .. math::

       h_O &= (h_I + 2h_P - h_K + s_Y - 1) / s_Y + 1,\\\\
       w_O &= (w_I + 2w_P - w_K + s_X - 1) / s_X + 1.

    If the bias vector is given, then it is added to all spatial locations of
    the output of convolution.

    The output of this function can be non-deterministic when it uses cuDNN.
    If ``chainer.configuration.config.cudnn_deterministic`` is ``True`` and
    cuDNN version is >= v3, it forces cuDNN to use a deterministic algorithm.

    Convolution links can use a feature of cuDNN called autotuning, which
    selects the most efficient CNN algorithm for images of fixed-size,
    can provide a significant performance boost for fixed neural nets.
    To enable, set `chainer.using_config('autotune', True)`

    When the dilation factor is greater than one, cuDNN is not used unless
    the version is 6.0 or higher.

    .. warning::

        ``deterministic`` argument is not supported anymore since v2.
        Instead, use ``chainer.using_config('cudnn_deterministic', value)``
        (value is either ``True`` or ``False``).
        See :func:`chainer.using_config`.

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Input variable of shape :math:`(n, c_I, h_I, w_I)`.
        W (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Weight variable of shape :math:`(c_O, c_I, h_K, w_K)`.
        b (None or :class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Bias variable of length :math:`c_O` (optional).
        stride (:class:`int` or pair of :class:`int` s):
            Stride of filter applications. ``stride=s`` and ``stride=(s, s)``
            are equivalent.
        pad (:class:`int` or pair of :class:`int` s):
            Spatial padding width for input arrays.
            ``pad=p`` and ``pad=(p, p)`` are equivalent.
        cover_all (:class:`bool`):
            If ``True``, all spatial locations are convoluted into some output
            pixels.
        dilate (:class:`int` or pair of :class:`int` s):
            Dilation factor of filter applications.
            ``dilate=d`` and ``dilate=(d, d)`` are equivalent.
        groups (:class:`int`): Number of groups of channels. If the number
            is greater than 1, input tensor :math:`W` is divided into some
            blocks by this value. For each tensor blocks, convolution
            operation will be executed independently. Input channel size
            :math:`c_I` and output channel size :math:`c_O` must be exactly
            divisible by this value.

    Returns:
        ~chainer.Variable:
            Output variable of shape :math:`(n, c_O, h_O, w_O)`.

    .. seealso:: :class:`~chainer.links.Convolution2D`

    .. admonition:: Example

        >>> n = 10
        >>> c_i, c_o = 3, 1
        >>> h_i, w_i = 30, 40
        >>> h_k, w_k = 10, 10
        >>> h_p, w_p = 5, 5
        >>> x = np.random.uniform(0, 1, (n, c_i, h_i, w_i)).astype(np.float32)
        >>> x.shape
        (10, 3, 30, 40)
        >>> W = np.random.uniform(0, 1, (c_o, c_i, h_k, w_k)).\
astype(np.float32)
        >>> W.shape
        (1, 3, 10, 10)
        >>> b = np.random.uniform(0, 1, (c_o,)).astype(np.float32)
        >>> b.shape
        (1,)
        >>> s_y, s_x = 5, 7
        >>> y = F.convolution_2d(x, W, b, stride=(s_y, s_x), pad=(h_p, w_p))
        >>> y.shape
        (10, 1, 7, 6)
        >>> h_o = int((h_i + 2 * h_p - h_k) / s_y + 1)
        >>> w_o = int((w_i + 2 * w_p - w_k) / s_x + 1)
        >>> y.shape == (n, c_o, h_o, w_o)
        True
        >>> y = F.convolution_2d(x, W, b, stride=(s_y, s_x), pad=(h_p, w_p), \
cover_all=True)
        >>> y.shape == (n, c_o, h_o, w_o + 1)
        True

    """  # NOQA
    dilate, groups, d_layout, w_layout = argument.parse_kwargs(
        kwargs, ('dilate', 1), ('groups', 1), ('d_layout', 'NHWC'), ('w_layout', 'NHWC'),
        deterministic="deterministic argument is not supported anymore. "
        "Use chainer.using_config('cudnn_deterministic', value) "
        "context where value is either `True` or `False`.")

    fnode = NhwcConvolution2DFunction(stride, pad, cover_all, dilate=dilate,
                                      groups=groups, d_layout=d_layout, w_layout=w_layout)
    if b is None:
        args = x, W
    else:
        args = x, W, b
    y, = fnode.apply(args)
    return y
