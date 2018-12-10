import numpy

import chainer
from chainer.backends import cuda
from chainer.backends import intel64
from chainer import configuration
from chainer import function_node
import chainer.functions
from chainer.functions.connection import convolution_2d
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


class NhwcDeconvolution2DFunction(function_node.FunctionNode):

    cover_all = None
    _use_ideep = False

    def __init__(self, stride=1, pad=0, outsize=None, **kwargs):
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
        self.outh, self.outw = (None, None) if outsize is None else outsize
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
        x_type, w_type = in_types[:2]

        type_check.expect(
            x_type.dtype.kind == 'f',
            w_type.dtype.kind == 'f',
            x_type.ndim == 4,
            w_type.ndim == 4,
            # x_type.shape[1] == w_type.shape[0]  # todo
        )

        if self.d_layout == 'NCHW':
            # _, xC, xH, xW = x_type.shape[0:4]
            xC = x_type.shape[1]
            xH = x_type.shape[2]
            xW = x_type.shape[3]
        else:
            # _, xH, xW, xC = x_type.shape[0:4]
            xH = x_type.shape[1]
            xW = x_type.shape[2]
            xC = x_type.shape[3]
        if self.w_layout == 'NCHW':
            # _, wC, wH, wW = w_type.shape[0:4]
            wC = w_type.shape[1]
            wH = w_type.shape[2]
            wW = w_type.shape[3]
        else:
            # _, wH, wW, wC = w_type.shape[0:4]
            wH = w_type.shape[1]
            wW = w_type.shape[2]
            wC = w_type.shape[3]

        if self.outh is not None:
            lower_bound = conv.get_conv_outsize(
                self.outh, wH, self.sy, self.ph,
                d=self.dy)
            upper_bound = conv.get_conv_outsize(
                self.outh, wH, self.sy, self.ph, cover_all=True,
                d=self.dy)
            type_check.expect(lower_bound <= xH, xH <= upper_bound)
        if self.outw is not None:
            lower_bound = conv.get_conv_outsize(
                self.outw, wW, self.sx, self.pw, d=self.dx)
            upper_bound = conv.get_conv_outsize(
                self.outw, wW, self.sx, self.pw, cover_all=True, d=self.dx)
            type_check.expect(
                lower_bound <= xW, xW <= upper_bound)

        if type_check.eval(n_in) == 3:
            b_type = in_types[2]
            type_check.expect(
                b_type.dtype == x_type.dtype,
                b_type.ndim == 1,
                # Need to consider the case that group count > 1.
                # b_type.shape[0] == w_type.shape[1],
            )

    def _calc_out_size(self, x, W):
        """Calculates and stores `outh` and `outw`."""
        if self.w_layout == 'NCHW':
            _, _, kh, kw = W.shape
        else:
            _, kh, kw, _ = W.shape
        if self.d_layout == 'NCHW':
            _, _, in_h, in_w = x.shape
        else:
            _, in_h, in_w, _ = x.shape
        # - k, m, n: shape of out_channel
        # - b: number of inputs
        # - h, w: height and width of kernels
        # k, m, n, b, h, w -> b, k, m, n, h, w
        if self.outh is None:
            self.outh = conv.get_deconv_outsize(
                in_h, kh, self.sy, self.ph, d=self.dy)
            if self.outh <= 0:
                raise RuntimeError('Height in the output must be positive.')

        if self.outw is None:
            self.outw = conv.get_deconv_outsize(
                in_w, kw, self.sx, self.pw, d=self.dx)
            if self.outw <= 0:
                raise RuntimeError('Width in the output must be positive.')

    def forward_cpu(self, inputs):
        if ((self.dy == 1 and self.dx == 1)
                and intel64.should_use_ideep('>=auto')
                and intel64.inputs_all_ready(inputs)):
            self._use_ideep = True

        self.retain_inputs((0, 1))  # only retain x and W
        if len(inputs) == 2:
            (x, W), b = inputs, None
        else:
            x, W, b = inputs

        self._calc_out_size(x, W)

        if self.groups > 1:
            # Grouped convolution implementation
            return self._forward_grouped_convolution(x, W, b)

        elif (intel64.should_use_ideep('>=auto')
              and intel64.inputs_all_ready(inputs)):
            # iDeep implementation
            self._use_ideep = True
            return self._forward_ideep(x, W, b)

        else:
            return self._forward_cpu_core(x, W, b)

    def _forward_cpu_core(self, x, W, b):
        if self._use_ideep:
            return self._forward_ideep(x, W, b)

        gcol = numpy.tensordot(W, x, (0, 1)).astype(x.dtype, copy=False)
        gcol = numpy.rollaxis(gcol, 3)
        y = conv.col2im_cpu(
            gcol, self.sy, self.sx, self.ph, self.pw, self.outh, self.outw,
            dy=self.dy, dx=self.dx)
        # b, k, h, w
        if b is not None:
            y += b.reshape((1, b.size, 1, 1))
        return y,

    # @prof.TimeRangeDecorator('Deconv.forward_gpu')
    def forward_gpu(self, inputs):
        self.retain_inputs((0, 1))  # only retain x and W
        if len(inputs) == 2:
            (x, W), b = inputs, None
        else:
            x, W, b = inputs

        self._calc_out_size(x, W)
        self._set_cover_all(x, W)

        use_cudnn = (
            chainer.should_use_cudnn('>=auto')
            and not self.cover_all
            and x.dtype == W.dtype
            and ((self.dy == 1 and self.dx == 1)
                 or (_cudnn_version >= 6000
                     and not configuration.config.cudnn_deterministic))
            and (self.groups <= 1 or _cudnn_version >= 7000)
        )
        if not use_cudnn:
            raise RuntimeError('this functions is available with cuDNN.')

        return self._forward_cudnn(x, W, b)

    # @prof.TimeRangeDecorator('Deconv.forward_cudnn')
    def _forward_cudnn(self, x, W, b):
        n = len(x)
        if self.w_layout == 'NCHW':
            yC = W.shape[1] * self.groups
        else:
            yC = W.shape[3] * self.groups

        if self.d_layout == 'NCHW':
            y = cuda.cupy.empty((n, yC, self.outh, self.outw), dtype=x.dtype)
        else:
            y = cuda.cupy.empty((n, self.outh, self.outw, yC), dtype=x.dtype)
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
        cuda.cudnn.convolution_backward_data(
            W, x, b, y, pad, stride, dilation, self.groups,
            deterministic=deterministic, auto_tune=auto_tune,
            tensor_core=tensor_core,
            d_layout=cudnn_d_layout, w_layout=cudnn_w_layout)

        return y,

    # @prof.TimeRangeDecorator('Deconv.backward')
    def backward(self, indexes, grad_outputs):
        x, W = self.get_retained_inputs()
        gy, = grad_outputs

        ret = []
        if 0 in indexes:
            if self.cover_all is None:
                self._set_cover_all(x, W)
            gx = chainer.functions.nhwc_convolution_2d(
                gy, W, stride=(self.sy, self.sx), pad=(self.ph, self.pw),
                cover_all=self.cover_all, dilate=(self.dy, self.dx),
                groups=self.groups, d_layout=self.d_layout, w_layout=self.w_layout)
            ret.append(gx)
        if 1 in indexes:
            if self.cover_all is None:
                self._set_cover_all(x, W)
            gW, = convolution_2d.NhwcConvolution2DGradW(self).apply((gy, x))
            ret.append(gW)
        if 2 in indexes:
            gb = chainer.functions.sum(gy, axis=(0, 2, 3))
            ret.append(gb)

        return ret

    def _set_cover_all(self, x, W):
        if self.d_layout == 'NCHW':
            _, _, in_h, in_w = x.shape
        else:
            _, in_h, in_w, _ = x.shape
        if self.d_layout == 'NCHW':
            _, _, kh, kw = W.shape
        else:
            _, kh, kw, _ = W.shape
        self.cover_all = (
            in_h != conv.get_conv_outsize(self.outh, kh, self.sy,
                                          self.ph, d=self.dy) or
            in_w != conv.get_conv_outsize(self.outw, kw, self.sx,
                                          self.pw, d=self.dx))


def nhwc_deconvolution_2d(x, W, b=None, stride=1, pad=0, outsize=None, **kwargs):
    """deconvolution_2d(x, W, b=None, stride=1, pad=0, outsize=None, *, dilate=1, groups=1)

    Two dimensional deconvolution function.

    This is an implementation of two-dimensional deconvolution. In most of deep
    learning frameworks and papers, this function is called
    **transposed convolution**. But because of historical reasons (e.g. paper
    by Ziller `Deconvolutional Networks`_) and backward compatibility, this
    function is called **deconvolution** in Chainer.

    .. _Deconvolutional Networks: \
http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf

    It takes three variables: input image ``x``,
    the filter weight ``W``, and the bias vector ``b``.

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

    Let :math:`(s_Y, s_X)` be the stride of filter application. Then, the
    output size :math:`(h_O, w_O)` is estimated by the following equations:

    .. math::

       h_O &= s_Y (h_I - 1) + h_K - 2h_P,\\\\
       w_O &= s_X (w_I - 1) + w_K - 2w_P.

    The output of this function can be non-deterministic when it uses cuDNN.
    If ``chainer.configuration.config.deterministic`` is ``True`` and
    cuDNN version is >= v3, it forces cuDNN to use a deterministic algorithm.

    Deconvolution links can use a feature of cuDNN called autotuning, which
    selects the most efficient CNN algorithm for images of fixed-size,
    can provide a significant performance boost for fixed neural nets.
    To enable, set `chainer.using_config('autotune', True)`

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
            Weight variable of shape :math:`(c_I, c_O, h_K, w_K)`.
        b (None or :class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Bias variable of length :math:`c_O` (optional).
        stride (:class:`int` or pair of :class:`int` s):
            Stride of filter applications. ``stride=s`` and ``stride=(s, s)``
            are equivalent.
        pad (:class:`int` or pair of :class:`int` s):
            Spatial padding width for input arrays.
            ``pad=p`` and ``pad=(p, p)`` are equivalent.
        outsize (None or :class:`tuple` of :class:`int` s):
            Expected output size of deconvolutional operation.
            It should be pair of height and width :math:`(h_O, w_O)`.
            Default value is ``None`` and the outsize is estimated by
            input size, stride and pad.
        dilate (:class:`int` or pair of :class:`int` s):
            Dilation factor of filter applications.
            ``dilate=d`` and ``dilate=(d, d)`` are equivalent.
        groups (:class:`int`):
            The number of groups to use grouped deconvolution.
            The default is one, where grouped deconvolution is not used.

    Returns:
        ~chainer.Variable:
            Output variable of shape :math:`(n, c_O, h_O, w_O)`.

    .. admonition:: Example

        >>> n = 10
        >>> c_i, c_o = 1, 3
        >>> h_i, w_i = 5, 10
        >>> h_k, w_k = 10, 10
        >>> h_p, w_p = 5, 5
        >>> x = np.random.uniform(0, 1, (n, c_i, h_i, w_i)).astype(np.float32)
        >>> x.shape
        (10, 1, 5, 10)
        >>> W = np.random.uniform(0, 1, (c_i, c_o, h_k, w_k)).\
astype(np.float32)
        >>> W.shape
        (1, 3, 10, 10)
        >>> b = np.random.uniform(0, 1, c_o).astype(np.float32)
        >>> b.shape
        (3,)
        >>> s_y, s_x = 5, 5
        >>> y = F.deconvolution_2d(x, W, b, stride=(s_y, s_x), pad=(h_p, w_p))
        >>> y.shape
        (10, 3, 20, 45)
        >>> h_o = s_y * (h_i - 1) + h_k - 2 * h_p
        >>> w_o = s_x * (w_i - 1) + w_k - 2 * w_p
        >>> y.shape == (n, c_o, h_o, w_o)
        True


    """  # NOQA
    argument.check_unexpected_kwargs(
        kwargs, deterministic="deterministic argument is not "
        "supported anymore. "
        "Use chainer.using_config('cudnn_deterministic', value) "
        "context where value is either `True` or `False`.")
    dilate, groups, d_layout, w_layout = argument.parse_kwargs(
        kwargs, ('dilate', 1), ('groups', 1), ('d_layout', 'NHWC'), ('w_layout', 'NHWC'))

    func = NhwcDeconvolution2DFunction(stride, pad, outsize, dilate=dilate,
                                       groups=groups, d_layout=d_layout, w_layout=w_layout)
    if b is None:
        args = x, W
    else:
        args = x, W, b
    y, = func.apply(args)
    return y
