import numpy
import six

from chainer import configuration
from chainer import cuda
from chainer import function
from chainer.utils import type_check

_debug = False

def _cu_conv_sum(y, x, n):
    # Convolutional sum
    # TODO(beam2d): Use scan computation
    rdim = x.size // (x.shape[0] * x.shape[1])
    cuda.elementwise(
        'raw T x, int32 rdim, int32 N, int32 n_', 'raw T y',
        '''
          int half_n = n_ / 2;
          int offset = i / rdim * N * rdim + i % rdim;

          float sum_part = 0;
          for (int j = 0; j < N + half_n; ++j) {
            if (j < N) {
              sum_part += x[offset + j * rdim];
            }
            if (j >= n_) {
              sum_part -= x[offset + (j - n_) * rdim];
            }
            if (j >= half_n) {
              y[offset + (j - half_n) * rdim] = sum_part;
            }
          }
        ''', 'lrn_conv_sum')(x, rdim, x.shape[1], n, y,
                             size=x.shape[0] * rdim)


def _cu_conv_square_sum(y, x, n):
    # Convolutional square sum
    # TODO(beam2d): Use scan computation
    rdim = x.size // (x.shape[0] * x.shape[1])
    cuda.elementwise(
        'raw T x, int32 rdim, int32 N, int32 n_', 'raw T y',
        '''
          int half_n = n_ / 2;
          int offset = i / rdim * N * rdim + i % rdim;

          float sum_part = 0;
          for (int j = 0; j < N + half_n; ++j) {
            if (j < N) {
              T xj = x[offset + j * rdim];
              sum_part += xj * xj;
            }
            if (j >= n_) {
              T xj = x[offset + (j - n_) * rdim];
              sum_part -= xj * xj;
            }
            if (j >= half_n) {
              y[offset + (j - half_n) * rdim] = sum_part;
            }
          }
        ''', 'lrn_conv_square_sum')(x, rdim, x.shape[1], n, y,
                                    size=x.shape[0] * rdim)


class LocalResponseNormalization(function.Function):

    """Cross-channel normalization function used in AlexNet."""

    def __init__(self, n=5, k=2, alpha=1e-4, beta=.75):
        self.n = n
        self.k = k
        self.alpha = alpha
        self.beta = beta

        self.use_blocking = False
        execution_policy = getattr(configuration.config, 'execution_policy', None)
        if execution_policy is 'memory_usage':
            self.use_blocking = True
        if _debug:
            print('# LRN:81, use_blocking: {}'.format(self.use_blocking))

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types

        type_check.expect(
            x_type.dtype.kind == 'f',
            x_type.ndim >= 2,
        )

    def forward_cpu(self, x):
        half_n = self.n // 2
        x2 = numpy.square(x[0])
        sum_part = x2.copy()
        for i in six.moves.range(1, half_n + 1):
            sum_part[:, i:] += x2[:, :-i]
            sum_part[:, :-i] += x2[:, i:]
        self.unit_scale = self.k + self.alpha * sum_part
        self.scale = self.unit_scale ** -self.beta
        self.y = x[0] * self.scale
        return self.y,

    def backward_cpu(self, x, gy):
        half_n = self.n // 2
        summand = self.y * gy[0] / self.unit_scale
        sum_part = summand.copy()
        for i in six.moves.range(1, half_n + 1):
            sum_part[:, i:] += summand[:, :-i]
            sum_part[:, :-i] += summand[:, i:]

        gx = gy[0] * self.scale - 2 * self.alpha * self.beta * x[0] * sum_part
        return gx,

    def forward_gpu(self, x):
        y = cuda.cupy.square(x[0])  # temporary
        scale = cuda.cupy.empty_like(x[0])
        _cu_conv_sum(scale, y, self.n)
        cuda.elementwise(
            'T x, T k, T alpha, T beta',
            'T y, T scale',
            '''scale = k + alpha * scale;
               y = x * pow(scale, -beta);''',
            'lrn_fwd')(x[0], self.k, self.alpha, self.beta,
                       y, scale)

        if x[0].ndim != 4:
            self.use_blocking = False
        if self.use_blocking:
            self.gx = scale

        self.retain_outputs((0,))
        return y,

    def backward_gpu(self, x, gy):
        y = self.output_data[0]
        if self.use_blocking is False:
            gx = self._backward_gpu_core(x[0], gy[0], y)
            return gx,

        gx = getattr(self, 'gx', None)
        if gx is None:
            gx = cuda.cupy.empty_like(x[0])
        else:
            self.gx = None

        _step_b, _step_h, _step_w = [32, 64, 64]
        _dim_b, _dim_c, _dim_h, _dim_w = x[0].shape
        for _start_b in range(0, _dim_b, _step_b):
            _end_b = min(_start_b + _step_b, _dim_b)
            _size_b = _end_b - _start_b

            for _start_h in range(0, _dim_h, _step_h):
                _end_h = min(_start_h + _step_h, _dim_h)
                _size_h = _end_h - _start_h

                for _start_w in range(0, _dim_w, _step_w):
                    _end_w = min(_start_w + _step_w, _dim_w)
                    _size_w = _end_w - _start_w

                    if _debug:
                        print('# LRN:162, _b:{}, _h:{}, _w:{}'.format([_start_b, _end_b], [_start_h, _end_h], [_start_w, _end_w]))

                    sub_x = x[0][_start_b:_end_b, :, _start_h:_end_h, _start_w:_end_w]
                    sub_gy = gy[0][_start_b:_end_b, :, _start_h:_end_h, _start_w:_end_w]
                    sub_y = y[_start_b:_end_b, :, _start_h:_end_h, _start_w:_end_w]

                    sub_gx = self._backward_gpu_core(sub_x, sub_gy, sub_y)
                    del sub_x
                    del sub_gy
                    del sub_y

                    gx[_start_b:_end_b, :, _start_h:_end_h, _start_w:_end_w] = sub_gx
                    del sub_gx

        return gx,

    def _backward_gpu_core(self, x, gy, y):
        scale = cuda.cupy.empty_like(x)
        _cu_conv_square_sum(scale, x, self.n)
        cuda.elementwise(
            'T k, T alpha, T beta',
            'T scale',
            '''scale = k + alpha * scale;''',
            'lrn_fwd_part')(self.k, self.alpha, self.beta,
                            scale)

        summand = cuda.elementwise(
            'T scale, T y, T gy', 'T summand',
            'summand = y * gy / scale',
            'lrn_bwd_summand')(scale, y, gy)
        gx = cuda.cupy.empty_like(x)
        _cu_conv_sum(gx, summand, self.n)
        cuda.elementwise(
            ' T x, T gy, T scale, T beta, T coeff', 'T gx',
            'gx = pow(scale, -beta) * gy - coeff * x * gx',
            'lrn_bwd')(x, gy, scale,
                       self.beta, 2 * self.alpha * self.beta, gx)
        del scale
        del summand

        return gx


def local_response_normalization(x, n=5, k=2, alpha=1e-4, beta=.75):
    """Local response normalization across neighboring channels.

    This function implements normalization across channels. Let :math:`x` an
    input image with :math:`N` channels. Then, this function computes an output
    image :math:`y` by following formula:

    .. math::
       y_i = {x_i \\over \\left( k + \\
              \\alpha \\sum_{j=\\max{1, i - n/2}}^{\\min{N, i + n/2}} \\
              x_j^2 \\right)^\\beta}.

    Args:
        x (Variable): Input variable.
        n (int): Normalization window width.
        k (float): Smoothing parameter.
        alpha (float): Normalizer scaling parameter.
        beta (float): Normalizer power parameter.

    Returns:
        Variable: Output variable.

    See: Section 3.3 of `ImageNet Classification with Deep Convolutional \\
    Neural Networks <http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf>`_

    """
    return LocalResponseNormalization(n, k, alpha, beta)(x)
