"""``ComplexConv2D`` -- 2D convolution over complex feature maps.

The complex convolution is not a separate kernel type: it is four ordinary real
convolutions recombined as ``(I_r*K_r - I_i*K_i) + i(I_r*K_i + I_i*K_r)``. The
kernel itself, however, is a genuine ``complex64`` weight, drawn by
:class:`dl_techniques.layers.complex.base.ComplexLayer` (Rayleigh magnitude,
uniform phase) rather than by any ``keras.initializers.Initializer`` -- which is
why this layer subclasses that base instead of ``keras.layers.Layer`` as the
four weightless layers of the package do. Its sibling
:class:`dl_techniques.layers.complex.complex_dense.ComplexDense` is the same
algebra with ``matmul`` in place of ``conv``.

Two subtleties are load-bearing here and both are pinned by tests. ``'SAME'``
padding must report ``ceil(dim / stride)``, because that is what
``keras.ops.conv`` actually computes -- floor division disagreed with the real
forward pass in every cell where the stride does not divide the input evenly.
And ``kernel_size``/``strides`` must accept any sequence, not only a ``tuple``,
because a ``.keras`` archive hands them back as JSON lists; see the DECISION
anchor in ``__init__`` and ``decisions.md`` D-005.

The forward path uses raw ``tf.complex`` / ``tf.math.real`` / ``tf.math.imag``
because ``keras.ops`` exposes no complex-tensor constructor, so this layer is
TensorFlow-backend-only.
"""

import keras
import tensorflow as tf
from typing import Optional, Tuple, Union, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.complex.base import ComplexLayer
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.complex.complex_conv2d")
class ComplexConv2D(ComplexLayer):
    """Complex-valued 2D convolution layer.

    Computes complex convolution as four real convolutions, following the
    algebraic expansion of complex multiplication.

    Architecture:

    .. code-block:: text

        Input [B, H, W, C] (complex)
                    │
                    ▼
        ┌───────────────────────────┐
        │ split: real, imag         │
        └─────────────┬─────────────┘
                       ▼
        ┌───────────────────────────┐
        │ 4 real convolutions       │
        │  real_out = conv(re,Kre)  │
        │           - conv(im,Kim)  │
        │  imag_out = conv(re,Kim)  │
        │           + conv(im,Kre)  │
        └─────────────┬─────────────┘
                       ▼
        combine + bias, complex(real_out, imag_out)
                       │
                       ▼
        Output [B, H', W', filters] (complex)

    For complex input ``I = I_r + iI_i`` and kernel ``K = K_r + iK_i``:
    ``output = (I_r*K_r - I_i*K_i) + i(I_r*K_i + I_i*K_r)``

    :param filters: Number of output filters. Each filter produces one
        complex-valued output channel.
    :type filters: int
    :param kernel_size: Height and width of the convolution window.
    :type kernel_size: int or Tuple[int, int]
    :param strides: Stride of the convolution. Defaults to ``1``.
    :type strides: int or Tuple[int, int]
    :param padding: ``'SAME'`` or ``'VALID'`` (case-insensitive). Defaults to
        ``'SAME'``.
    :type padding: str

    Example:

    .. code-block:: python

        conv = ComplexConv2D(
            filters=32, kernel_size=(5, 5), strides=(2, 2),
            padding='VALID', kernel_regularizer=keras.regularizers.L2(0.01),
        )
    """

    def __init__(
        self,
        filters: int,
        kernel_size: Union[int, Tuple[int, int]],
        strides: Union[int, Tuple[int, int]] = 1,
        padding: str = 'SAME',
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if filters <= 0:
            raise ValueError(f"filters must be positive, got {filters}")

        # Store configuration
        self.filters = filters
        # DECISION plan-2026-09-08T070501-528ded1a/D-005: coerce ANY sequence,
        # not just a tuple. Do NOT write `x if isinstance(x, tuple) else (x, x)`
        # here: get_config emits a tuple, the .keras archive stores it as a JSON
        # LIST, and __init__ gets that list back -- so the tuple-only test is
        # False on reload and re-wraps [3, 3] into ([3, 3], [3, 3]), which made
        # this layer impossible to load from disk at all (ValueError: Invalid
        # dtype: TrackedList). See decisions.md D-005.
        self.kernel_size = tuple(kernel_size) if isinstance(kernel_size, (list, tuple)) else (kernel_size, kernel_size)
        self.strides = tuple(strides) if isinstance(strides, (list, tuple)) else (strides, strides)
        self.padding = padding.upper()

        if self.padding not in ['SAME', 'VALID']:
            raise ValueError(f"padding must be 'SAME' or 'VALID', got {padding}")

        # Initialize weight attributes (created in build)
        self.kernel = None
        self.bias = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the layer's weights.

        :param input_shape: Shape of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        """
        if len(input_shape) != 4:
            raise ValueError(f"ComplexConv2D requires 4D input, got {len(input_shape)}D")

        input_channels = input_shape[-1]
        if input_channels is None:
            raise ValueError("Last dimension of input must be defined")

        kernel_shape = (*self.kernel_size, input_channels, self.filters)

        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self._init_complex_weights,
            regularizer=self.kernel_regularizer,
            trainable=True,
            dtype=tf.complex64
        )

        self.bias = self.add_weight(
            name='bias',
            shape=(self.filters,),
            initializer='zeros',
            trainable=True,
            dtype=tf.complex64
        )

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply complex convolution to the input tensor.

        :param inputs: Complex-valued input tensor.
        :type inputs: keras.KerasTensor
        :param training: Whether the layer is in training mode.
        :type training: Optional[bool]
        :return: Complex-valued output tensor.
        :rtype: keras.KerasTensor
        """
        inputs_real = tf.math.real(inputs)
        inputs_imag = tf.math.imag(inputs)
        kernel_real = tf.math.real(self.kernel)
        kernel_imag = tf.math.imag(self.kernel)

        # Convert padding for keras.ops.conv (which expects lowercase)
        padding_lower = self.padding.lower()

        # Compute convolution components using keras.ops
        real_output = keras.ops.conv(
            inputs_real, kernel_real, strides=self.strides, padding=padding_lower
        ) - keras.ops.conv(
            inputs_imag, kernel_imag, strides=self.strides, padding=padding_lower
        )

        imag_output = keras.ops.conv(
            inputs_real, kernel_imag, strides=self.strides, padding=padding_lower
        ) + keras.ops.conv(
            inputs_imag, kernel_real, strides=self.strides, padding=padding_lower
        )

        # Combine real and imaginary parts
        output = tf.complex(
            real_output + tf.math.real(self.bias),
            imag_output + tf.math.imag(self.bias)
        )

        return output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the convolution.

        :param input_shape: Input tensor shape.

        :return: Output tensor shape.
        """
        batch_size = input_shape[0]

        if self.padding == 'SAME':
            # ceil(dim / stride), matching what keras.ops.conv(padding='same')
            # actually computes. The expression is `ComplexAveragePooling2D`'s,
            # verbatim: floor division disagreed with the real forward pass in
            # every cell where the stride does not divide the input evenly.
            if input_shape[1] is not None:
                output_height = (input_shape[1] + self.strides[0] - 1) // self.strides[0]
            else:
                output_height = None
            if input_shape[2] is not None:
                output_width = (input_shape[2] + self.strides[1] - 1) // self.strides[1]
            else:
                output_width = None
        else:  # VALID padding
            if input_shape[1] is not None:
                output_height = (input_shape[1] - self.kernel_size[0]) // self.strides[0] + 1
            else:
                output_height = None
            if input_shape[2] is not None:
                output_width = (input_shape[2] - self.kernel_size[1]) // self.strides[1] + 1
            else:
                output_width = None

        return (batch_size, output_height, output_width, self.filters)

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        :return: Configuration dictionary containing all constructor parameters.
        """
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
        })
        return config

# ---------------------------------------------------------------------
