"""``ComplexAveragePooling2D`` -- windowed spatial average over complex maps.

Averaging is linear, so pooling the real and imaginary parts independently is
exactly pooling the complex tensor: no polar conversion and no magnitude
heuristic is involved. The layer owns no weights and no sub-layers, so it
extends ``keras.layers.Layer`` directly rather than
:class:`dl_techniques.layers.complex.base.ComplexLayer`, and defines no
``build()`` -- see ``decisions.md`` D-004. Its whole-map counterpart is
:class:`dl_techniques.layers.complex.complex_global_average_pooling2d.ComplexGlobalAveragePooling2D`.

``pool_size``/``strides`` must accept any sequence, not only a ``tuple``,
because a ``.keras`` archive hands them back as JSON lists; see the DECISION
anchor in ``__init__`` and ``decisions.md`` D-005. ``compute_output_shape``
mirrors what ``keras.ops.average_pool`` computes on each branch -- ``'SAME'`` is
``ceil(dim / stride)``, ``'VALID'`` is ``(dim - pool + stride) // stride`` --
and both branches are pinned against the real forward pass by test.

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

from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.complex.complex_average_pooling2d")
class ComplexAveragePooling2D(keras.layers.Layer):
    """Complex-valued 2D average pooling layer.

    Applies standard average pooling independently to the real and
    imaginary components: for ``z = x + iy``, ``output = AvgPool(x) + i*AvgPool(y)``.

    Architecture:

    .. code-block:: text

        Input [B, H, W, C] (complex)
                    │
                    ▼
        ┌────────────────────────────┐
        │ split: real, imag          │
        │ avg-pool each independently│
        └─────────────┬──────────────┘
                       ▼
        Output [B, H', W', C] (complex)

    :param pool_size: Height and width of the pooling window. Defaults to
        ``(2, 2)``.
    :type pool_size: int or Tuple[int, int]
    :param strides: Stride of the pooling operation. Defaults to
        ``pool_size``.
    :type strides: int or Tuple[int, int], optional
    :param padding: ``'SAME'`` or ``'VALID'`` (case-insensitive). Defaults to
        ``'VALID'``.
    :type padding: str
    """

    def __init__(
            self,
            pool_size: Union[int, Tuple[int, int]] = (2, 2),
            strides: Optional[Union[int, Tuple[int, int]]] = None,
            padding: str = 'VALID',
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Store and validate configuration
        # DECISION plan-2026-09-08T070501-528ded1a/D-005: same round-trip closure
        # as ComplexConv2D -- a JSON config hands these back as LISTS, so a
        # tuple-only isinstance test re-wraps [2, 2] into ([2, 2], [2, 2]) and
        # the forward pass then raises "Expected int for argument 'ksize'".
        # See decisions.md D-005.
        self.pool_size = tuple(pool_size) if isinstance(pool_size, (list, tuple)) else (pool_size, pool_size)
        self.strides = strides if strides is not None else self.pool_size
        self.strides = tuple(self.strides) if isinstance(self.strides, (list, tuple)) else (self.strides, self.strides)
        self.padding = padding.upper()

        if self.padding not in ['SAME', 'VALID']:
            raise ValueError(f"padding must be 'SAME' or 'VALID', got {padding}")

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply complex average pooling.

        :param inputs: Complex-valued input tensor.
        :param training: Boolean indicating whether in training mode (unused).

        :return: Complex-valued output tensor after pooling.
        """
        # Split into real and imaginary components
        inputs_real = tf.math.real(inputs)
        inputs_imag = tf.math.imag(inputs)

        # Apply average pooling to each component
        # Note: keras.ops.average_pool expects lowercase padding
        padding_lower = self.padding.lower()

        pooled_real = keras.ops.average_pool(
            inputs_real,
            pool_size=self.pool_size,
            strides=self.strides,
            padding=padding_lower
        )

        pooled_imag = keras.ops.average_pool(
            inputs_imag,
            pool_size=self.pool_size,
            strides=self.strides,
            padding=padding_lower
        )

        # Recombine into a complex tensor
        return tf.complex(pooled_real, pooled_imag)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the pooling layer.

        :param input_shape: Input tensor shape.

        :return: Output tensor shape.
        """
        if len(input_shape) != 4:
            raise ValueError(f"ComplexAveragePooling2D requires 4D input, got {len(input_shape)}D")

        batch_size, height, width, channels = input_shape

        def _compute_dim(dim, pool, stride, padding):
            if dim is None:
                return None
            if padding == 'VALID':
                return (dim - pool + stride) // stride
            else:  # SAME
                return (dim + stride - 1) // stride

        output_height = _compute_dim(height, self.pool_size[0], self.strides[0], self.padding)
        output_width = _compute_dim(width, self.pool_size[1], self.strides[1], self.padding)

        return (batch_size, output_height, output_width, channels)

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        :return: Configuration dictionary containing all constructor parameters.
        """
        config = super().get_config()
        config.update({
            'pool_size': self.pool_size,
            'strides': self.strides,
            'padding': self.padding
        })
        return config

# ---------------------------------------------------------------------
