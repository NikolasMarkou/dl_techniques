"""``ComplexDense`` -- fully connected layer over complex activations.

The dense counterpart of
:class:`dl_techniques.layers.complex.complex_conv2d.ComplexConv2D`: the same
four-real-product expansion of complex multiplication, with ``matmul`` in place
of ``conv``. It is the second and last weight-owning layer of the package, so it
subclasses :class:`dl_techniques.layers.complex.base.ComplexLayer` to inherit
the ``complex64`` Rayleigh-magnitude / uniform-phase weight draw and the
``get_config``/``from_config`` pair; the four weightless layers extend
``keras.layers.Layer`` directly.

Note that ``kernel_initializer``, inherited from the base, does not choose the
weight distribution -- ``_init_complex_weights`` does. It is accepted and
serialized only so existing checkpoints keep loading; see the DECISION anchor in
``base.py`` and ``decisions.md`` D-002.

The forward path uses raw ``tf.complex`` / ``tf.math.real`` / ``tf.math.imag``
because ``keras.ops`` exposes no complex-tensor constructor, so this layer is
TensorFlow-backend-only.
"""

import keras
import tensorflow as tf
from typing import Optional, Tuple, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.complex.base import ComplexLayer
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.complex.complex_dense")
class ComplexDense(ComplexLayer):
    """Complex-valued fully connected layer.

    Computes complex matrix multiplication as four real matmuls, following
    the algebraic expansion of complex multiplication.

    Architecture:

    .. code-block:: text

        Input [..., D] (complex)
                    │
                    ▼
        ┌───────────────────────────┐
        │ split: real, imag         │
        └─────────────┬─────────────┘
                       ▼
        ┌────────────────────────────┐
        │ 4 real matmuls             │
        │  real_out = re@Wre - im@Wim│
        │  imag_out = re@Wim + im@Wre│
        └─────────────┬──────────────┘
                       ▼
        combine + bias, complex(real_out, imag_out)
                       │
                       ▼
        Output [..., units] (complex)

    For complex input ``I = I_r + iI_i`` and weights ``W = W_r + iW_i``:
    ``output = (I_r@W_r - I_i@W_i) + i(I_r@W_i + I_i@W_r)``

    :param units: Number of output units.
    :type units: int

    Example:

    .. code-block:: python

        dense = ComplexDense(
            units=64, kernel_regularizer=keras.regularizers.L2(0.01),
        )
        inputs = keras.Input(shape=(784,), dtype=tf.complex64)
        x = ComplexDense(256)(inputs)
        outputs = ComplexDense(10)(x)
        model = keras.Model(inputs, outputs)
    """

    def __init__(
        self,
        units: int,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if units <= 0:
            raise ValueError(f"units must be positive, got {units}")

        self.units = units

        # Initialize weight attributes (created in build)
        self.kernel = None
        self.bias = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create the layer's weights.

        :param input_shape: Shape of the input tensor.
        """
        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError("Last dimension of input must be defined")

        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_dim, self.units),
            initializer=self._init_complex_weights,
            regularizer=self.kernel_regularizer,
            trainable=True,
            dtype=tf.complex64
        )

        self.bias = self.add_weight(
            name='bias',
            shape=(self.units,),
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
        """
        Apply complex dense transformation.

        :param inputs: Complex-valued input tensor.
        :param training: Boolean indicating whether in training mode.

        :return: Complex-valued output tensor after transformation.
        """
        # Split computations for numerical stability
        real_output = keras.ops.matmul(
            tf.math.real(inputs), tf.math.real(self.kernel)
        ) - keras.ops.matmul(
            tf.math.imag(inputs), tf.math.imag(self.kernel)
        )

        imag_output = keras.ops.matmul(
            tf.math.real(inputs), tf.math.imag(self.kernel)
        ) + keras.ops.matmul(
            tf.math.imag(inputs), tf.math.real(self.kernel)
        )

        return tf.complex(
            real_output + tf.math.real(self.bias),
            imag_output + tf.math.imag(self.bias)
        )

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the dense layer.

        :param input_shape: Input tensor shape.

        :return: Output tensor shape.
        """
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        :return: Configuration dictionary containing all constructor parameters.
        """
        config = super().get_config()
        config.update({
            'units': self.units,
        })
        return config

# ---------------------------------------------------------------------
