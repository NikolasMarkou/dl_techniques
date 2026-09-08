"""``ComplexDropout`` -- drops whole complex units, not their components.

The point of this layer is the single shared mask. A plain ``Dropout`` applied
to the real and imaginary parts separately would draw two independent masks and
routinely keep one component of a unit while zeroing the other, which rotates
the value rather than removing it. Here one real mask is drawn over
``ones_like(real(inputs))`` and multiplied into the complex tensor, so a dropped
unit vanishes entirely and a kept one keeps its phase.

The mask comes from a real ``keras.layers.Dropout`` sub-layer, created in
``__init__`` with an explicit ``name="dropout"`` and built in ``build()``: this
is the one layer of the package that owns a sub-layer, and therefore the one
that needs a ``build()`` at all (the other three weightless layers get none --
``decisions.md`` D-004). The explicit name matters because two instances in one
process would otherwise disagree, the second auto-naming itself ``dropout_1``.
Delegating to ``Dropout`` also means the ``training`` flag and the ``1/(1-rate)``
scaling are handled by Keras rather than reimplemented.

The forward path uses raw ``tf.math.real`` because ``keras.ops`` exposes no
complex-tensor accessor, so this layer is TensorFlow-backend-only.
"""

import keras
import tensorflow as tf
from typing import Optional, Tuple, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.complex.complex_dropout")
class ComplexDropout(keras.layers.Layer):
    """Complex-valued dropout layer.

    Generates one real-valued dropout mask and applies it to both the real
    and imaginary parts, so entire complex units drop together rather than
    their components dropping independently.

    Architecture:

    .. code-block:: text

        Input [..., D] (complex)
                    │
                    ▼ (training only)
        ┌───────────────────────────┐
        │ mask = Dropout(ones_like) │
        │ output = input * mask     │
        └─────────────┬─────────────┘
                       ▼
        Output [..., D] (complex)

    :param rate: Fraction of the input units to drop, in ``[0, 1)``.
    :type rate: float

    Example:

    .. code-block:: python

        inputs = keras.Input(shape=(256,), dtype=tf.complex64)
        x = ComplexDense(128)(inputs)
        x = ComplexReLU()(x)
        x = ComplexDropout(0.3)(x)
        outputs = ComplexDense(64)(x)
        model = keras.Model(inputs, outputs)
    """

    def __init__(self, rate: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        if not 0 <= rate < 1:
            raise ValueError(f"rate must be in the interval [0, 1), got {rate}")

        self.rate = rate
        self.dropout_layer = keras.layers.Dropout(self.rate, name="dropout")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the inner dropout sub-layer.

        The mask is drawn over ``tf.ones_like(tf.math.real(inputs))``, which has
        the input's own shape, so the sub-layer is built against ``input_shape``
        unchanged.

        :param input_shape: Shape of the complex-valued input tensor.
        """
        self.dropout_layer.build(input_shape)
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply complex dropout.

        :param inputs: Complex-valued input tensor.
        :param training: Boolean indicating whether in training mode.

        :return: Complex-valued output tensor after dropout.
        """
        # Generate a real-valued mask by applying dropout to a tensor of ones.
        # The internal Dropout layer handles the `training` flag and scaling.
        mask = self.dropout_layer(
            tf.ones_like(tf.math.real(inputs)),
            training=training
        )

        # The mask is real-valued. Multiplying it with the complex input
        # correctly scales both the real and imaginary parts simultaneously.
        return inputs * tf.cast(mask, dtype=inputs.dtype)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape (same as input for dropout).

        :param input_shape: Input tensor shape.

        :return: Output tensor shape (identical to input).
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        :return: Configuration dictionary.
        """
        config = super().get_config()
        config.update({
            'rate': self.rate,
        })
        return config

# ---------------------------------------------------------------------
