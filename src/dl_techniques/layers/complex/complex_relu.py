"""``ComplexReLU`` -- ReLU applied to the real and imaginary parts separately.

The cheapest complex nonlinearity in the package, and the one whose semantics
are worth stating explicitly: it is **not** magnitude-based. For ``z = x + iy``
the output is ``max(0, x) + i*max(0, y)``, so the phase of a value is not
preserved -- a point in the third quadrant collapses to the origin. That is the
"CReLU" of Trabelsi et al. 2018 (cited in
:mod:`dl_techniques.layers.complex.base`), chosen over ``modReLU`` because it
needs no learned bias and no polar conversion.

The layer owns no weights and no sub-layers, so it extends ``keras.layers.Layer``
directly rather than :class:`dl_techniques.layers.complex.base.ComplexLayer`,
and it defines no ``build()`` -- see ``decisions.md`` D-004.

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

from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.complex.complex_relu")
class ComplexReLU(keras.layers.Layer):
    """Complex ReLU activation.

    Applies ReLU to the real and imaginary parts independently: for
    ``z = x + iy``, ``output = max(0, x) + i*max(0, y)``.

    Architecture:

    .. code-block:: text

        Input [..., D] (complex)
                    │
                    ▼
        ┌───────────────────────────┐
        │ split: real, imag         │
        │ relu each independently   │
        └─────────────┬─────────────┘
                       ▼
        Output [..., D] (complex)

    Example:

    .. code-block:: python

        inputs = keras.Input(shape=(128,), dtype=tf.complex64)
        x = ComplexDense(64)(inputs)
        x = ComplexReLU()(x)
        outputs = ComplexDense(32)(x)
        model = keras.Model(inputs, outputs)
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply complex ReLU activation.

        :param inputs: Complex-valued input tensor.
        :param training: Boolean indicating whether in training mode.

        :return: Complex-valued output tensor with ReLU applied to both components.
        """
        return tf.complex(
            keras.ops.relu(tf.math.real(inputs)),
            keras.ops.relu(tf.math.imag(inputs))
        )

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape (same as input for activation).

        :param input_shape: Input tensor shape.

        :return: Output tensor shape (identical to input).
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        :return: Configuration dictionary.
        """
        return super().get_config()


# ---------------------------------------------------------------------
