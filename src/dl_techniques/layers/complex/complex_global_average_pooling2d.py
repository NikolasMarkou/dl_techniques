"""``ComplexGlobalAveragePooling2D`` -- one complex number per feature map.

Reduces the two spatial axes of a ``[B, H, W, C]`` complex tensor to a
``[B, C]`` (or ``[B, 1, 1, C]`` under ``keepdims``) summary by averaging the
real and imaginary parts independently -- valid because averaging is linear,
the same argument that lets
:class:`dl_techniques.layers.complex.complex_average_pooling2d.ComplexAveragePooling2D`
pool componentwise. This is the whole-map limit of that layer, and the usual
bridge from a complex convolutional trunk to a complex dense head.

The reduction is over axes ``[1, 2]`` specifically: averaging over the channel
axis as well would erase the per-filter response the head reads. The layer owns
no weights and no sub-layers, so it extends ``keras.layers.Layer`` directly
rather than :class:`dl_techniques.layers.complex.base.ComplexLayer`, and defines
no ``build()`` -- see ``decisions.md`` D-004.

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

@register_dl_technique("dl_techniques.layers.complex.complex_global_average_pooling2d")
class ComplexGlobalAveragePooling2D(keras.layers.Layer):
    """Complex-valued global 2D average pooling layer.

    Reduces each complex feature map to a single complex number by averaging
    over the spatial axes, independently on the real and imaginary parts.

    Architecture:

    .. code-block:: text

        Input [B, H, W, C] (complex)
                    │
                    ▼
        ┌───────────────────────────┐
        │ split: real, imag         │
        │ mean over axes [1, 2]     │
        └─────────────┬─────────────┘
                       ▼
        Output [B, C] (complex), or [B, 1, 1, C] if keepdims

    :param keepdims: If ``False`` (default), the output has shape
        ``(batch, channels)``. If ``True``, ``(batch, 1, 1, channels)``.
    :type keepdims: bool
    """
    def __init__(self, keepdims: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.keepdims = keepdims

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply complex global average pooling.

        :param inputs: Complex-valued input tensor.
        :param training: Boolean indicating whether in training mode (unused).

        :return: Complex-valued output tensor after pooling.
        """
        # Split into real and imaginary components
        inputs_real = tf.math.real(inputs)
        inputs_imag = tf.math.imag(inputs)

        # Apply global average pooling to each component
        # The spatial axes for a 4D tensor are 1 and 2.
        pooled_real = keras.ops.mean(inputs_real, axis=[1, 2], keepdims=self.keepdims)
        pooled_imag = keras.ops.mean(inputs_imag, axis=[1, 2], keepdims=self.keepdims)

        # Recombine into a complex tensor
        return tf.complex(pooled_real, pooled_imag)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the global pooling layer.

        :param input_shape: Input tensor shape.

        :return: Output tensor shape.
        """
        if len(input_shape) != 4:
            raise ValueError(f"ComplexGlobalAveragePooling2D requires 4D input, got {len(input_shape)}D")

        if self.keepdims:
            return (input_shape[0], 1, 1, input_shape[3])
        else:
            return (input_shape[0], input_shape[3])

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        :return: Configuration dictionary containing all constructor parameters.
        """
        config = super().get_config()
        config.update({
            'keepdims': self.keepdims,
        })
        return config


# ---------------------------------------------------------------------
