"""
Selective Gradient Masking Layer
==============================

Implementation of a custom layer that allows selective gradient flow control
through binary masking. This is particularly useful in scenarios where you want
to prevent gradients from flowing through certain parts of your network during
training.

Key Features:
------------
- Binary mask-based gradient control
- Preserves forward pass behavior
- Training/inference mode awareness
- Compatible with any tensor shape
- Proper gradient handling

Applications:
------------
- Adversarial training
- Feature disentanglement
- Partial network freezing
- Controlled feature learning
- Gradient manipulation studies

Architecture:
------------
The layer implements a dual-path mechanism during training:
1. Stopped gradient path: For masked regions (mask = 1)
2. Normal gradient path: For unmasked regions (mask = 0)

The computation flow is:
input -> [stop_gradient * mask, input * (1-mask)] -> sum -> output

Usage Examples:
-------------
```python
# Basic usage
signal = keras.layers.Input(shape=(28, 28, 1))
mask = keras.layers.Input(shape=(28, 28, 1))
masked = SelectiveGradientMask()([signal, mask])

# In a model
model = keras.Model(
    inputs=[signal, mask],
    outputs=masked
)

# With dynamic mask generation
mask = tf.cast(features > threshold, tf.float32)
masked = SelectiveGradientMask()([features, mask])
```

Implementation Notes:
------------------
- Mask should be binary (0 or 1)
- Input tensors must have matching shapes
- Gradients are completely blocked where mask = 1
- Forward pass is unchanged during inference
"""

import keras
import tensorflow as tf
from keras import layers
from typing import Tuple, List, Dict, Any, Optional, Union

@keras.utils.register_keras_serializable()
class SelectiveGradientMask(layers.Layer):
    """A layer that selectively stops gradients based on a binary mask.

    This layer takes two inputs of the same shape:
    1. signal: The primary input tensor to process
    2. mask: A binary mask (0s and 1s) where:
       - 1 indicates positions where gradients should be stopped
       - 0 indicates positions where gradients should flow normally

    The layer preserves the forward pass behavior while controlling gradient
    flow during backpropagation.

    Args:
        name: Optional name for the layer
        **kwargs: Additional layer keyword arguments

    Example:
        ```python
        signal = layers.Input(shape=(28, 28, 1))
        mask = layers.Input(shape=(28, 28, 1))
        masked = SelectiveGradientMask()([signal, mask])
        ```
    """

    def __init__(
            self,
            name: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(name=name, **kwargs)

    def build(
            self,
            input_shape: List[Tuple[Optional[int], ...]]
    ) -> None:
        """Validates and builds the layer.

        Args:
            input_shape: List containing shapes of [signal, mask]

        Raises:
            ValueError: If input_shape is not a list of exactly 2 shapes
            ValueError: If signal and mask shapes don't match
        """
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError(
                "Layer expects a list of 2 input shapes: [signal_shape, mask_shape]"
            )

        if input_shape[0] != input_shape[1]:
            raise ValueError(
                f"Signal shape {input_shape[0]} must match mask shape {input_shape[1]}"
            )

        super().build(input_shape)

    def call(
            self,
            inputs: List[tf.Tensor],
            training: Optional[bool] = None
    ) -> tf.Tensor:
        """Forward pass of the layer.

        During training, creates two paths:
        1. Stopped gradient path: signal * mask
        2. Normal gradient path: signal * (1 - mask)

        During inference, passes the signal through unchanged.

        Args:
            inputs: List containing [signal, mask] tensors
            training: Whether in training mode

        Returns:
            Processed tensor with selective gradient stopping

        Raises:
            ValueError: If inputs is not a list of exactly 2 tensors
        """
        if not isinstance(inputs, list) or len(inputs) != 2:
            raise ValueError("Layer expects [signal, mask] tensor inputs")

        signal, mask = inputs

        if tf.reduce_any(tf.logical_and(mask != 0, mask != 1)):
            tf.print("\nWarning: Mask contains values other than 0 and 1")

        if training:
            return (tf.stop_gradient(signal) * mask +
                    signal * (1.0 - mask))
        return signal

    def compute_output_shape(
            self,
            input_shape: List[Tuple[Optional[int], ...]]
    ) -> Tuple[Optional[int], ...]:
        """Computes the output shape.

        Args:
            input_shape: List of input shapes [signal_shape, mask_shape]

        Returns:
            Output shape (same as signal shape)

        Raises:
            ValueError: If input_shape is invalid
        """
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError(
                "Expected list of 2 input shapes: [signal_shape, mask_shape]"
            )
        return input_shape[0]

    def get_config(self) -> Dict[str, Any]:
        """Returns the layer configuration.

        Returns:
            Layer configuration dictionary
        """
        return super().get_config()
