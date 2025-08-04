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
- Backend-agnostic using keras.ops

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
mask = keras.ops.cast(features > threshold, "float32")
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
from keras import layers, ops
from typing import Tuple, List, Dict, Any, Optional, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
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
        name: Optional name for the layer.
        **kwargs: Additional layer keyword arguments.

    Example:
        ```python
        signal = keras.layers.Input(shape=(28, 28, 1))
        mask = keras.layers.Input(shape=(28, 28, 1))
        masked = SelectiveGradientMask()([signal, mask])

        model = keras.Model(inputs=[signal, mask], outputs=masked)
        ```
    """

    def __init__(
        self,
        name: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """Initialize the SelectiveGradientMask layer.

        Args:
            name: Optional name for the layer.
            **kwargs: Additional layer keyword arguments.
        """
        super().__init__(name=name, **kwargs)

    def build(
        self,
        input_shape: Union[List[Tuple[Optional[int], ...]], Tuple[Optional[int], ...]]
    ) -> None:
        """Build the layer and validate input shapes.

        Args:
            input_shape: List containing shapes of [signal, mask] or tuple for single input.

        Raises:
            ValueError: If input_shape is not a list of exactly 2 shapes.
            ValueError: If signal and mask shapes don't match.
        """
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError(
                "SelectiveGradientMask expects a list of 2 input shapes: "
                "[signal_shape, mask_shape]. "
                f"Got: {type(input_shape)} with length {len(input_shape) if isinstance(input_shape, list) else 'N/A'}"
            )

        signal_shape, mask_shape = input_shape

        if signal_shape != mask_shape:
            raise ValueError(
                f"Signal shape {signal_shape} must match mask shape {mask_shape}. "
                "Both inputs must have identical dimensions."
            )

        logger.info(f"Built SelectiveGradientMask with input shape: {signal_shape}")
        super().build(input_shape)

    def call(
        self,
        inputs: List[keras.KerasTensor],
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass of the layer.

        During training, creates two paths:
        1. Stopped gradient path: signal * mask (gradients blocked)
        2. Normal gradient path: signal * (1 - mask) (gradients flow)

        During inference, passes the signal through unchanged.

        Args:
            inputs: List containing [signal, mask] tensors.
            training: Whether in training mode. If None, uses the global training phase.

        Returns:
            Processed tensor with selective gradient stopping applied during training.

        Raises:
            ValueError: If inputs is not a list of exactly 2 tensors.
        """
        if not isinstance(inputs, list) or len(inputs) != 2:
            raise ValueError(
                "SelectiveGradientMask expects [signal, mask] tensor inputs. "
                f"Got: {type(inputs)} with length {len(inputs) if isinstance(inputs, list) else 'N/A'}"
            )

        signal, mask = inputs

        # Validate mask values during training
        if training:
            # Check if mask contains values other than 0 and 1
            mask_min = ops.min(mask)
            mask_max = ops.max(mask)

            # Use ops.logical_or to check if mask has invalid values
            has_invalid_values = ops.logical_or(
                ops.not_equal(mask_min, ops.cast(mask_min, mask.dtype)),
                ops.not_equal(mask_max, ops.cast(mask_max, mask.dtype))
            )

            # Check if values are outside [0, 1] range
            out_of_range = ops.logical_or(
                ops.less(mask_min, 0.0),
                ops.greater(mask_max, 1.0)
            )

            if has_invalid_values or out_of_range:
                logger.warning(
                    "SelectiveGradientMask: Mask contains values outside [0, 1] range. "
                    f"Range: [{mask_min}, {mask_max}]. Consider using binary mask (0, 1)."
                )

            # Create stopped gradient path for masked regions (mask = 1)
            stopped_path = ops.multiply(ops.stop_gradient(signal), mask)

            # Create normal gradient path for unmasked regions (mask = 0)
            normal_path = ops.multiply(signal, ops.subtract(1.0, mask))

            # Combine both paths
            return ops.add(stopped_path, normal_path)

        # Inference mode: Pass signal through unchanged
        return signal

    def compute_output_shape(
        self,
        input_shape: List[Tuple[Optional[int], ...]]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        Args:
            input_shape: List of input shapes [signal_shape, mask_shape].

        Returns:
            Output shape (same as signal shape).

        Raises:
            ValueError: If input_shape is invalid.
        """
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError(
                "Expected list of 2 input shapes: [signal_shape, mask_shape]. "
                f"Got: {type(input_shape)}"
            )

        # Return the signal shape (first input shape)
        return input_shape[0]

    def get_config(self) -> Dict[str, Any]:
        """Get the layer configuration.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get build configuration for serialization.

        Returns:
            Dictionary containing build configuration.
        """
        return {
            "input_shape": getattr(self, "_build_input_shape", None)
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build the layer from configuration.

        Args:
            config: Dictionary containing build configuration.
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

# ---------------------------------------------------------------------
