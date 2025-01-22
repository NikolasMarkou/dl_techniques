"""
Selective Gradient Masking Implementation Module.

This module provides a custom Keras layer that enables selective gradient masking
during backpropagation. It includes type hints and comprehensive documentation.

"""

import keras
import tensorflow as tf
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------

@dataclass
class LayerConfig:
    """Configuration dataclass for SelectiveGradientMask layer."""
    name: Optional[str] = None
    trainable: bool = True
    dtype: Optional[str] = None

# ---------------------------------------------------------------------


class SelectiveGradientMask(keras.layers.Layer):
    """
    Custom Keras layer implementing selective gradient masking during backpropagation.

    This layer allows fine-grained control over gradient flow by selectively stopping
    gradients at specified positions based on a binary mask. The layer maintains
    the forward pass signal while controlling backpropagation paths.

    Attributes:
        config: LayerConfig instance containing layer configuration

    Example:
        ```python
        # Create input tensors
        signal = keras.Input(shape=(28, 28, 1))
        mask = keras.Input(shape=(28, 28, 1))

        # Apply selective gradient masking
        masked_output = SelectiveGradientMask()([signal, mask])
        ```
    """

    def __init__(
            self,
            name: Optional[str] = None,
            trainable: bool = True,
            dtype: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        """
        Initialize the SelectiveGradientMask layer.

        Args:
            name: Optional name for the layer
            trainable: Boolean indicating if layer has trainable weights
            dtype: Optional datatype for layer computations
            **kwargs: Additional keyword arguments passed to parent class
        """
        self.config = LayerConfig(name=name, trainable=trainable, dtype=dtype)
        super().__init__(
            name=self.config.name,
            trainable=self.config.trainable,
            dtype=self.config.dtype,
            **kwargs
        )

    def build(self, input_shape: List[Tuple[Optional[int], ...]]) -> None:
        """
        Build the layer by validating input shapes.

        Args:
            input_shape: List containing shapes of [signal, mask] tensors

        Raises:
            ValueError: If input_shape is not a list of exactly 2 tensor shapes
            ValueError: If signal and mask shapes don't match
        """
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError(
                f"Expected list of 2 shapes, got {input_shape}"
            )

        signal_shape, mask_shape = input_shape
        if signal_shape != mask_shape:
            raise ValueError(
                f"Signal shape {signal_shape} must match mask shape {mask_shape}"
            )

        super().build(input_shape)

    def call(
            self,
            inputs: List[tf.Tensor],
            training: Optional[bool] = None
    ) -> tf.Tensor:
        """
        Forward pass implementation with selective gradient masking.

        Args:
            inputs: List containing [signal, mask] tensors where:
                   - signal: Input tensor to process (any shape)
                   - mask: Binary mask tensor (same shape as signal)
            training: Boolean indicating training phase

        Returns:
            tf.Tensor with selective gradient masking applied

        Raises:
            ValueError: If inputs is not a list of exactly 2 tensors
            ValueError: If mask contains non-binary values
        """
        self._validate_inputs(inputs)
        signal, mask = inputs

        # During inference, return unmodified signal
        if not training:
            return signal

        # Validate mask contains only binary values
        if not tf.reduce_all(tf.logical_or(
                tf.equal(mask, 0.0),
                tf.equal(mask, 1.0)
        )):
            raise ValueError("Mask must contain only binary values (0 or 1)")

        # Create parallel paths for gradient flow
        stopped_path = tf.stop_gradient(signal) * mask
        normal_path = signal * (1.0 - mask)

        return stopped_path + normal_path

    def _validate_inputs(self, inputs: List[tf.Tensor]) -> None:
        """
        Validate input tensors.

        Args:
            inputs: List of input tensors to validate

        Raises:
            ValueError: If inputs format is invalid
        """
        if not isinstance(inputs, list) or len(inputs) != 2:
            raise ValueError(
                f"Expected list of 2 tensors, got {type(inputs)}"
            )

    def compute_output_shape(
            self,
            input_shape: List[Tuple[Optional[int], ...]]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute output tensor shape.

        Args:
            input_shape: List of input tensor shapes

        Returns:
            Output tensor shape (same as input signal shape)
        """
        return input_shape[0]

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration.

        Returns:
            Dictionary containing layer configuration
        """
        return {
            "name": self.config.name,
            "trainable": self.config.trainable,
            "dtype": self.config.dtype
        }

# ---------------------------------------------------------------------


def create_model(
        input_shape: Tuple[int, ...] = (28, 28, 1),
        num_classes: int = 10,
        filters: int = 32,
        kernel_size: int = 3,
        pool_size: int = 2
) -> keras.Model:
    """
    Create a CNN model with selective gradient masking.

    Args:
        input_shape: Shape of input tensors (excluding batch dimension)
        num_classes: Number of output classes
        filters: Number of convolutional filters
        kernel_size: Size of convolutional kernel
        pool_size: Size of pooling window

    Returns:
        Compiled Keras model
    """
    # Input layers
    signal_input = keras.layers.Input(shape=input_shape, name="signal")
    mask_input = keras.layers.Input(shape=input_shape, name="mask")

    # Apply selective gradient masking
    masked = SelectiveGradientMask(name="gradient_mask")(
        [signal_input, mask_input]
    )

    # CNN architecture
    x = keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        activation="relu",
        padding="same",
        name="conv1"
    )(masked)

    x = keras.layers.MaxPooling2D(
        pool_size=pool_size,
        name="pool1"
    )(x)

    x = keras.layers.BatchNormalization(name="bn1")(x)
    x = keras.layers.Dropout(0.25, name="dropout1")(x)
    x = keras.layers.Flatten(name="flatten")(x)

    outputs = keras.layers.Dense(
        units=num_classes,
        activation="softmax",
        name="predictions"
    )(x)

    model = keras.Model(
        inputs=[signal_input, mask_input],
        outputs=outputs,
        name="selective_gradient_cnn"
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

# ---------------------------------------------------------------------

