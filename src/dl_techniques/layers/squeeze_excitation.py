"""
Squeeze-and-Excitation Layer
===========================

Implementation of the Squeeze-and-Excitation (SE) block as described in:
"Squeeze-and-Excitation Networks" (Hu et al., 2019)
https://arxiv.org/abs/1709.01507

Architecture:
------------
The SE block consists of:
1. Global average pooling (squeeze)
2. Two-layer bottleneck (excitation):
   - Dimensionality reduction (1x1 conv)
   - Configurable activation (default: ReLU)
   - Dimensionality restoration (1x1 conv)
   - Sigmoid activation
3. Channel-wise scaling

The computation flow is:
input -> global_pool -> conv_reduce -> activation -> conv_restore -> sigmoid -> scale -> output

Usage Example:
------------
```python
# Default configuration with ReLU activation
se_block = SqueezeExcitation(
    reduction_ratio=0.25,
    kernel_initializer='he_normal',
    kernel_regularizer=keras.regularizers.L2(l2=0.01)
)

# Custom activation
se_block = SqueezeExcitation(
    reduction_ratio=0.25,
    activation='swish',
    kernel_regularizer=keras.regularizers.L2(l2=0.01)
)
```
"""

import keras
from keras import layers, activations, ops
from typing import Dict, Optional, Tuple, Union, Callable, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class SqueezeExcitation(layers.Layer):
    """Squeeze-and-Excitation block for channel-wise feature recalibration.

    This layer implements the Squeeze-and-Excitation mechanism that adaptively
    recalibrates channel-wise feature responses by explicitly modeling
    interdependencies between channels.

    Args:
        reduction_ratio: Float between 0 and 1, determining the bottleneck width.
            Defaults to 0.25.
        kernel_initializer: Initializer for the kernel weights matrix.
            Defaults to "glorot_normal".
        kernel_regularizer: Optional regularizer function applied to kernel weights.
            Defaults to None.
        activation: Activation function to use after reduction convolution.
            Can be a string identifier or a callable. Defaults to "relu".
        use_bias: Boolean, whether the layer uses a bias vector.
            Defaults to False.
        **kwargs: Additional keyword arguments for the base Layer class.

    Raises:
        ValueError: If reduction_ratio is not in the range (0, 1].

    Input shape:
        4D tensor with shape: ``(batch_size, height, width, channels)``

    Output shape:
        4D tensor with shape: ``(batch_size, height, width, channels)``
        Same shape as input.

    Example:
        >>> inputs = keras.Input(shape=(32, 32, 64))
        >>> se_layer = SqueezeExcitation(reduction_ratio=0.25)
        >>> outputs = se_layer(inputs)
        >>> print(outputs.shape)
        (None, 32, 32, 64)
    """

    def __init__(
            self,
            reduction_ratio: float = 0.25,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_normal",
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            activation: Union[str, Callable] = "relu",
            use_bias: bool = False,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if not 0 < reduction_ratio <= 1.0:
            raise ValueError(
                f"reduction_ratio must be in range (0, 1], got {reduction_ratio}"
            )

        # Store configuration parameters
        self.reduction_ratio = reduction_ratio
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.activation = activations.get(activation)

        # Layer components (initialized in build)
        self.input_channels: int = -1
        self.bottleneck_channels: int = -1
        self.global_pool: Optional[layers.Layer] = None
        self.conv_reduce: Optional[layers.Conv2D] = None
        self.conv_restore: Optional[layers.Conv2D] = None

        # Store build input shape for serialization
        self._build_input_shape: Optional[Tuple[int, ...]] = None

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Builds the layer with given input shape.

        Args:
            input_shape: Tuple of integers defining the input shape.
                Expected format: (batch_size, height, width, channels)
        """
        # Store input shape for serialization
        self._build_input_shape = input_shape

        self.input_channels = input_shape[-1]
        self.bottleneck_channels = max(1, int(round(
            self.input_channels * self.reduction_ratio
        )))

        logger.info(f"Building SqueezeExcitation layer: "
                   f"input_channels={self.input_channels}, "
                   f"bottleneck_channels={self.bottleneck_channels}")

        # Global pooling for squeeze operation
        self.global_pool = layers.GlobalAveragePooling2D(keepdims=True)

        # Channel reduction convolution
        self.conv_reduce = layers.Conv2D(
            filters=self.bottleneck_channels,
            kernel_size=(1, 1),
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="conv_reduce"
        )

        # Channel restoration convolution
        self.conv_restore = layers.Conv2D(
            filters=self.input_channels,
            kernel_size=(1, 1),
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="conv_restore"
        )

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass of the layer.

        Args:
            inputs: Input tensor with shape (batch_size, height, width, channels).
            training: Boolean indicating whether in training mode.

        Returns:
            Output tensor after applying SE operations with same shape as input.
        """
        # Squeeze operation - global average pooling
        x = self.global_pool(inputs)

        # Excitation operation
        x = self.conv_reduce(x, training=training)
        x = self.activation(x)
        x = self.conv_restore(x, training=training)
        x = activations.sigmoid(x)

        # Scale original inputs by attention weights
        return ops.multiply(inputs, x)

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input tensor.

        Returns:
            Output shape tuple (same as input shape).
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Returns the config of the layer.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "reduction_ratio": self.reduction_ratio,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "activation": activations.serialize(self.activation),
            "use_bias": self.use_bias,
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get the build configuration for serialization.

        Returns:
            Dictionary containing the build configuration.
        """
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build the layer from a build configuration.

        Args:
            config: Dictionary containing the build configuration.
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

# ---------------------------------------------------------------------
