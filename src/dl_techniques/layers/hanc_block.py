"""
HANC Block Implementation for ACC-UNet.

This module implements the HANC block, which is the main building block of ACC-UNet.
It combines inverted bottleneck expansion, depthwise convolution, hierarchical
context aggregation, and squeeze-excitation in a single block.
"""

import keras
from keras import ops
from typing import Optional, Union, Tuple, Any

from .squeeze_excitation import SqueezeExcitation
from .hanc_layer import HANCLayer


class HANCBlock(keras.layers.Layer):
    """
    Hierarchical Aggregation of Neighborhood Context (HANC) Block.

    This block implements the main building block from ACC-UNet, combining:
    1. Inverted bottleneck expansion (1x1 conv to expand channels)
    2. Depthwise 3x3 convolution
    3. Hierarchical context aggregation (HANC layer)
    4. Residual connection
    5. Final 1x1 convolution with Squeeze-Excitation

    The block provides long-range dependencies through hierarchical pooling
    operations while maintaining efficiency through depthwise convolutions.

    Args:
        filters: Number of output filters.
        k: Hierarchical levels for HANC operation (1-5 supported).
        inv_factor: Inverted bottleneck expansion factor.
        kernel_initializer: Initializer for convolution kernels.
        bias_initializer: Initializer for bias vectors.
        kernel_regularizer: Regularizer for convolution kernels.
        bias_regularizer: Regularizer for bias vectors.
        **kwargs: Additional arguments for the Layer base class.

    Input shape:
        4D tensor with shape (batch_size, height, width, input_channels).

    Output shape:
        4D tensor with shape (batch_size, height, width, filters).

    Example:
        ```python
        # Basic usage
        block = HANCBlock(filters=64, k=3, inv_factor=3)

        # Custom configuration
        block = HANCBlock(
            filters=128,
            k=4,
            inv_factor=4,
            kernel_initializer='he_normal'
        )

        # In a model
        inputs = keras.Input(shape=(224, 224, 32))
        x = HANCBlock(filters=64)(inputs)
        ```

    Note:
        The input channels are automatically detected during the build phase.
        The block includes residual connections when input and output channels match.
    """

    def __init__(
            self,
            filters: int,
            k: int = 3,
            inv_factor: int = 3,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.filters = filters
        self.k = k
        self.inv_factor = inv_factor
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # Validate parameters
        if filters <= 0:
            raise ValueError(f"filters must be positive, got {filters}")
        if k < 1 or k > 5:
            raise ValueError(f"k must be between 1 and 5, got {k}")
        if inv_factor <= 0:
            raise ValueError(f"inv_factor must be positive, got {inv_factor}")

        # Will be initialized in build()
        self.input_channels = None
        self.expanded_channels = None
        self.use_residual = False

        # Layer components
        self.expand_conv = None
        self.expand_bn = None
        self.depthwise_conv = None
        self.depthwise_bn = None
        self.hanc_layer = None
        self.residual_bn = None
        self.output_conv = None
        self.output_bn = None
        self.squeeze_excitation = None
        self.activation = None

        self._build_input_shape = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer components."""
        self._build_input_shape = input_shape
        self.input_channels = input_shape[-1]
        self.expanded_channels = self.input_channels * self.inv_factor

        # Check if we can use residual connection
        self.use_residual = (self.input_channels == self.filters)

        # 1. Expansion convolution (1x1)
        self.expand_conv = keras.layers.Conv2D(
            filters=self.expanded_channels,
            kernel_size=1,
            padding='same',
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='expand_conv'
        )
        self.expand_bn = keras.layers.BatchNormalization(name='expand_bn')

        # 2. Depthwise convolution (3x3)
        self.depthwise_conv = keras.layers.DepthwiseConv2D(
            kernel_size=3,
            padding='same',
            use_bias=False,
            depthwise_initializer=self.kernel_initializer,
            depthwise_regularizer=self.kernel_regularizer,
            name='depthwise_conv'
        )
        self.depthwise_bn = keras.layers.BatchNormalization(name='depthwise_bn')

        # 3. HANC layer for hierarchical context aggregation
        self.hanc_layer = HANCLayer(
            in_channels=self.expanded_channels,
            out_channels=self.input_channels,
            k=self.k,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='hanc'
        )

        # 4. Residual connection batch norm (applied after adding residual)
        if self.use_residual:
            self.residual_bn = keras.layers.BatchNormalization(name='residual_bn')

        # 5. Output convolution (1x1)
        self.output_conv = keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=1,
            padding='same',
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='output_conv'
        )
        self.output_bn = keras.layers.BatchNormalization(name='output_bn')

        # 6. Squeeze-Excitation
        self.squeeze_excitation = SqueezeExcitation(
            reduction_ratio=0.25,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='se'
        )

        # 7. Activation
        self.activation = keras.layers.LeakyReLU(negative_slope=0.01, name='activation')

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass computation."""
        # 1. Expansion phase
        x = self.expand_conv(inputs)
        x = self.expand_bn(x, training=training)
        x = self.activation(x)

        # 2. Depthwise convolution
        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x, training=training)
        x = self.activation(x)

        # 3. Hierarchical context aggregation
        x = self.hanc_layer(x, training=training)

        # 4. Residual connection (if applicable)
        if self.use_residual:
            x = x + inputs
            x = self.residual_bn(x, training=training)

        # 5. Output projection
        x = self.output_conv(x)
        x = self.output_bn(x, training=training)
        x = self.activation(x)

        # 6. Squeeze-Excitation
        x = self.squeeze_excitation(x)

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        return tuple(list(input_shape[:-1]) + [self.filters])

    def get_config(self) -> dict:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'k': self.k,
            'inv_factor': self.inv_factor,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

    def get_build_config(self) -> dict:
        """Get build configuration."""
        return {
            'input_shape': self._build_input_shape,
        }

    def build_from_config(self, config: dict) -> None:
        """Build from configuration."""
        if config.get('input_shape') is not None:
            self.build(config['input_shape'])