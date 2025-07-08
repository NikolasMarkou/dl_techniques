"""
ConvNext Block Implementation
===========================

A modern implementation of the ConvNext block architecture as described in:
"A ConvNet for the 2020s" (Liu et al., 2022)
https://arxiv.org/abs/2201.03545

Key Features:
------------
- Depthwise convolution with large kernels
- Inverted bottleneck design
- Proper normalization strategy (LayerNorm)
- GELU activation by default
- Configurable dropout (both standard and spatial)
- Optional learnable scaling (gamma)
- Support for various kernel regularization strategies

Architecture:
------------
The ConvNext block consists of:
1. Depthwise Conv (7x7) for local feature extraction
2. LayerNorm for feature normalization
3. Two-layer MLP for feature transformation:
   - Expansion layer (pointwise conv, 4x channels)
   - GELU activation
   - Optional dropout
   - Reduction layer (pointwise conv, back to input channels)
4. Optional learnable scaling

The computation flow is:
input -> depthwise_conv -> layer_norm ->
        pointwise_conv1 -> activation -> dropout ->
        pointwise_conv2 -> gamma_scale -> output

Configuration:
-------------
Supports extensive customization through direct parameters:
- Kernel sizes and filter counts
- Stride configuration
- Activation functions
- Regularization strategies
- Bias terms

Features two types of dropout:
- Standard dropout for feature regularization
- Spatial dropout for structured regularization

Supports different multiplier types:
- Global: Single scaling factor
- Channel-wise: Per-channel scaling

Usage Examples:
-------------
```python
# Basic configuration
block = ConvNextV1Block(
    kernel_size=7,
    filters=64,
    activation="gelu"
)

# Advanced configuration with regularization
block = ConvNextV1Block(
    kernel_size=7,
    filters=128,
    kernel_regularizer=keras.regularizers.L2(0.01),
    dropout_rate=0.1,
    spatial_dropout_rate=0.1,
    use_softorthonormal_regularizer=True
)
```

Notes:
-----
- The block implements proper normalization ordering
- Uses truncated normal initialization (μ=0, σ=0.02)
- Supports serialization and deserialization
- Implements ResNet-style skip connections
- Compatible with TF/Keras model saving
"""
import copy
import keras
from typing import Optional, Dict, Union, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .layer_scale import LearnableMultiplier
from dl_techniques.regularizers.soft_orthogonal import (
    SoftOrthonormalConstraintRegularizer
)
from dl_techniques.constraints.value_range_constraint import (
    ValueRangeConstraint
)


# ---------------------------------------------------------------------


@keras.utils.register_keras_serializable()
class ConvNextV1Block(keras.layers.Layer):
    """Implementation of ConvNext block with modern best practices.

    Args:
        kernel_size: Size of the convolution kernel
        filters: Number of output filters
        strides: Convolution stride length
        activation: Activation function to use
        kernel_regularizer: Optional regularization for kernel weights
        use_bias: Whether to include a bias term
        dropout_rate: Optional dropout rate
        spatial_dropout_rate: Optional spatial dropout rate
        use_gamma: Whether to use learnable multiplier
        use_softorthonormal_regularizer: If true use soft orthonormal regularizer
        name: Name of the layer
    """

    # Important constants - following ConvNeXt paper specifications
    EXPANSION_FACTOR = 4  # Bottleneck expansion factor (filters * 4)
    INITIALIZER_MEAN = 0.0  # Mean for TruncatedNormal initializer
    INITIALIZER_STDDEV = 0.02  # Standard deviation for TruncatedNormal initializer
    LAYERNORM_EPSILON = 1e-6  # Epsilon for LayerNormalization
    POINTWISE_KERNEL_SIZE = 1  # Kernel size for pointwise convolutions
    GAMMA_L2_REGULARIZATION = 1e-5  # L2 regularization for gamma multiplier
    GAMMA_INITIAL_VALUE = 1.0  # Initial value for gamma multiplier
    GAMMA_MIN_VALUE = 0.0  # Minimum value for gamma constraint
    GAMMA_MAX_VALUE = 1.0  # Maximum value for gamma constraint

    def __init__(
            self,
            kernel_size: Union[int, Tuple[int, int]],
            filters: int,
            strides: Union[int, Tuple[int, int]] = (1, 1),
            activation: str = "gelu",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            use_bias: bool = True,
            dropout_rate: Optional[float] = 0.0,
            spatial_dropout_rate: Optional[float] = 0.0,
            use_gamma: bool = True,
            use_softorthonormal_regularizer: bool = False,
            **kwargs
    ):
        super().__init__(**kwargs)

        # Store configuration parameters
        self.kernel_size = kernel_size
        self.filters = filters
        self.strides = strides
        self.activation_name = activation
        self.kernel_regularizer = kernel_regularizer
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.spatial_dropout_rate = spatial_dropout_rate
        self.use_gamma = use_gamma
        self.use_softorthonormal_regularizer = use_softorthonormal_regularizer

        # Initialize layers
        self.conv_1 = None
        self.conv_2 = None
        self.conv_3 = None
        self.norm = None
        self.activation = None
        self.dropout = None
        self.spatial_dropout = None
        self.gamma = None

    def build(self, input_shape) -> None:
        """Initialize all layers with proper configuration."""
        # Store input shape for serialization
        self._build_input_shape = input_shape

        # Depthwise convolution
        self.conv_1 = keras.layers.DepthwiseConv2D(
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding="same",
            depthwise_initializer=keras.initializers.TruncatedNormal(
                mean=self.INITIALIZER_MEAN,
                stddev=self.INITIALIZER_STDDEV
            ),
            use_bias=self.use_bias,
            depthwise_regularizer=copy.deepcopy(self.kernel_regularizer),
        )
        self.conv_1.build(input_shape)

        # Normalization layer (LayerNorm works on the last axis)
        self.norm = (
            keras.layers.LayerNormalization(
                epsilon=self.LAYERNORM_EPSILON,
                center=self.use_bias,
                scale=True)
        )
        # LayerNorm after depthwise conv, so same input shape
        self.norm.build(input_shape)

        # Point-wise convolutions
        conv_params = {
            "kernel_size": self.POINTWISE_KERNEL_SIZE,
            "padding": "same",
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.TruncatedNormal(
                mean=self.INITIALIZER_MEAN,
                stddev=self.INITIALIZER_STDDEV
            ),
            "kernel_regularizer": copy.deepcopy(self.kernel_regularizer)
        }

        if self.use_softorthonormal_regularizer:
            conv_params["kernel_regularizer"] = SoftOrthonormalConstraintRegularizer()

        # First pointwise conv (expansion)
        self.conv_2 = keras.layers.Conv2D(
            filters=self.filters * self.EXPANSION_FACTOR,
            **conv_params
        )
        self.conv_2.build(input_shape)

        # Calculate intermediate shape after expansion
        intermediate_shape = list(input_shape)
        intermediate_shape[-1] = self.filters * self.EXPANSION_FACTOR
        intermediate_shape = tuple(intermediate_shape)

        # Second pointwise conv (reduction)
        self.conv_3 = keras.layers.Conv2D(
            filters=self.filters,
            **conv_params
        )
        self.conv_3.build(intermediate_shape)

        # Activation layers
        self.activation = keras.layers.Activation(self.activation_name)
        self.activation.build(intermediate_shape)

        # Dropout layers
        if self.dropout_rate is not None and self.dropout_rate > 0:
            self.dropout = keras.layers.Dropout(self.dropout_rate)
            self.dropout.build(intermediate_shape)
        else:
            self.dropout = keras.layers.Lambda(lambda x: x)
            self.dropout.build(intermediate_shape)

        if self.spatial_dropout_rate is not None and self.spatial_dropout_rate > 0:
            self.spatial_dropout = keras.layers.SpatialDropout2D(
                self.spatial_dropout_rate
            )
            self.spatial_dropout.build(intermediate_shape)
        else:
            self.spatial_dropout = keras.layers.Lambda(lambda x: x)
            self.spatial_dropout.build(intermediate_shape)

        # Calculate final shape after second pointwise conv
        final_shape = list(intermediate_shape)
        final_shape[-1] = self.filters
        final_shape = tuple(final_shape)

        # Learnable multiplier
        if self.use_gamma:
            self.gamma = LearnableMultiplier(
                multiplier_type="CHANNEL",
                regularizer=keras.regularizers.L2(self.GAMMA_L2_REGULARIZATION),
                initializer=keras.initializers.Constant(self.GAMMA_INITIAL_VALUE),
                constraint=ValueRangeConstraint(
                    min_value=self.GAMMA_MIN_VALUE,
                    max_value=self.GAMMA_MAX_VALUE
                ),
            )
            self.gamma.build(final_shape)
        else:
            self.gamma = keras.layers.Lambda(lambda x: x)
            self.gamma.build(final_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass of the ConvNext block.

        Args:
            inputs: Input tensor
            training: Whether in training mode

        Returns:
            Processed tensor
        """
        # Depthwise convolution
        x = self.conv_1(inputs, training=training)

        # Normalization (following proper order)
        x = self.norm(x, training=training)

        # First pointwise convolution
        x = self.conv_2(x, training=training)
        x = self.activation(x, training=training)

        # Apply dropouts if specified
        x = self.dropout(x, training=training)
        x = self.spatial_dropout(x, training=training)

        # Second pointwise convolution
        x = self.conv_3(x, training=training)

        # Apply learnable multiplier if specified
        x = self.gamma(x, training=training)

        return x

    def compute_output_shape(self, input_shape):
        """Computes the output shape of the ConvNeXt block.

        Args:
            input_shape: Shape tuple (tuple of integers)
                representing the input shape (batch_size, height, width, channels).

        Returns:
            tuple: Output shape after applying the ConvNeXt block,
            considering strides and output channels from configuration.

        Raises:
            ValueError: If input shape doesn't have 4 dimensions.
        """
        if isinstance(input_shape, list):
            return [self.compute_output_shape(shape) for shape in input_shape]

        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input tensor, got shape: {input_shape}")

        # Extract dimensions (NHWC format)
        batch_size, height, width, _ = input_shape

        # Normalize strides to tuple format
        if isinstance(self.strides, int):
            strides = (self.strides, self.strides)
        else:
            strides = self.strides

        # Calculate new height and width based on strides
        new_height = height // strides[0]
        new_width = width // strides[1]

        # Output channels determined by the filters parameter
        output_channels = self.filters

        return (batch_size, new_height, new_width, output_channels)

    def get_config(self) -> Dict:
        """Returns the config of the layer for serialization.

        Returns:
            Configuration dictionary
        """
        config = super().get_config()
        config.update({
            "kernel_size": self.kernel_size,
            "filters": self.filters,
            "strides": self.strides,
            "activation": self.activation_name,
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "use_bias": self.use_bias,
            "dropout_rate": self.dropout_rate,
            "spatial_dropout_rate": self.spatial_dropout_rate,
            "use_gamma": self.use_gamma,
            "use_softorthonormal_regularizer": self.use_softorthonormal_regularizer,
        })
        return config

    def get_build_config(self):
        """Get build configuration for proper serialization.

        Returns:
            Build configuration dictionary
        """
        return {
            "input_shape": getattr(self, '_build_input_shape', None),
        }

    def build_from_config(self, config):
        """Build layer from build configuration.

        Args:
            config: Build configuration dictionary
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

    @classmethod
    def from_config(cls, config):
        """Creates a layer from its config.

        Args:
            config: Configuration dictionary

        Returns:
            ConvNextV1Block instance
        """
        from copy import deepcopy

        # Make a copy of the config to avoid modifying the original
        config_copy = deepcopy(config)

        # Deserialize the kernel_regularizer if it exists
        if "kernel_regularizer" in config_copy and config_copy["kernel_regularizer"] is not None:
            config_copy["kernel_regularizer"] = keras.regularizers.deserialize(
                config_copy["kernel_regularizer"]
            )

        # Create the ConvNextV1Block with the configuration
        return cls(**config_copy)

# ---------------------------------------------------------------------