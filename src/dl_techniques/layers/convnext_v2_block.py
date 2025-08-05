"""
ConvNextV2 Block Implementation
===============================

A modern implementation of the ConvNextV2 block architecture as described in:
"ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders" (Woo et al., 2023)
https://arxiv.org/abs/2301.00808

Key Features:
------------
- Depthwise convolution with large kernels (7x7)
- Proper layer normalization
- Inverted bottleneck design (expand-reduce pattern)
- GELU activation function
- Global Response Normalization (GRN) for enhanced feature competition
- Flexible dropout options (standard and spatial)
- Residual connections throughout
- Customizable regularization strategy

Architecture:
------------
The ConvNextV2 block consists of:
1. Depthwise Conv (7x7) for local feature extraction
2. LayerNorm for feature normalization
3. Pointwise Conv (1x1) for channel expansion (4x)
4. GELU activation function
5. Global Response Normalization (GRN) - key innovation in V2
6. Optional dropout for regularization
7. Pointwise Conv (1x1) for channel reduction

The computation flow is:
input → depthwise_conv → layernorm → pointwise_conv1 → activation →
        GRN → dropout → pointwise_conv2 → output

Improvements over ConvNextV1:
----------------------------
- Global Response Normalization (GRN) enhances inter-channel feature competition
- Improved normalization strategy
- Enhanced feature representation capacity
- Better generalization to downstream tasks

Primary difference from V1 is the GRN layer, which:
1. Computes L2 norm across spatial dimensions
2. Normalizes by mean of L2 norm
3. Applies learnable scaling (gamma) and bias (beta)
4. Residual connection not included

Usage Examples:
-------------
```python
# Basic configuration
block = ConvNextV2Block(
    kernel_size=7,
    filters=64,
    activation="gelu"
)

# Advanced configuration with regularization
block = ConvNextV2Block(
    kernel_size=7,
    filters=128,
    kernel_regularizer=keras.regularizers.L2(0.01),
    dropout_rate=0.1,
    spatial_dropout_rate=0.05,
    use_softorthonormal_regularizer=True
)
```

Notes:
-----
- Follows proper normalization ordering (Linear/Conv → Norm → Activation)
- Uses truncated normal initialization (μ=0, σ=0.02)
"""

import copy
import keras
from typing import Optional, Dict, Union, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .layer_scale import LearnableMultiplier
from .norms.global_response_norm import GlobalResponseNormalization
from ..constraints.value_range_constraint import ValueRangeConstraint
from ..regularizers.soft_orthogonal import SoftOrthonormalConstraintRegularizer

# ---------------------------------------------------------------------


@keras.utils.register_keras_serializable()
class ConvNextV2Block(keras.layers.Layer):
    """Implementation of ConvNextV2 block with modern best practices.

    Args:
        kernel_size: Size of the convolution kernel
        filters: Number of output filters
        activation: Activation function to use
        kernel_regularizer: Optional regularization for kernel weights
        use_bias: Whether to include a bias term
        dropout_rate: Optional dropout rate
        spatial_dropout_rate: Optional spatial dropout rate
        use_gamma: Whether to use learnable multiplier
        use_softorthonormal_regularizer: If true use soft orthonormal regularizer
        name: Name of the layer
    """

    # Important constants - following ConvNeXt V2 paper specifications
    EXPANSION_FACTOR = 4  # Bottleneck expansion factor (filters * 4)
    INITIALIZER_MEAN = 0.0  # Mean for TruncatedNormal initializer
    INITIALIZER_STDDEV = 0.02  # Standard deviation for TruncatedNormal initializer
    LAYERNORM_EPSILON = 1e-6  # Epsilon for LayerNormalization
    GRN_EPSILON = 1e-6  # Epsilon for Global Response Normalization
    POINTWISE_KERNEL_SIZE = 1  # Kernel size for pointwise convolutions
    GAMMA_L2_REGULARIZATION = 1e-5  # L2 regularization for gamma multiplier
    GAMMA_INITIAL_VALUE = 1.0  # Initial value for gamma multiplier
    GAMMA_MIN_VALUE = 0.0  # Minimum value for gamma constraint
    GAMMA_MAX_VALUE = 1.0  # Maximum value for gamma constraint
    STRIDES = (1, 1) # Strides for depthwise

    def __init__(
            self,
            kernel_size: Union[int, Tuple[int, int]],
            filters: int,
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
        self.grn = None

    def build(self, input_shape) -> None:
        """Initialize all layers with proper configuration."""
        # Store input shape for serialization
        self._build_input_shape = input_shape

        # Depthwise convolution
        self.conv_1 = keras.layers.DepthwiseConv2D(
            kernel_size=self.kernel_size,
            strides=self.STRIDES,
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

        # Activation layers
        self.activation = keras.layers.Activation(self.activation_name)
        self.activation.build(intermediate_shape)

        # Global Response Normalization (GRN) - key feature of ConvNeXt V2
        self.grn = GlobalResponseNormalization(eps=self.GRN_EPSILON)
        self.grn.build(intermediate_shape)

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

        # Second pointwise conv (reduction)
        self.conv_3 = keras.layers.Conv2D(
            filters=self.filters,
            **conv_params
        )
        self.conv_3.build(intermediate_shape)

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

        # Global Response Normalization - key ConvNeXt V2 feature
        x = self.grn(x, training=training)

        # Apply dropouts if specified
        x = self.dropout(x, training=training)
        x = self.spatial_dropout(x, training=training)

        # Second pointwise convolution
        x = self.conv_3(x, training=training)

        # Apply learnable multiplier if specified
        x = self.gamma(x, training=training)

        return x

    def compute_output_shape(self, input_shape):
        """Computes the output shape of the ConvNeXt V2 block.

        Args:
            input_shape: Shape tuple (tuple of integers)
                representing the input shape (batch_size, height, width, channels).

        Returns:
            tuple: Output shape after applying the ConvNeXt V2 block,
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

        # Output channels determined by the filters parameter
        output_channels = self.filters

        return (batch_size, height, width, output_channels)

    def get_config(self) -> Dict:
        """Returns the config of the layer for serialization.

        Returns:
            Configuration dictionary
        """
        config = super().get_config()
        config.update({
            "kernel_size": self.kernel_size,
            "filters": self.filters,
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
            ConvNextV2Block instance
        """
        from copy import deepcopy

        # Make a copy of the config to avoid modifying the original
        config_copy = deepcopy(config)

        # Deserialize the kernel_regularizer if it exists
        if "kernel_regularizer" in config_copy and config_copy["kernel_regularizer"] is not None:
            config_copy["kernel_regularizer"] = keras.regularizers.deserialize(
                config_copy["kernel_regularizer"]
            )

        # Create the ConvNextV2Block with the configuration
        return cls(**config_copy)

# ---------------------------------------------------------------------