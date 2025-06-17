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
config = ConvNextV2Config(
    kernel_size=7,
    filters=64,
    activation="gelu"
)
block = ConvNextV2Block(config)

# Advanced configuration with regularization
config = ConvNextV2Config(
    kernel_size=7,
    filters=128,
    kernel_regularizer=keras.regularizers.L2(0.01)
)
block = ConvNextV2Block(
    conv_config=config,
    dropout_rate=0.1,
    spatial_dropout_rate=0.05
)
```

Notes:
-----
- Follows proper normalization ordering (Linear/Conv → Norm → Activation)
- Uses truncated normal initialization (μ=0, σ=0.02)
- Implements full serialization support
- Compatible with TF/Keras model saving
"""

import copy
import keras
import tensorflow as tf
from dataclasses import dataclass, asdict
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
from dl_techniques.layers.norms.global_response_norm import GlobalResponseNormalization


# ---------------------------------------------------------------------

@dataclass
class ConvNextV2Config:
    """Configuration for ConvNext block parameters.

    Args:
        kernel_size: Size of the convolution kernel
        filters: Number of output filters
        strides: Convolution stride length
        activation: Activation function to use
        kernel_regularizer: Optional regularization for kernel weights
        use_bias: Whether to include a bias term
    """
    kernel_size: Union[int, Tuple[int, int]]
    filters: int
    strides: Union[int, Tuple[int, int]] = (1, 1)
    activation: str = "gelu"
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None
    use_bias: bool = True


# ---------------------------------------------------------------------


@keras.utils.register_keras_serializable()
class ConvNextV2Block(keras.layers.Layer):
    """Implementation of ConvNext block with modern best practices.

    Args:
        conv_config: Configuration for convolution layers
        dropout_rate: Optional dropout rate
        spatial_dropout_rate: Optional spatial dropout rate
        use_gamma: Whether to use learnable multiplier
        use_softorthonormal_regularizer: If true use soft orthonormal regularizer
        name: Name of the layer
    """

    def __init__(
            self,
            conv_config: ConvNextV2Config,
            dropout_rate: Optional[float] = 0.0,
            spatial_dropout_rate: Optional[float] = 0.0,
            use_gamma: bool = False,
            use_softorthonormal_regularizer: bool = False,
            **kwargs
    ):
        super().__init__(**kwargs)

        # Store configurations
        self.conv_config = conv_config
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
        self.gamma= None
        self.grn = None

    def build(self, input_shape) -> None:
        """Initialize all layers with proper configuration."""
        # Depthwise convolution
        self.conv_1 = keras.layers.DepthwiseConv2D(
            kernel_size=self.conv_config.kernel_size,
            strides=self.conv_config.strides,
            padding="same",
            depthwise_initializer=keras.initializers.TruncatedNormal(
                mean=0.0, stddev=0.02
            ),
            use_bias=self.conv_config.use_bias,
            depthwise_regularizer=copy.deepcopy(self.conv_config.kernel_regularizer),
        )

        # Normalization layer
        self.norm = (
            keras.layers.LayerNormalization(
                epsilon=1e-6,
                center=self.conv_config.use_bias,
                scale=True)
        )

        # Point-wise convolutions
        conv_params = {
            "kernel_size": 1,
            "padding": "same",
            "use_bias": self.conv_config.use_bias,
            "kernel_initializer": keras.initializers.TruncatedNormal(
                mean=0.0, stddev=0.02
            ),
            "kernel_regularizer": copy.deepcopy(self.conv_config.kernel_regularizer)
        }

        if self.use_softorthonormal_regularizer == "orthonormal":
            conv_params["kernel_regularizer"] = SoftOrthonormalConstraintRegularizer()


        self.conv_2 = keras.layers.Conv2D(
            filters=self.conv_config.filters * 4,
            **conv_params
        )
        self.conv_3 = keras.layers.Conv2D(
            filters=self.conv_config.filters,
            **conv_params
        )

        # Activation layers
        self.activation = keras.layers.Activation(self.conv_config.activation)

        # Dropout layers
        if self.dropout_rate is not None and self.dropout_rate > 0:
            self.dropout = keras.layers.Dropout(self.dropout_rate)
        else:
            self.dropout = keras.layers.Lambda(lambda x: x)

        if self.spatial_dropout_rate is not None and self.spatial_dropout_rate > 0:
            self.spatial_dropout = keras.layers.SpatialDropout2D(
                self.spatial_dropout_rate
            )
        else:
            self.spatial_dropout = keras.layers.Lambda(lambda x: x)

        # Learnable multiplier
        if self.use_gamma:
            self.gamma = LearnableMultiplier(
                multiplier_type="CHANNEL",
                regularizer=keras.regularizers.L2(1e-5),
                initializer=keras.initializers.Constant(1.0),
                constraint=ValueRangeConstraint(min_value=0.0, max_value=1.0),
            )
        else:
            self.gamma = keras.layers.Lambda(lambda x: x)

        # Global Response Normalization (GRN) - key feature of ConvNeXt V2
        self.grn = GlobalResponseNormalization(eps=1e-6)


    def call(
            self,
            inputs: tf.Tensor,
            training: Optional[bool] = None
    ) -> tf.Tensor:
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
        """Computes the output shape of the ConvNeXt block.

        Args:
            input_shape: Shape tuple (tuple of integers)
                representing the input shape (batch_size, height, width, channels).

        Returns:
            tuple: Output shape after applying the ConvNeXt block,
            considering strides and output channels from conv_config.

        Raises:
            ValueError: If input shape doesn't have 4 dimensions.
        """
        if isinstance(input_shape, list):
            return [self.compute_output_shape(shape) for shape in input_shape]

        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input tensor, got shape: {input_shape}")

        # Extract dimensions (NHWC format)
        batch_size, height, width, _ = input_shape

        # Get strides from conv_config if it exists
        strides = getattr(self.conv_config, 'strides', (1, 1))

        # Calculate new height and width based on strides
        new_height = height // strides[0]
        new_width = width // strides[1]

        # Output channels determined by the conv_config
        output_channels = getattr(self.conv_config, 'filters', -1)

        return (batch_size, new_height, new_width, output_channels)

    def get_config(self) -> Dict:
        """Returns the config of the layer for serialization.

        Returns:
            Configuration dictionary
        """
        config = super().get_config()
        config.update({
            "conv_config": asdict(self.conv_config),
            "dropout_rate": self.dropout_rate,
            "spatial_dropout_rate": self.spatial_dropout_rate,
            "use_gamma": self.use_gamma,
            "use_softorthonormal_regularizer": self.use_softorthonormal_regularizer,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Creates a layer from its config."""
        from copy import deepcopy

        # Make a copy of the config to avoid modifying the original
        config_copy = deepcopy(config)

        # Extract the conv_config dictionary
        conv_config_dict = config_copy.pop("conv_config", {})

        # Recreate the ConvNextConfig object
        conv_config = ConvNextV2Config(**conv_config_dict)

        # Create the ConvNextBlock with the recreated ConvNextConfig
        return cls(conv_config=conv_config, **config_copy)


# ---------------------------------------------------------------------

