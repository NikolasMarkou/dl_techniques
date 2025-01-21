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
input -> depthwise_conv -> layer_norm -> activation ->
        pointwise_conv1 -> activation -> dropout ->
        pointwise_conv2 -> gamma_scale -> output

Configuration:
-------------
Supports extensive customization through ConvNextConfig:
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
config = ConvNextConfig(
    kernel_size=7,
    filters=64,
    activation="gelu"
)
block = ConvNextBlock(config)

# Advanced configuration with regularization
config = ConvNextConfig(
    kernel_size=7,
    filters=128,
    kernel_regularizer=keras.regularizers.L2(0.01)
)
block = ConvNextBlock(
    conv_config=config,
    dropout_rate=0.1,
    spatial_dropout_rate=0.1,
    kernel_regularization="orthogonal"
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

import keras
from enum import Enum
import tensorflow as tf
from dataclasses import dataclass
from typing import Optional, Dict, Union, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .layer_scale import LearnableMultiplier
from dl_techniques.regularizers.soft_orthogonal import \
    SoftOrthogonalConstraintRegularizer, \
    SoftOrthonormalConstraintRegularizer

# ---------------------------------------------------------------------


class MultiplierType(Enum):
    """Type of learnable multiplier to use."""
    Global = "global"
    Channel = "channel"


# ---------------------------------------------------------------------

@dataclass
class ConvNextConfig:
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
class ConvNextBlock(keras.layers.Layer):
    """Implementation of ConvNext block with modern best practices.

    Args:
        conv_config: Configuration for convolution layers
        dropout_rate: Optional dropout rate
        spatial_dropout_rate: Optional spatial dropout rate
        use_gamma: Whether to use learnable multiplier
        use_global_gamma: Whether to use global or per-channel multiplier
        kernel_regularization: Type of kernel regularization to use
        name: Name of the layer
    """

    def __init__(
            self,
            conv_config: ConvNextConfig,
            dropout_rate: Optional[float] = None,
            spatial_dropout_rate: Optional[float] = None,
            use_gamma: bool = True,
            use_global_gamma: bool = True,
            kernel_regularization: Optional[str] = None,
            name: Optional[str] = None,
    ):
        super().__init__(name=name)

        # Store configurations
        self.conv_config = conv_config
        self.dropout_rate = dropout_rate
        self.spatial_dropout_rate = spatial_dropout_rate
        self.use_gamma = use_gamma
        self.use_global_gamma = use_global_gamma
        self.kernel_regularization = kernel_regularization

        # Initialize layers
        self.conv_1 = None
        self.conv_2 = None
        self.conv_3 = None
        self.norm = None
        self.activation = None
        self.dropout = None
        self.spatial_dropout = None
        self.gamma= None

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
        )

        # Normalization layer
        self.norm = keras.layers.LayerNormalization(epsilon=1e-6)

        # Point-wise convolutions
        conv_params = {
            "kernel_size": 1,
            "padding": "same",
            "use_bias": self.conv_config.use_bias,
            "kernel_initializer": keras.initializers.TruncatedNormal(
                mean=0.0, stddev=0.02
            ),
            "kernel_regularizer": keras.regularizers.L2()
        }

        if self.kernel_regularization == "orthogonal":
            conv_params["kernel_regularizer"] = SoftOrthogonalConstraintRegularizer()
        elif self.kernel_regularization == "orthonormal":
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
        if self.dropout_rate is not None:
            self.dropout = keras.layers.Dropout(self.dropout_rate)
        if self.spatial_dropout_rate is not None:
            self.spatial_dropout = keras.layers.SpatialDropout2D(
                self.spatial_dropout_rate
            )

        # Learnable multiplier
        if self.use_gamma:
            self.gamma = LearnableMultiplier(
                multiplier_type=MultiplierType.Global if self.use_global_gamma else MultiplierType.Channel,
                capped=True,
                regularizer=keras.regularizers.l2(1e-2),
            )

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
        x = self.conv_1(inputs)

        # Normalization (following proper order)
        x = self.norm(x, training=training)

        # First activation
        x = self.activation(x)

        # First pointwise convolution
        x = self.conv_2(x)
        x = self.activation(x)

        # Apply dropouts if specified
        if hasattr(self, 'dropout'):
            x = self.dropout(x, training=training)
        if hasattr(self, 'spatial_dropout'):
            x = self.spatial_dropout(x, training=training)

        # Second pointwise convolution
        x = self.conv_3(x)

        # Apply learnable multiplier if specified
        if self.use_gamma:
            x = self.gamma(x)

        return x

    def get_config(self) -> Dict:
        """Returns the config of the layer for serialization.

        Returns:
            Configuration dictionary
        """
        config = super().get_config()
        config.update({
            "conv_config": self.conv_config.__dict__,
            "dropout_rate": self.dropout_rate,
            "spatial_dropout_rate": self.spatial_dropout_rate,
            "use_gamma": self.use_gamma,
            "use_global_gamma": self.use_global_gamma,
            "kernel_regularization": self.kernel_regularization,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict) -> 'ConvNextBlock':
        """Creates a layer from its config.

        Args:
            config: Layer configuration dictionary

        Returns:
            Instantiated ConvNextBlock layer
        """
        conv_config = ConvNextConfig(**config.pop("conv_config"))
        return cls(conv_config=conv_config, **config)

# ---------------------------------------------------------------------
