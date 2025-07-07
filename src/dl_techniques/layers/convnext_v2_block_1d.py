"""
ConvNextV2 Block 1D Implementation
=================================

A 1D implementation of the ConvNextV2 block architecture adapted for sequential data.
Based on "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders" (Woo et al., 2023)
https://arxiv.org/abs/2301.00808

Key Features:
------------
- Depthwise convolution with large kernels (7 by default) for temporal feature extraction
- Proper layer normalization
- Inverted bottleneck design (expand-reduce pattern)
- GELU activation function
- Global Response Normalization (GRN) adapted for 1D sequences
- Flexible dropout options (standard and spatial)
- Residual connections throughout
- Customizable regularization strategy

Architecture:
------------
The ConvNextV2 1D block consists of:
1. Depthwise Conv1D (kernel_size=7) for local temporal feature extraction
2. LayerNorm for feature normalization
3. Pointwise Conv1D (kernel_size=1) for channel expansion (4x)
4. GELU activation function
5. Global Response Normalization (GRN) - adapted for 1D sequences
6. Optional dropout for regularization
7. Pointwise Conv1D (kernel_size=1) for channel reduction

The computation flow is:
input → depthwise_conv1d → layernorm → pointwise_conv1d → activation →
        GRN → dropout → pointwise_conv1d → output

Improvements over ConvNextV1:
----------------------------
- Global Response Normalization (GRN) enhances inter-channel feature competition
- Improved normalization strategy for sequential data
- Enhanced feature representation capacity
- Better generalization to downstream tasks

Usage Examples:
-------------
```python
# Basic configuration for time series
config = ConvNextV2Config1D(
    kernel_size=7,
    filters=64,
    activation="gelu"
)
block = ConvNextV2Block1D(config)

# Advanced configuration with regularization
config = ConvNextV2Config1D(
    kernel_size=11,
    filters=128,
    kernel_regularizer=keras.regularizers.L2(0.01)
)
block = ConvNextV2Block1D(
    conv_config=config,
    dropout_rate=0.1,
    spatial_dropout_rate=0.05
)

# Input shape: (batch_size, sequence_length, channels)
x = keras.Input(shape=(1000, 64))
y = block(x)
```

Notes:
-----
- Follows proper normalization ordering (Linear/Conv → Norm → Activation)
- Uses truncated normal initialization (μ=0, σ=0.02)
- Implements full serialization support
- Compatible with TF/Keras model saving
- Adapted for sequential data processing
"""

import copy
import keras
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Tuple

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
class ConvNextV2Config1D:
    """Configuration for ConvNext 1D block parameters.

    Args:
        kernel_size: Size of the 1D convolution kernel
        filters: Number of output filters
        strides: Convolution stride length
        activation: Activation function to use
        kernel_regularizer: Optional regularization for kernel weights
        use_bias: Whether to include a bias term
    """
    kernel_size: int
    filters: int
    strides: int = 1
    activation: str = "gelu"
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None
    use_bias: bool = True


# ---------------------------------------------------------------------


@keras.utils.register_keras_serializable()
class ConvNextV2Block1D(keras.layers.Layer):
    """Implementation of ConvNext 1D block for sequential data processing.

    This layer adapts the ConvNextV2 architecture for 1D sequential data,
    such as time series, audio signals, or text sequences.

    Args:
        conv_config: Configuration for convolution layers
        dropout_rate: Optional dropout rate for regularization
        spatial_dropout_rate: Optional spatial dropout rate (applied along time dimension)
        use_gamma: Whether to use learnable multiplier for feature scaling
        use_softorthonormal_regularizer: If true use soft orthonormal regularizer
        name: Name of the layer
        **kwargs: Additional keyword arguments for the base Layer class

    Input shape:
        3D tensor with shape: (batch_size, sequence_length, channels)

    Output shape:
        3D tensor with shape: (batch_size, new_sequence_length, filters)
        where new_sequence_length depends on strides and padding

    Example:
        ```python
        # Create configuration
        config = ConvNextV2Config1D(
            kernel_size=7,
            filters=128,
            activation="gelu"
        )

        # Create block
        block = ConvNextV2Block1D(
            conv_config=config,
            dropout_rate=0.1
        )

        # Process sequential data
        x = keras.Input(shape=(1000, 64))  # 1000 time steps, 64 channels
        y = block(x)
        ```
    """

    def __init__(
            self,
            conv_config: ConvNextV2Config1D,
            dropout_rate: Optional[float] = 0.0,
            spatial_dropout_rate: Optional[float] = 0.0,
            use_gamma: bool = False,
            use_softorthonormal_regularizer: bool = False,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)

        # Store configurations
        self.conv_config = conv_config
        self.dropout_rate = dropout_rate
        self.spatial_dropout_rate = spatial_dropout_rate
        self.use_gamma = use_gamma
        self.use_softorthonormal_regularizer = use_softorthonormal_regularizer

        # Initialize layers to None (will be created in build)
        self.conv_1 = None
        self.conv_2 = None
        self.conv_3 = None
        self.norm = None
        self.activation = None
        self.dropout = None
        self.spatial_dropout = None
        self.gamma = None
        self.grn = None

        # Store build input shape for serialization
        self._build_input_shape = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Initialize all layers with proper configuration.

        Args:
            input_shape: Shape tuple (tuple of integers) indicating the input shape
                of the layer. Expected shape: (batch_size, sequence_length, channels)
        """
        # Store input shape for serialization
        self._build_input_shape = input_shape

        # Depthwise convolution for temporal feature extraction
        self.conv_1 = keras.layers.Conv1D(
            filters=input_shape[-1],  # Keep same number of channels for depthwise
            kernel_size=self.conv_config.kernel_size,
            strides=self.conv_config.strides,
            padding="same",
            groups=input_shape[-1],  # Depthwise convolution
            kernel_initializer=keras.initializers.TruncatedNormal(
                mean=0.0, stddev=0.02
            ),
            use_bias=self.conv_config.use_bias,
            kernel_regularizer=copy.deepcopy(self.conv_config.kernel_regularizer),
            name="depthwise_conv1d"
        )

        # Normalization layer
        self.norm = keras.layers.LayerNormalization(
            epsilon=1e-6,
            center=self.conv_config.use_bias,
            scale=True,
            name="layer_norm"
        )

        # Point-wise convolutions parameters
        conv_params = {
            "kernel_size": 1,
            "padding": "same",
            "use_bias": self.conv_config.use_bias,
            "kernel_initializer": keras.initializers.TruncatedNormal(
                mean=0.0, stddev=0.02
            ),
            "kernel_regularizer": copy.deepcopy(self.conv_config.kernel_regularizer)
        }

        # Apply soft orthonormal regularizer if specified
        if self.use_softorthonormal_regularizer:
            conv_params["kernel_regularizer"] = SoftOrthonormalConstraintRegularizer()

        # Expansion pointwise convolution (4x expansion)
        self.conv_2 = keras.layers.Conv1D(
            filters=self.conv_config.filters * 4,
            name="pointwise_conv1d_expand",
            **conv_params
        )

        # Reduction pointwise convolution
        self.conv_3 = keras.layers.Conv1D(
            filters=self.conv_config.filters,
            name="pointwise_conv1d_reduce",
            **conv_params
        )

        # Activation layer
        self.activation = keras.layers.Activation(
            self.conv_config.activation,
            name="activation"
        )

        # Dropout layers
        if self.dropout_rate is not None and self.dropout_rate > 0:
            self.dropout = keras.layers.Dropout(
                self.dropout_rate,
                name="dropout"
            )
        else:
            self.dropout = keras.layers.Lambda(lambda x: x, name="identity_dropout")

        if self.spatial_dropout_rate is not None and self.spatial_dropout_rate > 0:
            self.spatial_dropout = keras.layers.SpatialDropout1D(
                self.spatial_dropout_rate,
                name="spatial_dropout1d"
            )
        else:
            self.spatial_dropout = keras.layers.Lambda(lambda x: x, name="identity_spatial_dropout")

        # Learnable multiplier (gamma scaling)
        if self.use_gamma:
            self.gamma = LearnableMultiplier(
                multiplier_type="CHANNEL",
                regularizer=keras.regularizers.L2(1e-5),
                initializer=keras.initializers.Constant(1.0),
                constraint=ValueRangeConstraint(min_value=0.0, max_value=1.0),
                name="gamma_multiplier"
            )
        else:
            self.gamma = keras.layers.Lambda(lambda x: x, name="identity_gamma")

        # Global Response Normalization (GRN) - adapted for 1D sequences
        self.grn = GlobalResponseNormalization(
            eps=1e-6,
            name="global_response_norm"
        )

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass of the ConvNext 1D block.

        Args:
            inputs: Input tensor with shape (batch_size, sequence_length, channels)
            training: Whether in training mode

        Returns:
            Processed tensor with shape (batch_size, new_sequence_length, filters)
        """
        # Depthwise convolution for temporal feature extraction
        x = self.conv_1(inputs, training=training)

        # Normalization (following proper order: Conv → Norm → Activation)
        x = self.norm(x, training=training)

        # First pointwise convolution (expansion)
        x = self.conv_2(x, training=training)
        x = self.activation(x, training=training)

        # Global Response Normalization - key ConvNeXt V2 feature
        x = self.grn(x, training=training)

        # Apply dropouts if specified
        x = self.dropout(x, training=training)
        x = self.spatial_dropout(x, training=training)

        # Second pointwise convolution (reduction)
        x = self.conv_3(x, training=training)

        # Apply learnable multiplier if specified
        x = self.gamma(x, training=training)

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Computes the output shape of the ConvNeXt 1D block.

        Args:
            input_shape: Shape tuple (tuple of integers) representing the input shape
                (batch_size, sequence_length, channels).

        Returns:
            tuple: Output shape after applying the ConvNeXt 1D block,
            considering strides and output channels from conv_config.

        Raises:
            ValueError: If input shape doesn't have 3 dimensions.
        """
        if isinstance(input_shape, list):
            return [self.compute_output_shape(shape) for shape in input_shape]

        if len(input_shape) != 3:
            raise ValueError(f"Expected 3D input tensor, got shape: {input_shape}")

        # Extract dimensions (batch_size, sequence_length, channels)
        batch_size, sequence_length, _ = input_shape

        # Get stride from conv_config
        stride = getattr(self.conv_config, 'strides', 1)

        # Calculate new sequence length based on stride
        if sequence_length is not None:
            new_sequence_length = sequence_length // stride
        else:
            new_sequence_length = None

        # Output channels determined by the conv_config
        output_channels = getattr(self.conv_config, 'filters', input_shape[-1])

        return (batch_size, new_sequence_length, output_channels)

    def get_config(self) -> Dict:
        """Returns the config of the layer for serialization.

        Returns:
            Configuration dictionary containing all layer parameters
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

    def get_build_config(self) -> Dict:
        """Get the build configuration for serialization.

        Returns:
            Dictionary containing build configuration
        """
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict) -> None:
        """Build the layer from configuration.

        Args:
            config: Dictionary containing build configuration
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

    @classmethod
    def from_config(cls, config: Dict) -> 'ConvNextV2Block1D':
        """Creates a layer from its config.

        Args:
            config: Dictionary containing layer configuration

        Returns:
            ConvNextV2Block1D instance
        """
        from copy import deepcopy

        # Make a copy of the config to avoid modifying the original
        config_copy = deepcopy(config)

        # Extract the conv_config dictionary
        conv_config_dict = config_copy.pop("conv_config", {})

        # Handle kernel_regularizer deserialization if present
        if "kernel_regularizer" in conv_config_dict and conv_config_dict["kernel_regularizer"] is not None:
            conv_config_dict["kernel_regularizer"] = keras.regularizers.deserialize(
                conv_config_dict["kernel_regularizer"]
            )

        # Recreate the ConvNextV2Config1D object
        conv_config = ConvNextV2Config1D(**conv_config_dict)

        # Create the ConvNextV2Block1D with the recreated config
        return cls(conv_config=conv_config, **config_copy)


# ---------------------------------------------------------------------
# Utility functions for creating ConvNextV2 1D blocks
# ---------------------------------------------------------------------

def create_convnext_v2_block_1d(
        filters: int,
        kernel_size: int = 7,
        strides: int = 1,
        activation: str = "gelu",
        dropout_rate: float = 0.0,
        spatial_dropout_rate: float = 0.0,
        use_gamma: bool = False,
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        use_softorthonormal_regularizer: bool = False,
        **kwargs
) -> ConvNextV2Block1D:
    """Utility function to create a ConvNextV2Block1D with common parameters.

    Args:
        filters: Number of output filters
        kernel_size: Size of the convolution kernel (default: 7)
        strides: Convolution stride (default: 1)
        activation: Activation function (default: "gelu")
        dropout_rate: Dropout rate (default: 0.0)
        spatial_dropout_rate: Spatial dropout rate (default: 0.0)
        use_gamma: Whether to use learnable multiplier (default: False)
        kernel_regularizer: Optional kernel regularizer
        use_softorthonormal_regularizer: Whether to use soft orthonormal regularizer
        **kwargs: Additional arguments passed to ConvNextV2Block1D

    Returns:
        ConvNextV2Block1D instance

    Example:
        ```python
        # Create a simple ConvNextV2 1D block
        block = create_convnext_v2_block_1d(
            filters=128,
            kernel_size=11,
            dropout_rate=0.1
        )

        # Use in a model
        model = keras.Sequential([
            keras.layers.Input(shape=(1000, 64)),
            block,
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dense(10, activation='softmax')
        ])
        ```
    """
    config = ConvNextV2Config1D(
        kernel_size=kernel_size,
        filters=filters,
        strides=strides,
        activation=activation,
        kernel_regularizer=kernel_regularizer
    )

    return ConvNextV2Block1D(
        conv_config=config,
        dropout_rate=dropout_rate,
        spatial_dropout_rate=spatial_dropout_rate,
        use_gamma=use_gamma,
        use_softorthonormal_regularizer=use_softorthonormal_regularizer,
        **kwargs
    )

# ---------------------------------------------------------------------
