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
"""

import copy
import keras
from typing import Optional, Dict, Union, Tuple, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .layer_scale import LearnableMultiplier
from .norms.global_response_norm import GlobalResponseNormalization
from ..constraints.value_range_constraint import ValueRangeConstraint
from ..regularizers.soft_orthogonal import SoftOrthonormalConstraintRegularizer

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class ConvNextV2Block(keras.layers.Layer):
    """
    Implementation of ConvNextV2 block with modern best practices.

    This layer implements the ConvNextV2 block architecture from "ConvNeXt V2:
    Co-designing and Scaling ConvNets with Masked Autoencoders" (Woo et al., 2023).
    The main improvement over V1 is the addition of Global Response Normalization (GRN)
    which enhances inter-channel feature competition and improves representation learning.

    Key architectural features:
    - Depthwise convolution for spatial feature extraction
    - LayerNormalization following the proper normalization strategy
    - Inverted bottleneck MLP with 4x expansion factor
    - Global Response Normalization (GRN) - the key innovation in V2
    - Optional learnable scaling (gamma) for feature calibration
    - Configurable dropout strategies for regularization

    Mathematical formulation:
        x = DepthwiseConv(input)
        x = LayerNorm(x)
        x = Conv1x1_expand(x)  # 4x expansion
        x = GELU(x)
        x = GRN(x)  # Global Response Normalization - V2 innovation
        x = Dropout(x)
        x = Conv1x1_reduce(x)  # back to original channels
        output = gamma * x

    Global Response Normalization (GRN):
        The GRN layer computes:
        1. L2 norm across spatial dimensions for each channel
        2. Normalizes by the mean of the L2 norm
        3. Applies learnable scaling (gamma) and bias (beta)
        4. Enhances inter-channel feature competition

    Args:
        kernel_size: Integer or tuple of integers, size of the depthwise convolution kernel.
            For square kernels, can be a single integer. Must be positive.
        filters: Integer, number of output filters/channels. Must be positive.
        activation: String or callable, activation function to use in the MLP.
            Supports standard Keras activation names. Defaults to 'gelu'.
        kernel_regularizer: Optional regularizer for convolution kernels.
            Applied to all convolutional layers in the block.
        use_bias: Boolean, whether to include bias terms in convolutions.
            Defaults to True.
        dropout_rate: Optional float between 0 and 1, standard dropout rate
            applied after GRN. If None or 0, no dropout is applied.
        spatial_dropout_rate: Optional float between 0 and 1, spatial dropout rate
            for structured regularization. If None or 0, no spatial dropout is applied.
        use_gamma: Boolean, whether to use learnable scaling (gamma multiplier).
            When True, applies channel-wise learnable scaling. Defaults to True.
        use_softorthonormal_regularizer: Boolean, whether to apply soft orthonormal
            regularization to convolutional kernels. Defaults to False.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, channels)`

    Output shape:
        4D tensor with shape: `(batch_size, height, width, filters)`

    Attributes:
        conv_1: DepthwiseConv2D layer for spatial feature extraction.
        norm: LayerNormalization layer for feature normalization.
        conv_2: First pointwise Conv2D layer (expansion).
        activation_layer: Activation layer (GELU by default).
        grn: GlobalResponseNormalization layer - key V2 innovation.
        dropout: Dropout layer for regularization.
        spatial_dropout: SpatialDropout2D layer for structured regularization.
        conv_3: Second pointwise Conv2D layer (reduction).
        gamma: LearnableMultiplier for channel-wise scaling (if use_gamma=True).

    Example:
        ```python
        # Basic usage
        block = ConvNextV2Block(kernel_size=7, filters=64)

        # Advanced configuration
        block = ConvNextV2Block(
            kernel_size=7,
            filters=128,
            activation='gelu',
            kernel_regularizer=keras.regularizers.L2(0.01),
            dropout_rate=0.1,
            spatial_dropout_rate=0.05,
            use_gamma=True,
            use_softorthonormal_regularizer=False
        )

        # In a model
        inputs = keras.Input(shape=(224, 224, 3))
        x = ConvNextV2Block(kernel_size=7, filters=64)(inputs)
        outputs = keras.layers.Dense(1000)(keras.layers.GlobalAveragePooling2D()(x))
        model = keras.Model(inputs, outputs)
        ```

    References:
        - ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders,
          Woo et al., 2023
        - https://arxiv.org/abs/2301.00808

    Raises:
        ValueError: If kernel_size or filters is not positive.
        ValueError: If dropout rates are not between 0 and 1.

    Note:
        The key difference from ConvNextV1 is the Global Response Normalization (GRN)
        layer, which enhances feature competition and improves representation learning.
        This implementation follows the exact specifications from the original paper.
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
    STRIDES = (1, 1)  # Strides for depthwise convolution

    def __init__(
            self,
            kernel_size: Union[int, Tuple[int, int]],
            filters: int,
            activation: Union[str, keras.layers.Activation] = "gelu",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            use_bias: bool = True,
            dropout_rate: Optional[float] = 0.0,
            spatial_dropout_rate: Optional[float] = 0.0,
            use_gamma: bool = True,
            use_softorthonormal_regularizer: bool = False,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if isinstance(kernel_size, int):
            if kernel_size <= 0:
                raise ValueError(f"kernel_size must be positive, got {kernel_size}")
        elif isinstance(kernel_size, (tuple, list)):
            if len(kernel_size) != 2 or any(k <= 0 for k in kernel_size):
                raise ValueError(f"kernel_size tuple must have 2 positive values, got {kernel_size}")

        if filters <= 0:
            raise ValueError(f"filters must be positive, got {filters}")

        if dropout_rate is not None and not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")

        if spatial_dropout_rate is not None and not (0.0 <= spatial_dropout_rate <= 1.0):
            raise ValueError(f"spatial_dropout_rate must be between 0 and 1, got {spatial_dropout_rate}")

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

        # CREATE all sub-layers in __init__ (following modern Keras 3 pattern)
        # They will be built explicitly in build() method for robust serialization

        # Depthwise convolution layer
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
            name="depthwise_conv"
        )

        # Normalization layer
        self.norm = keras.layers.LayerNormalization(
            epsilon=self.LAYERNORM_EPSILON,
            center=self.use_bias,
            scale=True,
            name="layer_norm"
        )

        # Prepare convolution parameters
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

        # Apply soft orthonormal regularizer if requested
        if self.use_softorthonormal_regularizer:
            conv_params["kernel_regularizer"] = (
                SoftOrthonormalConstraintRegularizer(
                    lambda_coefficient=1.0,
                    l1_coefficient=1-5,
                    l2_coefficient=0.0,
                    use_matrix_scaling=True
                )
            )

        # First pointwise convolution (expansion)
        self.conv_2 = keras.layers.Conv2D(
            filters=self.filters * self.EXPANSION_FACTOR,
            name="expand_conv",
            **conv_params
        )

        # Second pointwise convolution (reduction)
        self.conv_3 = keras.layers.Conv2D(
            filters=self.filters,
            name="reduce_conv",
            **conv_params
        )

        # Activation layer
        self.activation_layer = keras.layers.Activation(
            self.activation_name,
            name="activation"
        )

        # Global Response Normalization (GRN) - key feature of ConvNeXt V2
        self.grn = GlobalResponseNormalization(
            eps=self.GRN_EPSILON,
            name="global_response_norm"
        )

        # Dropout layers
        if self.dropout_rate is not None and self.dropout_rate > 0:
            self.dropout = keras.layers.Dropout(
                self.dropout_rate,
                name="dropout"
            )
        else:
            self.dropout = keras.layers.Lambda(
                lambda x: x,
                name="no_dropout"
            )

        if self.spatial_dropout_rate is not None and self.spatial_dropout_rate > 0:
            self.spatial_dropout = keras.layers.SpatialDropout2D(
                self.spatial_dropout_rate,
                name="spatial_dropout"
            )
        else:
            self.spatial_dropout = keras.layers.Lambda(
                lambda x: x,
                name="no_spatial_dropout"
            )

        # Learnable multiplier (gamma scaling)
        if self.use_gamma:
            self.gamma = LearnableMultiplier(
                multiplier_type="CHANNEL",
                regularizer=keras.regularizers.L2(self.GAMMA_L2_REGULARIZATION),
                initializer=keras.initializers.Constant(self.GAMMA_INITIAL_VALUE),
                constraint=ValueRangeConstraint(
                    min_value=self.GAMMA_MIN_VALUE,
                    max_value=self.GAMMA_MAX_VALUE
                ),
                name="gamma_scale"
            )
        else:
            self.gamma = keras.layers.Lambda(
                lambda x: x,
                name="no_gamma"
            )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and all its sub-layers.

        CRITICAL: Explicitly build each sub-layer for robust serialization.
        This ensures all weight variables exist before weight restoration.

        Args:
            input_shape: Shape tuple of the input tensor.
        """
        # Build sub-layers in computational order for proper shape propagation

        # 1. Depthwise convolution
        self.conv_1.build(input_shape)

        # Shape after depthwise conv (same as input since stride=1, padding='same')
        post_depthwise_shape = input_shape

        # 2. Layer normalization
        self.norm.build(post_depthwise_shape)

        # 3. First pointwise convolution (expansion)
        self.conv_2.build(post_depthwise_shape)

        # Shape after expansion
        expansion_shape = list(post_depthwise_shape)
        expansion_shape[-1] = self.filters * self.EXPANSION_FACTOR
        expansion_shape = tuple(expansion_shape)

        # 4. Activation layer
        self.activation_layer.build(expansion_shape)

        # 5. Global Response Normalization (GRN) - key V2 feature
        self.grn.build(expansion_shape)

        # 6. Dropout layers
        self.dropout.build(expansion_shape)
        self.spatial_dropout.build(expansion_shape)

        # 7. Second pointwise convolution (reduction)
        self.conv_3.build(expansion_shape)

        # Final shape after reduction
        final_shape = list(expansion_shape)
        final_shape[-1] = self.filters
        final_shape = tuple(final_shape)

        # 8. Gamma scaling
        self.gamma.build(final_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass of the ConvNextV2 block.

        Args:
            inputs: Input tensor of shape (batch_size, height, width, channels).
            training: Boolean indicating whether in training mode.

        Returns:
            Output tensor of shape (batch_size, height, width, filters).
        """
        # 1. Depthwise convolution
        x = self.conv_1(inputs, training=training)

        # 2. Layer normalization
        x = self.norm(x, training=training)

        # 3. First pointwise convolution (expansion)
        x = self.conv_2(x, training=training)

        # 4. Activation
        x = self.activation_layer(x, training=training)

        # 5. Global Response Normalization - key ConvNeXt V2 feature
        x = self.grn(x, training=training)

        # 6. Apply dropout layers
        x = self.dropout(x, training=training)
        x = self.spatial_dropout(x, training=training)

        # 7. Second pointwise convolution (reduction)
        x = self.conv_3(x, training=training)

        # 8. Apply learnable scaling
        x = self.gamma(x, training=training)

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the ConvNextV2 block.

        Args:
            input_shape: Shape tuple representing input shape
                (batch_size, height, width, channels).

        Returns:
            Output shape tuple (batch_size, height, width, filters).

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
        # Height and width remain the same due to stride=1 and padding='same'
        return (batch_size, height, width, self.filters)

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration for serialization.

        Returns ALL constructor parameters for proper serialization.

        Returns:
            Dictionary containing the layer configuration.
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

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ConvNextV2Block":
        """
        Create layer from configuration dictionary.

        Args:
            config: Configuration dictionary.

        Returns:
            ConvNextV2Block instance.
        """
        # Make a copy to avoid modifying the original config
        config_copy = config.copy()

        # Deserialize the kernel_regularizer if it exists
        if "kernel_regularizer" in config_copy and config_copy["kernel_regularizer"] is not None:
            config_copy["kernel_regularizer"] = keras.regularizers.deserialize(
                config_copy["kernel_regularizer"]
            )

        return cls(**config_copy)

# ---------------------------------------------------------------------