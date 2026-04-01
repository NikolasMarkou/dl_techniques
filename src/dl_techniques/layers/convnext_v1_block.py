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
"""

import copy
import keras
from typing import Optional, Dict, Union, Tuple, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .layer_scale import LearnableMultiplier
from ..constraints.value_range_constraint import ValueRangeConstraint
from ..regularizers.soft_orthogonal import SoftOrthonormalConstraintRegularizer
from ..initializers.hypersphere_orthogonal_initializer import OrthogonalHypersphereInitializer

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class ConvNextV1Block(keras.layers.Layer):
    """ConvNeXt V1 block with depthwise convolution and inverted bottleneck MLP.

    Implements the block from "A ConvNet for the 2020s" (Liu et al., 2022).
    The computation is ``x = DepthwiseConv -> LayerNorm -> Conv1x1(4x expand)
    -> GELU -> Dropout -> Conv1x1(reduce) -> gamma * x``, using large-kernel
    depthwise convolutions and an inverted bottleneck design with a 4x
    expansion factor in the pointwise MLP.

    **Architecture Overview:**

    .. code-block:: text

        ┌───────────────────────────────────┐
        │  Input (B, H, W, C)               │
        └────────────────┬──────────────────┘
                         ▼
        ┌───────────────────────────────────┐
        │  DepthwiseConv2D (KxK, same)      │
        └────────────────┬──────────────────┘
                         ▼
        ┌───────────────────────────────────┐
        │  LayerNormalization               │
        └────────────────┬──────────────────┘
                         ▼
        ┌───────────────────────────────────┐
        │  Conv2D 1x1 (expand: F*4)         │
        └────────────────┬──────────────────┘
                         ▼
        ┌───────────────────────────────────┐
        │  GELU Activation                  │
        └────────────────┬──────────────────┘
                         ▼
        ┌───────────────────────────────────┐
        │  Dropout / SpatialDropout         │
        └────────────────┬──────────────────┘
                         ▼
        ┌───────────────────────────────────┐
        │  Conv2D 1x1 (reduce: F)           │
        └────────────────┬──────────────────┘
                         ▼
        ┌───────────────────────────────────┐
        │  gamma * x  (learnable scale)     │
        └────────────────┬──────────────────┘
                         ▼
        ┌───────────────────────────────────┐
        │  Output (B, H, W, F)              │
        └───────────────────────────────────┘

    :param kernel_size: Size of the depthwise convolution kernel. Must be positive.
    :type kernel_size: Union[int, Tuple[int, int]]
    :param filters: Number of output filters/channels. Must be positive.
    :type filters: int
    :param activation: Activation function name. Defaults to ``'gelu'``.
    :type activation: Union[str, keras.layers.Activation]
    :param kernel_regularizer: Optional regularizer for convolution kernels.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param use_bias: Whether to include bias terms. Defaults to ``True``.
    :type use_bias: bool
    :param dropout_rate: Standard dropout rate. Defaults to 0.0.
    :type dropout_rate: Optional[float]
    :param spatial_dropout_rate: Spatial dropout rate. Defaults to 0.0.
    :type spatial_dropout_rate: Optional[float]
    :param use_gamma: Whether to use learnable gamma scaling. Defaults to ``True``.
    :type use_gamma: bool
    :param use_softorthonormal_regularizer: Whether to apply soft orthonormal
        regularization. Defaults to ``False``.
    :type use_softorthonormal_regularizer: bool
    :param kwargs: Additional keyword arguments for the Layer base class.
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
                    l1_coefficient=1e-5,
                    l2_coefficient=0.0,
                    use_matrix_scaling=True
                )
            )

            conv_params["kernel_initializer"] = OrthogonalHypersphereInitializer()

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

            :param input_shape: Shape tuple of the input tensor.
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

        # 5. Dropout layers
        self.dropout.build(expansion_shape)
        self.spatial_dropout.build(expansion_shape)

        # 6. Second pointwise convolution (reduction)
        self.conv_3.build(expansion_shape)

        # Final shape after reduction
        final_shape = list(expansion_shape)
        final_shape[-1] = self.filters
        final_shape = tuple(final_shape)

        # 7. Gamma scaling
        self.gamma.build(final_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass of the ConvNext block.

            :param inputs: Input tensor of shape (batch_size, height, width, channels).
            :param training: Boolean indicating whether in training mode.

            :return: Output tensor of shape (batch_size, height, width, filters).
        """
        # 1. Depthwise convolution
        x = self.conv_1(inputs, training=training)

        # 2. Layer normalization
        x = self.norm(x, training=training)

        # 3. First pointwise convolution (expansion)
        x = self.conv_2(x, training=training)

        # 4. Activation
        x = self.activation_layer(x, training=training)

        # 5. Apply dropout layers
        x = self.dropout(x, training=training)
        x = self.spatial_dropout(x, training=training)

        # 6. Second pointwise convolution (reduction)
        x = self.conv_3(x, training=training)

        # 7. Apply learnable scaling
        x = self.gamma(x, training=training)

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the ConvNext block.

            :param input_shape: Shape tuple representing input shape
                (batch_size, height, width, channels).

            :return: Output shape tuple (batch_size, height, width, filters).

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

            :return: Dictionary containing the layer configuration.
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
    def from_config(cls, config: Dict[str, Any]) -> "ConvNextV1Block":
        """
        Create layer from configuration dictionary.

            :param config: Configuration dictionary.

            :return: ConvNextV1Block instance.
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
