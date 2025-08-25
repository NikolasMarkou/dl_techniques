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

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class ConvNextV1Block(keras.layers.Layer):
    """
    Implementation of ConvNext block with modern best practices.

    This layer implements the ConvNext block architecture from "A ConvNet for the 2020s"
    (Liu et al., 2022). The block uses depthwise convolutions with large kernels followed
    by an inverted bottleneck design with LayerNormalization.

    Key architectural features:
    - Depthwise convolution for spatial feature extraction
    - LayerNormalization following the normalization strategy
    - Inverted bottleneck MLP with 4x expansion factor
    - Optional learnable scaling (gamma) for feature calibration
    - Configurable dropout strategies for regularization

    Mathematical formulation:
        x = DepthwiseConv(input)
        x = LayerNorm(x)
        x = Conv1x1_expand(x)  # 4x expansion
        x = GELU(x)
        x = Dropout(x)
        x = Conv1x1_reduce(x)  # back to original channels
        output = gamma * x

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
            applied after activation. If None or 0, no dropout is applied.
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
        dropout: Dropout layer for regularization.
        spatial_dropout: SpatialDropout2D layer for structured regularization.
        conv_3: Second pointwise Conv2D layer (reduction).
        gamma: LearnableMultiplier for channel-wise scaling (if use_gamma=True).

    Example:
        ```python
        # Basic usage
        block = ConvNextV1Block(kernel_size=7, filters=64)

        # Advanced configuration
        block = ConvNextV1Block(
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
        x = ConvNextV1Block(kernel_size=7, filters=64)(inputs)
        outputs = keras.layers.Dense(1000)(keras.layers.GlobalAveragePooling2D()(x))
        model = keras.Model(inputs, outputs)
        ```

    References:
        - A ConvNet for the 2020s, Liu et al., 2022
        - https://arxiv.org/abs/2201.03545

    Raises:
        ValueError: If kernel_size or filters is not positive.
        ValueError: If dropout rates are not between 0 and 1.

    Note:
        This implementation follows the exact specifications from the original paper,
        including the 4x expansion factor, TruncatedNormal initialization, and
        LayerNormalization placement.
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
            activation: Union[str, keras.activations.Activation] = "gelu",
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
            conv_params["kernel_regularizer"] = SoftOrthonormalConstraintRegularizer()

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
    def from_config(cls, config: Dict[str, Any]) -> "ConvNextV1Block":
        """
        Create layer from configuration dictionary.

        Args:
            config: Configuration dictionary.

        Returns:
            ConvNextV1Block instance.
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
