"""
MobileNetV3: Efficient Mobile Networks with Hardware-Aware NAS
==============================================================

A complete implementation of MobileNetV3 architecture with inverted residual blocks,
squeeze-and-excite modules, and hard-swish activation. This version follows modern
Keras 3 best practices for custom models.

Based on: "Searching for MobileNetV3"
Paper: https://arxiv.org/abs/1905.02244

Key Features:
------------
- Inverted residual blocks with linear bottlenecks
- Squeeze-and-Excite attention modules
- Hard-swish (h-swish) activation for efficiency
- Optimized first and last layers
- Support for Large and Small variants
- Complete variant configurations

Architecture Overview:
---------------------
MobileNetV3 consists of:
1. **Stem**: Initial convolution with hard-swish
2. **Body**: Stack of inverted residual blocks with optional SE
3. **Head**: Efficient last stage with optimized structure

Model Variants:
--------------
- MobileNetV3-Large: High accuracy model for powerful devices
- MobileNetV3-Small: Lightweight model for resource-constrained devices

Usage Examples:
--------------
```python
# ImageNet model (224x224 input)
model = MobileNetV3.from_variant("large", num_classes=1000)

# CIFAR-10 model (32x32 input)
model = MobileNetV3.from_variant("small", num_classes=10, input_shape=(32, 32, 3))

# Custom configuration
model = MobileNetV3(num_classes=100, variant="large", width_multiplier=0.75)
```
"""

import keras
from keras import layers, regularizers, initializers
from typing import Tuple, Optional, Dict, Any, Literal, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.activations.hard_swish import HardSwish
from dl_techniques.layers.activations.hard_sigmoid import HardSigmoid

# ---------------------------------------------------------------------
# Squeeze and Excitation Module
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class SqueezeExcitationV3(layers.Layer):
    """Squeeze-and-Excitation module for MobileNetV3.

    Applies channel-wise attention using global pooling and two FC layers
    with hard-sigmoid activation for efficiency on mobile devices.

    Args:
        filters: Integer, number of input/output filters.
        se_ratio: Float, reduction ratio for squeeze operation.
            The squeeze layer will have filters // se_ratio channels.
        use_bias: Boolean, whether to use bias in convolution layers.
        kernel_initializer: Initializer for convolution kernels.
        kernel_regularizer: Regularizer for convolution kernels.
        **kwargs: Additional arguments for Layer base class.

    Example:
        >>> se = SqueezeExcitationV3(filters=96, se_ratio=4)
        >>> x = keras.random.normal((2, 7, 7, 96))
        >>> output = se(x)  # Shape: (2, 7, 7, 96)
    """

    def __init__(
            self,
            filters: int,
            se_ratio: float = 4.0,
            use_bias: bool = True,
            kernel_initializer: Union[str, initializers.Initializer] = "glorot_uniform",
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.filters = filters
        self.se_ratio = se_ratio
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = kernel_regularizer

        # Calculate squeeze dimension (fixed at 1/4 of expansion in MobileNetV3)
        self.squeeze_filters = max(1, int(filters // se_ratio))

        # Create layers in __init__
        self.global_pool = layers.GlobalAveragePooling2D(keepdims=True, name="se_pool")
        self.fc1 = layers.Conv2D(
            self.squeeze_filters,
            kernel_size=1,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="se_reduce"
        )
        self.relu = layers.ReLU(name="se_relu")
        self.fc2 = layers.Conv2D(
            filters,
            kernel_size=1,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="se_expand"
        )
        self.hard_sigmoid = HardSigmoid(name="se_hsigmoid")

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Apply squeeze-and-excitation."""
        x = self.global_pool(inputs)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.hard_sigmoid(x)
        return inputs * x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Output shape is same as input shape."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "se_ratio": self.se_ratio,
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
        })
        return config


# ---------------------------------------------------------------------
# Inverted Residual Block
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class InvertedResidualV3(layers.Layer):
    """MobileNetV3 inverted residual bottleneck block.

    Implements the inverted residual structure with:
    - 1x1 expansion convolution (if expansion != 1)
    - Depthwise convolution
    - Optional squeeze-and-excitation
    - 1x1 projection convolution
    - Residual connection when stride=1 and in_filters=out_filters

    Args:
        expansion_filters: Integer, number of channels in expansion.
        out_filters: Integer, number of output channels.
        kernel_size: Integer, size of depthwise convolution kernel.
        stride: Integer, stride for depthwise convolution.
        use_se: Boolean, whether to use squeeze-and-excitation.
        activation: String, activation function ("relu" or "hswish").
        use_residual: Boolean, whether to use residual connection.
        kernel_initializer: Initializer for convolution kernels.
        kernel_regularizer: Regularizer for convolution kernels.
        **kwargs: Additional arguments for Layer base class.

    Example:
        >>> block = InvertedResidualV3(
        ...     expansion_filters=96,
        ...     out_filters=40,
        ...     kernel_size=5,
        ...     stride=2,
        ...     use_se=True,
        ...     activation="relu"
        ... )
        >>> x = keras.random.normal((2, 56, 56, 24))
        >>> output = block(x)  # Shape: (2, 28, 28, 40)
    """

    def __init__(
            self,
            expansion_filters: int,
            out_filters: int,
            kernel_size: int = 3,
            stride: int = 1,
            use_se: bool = False,
            activation: Literal["relu", "hswish"] = "relu",
            use_residual: bool = True,
            kernel_initializer: Union[str, initializers.Initializer] = "glorot_uniform",
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.expansion_filters = expansion_filters
        self.out_filters = out_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.use_se = use_se
        self.activation = activation
        self.use_residual = use_residual
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = kernel_regularizer

        # Choose activation function
        if activation == "relu":
            self.activation_fn = layers.ReLU()
        elif activation == "hswish":
            self.activation_fn = HardSwish()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Build sub-layers - will be built properly in build() method
        self.expand_conv = None
        self.expand_bn = None
        self.expand_activation = None

        self.depthwise_conv = layers.DepthwiseConv2D(
            kernel_size=kernel_size,
            strides=stride,
            padding='same',
            use_bias=False,
            depthwise_initializer=kernel_initializer,
            depthwise_regularizer=kernel_regularizer,
            name="depthwise"
        )
        self.depthwise_bn = layers.BatchNormalization(name="depthwise_bn")
        self.depthwise_activation = self.activation_fn

        if use_se:
            self.se = SqueezeExcitationV3(
                filters=expansion_filters,
                se_ratio=4.0,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name="se"
            )
        else:
            self.se = None

        self.project_conv = layers.Conv2D(
            out_filters,
            kernel_size=1,
            strides=1,
            padding='same',
            use_bias=False,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name="project"
        )
        self.project_bn = layers.BatchNormalization(name="project_bn")
        # Note: No activation after projection (linear bottleneck)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer and sub-layers."""
        in_filters = input_shape[-1]

        # Determine if we need expansion
        self.use_expansion = (in_filters != self.expansion_filters)

        if self.use_expansion:
            self.expand_conv = layers.Conv2D(
                self.expansion_filters,
                kernel_size=1,
                strides=1,
                padding='same',
                use_bias=False,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name="expand"
            )
            self.expand_bn = layers.BatchNormalization(name="expand_bn")
            self.expand_activation = self.activation_fn

            # Build expansion layers
            self.expand_conv.build(input_shape)
            expand_shape = self.expand_conv.compute_output_shape(input_shape)
            self.expand_bn.build(expand_shape)
            depthwise_input_shape = expand_shape
        else:
            depthwise_input_shape = input_shape

        # Build depthwise layers
        self.depthwise_conv.build(depthwise_input_shape)
        depthwise_shape = self.depthwise_conv.compute_output_shape(depthwise_input_shape)
        self.depthwise_bn.build(depthwise_shape)

        # Build SE if used
        if self.se is not None:
            self.se.build(depthwise_shape)

        # Build projection layers
        self.project_conv.build(depthwise_shape)
        project_shape = self.project_conv.compute_output_shape(depthwise_shape)
        self.project_bn.build(project_shape)

        # Determine if residual connection should be used
        self.use_residual_connection = (
                self.use_residual and
                self.stride == 1 and
                in_filters == self.out_filters
        )

        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Forward pass through the block."""
        x = inputs

        # Expansion
        if self.use_expansion:
            x = self.expand_conv(x)
            x = self.expand_bn(x, training=training)
            x = self.expand_activation(x)

        # Depthwise
        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x, training=training)
        x = self.depthwise_activation(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x, training=training)

        # Projection
        x = self.project_conv(x)
        x = self.project_bn(x, training=training)

        # Residual connection
        if self.use_residual_connection:
            x = layers.Add()([inputs, x])

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape."""
        batch_size, height, width, _ = input_shape

        # Account for stride
        if self.stride > 1:
            height = height // self.stride if height else None
            width = width // self.stride if width else None

        return (batch_size, height, width, self.out_filters)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "expansion_filters": self.expansion_filters,
            "out_filters": self.out_filters,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "use_se": self.use_se,
            "activation": self.activation,
            "use_residual": self.use_residual,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
        })
        return config


# ---------------------------------------------------------------------
# MobileNetV3 Model
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class MobileNetV3(keras.Model):
    """MobileNetV3 model implementation with inverted residual blocks and SE.

    A highly efficient mobile architecture discovered through platform-aware NAS
    and optimized with NetAdapt, featuring hard-swish activation and an efficient
    last stage design.

    Args:
        num_classes: Integer, number of output classes for classification.
        variant: String, model variant ("large" or "small").
        width_multiplier: Float, multiplier for the number of filters.
        dropout_rate: Float, dropout rate for regularization.
        weight_decay: Float, L2 regularization factor.
        kernel_initializer: String or initializer, weight initialization.
        include_top: Boolean, whether to include the classification head.
        input_shape: Tuple, input shape. If None, defaults to (224, 224, 3).
        **kwargs: Additional keyword arguments for Model base class.

    Example:
        >>> # Create MobileNetV3-Large for ImageNet
        >>> model = MobileNetV3(num_classes=1000, variant="large")
        >>>
        >>> # Create MobileNetV3-Small for CIFAR-10
        >>> model = MobileNetV3(
        ...     num_classes=10,
        ...     variant="small",
        ...     input_shape=(32, 32, 3)
        ... )
        >>>
        >>> # With custom width multiplier
        >>> model = MobileNetV3(
        ...     num_classes=100,
        ...     variant="large",
        ...     width_multiplier=0.75
        ... )
    """

    # Model configurations from the paper
    LARGE_CONFIG = [
        # exp_size, out_channels, kernel_size, stride, use_se, activation
        # Stage 1
        (16, 16, 3, 1, False, "relu"),  # 112x112
        # Stage 2
        (64, 24, 3, 2, False, "relu"),  # 56x56
        (72, 24, 3, 1, False, "relu"),
        # Stage 3
        (72, 40, 5, 2, True, "relu"),  # 28x28
        (120, 40, 5, 1, True, "relu"),
        (120, 40, 5, 1, True, "relu"),
        # Stage 4
        (240, 80, 3, 2, False, "hswish"),  # 14x14
        (200, 80, 3, 1, False, "hswish"),
        (184, 80, 3, 1, False, "hswish"),
        (184, 80, 3, 1, False, "hswish"),
        (480, 112, 3, 1, True, "hswish"),
        (672, 112, 3, 1, True, "hswish"),
        # Stage 5
        (672, 160, 5, 2, True, "hswish"),  # 7x7
        (960, 160, 5, 1, True, "hswish"),
        (960, 160, 5, 1, True, "hswish"),
    ]

    SMALL_CONFIG = [
        # exp_size, out_channels, kernel_size, stride, use_se, activation
        # Stage 1
        (16, 16, 3, 2, True, "relu"),  # 56x56
        # Stage 2
        (72, 24, 3, 2, False, "relu"),  # 28x28
        (88, 24, 3, 1, False, "relu"),
        # Stage 3
        (96, 40, 5, 2, True, "hswish"),  # 14x14
        (240, 40, 5, 1, True, "hswish"),
        (240, 40, 5, 1, True, "hswish"),
        (120, 48, 5, 1, True, "hswish"),
        (144, 48, 5, 1, True, "hswish"),
        # Stage 4
        (288, 96, 5, 2, True, "hswish"),  # 7x7
        (576, 96, 5, 1, True, "hswish"),
        (576, 96, 5, 1, True, "hswish"),
    ]

    def __init__(
            self,
            num_classes: int = 1000,
            variant: Literal["large", "small"] = "large",
            width_multiplier: float = 1.0,
            dropout_rate: float = 0.2,
            weight_decay: float = 1e-5,
            kernel_initializer: Union[str, initializers.Initializer] = "glorot_uniform",
            include_top: bool = True,
            input_shape: Optional[Tuple[int, int, int]] = None,
            **kwargs
    ):
        super().__init__(**kwargs)

        # Validate variant
        if variant not in ["large", "small"]:
            raise ValueError(f"Unknown variant '{variant}'. Must be 'large' or 'small'.")

        # Validate input shape
        if input_shape and len(input_shape) != 3:
            raise ValueError(f"input_shape must be 3D, got {input_shape}")

        # Store configuration
        self.num_classes = num_classes
        self.variant = variant
        self.width_multiplier = width_multiplier
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.include_top = include_top
        self._input_shape = input_shape or (224, 224, 3)

        # Get configuration for the variant
        if variant == "large":
            self.block_configs = self.LARGE_CONFIG
            self.last_block_filters = 960
            self.last_conv_filters = 1280
        else:  # small
            self.block_configs = self.SMALL_CONFIG
            self.last_block_filters = 576
            self.last_conv_filters = 1024

        # Create regularizer
        self.kernel_regularizer = regularizers.L2(weight_decay) if weight_decay > 0 else None

        # Helper function to make divisible
        def make_divisible(value: float, divisor: int = 8) -> int:
            """Make value divisible by divisor."""
            new_value = max(divisor, int(value + divisor / 2) // divisor * divisor)
            # Make sure that round down does not go down by more than 10%.
            if new_value < 0.9 * value:
                new_value += divisor
            return new_value

        # Build the model layers
        # Initial stem
        first_filter = make_divisible(16 * width_multiplier)
        self.stem_conv = layers.Conv2D(
            first_filter,
            kernel_size=3,
            strides=2,
            padding='same',
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="stem_conv"
        )
        self.stem_bn = layers.BatchNormalization(name="stem_bn")
        self.stem_activation = HardSwish(name="stem_hswish")

        # Build inverted residual blocks
        self.blocks = []
        in_channels = first_filter

        for i, (exp_size, out_channels, kernel, stride, use_se, activation) in enumerate(self.block_configs):
            # Apply width multiplier
            exp_channels = make_divisible(exp_size * width_multiplier)
            out_channels = make_divisible(out_channels * width_multiplier)

            block = InvertedResidualV3(
                expansion_filters=exp_channels,
                out_filters=out_channels,
                kernel_size=kernel,
                stride=stride,
                use_se=use_se,
                activation=activation,
                use_residual=(stride == 1 and in_channels == out_channels),
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"block_{i}"
            )
            self.blocks.append(block)
            in_channels = out_channels

        # Last convolution block (efficient last stage)
        last_block_filters = make_divisible(self.last_block_filters * width_multiplier)
        self.last_conv = layers.Conv2D(
            last_block_filters,
            kernel_size=1,
            strides=1,
            padding='same',
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="last_conv"
        )
        self.last_bn = layers.BatchNormalization(name="last_bn")
        self.last_activation = HardSwish(name="last_hswish")

        # Head
        if self.include_top:
            self.global_pool = layers.GlobalAveragePooling2D(name="global_pool")

            # Final convolution (acts as FC after global pooling)
            last_conv_filters = make_divisible(self.last_conv_filters * width_multiplier)
            self.head_conv = layers.Dense(
                last_conv_filters,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name="head_conv"
            )
            self.head_activation = HardSwish(name="head_hswish")
            self.dropout = layers.Dropout(dropout_rate, name="dropout")

            # Classifier
            self.classifier = layers.Dense(
                num_classes,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name="classifier"
            )

    def call(self, x: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Forward pass through the model."""
        # Stem
        x = self.stem_conv(x)
        x = self.stem_bn(x, training=training)
        x = self.stem_activation(x)

        # Inverted residual blocks
        for block in self.blocks:
            x = block(x, training=training)

        # Last convolution block
        x = self.last_conv(x)
        x = self.last_bn(x, training=training)
        x = self.last_activation(x)

        # Head
        if self.include_top:
            x = self.global_pool(x)
            x = self.head_conv(x)
            x = self.head_activation(x)
            x = self.dropout(x, training=training)
            x = self.classifier(x)

        return x

    @classmethod
    def from_variant(
            cls,
            variant: Literal["large", "small"],
            num_classes: int = 1000,
            input_shape: Optional[Tuple[int, int, int]] = None,
            width_multiplier: float = 1.0,
            **kwargs
    ) -> "MobileNetV3":
        """Create a MobileNetV3 model from a predefined variant.

        Args:
            variant: String, "large" or "small"
            num_classes: Integer, number of output classes
            input_shape: Tuple, input shape. If None, uses (224, 224, 3)
            width_multiplier: Float, multiplier for filter dimensions
            **kwargs: Additional arguments passed to the constructor

        Returns:
            MobileNetV3 model instance

        Example:
            >>> # ImageNet model
            >>> model = MobileNetV3.from_variant("large", num_classes=1000)
            >>> # CIFAR-10 model
            >>> model = MobileNetV3.from_variant("small", num_classes=10, input_shape=(32, 32, 3))
        """
        logger.info(f"Creating MobileNetV3-{variant.capitalize()} model")

        return cls(
            num_classes=num_classes,
            variant=variant,
            width_multiplier=width_multiplier,
            input_shape=input_shape,
            **kwargs
        )

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        config = {
            "num_classes": self.num_classes,
            "variant": self.variant,
            "width_multiplier": self.width_multiplier,
            "dropout_rate": self.dropout_rate,
            "weight_decay": self.weight_decay,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "include_top": self.include_top,
            "input_shape": self._input_shape,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MobileNetV3":
        """Create model from configuration."""
        return cls(**config)

    def summary(self, **kwargs):
        """Print model summary with additional information."""
        # Build the model if it hasn't been built yet
        if not self.built and self._input_shape:
            self.build((None,) + self._input_shape)

        super().summary(**kwargs)

        # Print additional model information
        total_params = self.count_params()

        logger.info("MobileNetV3 Configuration:")
        logger.info(f"  - Variant: {self.variant}")
        logger.info(f"  - Input shape: {self._input_shape}")
        logger.info(f"  - Width multiplier: {self.width_multiplier}")
        logger.info(f"  - Number of blocks: {len(self.blocks)}")
        logger.info(f"  - Include top: {self.include_top}")
        if self.include_top:
            logger.info(f"  - Number of classes: {self.num_classes}")
        logger.info(f"  - Total parameters: {total_params:,}")


# ---------------------------------------------------------------------

def create_mobilenetv3(
        variant: Literal["large", "small"] = "large",
        num_classes: int = 1000,
        input_shape: Optional[Tuple[int, int, int]] = None,
        width_multiplier: float = 1.0,
        pretrained: bool = False,
        **kwargs
) -> MobileNetV3:
    """Convenience function to create MobileNetV3 models.

    Args:
        variant: String, model variant ("large" or "small")
        num_classes: Integer, number of output classes
        input_shape: Tuple, input shape. If None, uses (224, 224, 3)
        width_multiplier: Float, multiplier for filter dimensions
        pretrained: Boolean, whether to load pretrained weights (not implemented)
        **kwargs: Additional arguments passed to the model constructor

    Returns:
        MobileNetV3 model instance

    Example:
        >>> # Create MobileNetV3-Large for ImageNet
        >>> model = create_mobilenetv3("large", num_classes=1000)
        >>>
        >>> # Create MobileNetV3-Small for CIFAR-10
        >>> model = create_mobilenetv3("small", num_classes=10, input_shape=(32, 32, 3))
        >>>
        >>> # Create with custom width multiplier
        >>> model = create_mobilenetv3("large", width_multiplier=0.75)
    """
    if pretrained:
        logger.warning("Pretrained weights are not yet implemented")

    model = MobileNetV3.from_variant(
        variant,
        num_classes=num_classes,
        input_shape=input_shape,
        width_multiplier=width_multiplier,
        **kwargs
    )

    return model

# ------------------------------------------------------------------------