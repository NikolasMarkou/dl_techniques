"""
MobileNetV3: Efficient Mobile Networks with Hardware-Aware NAS
==============================================================

A complete implementation of MobileNetV3 architecture built with the flexible
Universal Inverted Bottleneck (UIB) layer. This version follows modern
Keras 3 best practices for custom models with proper serialization support.

Based on: "Searching for MobileNetV3"
Paper: https://arxiv.org/abs/1905.02244

Key Features:
------------
- Universal Inverted Bottleneck blocks configured to replicate MobileNetV3's design
- Squeeze-and-Excite attention modules (via UIB)
- Hard-swish (h-swish) activation for efficiency
- Optimized first and last layers
- Support for Large and Small variants
- Complete variant configurations

Architecture Overview:
---------------------
MobileNetV3 consists of:
1. **Stem**: Initial convolution with hard-swish
2. **Body**: Stack of `UniversalInvertedBottleneck` blocks with optional SE
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
from dl_techniques.layers.universal_inverted_bottleneck import UniversalInvertedBottleneck

# ---------------------------------------------------------------------
# MobileNetV3 Model
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class MobileNetV3(keras.Model):
    """MobileNetV3 model implemented with Universal Inverted Bottleneck blocks.

    A highly efficient mobile architecture discovered through platform-aware NAS
    and optimized with NetAdapt, featuring hard-swish activation and an efficient
    last stage design. This implementation leverages the `UniversalInvertedBottleneck`
    layer for its core building blocks.

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
        (240, 80, 3, 2, False, "hard_swish"),  # 14x14
        (200, 80, 3, 1, False, "hard_swish"),
        (184, 80, 3, 1, False, "hard_swish"),
        (184, 80, 3, 1, False, "hard_swish"),
        (480, 112, 3, 1, True, "hard_swish"),
        (672, 112, 3, 1, True, "hard_swish"),
        # Stage 5
        (672, 160, 5, 2, True, "hard_swish"),  # 7x7
        (960, 160, 5, 1, True, "hard_swish"),
        (960, 160, 5, 1, True, "hard_swish"),
    ]

    SMALL_CONFIG = [
        # exp_size, out_channels, kernel_size, stride, use_se, activation
        # Stage 1
        (16, 16, 3, 2, True, "relu"),  # 56x56
        # Stage 2
        (72, 24, 3, 2, False, "relu"),  # 28x28
        (88, 24, 3, 1, False, "relu"),
        # Stage 3
        (96, 40, 5, 2, True, "hard_swish"),  # 14x14
        (240, 40, 5, 1, True, "hard_swish"),
        (240, 40, 5, 1, True, "hard_swish"),
        (120, 48, 5, 1, True, "hard_swish"),
        (144, 48, 5, 1, True, "hard_swish"),
        # Stage 4
        (288, 96, 5, 2, True, "hard_swish"),  # 7x7
        (576, 96, 5, 1, True, "hard_swish"),
        (576, 96, 5, 1, True, "hard_swish"),
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
        self.input_shape_config = input_shape or (224, 224, 3)

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

        # --- Build the model layers ---

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
        self.stem_activation = HardSwish(name="stem_hard_swish")

        # Build inverted residual blocks using UniversalInvertedBottleneck
        self.blocks = []
        in_channels = first_filter

        for i, (exp_size, out_size, kernel, stride, use_se, activation) in enumerate(self.block_configs):
            # Apply width multiplier
            exp_channels = make_divisible(exp_size * self.width_multiplier)
            out_channels = make_divisible(out_size * self.width_multiplier)

            # Calculate expansion factor required by UIB
            if in_channels == 0:
                raise ValueError("Input channels cannot be zero.")
            expansion_factor = exp_channels // in_channels

            block = UniversalInvertedBottleneck(
                filters=out_channels,
                expansion_factor=expansion_factor,
                kernel_size=kernel,
                stride=stride,
                use_squeeze_excitation=use_se,
                activation_type=activation,
                normalization_type='batch_norm',
                use_bias=False,
                use_dw1=True,  # Standard inverted bottleneck structure
                use_dw2=False,
                se_ratio=4.0,  # MobileNetV3 uses a fixed SE ratio of 4
                se_activation='relu',  # Activation before expansion in SE
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
        self.last_activation = HardSwish(name="last_hard_swish")

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
            self.head_activation = HardSwish(name="head_hard_swish")
            self.dropout = layers.Dropout(dropout_rate, name="dropout")

            # Classifier
            self.classifier = layers.Dense(
                num_classes,
                activation='softmax',  # Add softmax for classification
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
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "variant": self.variant,
            "width_multiplier": self.width_multiplier,
            "dropout_rate": self.dropout_rate,
            "weight_decay": self.weight_decay,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "include_top": self.include_top,
            "input_shape": self.input_shape_config,
        })
        return config

    def summary(self, **kwargs):
        """Print model summary with additional information."""
        # Build the model if it hasn't been built yet
        if not self.built:
            input_tensor = keras.Input(shape=self.input_shape_config)
            self.build(input_tensor.shape)

        super().summary(**kwargs)

        # Print additional model information
        total_params = self.count_params()

        logger.info("MobileNetV3 Configuration:")
        logger.info(f"  - Variant: {self.variant}")
        logger.info(f"  - Input shape: {self.input_shape_config}")
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

# ---------------------------------------------------------------------
