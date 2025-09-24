"""
MobileNetV1: Efficient Convolutional Neural Networks for Mobile Vision Applications
================================================================================

A complete implementation of the original MobileNet architecture using depthwise
separable convolutions. This implementation follows modern Keras 3 best practices
for custom models with proper serialization support.

Based on: "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
Paper: https://arxiv.org/abs/1704.04861

Key Features:
------------
- Depthwise separable convolutions for efficiency
- Width multiplier (α) for model scaling
- Resolution flexibility through input_shape
- Modular design with proper serialization support
- Complete variant support (1.0, 0.75, 0.5, 0.25 width multipliers)

Architecture Overview:
---------------------
MobileNetV1 consists of:
1. **Initial Conv**: Standard 3x3 convolution with stride 2
2. **Depthwise Separable Blocks**: 13 depthwise separable convolution blocks
3. **Global Average Pooling**: Reduces spatial dimensions
4. **Classifier**: Fully connected layer for classification

Usage Examples:
--------------
```python
# Standard MobileNetV1 (α=1.0) for ImageNet
model = MobileNetV1.from_variant("1.0", num_classes=1000)

# Smaller model (α=0.75) for CIFAR-10
model = MobileNetV1.from_variant("0.75", num_classes=10, input_shape=(32, 32, 3))

# Custom configuration
model = MobileNetV1(num_classes=100, width_multiplier=0.5, input_shape=(128, 128, 3))
```
"""

import keras
from keras import layers, regularizers
from typing import Tuple, Optional, Dict, Any, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.depthwise_separable_block import DepthwiseSeparableBlock

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MobileNetV1(keras.Model):
    """MobileNetV1 model implementation with depthwise separable convolutions.

    The original efficient architecture for mobile and embedded vision_heads applications
    using depthwise separable convolutions to drastically reduce computation and
    model size while maintaining good accuracy.

    Args:
        num_classes: Integer, number of output classes for classification
        width_multiplier: Float, multiplier for the number of filters (α).
            Controls model width. Common values: 1.0, 0.75, 0.5, 0.25
        dropout_rate: Float, dropout rate before the classifier
        weight_decay: Float, L2 regularization factor for all layers
        kernel_initializer: String or initializer, weight initialization strategy
        include_top: Boolean, whether to include the classification head
        input_shape: Tuple, input shape. If None, defaults to (224, 224, 3)
        **kwargs: Additional keyword arguments for Model base class

    Raises:
        ValueError: If width_multiplier is not positive
        ValueError: If input_shape is invalid

    Example:
        >>> # Create standard MobileNetV1 for ImageNet
        >>> model = MobileNetV1(num_classes=1000, width_multiplier=1.0)
        >>>
        >>> # Create smaller model for CIFAR-10
        >>> model = MobileNetV1(num_classes=10, width_multiplier=0.5, input_shape=(32, 32, 3))
    """

    # Model variant configurations
    MODEL_VARIANTS = {
        "1.0": {"width_multiplier": 1.0},
        "0.75": {"width_multiplier": 0.75},
        "0.5": {"width_multiplier": 0.5},
        "0.25": {"width_multiplier": 0.25},
    }

    # Architecture definition (filters and strides for each block)
    # Format: (filters, stride)
    ARCHITECTURE = [
        # Initial standard conv is handled separately
        (64, 1),  # Block 1
        (128, 2),  # Block 2
        (128, 1),  # Block 3
        (256, 2),  # Block 4
        (256, 1),  # Block 5
        (512, 2),  # Block 6
        (512, 1),  # Block 7 (repeated 5 times)
        (512, 1),  # Block 8
        (512, 1),  # Block 9
        (512, 1),  # Block 10
        (512, 1),  # Block 11
        (1024, 2),  # Block 12
        (1024, 1),  # Block 13
    ]

    def __init__(
            self,
            num_classes: int = 1000,
            width_multiplier: float = 1.0,
            dropout_rate: float = 0.001,  # MobileNetV1 uses very light dropout
            weight_decay: float = 0.00004,  # Standard weight decay for MobileNet
            kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
            include_top: bool = True,
            input_shape: Optional[Tuple[int, ...]] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if width_multiplier <= 0:
            raise ValueError(f"width_multiplier must be positive, got {width_multiplier}")

        if input_shape is None:
            input_shape = (224, 224, 3)

        if len(input_shape) != 3:
            raise ValueError(f"input_shape must be 3D, got {input_shape}")

        height, width, channels = input_shape
        if channels not in [1, 3]:
            logger.warning(f"Unusual number of channels: {channels}")

        # Store configuration
        self.num_classes = num_classes
        self.width_multiplier = width_multiplier
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.include_top = include_top
        self._input_shape = input_shape

        # Create regularizer
        self.kernel_regularizer = regularizers.L2(weight_decay) if weight_decay > 0 else None

        # Build the model layers
        self._build_layers()

    def _build_layers(self) -> None:
        """Build all layers of the model."""
        # Initial standard convolution (not depthwise separable)
        self.initial_conv = layers.Conv2D(
            filters=int(32 * self.width_multiplier),
            kernel_size=3,
            strides=2,
            padding='same',
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='conv1'
        )
        self.initial_bn = layers.BatchNormalization(name='conv1_bn')
        self.initial_relu = layers.ReLU(name='conv1_relu')

        # Build depthwise separable blocks
        self.depthwise_blocks = []
        for block_id, (filters, stride) in enumerate(self.ARCHITECTURE, start=1):
            # Apply width multiplier to filter count
            actual_filters = int(filters * self.width_multiplier)

            block = DepthwiseSeparableBlock(
                filters=actual_filters,
                stride=stride,
                block_id=block_id,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f'block_{block_id}'
            )
            self.depthwise_blocks.append(block)

        # Global average pooling
        self.global_avg_pool = layers.GlobalAveragePooling2D(name='global_avg_pool')

        # Classification head
        if self.include_top:
            # Shape layer to ensure correct dimensions
            self.reshape = layers.Reshape((1, 1, int(1024 * self.width_multiplier)), name='reshape')

            # Dropout for regularization
            self.dropout = layers.Dropout(self.dropout_rate, name='dropout')

            # Final convolution as FC layer (MobileNetV1 uses Conv instead of Dense)
            self.classifier_conv = layers.Conv2D(
                filters=self.num_classes,
                kernel_size=1,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name='conv_preds'
            )

            # Reshape to get proper output shape
            self.output_reshape = layers.Reshape((self.num_classes,), name='output_reshape')

            # Softmax activation
            self.softmax = layers.Activation('softmax', name='act_softmax')

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass of the MobileNetV1 model."""
        # Initial convolution
        x = self.initial_conv(inputs)
        x = self.initial_bn(x, training=training)
        x = self.initial_relu(x)

        # Depthwise separable blocks
        for block in self.depthwise_blocks:
            x = block(x, training=training)

        # Global average pooling
        x = self.global_avg_pool(x)

        # Classification head
        if self.include_top:
            x = self.reshape(x)
            x = self.dropout(x, training=training)
            x = self.classifier_conv(x)
            x = self.output_reshape(x)
            x = self.softmax(x)

        return x

    @classmethod
    def from_variant(
            cls,
            variant: str,
            num_classes: int = 1000,
            input_shape: Optional[Tuple[int, ...]] = None,
            **kwargs: Any
    ) -> "MobileNetV1":
        """Create a MobileNetV1 model from a predefined variant.

        Args:
            variant: String, one of "1.0", "0.75", "0.5", "0.25"
            num_classes: Integer, number of output classes
            input_shape: Tuple, input shape. If None, uses (224, 224, 3)
            **kwargs: Additional arguments passed to the constructor

        Returns:
            MobileNetV1 model instance

        Raises:
            ValueError: If variant is not recognized

        Example:
            >>> # Standard MobileNetV1
            >>> model = MobileNetV1.from_variant("1.0", num_classes=1000)
            >>> # Smaller model for mobile
            >>> model = MobileNetV1.from_variant("0.5", num_classes=100)
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        config = cls.MODEL_VARIANTS[variant]

        # Set default input shape if not provided
        if input_shape is None:
            input_shape = (224, 224, 3)

        logger.info(f"Creating MobileNetV1-{variant} model")
        logger.info(f"Input shape: {input_shape}, Classes: {num_classes}")

        return cls(
            num_classes=num_classes,
            width_multiplier=config["width_multiplier"],
            input_shape=input_shape,
            **kwargs
        )

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        config = {
            "num_classes": self.num_classes,
            "width_multiplier": self.width_multiplier,
            "dropout_rate": self.dropout_rate,
            "weight_decay": self.weight_decay,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "include_top": self.include_top,
            "input_shape": self._input_shape,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MobileNetV1":
        """Create model from configuration."""
        # Deserialize the initializer if needed
        if isinstance(config.get("kernel_initializer"), dict):
            config["kernel_initializer"] = keras.initializers.deserialize(
                config["kernel_initializer"]
            )
        return cls(**config)

    def summary(self, **kwargs):
        """Print model summary with additional information."""
        # Build the model if it hasn't been built yet
        if not self.built and self._input_shape:
            self.build((None, *self._input_shape))

        super().summary(**kwargs)

        # Print additional model information
        total_blocks = len(self.depthwise_blocks)
        total_params = self.count_params()

        logger.info("MobileNetV1 Configuration:")
        logger.info(f"  - Input shape: {self._input_shape}")
        logger.info(f"  - Width multiplier (α): {self.width_multiplier}")
        logger.info(f"  - Number of depthwise blocks: {total_blocks}")
        logger.info(f"  - Include top: {self.include_top}")
        if self.include_top:
            logger.info(f"  - Number of classes: {self.num_classes}")
        logger.info(f"  - Dropout rate: {self.dropout_rate}")
        logger.info(f"  - Weight decay: {self.weight_decay}")
        logger.info(f"  - Total parameters: {total_params:,}")


# ---------------------------------------------------------------------

def create_mobilenetv1(
        variant: str = "1.0",
        num_classes: int = 1000,
        input_shape: Optional[Tuple[int, ...]] = None,
        pretrained: bool = False,
        **kwargs: Any
) -> MobileNetV1:
    """Convenience function to create MobileNetV1 models.

    Args:
        variant: String, model variant ("1.0", "0.75", "0.5", "0.25")
        num_classes: Integer, number of output classes
        input_shape: Tuple, input shape. If None, uses (224, 224, 3)
        pretrained: Boolean, whether to load pretrained weights (not implemented)
        **kwargs: Additional arguments passed to the model constructor

    Returns:
        MobileNetV1 model instance

    Example:
        >>> # Create standard MobileNetV1
        >>> model = create_mobilenetv1("1.0", num_classes=1000)
        >>>
        >>> # Create smaller model for CIFAR-10
        >>> model = create_mobilenetv1("0.5", num_classes=10, input_shape=(32, 32, 3))
        >>>
        >>> # Create tiny model for embedded devices
        >>> model = create_mobilenetv1("0.25", num_classes=100)
    """
    if pretrained:
        logger.warning("Pretrained weights are not yet implemented for MobileNetV1")

    model = MobileNetV1.from_variant(
        variant,
        num_classes=num_classes,
        input_shape=input_shape,
        **kwargs
    )

    return model

# ------------------------------------------------------------------------