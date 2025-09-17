"""
MobileNetV2: Inverted Residuals and Linear Bottlenecks
======================================================

A complete implementation of MobileNetV2 architecture using Universal
Inverted Bottleneck blocks configured to replicate the original inverted
residuals and linear bottlenecks. This implementation follows modern Keras 3
best practices for custom models with proper serialization support.

Based on: "MobileNetV2: Inverted Residuals and Linear Bottlenecks"
Paper: https://arxiv.org/abs/1801.04381

Key Features:
------------
- Universal Inverted Bottleneck blocks configured as inverted residuals
- Expansion layers with lightweight depthwise convolutions
- Residual connections between bottleneck layers
- Width multiplier (α) for model scaling
- Modular design with proper serialization support
- Complete variant support (1.0, 0.75, 0.5, 0.35 width multipliers)

Architecture Overview:
---------------------
MobileNetV2 consists of:
1. **Initial Conv**: Standard 3x3 convolution with stride 2 and 32 filters
2. **Bottleneck Blocks**: 17 `UniversalInvertedBottleneck` blocks organized in stages
3. **Final Conv**: 1x1 convolution expanding to 1280 channels
4. **Global Average Pooling**: Reduces spatial dimensions
5. **Classifier**: Fully connected layer for classification

Inverted Residual Block (emulated by UniversalInvertedBottleneck):
- Expansion: 1x1 conv to expand channels (with ReLU6)
- Depthwise: 3x3 depthwise conv (with ReLU6)
- Projection: 1x1 conv to project back (LINEAR - no activation)
- Residual: Skip connection when stride=1 and channels match

Usage Examples:
--------------
```python
# Standard MobileNetV2 (α=1.0) for ImageNet
model = MobileNetV2.from_variant("1.0", num_classes=1000)

# Smaller model (α=0.75) for CIFAR-10
model = MobileNetV2.from_variant("0.75", num_classes=10, input_shape=(32, 32, 3))

# Custom configuration
model = MobileNetV2(num_classes=100, width_multiplier=0.5, input_shape=(128, 128, 3))
```
"""

import keras
from keras import layers, regularizers
from typing import Tuple, Optional, Dict, Any, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.universal_inverted_bottleneck import UniversalInvertedBottleneck

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class MobileNetV2(keras.Model):
    """MobileNetV2 classification model built with Universal Inverted Bottleneck blocks.

    This class implements the full MobileNetV2 architecture, an efficient
    convolutional neural network designed for mobile and embedded vision
    applications. It utilizes `UniversalInvertedBottleneck` (UIB) layers configured
    to replicate the original's inverted residuals and linear bottlenecks.

    **Intent**: To provide a production-ready, configurable, and easily
    serializable implementation of the MobileNetV2 model. This serves as a
    best-practice example for building complex custom models in Keras 3,
    leveraging a flexible and unified building block (UIB).

    **Architecture**:
    ```
    Input(shape=[H, W, 3])
           ↓
    Initial Conv: 3x3 Conv2D(32, stride=2) -> BN -> ReLU6
           ↓
    Bottleneck Blocks: Sequence of 17 UniversalInvertedBottleneck layers
           ↓
    Final Conv: 1x1 Conv2D(1280) -> BN -> ReLU6
           ↓
    Pooling: GlobalAveragePooling2D
           ↓
    Classifier: Dropout -> Dense(num_classes, 'softmax')
           ↓
    Output(shape=[num_classes])
    ```

    **Data Flow**:
    1. An initial convolution layer performs downsampling and feature extraction.
    2. A series of 7 bottleneck stages, built from `UniversalInvertedBottleneck`
       layers, progressively extracts features and reduces spatial dimensions.
    3. A final 1x1 convolution expands the feature map to a high-dimensional space.
    4. Global average pooling converts the feature map into a single feature vector.
    5. A fully-connected classifier with softmax activation produces class probabilities.

    Args:
        num_classes: Integer, number of output classes for classification.
        width_multiplier: Float (α), scales the number of channels in each layer,
            allowing for control over model size and complexity.
        dropout_rate: Float, dropout rate applied before the final classifier.
        weight_decay: Float, L2 regularization factor for convolutional and dense layers.
        kernel_initializer: String or Initializer for kernel weight initialization.
        include_top: Boolean, whether to include the final pooling and classification layers.
        input_shape: Optional Tuple, the shape of the input tensor. Defaults to (224, 224, 3).
        **kwargs: Additional keyword arguments for the `keras.Model` base class.

    Input shape:
        3D tensor with shape `(height, width, channels)`, e.g., `(224, 224, 3)`.

    Output shape:
        2D tensor with shape `(batch_size, num_classes)` if `include_top=True`.
        4D feature map otherwise.

    Attributes:
        initial_conv, initial_bn, initial_relu: Layers for the first block.
        blocks: A list of `UniversalInvertedBottleneck` instances.
        last_conv, last_bn, last_relu: Layers for the final feature extraction.
        global_avg_pool: Global average pooling layer.
        dropout: Dropout layer (if used).
        classifier: Final Dense classification layer.

    Example:
        ```python
        # Standard MobileNetV2 (α=1.0) for ImageNet
        model = MobileNetV2(num_classes=1000, width_multiplier=1.0)

        # Smaller model (α=0.75) for CIFAR-10
        model = MobileNetV2(
            num_classes=10,
            width_multiplier=0.75,
            input_shape=(32, 32, 3)
        )
        ```

    References:
        - MobileNetV2 Paper: https://arxiv.org/abs/1801.04381
    """

    # Model variant configurations mapping variant name to width_multiplier
    MODEL_VARIANTS = {
        "1.4": 1.4, "1.0": 1.0, "0.75": 0.75, "0.5": 0.5, "0.35": 0.35,
    }

    # Architecture definition from Table 2 of the paper: (t, c, n, s)
    # t: expansion factor, c: output channels, n: repetitions, s: stride
    ARCHITECTURE = [
        (1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2), (6, 64, 4, 2),
        (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1),
    ]

    def __init__(
            self,
            num_classes: int = 1000,
            width_multiplier: float = 1.0,
            dropout_rate: float = 0.2,
            weight_decay: float = 4e-5,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
            include_top: bool = True,
            input_shape: Optional[Tuple[int, ...]] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # --- Configuration Validation and Storage ---
        if width_multiplier <= 0:
            raise ValueError(f"width_multiplier must be positive, got {width_multiplier}")
        self.input_shape_config = input_shape or (224, 224, 3)
        if len(self.input_shape_config) != 3:
            raise ValueError(f"input_shape must be a 3D tuple, got {self.input_shape_config}")

        # Store all configuration parameters for serialization
        self.num_classes = num_classes
        self.width_multiplier = width_multiplier
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.include_top = include_top
        self.kernel_regularizer = regularizers.L2(weight_decay) if weight_decay > 0 else None

        # --- CREATE all sub-layers in __init__ ---
        # For a keras.Model, all sub-layers should be created here. Keras will
        # handle calling their `build` methods automatically.
        self._build_model_layers()

    def _make_divisible(self, v: float, divisor: int = 8) -> int:
        """Ensures that layer channel counts are divisible by 8."""
        new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def _build_model_layers(self) -> None:
        """Create all layers of the model based on the configuration."""
        # Initial Convolution
        first_channels = self._make_divisible(32 * self.width_multiplier)
        self.initial_conv = layers.Conv2D(
            first_channels, 3, strides=2, padding='same', use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer, name='conv1'
        )
        self.initial_bn = layers.BatchNormalization(name='conv1_bn')
        self.initial_relu = layers.ReLU(max_value=6, name='conv1_relu6')

        # Bottleneck Blocks (using UniversalInvertedBottleneck)
        self.blocks = []
        block_id = 0
        for t, c, n, s in self.ARCHITECTURE:
            output_channels = self._make_divisible(c * self.width_multiplier)
            for i in range(n):
                stride = s if i == 0 else 1
                self.blocks.append(UniversalInvertedBottleneck(
                    filters=output_channels,
                    expansion_factor=t,
                    stride=stride,
                    kernel_size=3,              # Standard for MobileNetV2
                    use_dw1=True,               # Emulates MobileNetV2 block
                    use_dw2=False,              # Emulates MobileNetV2 block
                    activation_type='relu',     # Use ReLU...
                    activation_args={'max_value': 6}, # ...with max_value=6 (ReLU6)
                    normalization_type='batch_norm',
                    use_bias=False,
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f'block_{block_id}'
                ))
                block_id += 1

        # Final Convolution
        # FIX: Always scale the last convolution layer's channels. The original
        # conditional logic was incorrect for multipliers < 1.0.
        last_channels = self._make_divisible(1280 * self.width_multiplier)
        self.last_conv = layers.Conv2D(
            last_channels, 1, padding='same', use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer, name='conv_last'
        )
        self.last_bn = layers.BatchNormalization(name='conv_last_bn')
        self.last_relu = layers.ReLU(max_value=6, name='conv_last_relu6')

        # Top (Classification Head)
        if self.include_top:
            self.global_avg_pool = layers.GlobalAveragePooling2D(name='global_avg_pool')
            if self.dropout_rate > 0:
                self.dropout = layers.Dropout(self.dropout_rate, name='dropout')
            else:
                self.dropout = None
            self.classifier = layers.Dense(
                self.num_classes, activation='softmax', name='classifier',
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer
            )

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Forward pass of the MobileNetV2 model."""
        x = self.initial_conv(inputs)
        x = self.initial_bn(x, training=training)
        x = self.initial_relu(x)

        for block in self.blocks:
            x = block(x, training=training)

        x = self.last_conv(x)
        x = self.last_bn(x, training=training)
        x = self.last_relu(x)

        if self.include_top:
            x = self.global_avg_pool(x)
            if self.dropout:
                x = self.dropout(x, training=training)
            x = self.classifier(x)

        return x

    @classmethod
    def from_variant(
            cls,
            variant: str,
            num_classes: int = 1000,
            input_shape: Optional[Tuple[int, ...]] = None,
            **kwargs: Any
    ) -> "MobileNetV2":
        """Create a MobileNetV2 model from a predefined variant string."""
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(f"Unknown variant '{variant}'. Available: {list(cls.MODEL_VARIANTS.keys())}")

        width_multiplier = cls.MODEL_VARIANTS[variant]
        logger.info(f"Creating MobileNetV2 variant '{variant}' (α={width_multiplier})")

        return cls(
            num_classes=num_classes,
            width_multiplier=width_multiplier,
            input_shape=input_shape,
            **kwargs
        )

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "width_multiplier": self.width_multiplier,
            "dropout_rate": self.dropout_rate,
            "weight_decay": self.weight_decay,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
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
        total_blocks = len(self.blocks)
        total_params = self.count_params()

        logger.info("MobileNetV2 Configuration:")
        logger.info(f"  - Input shape: {self.input_shape_config}")
        logger.info(f"  - Width multiplier (α): {self.width_multiplier}")
        logger.info(f"  - Number of bottleneck blocks: {total_blocks}")
        logger.info(f"  - Include top: {self.include_top}")
        if self.include_top:
            logger.info(f"  - Number of classes: {self.num_classes}")
        logger.info(f"  - Dropout rate: {self.dropout_rate}")
        logger.info(f"  - Weight decay: {self.weight_decay}")
        logger.info(f"  - Total parameters: {total_params:,}")


# ---------------------------------------------------------------------

def create_mobilenetv2(
        variant: str = "1.0",
        num_classes: int = 1000,
        input_shape: Optional[Tuple[int, ...]] = None,
        pretrained: bool = False,
        **kwargs: Any
) -> MobileNetV2:
    """Convenience function to create MobileNetV2 models.

    Args:
        variant: String, model variant ("1.4", "1.0", "0.75", "0.5", "0.35")
        num_classes: Integer, number of output classes
        input_shape: Tuple, input shape. If None, uses (224, 224, 3)
        pretrained: Boolean, whether to load pretrained weights (not implemented)
        **kwargs: Additional arguments passed to the model constructor

    Returns:
        MobileNetV2 model instance

    Example:
        >>> # Create standard MobileNetV2
        >>> model = create_mobilenetv2("1.0", num_classes=1000)
        >>>
        >>> # Create smaller model for CIFAR-10
        >>> model = create_mobilenetv2("0.5", num_classes=10, input_shape=(32, 32, 3))
        >>>
        >>> # Create tiny model for embedded devices
        >>> model = create_mobilenetv2("0.35", num_classes=100)
    """
    if pretrained:
        logger.warning("Pretrained weights are not yet implemented for MobileNetV2")

    model = MobileNetV2.from_variant(
        variant,
        num_classes=num_classes,
        input_shape=input_shape,
        **kwargs
    )

    return model

# ---------------------------------------------------------------------
