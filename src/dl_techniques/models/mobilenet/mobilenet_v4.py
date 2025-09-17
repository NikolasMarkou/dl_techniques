"""
MobileNetV4: Universal and Efficient Neural Networks for Mobile Applications
========================================================================

A complete implementation of MobileNetV4 architecture with Universal Inverted Bottleneck
(UIB) blocks and optional Mobile Multi-Query Attention (MQA). This version follows modern
Keras 3 best practices for custom models.

Based on: "MobileNetV4: Universal Inverted Bottleneck and Mobile MQA"
Paper: https://arxiv.org/abs/2404.10518

Key Features:
------------
- Universal Inverted Bottleneck (UIB) blocks with multiple configurations
- Optional Mobile Multi-Query Attention for hybrid models
- Support for both Conv-only and Hybrid variants
- Modular design with proper serialization support
- Configurable width multipliers and input resolutions
- Complete variant support (Small, Medium, Large, Hybrid variants)

Architecture Overview:
---------------------
MobileNetV4 consists of:
1. **Stem**: Initial convolution block for feature extraction
2. **Body**: Stack of UIB blocks organized in stages with progressive downsampling
3. **Head**: Global average pooling and classification layer

UIB Block Types:
- IB (Inverted Bottleneck): Standard MobileNet block
- ConvNext: ConvNext-style block with depthwise conv first
- ExtraDW: Enhanced version with additional depthwise conv
- FFN: Feed-forward network style block

Model Variants:
--------------
- MobileNetV4-ConvSmall: Lightweight conv-only model
- MobileNetV4-ConvMedium: Balanced conv-only model
- MobileNetV4-ConvLarge: High-capacity conv-only model
- MobileNetV4-Hybrid-Medium: Medium model with Mobile MQA
- MobileNetV4-Hybrid-Large: Large model with Mobile MQA

Usage Examples:
--------------
```python
# CIFAR-10 model (32x32 input)
model = MobileNetV4.from_variant("conv_small", num_classes=10, input_shape=(32, 32, 3))

# ImageNet model (224x224 input)
model = MobileNetV4.from_variant("conv_medium", num_classes=1000)

# Custom input size with hybrid attention
model = MobileNetV4.from_variant("hybrid_medium", num_classes=100, input_shape=(128, 128, 3))
```
"""

import keras
from keras import layers, regularizers
from typing import List, Tuple, Optional, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.attention.mobile_mqa import MobileMQA
from dl_techniques.layers.universal_inverted_bottleneck import UniversalInvertedBottleneck

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class MobileNetV4(keras.Model):
    """MobileNetV4 model implementation with Universal Inverted Bottleneck blocks.

    A modern efficient architecture combining the best of MobileNets with new
    Universal Inverted Bottleneck (UIB) blocks that unify different architectural
    patterns. Supports both pure convolutional and hybrid variants with Mobile MQA.

    Args:
        num_classes: Integer, number of output classes for classification.
            Only used if include_top=True.
        depths: List of integers, number of UIB blocks in each stage.
            Default is [1, 2, 3, 4, 3, 3, 1] for MobileNetV4-ConvMedium.
        dims: List of integers, number of channels in each stage.
            Default is [16, 24, 40, 80, 112, 192, 320] for MobileNetV4-ConvMedium.
        block_types: List of strings, UIB block type for each stage.
            Options: "IB", "ConvNext", "ExtraDW", "FFN". Default optimized per stage.
        strides: List of integers, stride for the first block of each stage.
            Default is [1, 2, 2, 2, 1, 2, 1].
        width_multiplier: Float, multiplier for the number of filters.
            Values like 0.5, 0.75, 1.0, 1.25 control model capacity.
        use_attention: Boolean, whether to use Mobile MQA in later stages.
            Creates hybrid MobileNetV4-Hybrid variant when True.
        attention_stages: List of integers, which stages to add attention.
            Default is [5, 6] (last two stages) when use_attention=True.
        dropout_rate: Float, dropout rate for regularization in classifier head.
        weight_decay: Float, L2 regularization factor for all layers.
        kernel_initializer: String or initializer, weight initialization strategy.
        include_top: Boolean, whether to include the classification head.
        input_shape: Tuple, input shape. If None, defaults to (224, 224, 3).
        **kwargs: Additional keyword arguments for the Model base class.

    Raises:
        ValueError: If depths, dims, block_types, or strides have different lengths.
        ValueError: If invalid block type is specified.
        ValueError: If invalid attention stage indices are provided.

    Example:
        >>> # Create MobileNetV4-ConvMedium for ImageNet
        >>> model = MobileNetV4.from_variant("conv_medium", num_classes=1000)
        >>>
        >>> # Create MobileNetV4-ConvSmall for CIFAR-10
        >>> model = MobileNetV4.from_variant("conv_small", num_classes=10, input_shape=(32, 32, 3))
        >>>
        >>> # Create MobileNetV4-Hybrid with custom configuration
        >>> model = MobileNetV4(
        ...     num_classes=100,
        ...     depths=[1, 2, 3, 4, 3, 3, 1],
        ...     dims=[16, 24, 40, 80, 112, 192, 320],
        ...     use_attention=True,
        ...     attention_stages=[5, 6],
        ...     input_shape=(128, 128, 3)
        ... )
    """

    # Model variant configurations
    MODEL_VARIANTS = {
        "conv_small": {
            "depths": [1, 1, 2, 3, 2, 2, 1],
            "dims": [16, 24, 32, 64, 96, 128, 160],
            "block_types": ["IB", "IB", "IB", "ExtraDW", "IB", "IB", "IB"],
            "use_attention": False,
        },
        "conv_medium": {
            "depths": [1, 2, 3, 4, 3, 3, 1],
            "dims": [16, 24, 40, 80, 112, 192, 320],
            "block_types": ["IB", "IB", "ExtraDW", "ExtraDW", "IB", "ExtraDW", "IB"],
            "use_attention": False,
        },
        "conv_large": {
            "depths": [1, 2, 4, 5, 4, 4, 1],
            "dims": [24, 32, 48, 96, 136, 224, 384],
            "block_types": ["IB", "ExtraDW", "ExtraDW", "ExtraDW", "ExtraDW", "ExtraDW", "IB"],
            "use_attention": False,
        },
        "hybrid_medium": {
            "depths": [1, 2, 3, 4, 3, 3, 1],
            "dims": [16, 24, 40, 80, 112, 192, 320],
            "block_types": ["IB", "IB", "ExtraDW", "ExtraDW", "IB", "ExtraDW", "IB"],
            "use_attention": True,
            "attention_stages": [5, 6],
        },
        "hybrid_large": {
            "depths": [1, 2, 4, 5, 4, 4, 1],
            "dims": [24, 32, 48, 96, 136, 224, 384],
            "block_types": ["IB", "ExtraDW", "ExtraDW", "ExtraDW", "ExtraDW", "ExtraDW", "IB"],
            "use_attention": True,
            "attention_stages": [5, 6],
        },
    }

    # Architecture constants
    STEM_KERNEL_SIZE = 3
    STEM_STRIDE = 2
    DEFAULT_STRIDES = [1, 2, 2, 2, 1, 2, 1]
    DEFAULT_ATTENTION_STAGES = [5, 6]
    HEAD_HIDDEN_DIM = 1280
    LAYERNORM_EPSILON = 1e-6

    def __init__(
        self,
        num_classes: int = 1000,
        depths: List[int] = [1, 2, 3, 4, 3, 3, 1],
        dims: List[int] = [16, 24, 40, 80, 112, 192, 320],
        block_types: List[str] = ["IB", "IB", "ExtraDW", "ExtraDW", "IB", "ExtraDW", "IB"],
        strides: List[int] = [1, 2, 2, 2, 1, 2, 1],
        width_multiplier: float = 1.0,
        use_attention: bool = False,
        attention_stages: List[int] = [5, 6],
        dropout_rate: float = 0.2,
        weight_decay: float = 1e-5,
        kernel_initializer: str = "he_normal",
        include_top: bool = True,
        input_shape: Tuple[int, ...] = (224, 224, 3),
        **kwargs
    ):
        super().__init__(**kwargs)

        # Validate block types first to ensure tests get the expected error
        valid_block_types = {"IB", "ConvNext", "ExtraDW", "FFN"}
        for block_type in block_types:
            if block_type not in valid_block_types:
                raise ValueError(
                    f"Invalid block type '{block_type}'. "
                    f"Must be one of {valid_block_types}"
                )

        # Validate configuration lengths
        stage_configs = [depths, dims, block_types, strides]
        stage_lengths = [len(config) for config in stage_configs]
        if not all(length == stage_lengths[0] for length in stage_lengths):
            raise ValueError(
                f"All stage configurations must have same length. Got: "
                f"depths={len(depths)}, dims={len(dims)}, "
                f"block_types={len(block_types)}, strides={len(strides)}"
            )

        # Validate attention stages
        if use_attention:
            max_stage_idx = len(depths) - 1
            for stage_idx in attention_stages:
                if not (0 <= stage_idx <= max_stage_idx):
                    raise ValueError(
                        f"Attention stage index {stage_idx} out of range. "
                        f"Must be in [0, {max_stage_idx}]"
                    )

        # Validate input shape
        if input_shape and len(input_shape) != 3:
            raise ValueError(f"input_shape must be 3D, got {input_shape}")

        if input_shape:
            height, width, channels = input_shape
            if channels not in [1, 3]:
                logger.warning(f"Unusual number of channels: {channels}")


        # Store configuration
        self.num_classes = num_classes
        self.depths = depths
        self.dims = dims
        self.block_types = block_types
        self.strides = strides
        self.width_multiplier = width_multiplier
        self.use_attention = use_attention
        self.attention_stages = attention_stages
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.kernel_initializer = kernel_initializer
        self.include_top = include_top
        self._input_shape = input_shape

        # Apply width multiplier to dimensions
        self.actual_dims = [int(dim * width_multiplier) for dim in dims]

        # Create regularizer
        self.kernel_regularizer = regularizers.L2(weight_decay) if weight_decay > 0 else None
        
        # Instantiate layers in __init__ for proper tracking and serialization
        self.stem_conv, self.stem_bn, self.stem_activation = self._build_stem()

        self.stages = []
        for stage_idx in range(len(self.depths)):
            stage_layers = self._build_stage(stage_idx)
            self.stages.append(stage_layers)

        if self.include_top:
            self.head_layers = self._build_head()

    def call(self, x, training=None):
        """Forward pass of the MobileNetV4 model."""
        # Stem
        x = self.stem_conv(x)
        x = self.stem_bn(x, training=training)
        x = self.stem_activation(x)

        # Body (Stages)
        for stage_layers in self.stages:
            for layer in stage_layers:
                x = layer(x, training=training)

        # Head
        if self.include_top:
            for layer in self.head_layers:
                if isinstance(layer, layers.Dropout):
                    x = layer(x, training=training)
                else:
                    x = layer(x)
        return x

    def _build_stem(self):
        """Build and return the stem layers."""
        stem_conv = layers.Conv2D(
            filters=self.actual_dims[0],
            kernel_size=self.STEM_KERNEL_SIZE,
            strides=self.STEM_STRIDE,
            padding='same',
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="stem_conv"
        )
        stem_bn = layers.BatchNormalization(name="stem_bn")
        stem_activation = layers.ReLU(name="stem_relu")
        return stem_conv, stem_bn, stem_activation


    def _build_stage(self, stage_idx: int):
        """Build and return layers for a single stage."""
        stage_layers = []
        depth = self.depths[stage_idx]
        dim = self.actual_dims[stage_idx]
        block_type = self.block_types[stage_idx]
        stage_stride = self.strides[stage_idx]

        logger.info(
            f"Building stage {stage_idx}: {depth} blocks, {dim} dims, "
            f"type={block_type}, stride={stage_stride}"
        )

        for block_idx in range(depth):
            block_stride = stage_stride if block_idx == 0 else 1
            block = UniversalInvertedBottleneck(
                filters=dim,
                stride=block_stride,
                block_type=block_type,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"stage_{stage_idx}_block_{block_idx}"
            )
            stage_layers.append(block)

        if self.use_attention and stage_idx in self.attention_stages:
            logger.info(f"Adding Mobile MQA to stage {stage_idx}")
            mqa_layer = MobileMQA(
                dim=dim,
                use_downsampling=True,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"stage_{stage_idx}_mqa"
            )
            stage_layers.append(mqa_layer)

        return stage_layers


    def _build_head(self):
        """Build and return the head layers."""
        head_layers_list = []
        gap = layers.GlobalAveragePooling2D(name="global_avg_pool")
        head_layers_list.append(gap)

        if self.HEAD_HIDDEN_DIM > 0:
            hidden_dense = layers.Dense(
                self.HEAD_HIDDEN_DIM,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name="head_hidden"
            )
            hidden_activation = layers.ReLU(name="head_hidden_relu")
            hidden_dropout = layers.Dropout(self.dropout_rate, name="head_dropout")
            head_layers_list.extend([hidden_dense, hidden_activation, hidden_dropout])
        
        if self.num_classes > 0:
            classifier = layers.Dense(
                self.num_classes,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                activation='softmax',
                name="classifier"
            )
            head_layers_list.append(classifier)
            
        return head_layers_list


    @classmethod
    def from_variant(
        cls,
        variant: str,
        num_classes: int = 1000,
        input_shape: Optional[Tuple[int, ...]] = None,
        width_multiplier: float = 1.0,
        **kwargs
    ) -> "MobileNetV4":
        """Create a MobileNetV4 model from a predefined variant.

        Args:
            variant: String, one of "conv_small", "conv_medium", "conv_large",
                "hybrid_medium", "hybrid_large"
            num_classes: Integer, number of output classes
            input_shape: Tuple, input shape. If None, uses (224, 224, 3)
            width_multiplier: Float, multiplier for filter dimensions
            **kwargs: Additional arguments passed to the constructor

        Returns:
            MobileNetV4 model instance

        Raises:
            ValueError: If variant is not recognized

        Example:
            >>> # CIFAR-10 model
            >>> model = MobileNetV4.from_variant("conv_small", num_classes=10, input_shape=(32, 32, 3))
            >>> # ImageNet model with attention
            >>> model = MobileNetV4.from_variant("hybrid_medium", num_classes=1000)
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

        logger.info(f"Creating MobileNetV4-{variant} model")
        logger.info(f"Input shape: {input_shape}, Classes: {num_classes}")

        return cls(
            num_classes=num_classes,
            depths=config["depths"],
            dims=config["dims"],
            block_types=config["block_types"],
            strides=config.get("strides", cls.DEFAULT_STRIDES),
            width_multiplier=width_multiplier,
            use_attention=config["use_attention"],
            attention_stages=config.get("attention_stages", cls.DEFAULT_ATTENTION_STAGES),
            input_shape=input_shape,
            **kwargs
        )

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization.

        Returns:
            Configuration dictionary
        """
        config = {
            "num_classes": self.num_classes,
            "depths": self.depths,
            "dims": self.dims,
            "block_types": self.block_types,
            "strides": self.strides,
            "width_multiplier": self.width_multiplier,
            "use_attention": self.use_attention,
            "attention_stages": self.attention_stages,
            "dropout_rate": self.dropout_rate,
            "weight_decay": self.weight_decay,
            "kernel_initializer": self.kernel_initializer,
            "include_top": self.include_top,
            "input_shape": self._input_shape,
        }
        base_config = super().get_config()
        return {**base_config, **config}


    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MobileNetV4":
        """Create model from configuration.

        Args:
            config: Configuration dictionary

        Returns:
            MobileNetV4 model instance
        """
        return cls(**config)

    def summary(self, **kwargs):
        """Print model summary with additional information."""
        # Build the model on a dummy input if it hasn't been built yet
        if not self.built and self._input_shape:
            self.build(self._input_shape)
            # In Keras 3, it's better to build by calling with dummy data
            # self(ops.zeros((1, *self._input_shape)))

        super().summary(**kwargs)

        # Print additional model information
        total_blocks = sum(self.depths)
        total_params = self.count_params()

        logger.info("MobileNetV4 Configuration:")
        logger.info(f"  - Input shape: {self._input_shape}")
        logger.info(f"  - Stages: {len(self.depths)}")
        logger.info(f"  - Depths: {self.depths}")
        logger.info(f"  - Original dimensions: {self.dims}")
        logger.info(f"  - Actual dimensions: {self.actual_dims}")
        logger.info(f"  - Block types: {self.block_types}")
        logger.info(f"  - Total blocks: {total_blocks}")
        logger.info(f"  - Width multiplier: {self.width_multiplier}")
        logger.info(f"  - Use attention: {self.use_attention}")
        if self.use_attention:
            logger.info(f"  - Attention stages: {self.attention_stages}")
        logger.info(f"  - Include top: {self.include_top}")
        if self.include_top:
            logger.info(f"  - Number of classes: {self.num_classes}")
        logger.info(f"  - Total parameters: {total_params:,}")


# ---------------------------------------------------------------------

def create_mobilenetv4(
    variant: str = "conv_medium",
    num_classes: int = 1000,
    input_shape: Optional[Tuple[int, ...]] = None,
    width_multiplier: float = 1.0,
    pretrained: bool = False,
    **kwargs
) -> MobileNetV4:
    """Convenience function to create MobileNetV4 models.

    Args:
        variant: String, model variant ("conv_small", "conv_medium", "conv_large",
            "hybrid_medium", "hybrid_large")
        num_classes: Integer, number of output classes
        input_shape: Tuple, input shape. If None, uses (224, 224, 3)
        width_multiplier: Float, multiplier for filter dimensions
        pretrained: Boolean, whether to load pretrained weights (not implemented)
        **kwargs: Additional arguments passed to the model constructor

    Returns:
        MobileNetV4 model instance

    Example:
        >>> # Create MobileNetV4-ConvSmall for CIFAR-10
        >>> model = create_mobilenetv4("conv_small", num_classes=10, input_shape=(32, 32, 3))
        >>>
        >>> # Create MobileNetV4-HybridMedium for ImageNet
        >>> model = create_mobilenetv4("hybrid_medium", num_classes=1000)
        >>>
        >>> # Create custom width multiplier model
        >>> model = create_mobilenetv4("conv_medium", width_multiplier=0.75)
    """
    if pretrained:
        logger.warning("Pretrained weights are not yet implemented")

    model = MobileNetV4.from_variant(
        variant,
        num_classes=num_classes,
        input_shape=input_shape,
        width_multiplier=width_multiplier,
        **kwargs
    )

    return model

# ------------------------------------------------------------------------
