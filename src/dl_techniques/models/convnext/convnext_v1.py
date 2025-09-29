"""
ConvNeXt V1 Model Implementation
==================================================

A complete implementation of the ConvNeXt V1 architecture .
This version can natively handle different input sizes without requiring preprocessing.

Based on: "A ConvNet for the 2020s" (Liu et al., 2022)
https://arxiv.org/abs/2201.03545

Key Features:
------------
- Modular design using ConvNextV1Block as building blocks
- Support for all standard ConvNeXt variants
- Smart stem and downsampling strategies
- Configurable stochastic depth (drop path)
- Proper normalization and initialization strategies
- Flexible head design (classification, feature extraction)
- Complete serialization support
- Production-ready implementation

Architecture Adaptations:
------------------------
- Small inputs (< 64x64): Uses 3x3 stem with stride 1, gentle downsampling
- Medium inputs (64-128): Uses 4x4 stem with stride 2, moderate downsampling
- Large inputs (>= 128): Uses original 4x4 stem with stride 4, standard downsampling
- Smart downsampling layer configuration that prevents over-downsampling

Model Variants:
--------------
- ConvNeXt-T: [3, 3, 9, 3] blocks, [96, 192, 384, 768] dims
- ConvNeXt-S: [3, 3, 27, 3] blocks, [96, 192, 384, 768] dims
- ConvNeXt-B: [3, 3, 27, 3] blocks, [128, 256, 512, 1024] dims
- ConvNeXt-L: [3, 3, 27, 3] blocks, [192, 384, 768, 1536] dims
- ConvNeXt-XL: [3, 3, 27, 3] blocks, [256, 512, 1024, 2048] dims

Usage Examples:
-------------
```python
# CIFAR-10 model (32x32 input)
model = ConvNeXtV1.from_variant("tiny", num_classes=10, input_shape=(32, 32, 3))

# MNIST model (28x28 input)
model = ConvNeXtV1.from_variant("small", num_classes=10, input_shape=(28, 28, 3))

# ImageNet model (224x224 input)
model = ConvNeXtV1.from_variant("base", num_classes=1000)

# Custom dataset model (64x64 input)
model = create_convnext_v1("large", num_classes=100, input_shape=(64, 64, 3))
```
"""

import keras
from typing import List, Optional, Union, Tuple, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.convnext_v1_block import ConvNextV1Block


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class ConvNeXtV1(keras.Model):
    """ConvNeXt V1 model implementation

    A modern ConvNet architecture that achieves competitive performance
    with Vision Transformers while maintaining the simplicity and efficiency
    of convolutional networks. This version adapts to different input sizes.

    Args:
        num_classes: Integer, number of output classes for classification.
            Only used if include_top=True.
        depths: List of integers, number of ConvNext blocks in each stage.
            Default is [3, 3, 9, 3] for ConvNeXt-Tiny.
        dims: List of integers, number of channels in each stage.
            Default is [96, 192, 384, 768] for ConvNeXt-Tiny.
        drop_path_rate: Float, stochastic depth rate. Linearly increases
            from 0 to this value across all blocks.
        kernel_size: Integer or tuple, kernel size for ConvNext blocks.
            Default is 7 following the original paper.
        activation: String or callable, activation function for blocks.
            Default is "gelu" as used in the original paper.
        use_bias: Boolean, whether to use bias in convolutions.
        kernel_regularizer: Regularizer function applied to kernels.
        dropout_rate: Float, dropout rate applied within blocks.
        spatial_dropout_rate: Float, spatial dropout rate for blocks.
        strides: int, Strides for downsampling.
        use_gamma: Boolean, whether to use learnable scaling in blocks.
        use_softorthonormal_regularizer: Boolean, whether to use soft
            orthonormal regularization in blocks.
        include_top: Boolean, whether to include the classification head.
        input_shape: Tuple, input shape. If None and include_top=True,
            uses (224, 224, 3) for ImageNet. Must be provided for non-ImageNet inputs.
        **kwargs: Additional keyword arguments for the Model base class.

    Raises:
        ValueError: If depths and dims have different lengths.
        ValueError: If invalid model configuration is provided.

    Example:
        >>> # Create ConvNeXt-Tiny model for CIFAR-10
        >>> model = ConvNeXtV1.from_variant("tiny", num_classes=10, input_shape=(32, 32, 3))
        >>>
        >>> # Create ConvNeXt-Small for MNIST
        >>> model = ConvNeXtV1.from_variant("small", num_classes=10, input_shape=(28, 28, 3))
        >>>
        >>> # Create standard ImageNet model
        >>> model = ConvNeXtV1.from_variant("base", num_classes=1000)
    """

    # Model variant configurations
    MODEL_VARIANTS = {
        "cifar10": {"depths": [5, 5], "dims": [96, 192]},
        "tiny": {"depths": [3, 3, 9, 3], "dims": [96, 192, 384, 768]},
        "small": {"depths": [3, 3, 27, 3], "dims": [96, 192, 384, 768]},
        "base": {"depths": [3, 3, 27, 3], "dims": [128, 256, 512, 1024]},
        "large": {"depths": [3, 3, 27, 3], "dims": [192, 384, 768, 1536]},
        "xlarge": {"depths": [3, 3, 27, 3], "dims": [256, 512, 1024, 2048]},
    }

    # Architecture constants
    LAYERNORM_EPSILON = 1e-6
    STEM_INITIALIZER = "truncated_normal"
    HEAD_INITIALIZER = "truncated_normal"

    def __init__(
            self,
            num_classes: int = 1000,
            depths: List[int] = [3, 3, 9, 3],
            dims: List[int] = [96, 192, 384, 768],
            drop_path_rate: float = 0.0,
            kernel_size: Union[int, Tuple[int, int]] = 7,
            activation: str = "gelu",
            use_bias: bool = True,
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            dropout_rate: float = 0.0,
            spatial_dropout_rate: float = 0.0,
            strides: int = 4,
            use_gamma: bool = True,
            use_softorthonormal_regularizer: bool = True,
            include_top: bool = True,
            input_shape: Tuple[int, ...] = (None, None, 3),
            **kwargs
    ):
        super().__init__(**kwargs)

        # Validate configuration
        if len(depths) != len(dims):
            raise ValueError(
                f"Length of depths ({len(depths)}) must equal length of dims ({len(dims)})"
            )

        if len(depths) != 4:
            logger.warning(
                f"ConvNeXt typically uses 4 stages, got {len(depths)} stages"
            )

        # Store configuration
        self.num_classes = num_classes
        self.depths = depths
        self.dims = dims
        self.drop_path_rate = drop_path_rate
        self.kernel_size = kernel_size
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_regularizer = kernel_regularizer
        self.dropout_rate = dropout_rate
        self.spatial_dropout_rate = spatial_dropout_rate
        self.use_gamma = use_gamma
        self.use_softorthonormal_regularizer = use_softorthonormal_regularizer
        self.include_top = include_top
        self.strides = strides
        self.input_shape = input_shape

        # Validate and store input shape details
        if input_shape is None:
            input_shape = (None, None, 3)
        if len(input_shape) != 3:
            raise ValueError(f"input_shape must be 3D, got {input_shape}")

        self.input_height, self.input_width, self.input_channels = input_shape
        if self.input_channels not in [1, 3]:
            logger.warning(
                f"Unusual number of channels: {self.input_channels}. ConvNeXt typically uses 3 channels")

        # --- Build layers ---
        # This follows the Keras subclassing model best practice.
        # Layers are created in __init__ and used in call().

        # 1. Stem
        self._build_stem()

        # 2. Downsample layers and Stages
        self.downsample_layers_list = []
        self.stages_list = []
        for i in range(len(self.depths)):
            # Downsample layer (except for the first stage)
            if i > 0:
                self._build_downsample_layer(i)
            # Stage of ConvNeXt blocks
            self._build_stage(i)

        # 3. Head
        if self.include_top:
            self._build_head()

        logger.info(
            f"Created ConvNeXt V1 model for input {input_shape} "
            f"with {sum(depths)} blocks"
        )

    def _build_stem(self):
        """Build and assign stem layers."""
        stem_kernel_size = self.strides
        stem_stride = self.strides
        self.stem_conv = keras.layers.Conv2D(
            filters=self.dims[0],
            kernel_size=stem_kernel_size,
            strides=stem_stride,
            padding="same" if stem_stride == 1 else "valid",
            use_bias=self.use_bias,
            kernel_initializer=self.STEM_INITIALIZER,
            kernel_regularizer=self.kernel_regularizer,
            name="stem_conv"
        )
        self.stem_norm = keras.layers.LayerNormalization(
            epsilon=self.LAYERNORM_EPSILON,
            center=self.use_bias,
            scale=True,
            name="stem_norm"
        )

    def _build_downsample_layer(self, stage_idx: int):
        """Build and assign a downsample layer."""
        downsample_kernel_size, downsample_stride = self.strides, self.strides
        downsample_norm = keras.layers.LayerNormalization(
            epsilon=self.LAYERNORM_EPSILON,
            center=self.use_bias,
            scale=True,
            name=f"downsample_norm_{stage_idx - 1}"
        )
        downsample_conv = keras.layers.Conv2D(
            filters=self.dims[stage_idx],
            kernel_size=downsample_kernel_size,
            strides=downsample_stride,
            padding="valid",
            use_bias=self.use_bias,
            kernel_initializer=self.STEM_INITIALIZER,
            kernel_regularizer=self.kernel_regularizer,
            name=f"downsample_conv_{stage_idx - 1}"
        )
        self.downsample_layers_list.append([downsample_norm, downsample_conv])

    def _build_stage(self, stage_idx: int):
        """Build and assign a stage of ConvNeXt blocks."""
        stage_blocks = []
        depth = self.depths[stage_idx]
        dim = self.dims[stage_idx]
        total_blocks = sum(self.depths)
        block_start_idx = sum(self.depths[:stage_idx])

        for block_idx in range(depth):
            current_block_idx = block_start_idx + block_idx
            if total_blocks > 1:
                drop_rate = self.drop_path_rate * current_block_idx / (total_blocks - 1)
            else:
                drop_rate = 0.0

            block = ConvNextV1Block(
                kernel_size=self.kernel_size,
                filters=dim,
                activation=self.activation,
                kernel_regularizer=self.kernel_regularizer,
                use_bias=self.use_bias,
                dropout_rate=self.dropout_rate,
                spatial_dropout_rate=self.spatial_dropout_rate,
                use_gamma=self.use_gamma,
                use_softorthonormal_regularizer=self.use_softorthonormal_regularizer,
                name=f"stage_{stage_idx}_block_{block_idx}"
            )
            drop_path = keras.layers.Dropout(
                rate=drop_rate, noise_shape=(None, 1, 1, 1),
                name=f"stage_{stage_idx}_block_{block_idx}_drop_path"
            ) if drop_rate > 0 else None
            stage_blocks.append({"block": block, "drop_path": drop_path})
        self.stages_list.append(stage_blocks)

    def _build_head(self):
        """Build and assign head layers."""
        self.gap = keras.layers.GlobalAveragePooling2D(name="global_avg_pool")
        self.head_norm = keras.layers.LayerNormalization(
            epsilon=self.LAYERNORM_EPSILON,
            center=self.use_bias,
            scale=True,
            name="head_norm"
        )
        if self.num_classes > 0:
            self.classifier = keras.layers.Dense(
                units=self.num_classes,
                use_bias=self.use_bias,
                kernel_initializer=self.HEAD_INITIALIZER,
                kernel_regularizer=self.kernel_regularizer,
                name="classifier"
            )
        else:
            self.classifier = None

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Defines the forward pass of the model."""
        x = self.stem_conv(inputs)
        x = self.stem_norm(x)

        for stage_idx, stage_blocks in enumerate(self.stages_list):
            if stage_idx > 0:
                norm_layer, conv_layer = self.downsample_layers_list[stage_idx - 1]
                x = norm_layer(x)
                x = conv_layer(x)

            for block_info in stage_blocks:
                residual = x
                x = block_info["block"](x, training=training)
                if block_info["drop_path"]:
                    x = block_info["drop_path"](x, training=training)
                x = keras.layers.add([residual, x])

        if self.include_top:
            x = self.gap(x)
            x = self.head_norm(x)
            if self.classifier:
                x = self.classifier(x)

        return x

    @classmethod
    def from_variant(
            cls,
            variant: str,
            num_classes: int = 1000,
            input_shape: Optional[Tuple[int, ...]] = None,
            **kwargs
    ) -> "ConvNeXtV1":
        """Create a ConvNeXt model from a predefined variant.

        Args:
            variant: String, one of "tiny", "small", "base", "large", "xlarge"
            num_classes: Integer, number of output classes
            input_shape: Tuple, input shape. If None and include_top=True, uses (224, 224, 3)
            **kwargs: Additional arguments passed to the constructor

        Returns:
            ConvNeXtV1 model instance

        Raises:
            ValueError: If variant is not recognized

        Example:
            >>> # CIFAR-10 model
            >>> model = ConvNeXtV1.from_variant("tiny", num_classes=10, input_shape=(32, 32, 3))
            >>> # MNIST model
            >>> model = ConvNeXtV1.from_variant("small", num_classes=10, input_shape=(28, 28, 3))
            >>> # ImageNet model
            >>> model = ConvNeXtV1.from_variant("base", num_classes=1000)
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        config = cls.MODEL_VARIANTS[variant]

        logger.info(f"Creating ConvNeXt-{variant.upper()} model")
        logger.info(f"from_variant received input_shape: {input_shape}")

        return cls(
            num_classes=num_classes,
            depths=config["depths"],
            dims=config["dims"],
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
            "drop_path_rate": self.drop_path_rate,
            "kernel_size": self.kernel_size,
            "activation": self.activation,
            "use_bias": self.use_bias,
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "dropout_rate": self.dropout_rate,
            "spatial_dropout_rate": self.spatial_dropout_rate,
            "use_gamma": self.use_gamma,
            "use_softorthonormal_regularizer": self.use_softorthonormal_regularizer,
            "include_top": self.include_top,
            "input_shape": self.input_shape,
            "strides": self.strides
        }
        base_config = super().get_config()
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ConvNeXtV1":
        """Create model from configuration.

        Args:
            config: Configuration dictionary

        Returns:
            ConvNeXtV1 model instance
        """
        # Deserialize regularizer if present
        if config.get("kernel_regularizer"):
            config["kernel_regularizer"] = keras.regularizers.deserialize(
                config["kernel_regularizer"]
            )

        return cls(**config)

    def summary(self, **kwargs):
        """Print model summary with additional information."""
        # Build the model first if it hasn't been built
        if not self.built:
            dummy_input = keras.KerasTensor(self.input_shape)
            self.build(dummy_input.shape)

        super().summary(**kwargs)

        # Print additional model information
        total_blocks = sum(self.depths)
        logger.info(f"ConvNeXt V1 configuration:")
        logger.info(f"  - Input shape: ({self.input_height}, {self.input_width}, {self.input_channels})")
        logger.info(f"  - Stages: {len(self.depths)}")
        logger.info(f"  - Depths: {self.depths}")
        logger.info(f"  - Original dimensions: {self.dims}")
        logger.info(f"  - Total blocks: {total_blocks}")
        logger.info(f"  - Drop path rate: {self.drop_path_rate}")
        logger.info(f"  - Kernel size: {self.kernel_size}")
        logger.info(f"  - Include top: {self.include_top}")
        if self.include_top:
            logger.info(f"  - Number of classes: {self.num_classes}")


# ---------------------------------------------------------------------

def create_convnext_v1(
        variant: str = "tiny",
        num_classes: int = 1000,
        input_shape: Optional[Tuple[int, ...]] = (None, None, 3),
        pretrained: bool = False,
        **kwargs
) -> ConvNeXtV1:
    """Convenience function to create ConvNeXt V1 models.

    Args:
        variant: String, model variant ("tiny", "small", "base", "large", "xlarge")
        num_classes: Integer, number of output classes
        input_shape: Tuple, input shape.
        pretrained: Boolean, whether to load pretrained weights (not implemented)
        **kwargs: Additional arguments passed to the model constructor

    Returns:
        ConvNeXtV1 model instance

    Example:
        >>> # Create ConvNeXt-Tiny for CIFAR-10
        >>> model = create_convnext_v1("tiny", num_classes=10, input_shape=(32, 32, 3))
        >>>
        >>> # Create ConvNeXt-Small for MNIST
        >>> model = create_convnext_v1("small", num_classes=10, input_shape=(28, 28, 3))
        >>>
        >>> # Create ConvNeXt-Base for ImageNet
        >>> model = create_convnext_v1("base", num_classes=1000)
    """
    if pretrained:
        logger.warning("Pretrained weights are not yet implemented")

    model = ConvNeXtV1.from_variant(
        variant,
        num_classes=num_classes,
        input_shape=input_shape,
        **kwargs
    )

    return model

# ---------------------------------------------------------------------