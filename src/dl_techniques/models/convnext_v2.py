"""
ConvNeXt V2 Model Implementation
===============================

A complete implementation of the ConvNeXt V2 architecture as described in:
"ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders" (Woo et al., 2023)
https://arxiv.org/abs/2301.00808

This implementation builds upon the ConvNextV2Block to create full model variants
including ConvNeXt-Atto, ConvNeXt-Femto, ConvNeXt-Pico, ConvNeXt-Nano,
ConvNeXt-Tiny, ConvNeXt-Base, ConvNeXt-Large, and ConvNeXt-Huge.

Key Features:
------------
- Modular design using ConvNextV2Block as building blocks
- Global Response Normalization (GRN) for enhanced feature competition
- Support for all standard ConvNeXt V2 variants including micro variants
- Configurable stochastic depth (drop path)
- Proper normalization and initialization strategies
- Flexible head design (classification, feature extraction)
- Complete serialization support
- Production-ready implementation

Architecture:
------------
The ConvNeXt V2 model consists of:
1. Patchify stem (4x4 conv with stride 4) + LayerNorm
2. Four stages with varying depths and channel dimensions
3. Downsampling layers between stages (2x2 conv with stride 2)
4. Global average pooling + LayerNorm + Linear classifier

Key V2 Improvements:
-------------------
- Global Response Normalization (GRN) within blocks
- Enhanced feature competition and representation capacity
- Better transfer learning performance
- Improved masked autoencoder compatibility

Model Variants:
--------------
- ConvNeXt-Atto: [2, 2, 6, 2] blocks, [40, 80, 160, 320] dims (3.7M params)
- ConvNeXt-Femto: [2, 2, 6, 2] blocks, [48, 96, 192, 384] dims (5.2M params)
- ConvNeXt-Pico: [2, 2, 6, 2] blocks, [64, 128, 256, 512] dims (9.1M params)
- ConvNeXt-Nano: [2, 2, 8, 2] blocks, [80, 160, 320, 640] dims (15.6M params)
- ConvNeXt-Tiny: [3, 3, 9, 3] blocks, [96, 192, 384, 768] dims (28.6M params)
- ConvNeXt-Base: [3, 3, 27, 3] blocks, [128, 256, 512, 1024] dims (89M params)
- ConvNeXt-Large: [3, 3, 27, 3] blocks, [192, 384, 768, 1536] dims (198M params)
- ConvNeXt-Huge: [3, 3, 27, 3] blocks, [352, 704, 1408, 2816] dims (660M params)

Usage Examples:
-------------
```python
# Standard ConvNeXt-Tiny for ImageNet classification
model = ConvNeXtV2.from_variant("tiny", num_classes=1000)

# Custom configuration
model = ConvNeXtV2(
    num_classes=10,
    depths=[3, 3, 9, 3],
    dims=[96, 192, 384, 768],
    drop_path_rate=0.1
)

# Feature extractor (no classification head)
feature_extractor = ConvNeXtV2(
    depths=[3, 3, 9, 3],
    dims=[96, 192, 384, 768],
    include_top=False
)
```
"""

import keras
from typing import List, Optional, Union, Tuple, Dict, Any

from dl_techniques.utils.logger import logger
from dl_techniques.layers.convnext_v2_block import ConvNextV2Block


@keras.saving.register_keras_serializable()
class ConvNeXtV2(keras.Model):
    """ConvNeXt V2 model implementation.

    A modern ConvNet architecture that incorporates Global Response Normalization
    for enhanced inter-channel feature competition, achieving superior performance
    in both supervised learning and self-supervised masked autoencoder training.

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
        use_gamma: Boolean, whether to use learnable scaling in blocks.
        use_softorthonormal_regularizer: Boolean, whether to use soft
            orthonormal regularization in blocks.
        include_top: Boolean, whether to include the classification head.
        input_shape: Tuple, input shape. Only used if include_top=False.
        **kwargs: Additional keyword arguments for the Model base class.

    Raises:
        ValueError: If depths and dims have different lengths.
        ValueError: If invalid model configuration is provided.

    Example:
        >>> # Create ConvNeXt-Tiny model
        >>> model = ConvNeXtV2.from_variant("tiny", num_classes=1000)
        >>>
        >>> # Custom model configuration
        >>> model = ConvNeXtV2(
        ...     num_classes=10,
        ...     depths=[2, 2, 6, 2],
        ...     dims=[64, 128, 256, 512],
        ...     drop_path_rate=0.1
        ... )
    """

    # Model variant configurations
    MODEL_VARIANTS = {
        "atto": {"depths": [2, 2, 6, 2], "dims": [40, 80, 160, 320]},
        "femto": {"depths": [2, 2, 6, 2], "dims": [48, 96, 192, 384]},
        "pico": {"depths": [2, 2, 6, 2], "dims": [64, 128, 256, 512]},
        "nano": {"depths": [2, 2, 8, 2], "dims": [80, 160, 320, 640]},
        "tiny": {"depths": [3, 3, 9, 3], "dims": [96, 192, 384, 768]},
        "base": {"depths": [3, 3, 27, 3], "dims": [128, 256, 512, 1024]},
        "large": {"depths": [3, 3, 27, 3], "dims": [192, 384, 768, 1536]},
        "huge": {"depths": [3, 3, 27, 3], "dims": [352, 704, 1408, 2816]},
    }

    # Architecture constants
    STEM_KERNEL_SIZE = 4
    STEM_STRIDE = 4
    DOWNSAMPLE_KERNEL_SIZE = 2
    DOWNSAMPLE_STRIDE = 2
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
        use_gamma: bool = True,
        use_softorthonormal_regularizer: bool = False,
        include_top: bool = True,
        input_shape: Optional[Tuple[int, ...]] = None,
        **kwargs
    ):
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
        self._input_shape = input_shape

        # Initialize layer lists
        self.stem_layers = []
        self.stages = []
        self.downsample_layers = []
        self.head_layers = []

        # Set input shape for the model
        if include_top:
            # Default ImageNet input shape, but allow dynamic if input_shape provided
            if input_shape is None:
                inputs = keras.Input(shape=(224, 224, 3))
            else:
                inputs = keras.Input(shape=input_shape)
        else:
            if input_shape is None:
                raise ValueError("input_shape must be provided when include_top=False")
            inputs = keras.Input(shape=input_shape)

        # Build the model
        outputs = self._build_model(inputs)

        # Initialize the Model
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

        logger.info(
            f"Created ConvNeXt V2 model with {sum(depths)} blocks, "
            f"dims={dims}, drop_path_rate={drop_path_rate}"
        )

    def _build_model(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Build the complete ConvNeXt V2 model architecture.

        Args:
            inputs: Input tensor

        Returns:
            Output tensor
        """
        x = inputs

        # Build stem
        x = self._build_stem(x)

        # Build stages with downsampling
        for stage_idx in range(len(self.depths)):
            # Add downsampling layer (except for first stage)
            if stage_idx > 0:
                x = self._build_downsample_layer(x, self.dims[stage_idx])

            # Build stage
            x = self._build_stage(x, stage_idx)

        # Build classification head if requested
        if self.include_top:
            x = self._build_head(x)
        else:
            # For feature extraction, apply global average pooling and normalization
            x = self._build_feature_head(x)

        return x

    def _build_stem(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Build the stem (patchify) layer.

        Args:
            x: Input tensor

        Returns:
            Processed tensor after stem
        """
        # Patchify layer: 4x4 conv with stride 4
        stem_conv = keras.layers.Conv2D(
            filters=self.dims[0],
            kernel_size=self.STEM_KERNEL_SIZE,
            strides=self.STEM_STRIDE,
            padding="valid",  # No padding for patchify
            use_bias=self.use_bias,
            kernel_initializer=self.STEM_INITIALIZER,
            kernel_regularizer=self.kernel_regularizer,
            name="stem_conv"
        )
        x = stem_conv(x)

        # Layer normalization after stem
        stem_norm = keras.layers.LayerNormalization(
            epsilon=self.LAYERNORM_EPSILON,
            center=self.use_bias,
            scale=True,
            name="stem_norm"
        )
        x = stem_norm(x)

        self.stem_layers = [stem_conv, stem_norm]

        return x

    def _build_downsample_layer(
        self,
        x: keras.KerasTensor,
        output_dim: int
    ) -> keras.KerasTensor:
        """Build downsampling layer between stages.

        Args:
            x: Input tensor
            output_dim: Output channel dimension

        Returns:
            Downsampled tensor
        """
        # LayerNorm before downsampling
        downsample_norm = keras.layers.LayerNormalization(
            epsilon=self.LAYERNORM_EPSILON,
            center=self.use_bias,
            scale=True,
            name=f"downsample_norm_{len(self.downsample_layers)}"
        )
        x = downsample_norm(x)

        # 2x2 conv with stride 2 for spatial downsampling
        downsample_conv = keras.layers.Conv2D(
            filters=output_dim,
            kernel_size=self.DOWNSAMPLE_KERNEL_SIZE,
            strides=self.DOWNSAMPLE_STRIDE,
            padding="valid",
            use_bias=self.use_bias,
            kernel_initializer=self.STEM_INITIALIZER,
            kernel_regularizer=self.kernel_regularizer,
            name=f"downsample_conv_{len(self.downsample_layers)}"
        )
        x = downsample_conv(x)

        self.downsample_layers.append([downsample_norm, downsample_conv])

        return x

    def _build_stage(self, x: keras.KerasTensor, stage_idx: int) -> keras.KerasTensor:
        """Build a stage consisting of multiple ConvNext V2 blocks.

        Args:
            x: Input tensor
            stage_idx: Index of the current stage

        Returns:
            Processed tensor after the stage
        """
        stage_blocks = []
        depth = self.depths[stage_idx]
        dim = self.dims[stage_idx]

        # Calculate drop path rates for this stage
        total_blocks = sum(self.depths)
        block_start_idx = sum(self.depths[:stage_idx])

        for block_idx in range(depth):
            # Calculate current drop path rate (linearly increasing)
            current_block_idx = block_start_idx + block_idx
            if total_blocks > 1:
                current_drop_path_rate = (
                    self.drop_path_rate * current_block_idx / (total_blocks - 1)
                )
            else:
                current_drop_path_rate = 0.0

            # Create ConvNext V2 block
            block = ConvNextV2Block(
                kernel_size=self.kernel_size,
                filters=dim,
                strides=(1, 1),  # No spatial reduction within stage
                activation=self.activation,
                kernel_regularizer=self.kernel_regularizer,
                use_bias=self.use_bias,
                dropout_rate=self.dropout_rate,
                spatial_dropout_rate=self.spatial_dropout_rate,
                use_gamma=self.use_gamma,
                use_softorthonormal_regularizer=self.use_softorthonormal_regularizer,
                name=f"stage_{stage_idx}_block_{block_idx}"
            )

            # Apply block with residual connection
            residual = x
            x = block(x)

            # Add stochastic depth if specified
            if current_drop_path_rate > 0:
                stochastic_depth = keras.layers.Dropout(
                    rate=current_drop_path_rate,
                    noise_shape=(None, 1, 1, 1),  # Drop entire samples
                    name=f"stage_{stage_idx}_block_{block_idx}_drop_path"
                )
                x = stochastic_depth(x, training=True)

            # Residual connection
            x = keras.layers.Add(name=f"stage_{stage_idx}_block_{block_idx}_add")([residual, x])

            stage_blocks.append(block)

        self.stages.append(stage_blocks)

        return x

    def _build_feature_head(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Build the feature extraction head (without classification).

        Args:
            x: Input feature tensor

        Returns:
            Feature tensor after global pooling and normalization
        """
        # Global average pooling
        gap = keras.layers.GlobalAveragePooling2D(name="global_avg_pool")
        x = gap(x)

        # Layer normalization before output
        head_norm = keras.layers.LayerNormalization(
            epsilon=self.LAYERNORM_EPSILON,
            center=self.use_bias,
            scale=True,
            name="head_norm"
        )
        x = head_norm(x)

        self.head_layers = [gap, head_norm]

        return x

    def _build_head(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Build the classification head.

        Args:
            x: Input feature tensor

        Returns:
            Classification logits
        """
        # Global average pooling
        gap = keras.layers.GlobalAveragePooling2D(name="global_avg_pool")
        x = gap(x)

        # Layer normalization before classifier
        head_norm = keras.layers.LayerNormalization(
            epsilon=self.LAYERNORM_EPSILON,
            center=self.use_bias,
            scale=True,
            name="head_norm"
        )
        x = head_norm(x)

        # Classification layer
        if self.num_classes > 0:
            classifier = keras.layers.Dense(
                units=self.num_classes,
                use_bias=self.use_bias,
                kernel_initializer=self.HEAD_INITIALIZER,
                kernel_regularizer=self.kernel_regularizer,
                name="classifier"
            )
            x = classifier(x)

            self.head_layers = [gap, head_norm, classifier]
        else:
            self.head_layers = [gap, head_norm]

        return x

    @classmethod
    def from_variant(
        cls,
        variant: str,
        num_classes: int = 1000,
        **kwargs
    ) -> "ConvNeXtV2":
        """Create a ConvNeXt V2 model from a predefined variant.

        Args:
            variant: String, one of "atto", "femto", "pico", "nano",
                "tiny", "base", "large", "huge"
            num_classes: Integer, number of output classes
            **kwargs: Additional arguments passed to the constructor

        Returns:
            ConvNeXtV2 model instance

        Raises:
            ValueError: If variant is not recognized

        Example:
            >>> model = ConvNeXtV2.from_variant("tiny", num_classes=10)
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        config = cls.MODEL_VARIANTS[variant]

        logger.info(f"Creating ConvNeXt V2-{variant.upper()} model")

        return cls(
            num_classes=num_classes,
            depths=config["depths"],
            dims=config["dims"],
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
            "input_shape": self._input_shape,
        }
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ConvNeXtV2":
        """Create model from configuration.

        Args:
            config: Configuration dictionary

        Returns:
            ConvNeXtV2 model instance
        """
        # Deserialize regularizer if present
        if config.get("kernel_regularizer"):
            config["kernel_regularizer"] = keras.regularizers.deserialize(
                config["kernel_regularizer"]
            )

        return cls(**config)

    def summary(self, **kwargs):
        """Print model summary with additional information."""
        super().summary(**kwargs)

        # Print additional model information
        total_blocks = sum(self.depths)
        logger.info(f"Model configuration:")
        logger.info(f"  - Stages: {len(self.depths)}")
        logger.info(f"  - Depths: {self.depths}")
        logger.info(f"  - Dimensions: {self.dims}")
        logger.info(f"  - Total blocks: {total_blocks}")
        logger.info(f"  - Drop path rate: {self.drop_path_rate}")
        logger.info(f"  - Kernel size: {self.kernel_size}")
        logger.info(f"  - Include top: {self.include_top}")
        if self.include_top:
            logger.info(f"  - Number of classes: {self.num_classes}")


def create_convnext_v2(
    variant: str = "tiny",
    num_classes: int = 1000,
    pretrained: bool = False,
    **kwargs
) -> ConvNeXtV2:
    """Convenience function to create ConvNeXt V2 models.

    Args:
        variant: String, model variant ("atto", "femto", "pico", "nano",
            "tiny", "base", "large", "huge")
        num_classes: Integer, number of output classes
        pretrained: Boolean, whether to load pretrained weights (not implemented)
        **kwargs: Additional arguments passed to the model constructor

    Returns:
        ConvNeXtV2 model instance

    Example:
        >>> # Create ConvNeXt-Small for CIFAR-10
        >>> model = create_convnext_v2("pico", num_classes=10)
        >>>
        >>> # Create custom ConvNeXt with regularization
        >>> model = create_convnext_v2(
        ...     "base",
        ...     num_classes=100,
        ...     drop_path_rate=0.1,
        ...     kernel_regularizer=keras.regularizers.L2(1e-4)
        ... )
    """
    if pretrained:
        logger.warning("Pretrained weights are not yet implemented")

    model = ConvNeXtV2.from_variant(variant, num_classes=num_classes, **kwargs)

    return model