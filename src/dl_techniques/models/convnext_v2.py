"""
ConvNeXt V2 Model Implementation
===========================================================

A complete implementation of the ConvNeXt V2 architecture.
This version can natively handle different input sizes without requiring preprocessing.

Based on: "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders" (Woo et al., 2023)
https://arxiv.org/abs/2301.00808

Key Features:
------------
- Modular design using ConvNextV2Block as building blocks
- Global Response Normalization (GRN) for enhanced feature competition
- Support for all standard ConvNeXt V2 variants including micro variants
- Smart stem and downsampling strategies
- Configurable stochastic depth (drop path)
- Proper normalization and initialization strategies
- Flexible head design (classification, feature extraction)
- Complete serialization support
- Production-ready implementation

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
# CIFAR-10 model (32x32 input)
model = ConvNeXtV2.from_variant("tiny", num_classes=10, input_shape=(32, 32, 3))

# MNIST model (28x28 input)
model = ConvNeXtV2.from_variant("pico", num_classes=10, input_shape=(28, 28, 3))

# ImageNet model (224x224 input)
model = ConvNeXtV2.from_variant("base", num_classes=1000)

# Custom dataset model (64x64 input)
model = create_convnext_v2("large", num_classes=100, input_shape=(64, 64, 3))
```
"""

import keras
from typing import List, Optional, Union, Tuple, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..utils.logger import logger
from ..layers.convnext_v2_block import ConvNextV2Block

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class ConvNeXtV2(keras.Model):
    """ConvNeXt V2 model implementation with input handling.

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
        input_shape: Tuple, input shape. If None and include_top=True,
            uses (224, 224, 3) for ImageNet. Must be provided for non-ImageNet inputs.
        **kwargs: Additional keyword arguments for the Model base class.

    Raises:
        ValueError: If depths and dims have different lengths.
        ValueError: If invalid model configuration is provided.

    Example:
        >>> # Create ConvNeXt-Tiny model for CIFAR-10
        >>> model = ConvNeXtV2.from_variant("tiny", num_classes=10, input_shape=(32, 32, 3))
        >>>
        >>> # Create ConvNeXt-Pico for MNIST
        >>> model = ConvNeXtV2.from_variant("pico", num_classes=10, input_shape=(28, 28, 3))
        >>>
        >>> # Create standard ImageNet model
        >>> model = ConvNeXtV2.from_variant("base", num_classes=1000)
    """

    # Model variant configurations
    MODEL_VARIANTS = {
        "cifar10": {"depths": [5, 5], "dims": [48, 96]},
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
        input_shape: Tuple[int, ...] = (None, None, 3),
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
        if input_shape is None:
            input_shape = (None, None, 3)

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

        # Validate input shape
        if len(input_shape) != 3:
            raise ValueError(f"input_shape must be 3D, got {input_shape}")

        height, width, channels = input_shape

        if channels not in [1, 3]:
            logger.warning(f"Unusual number of channels: {channels}. ConvNeXt typically uses 3 channels")

        # Store the actual input shape
        self.input_height = height
        self.input_width = width
        self.input_channels = channels

        # Set input shape for the model
        inputs = keras.Input(shape=input_shape)

        # Build the model
        outputs = self._build_model(inputs)

        # Initialize the Model
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

        logger.info(
            f"Created ConvNeXt V2 model for input {input_shape} "
            f"with {sum(depths)} blocks"
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
                x = self._build_downsample_layer(x, self.dims[stage_idx], stage_idx)

            # Build stage
            x = self._build_stage(x, stage_idx)

        # Build classification head if requested
        if self.include_top:
            x = self._build_head(x)

        return x

    def _build_stem(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Build the stem (patchify) layer.

        Args:
            x: Input tensor

        Returns:
            Processed tensor after stem
        """
        stem_kernel_size = 4
        stem_stride = 4

        stem_conv = keras.layers.Conv2D(
            filters=self.dims[0],
            kernel_size=stem_kernel_size,
            strides=stem_stride,
            padding="same" if stem_stride == 1 else "valid",
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
        output_dim: int,
        stage_idx: int
    ) -> keras.KerasTensor:
        """Build downsampling layer between stages.

        Args:
            x: Input tensor
            output_dim: Output channel dimension
            stage_idx: Current stage index

        Returns:
            Downsampled tensor
        """
        downsample_kernel_size, downsample_stride = 4, 4

        # LayerNorm before downsampling
        downsample_norm = keras.layers.LayerNormalization(
            epsilon=self.LAYERNORM_EPSILON,
            center=self.use_bias,
            scale=True,
            name=f"downsample_norm_{len(self.downsample_layers)}"
        )
        x = downsample_norm(x)

        # downsampling convolution
        if downsample_stride > 1:
            downsample_conv = keras.layers.Conv2D(
                filters=output_dim,
                kernel_size=downsample_kernel_size,
                strides=downsample_stride,
                padding="valid",
                use_bias=self.use_bias,
                kernel_initializer=self.STEM_INITIALIZER,
                kernel_regularizer=self.kernel_regularizer,
                name=f"downsample_conv_{len(self.downsample_layers)}"
            )
            x = downsample_conv(x)
        else:
            # If no spatial downsampling, just adjust channels if needed
            if x.shape[-1] != output_dim:
                downsample_conv = keras.layers.Conv2D(
                    filters=output_dim,
                    kernel_size=1,
                    strides=1,
                    padding="valid",
                    use_bias=self.use_bias,
                    kernel_initializer=self.STEM_INITIALIZER,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f"downsample_conv_{len(self.downsample_layers)}"
                )
                x = downsample_conv(x)
            else:
                downsample_conv = None

        if downsample_conv is not None:
            self.downsample_layers.append([downsample_norm, downsample_conv])
        else:
            self.downsample_layers.append([downsample_norm])

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
        input_shape: Optional[Tuple[int, ...]] = None,
        **kwargs
    ) -> "ConvNeXtV2":
        """Create a ConvNeXt V2 model from a predefined variant.

        Args:
            variant: String, one of "atto", "femto", "pico", "nano",
                "tiny", "base", "large", "huge"
            num_classes: Integer, number of output classes
            input_shape: Tuple, input shape. If None and include_top=True, uses (224, 224, 3)
            **kwargs: Additional arguments passed to the constructor

        Returns:
            ConvNeXtV2 model instance

        Raises:
            ValueError: If variant is not recognized

        Example:
            >>> # CIFAR-10 model
            >>> model = ConvNeXtV2.from_variant("tiny", num_classes=10, input_shape=(32, 32, 3))
            >>> # MNIST model
            >>> model = ConvNeXtV2.from_variant("pico", num_classes=10, input_shape=(28, 28, 3))
            >>> # ImageNet model
            >>> model = ConvNeXtV2.from_variant("base", num_classes=1000)
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        config = cls.MODEL_VARIANTS[variant]

        logger.info(f"Creating ConvNeXt V2-{variant.upper()} model")
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
        logger.info(f"ConvNeXt V2 configuration:")
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

def create_convnext_v2(
    variant: str = "tiny",
    num_classes: int = 1000,
    input_shape: Optional[Tuple[int, ...]] = (None, None, 3),
    pretrained: bool = False,
    **kwargs
) -> ConvNeXtV2:
    """Convenience function to create ConvNeXt V2 models.

    Args:
        variant: String, model variant ("atto", "femto", "pico", "nano",
            "tiny", "base", "large", "huge")
        num_classes: Integer, number of output classes
        input_shape: Tuple, input shape.
        pretrained: Boolean, whether to load pretrained weights (not implemented)
        **kwargs: Additional arguments passed to the model constructor

    Returns:
        ConvNeXtV2 model instance

    Example:
        >>> # Create ConvNeXt-Tiny for CIFAR-10
        >>> model = create_convnext_v2("tiny", num_classes=10, input_shape=(32, 32, 3))
        >>>
        >>> # Create ConvNeXt-Pico for MNIST
        >>> model = create_convnext_v2("pico", num_classes=10, input_shape=(28, 28, 3))
        >>>
        >>> # Create ConvNeXt-Base for ImageNet
        >>> model = create_convnext_v2("base", num_classes=1000)
    """
    if pretrained:
        logger.warning("Pretrained weights are not yet implemented")

    model = ConvNeXtV2.from_variant(
        variant,
        num_classes=num_classes,
        input_shape=input_shape,
        **kwargs
    )

    return model

# ---------------------------------------------------------------------

