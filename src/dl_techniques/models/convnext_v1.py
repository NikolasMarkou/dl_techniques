"""
ConvNeXt V1 Model Implementation
==================================================

A complete implementation of the ConvNeXt V1 architecture with adaptive input handling.
This version can natively handle different input sizes without requiring preprocessing.

Based on: "A ConvNet for the 2020s" (Liu et al., 2022)
https://arxiv.org/abs/2201.03545

Key Features:
------------
- Modular design using ConvNextV1Block as building blocks
- Support for all standard ConvNeXt variants
- Adaptive input size handling (16x16 to any larger size)
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
- Adaptive channel scaling for very small inputs
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

from dl_techniques.utils.logger import logger
from dl_techniques.layers.convnext_v1_block import ConvNextV1Block


@keras.saving.register_keras_serializable()
class ConvNeXtV1(keras.Model):
    """ConvNeXt V1 model implementation with adaptive input handling.

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

        # Determine input shape with improved logic
        actual_input_shape = self._determine_input_shape(include_top, input_shape)

        # Validate input shape
        if len(actual_input_shape) != 3:
            raise ValueError(f"input_shape must be 3D, got {actual_input_shape}")

        height, width, channels = actual_input_shape
        if height < 16 or width < 16:
            raise ValueError(f"Input size too small: {height}x{width}. Minimum is 16x16")

        if channels not in [1, 3]:
            logger.warning(f"Unusual number of channels: {channels}. ConvNeXt typically uses 3 channels")

        # Store the actual input shape for stem adaptation
        self.input_height = height
        self.input_width = width
        self.input_channels = channels

        # Adapt dimensions for very small inputs
        self.adapted_dims = self._adapt_dimensions_for_input_size(dims, height, width)

        # Set input shape for the model
        inputs = keras.Input(shape=actual_input_shape)

        # Build the model
        outputs = self._build_model(inputs)

        # Initialize the Model
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

        logger.info(
            f"Created ConvNeXt V1 model for input {actual_input_shape} "
            f"with {sum(depths)} blocks"
        )

    def _determine_input_shape(self, include_top: bool, input_shape: Optional[Tuple[int, ...]]) -> Tuple[int, int, int]:
        """Determine the actual input shape to use for the model.

        Args:
            include_top: Whether the model includes the classification head
            input_shape: Provided input shape

        Returns:
            Tuple representing the actual input shape to use

        Raises:
            ValueError: If input_shape is required but not provided
        """
        if input_shape is not None:
            # Use provided input shape
            actual_input_shape = input_shape
            logger.info(f"Using provided input shape: {actual_input_shape}")
        elif include_top:
            # Default ImageNet input shape for classification models
            actual_input_shape = (224, 224, 3)
            logger.info(f"Using default ImageNet input shape: {actual_input_shape}")
        else:
            # For feature extraction models, input_shape must be provided
            raise ValueError("input_shape must be provided when include_top=False")

        return actual_input_shape

    def _adapt_dimensions_for_input_size(self, dims: List[int], height: int, width: int) -> List[int]:
        """Adapt channel dimensions based on input size."""
        min_size = min(height, width)

        # For very small inputs, reduce dimensions to prevent overfitting
        if min_size <= 32:
            # Scale down dimensions for small inputs like CIFAR/MNIST
            scale_factor = 0.75
            adapted_dims = [max(32, int(d * scale_factor)) for d in dims]
            logger.info(f"Adapted dimensions for small input {height}x{width}: {dims} -> {adapted_dims}")
            return adapted_dims
        elif min_size <= 64:
            # Slight reduction for medium-small inputs
            scale_factor = 0.875
            adapted_dims = [max(48, int(d * scale_factor)) for d in dims]
            logger.info(f"Adapted dimensions for medium input {height}x{width}: {dims} -> {adapted_dims}")
            return adapted_dims
        else:
            # Use original dimensions for larger inputs
            return dims

    def _get_stem_config(self) -> Tuple[int, int]:
        """Determine stem kernel size and stride based on input size."""
        min_size = min(self.input_height, self.input_width)

        if min_size <= 32:
            # Small inputs (MNIST 28x28, CIFAR 32x32): gentle downsampling
            kernel_size, stride = 3, 1
            logger.info(f"Using small-input stem: {kernel_size}x{kernel_size} conv, stride {stride}")
        elif min_size <= 64:
            # Medium inputs: moderate downsampling
            kernel_size, stride = 4, 2
            logger.info(f"Using medium-input stem: {kernel_size}x{kernel_size} conv, stride {stride}")
        elif min_size <= 128:
            # Medium-large inputs: standard downsampling
            kernel_size, stride = 4, 3
            logger.info(f"Using medium-large-input stem: {kernel_size}x{kernel_size} conv, stride {stride}")
        else:
            # Large inputs (ImageNet): original ConvNeXt stem
            kernel_size, stride = 4, 4
            logger.info(f"Using large-input stem: {kernel_size}x{kernel_size} conv, stride {stride}")

        return kernel_size, stride

    def _get_downsample_config(self, stage_idx: int) -> Tuple[int, int]:
        """Determine downsampling configuration based on input size and stage."""
        min_size = min(self.input_height, self.input_width)

        # Calculate current feature map size after stem and previous downsamples
        stem_kernel, stem_stride = self._get_stem_config()
        current_size = min_size // stem_stride
        for i in range(stage_idx):
            current_size = current_size // 2  # Each downsample halves the size

        # If feature map would become too small, skip downsampling
        if current_size <= 4:
            kernel_size, stride = 1, 1  # No downsampling
            logger.info(f"Stage {stage_idx}: Skipping downsample (feature map too small: {current_size}x{current_size})")
        else:
            kernel_size, stride = 2, 2  # Standard downsampling

        return kernel_size, stride

    def _build_model(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Build the complete ConvNeXt model architecture.

        Args:
            inputs: Input tensor

        Returns:
            Output tensor
        """
        x = inputs

        # Build adaptive stem
        x = self._build_stem(x)

        # Build stages with adaptive downsampling
        for stage_idx in range(len(self.depths)):
            # Add downsampling layer (except for first stage)
            if stage_idx > 0:
                x = self._build_downsample_layer(x, self.adapted_dims[stage_idx], stage_idx)

            # Build stage
            x = self._build_stage(x, stage_idx)

        # Build classification head if requested
        if self.include_top:
            x = self._build_head(x)

        return x

    def _build_stem(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Build the adaptive stem (patchify) layer.

        Args:
            x: Input tensor

        Returns:
            Processed tensor after stem
        """
        stem_kernel_size, stem_stride = self._get_stem_config()

        # Adaptive stem convolution
        stem_conv = keras.layers.Conv2D(
            filters=self.adapted_dims[0],
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
        """Build adaptive downsampling layer between stages.

        Args:
            x: Input tensor
            output_dim: Output channel dimension
            stage_idx: Current stage index

        Returns:
            Downsampled tensor
        """
        downsample_kernel_size, downsample_stride = self._get_downsample_config(stage_idx)

        # LayerNorm before downsampling
        downsample_norm = keras.layers.LayerNormalization(
            epsilon=self.LAYERNORM_EPSILON,
            center=self.use_bias,
            scale=True,
            name=f"downsample_norm_{len(self.downsample_layers)}"
        )
        x = downsample_norm(x)

        # Adaptive downsampling convolution
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
        """Build a stage consisting of multiple ConvNext blocks.

        Args:
            x: Input tensor
            stage_idx: Index of the current stage

        Returns:
            Processed tensor after the stage
        """
        stage_blocks = []
        depth = self.depths[stage_idx]
        dim = self.adapted_dims[stage_idx]

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

            # Create ConvNext block
            block = ConvNextV1Block(
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
            "input_shape": self._input_shape,
        }
        return config

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
        super().summary(**kwargs)

        # Print additional model information
        total_blocks = sum(self.depths)
        logger.info(f"Adaptive ConvNeXt V1 configuration:")
        logger.info(f"  - Input shape: ({self.input_height}, {self.input_width}, {self.input_channels})")
        logger.info(f"  - Stages: {len(self.depths)}")
        logger.info(f"  - Depths: {self.depths}")
        logger.info(f"  - Original dimensions: {self.dims}")
        logger.info(f"  - Adapted dimensions: {self.adapted_dims}")
        logger.info(f"  - Total blocks: {total_blocks}")
        logger.info(f"  - Drop path rate: {self.drop_path_rate}")
        logger.info(f"  - Kernel size: {self.kernel_size}")
        logger.info(f"  - Include top: {self.include_top}")
        if self.include_top:
            logger.info(f"  - Number of classes: {self.num_classes}")


def create_convnext_v1(
    variant: str = "tiny",
    num_classes: int = 1000,
    input_shape: Optional[Tuple[int, ...]] = None,
    pretrained: bool = False,
    **kwargs
) -> ConvNeXtV1:
    """Convenience function to create adaptive ConvNeXt V1 models.

    Args:
        variant: String, model variant ("tiny", "small", "base", "large", "xlarge")
        num_classes: Integer, number of output classes
        input_shape: Tuple, input shape. If None and include_top=True, uses (224, 224, 3)
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