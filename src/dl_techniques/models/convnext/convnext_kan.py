"""
ConvNeXt V1 Model with KAN Linear Head Implementation
=====================================================

A ConvNeXt V1 architecture enhanced with a Kolmogorov-Arnold Network (KAN) linear
layer as the classification head. This combines the excellent spatial feature
extraction capabilities of ConvNeXt with the learnable activation functions of KAN.

The model replaces the traditional Dense classification layer with a KANLinear layer,
allowing the network to learn more flexible and expressive decision boundaries through
adaptive B-spline-based activation functions.

Key Benefits:
------------
- Improved expressiveness in the classification head
- Learnable activation functions adapt to data distribution
- Potential for better performance on complex classification tasks
- Maintains all ConvNeXt architectural benefits (efficiency, scalability)

Architecture:
------------
ConvNeXt Feature Extractor → Global Average Pooling → Layer Normalization → KAN Linear → Output

Based on:
- "A ConvNet for the 2020s" (Liu et al., 2022) - ConvNeXt architecture
- "KAN: Kolmogorov-Arnold Networks" (Liu et al., 2024) - KAN linear layers

Usage Examples:
--------------
```python
# CIFAR-10 model with KAN head
model = ConvNeXtKAN.from_variant(
    "tiny",
    num_classes=10,
    input_shape=(32, 32, 3),
    kan_grid_size=8,
    kan_spline_order=3
)

# ImageNet model with custom KAN configuration
model = ConvNeXtKAN.from_variant(
    "base",
    num_classes=1000,
    kan_grid_size=12,
    kan_activation='gelu',
    kan_regularization_factor=0.001
)
```
"""

import keras
from typing import List, Optional, Union, Tuple, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.kan_linear import KANLinear
from dl_techniques.layers.convnext_v1_block import ConvNextV1Block

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class ConvNeXtKAN(keras.Model):
    """ConvNeXt V1 model with KAN Linear classification head.

    This model enhances the standard ConvNeXt architecture by replacing the final
    Dense classification layer with a KANLinear layer. This allows the model to
    learn more flexible and expressive activation functions in the classification
    head through B-spline basis functions.

    The feature extraction backbone remains unchanged from ConvNeXt V1, ensuring
    all the proven benefits of the architecture while adding the expressiveness
    of learnable activation functions in the final layer.

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
            uses (224, 224, 3) for ImageNet.
        kan_grid_size: Integer, size of the grid for B-splines in KAN layer.
            Must be >= kan_spline_order. Defaults to 8.
        kan_spline_order: Integer, order of B-splines in KAN layer.
            Must be positive. Defaults to 3.
        kan_activation: String or callable, activation function for KAN layer.
            Defaults to 'swish'.
        kan_regularization_factor: Float, L2 regularization factor for KAN layer.
            Must be non-negative. Defaults to 0.01.
        kan_grid_range: Tuple of two floats, range for the KAN grid as (min, max).
            Defaults to (-2, 2) for better coverage of feature distributions.
        kan_use_residual: Boolean, whether to use residual connections in KAN layer.
            Defaults to True.
        kan_kernel_initializer: String or Initializer, initializer for KAN base weights.
            Defaults to 'orthogonal'.
        kan_spline_initializer: String or Initializer, initializer for KAN spline weights.
            Defaults to 'glorot_uniform'.
        kan_kernel_regularizer: Optional regularizer for KAN base weights.
        kan_spline_regularizer: Optional regularizer for KAN spline weights.
        **kwargs: Additional keyword arguments for the Model base class.

    Raises:
        ValueError: If depths and dims have different lengths.
        ValueError: If invalid model configuration is provided.
        ValueError: If KAN parameters are invalid.

    Example:
        >>> # Create ConvNeXt-KAN for CIFAR-10 with moderate KAN complexity
        >>> model = ConvNeXtKAN.from_variant(
        ...     "tiny",
        ...     num_classes=10,
        ...     input_shape=(32, 32, 3),
        ...     kan_grid_size=6,
        ...     kan_activation='gelu'
        ... )
        >>>
        >>> # Create ConvNeXt-KAN for ImageNet with high KAN expressiveness
        >>> model = ConvNeXtKAN.from_variant(
        ...     "base",
        ...     num_classes=1000,
        ...     kan_grid_size=12,
        ...     kan_spline_order=4,
        ...     kan_regularization_factor=0.005
        ... )
    """

    # Model variant configurations (inherited from ConvNeXt V1)
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
        use_gamma: bool = True,
        use_softorthonormal_regularizer: bool = True,
        include_top: bool = True,
        input_shape: Tuple[int, ...] = (None, None, 3),
        # KAN-specific parameters
        kan_grid_size: int = 8,
        kan_spline_order: int = 3,
        kan_activation: str = 'swish',
        kan_regularization_factor: float = 0.01,
        kan_grid_range: Tuple[float, float] = (-2.0, 2.0),
        kan_use_residual: bool = True,
        kan_kernel_initializer: Union[str, keras.initializers.Initializer] = 'orthogonal',
        kan_spline_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        kan_kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        kan_spline_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        **kwargs
    ):
        # Validate ConvNeXt configuration
        if len(depths) != len(dims):
            raise ValueError(
                f"Length of depths ({len(depths)}) must equal length of dims ({len(dims)})"
            )

        if len(depths) != 4:
            logger.warning(
                f"ConvNeXt typically uses 4 stages, got {len(depths)} stages"
            )

        # Validate KAN configuration
        if kan_grid_size < kan_spline_order:
            raise ValueError(
                f"KAN grid_size ({kan_grid_size}) must be >= spline_order ({kan_spline_order})"
            )

        if kan_grid_range[0] >= kan_grid_range[1]:
            raise ValueError(f"Invalid KAN grid range: {kan_grid_range}")

        if input_shape is None:
            input_shape = (None, None, 3)

        # Store ConvNeXt configuration
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

        # Store KAN configuration
        self.kan_grid_size = kan_grid_size
        self.kan_spline_order = kan_spline_order
        self.kan_activation = kan_activation
        self.kan_regularization_factor = kan_regularization_factor
        self.kan_grid_range = kan_grid_range
        self.kan_use_residual = kan_use_residual
        self.kan_kernel_initializer = kan_kernel_initializer
        self.kan_spline_initializer = kan_spline_initializer
        self.kan_kernel_regularizer = kan_kernel_regularizer
        self.kan_spline_regularizer = kan_spline_regularizer

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
            f"Created ConvNeXt-KAN model for input {input_shape} "
            f"with {sum(depths)} blocks and KAN head (grid_size={kan_grid_size})"
        )

    def _build_model(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Build the complete ConvNeXt-KAN model architecture.

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

        # Build KAN classification head if requested
        if self.include_top:
            x = self._build_kan_head(x)

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

        # Downsampling convolution
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

            # Create ConvNext block
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

    def _build_kan_head(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Build the KAN-based classification head.

        This replaces the traditional Dense layer with a KANLinear layer,
        providing learnable activation functions in the classification head.

        Args:
            x: Input feature tensor

        Returns:
            Classification logits
        """
        # Global average pooling
        gap = keras.layers.GlobalAveragePooling2D(name="global_avg_pool")
        x = gap(x)

        # Layer normalization before KAN classifier
        head_norm = keras.layers.LayerNormalization(
            epsilon=self.LAYERNORM_EPSILON,
            center=self.use_bias,
            scale=True,
            name="head_norm"
        )
        x = head_norm(x)

        # KAN Linear classification layer
        if self.num_classes > 0:
            kan_classifier = KANLinear(
                features=self.num_classes,
                grid_size=self.kan_grid_size,
                spline_order=self.kan_spline_order,
                activation=self.kan_activation,
                regularization_factor=self.kan_regularization_factor,
                grid_range=self.kan_grid_range,
                use_residual=self.kan_use_residual,
                kernel_initializer=self.kan_kernel_initializer,
                spline_initializer=self.kan_spline_initializer,
                kernel_regularizer=self.kan_kernel_regularizer,
                spline_regularizer=self.kan_spline_regularizer,
                name="kan_classifier"
            )
            x = kan_classifier(x)

            self.head_layers = [gap, head_norm, kan_classifier]
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
    ) -> "ConvNeXtKAN":
        """Create a ConvNeXt-KAN model from a predefined variant.

        Args:
            variant: String, one of "tiny", "small", "base", "large", "xlarge"
            num_classes: Integer, number of output classes
            input_shape: Tuple, input shape. If None and include_top=True, uses (224, 224, 3)
            **kwargs: Additional arguments passed to the constructor, including KAN parameters

        Returns:
            ConvNeXtKAN model instance

        Raises:
            ValueError: If variant is not recognized

        Example:
            >>> # CIFAR-10 model with moderate KAN complexity
            >>> model = ConvNeXtKAN.from_variant(
            ...     "tiny",
            ...     num_classes=10,
            ...     input_shape=(32, 32, 3),
            ...     kan_grid_size=6,
            ...     kan_activation='gelu'
            ... )
            >>>
            >>> # ImageNet model with high expressiveness
            >>> model = ConvNeXtKAN.from_variant(
            ...     "base",
            ...     num_classes=1000,
            ...     kan_grid_size=12,
            ...     kan_spline_order=4
            ... )
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        config = cls.MODEL_VARIANTS[variant]

        logger.info(f"Creating ConvNeXt-KAN-{variant.upper()} model")
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
            # ConvNeXt configuration
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
            # KAN configuration
            "kan_grid_size": self.kan_grid_size,
            "kan_spline_order": self.kan_spline_order,
            "kan_activation": self.kan_activation,
            "kan_regularization_factor": self.kan_regularization_factor,
            "kan_grid_range": self.kan_grid_range,
            "kan_use_residual": self.kan_use_residual,
            "kan_kernel_initializer": keras.initializers.serialize(
                keras.initializers.get(self.kan_kernel_initializer)
            ),
            "kan_spline_initializer": keras.initializers.serialize(
                keras.initializers.get(self.kan_spline_initializer)
            ),
            "kan_kernel_regularizer": keras.regularizers.serialize(self.kan_kernel_regularizer),
            "kan_spline_regularizer": keras.regularizers.serialize(self.kan_spline_regularizer),
        }
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ConvNeXtKAN":
        """Create model from configuration.

        Args:
            config: Configuration dictionary

        Returns:
            ConvNeXtKAN model instance
        """
        # Deserialize regularizers if present
        if config.get("kernel_regularizer"):
            config["kernel_regularizer"] = keras.regularizers.deserialize(
                config["kernel_regularizer"]
            )
        if config.get("kan_kernel_regularizer"):
            config["kan_kernel_regularizer"] = keras.regularizers.deserialize(
                config["kan_kernel_regularizer"]
            )
        if config.get("kan_spline_regularizer"):
            config["kan_spline_regularizer"] = keras.regularizers.deserialize(
                config["kan_spline_regularizer"]
            )

        # Deserialize initializers if present
        if config.get("kan_kernel_initializer"):
            config["kan_kernel_initializer"] = keras.initializers.deserialize(
                config["kan_kernel_initializer"]
            )
        if config.get("kan_spline_initializer"):
            config["kan_spline_initializer"] = keras.initializers.deserialize(
                config["kan_spline_initializer"]
            )

        return cls(**config)

    def summary(self, **kwargs):
        """Print model summary with additional information."""
        super().summary(**kwargs)

        # Print additional model information
        total_blocks = sum(self.depths)
        logger.info(f"ConvNeXt-KAN configuration:")
        logger.info(f"  - Input shape: ({self.input_height}, {self.input_width}, {self.input_channels})")
        logger.info(f"  - Stages: {len(self.depths)}")
        logger.info(f"  - Depths: {self.depths}")
        logger.info(f"  - Dimensions: {self.dims}")
        logger.info(f"  - Total blocks: {total_blocks}")
        logger.info(f"  - Drop path rate: {self.drop_path_rate}")
        logger.info(f"  - Kernel size: {self.kernel_size}")
        logger.info(f"  - Include top: {self.include_top}")
        if self.include_top:
            logger.info(f"  - Number of classes: {self.num_classes}")
            logger.info(f"  - KAN grid size: {self.kan_grid_size}")
            logger.info(f"  - KAN spline order: {self.kan_spline_order}")
            logger.info(f"  - KAN activation: {self.kan_activation}")
            logger.info(f"  - KAN grid range: {self.kan_grid_range}")

# ---------------------------------------------------------------------

def create_convnext_kan(
    variant: str = "tiny",
    num_classes: int = 1000,
    input_shape: Optional[Tuple[int, ...]] = (None, None, 3),
    pretrained: bool = False,
    **kwargs
) -> ConvNeXtKAN:
    """Convenience function to create ConvNeXt-KAN models.

    Args:
        variant: String, model variant ("tiny", "small", "base", "large", "xlarge")
        num_classes: Integer, number of output classes
        input_shape: Tuple, input shape.
        pretrained: Boolean, whether to load pretrained weights (not implemented)
        **kwargs: Additional arguments passed to the model constructor, including KAN parameters

    Returns:
        ConvNeXtKAN model instance

    Example:
        >>> # Create ConvNeXt-KAN-Tiny for CIFAR-10
        >>> model = create_convnext_kan(
        ...     "tiny",
        ...     num_classes=10,
        ...     input_shape=(32, 32, 3),
        ...     kan_grid_size=8,
        ...     kan_activation='gelu'
        ... )
        >>>
        >>> # Create ConvNeXt-KAN-Base for ImageNet with custom KAN settings
        >>> model = create_convnext_kan(
        ...     "base",
        ...     num_classes=1000,
        ...     kan_grid_size=12,
        ...     kan_spline_order=4,
        ...     kan_regularization_factor=0.005
        ... )
    """
    if pretrained:
        logger.warning("Pretrained weights are not yet implemented for ConvNeXt-KAN models")

    model = ConvNeXtKAN.from_variant(
        variant,
        num_classes=num_classes,
        input_shape=input_shape,
        **kwargs
    )

    return model

# ---------------------------------------------------------------------