"""
ConvNeXt V1 Model Implementation with Pretrained Support
==================================================

A complete implementation of the ConvNeXt V1 architecture with support for
loading pretrained weights. This version can natively handle different input
sizes without requiring preprocessing.

Based on: "A ConvNet for the 2020s" (Liu et al., 2022)
https://arxiv.org/abs/2201.03545

Key Features:
------------
- Modular design using ConvNextV1Block as building blocks
- Support for all standard ConvNeXt variants
- Pretrained weight loading from URLs or local paths
- Smart handling of classifier layer mismatches
- Smart stem and downsampling strategies
- Configurable stochastic depth (drop path)
- Proper normalization and initialization strategies
- Flexible head design (classification, feature extraction)
- Complete serialization support
- Production-ready implementation

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
# Load pretrained ImageNet weights
model = ConvNeXtV1.from_variant("tiny", pretrained=True, num_classes=1000)

# Load pretrained as feature extractor
model = ConvNeXtV1.from_variant("base", pretrained=True, include_top=False)

# Fine-tune on custom dataset
model = create_convnext_v1("small", num_classes=10, input_shape=(32, 32, 3),
                           pretrained=True, weights_input_shape=(224, 224, 3))

# Load from local weights file
model = ConvNeXtV1.from_variant("large", pretrained="path/to/weights.keras")
```
"""

import keras
import os
from typing import List, Optional, Union, Tuple, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.convnext_v1_block import ConvNextV1Block


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class ConvNeXtV1(keras.Model):
    """ConvNeXt V1 model implementation with pretrained support

    A modern ConvNet architecture that achieves competitive performance
    with Vision Transformers while maintaining the simplicity and efficiency
    of convolutional networks. This version adapts to different input sizes
    and supports loading pretrained weights.

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
        >>> # Load pretrained ImageNet model
        >>> model = ConvNeXtV1.from_variant("tiny", pretrained=True)
        >>>
        >>> # Load as feature extractor
        >>> model = ConvNeXtV1.from_variant("base", pretrained=True, include_top=False)
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

    # Pretrained weights URLs (update these with actual URLs when available)
    PRETRAINED_WEIGHTS = {
        "tiny": {
            "imagenet": "https://example.com/convnext_tiny_imagenet.keras",
            "imagenet22k": "https://example.com/convnext_tiny_imagenet22k.keras",
        },
        "small": {
            "imagenet": "https://example.com/convnext_small_imagenet.keras",
            "imagenet22k": "https://example.com/convnext_small_imagenet22k.keras",
        },
        "base": {
            "imagenet": "https://example.com/convnext_base_imagenet.keras",
            "imagenet22k": "https://example.com/convnext_base_imagenet22k.keras",
        },
        "large": {
            "imagenet": "https://example.com/convnext_large_imagenet.keras",
            "imagenet22k": "https://example.com/convnext_large_imagenet22k.keras",
        },
        "xlarge": {
            "imagenet": "https://example.com/convnext_xlarge_imagenet.keras",
            "imagenet22k": "https://example.com/convnext_xlarge_imagenet22k.keras",
        },
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

        if strides <= 0:
            raise ValueError(
                f"Strides {strides} must be positive."
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
                rate=drop_rate,
                noise_shape=(None, 1, 1, 1),
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
        """Defines the forward pass of the model.

        Args:
            inputs: Input tensor of shape (batch_size, height, width, channels).
            training: Boolean or None, whether the model is in training mode.

        Returns:
            Output tensor. Shape depends on include_top:
                - If include_top=True: (batch_size, num_classes)
                - If include_top=False: (batch_size, H', W', channels)
        """
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

    def load_pretrained_weights(
            self,
            weights_path: str,
            skip_mismatch: bool = True,
            by_name: bool = True
    ) -> None:
        """Load pretrained weights into the model.

        This method handles loading weights with smart mismatch handling,
        particularly useful when the number of classes differs or when
        loading weights without the top classifier.

        Args:
            weights_path: String, path to the weights file (.keras format).
            skip_mismatch: Boolean, whether to skip layers with mismatched shapes.
                Useful when loading weights with different num_classes.
            by_name: Boolean, whether to load weights by layer name.

        Raises:
            FileNotFoundError: If weights_path doesn't exist.
            ValueError: If weights cannot be loaded.

        Example:
            >>> model = ConvNeXtV1.from_variant("tiny", num_classes=10)
            >>> model.load_pretrained_weights("convnext_tiny_imagenet.keras", skip_mismatch=True)
        """
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        try:
            # Build model if not already built
            if not self.built:
                dummy_input = keras.random.normal((1,) + tuple(self.input_shape))
                self(dummy_input, training=False)

            logger.info(f"Loading pretrained weights from {weights_path}")

            # Load weights with appropriate settings
            self.load_weights(
                weights_path,
                skip_mismatch=skip_mismatch,
                by_name=by_name
            )

            if skip_mismatch:
                logger.info(
                    "Weights loaded with skip_mismatch=True. "
                    "Layers with shape mismatches were skipped (e.g., classifier layer)."
                )
            else:
                logger.info("All weights loaded successfully.")

        except Exception as e:
            raise ValueError(f"Failed to load weights from {weights_path}: {str(e)}")

    @staticmethod
    def _download_weights(
            variant: str,
            dataset: str = "imagenet",
            cache_dir: Optional[str] = None
    ) -> str:
        """Download pretrained weights from URL.

        Args:
            variant: String, model variant name.
            dataset: String, dataset the weights were trained on.
                Options: "imagenet", "imagenet22k".
            cache_dir: Optional string, directory to cache downloaded weights.
                If None, uses default Keras cache directory.

        Returns:
            String, path to the downloaded weights file.

        Raises:
            ValueError: If variant or dataset is not available.
        """
        if variant not in ConvNeXtV1.PRETRAINED_WEIGHTS:
            raise ValueError(
                f"No pretrained weights available for variant '{variant}'. "
                f"Available variants: {list(ConvNeXtV1.PRETRAINED_WEIGHTS.keys())}"
            )

        if dataset not in ConvNeXtV1.PRETRAINED_WEIGHTS[variant]:
            raise ValueError(
                f"No pretrained weights available for dataset '{dataset}'. "
                f"Available datasets for {variant}: "
                f"{list(ConvNeXtV1.PRETRAINED_WEIGHTS[variant].keys())}"
            )

        url = ConvNeXtV1.PRETRAINED_WEIGHTS[variant][dataset]

        logger.info(f"Downloading ConvNeXt-{variant} weights from {dataset}...")

        # Download weights using Keras utility
        weights_path = keras.utils.get_file(
            fname=f"convnext_{variant}_{dataset}.keras",
            origin=url,
            cache_dir=cache_dir,
            cache_subdir="models/convnext_v1"
        )

        logger.info(f"Weights downloaded to: {weights_path}")
        return weights_path

    @classmethod
    def from_variant(
            cls,
            variant: str,
            num_classes: int = 1000,
            input_shape: Optional[Tuple[int, ...]] = None,
            pretrained: Union[bool, str] = False,
            weights_dataset: str = "imagenet",
            weights_input_shape: Optional[Tuple[int, ...]] = None,
            cache_dir: Optional[str] = None,
            **kwargs
    ) -> "ConvNeXtV1":
        """Create a ConvNeXt model from a predefined variant.

        Args:
            variant: String, one of "tiny", "small", "base", "large", "xlarge"
            num_classes: Integer, number of output classes
            input_shape: Tuple, input shape. If None and include_top=True, uses (224, 224, 3)
            pretrained: Boolean or string. If True, loads pretrained weights from
                default URL. If string, treats it as a path to local weights file.
            weights_dataset: String, dataset for pretrained weights.
                Options: "imagenet", "imagenet22k". Only used if pretrained=True.
            weights_input_shape: Tuple, input shape used during weight pretraining.
                Only needed if loading pretrained weights with different input_shape.
                Defaults to (224, 224, 3) for ImageNet weights.
            cache_dir: Optional string, directory to cache downloaded weights.
            **kwargs: Additional arguments passed to the constructor

        Returns:
            ConvNeXtV1 model instance

        Raises:
            ValueError: If variant is not recognized

        Example:
            >>> # Load pretrained ImageNet model
            >>> model = ConvNeXtV1.from_variant("tiny", pretrained=True)
            >>>
            >>> # Load pretrained as feature extractor for fine-tuning
            >>> model = ConvNeXtV1.from_variant("base", pretrained=True, include_top=False)
            >>>
            >>> # Fine-tune on custom dataset with different input size
            >>> model = ConvNeXtV1.from_variant(
            ...     "small",
            ...     num_classes=10,
            ...     input_shape=(32, 32, 3),
            ...     pretrained=True,
            ...     weights_input_shape=(224, 224, 3)
            ... )
            >>>
            >>> # Load from local weights file
            >>> model = ConvNeXtV1.from_variant("large", pretrained="path/to/weights.keras")
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        config = cls.MODEL_VARIANTS[variant]

        logger.info(f"Creating ConvNeXt-{variant.upper()} model")
        logger.info(f"from_variant received input_shape: {input_shape}")

        # Handle pretrained weights
        load_weights_path = None
        skip_mismatch = False

        if pretrained:
            if isinstance(pretrained, str):
                # Load from local file path
                load_weights_path = pretrained
                logger.info(f"Will load weights from local file: {load_weights_path}")
            else:
                # Download from URL
                try:
                    load_weights_path = cls._download_weights(
                        variant=variant,
                        dataset=weights_dataset,
                        cache_dir=cache_dir
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to download pretrained weights: {str(e)}. "
                        f"Continuing with random initialization."
                    )
                    load_weights_path = None

            # Determine if we need to skip mismatches
            include_top = kwargs.get("include_top", True)
            if include_top:
                # Check if num_classes matches pretrained weights
                pretrained_classes = 1000 if weights_dataset == "imagenet" else 21841
                if num_classes != pretrained_classes:
                    skip_mismatch = True
                    logger.info(
                        f"num_classes ({num_classes}) differs from pretrained "
                        f"({pretrained_classes}). Will skip classifier weights."
                    )

            # Handle different input shapes
            if weights_input_shape and input_shape and weights_input_shape != input_shape:
                logger.info(
                    f"Loading weights pretrained on {weights_input_shape} "
                    f"for model with input shape {input_shape}. "
                    f"Only backbone weights will be loaded."
                )
                skip_mismatch = True

        # Create model
        model = cls(
            num_classes=num_classes,
            depths=config["depths"],
            dims=config["dims"],
            input_shape=input_shape,
            **kwargs
        )

        # Load pretrained weights if available
        if load_weights_path:
            try:
                model.load_pretrained_weights(
                    weights_path=load_weights_path,
                    skip_mismatch=skip_mismatch,
                    by_name=True
                )
            except Exception as e:
                logger.error(f"Failed to load pretrained weights: {str(e)}")
                raise

        return model

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
        pretrained: Union[bool, str] = False,
        weights_dataset: str = "imagenet",
        weights_input_shape: Optional[Tuple[int, ...]] = None,
        cache_dir: Optional[str] = None,
        **kwargs
) -> ConvNeXtV1:
    """Convenience function to create ConvNeXt V1 models.

    Args:
        variant: String, model variant ("tiny", "small", "base", "large", "xlarge")
        num_classes: Integer, number of output classes
        input_shape: Tuple, input shape.
        pretrained: Boolean or string. If True, loads pretrained weights from
            default URL. If string, treats it as a path to local weights file.
        weights_dataset: String, dataset for pretrained weights.
            Options: "imagenet", "imagenet22k". Only used if pretrained=True.
        weights_input_shape: Tuple, input shape used during weight pretraining.
            Only needed if loading pretrained weights with different input_shape.
        cache_dir: Optional string, directory to cache downloaded weights.
        **kwargs: Additional arguments passed to the model constructor

    Returns:
        ConvNeXtV1 model instance

    Example:
        >>> # Create ConvNeXt-Tiny with pretrained ImageNet weights
        >>> model = create_convnext_v1("tiny", pretrained=True)
        >>>
        >>> # Create ConvNeXt-Base as feature extractor
        >>> model = create_convnext_v1("base", pretrained=True, include_top=False)
        >>>
        >>> # Fine-tune on CIFAR-10 with pretrained backbone
        >>> model = create_convnext_v1(
        ...     "small",
        ...     num_classes=10,
        ...     input_shape=(32, 32, 3),
        ...     pretrained=True,
        ...     weights_input_shape=(224, 224, 3)
        ... )
        >>>
        >>> # Load from local weights
        >>> model = create_convnext_v1("large", pretrained="path/to/weights.keras")
    """
    model = ConvNeXtV1.from_variant(
        variant,
        num_classes=num_classes,
        input_shape=input_shape,
        pretrained=pretrained,
        weights_dataset=weights_dataset,
        weights_input_shape=weights_input_shape,
        cache_dir=cache_dir,
        **kwargs
    )

    return model

# ---------------------------------------------------------------------