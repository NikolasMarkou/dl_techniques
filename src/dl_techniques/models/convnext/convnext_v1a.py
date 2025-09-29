"""
ConvNeXt V1 Model Implementation with Configurable FFN and Normalization
=======================================================================

A highly configurable implementation of the ConvNeXt V1 architecture with support for
different FFN types and normalization strategies through factory patterns.

Based on: "A ConvNet for the 2020s" (Liu et al., 2022)
https://arxiv.org/abs/2201.03545

Key Features:
------------
- Configurable normalization layers via normalization factory
- Configurable FFN layers via FFN factory
- Full support for pretrained weights
- Dynamic input size handling

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
# Load with configurable normalization
model = ConvNeXtV1.from_variant(
    "tiny",
    pretrained=True,
    normalization_type='rms_norm'
)

# Use different FFN type
model = ConvNeXtV1.from_variant(
    "base",
    pretrained=True,
    ffn_type='swiglu',
    normalization_type='layer_norm'
)

# Fine-tune with custom configuration
model = create_convnext_v1(
    "small",
    num_classes=10,
    input_shape=(32, 32, 3),
    pretrained=True,
    ffn_type='geglu',
    normalization_type='rms_norm',
    normalization_kwargs={'epsilon': 1e-5},
    ffn_kwargs={'dropout_rate': 0.1}
)
```
"""

import os
import keras
from typing import List, Optional, Union, Tuple, Dict, Any, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.layers.ffn.factory import create_ffn_from_config

# ---------------------------------------------------------------------
# Type definitions
# ---------------------------------------------------------------------

NormalizationType = Literal[
    'layer_norm', 'rms_norm', 'batch_norm', 'band_rms',
    'adaptive_band_rms', 'band_logit_norm', 'global_response_norm',
    'logit_norm', 'max_logit_norm', 'decoupled_max_logit',
    'dml_plus_focal', 'dml_plus_center', 'dynamic_tanh',
    'zero_centered_rms_norm'
]

FFNType = Literal[
    'mlp', 'swiglu', 'differential', 'glu', 'geglu',
    'residual', 'swin_mlp'
]


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class ConvNextV1Block(keras.layers.Layer):
    """
    Configurable ConvNext V1 block with factory-based normalization and FFN.

    The block consists of:
    1. Depthwise convolution (spatial mixing)
    2. Normalization layer (factory-created)
    3. FFN layer (factory-created) - replaces pointwise convolutions
    4. Optional learnable scaling (gamma)

    Args:
        kernel_size: Integer or tuple, kernel size for depthwise convolution.
        filters: Integer, number of output channels.
        normalization_type: String, type of normalization layer.
        ffn_type: String, type of FFN layer.
        activation: String or callable, activation function for FFN.
        use_bias: Boolean, whether to use bias in convolutions.
        kernel_regularizer: Regularizer function applied to kernels.
        dropout_rate: Float, dropout rate for FFN.
        spatial_dropout_rate: Float, spatial dropout rate (deprecated).
        use_gamma: Boolean, whether to use learnable scaling.
        use_softorthonormal_regularizer: Boolean, regularization flag (deprecated).
        normalization_kwargs: Optional dict, custom arguments for normalization.
        ffn_kwargs: Optional dict, custom arguments for FFN.
        **kwargs: Additional keyword arguments for the Layer base class.

    Example:
        >>> block = ConvNextV1Block(
        ...     kernel_size=7,
        ...     filters=96,
        ...     normalization_type='rms_norm',
        ...     ffn_type='swiglu'
        ... )
    """

    def __init__(
            self,
            kernel_size: Union[int, Tuple[int, int]],
            filters: int,
            normalization_type: NormalizationType = 'layer_norm',
            ffn_type: FFNType = 'mlp',
            activation: Union[str, keras.layers.Activation] = 'gelu',
            use_bias: bool = True,
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            dropout_rate: float = 0.0,
            spatial_dropout_rate: float = 0.0,
            use_gamma: bool = True,
            use_softorthonormal_regularizer: bool = False,
            normalization_kwargs: Optional[Dict[str, Any]] = None,
            ffn_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs
    ):
        super().__init__(**kwargs)

        # Store configuration
        self.kernel_size = kernel_size
        self.filters = filters
        self.normalization_type = normalization_type
        self.ffn_type = ffn_type
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_regularizer = kernel_regularizer
        self.dropout_rate = dropout_rate
        self.spatial_dropout_rate = spatial_dropout_rate
        self.use_gamma = use_gamma
        self.use_softorthonormal_regularizer = use_softorthonormal_regularizer
        self.normalization_kwargs = normalization_kwargs or {}
        self.ffn_kwargs = ffn_kwargs or {}

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer by creating all sublayers.

        Args:
            input_shape: Shape tuple of the input.
        """
        # Depthwise convolution for spatial mixing
        self.depthwise_conv = keras.layers.DepthwiseConv2D(
            kernel_size=self.kernel_size,
            padding='same',
            use_bias=self.use_bias,
            depthwise_regularizer=self.kernel_regularizer,
            name=f"{self.name}_dwconv"
        )

        # Normalization layer via factory
        self.norm = create_normalization_layer(
            normalization_type=self.normalization_type,
            name=f"{self.name}_norm",
            **self.normalization_kwargs
        )

        # FFN layer via factory (replaces the two pointwise convolutions)
        # Prepare FFN configuration
        ffn_config = {
            'ffn_type': self.ffn_type,
            'output_dim': self.filters,
            'activation': self.activation,
            'use_bias': self.use_bias,
            'kernel_regularizer': self.kernel_regularizer,
            'dropout_rate': self.dropout_rate,
            'name': f"{self.name}_ffn"
        }

        # Add FFN-specific parameters
        if self.ffn_type == 'mlp':
            # MLP needs hidden_dim (typically 4x expansion)
            ffn_config['hidden_dim'] = self.filters * 4
        elif self.ffn_type in ['swiglu', 'glu', 'geglu']:
            # Gated FFNs typically don't need hidden_dim specified
            # They use ffn_expansion_factor instead
            if 'ffn_expansion_factor' not in self.ffn_kwargs:
                ffn_config['ffn_expansion_factor'] = 4
        elif self.ffn_type in ['differential', 'residual']:
            ffn_config['hidden_dim'] = self.filters * 4
        elif self.ffn_type == 'swin_mlp':
            ffn_config['hidden_dim'] = self.filters * 4

        # Merge with user-provided kwargs
        ffn_config.update(self.ffn_kwargs)

        # Create FFN layer
        self.ffn = create_ffn_from_config(ffn_config)

        # Learnable scaling parameter (gamma)
        if self.use_gamma:
            self.gamma = self.add_weight(
                name=f"{self.name}_gamma",
                shape=(self.filters,),
                initializer=keras.initializers.Zeros(),
                trainable=True
            )

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass of the ConvNext block.

        Args:
            inputs: Input tensor of shape (batch_size, height, width, channels).
            training: Boolean or None, whether the model is in training mode.

        Returns:
            Output tensor with same shape as input.
        """
        # Depthwise convolution for spatial mixing
        x = self.depthwise_conv(inputs)

        # Normalization
        x = self.norm(x)

        # FFN for channel mixing (replaces the pointwise conv -> activation -> pointwise conv)
        x = self.ffn(x, training=training)

        # Apply learnable scaling if enabled
        if self.use_gamma:
            x = x * self.gamma

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute output shape.

        Args:
            input_shape: Shape tuple of the input.

        Returns:
            Output shape tuple.
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns:
            Configuration dictionary.
        """
        config = super().get_config()
        config.update({
            'kernel_size': self.kernel_size,
            'filters': self.filters,
            'normalization_type': self.normalization_type,
            'ffn_type': self.ffn_type,
            'activation': self.activation,
            'use_bias': self.use_bias,
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'dropout_rate': self.dropout_rate,
            'spatial_dropout_rate': self.spatial_dropout_rate,
            'use_gamma': self.use_gamma,
            'use_softorthonormal_regularizer': self.use_softorthonormal_regularizer,
            'normalization_kwargs': self.normalization_kwargs,
            'ffn_kwargs': self.ffn_kwargs,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ConvNextV1Block":
        """
        Create layer from configuration.

        Args:
            config: Configuration dictionary.

        Returns:
            ConvNextV1Block instance.
        """
        # Deserialize regularizer if present
        if config.get('kernel_regularizer'):
            config['kernel_regularizer'] = keras.regularizers.deserialize(
                config['kernel_regularizer']
            )
        return cls(**config)


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class ConvNeXtV1(keras.Model):
    """
    Configurable ConvNeXt V1 model with factory-based normalization and FFN.

    A modern ConvNet architecture that achieves competitive performance
    with Vision Transformers while maintaining the simplicity and efficiency
    of convolutional networks. This version supports configurable normalization
    and FFN types through factory patterns.

    Args:
        num_classes: Integer, number of output classes for classification.
        depths: List of integers, number of ConvNext blocks in each stage.
        dims: List of integers, number of channels in each stage.
        drop_path_rate: Float, stochastic depth rate.
        kernel_size: Integer or tuple, kernel size for ConvNext blocks.
        normalization_type: String, type of normalization layer.
        ffn_type: String, type of FFN layer.
        activation: String or callable, activation function for blocks.
        use_bias: Boolean, whether to use bias in convolutions.
        kernel_regularizer: Regularizer function applied to kernels.
        dropout_rate: Float, dropout rate applied within blocks.
        spatial_dropout_rate: Float, spatial dropout rate for blocks.
        strides: Integer, strides for downsampling.
        use_gamma: Boolean, whether to use learnable scaling in blocks.
        use_softorthonormal_regularizer: Boolean, regularization flag.
        include_top: Boolean, whether to include the classification head.
        input_shape: Tuple, input shape.
        normalization_kwargs: Optional dict, custom arguments for normalization.
        ffn_kwargs: Optional dict, custom arguments for FFN.
        **kwargs: Additional keyword arguments for the Model base class.

    Example:
        >>> model = ConvNeXtV1.from_variant(
        ...     "tiny",
        ...     num_classes=10,
        ...     normalization_type='rms_norm',
        ...     ffn_type='swiglu'
        ... )
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

    # Pretrained weights URLs
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
            normalization_type: NormalizationType = 'layer_norm',
            ffn_type: FFNType = 'mlp',
            activation: str = "gelu",
            use_bias: bool = True,
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            dropout_rate: float = 0.0,
            spatial_dropout_rate: float = 0.0,
            strides: int = 4,
            use_gamma: bool = True,
            use_softorthonormal_regularizer: bool = False,
            include_top: bool = True,
            input_shape: Tuple[int, ...] = (None, None, 3),
            normalization_kwargs: Optional[Dict[str, Any]] = None,
            ffn_kwargs: Optional[Dict[str, Any]] = None,
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
            raise ValueError(f"Strides {strides} must be positive.")

        # Store configuration
        self.num_classes = num_classes
        self.depths = depths
        self.dims = dims
        self.drop_path_rate = drop_path_rate
        self.kernel_size = kernel_size
        self.normalization_type = normalization_type
        self.ffn_type = ffn_type
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_regularizer = kernel_regularizer
        self.dropout_rate = dropout_rate
        self.spatial_dropout_rate = spatial_dropout_rate
        self.use_gamma = use_gamma
        self.use_softorthonormal_regularizer = use_softorthonormal_regularizer
        self.include_top = include_top
        self.strides = strides
        self.input_shape_config = input_shape
        self.normalization_kwargs = normalization_kwargs or {}
        self.ffn_kwargs = ffn_kwargs or {}

        # Validate input shape
        if input_shape is None:
            input_shape = (None, None, 3)
        if len(input_shape) != 3:
            raise ValueError(f"input_shape must be 3D, got {input_shape}")

        self.input_height, self.input_width, self.input_channels = input_shape
        if self.input_channels not in [1, 3]:
            logger.warning(
                f"Unusual number of channels: {self.input_channels}. "
                f"ConvNeXt typically uses 3 channels"
            )

        # Build layers
        self._build_stem()

        # Downsample layers and Stages
        self.downsample_layers_list = []
        self.stages_list = []
        for i in range(len(self.depths)):
            if i > 0:
                self._build_downsample_layer(i)
            self._build_stage(i)

        # Head
        if self.include_top:
            self._build_head()

        logger.info(
            f"Created Configurable ConvNeXt V1 model for input {input_shape} "
            f"with {sum(depths)} blocks, normalization_type={normalization_type}, "
            f"ffn_type={ffn_type}"
        )

    def _build_stem(self) -> None:
        """Build stem layers."""
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

        # Use factory for stem normalization
        self.stem_norm = create_normalization_layer(
            normalization_type=self.normalization_type,
            name="stem_norm",
            **self.normalization_kwargs
        )

    def _build_downsample_layer(self, stage_idx: int) -> None:
        """Build downsample layer."""
        downsample_kernel_size, downsample_stride = self.strides, self.strides

        # Use factory for downsample normalization
        downsample_norm = create_normalization_layer(
            normalization_type=self.normalization_type,
            name=f"downsample_norm_{stage_idx - 1}",
            **self.normalization_kwargs
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

    def _build_stage(self, stage_idx: int) -> None:
        """Build stage of ConvNeXt blocks."""
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

            # Create configurable ConvNeXt block
            block = ConvNextV1Block(
                kernel_size=self.kernel_size,
                filters=dim,
                normalization_type=self.normalization_type,
                ffn_type=self.ffn_type,
                activation=self.activation,
                kernel_regularizer=self.kernel_regularizer,
                use_bias=self.use_bias,
                dropout_rate=self.dropout_rate,
                spatial_dropout_rate=self.spatial_dropout_rate,
                use_gamma=self.use_gamma,
                use_softorthonormal_regularizer=self.use_softorthonormal_regularizer,
                normalization_kwargs=self.normalization_kwargs,
                ffn_kwargs=self.ffn_kwargs,
                name=f"stage_{stage_idx}_block_{block_idx}"
            )

            drop_path = keras.layers.Dropout(
                rate=drop_rate,
                noise_shape=(None, 1, 1, 1),
                name=f"stage_{stage_idx}_block_{block_idx}_drop_path"
            ) if drop_rate > 0 else None

            stage_blocks.append({"block": block, "drop_path": drop_path})

        self.stages_list.append(stage_blocks)

    def _build_head(self) -> None:
        """Build classification head."""
        self.gap = keras.layers.GlobalAveragePooling2D(name="global_avg_pool")

        # Use factory for head normalization
        self.head_norm = create_normalization_layer(
            normalization_type=self.normalization_type,
            name="head_norm",
            **self.normalization_kwargs
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

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass of the model.

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
        """
        Load pretrained weights into the model.

        Args:
            weights_path: String, path to the weights file (.keras format).
            skip_mismatch: Boolean, whether to skip layers with mismatched shapes.
            by_name: Boolean, whether to load weights by layer name.

        Raises:
            FileNotFoundError: If weights_path doesn't exist.
            ValueError: If weights cannot be loaded.
        """
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        try:
            # Build model if not already built
            if not self.built:
                dummy_input = keras.random.normal((1,) + tuple(self.input_shape_config))
                self(dummy_input, training=False)

            logger.info(f"Loading pretrained weights from {weights_path}")

            self.load_weights(
                weights_path,
                skip_mismatch=skip_mismatch,
                by_name=by_name
            )

            if skip_mismatch:
                logger.info(
                    "Weights loaded with skip_mismatch=True. "
                    "Layers with shape mismatches were skipped."
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
        """
        Download pretrained weights from URL.

        Args:
            variant: String, model variant name.
            dataset: String, dataset the weights were trained on.
            cache_dir: Optional string, directory to cache downloaded weights.

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
            normalization_type: NormalizationType = 'layer_norm',
            ffn_type: FFNType = 'mlp',
            normalization_kwargs: Optional[Dict[str, Any]] = None,
            ffn_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs
    ) -> "ConvNeXtV1":
        """
        Create a ConvNeXt model from a predefined variant.

        Args:
            variant: String, one of "tiny", "small", "base", "large", "xlarge".
            num_classes: Integer, number of output classes.
            input_shape: Tuple, input shape.
            pretrained: Boolean or string. If True, loads pretrained weights.
            weights_dataset: String, dataset for pretrained weights.
            weights_input_shape: Tuple, input shape used during weight pretraining.
            cache_dir: Optional string, directory to cache downloaded weights.
            normalization_type: String, type of normalization layer.
            ffn_type: String, type of FFN layer.
            normalization_kwargs: Optional dict, custom arguments for normalization.
            ffn_kwargs: Optional dict, custom arguments for FFN.
            **kwargs: Additional arguments passed to the constructor.

        Returns:
            ConvNeXtV1 model instance.

        Example:
            >>> model = ConvNeXtV1.from_variant(
            ...     "tiny",
            ...     pretrained=True,
            ...     normalization_type='rms_norm',
            ...     ffn_type='swiglu'
            ... )
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        config = cls.MODEL_VARIANTS[variant]

        logger.info(
            f"Creating ConvNeXt-{variant.upper()} model with "
            f"normalization_type={normalization_type}, ffn_type={ffn_type}"
        )

        # Handle pretrained weights
        load_weights_path = None
        skip_mismatch = False

        if pretrained:
            if isinstance(pretrained, str):
                load_weights_path = pretrained
                logger.info(f"Will load weights from local file: {load_weights_path}")
            else:
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
            normalization_type=normalization_type,
            ffn_type=ffn_type,
            normalization_kwargs=normalization_kwargs,
            ffn_kwargs=ffn_kwargs,
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
        """
        Get model configuration for serialization.

        Returns:
            Configuration dictionary.
        """
        config = {
            "num_classes": self.num_classes,
            "depths": self.depths,
            "dims": self.dims,
            "drop_path_rate": self.drop_path_rate,
            "kernel_size": self.kernel_size,
            "normalization_type": self.normalization_type,
            "ffn_type": self.ffn_type,
            "activation": self.activation,
            "use_bias": self.use_bias,
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "dropout_rate": self.dropout_rate,
            "spatial_dropout_rate": self.spatial_dropout_rate,
            "use_gamma": self.use_gamma,
            "use_softorthonormal_regularizer": self.use_softorthonormal_regularizer,
            "include_top": self.include_top,
            "input_shape": self.input_shape_config,
            "strides": self.strides,
            "normalization_kwargs": self.normalization_kwargs,
            "ffn_kwargs": self.ffn_kwargs,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ConvNeXtV1":
        """
        Create model from configuration.

        Args:
            config: Configuration dictionary.

        Returns:
            ConvNeXtV1 model instance.
        """
        # Deserialize regularizer if present
        if config.get("kernel_regularizer"):
            config["kernel_regularizer"] = keras.regularizers.deserialize(
                config["kernel_regularizer"]
            )

        return cls(**config)

    def summary(self, **kwargs) -> None:
        """Print model summary with additional information."""
        if not self.built:
            dummy_input = keras.KerasTensor(self.input_shape_config)
            self.build(dummy_input.shape)

        super().summary(**kwargs)

        # Print additional model information
        total_blocks = sum(self.depths)
        logger.info(f"Configurable ConvNeXt V1 configuration:")
        logger.info(f"  - Input shape: {self.input_shape_config}")
        logger.info(f"  - Stages: {len(self.depths)}")
        logger.info(f"  - Depths: {self.depths}")
        logger.info(f"  - Dimensions: {self.dims}")
        logger.info(f"  - Total blocks: {total_blocks}")
        logger.info(f"  - Normalization type: {self.normalization_type}")
        logger.info(f"  - FFN type: {self.ffn_type}")
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
        normalization_type: NormalizationType = 'layer_norm',
        ffn_type: FFNType = 'mlp',
        normalization_kwargs: Optional[Dict[str, Any]] = None,
        ffn_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
) -> ConvNeXtV1:
    """
    Convenience function to create configurable ConvNeXt V1 models.

    Args:
        variant: String, model variant ("tiny", "small", "base", "large", "xlarge").
        num_classes: Integer, number of output classes.
        input_shape: Tuple, input shape.
        pretrained: Boolean or string. If True, loads pretrained weights.
        weights_dataset: String, dataset for pretrained weights.
        weights_input_shape: Tuple, input shape used during weight pretraining.
        cache_dir: Optional string, directory to cache downloaded weights.
        normalization_type: String, type of normalization layer.
        ffn_type: String, type of FFN layer.
        normalization_kwargs: Optional dict, custom arguments for normalization.
        ffn_kwargs: Optional dict, custom arguments for FFN.
        **kwargs: Additional arguments passed to the model constructor.

    Returns:
        ConvNeXtV1 model instance.

    Example:
        >>> # Standard ConvNeXt-Tiny with pretrained ImageNet weights
        >>> model = create_convnext_v1("tiny", pretrained=True)
        >>>
        >>> # ConvNeXt-Base with RMSNorm and SwiGLU FFN
        >>> model = create_convnext_v1(
        ...     "base",
        ...     pretrained=True,
        ...     normalization_type='rms_norm',
        ...     ffn_type='swiglu'
        ... )
        >>>
        >>> # Fine-tune on CIFAR-10 with custom configuration
        >>> model = create_convnext_v1(
        ...     "small",
        ...     num_classes=10,
        ...     input_shape=(32, 32, 3),
        ...     pretrained=True,
        ...     ffn_type='geglu',
        ...     normalization_type='rms_norm',
        ...     normalization_kwargs={'epsilon': 1e-5},
        ...     ffn_kwargs={'dropout_rate': 0.1}
        ... )
    """
    model = ConvNeXtV1.from_variant(
        variant=variant,
        num_classes=num_classes,
        input_shape=input_shape,
        pretrained=pretrained,
        weights_dataset=weights_dataset,
        weights_input_shape=weights_input_shape,
        cache_dir=cache_dir,
        normalization_type=normalization_type,
        ffn_type=ffn_type,
        normalization_kwargs=normalization_kwargs,
        ffn_kwargs=ffn_kwargs,
        **kwargs
    )

    return model