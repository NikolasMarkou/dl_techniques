"""
ConvNeXt V1 Model Implementation - Refactored
==================================================

A complete implementation of the ConvNeXt V1 architecture with separated backbone and head.
This version allows using the same backbone with different task heads.

Based on: "A ConvNet for the 2020s" (Liu et al., 2022)
https://arxiv.org/abs/2201.03545

Key Features:
------------
- Separated ConvNeXtV1Backbone for feature extraction
- Compatible with vision heads framework for multi-task learning
- Support for multi-scale feature extraction (FPN-style)
- Configurable stochastic depth (drop path)
- Complete serialization support
- Production-ready implementation

Usage Examples:
-------------
```python
# Example 1: Classification with vision head
from dl_techniques.layers.vision_heads import create_vision_head, TaskType

backbone = ConvNeXtV1Backbone.from_variant("tiny", input_shape=(32, 32, 3))
head = create_vision_head(TaskType.CLASSIFICATION, num_classes=10, hidden_dim=768)

model = keras.Sequential([backbone, head])

# Example 2: Detection with custom head
backbone = ConvNeXtV1Backbone.from_variant("small", input_shape=(640, 640, 3))
detection_head = create_vision_head(TaskType.DETECTION, num_classes=80, hidden_dim=768)

images = keras.Input(shape=(640, 640, 3))
features = backbone(images)
outputs = detection_head(features)
model = keras.Model(inputs=images, outputs=outputs)

# Example 3: Multi-scale features for FPN
backbone = ConvNeXtV1Backbone.from_variant("base", return_multi_scale=True)
p3, p4, p5 = backbone(images)  # Multi-scale features

# Example 4: Backward compatible - complete model
model = create_convnext_v1_classifier("tiny", num_classes=10, input_shape=(32, 32, 3))
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
# ConvNeXt V1 Backbone
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class ConvNeXtV1Backbone(keras.Model):
    """ConvNeXt V1 Backbone for feature extraction.

    Extracts visual features from images using the ConvNeXt architecture.
    Can be combined with any task-specific head.

    Args:
        depths: List[int], number of ConvNext blocks in each stage.
            Default is [3, 3, 9, 3] for ConvNeXt-Tiny.
        dims: List[int], number of channels in each stage.
            Default is [96, 192, 384, 768] for ConvNeXt-Tiny.
        drop_path_rate: float, stochastic depth rate. Linearly increases
            from 0 to this value across all blocks.
        kernel_size: Union[int, Tuple[int, int]], kernel size for ConvNext blocks.
            Default is 7 following the original paper.
        activation: str, activation function for blocks.
            Default is "gelu" as used in the original paper.
        use_bias: bool, whether to use bias in convolutions.
        kernel_regularizer: Optional regularizer function applied to kernels.
        dropout_rate: float, dropout rate applied within blocks.
        spatial_dropout_rate: float, spatial dropout rate for blocks.
        strides: int, strides for downsampling.
        use_gamma: bool, whether to use learnable scaling in blocks.
        use_softorthonormal_regularizer: bool, whether to use soft
            orthonormal regularization in blocks.
        input_shape: Tuple[int, ...], input shape (height, width, channels).
        return_multi_scale: bool, whether to return multi-scale features.
            If True, returns [stage_1_out, stage_2_out, ..., stage_n_out].
            If False, returns only the final stage output.
        **kwargs: Additional keyword arguments for the Model base class.

    Returns:
        If return_multi_scale=False: Tensor of shape [batch, H', W', C']
        If return_multi_scale=True: List of tensors for each stage

    Example:
        >>> # Single-scale output
        >>> backbone = ConvNeXtV1Backbone.from_variant("tiny", input_shape=(224, 224, 3))
        >>> features = backbone(images)  # Shape: (batch, H/32, W/32, 768)
        >>>
        >>> # Multi-scale output for FPN
        >>> backbone = ConvNeXtV1Backbone.from_variant("tiny", return_multi_scale=True)
        >>> stage_outputs = backbone(images)  # List of 4 feature maps
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

    def __init__(
            self,
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
            input_shape: Tuple[int, ...] = (None, None, 3),
            return_multi_scale: bool = False,
            **kwargs
    ):
        super().__init__(**kwargs)

        # Validate configuration
        if len(depths) != len(dims):
            raise ValueError(
                f"Length of depths ({len(depths)}) must equal length of dims ({len(dims)})"
            )

        if strides <= 0:
            raise ValueError(f"Strides {strides} must be positive.")

        # Store configuration
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
        self.strides = strides
        self.input_shape_tuple = input_shape
        self.return_multi_scale = return_multi_scale

        # Validate and store input shape details
        if input_shape is None:
            input_shape = (None, None, 3)
        if len(input_shape) != 3:
            raise ValueError(f"input_shape must be 3D, got {input_shape}")

        self.input_height, self.input_width, self.input_channels = input_shape

        # Build layers
        self._build_stem()

        # Downsample layers and Stages
        self.downsample_layers_list = []
        self.stages_list = []
        for i in range(len(self.depths)):
            if i > 0:
                self._build_downsample_layer(i)
            self._build_stage(i)

        logger.info(
            f"Created ConvNeXt V1 Backbone for input {input_shape} "
            f"with {sum(depths)} blocks, return_multi_scale={return_multi_scale}"
        )

    def _build_stem(self) -> None:
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

    def _build_downsample_layer(self, stage_idx: int) -> None:
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

    def _build_stage(self, stage_idx: int) -> None:
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

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> Union[keras.KerasTensor, List[keras.KerasTensor]]:
        """Forward pass through the backbone.

        Args:
            inputs: Input images of shape [batch, height, width, channels]
            training: Boolean indicating training mode

        Returns:
            If return_multi_scale=False: Feature tensor [batch, H', W', C']
            If return_multi_scale=True: List of feature tensors for each stage
        """
        x = self.stem_conv(inputs)
        x = self.stem_norm(x)

        stage_outputs = []

        for stage_idx, stage_blocks in enumerate(self.stages_list):
            # Apply downsampling if not first stage
            if stage_idx > 0:
                norm_layer, conv_layer = self.downsample_layers_list[stage_idx - 1]
                x = norm_layer(x)
                x = conv_layer(x)

            # Apply blocks in this stage
            for block_info in stage_blocks:
                residual = x
                x = block_info["block"](x, training=training)
                if block_info["drop_path"]:
                    x = block_info["drop_path"](x, training=training)
                x = keras.layers.add([residual, x])

            # Store stage output if multi-scale
            if self.return_multi_scale:
                stage_outputs.append(x)

        # Return appropriate output format
        if self.return_multi_scale:
            return stage_outputs
        else:
            return x

    @property
    def output_channels(self) -> int:
        """Get the number of output channels from the backbone."""
        return self.dims[-1]

    @property
    def output_stride(self) -> int:
        """Get the total output stride of the backbone."""
        num_downsamples = len(self.depths)  # stem + (num_stages - 1) downsamples
        return self.strides ** num_downsamples

    @classmethod
    def from_variant(
            cls,
            variant: str,
            input_shape: Optional[Tuple[int, ...]] = None,
            **kwargs
    ) -> "ConvNeXtV1Backbone":
        """Create a ConvNeXt backbone from a predefined variant.

        Args:
            variant: str, one of "cifar10", "tiny", "small", "base", "large", "xlarge"
            input_shape: Tuple, input shape (height, width, channels).
                If None, uses (None, None, 3) for flexible input size.
            **kwargs: Additional arguments passed to the constructor

        Returns:
            ConvNeXtV1Backbone instance

        Raises:
            ValueError: If variant is not recognized

        Example:
            >>> # CIFAR-10 backbone
            >>> backbone = ConvNeXtV1Backbone.from_variant("tiny", input_shape=(32, 32, 3))
            >>> # ImageNet backbone with flexible input size
            >>> backbone = ConvNeXtV1Backbone.from_variant("base")
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        config = cls.MODEL_VARIANTS[variant]

        logger.info(f"Creating ConvNeXt-{variant.upper()} backbone")

        return cls(
            depths=config["depths"],
            dims=config["dims"],
            input_shape=input_shape,
            **kwargs
        )

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        config = {
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
            "input_shape": self.input_shape_tuple,
            "strides": self.strides,
            "return_multi_scale": self.return_multi_scale
        }
        base_config = super().get_config()
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ConvNeXtV1Backbone":
        """Create model from configuration."""
        if config.get("kernel_regularizer"):
            config["kernel_regularizer"] = keras.regularizers.deserialize(
                config["kernel_regularizer"]
            )
        return cls(**config)


# ---------------------------------------------------------------------
# ConvNeXt V1 Classifier (Backbone + Classification Head)
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class ConvNeXtV1Classifier(keras.Model):
    """ConvNeXt V1 model with classification head.

    Complete model combining ConvNeXt backbone with a classification head.
    This provides backward compatibility with the original monolithic implementation.

    Args:
        num_classes: int, number of output classes for classification.
        backbone_config: Dict, configuration for the backbone.
            Can be created using ConvNeXtV1Backbone.MODEL_VARIANTS[variant].
        dropout_rate: float, dropout rate before final classification.
        use_bias: bool, whether to use bias in classification head.
        kernel_regularizer: Optional regularizer for classification head.
        **kwargs: Additional keyword arguments for the Model base class.

    Example:
        >>> # Create classifier
        >>> model = ConvNeXtV1Classifier.from_variant(
        ...     "tiny",
        ...     num_classes=10,
        ...     input_shape=(32, 32, 3)
        ... )
        >>>
        >>> # Train
        >>> model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        >>> model.fit(x_train, y_train, epochs=10)
    """

    # Architecture constants
    LAYERNORM_EPSILON = 1e-6
    HEAD_INITIALIZER = "truncated_normal"

    def __init__(
            self,
            num_classes: int,
            backbone_config: Dict[str, Any],
            dropout_rate: float = 0.0,
            use_bias: bool = True,
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.backbone_config = backbone_config
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_regularizer = kernel_regularizer

        # Create backbone (ensure multi-scale is False for classification)
        backbone_config['return_multi_scale'] = False
        self.backbone = ConvNeXtV1Backbone(**backbone_config)

        # Build classification head
        self._build_head()

        logger.info(f"Created ConvNeXt V1 Classifier with {num_classes} classes")

    def _build_head(self) -> None:
        """Build classification head."""
        self.gap = keras.layers.GlobalAveragePooling2D(name="global_avg_pool")
        self.head_norm = keras.layers.LayerNormalization(
            epsilon=self.LAYERNORM_EPSILON,
            center=self.use_bias,
            scale=True,
            name="head_norm"
        )

        if self.dropout_rate > 0:
            self.dropout = keras.layers.Dropout(self.dropout_rate, name="head_dropout")

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
        """Forward pass through the model.

        Args:
            inputs: Input images of shape [batch, height, width, channels]
            training: Boolean indicating training mode

        Returns:
            Classification logits of shape [batch, num_classes]
        """
        # Extract features from backbone
        features = self.backbone(inputs, training=training)

        # Apply classification head
        x = self.gap(features)
        x = self.head_norm(x)

        if self.dropout_rate > 0:
            x = self.dropout(x, training=training)

        if self.classifier:
            x = self.classifier(x)

        return x

    @classmethod
    def from_variant(
            cls,
            variant: str,
            num_classes: int,
            input_shape: Optional[Tuple[int, ...]] = None,
            **kwargs
    ) -> "ConvNeXtV1Classifier":
        """Create a ConvNeXt classifier from a predefined variant.

        Args:
            variant: str, one of "cifar10", "tiny", "small", "base", "large", "xlarge"
            num_classes: int, number of output classes
            input_shape: Tuple, input shape (height, width, channels)
            **kwargs: Additional arguments (drop_path_rate, dropout_rate, etc.)

        Returns:
            ConvNeXtV1Classifier instance

        Example:
            >>> model = ConvNeXtV1Classifier.from_variant(
            ...     "tiny",
            ...     num_classes=10,
            ...     input_shape=(32, 32, 3),
            ...     drop_path_rate=0.1
            ... )
        """
        if variant not in ConvNeXtV1Backbone.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(ConvNeXtV1Backbone.MODEL_VARIANTS.keys())}"
            )

        # Get backbone configuration
        variant_config = ConvNeXtV1Backbone.MODEL_VARIANTS[variant]

        # Extract backbone-specific kwargs
        backbone_kwargs = {
            'depths': variant_config['depths'],
            'dims': variant_config['dims'],
            'input_shape': input_shape,
            'drop_path_rate': kwargs.pop('drop_path_rate', 0.0),
            'kernel_size': kwargs.pop('kernel_size', 7),
            'activation': kwargs.pop('activation', 'gelu'),
            'use_bias': kwargs.pop('use_bias', True),
            'kernel_regularizer': kwargs.pop('kernel_regularizer', None),
            'dropout_rate': kwargs.pop('block_dropout_rate', 0.0),
            'spatial_dropout_rate': kwargs.pop('spatial_dropout_rate', 0.0),
            'strides': kwargs.pop('strides', 4),
            'use_gamma': kwargs.pop('use_gamma', True),
            'use_softorthonormal_regularizer': kwargs.pop('use_softorthonormal_regularizer', True),
        }

        # Extract head-specific kwargs
        head_dropout_rate = kwargs.pop('dropout_rate', 0.0)
        head_kernel_regularizer = kwargs.pop('head_kernel_regularizer', None)

        logger.info(f"Creating ConvNeXt-{variant.upper()} classifier")

        return cls(
            num_classes=num_classes,
            backbone_config=backbone_kwargs,
            dropout_rate=head_dropout_rate,
            kernel_regularizer=head_kernel_regularizer,
            **kwargs
        )

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        config = {
            "num_classes": self.num_classes,
            "backbone_config": self.backbone_config,
            "dropout_rate": self.dropout_rate,
            "use_bias": self.use_bias,
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer)
        }
        base_config = super().get_config()
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ConvNeXtV1Classifier":
        """Create model from configuration."""
        if config.get("kernel_regularizer"):
            config["kernel_regularizer"] = keras.regularizers.deserialize(
                config["kernel_regularizer"]
            )
        if config.get("backbone_config", {}).get("kernel_regularizer"):
            config["backbone_config"]["kernel_regularizer"] = keras.regularizers.deserialize(
                config["backbone_config"]["kernel_regularizer"]
            )
        return cls(**config)


# ---------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------


def create_convnext_v1_backbone(
        variant: str = "tiny",
        input_shape: Optional[Tuple[int, ...]] = None,
        return_multi_scale: bool = False,
        **kwargs
) -> ConvNeXtV1Backbone:
    """Convenience function to create ConvNeXt V1 backbone.

    Args:
        variant: str, model variant ("cifar10", "tiny", "small", "base", "large", "xlarge")
        input_shape: Tuple, input shape (height, width, channels)
        return_multi_scale: bool, whether to return multi-scale features
        **kwargs: Additional arguments passed to the backbone constructor

    Returns:
        ConvNeXtV1Backbone instance

    Example:
        >>> # Single-scale backbone for classification
        >>> backbone = create_convnext_v1_backbone("tiny", input_shape=(224, 224, 3))
        >>>
        >>> # Multi-scale backbone for detection
        >>> backbone = create_convnext_v1_backbone(
        ...     "small",
        ...     input_shape=(640, 640, 3),
        ...     return_multi_scale=True
        ... )
    """
    return ConvNeXtV1Backbone.from_variant(
        variant,
        input_shape=input_shape,
        return_multi_scale=return_multi_scale,
        **kwargs
    )


def create_convnext_v1_classifier(
        variant: str = "tiny",
        num_classes: int = 1000,
        input_shape: Optional[Tuple[int, ...]] = None,
        **kwargs
) -> ConvNeXtV1Classifier:
    """Convenience function to create ConvNeXt V1 classifier.

    Provides backward compatibility with the original create_convnext_v1 function.

    Args:
        variant: str, model variant ("cifar10", "tiny", "small", "base", "large", "xlarge")
        num_classes: int, number of output classes
        input_shape: Tuple, input shape (height, width, channels)
        **kwargs: Additional arguments passed to the model constructor

    Returns:
        ConvNeXtV1Classifier instance

    Example:
        >>> # CIFAR-10 model
        >>> model = create_convnext_v1_classifier(
        ...     "tiny",
        ...     num_classes=10,
        ...     input_shape=(32, 32, 3),
        ...     drop_path_rate=0.1
        ... )
        >>>
        >>> # ImageNet model
        >>> model = create_convnext_v1_classifier("base", num_classes=1000)
    """
    return ConvNeXtV1Classifier.from_variant(
        variant,
        num_classes=num_classes,
        input_shape=input_shape,
        **kwargs
    )


# Backward compatibility alias
def create_convnext_v1(
        variant: str = "tiny",
        num_classes: int = 1000,
        input_shape: Optional[Tuple[int, ...]] = None,
        pretrained: bool = False,
        **kwargs
) -> ConvNeXtV1Classifier:
    """Backward compatible function to create ConvNeXt V1 models.

    This function maintains API compatibility with the original implementation.

    Args:
        variant: str, model variant
        num_classes: int, number of output classes
        input_shape: Tuple, input shape
        pretrained: bool, whether to load pretrained weights (not implemented)
        **kwargs: Additional arguments

    Returns:
        ConvNeXtV1Classifier instance
    """
    if pretrained:
        logger.warning("Pretrained weights are not yet implemented")

    return create_convnext_v1_classifier(
        variant,
        num_classes=num_classes,
        input_shape=input_shape,
        **kwargs
    )


# For backward compatibility with existing imports
ConvNeXtV1 = ConvNeXtV1Classifier


# ---------------------------------------------------------------------
