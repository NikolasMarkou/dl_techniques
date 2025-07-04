"""
Vision Transformer (ViT) Model Implementation

This module provides a complete Vision Transformer model implementation
that can be used for various computer vision tasks including image classification,
feature extraction, and transfer learning.

The model supports different scales and configurations similar to the original
"An Image is Worth 16x16 Words" paper and its variants.

File: src/dl_techniques/models/vit.py
"""

import keras
from keras import ops
from typing import Optional, Tuple, Dict, Any, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.norms.rms_norm import RMSNorm
from dl_techniques.layers.patch_embedding import PatchEmbedding2D
from dl_techniques.layers.positional_embedding import PositionalEmbedding
from dl_techniques.layers.vision_transformer import VisionTransformerLayer

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ViT(keras.Model):
    """
    Vision Transformer (ViT) Model.

    This model implements the complete Vision Transformer architecture with support
    for different scales and configurations. It can be used for classification,
    feature extraction, and other vision tasks.

    Args:
        input_shape: Input image shape (height, width, channels).
        num_classes: Number of output classes for classification.
        scale: Model scale configuration ('tiny', 'small', 'base', 'large', 'huge').
        patch_size: Size of patches to extract from input images.
        include_top: Whether to include the classification head.
        pooling: Pooling mode for feature extraction ('cls', 'mean', 'max', None).
        dropout_rate: Dropout rate for regularization.
        attention_dropout_rate: Dropout rate for attention weights.
        kernel_initializer: Weight initializer for all layers.
        kernel_regularizer: Weight regularizer for all layers.
        norm_type: Type of normalization ('layer' or 'rms').
        name: Model name.

    Returns:
        Model outputs depend on include_top and pooling settings.
    """

    # Scale configurations: [embed_dim, num_heads, num_layers, mlp_ratio]
    SCALE_CONFIGS = {
        "tiny": [192, 3, 12, 4.0],  # ViT-Tiny
        "small": [384, 6, 12, 4.0],  # ViT-Small
        "base": [768, 12, 12, 4.0],  # ViT-Base
        "large": [1024, 16, 24, 4.0],  # ViT-Large
        "huge": [1280, 16, 32, 4.0],  # ViT-Huge
    }

    def __init__(
            self,
            input_shape: Tuple[int, int, int] = (224, 224, 3),
            num_classes: int = 1000,
            scale: str = "base",
            patch_size: Union[int, Tuple[int, int]] = 16,
            include_top: bool = True,
            pooling: Optional[str] = None,
            dropout_rate: float = 0.0,
            attention_dropout_rate: float = 0.0,
            kernel_initializer: str = "he_normal",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            norm_type: str = "layer",
            name: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        """
        Initialize Vision Transformer model.

        Args:
            input_shape: Input image shape (height, width, channels).
            num_classes: Number of output classes for classification.
            scale: Model scale ('tiny', 'small', 'base', 'large', 'huge').
            patch_size: Size of patches to extract from input images.
            include_top: Whether to include the classification head.
            pooling: Pooling mode for feature extraction ('cls', 'mean', 'max', None).
            dropout_rate: Dropout rate for regularization.
            attention_dropout_rate: Dropout rate for attention weights.
            kernel_initializer: Weight initializer for all layers.
            kernel_regularizer: Weight regularizer for all layers.
            norm_type: Type of normalization ('layer' or 'rms').
            name: Model name.
            **kwargs: Additional keyword arguments.
        """
        if name is None:
            name = f"vision_transformer_{scale}"
        super().__init__(name=name, **kwargs)

        self.input_shape_config = input_shape
        self.num_classes = num_classes
        self.scale = scale
        self.patch_size = patch_size
        self.include_top = include_top
        self.pooling = pooling
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.norm_type = norm_type

        if scale not in self.SCALE_CONFIGS:
            raise ValueError(
                f"Unsupported scale: {scale}. Choose from {list(self.SCALE_CONFIGS.keys())}"
            )

        if pooling not in [None, "cls", "mean", "max"]:
            raise ValueError(
                f"Unsupported pooling: {pooling}. Choose from [None, 'cls', 'mean', 'max']"
            )

        if not include_top and pooling is None:
            logger.warning(
                "include_top=False and pooling=None will return the raw transformer output"
            )

        # Get model configuration
        self.embed_dim, self.num_heads, self.num_layers, self.mlp_ratio = self.SCALE_CONFIGS[scale]

        # Calculate number of patches
        if isinstance(patch_size, int):
            patch_h = patch_w = patch_size
        else:
            patch_h, patch_w = patch_size

        img_h, img_w = input_shape[:2]
        self.num_patches = (img_h // patch_h) * (img_w // patch_w)
        self.seq_len = self.num_patches + 1  # +1 for CLS token

        # Store build state for serialization
        self._build_input_shape = None
        self._layers_built = False

        logger.info(f"Created VisionTransformer-{scale} with {self.embed_dim}d, {self.num_heads}h, {self.num_layers}L")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the model layers."""
        if self._layers_built:
            return

        self._build_input_shape = input_shape
        self._build_layers()
        self._layers_built = True
        super().build(input_shape)

    def _build_layers(self) -> None:
        """Initialize all model layers."""
        # Patch embedding layer
        self.patch_embed = PatchEmbedding2D(
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="patch_embed"
        )

        # CLS token
        self.cls_token = self.add_weight(
            name="cls_token",
            shape=(1, 1, self.embed_dim),
            initializer="zeros",
            trainable=True
        )

        # Positional embedding
        self.pos_embed = PositionalEmbedding(
            sequence_length=self.seq_len,
            embed_dim=self.embed_dim,
            name="pos_embed"
        )

        # Transformer layers
        self.transformer_layers = []
        for i in range(self.num_layers):
            layer = VisionTransformerLayer(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                norm_type=self.norm_type,
                name=f"transformer_layer_{i}"
            )
            self.transformer_layers.append(layer)

        # Final normalization
        if self.norm_type == "rms":
            self.norm = RMSNorm(name="norm")
        else:
            self.norm = keras.layers.LayerNormalization(epsilon=1e-6, name="norm")

        # Classification head (if include_top)
        if self.include_top:
            self.head_dropout = keras.layers.Dropout(self.dropout_rate, name="head_dropout")
            self.head = keras.layers.Dense(
                self.num_classes,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name="head"
            )

        # Global pooling layers (if needed)
        if self.pooling == "mean":
            self.global_pool = keras.layers.GlobalAveragePooling1D(name="global_avg_pool")
        elif self.pooling == "max":
            self.global_pool = keras.layers.GlobalMaxPooling1D(name="global_max_pool")

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass through the model.

        Args:
            inputs: Input tensor (batch_size, height, width, channels).
            training: Whether in training mode.

        Returns:
            Model output tensor. Shape depends on include_top and pooling settings.
        """
        # Convert image to patches
        x = self.patch_embed(inputs)  # (batch_size, num_patches, embed_dim)

        # Add CLS token
        batch_size = ops.shape(x)[0]
        cls_tokens = ops.broadcast_to(self.cls_token, (batch_size, 1, self.embed_dim))
        x = ops.concatenate([cls_tokens, x], axis=1)  # (batch_size, seq_len, embed_dim)

        # Add positional embeddings
        x = self.pos_embed(x)

        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x, training=training)

        # Apply final normalization
        x = self.norm(x, training=training)

        # Handle different output modes
        if self.include_top:
            # Extract CLS token for classification
            cls_token = x[:, 0, :]  # (batch_size, embed_dim)
            x = self.head_dropout(cls_token, training=training)
            x = self.head(x)  # (batch_size, num_classes)
            return x
        else:
            # Feature extraction mode
            if self.pooling == "cls":
                # Return CLS token
                return x[:, 0, :]  # (batch_size, embed_dim)
            elif self.pooling == "mean":
                # Global average pooling over sequence
                return self.global_pool(x)  # (batch_size, embed_dim)
            elif self.pooling == "max":
                # Global max pooling over sequence
                return self.global_pool(x)  # (batch_size, embed_dim)
            else:
                # Return full transformer output
                return x  # (batch_size, seq_len, embed_dim)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute output shape.

        Args:
            input_shape: Input tensor shape.

        Returns:
            Output shape.
        """
        batch_size = input_shape[0]

        if self.include_top:
            return (batch_size, self.num_classes)
        else:
            if self.pooling in ["cls", "mean", "max"]:
                return (batch_size, self.embed_dim)
            else:
                return (batch_size, self.seq_len, self.embed_dim)

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        config = super().get_config()
        config.update({
            "input_shape": self.input_shape_config,
            "num_classes": self.num_classes,
            "scale": self.scale,
            "patch_size": self.patch_size,
            "include_top": self.include_top,
            "pooling": self.pooling,
            "dropout_rate": self.dropout_rate,
            "attention_dropout_rate": self.attention_dropout_rate,
            "kernel_initializer": self.kernel_initializer,
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "norm_type": self.norm_type,
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get build configuration for serialization."""
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build model from configuration."""
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ViT":
        """Create model from configuration."""
        return cls(**config)

    def get_feature_extractor(self) -> "ViT":
        """
        Get a feature extractor version of this model.

        Returns:
            New ViT configured for feature extraction.
        """
        return ViT(
            input_shape=self.input_shape_config,
            num_classes=self.num_classes,
            scale=self.scale,
            patch_size=self.patch_size,
            include_top=False,
            pooling="cls",
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            norm_type=self.norm_type,
            name=f"{self.name}_feature_extractor"
        )

    def summary_detailed(self) -> None:
        """Print detailed model summary."""
        logger.info(f"Vision Transformer Model Summary")
        logger.info(f"Scale: {self.scale}")
        logger.info(f"Input Shape: {self.input_shape_config}")
        logger.info(f"Patch Size: {self.patch_size}")
        logger.info(f"Number of Patches: {self.num_patches}")
        logger.info(f"Sequence Length: {self.seq_len}")
        logger.info(f"Embedding Dimension: {self.embed_dim}")
        logger.info(f"Number of Heads: {self.num_heads}")
        logger.info(f"Number of Layers: {self.num_layers}")
        logger.info(f"MLP Ratio: {self.mlp_ratio}")
        logger.info(f"Dropout Rate: {self.dropout_rate}")
        logger.info(f"Attention Dropout Rate: {self.attention_dropout_rate}")
        logger.info(f"Normalization Type: {self.norm_type}")
        logger.info(f"Include Top: {self.include_top}")
        logger.info(f"Pooling: {self.pooling}")
        logger.info(f"Number of Classes: {self.num_classes}")
        logger.info(f"Total Parameters: {self.count_params():,}")


def create_vision_transformer(
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        num_classes: int = 1000,
        scale: str = "base",
        patch_size: Union[int, Tuple[int, int]] = 16,
        include_top: bool = True,
        pooling: Optional[str] = None,
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        kernel_initializer: str = "he_normal",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        norm_type: str = "layer",
        **kwargs: Any
) -> ViT:
    """
    Create a Vision Transformer model with specified configuration.

    Args:
        input_shape: Input image shape.
        num_classes: Number of output classes for classification.
        scale: Model scale ('tiny', 'small', 'base', 'large', 'huge').
        patch_size: Size of patches to extract from input images.
        include_top: Whether to include the classification head.
        pooling: Pooling mode for feature extraction ('cls', 'mean', 'max', None).
        dropout_rate: Dropout rate for regularization.
        attention_dropout_rate: Dropout rate for attention weights.
        kernel_initializer: Weight initializer for all layers.
        kernel_regularizer: Weight regularizer for all layers.
        norm_type: Type of normalization ('layer' or 'rms').
        **kwargs: Additional arguments for ViT.

    Returns:
        ViT instance.

    Examples:
        >>> # Create a ViT-Base model for ImageNet classification
        >>> model = create_vision_transformer(
        ...     input_shape=(224, 224, 3),
        ...     num_classes=1000,
        ...     scale="base"
        ... )

        >>> # Create a ViT-Small feature extractor
        >>> feature_extractor = create_vision_transformer(
        ...     input_shape=(224, 224, 3),
        ...     scale="small",
        ...     include_top=False,
        ...     pooling="cls"
        ... )

        >>> # Create a ViT-Tiny for CIFAR-10
        >>> cifar_model = create_vision_transformer(
        ...     input_shape=(32, 32, 3),
        ...     num_classes=10,
        ...     scale="tiny",
        ...     patch_size=4,
        ...     dropout_rate=0.1
        ... )
    """
    model = ViT(
        input_shape=input_shape,
        num_classes=num_classes,
        scale=scale,
        patch_size=patch_size,
        include_top=include_top,
        pooling=pooling,
        dropout_rate=dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        norm_type=norm_type,
        **kwargs
    )

    logger.info(f"VisionTransformer-{scale} created successfully")
    return model


def create_vit_tiny(**kwargs: Any) -> ViT:
    """Create ViT-Tiny model."""
    return create_vision_transformer(scale="tiny", **kwargs)


def create_vit_small(**kwargs: Any) -> ViT:
    """Create ViT-Small model."""
    return create_vision_transformer(scale="small", **kwargs)


def create_vit_base(**kwargs: Any) -> ViT:
    """Create ViT-Base model."""
    return create_vision_transformer(scale="base", **kwargs)


def create_vit_large(**kwargs: Any) -> ViT:
    """Create ViT-Large model."""
    return create_vision_transformer(scale="large", **kwargs)


def create_vit_huge(**kwargs: Any) -> ViT:
    """Create ViT-Huge model."""
    return create_vision_transformer(scale="huge", **kwargs)

# ---------------------------------------------------------------------