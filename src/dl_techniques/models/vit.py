"""
Vision Transformer (ViT) Model Implementation

This module provides a complete Vision Transformer model implementation
that can be used for various computer vision tasks including image classification,
feature extraction, and transfer learning.

The model supports different scales and configurations similar to the original
"An Image is Worth 16x16 Words" paper and its variants.
"""

import keras
from keras import ops
from typing import Optional, Tuple, Dict, Any, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..utils.logger import logger
from ..layers.norms.rms_norm import RMSNorm
from ..layers.transformer import TransformerLayer
from ..layers.embedding.patch_embedding import PatchEmbedding2D
from ..layers.embedding.positional_embedding import PositionalEmbedding

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
        pos_dropout_rate: Dropout rate for positional embeddings.
        kernel_initializer: Weight initializer for all layers.
        kernel_regularizer: Weight regularizer for all layers.
        bias_initializer: Bias initializer for all layers.
        bias_regularizer: Bias regularizer for all layers.
        norm_type: Type of normalization ('layer' or 'rms').
        normalization_position: Position of normalization ('pre' or 'post').
        ffn_type: Type of feed-forward network ('mlp', 'swiglu', etc.).
        activation: Activation function for feed-forward networks.
        name: Model name.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, channels)`

    Output shape:
        - If `include_top=True`: `(batch_size, num_classes)`
        - If `include_top=False` and `pooling='cls'`: `(batch_size, embed_dim)`
        - If `include_top=False` and `pooling='mean'`: `(batch_size, embed_dim)`
        - If `include_top=False` and `pooling='max'`: `(batch_size, embed_dim)`
        - If `include_top=False` and `pooling=None`: `(batch_size, seq_len, embed_dim)`

    Example:
        ```python
        # Create ViT-Base for ImageNet classification
        model = ViT(
            input_shape=(224, 224, 3),
            num_classes=1000,
            scale='base'
        )

        # Create feature extractor
        feature_model = ViT(
            input_shape=(224, 224, 3),
            scale='base',
            include_top=False,
            pooling='cls'
        )

        # Custom configuration with different transformer settings
        custom_model = ViT(
            input_shape=(384, 384, 3),
            num_classes=10,
            scale='small',
            normalization_position='pre',
            ffn_type='swiglu',
            activation='gelu'
        )
        ```
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
        pos_dropout_rate: float = 0.0,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
        bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
        norm_type: str = "layer",
        normalization_position: str = "post",
        ffn_type: str = "mlp",
        activation: Union[str, callable] = "gelu",
        name: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """Initialize Vision Transformer model."""
        if name is None:
            name = f"vision_transformer_{scale}"
        super().__init__(name=name, **kwargs)

        # Validate input_shape
        if len(input_shape) != 3:
            raise ValueError(f"input_shape must be a 3-tuple (height, width, channels), got {input_shape}")

        img_h, img_w, img_c = input_shape
        if img_h <= 0 or img_w <= 0 or img_c <= 0:
            raise ValueError(f"All input_shape dimensions must be positive, got {input_shape}")

        # Validate and normalize patch_size
        if isinstance(patch_size, int):
            if patch_size <= 0:
                raise ValueError(f"patch_size must be positive, got {patch_size}")
            patch_h = patch_w = patch_size
        else:
            if len(patch_size) != 2:
                raise ValueError(f"patch_size must be int or tuple of 2 ints, got {patch_size}")
            patch_h, patch_w = patch_size
            if patch_h <= 0 or patch_w <= 0:
                raise ValueError(f"patch_size dimensions must be positive, got {patch_size}")

        # Validate that image dimensions are divisible by patch dimensions
        if img_h % patch_h != 0:
            raise ValueError(f"Image height ({img_h}) must be divisible by patch height ({patch_h})")
        if img_w % patch_w != 0:
            raise ValueError(f"Image width ({img_w}) must be divisible by patch width ({patch_w})")

        # Store configuration
        self.input_shape_config = tuple(input_shape)
        self.num_classes = int(num_classes)
        self.scale = str(scale)
        self.patch_size = (patch_h, patch_w)
        self.include_top = bool(include_top)
        self.pooling = pooling
        self.dropout_rate = float(dropout_rate)
        self.attention_dropout_rate = float(attention_dropout_rate)
        self.pos_dropout_rate = float(pos_dropout_rate)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = kernel_regularizer
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.bias_regularizer = bias_regularizer
        self.norm_type = str(norm_type)
        self.normalization_position = str(normalization_position)
        self.ffn_type = str(ffn_type)
        self.activation = activation

        # Validate inputs
        if scale not in self.SCALE_CONFIGS:
            raise ValueError(
                f"Unsupported scale: {scale}. Choose from {list(self.SCALE_CONFIGS.keys())}"
            )

        if pooling not in [None, "cls", "mean", "max"]:
            raise ValueError(
                f"Unsupported pooling: {pooling}. Choose from [None, 'cls', 'mean', 'max']"
            )

        if norm_type not in ["layer", "rms"]:
            raise ValueError(f"Unsupported norm_type: {norm_type}. Choose from ['layer', 'rms']")

        if normalization_position not in ["pre", "post"]:
            raise ValueError(f"Unsupported normalization_position: {normalization_position}. Choose from ['pre', 'post']")

        # Get model configuration
        self.embed_dim, self.num_heads, self.num_layers, self.mlp_ratio = self.SCALE_CONFIGS[scale]

        # Calculate intermediate size for transformer layers
        self.intermediate_size = int(self.embed_dim * self.mlp_ratio)

        # Calculate number of patches
        self.num_patches = (img_h // patch_h) * (img_w // patch_w)
        self.max_seq_len = self.num_patches + 1  # +1 for CLS token

        # Validate we have at least one patch
        if self.num_patches <= 0:
            raise ValueError(f"Number of patches must be positive, got {self.num_patches}")

        # Initialize layers as None - they will be created in build()
        self.patch_embed = None
        self.cls_token = None
        self.pos_embed = None
        self.transformer_layers = []
        self.norm = None
        self.head_dropout = None
        self.head = None
        self.global_pool = None

        # Store build state for serialization
        self._build_input_shape = None

        logger.info(f"Created VisionTransformer-{scale} with {self.embed_dim}d, {self.num_heads}h, {self.num_layers}L")
        logger.info(f"Image shape: {self.input_shape_config}, Patch size: {self.patch_size}, Num patches: {self.num_patches}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the model layers."""
        if self.built:
            return

        self._build_input_shape = input_shape

        # Map norm_type to TransformerLayer's normalization_type
        transformer_norm_type = "layer_norm" if self.norm_type == "layer" else "rms_norm"

        # Patch embedding layer
        self.patch_embed = PatchEmbedding2D(
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="patch_embed"
        )

        # Build patch embedding to validate dimensions
        dummy_input_shape = (None,) + self.input_shape_config
        self.patch_embed.build(dummy_input_shape)

        # CLS token
        self.cls_token = self.add_weight(
            name="cls_token",
            shape=(1, 1, self.embed_dim),
            initializer="zeros",
            trainable=True
        )

        # Positional embedding
        self.pos_embed = PositionalEmbedding(
            max_seq_len=self.max_seq_len,
            dim=self.embed_dim,
            dropout=self.pos_dropout_rate,
            name="pos_embed"
        )
        # Build positional embedding
        pos_input_shape = (None, self.max_seq_len, self.embed_dim)
        self.pos_embed.build(pos_input_shape)

        # Transformer layers using the generic TransformerLayer
        self.transformer_layers = []
        for i in range(self.num_layers):
            layer = TransformerLayer(
                hidden_size=self.embed_dim,
                num_heads=self.num_heads,
                intermediate_size=self.intermediate_size,
                attention_type="multi_head_attention",
                normalization_type=transformer_norm_type,
                normalization_position=self.normalization_position,
                ffn_type=self.ffn_type,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                activation=self.activation,
                use_bias=True,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name=f"transformer_layer_{i}"
            )
            # Build transformer layer
            transformer_input_shape = (None, self.max_seq_len, self.embed_dim)
            layer.build(transformer_input_shape)
            self.transformer_layers.append(layer)

        # Final normalization
        if self.norm_type == "rms":
            self.norm = RMSNorm(name="norm")
        else:
            self.norm = keras.layers.LayerNormalization(epsilon=1e-6, name="norm")

        # Build final norm layer
        norm_input_shape = (None, self.max_seq_len, self.embed_dim)
        self.norm.build(norm_input_shape)

        # Classification head (if include_top)
        if self.include_top:
            if self.dropout_rate > 0.0:
                self.head_dropout = keras.layers.Dropout(self.dropout_rate, name="head_dropout")

            self.head = keras.layers.Dense(
                self.num_classes,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name="head"
            )
            # Build head layer
            head_input_shape = (None, self.embed_dim)
            self.head.build(head_input_shape)

        # Global pooling layers (if needed)
        if self.pooling == "mean":
            self.global_pool = keras.layers.GlobalAveragePooling1D(name="global_avg_pool")
        elif self.pooling == "max":
            self.global_pool = keras.layers.GlobalMaxPooling1D(name="global_max_pool")

        if self.global_pool is not None:
            pool_input_shape = (None, self.max_seq_len, self.embed_dim)
            self.global_pool.build(pool_input_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through the model.

        Args:
            inputs: Input tensor (batch_size, height, width, channels).
            training: Whether in training mode.

        Returns:
            Model output tensor. Shape depends on include_top and pooling settings.
        """
        # Convert image to patches
        x = self.patch_embed(inputs, training=training)  # (batch_size, num_patches, embed_dim)

        # Add CLS token
        batch_size = ops.shape(x)[0]
        cls_tokens = ops.broadcast_to(self.cls_token, (batch_size, 1, self.embed_dim))
        x = ops.concatenate([cls_tokens, x], axis=1)  # (batch_size, seq_len, embed_dim)

        # Add positional embeddings (includes dropout)
        x = self.pos_embed(x, training=training)

        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x, training=training)

        # Apply final normalization
        x = self.norm(x, training=training)

        # Handle different output modes
        if self.include_top:
            # Extract CLS token for classification
            cls_token = x[:, 0, :]  # (batch_size, embed_dim)
            if self.head_dropout is not None:
                cls_token = self.head_dropout(cls_token, training=training)
            x = self.head(cls_token)  # (batch_size, num_classes)
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
        """Compute output shape.

        Args:
            input_shape: Input tensor shape.

        Returns:
            Output shape.
        """
        if len(input_shape) < 4:
            raise ValueError(f"Expected 4D input shape (batch, height, width, channels), got {input_shape}")

        batch_size = input_shape[0]

        if self.include_top:
            return (batch_size, self.num_classes)
        else:
            if self.pooling in ["cls", "mean", "max"]:
                return (batch_size, self.embed_dim)
            else:
                return (batch_size, self.max_seq_len, self.embed_dim)

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
            "pos_dropout_rate": self.pos_dropout_rate,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
            "norm_type": self.norm_type,
            "normalization_position": self.normalization_position,
            "ffn_type": self.ffn_type,
            "activation": self.activation,
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get build configuration for serialization."""
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build model from configuration."""
        input_shape = config.get("input_shape")
        if input_shape is not None:
            if not isinstance(input_shape, tuple):
                input_shape = tuple(input_shape)
            self.build(input_shape)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ViT":
        """Create model from configuration."""
        return cls(**config)

    def get_feature_extractor(self) -> "ViT":
        """Get a feature extractor version of this model.

        Returns:
            New ViT configured for feature extraction.
        """
        if not hasattr(self, 'input_shape_config') or not self.input_shape_config:
            raise ValueError("Model must be properly initialized before creating feature extractor")

        return ViT(
            input_shape=self.input_shape_config,
            num_classes=self.num_classes,
            scale=self.scale,
            patch_size=self.patch_size,
            include_top=False,
            pooling="cls",
            dropout_rate=self.dropout_rate,
            attention_dropout_rate=self.attention_dropout_rate,
            pos_dropout_rate=self.pos_dropout_rate,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bias_regularizer,
            norm_type=self.norm_type,
            normalization_position=self.normalization_position,
            ffn_type=self.ffn_type,
            activation=self.activation,
            name=f"{self.name}_feature_extractor"
        )

    def summary_detailed(self) -> None:
        """Print detailed model summary."""
        logger.info(f"Vision Transformer Model Summary")
        logger.info(f"Scale: {self.scale}")
        logger.info(f"Input Shape: {self.input_shape_config}")
        logger.info(f"Patch Size: {self.patch_size}")
        logger.info(f"Number of Patches: {self.num_patches}")
        logger.info(f"Sequence Length: {self.max_seq_len}")
        logger.info(f"Embedding Dimension: {self.embed_dim}")
        logger.info(f"Number of Heads: {self.num_heads}")
        logger.info(f"Number of Layers: {self.num_layers}")
        logger.info(f"MLP Ratio: {self.mlp_ratio}")
        logger.info(f"Intermediate Size: {self.intermediate_size}")
        logger.info(f"Dropout Rate: {self.dropout_rate}")
        logger.info(f"Attention Dropout Rate: {self.attention_dropout_rate}")
        logger.info(f"Positional Dropout Rate: {self.pos_dropout_rate}")
        logger.info(f"Normalization Type: {self.norm_type}")
        logger.info(f"Normalization Position: {self.normalization_position}")
        logger.info(f"FFN Type: {self.ffn_type}")
        logger.info(f"Activation: {self.activation}")
        logger.info(f"Include Top: {self.include_top}")
        logger.info(f"Pooling: {self.pooling}")
        logger.info(f"Number of Classes: {self.num_classes}")
        if self.built:
            logger.info(f"Total Parameters: {self.count_params():,}")

        # Additional safety information
        patch_h, patch_w = self.patch_size
        img_h, img_w = self.input_shape_config[:2]
        logger.info(f"Patches per dimension: {img_h // patch_h} x {img_w // patch_w}")
        logger.info(f"Patch coverage: {(img_h // patch_h) * patch_h}x{(img_w // patch_w) * patch_w} of {img_h}x{img_w}")


def create_vision_transformer(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 1000,
    scale: str = "base",
    patch_size: Union[int, Tuple[int, int]] = 16,
    include_top: bool = True,
    pooling: Optional[str] = None,
    dropout_rate: float = 0.0,
    attention_dropout_rate: float = 0.0,
    pos_dropout_rate: float = 0.0,
    kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
    bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
    norm_type: str = "layer",
    normalization_position: str = "post",
    ffn_type: str = "mlp",
    activation: Union[str, callable] = "gelu",
    **kwargs: Any
) -> ViT:
    """Create a Vision Transformer model with specified configuration.

    Args:
        input_shape: Input image shape.
        num_classes: Number of output classes for classification.
        scale: Model scale ('tiny', 'small', 'base', 'large', 'huge').
        patch_size: Size of patches to extract from input images.
        include_top: Whether to include the classification head.
        pooling: Pooling mode for feature extraction ('cls', 'mean', 'max', None).
        dropout_rate: Dropout rate for regularization.
        attention_dropout_rate: Dropout rate for attention weights.
        pos_dropout_rate: Dropout rate for positional embeddings.
        kernel_initializer: Weight initializer for all layers.
        kernel_regularizer: Weight regularizer for all layers.
        bias_initializer: Bias initializer for all layers.
        bias_regularizer: Bias regularizer for all layers.
        norm_type: Type of normalization ('layer' or 'rms').
        normalization_position: Position of normalization ('pre' or 'post').
        ffn_type: Type of feed-forward network ('mlp', 'swiglu', etc.).
        activation: Activation function for feed-forward networks.
        **kwargs: Additional arguments for ViT.

    Returns:
        ViT instance.
    """
    # Validate basic parameters
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")

    if not isinstance(input_shape, (tuple, list)) or len(input_shape) != 3:
        raise ValueError(f"input_shape must be a 3-element tuple/list, got {input_shape}")

    if any(dim <= 0 for dim in input_shape):
        raise ValueError(f"All input_shape dimensions must be positive, got {input_shape}")

    # Validate patch_size and ensure compatibility with input_shape
    if isinstance(patch_size, int):
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        patch_h = patch_w = patch_size
    else:
        if not isinstance(patch_size, (tuple, list)) or len(patch_size) != 2:
            raise ValueError(f"patch_size must be int or 2-element tuple/list, got {patch_size}")
        patch_h, patch_w = patch_size
        if patch_h <= 0 or patch_w <= 0:
            raise ValueError(f"patch_size dimensions must be positive, got {patch_size}")

    img_h, img_w = input_shape[:2]
    if img_h % patch_h != 0:
        raise ValueError(f"Image height ({img_h}) must be divisible by patch height ({patch_h})")
    if img_w % patch_w != 0:
        raise ValueError(f"Image width ({img_w}) must be divisible by patch width ({patch_w})")

    # Calculate number of patches to ensure it's reasonable
    num_patches = (img_h // patch_h) * (img_w // patch_w)
    if num_patches <= 0:
        raise ValueError(f"Number of patches must be positive, got {num_patches}")
    if num_patches > 10000:  # Reasonable upper limit
        logger.warning(f"Large number of patches ({num_patches}) may cause memory issues")

    model = ViT(
        input_shape=input_shape,
        num_classes=num_classes,
        scale=scale,
        patch_size=patch_size,
        include_top=include_top,
        pooling=pooling,
        dropout_rate=dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        pos_dropout_rate=pos_dropout_rate,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_initializer=bias_initializer,
        bias_regularizer=bias_regularizer,
        norm_type=norm_type,
        normalization_position=normalization_position,
        ffn_type=ffn_type,
        activation=activation,
        **kwargs
    )

    logger.info(f"VisionTransformer-{scale} created successfully")
    logger.info(f"Configuration: {num_patches} patches ({img_h//patch_h}x{img_w//patch_w}), {num_classes} classes")
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
