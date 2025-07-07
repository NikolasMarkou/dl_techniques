"""
Vision Transformer (ViT) Model Implementation (Refined Version)

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
        pos_dropout_rate: Dropout rate for positional embeddings.
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
            pos_dropout_rate: float = 0.0,
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
            pos_dropout_rate: Dropout rate for positional embeddings.
            kernel_initializer: Weight initializer for all layers.
            kernel_regularizer: Weight regularizer for all layers.
            norm_type: Type of normalization ('layer' or 'rms').
            name: Model name.
            **kwargs: Additional keyword arguments.
        """
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

        # Store configuration - ensure input_shape is always a tuple
        self.input_shape_config = tuple(input_shape) if not isinstance(input_shape, tuple) else input_shape
        self.num_classes = num_classes
        self.scale = scale
        self.patch_size = (patch_h, patch_w)  # Always store as tuple
        self.include_top = include_top
        self.pooling = pooling
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.pos_dropout_rate = pos_dropout_rate
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.norm_type = norm_type

        # Validate inputs
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

        # Calculate number of patches (safe from division by zero due to validation above)
        self.num_patches = (img_h // patch_h) * (img_w // patch_w)
        self.max_seq_len = self.num_patches + 1  # +1 for CLS token

        # Validate we have at least one patch
        if self.num_patches <= 0:
            raise ValueError(f"Number of patches must be positive, got {self.num_patches}")

        # Initialize layers as None - they will be created in build()
        self.patch_embed = None
        self.cls_token = None
        self.pos_embed = None
        self.transformer_layers = None
        self.norm = None
        self.head_dropout = None
        self.head = None
        self.global_pool = None

        # Store build state for serialization
        self._build_input_shape = None
        self._layers_built = False

        logger.info(f"Created VisionTransformer-{scale} with {self.embed_dim}d, {self.num_heads}h, {self.num_layers}L")
        logger.info(f"Image shape: {self.input_shape_config}, Patch size: {self.patch_size}, Num patches: {self.num_patches}")

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
            patch_size=self.patch_size,  # Already validated and stored as tuple
            embed_dim=self.embed_dim,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="patch_embed"
        )

        # Build patch embedding to get actual number of patches
        # Convert input_shape_config to tuple to handle TrackedList during deserialization
        try:
            input_shape_tuple = tuple(self.input_shape_config)
        except (TypeError, ValueError):
            # Fallback if conversion fails
            if hasattr(self.input_shape_config, '__iter__'):
                input_shape_tuple = tuple(list(self.input_shape_config))
            else:
                # Last resort - use what we have
                input_shape_tuple = self.input_shape_config

        dummy_input_shape = (None,) + input_shape_tuple
        self.patch_embed.build(dummy_input_shape)

        # Get actual number of patches from built layer and validate
        actual_num_patches = self.patch_embed.num_patches
        if actual_num_patches is not None:
            if actual_num_patches <= 0:
                raise ValueError(f"Invalid number of patches: {actual_num_patches}")
            self.num_patches = actual_num_patches
            self.max_seq_len = self.num_patches + 1
        elif self.num_patches <= 0:
            raise ValueError(f"Invalid number of patches: {self.num_patches}")

        # Validate max_seq_len is reasonable
        if self.max_seq_len <= 1:
            raise ValueError(f"Invalid sequence length: {self.max_seq_len}")

        # CLS token
        self.cls_token = self.add_weight(
            name="cls_token",
            shape=(1, 1, self.embed_dim),
            initializer="zeros",
            trainable=True
        )

        # Positional embedding with proper parameter names
        self.pos_embed = PositionalEmbedding(
            max_seq_len=self.max_seq_len,
            dim=self.embed_dim,
            dropout=self.pos_dropout_rate,
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
        if len(input_shape) < 4:
            raise ValueError(f"Expected 4D input shape (batch, height, width, channels), got {input_shape}")

        batch_size = input_shape[0]

        if self.include_top:
            return (batch_size, self.num_classes)
        else:
            if self.pooling in ["cls", "mean", "max"]:
                return (batch_size, self.embed_dim)
            else:
                # Safely calculate the number of patches if not already set
                if hasattr(self, 'max_seq_len') and self.max_seq_len > 0:
                    return (batch_size, self.max_seq_len, self.embed_dim)
                else:
                    # Fallback calculation
                    img_h, img_w = input_shape[1], input_shape[2]
                    if img_h is not None and img_w is not None:
                        patch_h, patch_w = self.patch_size
                        if patch_h > 0 and patch_w > 0:  # Avoid division by zero
                            num_patches = (img_h // patch_h) * (img_w // patch_w)
                            seq_len = num_patches + 1  # +1 for CLS token
                            return (batch_size, seq_len, self.embed_dim)

                    # If we can't calculate, return None for unknown dimensions
                    return (batch_size, None, self.embed_dim)

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        config = super().get_config()

        # Ensure all values are properly serializable
        input_shape_tuple = tuple(self.input_shape_config)
        if hasattr(self.input_shape_config, '__iter__') and not isinstance(self.input_shape_config, (str, tuple)):
            try:
                input_shape_tuple = tuple(list(self.input_shape_config))
            except (TypeError, ValueError):
                input_shape_tuple = self.input_shape_config

        # Ensure patch_size is properly serialized
        patch_size_tuple = tuple(self.patch_size) if hasattr(self.patch_size, '__iter__') else self.patch_size

        config.update({
            "input_shape": input_shape_tuple,
            "num_classes": int(self.num_classes),
            "scale": str(self.scale),
            "patch_size": patch_size_tuple,
            "include_top": bool(self.include_top),
            "pooling": self.pooling,
            "dropout_rate": float(self.dropout_rate),
            "attention_dropout_rate": float(self.attention_dropout_rate),
            "pos_dropout_rate": float(self.pos_dropout_rate),
            "kernel_initializer": str(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "norm_type": str(self.norm_type),
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
            # Ensure input_shape is a tuple
            if not isinstance(input_shape, tuple):
                input_shape = tuple(input_shape)
            self.build(input_shape)

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
        # Ensure we have valid configuration before creating feature extractor
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
            norm_type=self.norm_type,
            name=f"{self.name}_feature_extractor"
        )

    def get_attention_weights(
        self,
        inputs: keras.KerasTensor,
        layer_idx: Optional[int] = None
    ) -> Union[keras.KerasTensor, Dict[int, keras.KerasTensor]]:
        """
        Extract attention weights from transformer layers.

        Args:
            inputs: Input tensor.
            layer_idx: Specific layer index to extract weights from.
                      If None, returns weights from all layers.

        Returns:
            Attention weights tensor or dictionary of weights.
        """
        if not self._layers_built:
            raise ValueError("Model must be built before extracting attention weights")

        if not hasattr(self, 'transformer_layers') or not self.transformer_layers:
            raise ValueError("No transformer layers found in the model")

        if layer_idx is not None and (layer_idx < 0 or layer_idx >= len(self.transformer_layers)):
            raise ValueError(f"layer_idx must be between 0 and {len(self.transformer_layers)-1}, got {layer_idx}")

        # Forward pass through patch embedding and positional encoding
        x = self.patch_embed(inputs)
        batch_size = ops.shape(x)[0]
        cls_tokens = ops.broadcast_to(self.cls_token, (batch_size, 1, self.embed_dim))
        x = ops.concatenate([cls_tokens, x], axis=1)
        x = self.pos_embed(x)

        attention_weights = {}

        # Extract attention weights from each transformer layer
        for i, layer in enumerate(self.transformer_layers):
            if hasattr(layer, 'get_attention_weights'):
                try:
                    attention_weights[i] = layer.get_attention_weights(x)
                    x = layer(x)
                except Exception as e:
                    logger.warning(f"Failed to extract attention weights from layer {i}: {e}")
                    x = layer(x)
            else:
                # If layer doesn't support attention weight extraction, just forward
                x = layer(x)

        if layer_idx is not None:
            return attention_weights.get(layer_idx, None)

        return attention_weights

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
        logger.info(f"Dropout Rate: {self.dropout_rate}")
        logger.info(f"Attention Dropout Rate: {self.attention_dropout_rate}")
        logger.info(f"Positional Dropout Rate: {self.pos_dropout_rate}")
        logger.info(f"Normalization Type: {self.norm_type}")
        logger.info(f"Include Top: {self.include_top}")
        logger.info(f"Pooling: {self.pooling}")
        logger.info(f"Number of Classes: {self.num_classes}")
        if self._layers_built:
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
        pos_dropout_rate: Dropout rate for positional embeddings.
        kernel_initializer: Weight initializer for all layers.
        kernel_regularizer: Weight regularizer for all layers.
        norm_type: Type of normalization ('layer' or 'rms').
        **kwargs: Additional arguments for ViT.

    Returns:
        ViT instance.

    Raises:
        ValueError: If input parameters are invalid.

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
        ...     dropout_rate=0.1,
        ...     pos_dropout_rate=0.1
        ... )
    """
    # Validate basic parameters before creating the model
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
        norm_type=norm_type,
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