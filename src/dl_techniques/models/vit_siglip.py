"""
SigLIP Vision Transformer Model

This module provides `SigLIPVisionTransformer`, a complete Vision Transformer model
that implements the SigLIP (Sigmoid-Loss for Language-Image Pre-training) architecture
with two-stage patch embedding, positional encoding, and transformer blocks.

A Vision Transformer model that applies the Transformer architecture to computer vision
by treating images as sequences of patches. This implementation follows the SigLIP
approach with enhanced patch embedding strategy.

Key Architectural Components:

1.  **SigLIP-Style Two-Stage Patch Embedding:**
    -   Two-stage convolution approach for better feature extraction
    -   Stage 1: Coarse-grained patching with medium kernel
    -   Stage 2: Refinement to final embedding dimension

2.  **Learnable [CLS] Token:**
    -   Global summary token for classification tasks
    -   Prepended to patch sequence

3.  **Positional Embeddings:**
    -   Learnable positional encoding for spatial awareness
    -   Applied to both CLS and patch tokens

4.  **Stack of Transformer Blocks:**
    -   Configurable transformer encoder blocks
    -   Multi-head self-attention and feed-forward networks

5.  **Final Normalization:**
    -   Layer normalization for well-conditioned outputs
"""

import keras
from keras import ops, layers
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict, Any


# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..utils.logger import logger
from ..layers.transformer import TransformerLayer
from ..layers.positional_embedding import PositionalEmbedding

# ---------------------------------------------------------------------

@dataclass
class ViTSigLIPConfig:
    """Configuration for SigLIP Vision Transformer model."""
    img_size: int = 224
    patch_size: int = 16
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    activation: str = 'gelu'
    attention_type: str = 'multi_head_attention'
    normalization_type: str = 'layer_norm'
    normalization_position: str = 'post'
    ffn_type: str = 'mlp'
    use_bias: bool = True
    kernel_initializer: str = 'glorot_uniform'
    bias_initializer: str = 'zeros'
    kernel_regularizer: Optional[str] = None
    bias_regularizer: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            'img_size': self.img_size,
            'patch_size': self.patch_size,
            'embed_dim': self.embed_dim,
            'depth': self.depth,
            'num_heads': self.num_heads,
            'mlp_ratio': self.mlp_ratio,
            'dropout': self.dropout,
            'activation': self.activation,
            'attention_type': self.attention_type,
            'normalization_type': self.normalization_type,
            'normalization_position': self.normalization_position,
            'ffn_type': self.ffn_type,
            'use_bias': self.use_bias,
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'bias_regularizer': self.bias_regularizer,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ViTSigLIP':
        """Create configuration from dictionary."""
        return cls(**config_dict)


@keras.saving.register_keras_serializable()
class ViTSigLIP(keras.Model):
    """SigLIP Vision Transformer Model

    Complete Vision Transformer model following SigLIP architecture with
    two-stage patch embedding, positional encoding, and configurable transformer blocks.

    This model is ready for training and inference, providing both CLS token output
    for classification tasks and patch token outputs for dense prediction tasks.

    Args:
        img_size: Input image size. Defaults to 224.
        patch_size: Size of image patches. Defaults to 16.
        embed_dim: Embedding dimension. Defaults to 768.
        depth: Number of transformer blocks. Defaults to 12.
        num_heads: Number of attention heads. Defaults to 12.
        mlp_ratio: MLP expansion ratio. Defaults to 4.0.
        dropout: Dropout rate. Defaults to 0.0.
        attention_type: Type of attention mechanism. Defaults to 'multi_head_attention'.
        normalization_type: Type of normalization. Defaults to 'layer_norm'.
        normalization_position: Position of normalization ('pre' or 'post'). Defaults to 'post'.
        ffn_type: Type of feed-forward network. Defaults to 'mlp'.
        use_bias: Whether to use bias in linear layers. Defaults to True.
        kernel_initializer: Initializer for convolution kernels. Defaults to 'glorot_uniform'.
        bias_initializer: Initializer for bias vectors. Defaults to 'zeros'.
        kernel_regularizer: Regularizer for convolution kernels. Defaults to None.
        bias_regularizer: Regularizer for bias vectors. Defaults to None.
        **kwargs: Additional keyword arguments.

    Examples:
        >>> # Create and use the model
        >>> model = ViTSigLIP(
        ...     img_size=224,
        ...     patch_size=16,
        ...     embed_dim=768,
        ...     depth=12,
        ...     num_heads=12
        ... )
        >>>
        >>> # Build the model
        >>> inputs = keras.Input(shape=(224, 224, 3))
        >>> outputs = model(inputs)
        >>>
        >>> # For classification
        >>> cls_features = model.get_cls_token(outputs)  # Shape: (batch, embed_dim)
        >>>
        >>> # For dense prediction
        >>> patch_features = model.get_patch_tokens(outputs)  # Shape: (batch, num_patches, embed_dim)

        >>> # Advanced configuration
        >>> model = ViTSigLIP(
        ...     embed_dim=512,
        ...     depth=8,
        ...     attention_type='window_attention',
        ...     normalization_type='rms_norm',
        ...     normalization_position='pre',
        ...     ffn_type='swiglu',
        ...     kernel_regularizer='l2',
        ...     dropout=0.1
        ... )
    """

    def __init__(
            self,
            img_size: int = 224,
            patch_size: int = 16,
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,
            dropout: float = 0.0,
            activation: str = 'gelu',
            attention_type: str = 'multi_head_attention',
            normalization_type: str = 'layer_norm',
            normalization_position: str = 'post',
            ffn_type: str = 'mlp',
            use_bias: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        self._validate_config(img_size, patch_size, embed_dim, num_heads, dropout)

        # Store configuration parameters - following the "Create vs Build" pattern
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.activation = activation
        self.attention_type = attention_type
        self.normalization_type = normalization_type
        self.normalization_position = normalization_position
        self.ffn_type = ffn_type
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # Computed properties
        self.num_patches = (img_size // patch_size) ** 2
        self.intermediate_size = int(embed_dim * mlp_ratio)

        # Initialize model components to None - will be created in build()
        self.siglip_patch_embed = None
        self.cls_token = None
        self.pos_embed = None
        self.dropout_layer = None
        self.transformer_blocks = None
        self.norm = None

        # Store build input shape for serialization - following the "Managed State" principle
        self._build_input_shape = None

        logger.info(f"Initialized ViTSigLIP with {self.num_patches} patches, "
                    f"{self.depth} transformer blocks, and {embed_dim} embedding dimension")

    def _validate_config(
            self,
            img_size: int,
            patch_size: int,
            embed_dim: int,
            num_heads: int,
            dropout: float
    ) -> None:
        """Validate model configuration parameters.

        Args:
            img_size: Input image size.
            patch_size: Size of image patches.
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            dropout: Dropout rate.

        Raises:
            ValueError: If configuration is invalid.
        """
        if img_size <= 0:
            raise ValueError(f"img_size must be positive, got {img_size}")
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        if img_size % patch_size != 0:
            raise ValueError(f"Image size ({img_size}) must be divisible by patch size ({patch_size})")
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if embed_dim % num_heads != 0:
            raise ValueError(f"Embedding dimension ({embed_dim}) must be divisible by number of heads ({num_heads})")
        if not (0.0 <= dropout <= 1.0):
            raise ValueError(f"dropout must be between 0 and 1, got {dropout}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the vision transformer model components.

        Args:
            input_shape: Shape tuple of the input tensor.

        Raises:
            ValueError: If input shape is invalid.
        """
        if self.built:
            return

        # Store input shape for serialization
        self._build_input_shape = input_shape

        # Validate input shape
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input shape (batch, height, width, channels), got {input_shape}")

        if input_shape[1] != self.img_size or input_shape[2] != self.img_size:
            raise ValueError(f"Input shape height/width {input_shape[1:3]} must match img_size ({self.img_size})")

        logger.info(f"Building SigLIPVisionTransformer model with input shape: {input_shape}")

        # Create SigLIP-style two-stage patch embedding - following the "Create vs Build" pattern
        self.siglip_patch_embed = self._create_patch_embedding()

        # Create CLS token using managed state principle
        self.cls_token = self.add_weight(
            shape=(1, 1, self.embed_dim),
            initializer='truncated_normal',
            trainable=True,
            name='cls_token'
        )

        # Create positional embedding
        self.pos_embed = PositionalEmbedding(
            max_seq_len=self.num_patches + 1,  # +1 for CLS token
            dim=self.embed_dim,
            dropout=0.0,  # Handle dropout separately
            name='pos_embed'
        )

        # Create dropout layer for embeddings
        if self.dropout > 0.0:
            self.dropout_layer = layers.Dropout(
                rate=self.dropout,
                name='embed_dropout'
            )

        # Create transformer blocks - following the "Parent Controls the Build" rule
        self.transformer_blocks = []
        for i in range(self.depth):
            block = TransformerLayer(
                hidden_size=self.embed_dim,
                num_heads=self.num_heads,
                intermediate_size=self.intermediate_size,
                attention_type=self.attention_type,
                normalization_type=self.normalization_type,
                normalization_position=self.normalization_position,
                ffn_type=self.ffn_type,
                dropout_rate=self.dropout,
                attention_dropout_rate=self.dropout,
                activation=self.activation,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name=f'transformer_block_{i}'
            )
            self.transformer_blocks.append(block)

        # Create final layer normalization
        self.norm = layers.LayerNormalization(name='final_norm')

        # Build all child components explicitly - following the "Parent Controls the Build" rule
        self._build_child_components(input_shape)

        # Build parent model last
        super().build(input_shape)
        logger.info("SigLIPVisionTransformer model build completed")

    def _create_patch_embedding(self) -> keras.Sequential:
        """Create SigLIP-style two-stage patch embedding.

        Returns:
            Sequential model for patch embedding.
        """
        return keras.Sequential([
            layers.Conv2D(
                filters=self.embed_dim // 2,
                kernel_size=self.patch_size // 2,
                strides=self.patch_size // 2,
                padding='valid',
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name='patch_embed_conv1'
            ),
            layers.LayerNormalization(name='patch_embed_norm1'),
            layers.Activation('gelu', name='patch_embed_gelu'),
            # Stage 2: Refine to final embedding dimension
            layers.Conv2D(
                filters=self.embed_dim,
                kernel_size=2,
                strides=2,
                padding='valid',
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name='patch_embed_conv2'
            ),
        ], name='siglip_patch_embed')

    def _build_child_components(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build all child components in the correct order.

        Args:
            input_shape: Shape tuple of the input tensor.
        """
        # Build patch embedding
        if not self.siglip_patch_embed.built:
            self.siglip_patch_embed.build(input_shape)

        # Compute shapes after patch embedding
        patch_height = self.img_size // self.patch_size
        patch_width = self.img_size // self.patch_size
        sequence_length = patch_height * patch_width + 1  # +1 for CLS token
        sequence_shape = (None, sequence_length, self.embed_dim)

        # Build positional embedding
        if not self.pos_embed.built:
            self.pos_embed.build(sequence_shape)

        # Build transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            if not block.built:
                block.build(sequence_shape)

        # Build final normalization
        if not self.norm.built:
            self.norm.build(sequence_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through the vision transformer model.

        Args:
            inputs: Input images of shape [batch_size, height, width, channels].
            training: Whether in training mode.

        Returns:
            Vision features of shape [batch_size, num_patches + 1, embed_dim].
            The first token is the CLS token, followed by patch tokens.
        """
        batch_size = ops.shape(inputs)[0]

        # SigLIP patch embedding
        x = self.siglip_patch_embed(inputs, training=training)  # [batch, h', w', embed_dim]

        # Reshape to sequence format
        patch_height = self.img_size // self.patch_size
        patch_width = self.img_size // self.patch_size
        x = ops.reshape(x, [batch_size, patch_height * patch_width, self.embed_dim])

        # Add CLS token
        cls_tokens = ops.tile(self.cls_token, [batch_size, 1, 1])
        x = ops.concatenate([cls_tokens, x], axis=1)

        # Add positional embeddings
        x = self.pos_embed(x, training=training)

        # Apply dropout to embeddings
        if self.dropout_layer is not None:
            x = self.dropout_layer(x, training=training)

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, training=training)

        # Final normalization. Only needed for pre-norm, as post-norm blocks already output normalized tensors.
        if self.normalization_position == 'pre':
            x = self.norm(x, training=training)

        return x

    def get_cls_token(self, features: keras.KerasTensor) -> keras.KerasTensor:
        """Extract CLS token from vision features for classification tasks.

        Args:
            features: Vision features from forward pass.

        Returns:
            CLS token of shape [batch_size, embed_dim].
        """
        return features[:, 0, :]  # First token is CLS

    def get_patch_tokens(self, features: keras.KerasTensor) -> keras.KerasTensor:
        """Extract patch tokens from vision features for dense prediction tasks.

        Args:
            features: Vision features from forward pass.

        Returns:
            Patch tokens of shape [batch_size, num_patches, embed_dim].
        """
        return features[:, 1:, :]  # Skip CLS token

    def get_spatial_features(self, features: keras.KerasTensor) -> keras.KerasTensor:
        """Reshape patch tokens back to spatial format for dense tasks.

        Args:
            features: Vision features from forward pass.

        Returns:
            Spatial features of shape [batch_size, patch_height, patch_width, embed_dim].
        """
        patch_tokens = self.get_patch_tokens(features)
        batch_size = ops.shape(patch_tokens)[0]
        patch_height = self.img_size // self.patch_size
        patch_width = self.img_size // self.patch_size

        return ops.reshape(
            patch_tokens,
            [batch_size, patch_height, patch_width, self.embed_dim]
        )

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization.

        Returns:
            Dictionary containing the model configuration.
        """
        config = super().get_config()
        config.update({
            "img_size": self.img_size,
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "dropout": self.dropout,
            "activation": self.activation,
            "attention_type": self.attention_type,
            "normalization_type": self.normalization_type,
            "normalization_position": self.normalization_position,
            "ffn_type": self.ffn_type,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get build configuration for serialization.

        Returns:
            Dictionary containing build configuration.
        """
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build from configuration for serialization.

        Args:
            config: Dictionary containing build configuration.
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ViTSigLIP':
        """Create model from configuration.

        Args:
            config: Dictionary containing model configuration.

        Returns:
            SigLIPVisionTransformer instance.
        """
        return cls(**config)


def create_siglip_vit(
        config: Optional[ViTSigLIPConfig] = None,
        **kwargs
) -> ViTSigLIP:
    """Factory function to create SigLIP Vision Transformer model.

    Args:
        config: Model configuration. If None, uses default configuration.
        **kwargs: Additional arguments to override configuration.

    Returns:
        SigLIPVisionTransformer model instance.
    """
    if config is None:
        config = ViTSigLIPConfig()

    # Override config with kwargs
    config_dict = config.to_dict()
    config_dict.update(kwargs)

    return ViTSigLIP(**config_dict)


def create_siglip_vit_base(**kwargs) -> ViTSigLIP:
    """Create SigLIP ViT-Base model configuration.

    Args:
        **kwargs: Additional arguments to override base configuration.

    Returns:
        SigLIPVisionTransformer with Base configuration.
    """
    config = ViTSigLIPConfig(
        embed_dim=768,
        depth=12,
        num_heads=12,
    )
    return create_siglip_vit(config, **kwargs)


def create_siglip_vit_large(**kwargs) -> ViTSigLIP:
    """Create SigLIP ViT-Large model configuration.

    Args:
        **kwargs: Additional arguments to override large configuration.

    Returns:
        SigLIPVisionTransformer with Large configuration.
    """
    config = ViTSigLIPConfig(
        embed_dim=1024,
        depth=24,
        num_heads=16,
    )
    return create_siglip_vit(config, **kwargs)


def create_siglip_vit_small(**kwargs) -> ViTSigLIP:
    """Create SigLIP ViT-Small model configuration.

    Args:
        **kwargs: Additional arguments to override small configuration.

    Returns:
        SigLIPVisionTransformer with Small configuration.
    """
    config = ViTSigLIPConfig(
        embed_dim=384,
        depth=12,
        num_heads=6,
    )
    return create_siglip_vit(config, **kwargs)


def build_and_initialize_siglip_vit(
        model: ViTSigLIP,
        input_shape: Tuple[int, int, int],
        compile_config: Optional[Dict[str, Any]] = None
) -> ViTSigLIP:
    """Build and initialize SigLIP Vision Transformer model.

    Args:
        model: SigLIPVisionTransformer model instance.
        input_shape: Input shape tuple (height, width, channels).
        compile_config: Optional compilation configuration.

    Returns:
        Built and optionally compiled model.
    """
    # Build model with proper input shape
    batch_input_shape = (None,) + input_shape
    model.build(batch_input_shape)

    logger.info(f"Model built with input shape: {batch_input_shape}")
    logger.info(f"Model has {model.count_params()} parameters")

    # Compile if configuration provided
    if compile_config:
        model.compile(**compile_config)
        logger.info("Model compiled successfully")

    return model

# ---------------------------------------------------------------------
