"""
SigLIP Vision Transformer Model

This module provides `SigLIPVisionTransformer`, a Keras Model that implements the
SigLIP (Sigmoid-Loss for Language-Image Pre-training) Vision Transformer architecture
with modern Keras 3 best practices.

A Vision Transformer model that applies the Transformer architecture to computer vision
by treating images as sequences of patches. This implementation follows the SigLIP
approach with enhanced two-stage patch embedding strategy.

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

Example:
    ```python
    # Basic usage
    model = SigLIPVisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12
    )

    # Compile and train
    model.compile(
        optimizer='adamw',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Use factory functions for standard configurations
    model = create_siglip_vit_base()  # ViT-Base configuration
    model = create_siglip_vit_large()  # ViT-Large configuration
    model = create_siglip_vit_small()  # ViT-Small configuration
    ```
"""

import keras
from keras import ops, layers, initializers, regularizers
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict, Any, List

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.transformer import TransformerLayer
from dl_techniques.layers.positional_embedding import PositionalEmbedding

# ---------------------------------------------------------------------


@dataclass
class ViTSigLIPConfig:
    """
    Configuration class for SigLIP Vision Transformer model.

    This dataclass contains all configuration parameters needed to instantiate
    a SigLIP Vision Transformer model. It provides methods for serialization
    and deserialization for easy model configuration management.

    Args:
        img_size: Input image size (square images). Must be divisible by patch_size.
        patch_size: Size of image patches. Must divide img_size evenly.
        embed_dim: Embedding dimension for tokens. Must be divisible by num_heads.
        depth: Number of transformer layers. Must be positive.
        num_heads: Number of attention heads. Must divide embed_dim evenly.
        mlp_ratio: Ratio of MLP hidden dimension to embedding dimension.
        dropout: Dropout rate for embeddings and transformer blocks.
        activation: Activation function for MLP layers.
        attention_type: Type of attention mechanism to use.
        normalization_type: Type of normalization to use.
        normalization_position: Position of normalization ('pre' or 'post').
        ffn_type: Type of feed-forward network to use.
        use_bias: Whether to use bias terms in linear layers.
        kernel_initializer: Initializer for kernel weights.
        bias_initializer: Initializer for bias weights.
        kernel_regularizer: Regularizer for kernel weights.
        bias_regularizer: Regularizer for bias weights.
    """
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
        """
        Convert configuration to dictionary for serialization.

        Returns:
            Dictionary containing all configuration parameters.
        """
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
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ViTSigLIPConfig':
        """
        Create configuration from dictionary.

        Args:
            config_dict: Dictionary containing configuration parameters.

        Returns:
            ViTSigLIPConfig instance.
        """
        return cls(**config_dict)


@keras.saving.register_keras_serializable()
class SigLIPVisionTransformer(keras.Model):
    """
    SigLIP Vision Transformer Model with modern Keras 3 implementation.

    This model implements the SigLIP Vision Transformer architecture with proper
    Keras 3 serialization support and modern best practices. It follows the
    create-in-init, build-weights pattern for robust serialization.

    The model treats images as sequences of patches and processes them through
    a stack of transformer layers. A learnable [CLS] token is used for global
    image representation.

    Args:
        img_size: Integer, input image size (square). Must be positive and
            divisible by patch_size. Defaults to 224.
        patch_size: Integer, size of image patches. Must be positive and
            divide img_size evenly. Defaults to 16.
        embed_dim: Integer, embedding dimension for tokens. Must be positive and
            divisible by num_heads. Defaults to 768.
        depth: Integer, number of transformer layers. Must be positive.
            Defaults to 12.
        num_heads: Integer, number of attention heads. Must be positive and
            divide embed_dim evenly. Defaults to 12.
        mlp_ratio: Float, ratio of MLP hidden dimension to embedding dimension.
            Must be positive. Defaults to 4.0.
        dropout: Float, dropout rate for embeddings and transformer blocks.
            Must be between 0 and 1. Defaults to 0.0.
        activation: String or callable, activation function for MLP layers.
            Defaults to 'gelu'.
        attention_type: String, type of attention mechanism. Passed to
            TransformerLayer. Defaults to 'multi_head_attention'.
        normalization_type: String, type of normalization. Passed to
            TransformerLayer. Defaults to 'layer_norm'.
        normalization_position: String, position of normalization ('pre' or 'post').
            Defaults to 'post'.
        ffn_type: String, type of feed-forward network. Passed to TransformerLayer.
            Defaults to 'mlp'.
        use_bias: Boolean, whether to use bias terms in linear layers.
            Defaults to True.
        kernel_initializer: String or initializer, initializer for kernel weights.
            Defaults to 'glorot_uniform'.
        bias_initializer: String or initializer, initializer for bias weights.
            Defaults to 'zeros'.
        kernel_regularizer: Optional regularizer for kernel weights.
        bias_regularizer: Optional regularizer for bias weights.
        **kwargs: Additional keyword arguments for Model base class.

    Input shape:
        4D tensor with shape: `(batch_size, img_size, img_size, 3)`

    Output shape:
        3D tensor with shape: `(batch_size, num_patches + 1, embed_dim)`
        Where num_patches = (img_size // patch_size) ** 2
        The first token (index 0) is the [CLS] token for global representation.

    Raises:
        ValueError: If img_size is not divisible by patch_size.
        ValueError: If embed_dim is not divisible by num_heads.
        ValueError: If any parameter is invalid.

    Example:
        ```python
        # Basic usage
        model = SigLIPVisionTransformer(
            img_size=224,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            dropout=0.1
        )

        # Build and compile
        model.build((None, 224, 224, 3))
        model.compile(
            optimizer='adamw',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Get features for an image
        images = keras.random.normal((8, 224, 224, 3))
        features = model(images)  # Shape: (8, 197, 768)

        # Extract CLS token for classification
        cls_features = model.get_cls_token(features)  # Shape: (8, 768)

        # Extract patch tokens for dense prediction
        patch_features = model.get_patch_tokens(features)  # Shape: (8, 196, 768)
        ```

    Note:
        This implementation follows the modern Keras 3 pattern where all sub-layers
        are created in __init__ and weights are created in build(). This ensures
        proper serialization and eliminates build errors.
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
        kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
        bias_initializer: Union[str, initializers.Initializer] = 'zeros',
        kernel_regularizer: Optional[Union[str, regularizers.Regularizer]] = None,
        bias_regularizer: Optional[Union[str, regularizers.Regularizer]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate configuration parameters
        self._validate_config(img_size, patch_size, embed_dim, depth, num_heads,
                            mlp_ratio, dropout)

        # Store configuration parameters
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

        # Store initializers and regularizers
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        # Calculate derived parameters
        self.num_patches = (img_size // patch_size) ** 2
        self.intermediate_size = int(embed_dim * mlp_ratio)

        # CREATE all sub-layers in __init__ (Modern Keras 3 pattern)

        # 1. SigLIP-style two-stage patch embedding
        self.siglip_patch_embed = self._create_patch_embedding()

        # 2. Positional embedding for all tokens (CLS + patches)
        self.pos_embed = PositionalEmbedding(
            max_seq_len=self.num_patches + 1,  # +1 for CLS token
            dim=self.embed_dim,
            dropout=0.0,  # We handle dropout separately
            name='pos_embed'
        )

        # 3. Embedding dropout (if needed)
        self.dropout_layer = None
        if self.dropout > 0.0:
            self.dropout_layer = layers.Dropout(
                rate=self.dropout,
                name='embed_dropout'
            )

        # 4. Stack of transformer blocks
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

        # 5. Final normalization (only if using pre-normalization)
        self.final_norm = None
        if self.normalization_position == 'pre':
            self.final_norm = layers.LayerNormalization(
                epsilon=1e-6,
                name='final_norm'
            )

        # Initialize weight attributes to None - created in build()
        self.cls_token = None

        logger.info(
            f"Initialized SigLIPVisionTransformer: "
            f"{self.num_patches} patches, {self.depth} blocks, "
            f"{embed_dim}D embeddings"
        )

    def _validate_config(
        self,
        img_size: int,
        patch_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float
    ) -> None:
        """
        Validate configuration parameters.

        Args:
            img_size: Input image size.
            patch_size: Patch size.
            embed_dim: Embedding dimension.
            depth: Number of layers.
            num_heads: Number of attention heads.
            mlp_ratio: MLP expansion ratio.
            dropout: Dropout rate.

        Raises:
            ValueError: If any parameter is invalid.
        """
        if img_size <= 0:
            raise ValueError(f"img_size must be positive, got {img_size}")

        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")

        if img_size % patch_size != 0:
            raise ValueError(
                f"img_size ({img_size}) must be divisible by "
                f"patch_size ({patch_size})"
            )

        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")

        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )

        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")

        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")

        if not (0.0 <= dropout <= 1.0):
            raise ValueError(f"dropout must be between 0 and 1, got {dropout}")

    def _create_patch_embedding(self) -> keras.Sequential:
        """
        Create SigLIP-style two-stage patch embedding.

        The two-stage approach provides better feature extraction compared to
        single-stage patch embedding used in standard ViT.

        Returns:
            Sequential model for patch embedding.
        """
        # Calculate intermediate filter size and stride
        stage1_filters = self.embed_dim // 2
        stage1_kernel_size = self.patch_size // 2
        stage1_stride = self.patch_size // 2

        return keras.Sequential([
            # Stage 1: Coarse-grained patching
            layers.Conv2D(
                filters=stage1_filters,
                kernel_size=stage1_kernel_size,
                strides=stage1_stride,
                padding='valid',
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name='patch_embed_conv1'
            ),

            # Normalization after first conv
            layers.LayerNormalization(
                epsilon=1e-6,
                name='patch_embed_norm1'
            ),

            # Activation
            layers.Activation('gelu', name='patch_embed_gelu'),

            # Stage 2: Refinement to final embedding dimension
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

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create the model's weights.

        This method creates the learnable CLS token weight. All sub-layers
        are created in __init__ following modern Keras 3 pattern.

        Args:
            input_shape: Shape tuple of the input tensor.
        """
        # Validate input shape
        if len(input_shape) != 4:
            raise ValueError(
                f"Expected 4D input (batch, height, width, channels), "
                f"got {len(input_shape)}D: {input_shape}"
            )

        expected_shape = (None, self.img_size, self.img_size, 3)
        if (input_shape[1] != self.img_size or
            input_shape[2] != self.img_size or
            input_shape[3] != 3):
            logger.warning(
                f"Input shape {input_shape} doesn't match expected "
                f"{expected_shape}. Model may not work correctly."
            )

        # CREATE the learnable CLS token weight
        self.cls_token = self.add_weight(
            name='cls_token',
            shape=(1, 1, self.embed_dim),
            initializer='truncated_normal',
            regularizer=self.kernel_regularizer,
            trainable=True,
        )

        # Let Keras know the build is complete
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass of the SigLIP Vision Transformer.

        Args:
            inputs: Input tensor with shape (batch_size, img_size, img_size, 3).
            training: Boolean indicating whether the model is in training mode.

        Returns:
            Output tensor with shape (batch_size, num_patches + 1, embed_dim).
            The first token (index 0) is the [CLS] token.
        """
        # Get batch size for dynamic tensor operations
        batch_size = ops.shape(inputs)[0]

        # 1. Patch embedding: (B, H, W, C) -> (B, P, D)
        # where P = num_patches, D = embed_dim
        x = self.siglip_patch_embed(inputs, training=training)

        # Reshape from conv output to sequence: (B, H', W', D) -> (B, P, D)
        x = ops.reshape(x, [batch_size, self.num_patches, self.embed_dim])

        # 2. Prepend CLS token: (B, P, D) -> (B, P+1, D)
        # Tile CLS token for the batch
        cls_tokens = ops.tile(self.cls_token, [batch_size, 1, 1])
        x = ops.concatenate([cls_tokens, x], axis=1)

        # 3. Add positional embeddings
        x = self.pos_embed(x, training=training)

        # 4. Apply embedding dropout if configured
        if self.dropout_layer is not None:
            x = self.dropout_layer(x, training=training)

        # 5. Pass through transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, layer_idx=i, training=training)

        # 6. Apply final normalization for pre-norm architecture
        if self.final_norm is not None:
            x = self.final_norm(x, training=training)

        return x

    def get_cls_token(self, features: keras.KerasTensor) -> keras.KerasTensor:
        """
        Extract CLS token from model features.

        Args:
            features: Output tensor from the model with shape
                (batch_size, num_patches + 1, embed_dim).

        Returns:
            CLS token tensor with shape (batch_size, embed_dim).
        """
        return features[:, 0, :]

    def get_patch_tokens(self, features: keras.KerasTensor) -> keras.KerasTensor:
        """
        Extract patch tokens from model features.

        Args:
            features: Output tensor from the model with shape
                (batch_size, num_patches + 1, embed_dim).

        Returns:
            Patch tokens tensor with shape (batch_size, num_patches, embed_dim).
        """
        return features[:, 1:, :]

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute output shape of the model.

        Args:
            input_shape: Shape tuple of the input.

        Returns:
            Output shape tuple.
        """
        batch_size = input_shape[0]
        sequence_length = self.num_patches + 1  # +1 for CLS token
        return (batch_size, sequence_length, self.embed_dim)

    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration for serialization.

        Returns:
            Dictionary containing all configuration parameters.
        """
        config = super().get_config()
        config.update({
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
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        })
        return config


# ---------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------


def create_siglip_vit(
    config: Optional[ViTSigLIPConfig] = None,
    **kwargs: Any
) -> SigLIPVisionTransformer:
    """
    Factory function to create SigLIP Vision Transformer model.

    Args:
        config: Model configuration. If None, uses default configuration.
        **kwargs: Additional arguments to override configuration.

    Returns:
        SigLIPVisionTransformer model instance.

    Example:
        ```python
        # Using default config
        model = create_siglip_vit()

        # Using custom config
        config = ViTSigLIPConfig(img_size=384, patch_size=32)
        model = create_siglip_vit(config)

        # Overriding specific parameters
        model = create_siglip_vit(embed_dim=1024, depth=24)
        ```
    """
    if config is None:
        config = ViTSigLIPConfig()

    # Override config with kwargs
    config_dict = config.to_dict()
    config_dict.update(kwargs)

    return SigLIPVisionTransformer(**config_dict)


def create_siglip_vit_base(**kwargs: Any) -> SigLIPVisionTransformer:
    """
    Create SigLIP ViT-Base model configuration.

    ViT-Base configuration:
    - 768 embedding dimensions
    - 12 transformer layers
    - 12 attention heads
    - ~86M parameters

    Args:
        **kwargs: Additional arguments to override base configuration.

    Returns:
        SigLIPVisionTransformer with Base configuration.

    Example:
        ```python
        model = create_siglip_vit_base()
        model = create_siglip_vit_base(dropout=0.1, img_size=384)
        ```
    """
    config = ViTSigLIPConfig(
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0
    )
    return create_siglip_vit(config, **kwargs)


def create_siglip_vit_large(**kwargs: Any) -> SigLIPVisionTransformer:
    """
    Create SigLIP ViT-Large model configuration.

    ViT-Large configuration:
    - 1024 embedding dimensions
    - 24 transformer layers
    - 16 attention heads
    - ~307M parameters

    Args:
        **kwargs: Additional arguments to override large configuration.

    Returns:
        SigLIPVisionTransformer with Large configuration.

    Example:
        ```python
        model = create_siglip_vit_large()
        model = create_siglip_vit_large(dropout=0.1, normalization_position='pre')
        ```
    """
    config = ViTSigLIPConfig(
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0
    )
    return create_siglip_vit(config, **kwargs)


def create_siglip_vit_small(**kwargs: Any) -> SigLIPVisionTransformer:
    """
    Create SigLIP ViT-Small model configuration.

    ViT-Small configuration:
    - 384 embedding dimensions
    - 12 transformer layers
    - 6 attention heads
    - ~22M parameters

    Args:
        **kwargs: Additional arguments to override small configuration.

    Returns:
        SigLIPVisionTransformer with Small configuration.

    Example:
        ```python
        model = create_siglip_vit_small()
        model = create_siglip_vit_small(dropout=0.2, patch_size=8)
        ```
    """
    config = ViTSigLIPConfig(
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0
    )
    return create_siglip_vit(config, **kwargs)


def build_and_initialize_siglip_vit(
    model: SigLIPVisionTransformer,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    compile_config: Optional[Dict[str, Any]] = None
) -> SigLIPVisionTransformer:
    """
    Build and initialize SigLIP Vision Transformer model.

    This helper function builds the model with the specified input shape
    and optionally compiles it with the provided configuration.

    Args:
        model: SigLIPVisionTransformer model instance.
        input_shape: Input shape tuple (height, width, channels).
            Defaults to (224, 224, 3).
        compile_config: Optional compilation configuration dictionary.
            Should contain 'optimizer', 'loss', and optionally 'metrics'.

    Returns:
        Built and optionally compiled model.

    Example:
        ```python
        model = create_siglip_vit_base()

        compile_config = {
            'optimizer': 'adamw',
            'loss': 'categorical_crossentropy',
            'metrics': ['accuracy', 'top_5_categorical_accuracy']
        }

        model = build_and_initialize_siglip_vit(
            model,
            input_shape=(224, 224, 3),
            compile_config=compile_config
        )
        ```
    """
    # Build model with proper input shape
    batch_input_shape = (None,) + input_shape
    model.build(batch_input_shape)

    logger.info(f"Model built with input shape: {batch_input_shape}")
    logger.info(f"Model has {model.count_params():,} parameters")

    # Compile if configuration provided
    if compile_config:
        model.compile(**compile_config)
        logger.info("Model compiled successfully")

    return model