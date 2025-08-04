import keras
from keras import ops
from typing import Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from .positional_embedding import PositionalEmbedding
from .vision_transformer import VisionTransformerLayer

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class SigLIPVisionTransformer(keras.layers.Layer):
    """SigLIP-based Vision Transformer for nanoVLM.

    Implements a vision transformer following SigLIP architecture with
    two-stage patch embedding, positional encoding, and transformer blocks.
    SigLIP uses a different patch embedding strategy compared to standard ViT,
    employing a two-stage convolution approach for better feature extraction.

    Args:
        img_size: Input image size. Defaults to 224.
        patch_size: Size of image patches. Defaults to 16.
        embed_dim: Embedding dimension. Defaults to 768.
        depth: Number of transformer blocks. Defaults to 12.
        num_heads: Number of attention heads. Defaults to 12.
        mlp_ratio: MLP expansion ratio. Defaults to 4.0.
        dropout: Dropout rate. Defaults to 0.0.
        use_bias: Whether to use bias in linear layers. Defaults to True.
        kernel_initializer: Initializer for convolution kernels. Defaults to 'glorot_uniform'.
        bias_initializer: Initializer for bias vectors. Defaults to 'zeros'.
        kernel_regularizer: Regularizer for convolution kernels. Defaults to None.
        bias_regularizer: Regularizer for bias vectors. Defaults to None.
        **kwargs: Additional keyword arguments.

    Examples:
        >>> # Basic usage
        >>> vit = SigLIPVisionTransformer(
        ...     img_size=224,
        ...     patch_size=16,
        ...     embed_dim=768,
        ...     depth=12,
        ...     num_heads=12
        ... )
        >>> inputs = keras.random.uniform((2, 224, 224, 3))
        >>> outputs = vit(inputs)
        >>> print(outputs.shape)  # (2, 197, 768)

        >>> # With custom regularization
        >>> vit = SigLIPVisionTransformer(
        ...     embed_dim=512,
        ...     depth=8,
        ...     kernel_regularizer=keras.regularizers.L2(1e-4),
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
            use_bias: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if img_size % patch_size != 0:
            raise ValueError(f"Image size ({img_size}) must be divisible by patch size ({patch_size})")
        if embed_dim % num_heads != 0:
            raise ValueError(f"Embedding dimension ({embed_dim}) must be divisible by number of heads ({num_heads})")

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        self.num_patches = (img_size // patch_size) ** 2

        # Will be initialized in build()
        self.siglip_patch_embed = None
        self.cls_token = None
        self.pos_embed = None
        self.dropout_layer = None
        self.transformer_blocks = None
        self.norm = None
        self._build_input_shape = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the vision transformer components.

        Args:
            input_shape: Shape tuple of the input tensor.
        """
        self._build_input_shape = input_shape

        logger.info(f"Building SigLIPVisionTransformer with input shape: {input_shape}")

        # SigLIP-style two-stage patch embedding
        # Stage 1: Large patch embedding
        self.siglip_patch_embed = keras.Sequential([
            keras.layers.Conv2D(
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
            keras.layers.LayerNormalization(name='patch_embed_norm1'),
            keras.layers.Activation('gelu', name='patch_embed_gelu'),
            # Stage 2: Refine to final embedding dimension
            keras.layers.Conv2D(
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

        # CLS token
        self.cls_token = self.add_weight(
            shape=(1, 1, self.embed_dim),
            initializer='truncated_normal',
            trainable=True,
            name='cls_token'
        )

        # Positional embedding using existing layer
        self.pos_embed = PositionalEmbedding(
            max_seq_len=self.num_patches + 1,  # +1 for CLS token
            dim=self.embed_dim,
            dropout=0.0,  # Handle dropout separately
            name='pos_embed'
        )

        # Dropout layer for embeddings
        if self.dropout > 0.0:
            self.dropout_layer = keras.layers.Dropout(
                rate=self.dropout,
                name='embed_dropout'
            )

        # Transformer blocks using existing VisionTransformerLayer
        self.transformer_blocks = []
        for i in range(self.depth):
            block = VisionTransformerLayer(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                dropout_rate=self.dropout,
                name=f'transformer_block_{i}'
            )
            self.transformer_blocks.append(block)

        # Final layer normalization
        self.norm = keras.layers.LayerNormalization(
            name='final_norm'
        )

        super().build(input_shape)
        logger.info("SigLIPVisionTransformer build completed")

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Forward pass through vision transformer.

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

        # Final normalization
        x = self.norm(x, training=training)

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape.

        Args:
            input_shape: Shape tuple of the input.

        Returns:
            Output shape tuple.
        """
        batch_size = input_shape[0]
        return (batch_size, self.num_patches + 1, self.embed_dim)

    def get_config(self) -> dict:
        """Get layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
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
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

    def get_build_config(self) -> dict:
        """Get build configuration for serialization.

        Returns:
            Dictionary containing the build configuration.
        """
        return {"input_shape": self._build_input_shape}

    def build_from_config(self, config: dict) -> None:
        """Build from configuration for serialization.

        Args:
            config: Dictionary containing build configuration.
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

    def get_cls_token(self, features: keras.KerasTensor) -> keras.KerasTensor:
        """Extract CLS token from vision features.

        Args:
            features: Vision features from forward pass.

        Returns:
            CLS token of shape [batch_size, embed_dim].
        """
        return features[:, 0, :]  # First token is CLS

    def get_patch_tokens(self, features: keras.KerasTensor) -> keras.KerasTensor:
        """Extract patch tokens from vision features.

        Args:
            features: Vision features from forward pass.

        Returns:
            Patch tokens of shape [batch_size, num_patches, embed_dim].
        """
        return features[:, 1:, :]  # Skip CLS token

    def get_spatial_features(self, features: keras.KerasTensor) -> keras.KerasTensor:
        """Reshape patch tokens back to spatial format.

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

# ---------------------------------------------------------------------
