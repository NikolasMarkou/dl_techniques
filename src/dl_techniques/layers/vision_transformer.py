"""
Vision Transformer (ViT) Layer Implementation

This module provides an improved Vision Transformer layer implementation that follows
the project's best practices and reuses existing components where possible.
"""

import keras
from keras import ops
from typing import Optional, Tuple, Union, Any, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .ffn.mlp import MLPBlock
from .norms.rms_norm import RMSNorm
from .multi_head_attention import MultiHeadAttention
from .patch_embedding import PatchEmbedding2D
from .positional_embedding import PositionalEmbedding

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class VisionTransformerLayer(keras.layers.Layer):
    """Vision Transformer (ViT) Layer.

    Implements a single transformer encoder layer for vision tasks, reusing
    existing project components where possible.

    Args:
        embed_dim: Dimension of input embeddings.
        num_heads: Number of attention heads.
        mlp_ratio: Ratio of MLP hidden dim to embedding dim.
        dropout_rate: Dropout rate for residual connections.
        attention_dropout_rate: Dropout rate for attention weights.
        kernel_initializer: Initializer for weight matrices.
        kernel_regularizer: Regularizer for weight matrices.
        use_bias: Whether to use bias in dense layers.
        norm_type: Type of normalization ('layer' or 'rms').
        **kwargs: Additional layer arguments.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        use_bias: bool = True,
        norm_type: str = "layer",
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.use_bias = use_bias
        self.norm_type = norm_type

        if norm_type not in ["layer", "rms"]:
            raise ValueError(f"norm_type must be 'layer' or 'rms', got {norm_type}")

        # Initialize to None, will be created in build()
        self.norm1 = None
        self.attn = None
        self.dropout1 = None
        self.norm2 = None
        self.mlp = None
        self._build_input_shape = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer's weights.

        Args:
            input_shape: Shape of the input tensor.
        """
        self._build_input_shape = input_shape

        # Create normalization layers
        if self.norm_type == "rms":
            self.norm1 = RMSNorm(name="norm1")
            self.norm2 = RMSNorm(name="norm2")
        else:
            self.norm1 = keras.layers.LayerNormalization(epsilon=1e-6, name="norm1")
            self.norm2 = keras.layers.LayerNormalization(epsilon=1e-6, name="norm2")

        # Create multi-head attention
        self.attn = MultiHeadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout_rate=self.attention_dropout_rate,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            use_bias=self.use_bias,
            name="attn"
        )

        # Create dropout layer
        self.dropout1 = keras.layers.Dropout(self.dropout_rate)

        # Create MLP block using existing MLP layer
        mlp_hidden_dim = int(self.embed_dim * self.mlp_ratio)
        self.mlp = MLPBlock(
            hidden_units=[mlp_hidden_dim, self.embed_dim],
            activation="gelu",
            dropout_rate=self.dropout_rate,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            use_bias=self.use_bias,
            name="mlp"
        )

        # Build sublayers
        self.norm1.build(input_shape)
        self.attn.build(input_shape)
        self.dropout1.build(input_shape)
        self.norm2.build(input_shape)
        self.mlp.build(input_shape)

        super().build(input_shape)

    def call(self, x: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Forward pass of the layer.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim).
            training: Whether in training mode.

        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim).
        """
        # Attention block with residual connection
        x1 = self.norm1(x, training=training)
        x1 = self.attn(x1, training=training)
        x1 = self.dropout1(x1, training=training)
        x = x + x1

        # MLP block with residual connection
        x2 = self.norm2(x, training=training)
        x2 = self.mlp(x2, training=training)
        x = x + x2

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input.

        Returns:
            Output shape.
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Get the layer configuration.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "dropout_rate": self.dropout_rate,
            "attention_dropout_rate": self.attention_dropout_rate,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "use_bias": self.use_bias,
            "norm_type": self.norm_type,
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get the build configuration.

        Returns:
            Dictionary containing the build configuration.
        """
        return {"input_shape": self._build_input_shape}

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build the layer from a config.

        Args:
            config: Dictionary containing the build configuration.
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class VisionTransformerBlock(keras.layers.Layer):
    """Complete Vision Transformer Block.

    Combines patch embedding, positional embedding, and transformer layers
    to create a complete vision transformer block.

    Args:
        patch_size: Size of patches to extract from input images.
        embed_dim: Dimension of patch embeddings.
        num_heads: Number of attention heads.
        num_layers: Number of transformer layers.
        mlp_ratio: Ratio of MLP hidden dim to embedding dim.
        dropout_rate: Dropout rate for residual connections.
        attention_dropout_rate: Dropout rate for attention weights.
        kernel_initializer: Initializer for weight matrices.
        kernel_regularizer: Regularizer for weight matrices.
        use_bias: Whether to use bias in dense layers.
        norm_type: Type of normalization ('layer' or 'rms').
        use_cls_token: Whether to add a classification token.
        **kwargs: Additional layer arguments.
    """

    def __init__(
        self,
        patch_size: Union[int, Tuple[int, int]] = 16,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        mlp_ratio: float = 4.0,
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        use_bias: bool = True,
        norm_type: str = "layer",
        use_cls_token: bool = True,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mlp_ratio = mlp_ratio
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.use_bias = use_bias
        self.norm_type = norm_type
        self.use_cls_token = use_cls_token

        # Initialize to None, will be created in build()
        self.patch_embed = None
        self.pos_embed = None
        self.cls_token = None
        self.transformer_layers = None
        self.norm = None
        self._build_input_shape = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer's weights.

        Args:
            input_shape: Shape of the input tensor.
        """
        self._build_input_shape = input_shape

        # Create patch embedding layer
        self.patch_embed = PatchEmbedding2D(
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="patch_embed"
        )

        # Calculate number of patches
        if isinstance(self.patch_size, int):
            patch_h = patch_w = self.patch_size
        else:
            patch_h, patch_w = self.patch_size

        img_h, img_w = input_shape[1], input_shape[2]
        num_patches = (img_h // patch_h) * (img_w // patch_w)

        # Add cls token if requested
        if self.use_cls_token:
            self.cls_token = self.add_weight(
                name="cls_token",
                shape=(1, 1, self.embed_dim),
                initializer="zeros",
                trainable=True
            )
            seq_len = num_patches + 1
        else:
            seq_len = num_patches

        # Create positional embedding
        self.pos_embed = PositionalEmbedding(
            sequence_length=seq_len,
            embed_dim=self.embed_dim,
            name="pos_embed"
        )

        # Create transformer layers
        self.transformer_layers = [
            VisionTransformerLayer(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                use_bias=self.use_bias,
                norm_type=self.norm_type,
                name=f"transformer_{i}"
            )
            for i in range(self.num_layers)
        ]

        # Create final normalization layer
        if self.norm_type == "rms":
            self.norm = RMSNorm(name="norm")
        else:
            self.norm = keras.layers.LayerNormalization(epsilon=1e-6, name="norm")

        # Build sublayers
        patch_embed_shape = (input_shape[0], num_patches, self.embed_dim)
        self.patch_embed.build(input_shape)
        self.pos_embed.build(patch_embed_shape)

        if self.use_cls_token:
            transformer_input_shape = (input_shape[0], seq_len, self.embed_dim)
        else:
            transformer_input_shape = patch_embed_shape

        for layer in self.transformer_layers:
            layer.build(transformer_input_shape)

        self.norm.build(transformer_input_shape)

        super().build(input_shape)

    def call(self, x: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Forward pass of the layer.

        Args:
            x: Input tensor of shape (batch_size, height, width, channels).
            training: Whether in training mode.

        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim).
        """
        # Convert image to patches
        x = self.patch_embed(x)  # (batch_size, num_patches, embed_dim)

        # Add cls token if requested
        if self.use_cls_token:
            batch_size = ops.shape(x)[0]
            cls_tokens = ops.broadcast_to(self.cls_token, (batch_size, 1, self.embed_dim))
            x = ops.concatenate([cls_tokens, x], axis=1)

        # Add positional embeddings
        x = self.pos_embed(x)

        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x, training=training)

        # Apply final normalization
        x = self.norm(x, training=training)

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input.

        Returns:
            Output shape.
        """
        if isinstance(self.patch_size, int):
            patch_h = patch_w = self.patch_size
        else:
            patch_h, patch_w = self.patch_size

        img_h, img_w = input_shape[1], input_shape[2]
        num_patches = (img_h // patch_h) * (img_w // patch_w)

        if self.use_cls_token:
            seq_len = num_patches + 1
        else:
            seq_len = num_patches

        return (input_shape[0], seq_len, self.embed_dim)

    def get_config(self) -> Dict[str, Any]:
        """Get the layer configuration.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "mlp_ratio": self.mlp_ratio,
            "dropout_rate": self.dropout_rate,
            "attention_dropout_rate": self.attention_dropout_rate,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "use_bias": self.use_bias,
            "norm_type": self.norm_type,
            "use_cls_token": self.use_cls_token,
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get the build configuration.

        Returns:
            Dictionary containing the build configuration.
        """
        return {"input_shape": self._build_input_shape}

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build the layer from a config.

        Args:
            config: Dictionary containing the build configuration.
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

# ---------------------------------------------------------------------
