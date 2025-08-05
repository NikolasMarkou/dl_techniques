"""
Encoder-only Mask Transformer (EoMT) Layer Implementation

Based on: "Your ViT is Secretly an Image Segmentation Model" by Kerssies et al.
"""

import keras
from keras import ops
from typing import Optional, Any, Tuple

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class EoMTLayer(keras.layers.Layer):
    """Encoder-only Mask Transformer Layer.

    This layer processes both patch tokens and learnable queries together using
    the standard Vision Transformer self-attention mechanism. It supports masked
    attention during training with annealing.

    Args:
        embed_dim: Integer, embedding dimension.
        num_heads: Integer, number of attention heads.
        mlp_ratio: Float, ratio of MLP hidden dim to embedding dim.
        dropout: Float, dropout rate.
        attention_dropout: Float, attention dropout rate.
        use_layer_norm: Boolean, whether to use layer normalization.
        activation: String, activation function for MLP.
        use_masked_attention: Boolean, whether to use masked attention.
        mask_probability: Float, probability of applying mask (for annealing).
        **kwargs: Additional keyword arguments for the Layer base class.
    """

    def __init__(
            self,
            embed_dim: int,
            num_heads: int = 8,
            mlp_ratio: float = 4.0,
            dropout: float = 0.0,
            attention_dropout: float = 0.0,
            use_layer_norm: bool = True,
            activation: str = "gelu",
            use_masked_attention: bool = False,
            mask_probability: float = 1.0,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.use_layer_norm = use_layer_norm
        self.activation = activation
        self.use_masked_attention = use_masked_attention
        self.mask_probability = mask_probability

        # Validate inputs
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Initialize components to None
        self.norm1 = None
        self.norm2 = None
        self.qkv = None
        self.proj = None
        self.mlp = None
        self.dropout_layer = None
        self.attention_dropout_layer = None

        # Build input shape storage
        self._build_input_shape = None

    def build(self, input_shape):
        """Build the layer."""
        self._build_input_shape = input_shape

        # Normalization layers
        if self.use_layer_norm:
            self.norm1 = keras.layers.LayerNormalization(
                epsilon=1e-6,
                name="norm1"
            )
            self.norm2 = keras.layers.LayerNormalization(
                epsilon=1e-6,
                name="norm2"
            )

        # Attention layers
        self.qkv = keras.layers.Dense(
            self.embed_dim * 3,
            use_bias=False,
            name="qkv"
        )

        self.proj = keras.layers.Dense(
            self.embed_dim,
            name="proj"
        )

        # MLP layers
        mlp_hidden_dim = int(self.embed_dim * self.mlp_ratio)
        self.mlp = keras.Sequential([
            keras.layers.Dense(
                mlp_hidden_dim,
                activation=self.activation,
                name="mlp_fc1"
            ),
            keras.layers.Dropout(self.dropout, name="mlp_dropout1"),
            keras.layers.Dense(
                self.embed_dim,
                name="mlp_fc2"
            ),
            keras.layers.Dropout(self.dropout, name="mlp_dropout2")
        ], name="mlp")

        # Dropout layers
        if self.dropout > 0:
            self.dropout_layer = keras.layers.Dropout(self.dropout)

        if self.attention_dropout > 0:
            self.attention_dropout_layer = keras.layers.Dropout(self.attention_dropout)

        super().build(input_shape)

    def _apply_masked_attention(
            self,
            attn_weights: keras.KerasTensor,
            mask: Optional[keras.KerasTensor],
            num_patches: int,
            num_queries: int
    ) -> keras.KerasTensor:
        """Apply masked attention to query-to-patch interactions.

        Args:
            attn_weights: Attention weights [batch, heads, seq_len, seq_len]
            mask: Mask tensor [batch, num_queries, H, W]
            num_patches: Number of patch tokens
            num_queries: Number of query tokens

        Returns:
            Masked attention weights
        """
        if mask is None or not self.use_masked_attention:
            return attn_weights

        # Apply mask probabilistically during training
        if self.mask_probability < 1.0:
            should_mask = ops.random.uniform([]) < self.mask_probability
            if not should_mask:
                return attn_weights

        batch_size = ops.shape(attn_weights)[0]

        # Flatten mask for patch matching
        mask_flat = ops.reshape(mask, [batch_size, num_queries, -1])  # [B, Q, H*W]

        # Create attention mask for query-to-patch interactions
        # attn_weights shape: [B, H, seq_len, seq_len]
        # We need to mask the query-to-patch part: [B, H, Q, P]

        # Extract query-to-patch attention
        query_to_patch_attn = attn_weights[:, :, num_patches:, :num_patches]  # [B, H, Q, P]

        # Expand mask to match attention heads
        mask_expanded = ops.expand_dims(mask_flat, axis=1)  # [B, 1, Q, P]
        mask_expanded = ops.tile(mask_expanded, [1, self.num_heads, 1, 1])  # [B, H, Q, P]

        # Apply mask (set to large negative value where mask is 0)
        masked_attn = ops.where(
            mask_expanded > 0.5,
            query_to_patch_attn,
            ops.full_like(query_to_patch_attn, -1e9)
        )

        # Update attention weights
        attn_weights = ops.concatenate([
            attn_weights[:, :, :num_patches, :],  # patch-to-all unchanged
            ops.concatenate([
                masked_attn,  # query-to-patch masked
                attn_weights[:, :, num_patches:, num_patches:]  # query-to-query unchanged
            ], axis=-1)
        ], axis=-2)

        return attn_weights

    def call(
            self,
            inputs: keras.KerasTensor,
            mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass.

        Args:
            inputs: Input tensor containing both patch tokens and queries
                   [batch, seq_len, embed_dim] where seq_len = num_patches + num_queries
            mask: Optional mask tensor [batch, num_queries, H, W]
            training: Boolean indicating training mode

        Returns:
            Output tensor with same shape as inputs
        """
        batch_size, seq_len, embed_dim = ops.shape(inputs)[0], ops.shape(inputs)[1], ops.shape(inputs)[2]

        # Determine number of patches and queries
        # This assumes queries are appended after patches
        if mask is not None:
            num_queries = ops.shape(mask)[1]
            num_patches = seq_len - num_queries
        else:
            # Default assumption: equal split or no queries
            num_patches = seq_len
            num_queries = 0

        # Self-attention
        shortcut = inputs

        if self.use_layer_norm:
            x = self.norm1(inputs)
        else:
            x = inputs

        # Multi-head self-attention
        qkv = self.qkv(x)
        qkv = ops.reshape(qkv, [batch_size, seq_len, 3, self.num_heads, self.head_dim])
        qkv = ops.transpose(qkv, [2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]  # [batch, heads, seq_len, head_dim]

        # Compute attention
        attn_weights = ops.matmul(q, ops.transpose(k, [0, 1, 3, 2])) * self.scale

        # Apply masked attention if needed
        if self.use_masked_attention and mask is not None and training:
            attn_weights = self._apply_masked_attention(
                attn_weights, mask, num_patches, num_queries
            )

        attn_weights = ops.softmax(attn_weights, axis=-1)

        if self.attention_dropout_layer is not None:
            attn_weights = self.attention_dropout_layer(attn_weights, training=training)

        # Apply attention to values
        attn_output = ops.matmul(attn_weights, v)  # [batch, heads, seq_len, head_dim]
        attn_output = ops.transpose(attn_output, [0, 2, 1, 3])  # [batch, seq_len, heads, head_dim]
        attn_output = ops.reshape(attn_output, [batch_size, seq_len, embed_dim])

        # Project back
        attn_output = self.proj(attn_output)

        if self.dropout_layer is not None:
            attn_output = self.dropout_layer(attn_output, training=training)

        # First residual connection
        x = shortcut + attn_output

        # MLP
        shortcut = x

        if self.use_layer_norm:
            x = self.norm2(x)

        x = self.mlp(x, training=training)

        # Second residual connection
        x = shortcut + x

        return x

    def compute_output_shape(self, input_shape):
        """Compute output shape."""
        return input_shape

    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "dropout": self.dropout,
            "attention_dropout": self.attention_dropout,
            "use_layer_norm": self.use_layer_norm,
            "activation": self.activation,
            "use_masked_attention": self.use_masked_attention,
            "mask_probability": self.mask_probability,
        })
        return config

    def get_build_config(self):
        """Get build configuration."""
        return {"input_shape": self._build_input_shape}

    def build_from_config(self, config):
        """Build from configuration."""
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MaskModule(keras.layers.Layer):
    """Mask prediction module for EoMT.

    This module predicts class logits and mask logits for each query token.

    Args:
        num_classes: Integer, number of classes.
        hidden_dim: Integer, hidden dimension for mask MLP.
        mask_dim: Integer, dimension of mask embeddings.
        **kwargs: Additional keyword arguments for the Layer base class.
    """

    def __init__(
            self,
            num_classes: int,
            hidden_dim: int = 256,
            mask_dim: int = 256,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.mask_dim = mask_dim

        # Initialize components to None
        self.class_head = None
        self.mask_mlp = None

        self._build_input_shape = None

    def build(self, input_shape):
        """Build the layer."""
        self._build_input_shape = input_shape

        # Class prediction head
        self.class_head = keras.layers.Dense(
            self.num_classes,
            name="class_head"
        )

        # Mask embedding MLP
        self.mask_mlp = keras.Sequential([
            keras.layers.Dense(
                self.hidden_dim,
                activation="relu",
                name="mask_mlp_1"
            ),
            keras.layers.Dense(
                self.hidden_dim,
                activation="relu",
                name="mask_mlp_2"
            ),
            keras.layers.Dense(
                self.mask_dim,
                name="mask_mlp_3"
            )
        ], name="mask_mlp")

        super().build(input_shape)

    def call(
            self,
            query_tokens: keras.KerasTensor,
            pixel_features: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Forward pass.

        Args:
            query_tokens: Query tokens [batch, num_queries, embed_dim]
            pixel_features: Pixel features [batch, H, W, embed_dim]
            training: Boolean indicating training mode

        Returns:
            Tuple of (class_logits, mask_logits)
            - class_logits: [batch, num_queries, num_classes]
            - mask_logits: [batch, num_queries, H, W]
        """
        # Predict class logits
        class_logits = self.class_head(query_tokens)

        # Predict mask embeddings
        mask_embeddings = self.mask_mlp(query_tokens)  # [batch, num_queries, mask_dim]

        # Compute mask logits via dot product
        batch_size, height, width, embed_dim = (
            ops.shape(pixel_features)[0],
            ops.shape(pixel_features)[1],
            ops.shape(pixel_features)[2],
            ops.shape(pixel_features)[3]
        )

        # Reshape pixel features for dot product
        pixel_features_flat = ops.reshape(
            pixel_features, [batch_size, height * width, embed_dim]
        )

        # Compute dot product: [batch, num_queries, mask_dim] @ [batch, mask_dim, H*W]
        mask_embeddings_expanded = ops.expand_dims(mask_embeddings, axis=2)  # [batch, num_queries, 1, mask_dim]
        pixel_features_expanded = ops.expand_dims(pixel_features_flat, axis=1)  # [batch, 1, H*W, embed_dim]

        # Compute dot product
        mask_logits = ops.sum(
            mask_embeddings_expanded * pixel_features_expanded, axis=-1
        )  # [batch, num_queries, H*W]

        # Reshape back to spatial dimensions
        mask_logits = ops.reshape(
            mask_logits, [batch_size, ops.shape(query_tokens)[1], height, width]
        )

        return class_logits, mask_logits

    def compute_output_shape(self, input_shape):
        """Compute output shape."""
        query_shape, pixel_shape = input_shape
        batch_size = query_shape[0]
        num_queries = query_shape[1]
        height, width = pixel_shape[1], pixel_shape[2]

        class_shape = (batch_size, num_queries, self.num_classes)
        mask_shape = (batch_size, num_queries, height, width)

        return class_shape, mask_shape

    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "hidden_dim": self.hidden_dim,
            "mask_dim": self.mask_dim,
        })
        return config

    def get_build_config(self):
        """Get build configuration."""
        return {"input_shape": self._build_input_shape}

    def build_from_config(self, config):
        """Build from configuration."""
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

# ---------------------------------------------------------------------
