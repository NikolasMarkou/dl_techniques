"""Anchor-based hierarchical attention mechanism."""

import keras
from keras import ops
from typing import Optional, Any, Dict, Tuple

from dl_techniques.utils.logger import logger


@keras.saving.register_keras_serializable()
class AnchorAttention(keras.layers.Layer):
    """Hierarchical attention where anchor tokens have full self-attention.

    This layer implements a memory-efficient attention mechanism where:
    - Anchor tokens have full self-attention among themselves
    - Query tokens only cross-attend to anchor tokens (no self-attention)

    This creates a hierarchical attention pattern that can significantly reduce
    computational complexity for large sequences while maintaining representational
    power through the anchor tokens.

    Args:
        dim: Integer, input/output dimension of the attention layer.
        num_heads: Integer, number of attention heads. Defaults to 8.
        dropout: Float, dropout rate for attention weights. Defaults to 0.0.
        use_bias: Boolean, whether to use bias in linear projections. Defaults to True.
        kernel_initializer: Initializer for the kernel weights.
        bias_initializer: Initializer for the bias vector.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        3D tensor with shape: `(batch_size, sequence_length, dim)`

    Output shape:
        3D tensor with shape: `(batch_size, sequence_length, dim)`

    Call arguments:
        x: Input tensor of shape (batch_size, sequence_length, dim).
        num_anchor_tokens: Optional integer specifying how many tokens from the
            beginning of the sequence are anchor tokens. If None, all tokens
            are treated as anchors (standard self-attention). If provided,
            the first num_anchor_tokens will be anchors with full self-attention,
            and remaining tokens will be queries that only attend to anchors.
        training: Boolean indicating whether the layer should behave in training
            mode (applying dropout) or inference mode.

    Returns:
        Output tensor with same shape as input.

    Example:
        >>> # Standard self-attention (all tokens are anchors)
        >>> x = keras.random.normal((2, 100, 256))
        >>> attn = AnchorAttention(dim=256, num_heads=8)
        >>> output = attn(x)
        >>> print(output.shape)  # (2, 100, 256)

        >>> # Hierarchical attention (first 20 tokens are anchors)
        >>> output = attn(x, num_anchor_tokens=20)
        >>> print(output.shape)  # (2, 100, 256)
        >>> # First 20 tokens attend to each other, last 80 only attend to first 20
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            dropout: float = 0.0,
            use_bias: bool = True,
            kernel_initializer: str = "glorot_uniform",
            bias_initializer: str = "zeros",
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout_rate = dropout
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)

        # Scale factor for attention scores
        self.scale = 1.0 / ops.sqrt(float(self.head_dim))

        # Store build information
        self._build_input_shape = None

        # Will be created in build()
        self.qkv_dense = None
        self.q_dense = None  # For query tokens when using anchors
        self.proj_dense = None
        self.dropout_layer = None

    def build(self, input_shape):
        """Build the layer weights."""
        self._build_input_shape = input_shape

        if len(input_shape) != 3:
            raise ValueError(f"Input must be 3D, got shape {input_shape}")

        if input_shape[-1] != self.dim:
            raise ValueError(f"Last dimension of input ({input_shape[-1]}) "
                             f"must match dim ({self.dim})")

        # QKV projection for anchor tokens (or all tokens in standard mode)
        self.qkv_dense = keras.layers.Dense(
            self.dim * 3,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            name="qkv"
        )

        # Q projection for query tokens (used only in hierarchical mode)
        self.q_dense = keras.layers.Dense(
            self.dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            name="q_query"
        )

        # Output projection
        self.proj_dense = keras.layers.Dense(
            self.dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            name="proj"
        )

        # Dropout layer
        if self.dropout_rate > 0.0:
            self.dropout_layer = keras.layers.Dropout(self.dropout_rate)

        super().build(input_shape)

    def call(self, x, num_anchor_tokens=None, training=None):
        """Apply anchor-based attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, dim).
            num_anchor_tokens: Number of anchor tokens from the beginning.
                If None, standard self-attention is applied.
            training: Boolean indicating training mode.

        Returns:
            Output tensor with same shape as input.
        """
        batch_size, seq_len, _ = x.shape

        if num_anchor_tokens is None:
            # Standard self-attention mode
            return self._standard_attention(x, training)
        else:
            # Hierarchical anchor-query attention mode
            return self._hierarchical_attention(x, num_anchor_tokens, training)

    def _standard_attention(self, x, training):
        """Standard multi-head self-attention."""
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V
        qkv = self.qkv_dense(x)  # (batch_size, seq_len, dim * 3)
        qkv = ops.reshape(qkv, (batch_size, seq_len, 3, self.num_heads, self.head_dim))
        qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))  # (3, batch_size, num_heads, seq_len, head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        scores = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scale
        attn_weights = ops.softmax(scores, axis=-1)

        if self.dropout_layer is not None:
            attn_weights = self.dropout_layer(attn_weights, training=training)

        # Apply attention to values
        out = ops.matmul(attn_weights, v)  # (batch_size, num_heads, seq_len, head_dim)

        # Reshape and project
        out = ops.transpose(out, (0, 2, 1, 3))  # (batch_size, seq_len, num_heads, head_dim)
        out = ops.reshape(out, (batch_size, seq_len, self.dim))
        out = self.proj_dense(out)

        return out

    def _hierarchical_attention(self, x, num_anchor_tokens, training):
        """Hierarchical anchor-query attention."""
        batch_size, seq_len, _ = x.shape

        if num_anchor_tokens >= seq_len:
            # All tokens are anchors, fallback to standard attention
            return self._standard_attention(x, training)

        # Split into anchor and query tokens
        anchor_tokens = x[:, :num_anchor_tokens, :]  # (batch_size, num_anchor_tokens, dim)
        query_tokens = x[:, num_anchor_tokens:, :]  # (batch_size, num_query_tokens, dim)
        num_query_tokens = seq_len - num_anchor_tokens

        # Process anchor tokens with full self-attention
        anchor_qkv = self.qkv_dense(anchor_tokens)  # (batch_size, num_anchor_tokens, dim * 3)
        anchor_qkv = ops.reshape(anchor_qkv, (batch_size, num_anchor_tokens, 3, self.num_heads, self.head_dim))
        anchor_qkv = ops.transpose(anchor_qkv,
                                   (2, 0, 3, 1, 4))  # (3, batch_size, num_heads, num_anchor_tokens, head_dim)

        anchor_q, anchor_k, anchor_v = anchor_qkv[0], anchor_qkv[1], anchor_qkv[2]

        # Process query tokens (only Q projection)
        query_q = self.q_dense(query_tokens)  # (batch_size, num_query_tokens, dim)
        query_q = ops.reshape(query_q, (batch_size, num_query_tokens, self.num_heads, self.head_dim))
        query_q = ops.transpose(query_q, (0, 2, 1, 3))  # (batch_size, num_heads, num_query_tokens, head_dim)

        # Combine Q vectors: anchors + queries
        combined_q = ops.concatenate([anchor_q, query_q], axis=2)  # (batch_size, num_heads, seq_len, head_dim)

        # Attention scores: all tokens attend to anchor tokens only
        scores = ops.matmul(combined_q, ops.transpose(anchor_k, (0, 1, 3, 2))) * self.scale
        # scores shape: (batch_size, num_heads, seq_len, num_anchor_tokens)

        attn_weights = ops.softmax(scores, axis=-1)

        if self.dropout_layer is not None:
            attn_weights = self.dropout_layer(attn_weights, training=training)

        # Apply attention to anchor values
        out = ops.matmul(attn_weights, anchor_v)  # (batch_size, num_heads, seq_len, head_dim)

        # Reshape and project
        out = ops.transpose(out, (0, 2, 1, 3))  # (batch_size, seq_len, num_heads, head_dim)
        out = ops.reshape(out, (batch_size, seq_len, self.dim))
        out = self.proj_dense(out)

        return out

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the layer."""
        return input_shape

    def get_config(self):
        """Returns the layer configuration."""
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "dropout": self.dropout_rate,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
        })
        return config

    def get_build_config(self):
        """Get build configuration."""
        return {"input_shape": self._build_input_shape}

    def build_from_config(self, config):
        """Build from configuration."""
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])