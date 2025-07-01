"""Perceiver-style cross-attention mechanism."""

import keras
from keras import ops
from typing import Optional, Any, Dict, Tuple

from dl_techniques.utils.logger import logger


@keras.saving.register_keras_serializable()
class PerceiverAttention(keras.layers.Layer):
    """Cross-attention mechanism from the Perceiver architecture.

    This layer implements cross-attention where queries and key-value pairs
    come from different sources, similar to the Perceiver architecture.
    This is useful for set-to-set transformations, cross-modal processing,
    and latent space learning.

    The key difference from standard self-attention is that:
    - Queries come from one input (query_input)
    - Keys and Values come from another input (kv_input)
    - This enables flexible cross-modal or cross-domain attention

    Args:
        dim: Integer, input/output dimension of the attention layer.
        num_heads: Integer, number of attention heads. Defaults to 8.
        dropout: Float, dropout rate for attention weights. Defaults to 0.0.
        use_bias: Boolean, whether to use bias in linear projections. Defaults to True.
        kernel_initializer: Initializer for the kernel weights.
        bias_initializer: Initializer for the bias vector.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        Two inputs:
        - query_input: 3D tensor with shape `(batch_size, query_seq_len, dim)`
        - kv_input: 3D tensor with shape `(batch_size, kv_seq_len, dim)`

    Output shape:
        3D tensor with shape: `(batch_size, query_seq_len, dim)`

    Call arguments:
        query_input: Query tensor of shape (batch_size, query_seq_len, dim).
        kv_input: Key-Value tensor of shape (batch_size, kv_seq_len, dim).
        training: Boolean indicating whether the layer should behave in training
            mode (applying dropout) or inference mode.

    Returns:
        Output tensor with same shape as query_input.

    Example:
        >>> # Cross-attention between different modalities
        >>> visual_features = keras.random.normal((2, 196, 256))  # ViT patches
        >>> text_features = keras.random.normal((2, 77, 256))    # Text tokens
        >>>
        >>> perceiver_attn = PerceiverAttention(dim=256, num_heads=8)
        >>>
        >>> # Text attending to visual features
        >>> text_to_visual = perceiver_attn(text_features, visual_features)
        >>> print(text_to_visual.shape)  # (2, 77, 256)
        >>>
        >>> # Visual attending to text features
        >>> visual_to_text = perceiver_attn(visual_features, text_features)
        >>> print(visual_to_text.shape)  # (2, 196, 256)
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
        self.q_dense = None
        self.kv_dense = None
        self.proj_dense = None
        self.dropout_layer = None

    def build(self, input_shape):
        """Build the layer weights."""
        # Handle different input formats
        if isinstance(input_shape, list):
            # Two separate inputs
            if len(input_shape) != 2:
                raise ValueError(f"Expected 2 inputs, got {len(input_shape)}")
            query_shape, kv_shape = input_shape
            self._build_input_shape = input_shape
        else:
            # Single input shape (will be used for both query and kv)
            query_shape = kv_shape = input_shape
            self._build_input_shape = input_shape

        # Validate shapes
        if len(query_shape) != 3:
            raise ValueError(f"Query input must be 3D, got shape {query_shape}")
        if len(kv_shape) != 3:
            raise ValueError(f"KV input must be 3D, got shape {kv_shape}")

        if query_shape[-1] != self.dim:
            raise ValueError(f"Query last dimension ({query_shape[-1]}) "
                             f"must match dim ({self.dim})")
        if kv_shape[-1] != self.dim:
            raise ValueError(f"KV last dimension ({kv_shape[-1]}) "
                             f"must match dim ({self.dim})")

        # Query projection
        self.q_dense = keras.layers.Dense(
            self.dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            name="q"
        )

        # Key-Value projection
        self.kv_dense = keras.layers.Dense(
            self.dim * 2,  # For both key and value
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            name="kv"
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

    def call(self, query_input, kv_input=None, training=None):
        """Apply Perceiver cross-attention.

        Args:
            query_input: Query tensor of shape (batch_size, query_seq_len, dim).
            kv_input: Key-Value tensor of shape (batch_size, kv_seq_len, dim).
                If None, uses query_input for both (self-attention mode).
            training: Boolean indicating training mode.

        Returns:
            Output tensor with same shape as query_input.
        """
        if kv_input is None:
            kv_input = query_input

        batch_size = ops.shape(query_input)[0]
        query_seq_len = ops.shape(query_input)[1]
        kv_seq_len = ops.shape(kv_input)[1]

        # Project queries
        q = self.q_dense(query_input)  # (batch_size, query_seq_len, dim)
        q = ops.reshape(q, (batch_size, query_seq_len, self.num_heads, self.head_dim))
        q = ops.transpose(q, (0, 2, 1, 3))  # (batch_size, num_heads, query_seq_len, head_dim)

        # Project keys and values
        kv = self.kv_dense(kv_input)  # (batch_size, kv_seq_len, dim * 2)
        kv = ops.reshape(kv, (batch_size, kv_seq_len, 2, self.num_heads, self.head_dim))
        kv = ops.transpose(kv, (2, 0, 3, 1, 4))  # (2, batch_size, num_heads, kv_seq_len, head_dim)

        k, v = kv[0], kv[1]

        # Scaled dot-product attention
        scores = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scale
        # scores shape: (batch_size, num_heads, query_seq_len, kv_seq_len)

        attn_weights = ops.softmax(scores, axis=-1)

        if self.dropout_layer is not None:
            attn_weights = self.dropout_layer(attn_weights, training=training)

        # Apply attention to values
        out = ops.matmul(attn_weights, v)  # (batch_size, num_heads, query_seq_len, head_dim)

        # Reshape and project
        out = ops.transpose(out, (0, 2, 1, 3))  # (batch_size, query_seq_len, num_heads, head_dim)
        out = ops.reshape(out, (batch_size, query_seq_len, self.dim))
        out = self.proj_dense(out)

        return out

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the layer."""
        if isinstance(input_shape, list):
            return input_shape[0]  # Same as query input shape
        else:
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