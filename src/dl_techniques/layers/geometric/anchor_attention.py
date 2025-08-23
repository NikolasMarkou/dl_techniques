import keras
from keras import ops
from typing import Optional, Any, Dict, Tuple, Union

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class AnchorAttention(keras.layers.Layer):
    """
    Hierarchical attention where anchor tokens have full self-attention.

    This layer implements a memory-efficient attention mechanism where:
    - Anchor tokens have full self-attention among themselves
    - Query tokens only cross-attend to anchor tokens (no self-attention)

    This creates a hierarchical attention pattern that can significantly reduce
    computational complexity for large sequences while maintaining representational
    power through the anchor tokens.

    Mathematical formulation:
        For standard mode: Attention(Q, K, V) = softmax(QK^T / √d_k)V
        For hierarchical mode: Query tokens only attend to anchor K, V

    Args:
        dim: Integer, input/output dimension of the attention layer. Must be positive
            and divisible by num_heads.
        num_heads: Integer, number of attention heads. Must be positive and divide dim.
            Defaults to 8.
        dropout: Float, dropout rate for attention weights. Must be between 0 and 1.
            Defaults to 0.0.
        use_bias: Boolean, whether to use bias in linear projections. Defaults to True.
        kernel_initializer: String or Initializer instance for kernel weights.
            Defaults to 'glorot_uniform'.
        bias_initializer: String or Initializer instance for bias vector.
            Defaults to 'zeros'.
        kernel_regularizer: Optional regularizer for kernel weights. Defaults to None.
        bias_regularizer: Optional regularizer for bias weights. Defaults to None.
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

    Raises:
        ValueError: If dim is not divisible by num_heads.
        ValueError: If dim or num_heads is not positive.
        ValueError: If dropout is not between 0 and 1.

    Example:
        ```python
        # Standard self-attention (all tokens are anchors)
        x = keras.random.normal((2, 100, 256))
        attn = AnchorAttention(dim=256, num_heads=8)
        output = attn(x)
        print(output.shape)  # (2, 100, 256)

        # Hierarchical attention (first 20 tokens are anchors)
        output = attn(x, num_anchor_tokens=20)
        print(output.shape)  # (2, 100, 256)
        # First 20 tokens attend to each other, last 80 only attend to first 20

        # With regularization
        attn = AnchorAttention(
            dim=512,
            num_heads=16,
            dropout=0.1,
            kernel_regularizer=keras.regularizers.L2(1e-4)
        )
        ```
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            dropout: float = 0.0,
            use_bias: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
            bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        if not (0.0 <= dropout <= 1.0):
            raise ValueError(f"dropout must be between 0 and 1, got {dropout}")

        # Store ALL configuration parameters
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout_rate = dropout
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # Scale factor for attention scores
        self.scale = 1.0 / ops.sqrt(float(self.head_dim))

        # CREATE all sub-layers in __init__ (following modern Keras 3 pattern)
        self.qkv_dense = keras.layers.Dense(
            self.dim * 3,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="qkv"
        )

        # Q projection for query tokens (used only in hierarchical mode)
        self.q_dense = keras.layers.Dense(
            self.dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="q_query"
        )

        # Output projection
        self.proj_dense = keras.layers.Dense(
            self.dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="proj"
        )

        # Dropout layer
        if self.dropout_rate > 0.0:
            self.dropout_layer = keras.layers.Dropout(self.dropout_rate, name="dropout")
        else:
            self.dropout_layer = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and all its sub-layers.

        CRITICAL: Explicitly build each sub-layer for robust serialization.

        Args:
            input_shape: Shape of the input tensor.
        """
        if len(input_shape) != 3:
            raise ValueError(f"Input must be 3D, got shape {input_shape}")

        if input_shape[-1] is None:
            raise ValueError("Last dimension of input must be defined")

        if input_shape[-1] != self.dim:
            raise ValueError(f"Last dimension of input ({input_shape[-1]}) "
                             f"must match dim ({self.dim})")

        # Build sub-layers explicitly for robust serialization
        self.qkv_dense.build(input_shape)
        self.q_dense.build(input_shape)
        self.proj_dense.build(input_shape)

        if self.dropout_layer is not None:
            # Dropout doesn't change shape, but we still build it
            attention_shape = input_shape[:-1] + (self.num_heads, input_shape[-1], input_shape[-1])
            self.dropout_layer.build(attention_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
            self,
            x: keras.KerasTensor,
            num_anchor_tokens: Optional[int] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply anchor-based attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, dim).
            num_anchor_tokens: Number of anchor tokens from the beginning.
                If None, standard self-attention is applied.
            training: Boolean indicating training mode.

        Returns:
            Output tensor with same shape as input.
        """
        if num_anchor_tokens is None:
            # Standard self-attention mode
            return self._standard_attention(x, training)
        else:
            # Hierarchical anchor-query attention mode
            return self._hierarchical_attention(x, num_anchor_tokens, training)

    def _standard_attention(
            self,
            x: keras.KerasTensor,
            training: Optional[bool]
    ) -> keras.KerasTensor:
        """Standard multi-head self-attention."""
        batch_size, seq_len, _ = ops.shape(x)

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

    def _hierarchical_attention(
            self,
            x: keras.KerasTensor,
            num_anchor_tokens: int,
            training: Optional[bool]
    ) -> keras.KerasTensor:
        """Hierarchical anchor-query attention."""
        batch_size, seq_len, _ = ops.shape(x)

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

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration for serialization.

        Returns:
            Dictionary containing ALL __init__ parameters for complete serialization.
        """
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "dropout": self.dropout_rate,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------
