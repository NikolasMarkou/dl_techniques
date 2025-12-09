"""
A hierarchical, memory-efficient anchor-based attention layer.

This layer provides a scalable alternative to standard self-attention by
creating an information bottleneck through a small, fixed set of "anchor"
tokens. It is designed to reduce the quadratic complexity of attention for
long sequences while preserving the model's ability to access global context.

Architecture:
    The architecture transforms the standard all-to-all attention graph
    into a two-tier, hub-and-spoke model. The input sequence is conceptually
    divided into two distinct groups:

    1.  **Anchor Tokens**: A small subset of tokens (e.g., the first `K`
        tokens of the sequence) that perform full, quadratic self-attention
        among themselves. These tokens are tasked with aggregating
        information and forming a compressed, global summary of the entire
        sequence context.

    2.  **Query Tokens**: The remaining tokens in the sequence. To save
        computation, these tokens do not attend to each other. Instead,
        they perform cross-attention *only* to the set of anchor tokens.
        Each query token can read information from the global summary
        created by the anchors but cannot interact directly with other
        query tokens.

    This design reduces the computational complexity from O(N^2) for a
    sequence of length N to a much more manageable O(K^2 + (N-K)*K), which
    is approximately linear in N when K is small and fixed.

References:
    - Beltagy, I., et al. (2020). "Longformer: The Long-Document Transformer".
    - Lee, J., et al. (2019). "Set Transformer".
"""

import keras
import numpy as np
from keras import ops, layers, initializers, regularizers
from typing import Optional, Any, Dict, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.activations import ProbabilityOutput

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class AnchorAttention(keras.layers.Layer):
    """
    Hierarchical attention mechanism where anchor tokens have full self-attention.

    This layer implements a memory-efficient attention mechanism that reduces computational
    complexity for large sequences while maintaining representational power through a
    hierarchical structure.

    Args:
        dim: Integer, input/output dimension of the attention layer. Must be positive
            and divisible by num_heads.
        num_heads: Integer, number of attention heads. Must be positive and divide dim
            evenly.
        head_dim: Optional integer, dimension of each attention head. If None,
            computed as dim // num_heads.
        dropout_rate: Float, dropout rate applied to attention weights. Must be in
            range [0, 1]. Defaults to 0.0.
        use_bias: Boolean, whether to use bias in linear projections. Defaults to True.
        probability_type: String, type of probability function to use for attention scores
            (e.g., 'softmax', 'sparsemax', 'adaptive'). Defaults to 'softmax'.
        probability_config: Optional dictionary containing configuration for the
            probability layer.
        kernel_initializer: String or Initializer instance for kernel weights.
            Defaults to 'glorot_uniform'.
        bias_initializer: String or Initializer instance for bias vectors.
            Defaults to 'zeros'.
        kernel_regularizer: Optional regularizer for kernel weights. Defaults to None.
        bias_regularizer: Optional regularizer for bias weights. Defaults to None.
        **kwargs: Additional keyword arguments for the Layer base class.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            head_dim: Optional[int] = None,
            dropout_rate: float = 0.0,
            use_bias: bool = True,
            probability_type: str = "softmax",
            probability_config: Optional[Dict[str, Any]] = None,
            kernel_initializer: Union[str, initializers.Initializer] = "glorot_uniform",
            bias_initializer: Union[str, initializers.Initializer] = "zeros",
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            bias_regularizer: Optional[regularizers.Regularizer] = None,
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
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else dim // num_heads
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.probability_type = probability_type
        self.probability_config = probability_config
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        # Scale factor for attention scores
        self.scale = 1.0 / np.sqrt(float(self.head_dim))

        # ---------------------------------------------------------------------
        # Create Sub-layers
        # ---------------------------------------------------------------------

        # Projections for Anchor Tokens (and Standard Mode)
        common_kwargs = {
            "use_bias": self.use_bias,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "bias_regularizer": self.bias_regularizer,
        }

        self.query_proj = keras.layers.Dense(
            self.num_heads * self.head_dim,
            name="query_proj",
            **common_kwargs
        )
        self.key_proj = keras.layers.Dense(
            self.num_heads * self.head_dim,
            name="key_proj",
            **common_kwargs
        )
        self.value_proj = keras.layers.Dense(
            self.num_heads * self.head_dim,
            name="value_proj",
            **common_kwargs
        )

        # Separate Query Projection for Query Tokens (Hierarchical Mode)
        # These tokens use a different projection to attend to anchors
        self.query_token_proj = keras.layers.Dense(
            self.dim,
            name="query_token_proj",
            **common_kwargs
        )

        # Output Projection
        self.output_proj = keras.layers.Dense(
            self.dim,
            name="output_proj",
            **common_kwargs
        )

        # Probability Output (Softmax replacement)
        self.score_activation = ProbabilityOutput(
            probability_type=self.probability_type,
            type_config=self.probability_config,
            name="score_activation"
        )

        # Dropout
        if self.dropout_rate > 0.0:
            self.dropout_layer = keras.layers.Dropout(
                self.dropout_rate,
                name="dropout"
            )
        else:
            self.dropout_layer = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and all its sub-layers.
        """
        if len(input_shape) != 3:
            raise ValueError(f"Input must be 3D, got shape {input_shape}")

        if input_shape[-1] is not None and input_shape[-1] != self.dim:
            raise ValueError(
                f"Last dimension of input ({input_shape[-1]}) "
                f"must match dim ({self.dim})"
            )

        # Build sub-layers explicitly
        self.query_proj.build(input_shape)
        self.key_proj.build(input_shape)
        self.value_proj.build(input_shape)
        self.query_token_proj.build(input_shape)
        self.output_proj.build(input_shape)

        # Build probability layer
        # Assuming typical attention score shape (Batch, Heads, Seq, Seq) for build context
        # The probability layer usually needs to know the axis
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        score_shape = (batch_size, self.num_heads, seq_len, seq_len)
        self.score_activation.build(score_shape)

        if self.dropout_layer is not None:
            self.dropout_layer.build(score_shape)

        super().build(input_shape)

    def call(
            self,
            x: keras.KerasTensor,
            num_anchor_tokens: Optional[int] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply anchor-based attention to input tensor.

        Args:
            x: Input tensor of shape (batch_size, seq_len, dim).
            num_anchor_tokens: Number of anchor tokens from the beginning.
                If None, applies standard self-attention to all tokens.
            training: Boolean indicating training mode for dropout.
        """
        if num_anchor_tokens is None:
            return self._standard_attention(x, training)
        else:
            return self._hierarchical_attention(x, num_anchor_tokens, training)

    def _standard_attention(
            self,
            x: keras.KerasTensor,
            training: Optional[bool]
    ) -> keras.KerasTensor:
        """Apply standard multi-head self-attention to all tokens."""
        batch_size, seq_len, _ = keras.ops.shape(x)

        # Projections
        q = self.query_proj(x)
        k = self.key_proj(x)
        v = self.value_proj(x)

        # Reshape to (batch_size, seq_len, num_heads, head_dim)
        q = ops.reshape(q, (batch_size, seq_len, self.num_heads, self.head_dim))
        k = ops.reshape(k, (batch_size, seq_len, self.num_heads, self.head_dim))
        v = ops.reshape(v, (batch_size, seq_len, self.num_heads, self.head_dim))

        # Transpose to (batch_size, num_heads, seq_len, head_dim)
        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))

        # Scaled dot-product attention
        # (batch, heads, seq, seq)
        scores = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scale

        # Apply Probability Output (Softmax/Sparsemax/etc)
        attn_weights = self.score_activation(scores)

        if self.dropout_layer is not None:
            attn_weights = self.dropout_layer(attn_weights, training=training)

        # Apply attention to values
        out = ops.matmul(attn_weights, v)

        # Reshape for output projection
        out = ops.transpose(out, (0, 2, 1, 3)) # (batch, seq, heads, head_dim)
        out = ops.reshape(out, (batch_size, seq_len, self.dim))

        return self.output_proj(out)

    def _hierarchical_attention(
            self,
            x: keras.KerasTensor,
            num_anchor_tokens: int,
            training: Optional[bool]
    ) -> keras.KerasTensor:
        """Apply hierarchical anchor-query attention pattern."""
        batch_size, seq_len, _ = ops.shape(x)

        # Fallback to standard if all tokens are anchors
        if num_anchor_tokens >= seq_len:
            return self._standard_attention(x, training)

        # Split input
        anchor_tokens = x[:, :num_anchor_tokens, :]
        query_tokens = x[:, num_anchor_tokens:, :]
        num_query_tokens = seq_len - num_anchor_tokens

        # 1. Process Anchor Tokens (Full Q, K, V)
        anchor_q = self.query_proj(anchor_tokens)
        anchor_k = self.key_proj(anchor_tokens)
        anchor_v = self.value_proj(anchor_tokens)

        # Reshape anchors
        anchor_q = ops.reshape(anchor_q, (batch_size, num_anchor_tokens, self.num_heads, self.head_dim))
        anchor_k = ops.reshape(anchor_k, (batch_size, num_anchor_tokens, self.num_heads, self.head_dim))
        anchor_v = ops.reshape(anchor_v, (batch_size, num_anchor_tokens, self.num_heads, self.head_dim))

        # Transpose anchors (batch, heads, num_anchors, head_dim)
        anchor_q = ops.transpose(anchor_q, (0, 2, 1, 3))
        anchor_k = ops.transpose(anchor_k, (0, 2, 1, 3))
        anchor_v = ops.transpose(anchor_v, (0, 2, 1, 3))

        # 2. Process Query Tokens (Only Q, using separate projection)
        query_q = self.query_token_proj(query_tokens)
        query_q = ops.reshape(query_q, (batch_size, num_query_tokens, self.num_heads, self.head_dim))
        query_q = ops.transpose(query_q, (0, 2, 1, 3)) # (batch, heads, num_queries, head_dim)

        # 3. Combine Q vectors: Anchors first, then Queries
        # Shape: (batch, heads, seq_len, head_dim)
        combined_q = ops.concatenate([anchor_q, query_q], axis=2)

        # 4. Attention Computation
        # All tokens attend ONLY to anchor tokens (Keys and Values)
        # combined_q: (batch, heads, seq_len, head_dim)
        # anchor_k^T: (batch, heads, head_dim, num_anchors)
        # scores: (batch, heads, seq_len, num_anchors)
        scores = ops.matmul(combined_q, ops.transpose(anchor_k, (0, 1, 3, 2))) * self.scale

        # Apply Probability Output
        attn_weights = self.score_activation(scores)

        if self.dropout_layer is not None:
            attn_weights = self.dropout_layer(attn_weights, training=training)

        # Apply attention to anchor values
        # attn_weights: (batch, heads, seq_len, num_anchors)
        # anchor_v: (batch, heads, num_anchors, head_dim)
        # out: (batch, heads, seq_len, head_dim)
        out = ops.matmul(attn_weights, anchor_v)

        # Reshape for output projection
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (batch_size, seq_len, self.dim))

        return self.output_proj(out)

    def compute_output_shape(
            self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "dropout_rate": self.dropout_rate,
            "use_bias": self.use_bias,
            "probability_type": self.probability_type,
            "probability_config": self.probability_config,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------
