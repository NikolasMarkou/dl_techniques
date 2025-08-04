"""
Grouped Query Attention (GQA) Implementation with Rotary Position Embeddings

This module implements Grouped Query Attention, an efficient attention mechanism that
reduces the computational and memory requirements of multi-head attention by sharing
key and value projections across multiple query heads while maintaining model quality.

Mathematical Formulation:
    Standard Multi-Head Attention uses:
    - n_head query projections
    - n_head key projections  
    - n_head value projections

    Grouped Query Attention uses:
    - n_head query projections
    - n_kv_head key projections (where n_kv_head < n_head)
    - n_kv_head value projections

    The key insight is that n_head % n_kv_head == 0, creating groups where:
    group_size = n_head // n_kv_head

    Each K,V pair is shared across group_size query heads by repeating the K,V
    tensors along the head dimension.

Computational Benefits:
    - Reduces memory usage for K,V caches in autoregressive generation
    - Maintains most of the representational power of full multi-head attention
    - Particularly effective in large language models where KV cache dominates memory
    - Enables longer sequence lengths with same memory budget

Architecture Details:
    1. Project input to Q (n_head), K (n_kv_head), V (n_kv_head)
    2. Apply rotary position embeddings to Q and K
    3. Repeat K,V tensors to match Q head count
    4. Perform standard scaled dot-product attention
    5. Project output back to model dimension

References:
    - Ainslie, J., et al. (2023). "GQA: Training Generalized Multi-Query Transformer 
      Models from Multi-Head Checkpoints." https://arxiv.org/abs/2305.13245

    - Shazeer, N. (2019). "Fast Transformer Decoding: One Write-Head is All You Need."
      https://arxiv.org/abs/1911.02150 (Multi-Query Attention predecessor)

    - Su, J., et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position 
      Embedding." https://arxiv.org/abs/2104.09864 (RoPE integration)
"""

import keras
from keras import layers, ops
from typing import Optional, Any, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from ..rope import RotaryPositionEmbedding

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class GroupedQueryAttention(layers.Layer):
    """
    Grouped Query Attention layer with rotary position embeddings.

    This layer implements an efficient attention mechanism that reduces computational
    and memory costs by sharing key-value projections across multiple query heads
    while maintaining most of the representational power of full multi-head attention.

    Args:
        d_model: int
            Model dimension (embedding size).
        n_head: int
            Number of attention heads for queries.
        n_kv_head: int
            Number of key-value heads. Must divide n_head evenly.
        max_seq_len: int
            Maximum sequence length for positional embeddings.
        attention_dropout: float, default=0.0
            Dropout rate applied to attention weights.
        rope_percentage: float, default=1.0
            Fraction of head dimensions to apply rotary embeddings to.
        use_bias: bool, default=False
            Whether to use bias in linear projections.
        **kwargs: Additional keyword arguments for the Layer parent class.

    Input shape:
        3D tensor with shape: (batch_size, sequence_length, d_model)

    Output shape:
        3D tensor with shape: (batch_size, sequence_length, d_model)

    Raises:
        ValueError: If d_model is not divisible by n_head, or if n_head is not 
                   divisible by n_kv_head.

    Usage Examples:
        Basic usage:
        >>> gqa = GroupedQueryAttention(
        ...     d_model=512,
        ...     n_head=8,
        ...     n_kv_head=2,  # 4 query heads per key/value head
        ...     max_seq_len=2048
        ... )
        >>> output = gqa(inputs)

        In a transformer decoder:
        >>> def decoder_layer(x):
        ...     # Self-attention with GQA
        ...     attn_out = GroupedQueryAttention(...)(x)
        ...     x = x + attn_out
        ...
        ...     # Feed-forward network
        ...     ffn_out = FeedForward(...)(LayerNorm()(x))
        ...     return x + ffn_out
    """

    def __init__(
            self,
            d_model: int,
            n_head: int,
            n_kv_head: int,
            max_seq_len: int,
            attention_dropout: float = 0.0,
            rope_percentage: float = 1.0,
            use_bias: bool = False,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        self._validate_inputs(d_model, n_head, n_kv_head, attention_dropout, rope_percentage)

        self.d_model = d_model
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.max_seq_len = max_seq_len
        self.attention_dropout = attention_dropout
        self.rope_percentage = rope_percentage
        self.use_bias = use_bias

        # Derived parameters
        self.head_dim = self.d_model // self.n_head
        self.n_group = self.n_head // self.n_kv_head

        # Will be initialized in build()
        self.w_q = None
        self.w_k = None
        self.w_v = None
        self.w_o = None
        self.dropout = None
        self.rope = None
        self._build_input_shape = None

        logger.info(f"GroupedQueryAttention initialized: d_model={d_model}, "
                    f"n_head={n_head}, n_kv_head={n_kv_head}, groups={self.n_group}")

    def _validate_inputs(
            self,
            d_model: int,
            n_head: int,
            n_kv_head: int,
            attention_dropout: float,
            rope_percentage: float
    ) -> None:
        """Validate initialization parameters."""
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if n_head <= 0:
            raise ValueError(f"n_head must be positive, got {n_head}")
        if n_kv_head <= 0:
            raise ValueError(f"n_kv_head must be positive, got {n_kv_head}")
        if d_model % n_head != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_head ({n_head})")
        if n_head % n_kv_head != 0:
            raise ValueError(f"n_head ({n_head}) must be divisible by n_kv_head ({n_kv_head})")
        if not 0.0 <= attention_dropout <= 1.0:
            raise ValueError(f"attention_dropout must be in [0, 1], got {attention_dropout}")
        if not 0.0 <= rope_percentage <= 1.0:
            raise ValueError(f"rope_percentage must be in [0, 1], got {rope_percentage}")

    def build(self, input_shape) -> None:
        """Build the layer weights and sublayers."""
        self._build_input_shape = input_shape

        # Query projection: maps to n_head * head_dim
        self.w_q = layers.Dense(
            self.n_head * self.head_dim,
            use_bias=self.use_bias,
            name='w_q'
        )

        # Key projection: maps to n_kv_head * head_dim (fewer heads)
        self.w_k = layers.Dense(
            self.n_kv_head * self.head_dim,
            use_bias=self.use_bias,
            name='w_k'
        )

        # Value projection: maps to n_kv_head * head_dim (fewer heads)
        self.w_v = layers.Dense(
            self.n_kv_head * self.head_dim,
            use_bias=self.use_bias,
            name='w_v'
        )

        # Output projection: maps back to d_model
        self.w_o = layers.Dense(
            self.d_model,
            use_bias=self.use_bias,
            name='w_o'
        )

        # Attention dropout
        self.dropout = layers.Dropout(self.attention_dropout)

        # Rotary position embeddings
        self.rope = RotaryPositionEmbedding(
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
            rope_percentage=self.rope_percentage,
            name='rope'
        )

        # Build sublayers
        self.w_q.build(input_shape)
        self.w_k.build(input_shape)
        self.w_v.build(input_shape)

        # Output projection input shape is (batch, seq_len, d_model)
        self.w_o.build(input_shape)

        # Build RoPE with appropriate shape
        # RoPE expects (batch, n_heads, seq_len, head_dim)
        rope_shape = (input_shape[0], self.n_head, input_shape[1], self.head_dim)
        self.rope.build(rope_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None,
            mask: Optional[keras.KerasTensor] = None,
            return_attention_weights: bool = False
    ) -> keras.KerasTensor:
        """
        Apply grouped query attention.

        Args:
            inputs: Input tensor of shape (batch_size, seq_len, d_model)
            training: Whether in training mode
            mask: Optional attention mask of shape (batch_size, seq_len, seq_len)
            return_attention_weights: Whether to return attention weights

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
            If return_attention_weights=True, returns (output, attention_weights)
        """
        batch_size = ops.shape(inputs)[0]
        seq_len = ops.shape(inputs)[1]

        # Project to Q, K, V with different head counts
        q = self.w_q(inputs)  # (batch, seq_len, n_head * head_dim)
        k = self.w_k(inputs)  # (batch, seq_len, n_kv_head * head_dim)
        v = self.w_v(inputs)  # (batch, seq_len, n_kv_head * head_dim)

        # Reshape for multi-head attention
        q = ops.reshape(q, (batch_size, seq_len, self.n_head, self.head_dim))
        k = ops.reshape(k, (batch_size, seq_len, self.n_kv_head, self.head_dim))
        v = ops.reshape(v, (batch_size, seq_len, self.n_kv_head, self.head_dim))

        # Transpose to (batch, num_heads, seq_len, head_dim)
        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))

        # Apply rotary position embeddings to Q and K
        q = self.rope.apply_rope(q, seq_len)
        k = self.rope.apply_rope(k, seq_len)

        # Key insight: Repeat K,V for each group to match Q head count
        # Each K,V head serves n_group query heads
        k = ops.repeat(k, self.n_group, axis=1)  # (batch, n_head, seq_len, head_dim)
        v = ops.repeat(v, self.n_group, axis=1)  # (batch, n_head, seq_len, head_dim)

        # Scaled dot-product attention
        scores = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2)))
        scores = scores / ops.sqrt(ops.cast(self.head_dim, scores.dtype))

        # Apply causal mask if provided
        if mask is not None:
            # Expand mask to match attention score dimensions
            # mask shape: (batch, seq_len, seq_len) -> (batch, n_head, seq_len, seq_len)
            if len(ops.shape(mask)) == 3:
                mask = ops.expand_dims(mask, axis=1)  # Add head dimension
                mask = ops.repeat(mask, self.n_head, axis=1)

            # Apply mask by setting masked positions to large negative value
            scores = ops.where(mask, scores, -1e9)

        # Softmax and dropout
        attention_weights = ops.softmax(scores, axis=-1)
        attention_weights = self.dropout(attention_weights, training=training)

        # Apply attention to values
        out = ops.matmul(attention_weights, v)  # (batch, n_head, seq_len, head_dim)

        # Transpose back and reshape to original format
        out = ops.transpose(out, (0, 2, 1, 3))  # (batch, seq_len, n_head, head_dim)
        out = ops.reshape(out, (batch_size, seq_len, self.d_model))

        # Final output projection
        output = self.w_o(out)

        if return_attention_weights:
            return output, attention_weights
        return output

    def compute_output_shape(self, input_shape) -> tuple:
        """Compute output shape (same as input shape)."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "n_head": self.n_head,
            "n_kv_head": self.n_kv_head,
            "max_seq_len": self.max_seq_len,
            "attention_dropout": self.attention_dropout,
            "rope_percentage": self.rope_percentage,
            "use_bias": self.use_bias,
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get build configuration for serialization."""
        return {"input_shape": self._build_input_shape}

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build layer from configuration."""
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

# ---------------------------------------------------------------------
