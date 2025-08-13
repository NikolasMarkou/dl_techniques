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
from keras import ops
from typing import Optional, Any, Dict, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from ..rotary_position_embedding import RotaryPositionEmbedding

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class GroupedQueryAttention(keras.layers.Layer):
    """
    Grouped Query Attention layer with rotary position embeddings.

    This layer implements an efficient attention mechanism that reduces computational
    and memory costs by sharing key-value projections across multiple query heads
    while maintaining most of the representational power of full multi-head attention.

    Args:
        d_model: Integer, model dimension (embedding size). Must be positive and
            divisible by n_head.
        n_head: Integer, number of attention heads for queries. Must be positive.
        n_kv_head: Integer, number of key-value heads. Must be positive and must
            divide n_head evenly.
        max_seq_len: Integer, maximum sequence length for positional embeddings.
            Must be positive. Defaults to 2048.
        dropout_rate: Float, dropout rate applied to attention weights.
            Must be between 0 and 1. Defaults to 0.0.
        rope_percentage: Float, fraction of head dimensions to apply rotary
            embeddings to. Must be between 0 and 1 (exclusive of 0). Defaults to 1.0.
        rope_theta: Float, base for the rotary position embedding frequency computation.
            Defaults to 10000.0.
        use_bias: Boolean, whether to use bias in linear projections. Defaults to False.
        kernel_initializer: String or keras.initializers.Initializer, initializer for
            the kernel weights. Defaults to 'glorot_uniform'.
        bias_initializer: String or keras.initializers.Initializer, initializer for
            the bias weights. Defaults to 'zeros'.
        kernel_regularizer: Optional keras.regularizers.Regularizer, regularizer
            applied to kernel weights.
        bias_regularizer: Optional keras.regularizers.Regularizer, regularizer
            applied to bias weights.
        **kwargs: Additional keyword arguments for the Layer parent class.

    Input shape:
        3D tensor with shape: `(batch_size, sequence_length, d_model)`

    Output shape:
        3D tensor with shape: `(batch_size, sequence_length, d_model)`

    Raises:
        ValueError: If d_model is not divisible by n_head, or if n_head is not
                   divisible by n_kv_head, or if any parameter is out of valid range.

    Example:
        ```python
        # Basic usage
        gqa = GroupedQueryAttention(
            d_model=512,
            n_head=8,
            n_kv_head=2,  # 4 query heads per key/value head
            max_seq_len=2048
        )
        output = gqa(inputs)

        # Advanced configuration
        gqa = GroupedQueryAttention(
            d_model=768,
            n_head=12,
            n_kv_head=4,
            max_seq_len=4096,
            dropout_rate=0.1,
            rope_percentage=0.75,
            rope_theta=50000.0,
            use_bias=True,
            kernel_regularizer=keras.regularizers.L2(1e-4)
        )

        # In a transformer decoder
        def decoder_layer(x):
            # Self-attention with GQA
            attn_out = GroupedQueryAttention(
                d_model=512, n_head=8, n_kv_head=2
            )(x)
            x = x + attn_out

            # Feed-forward network
            ffn_out = FeedForward(...)(LayerNorm()(x))
            return x + ffn_out
        ```

    Note:
        This implementation follows the modern Keras 3 pattern where all sub-layers
        are created in __init__ and Keras handles the building automatically. This
        ensures proper serialization and eliminates common build errors.
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        n_kv_head: int,
        max_seq_len: int = 2048,
        dropout_rate: float = 0.0,
        rope_percentage: float = 1.0,
        rope_theta: float = 10000.0,
        use_bias: bool = False,
        kernel_initializer: keras.initializers.Initializer = 'glorot_uniform',
        bias_initializer: keras.initializers.Initializer = 'zeros',
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        self._validate_inputs(d_model, n_head, n_kv_head, max_seq_len,
                            dropout_rate, rope_percentage, rope_theta)

        # Store configuration parameters
        self.d_model = d_model
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout_rate
        self.rope_percentage = rope_percentage
        self.rope_theta = rope_theta
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # Derived parameters
        self.head_dim = self.d_model // self.n_head
        self.n_group = self.n_head // self.n_kv_head

        # CREATE all sub-layers in __init__ following modern Keras 3 pattern
        # Query projection: maps to n_head * head_dim
        self.w_q = keras.layers.Dense(
            self.n_head * self.head_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='w_q'
        )

        # Key projection: maps to n_kv_head * head_dim (fewer heads)
        self.w_k = keras.layers.Dense(
            self.n_kv_head * self.head_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='w_k'
        )

        # Value projection: maps to n_kv_head * head_dim (fewer heads)
        self.w_v = keras.layers.Dense(
            self.n_kv_head * self.head_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='w_v'
        )

        # Output projection: maps back to d_model
        self.w_o = keras.layers.Dense(
            self.d_model,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='w_o'
        )

        # Attention dropout
        self.dropout = keras.layers.Dropout(self.dropout_rate, name='attention_dropout')

        # Rotary position embeddings
        self.rope = RotaryPositionEmbedding(
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
            rope_percentage=self.rope_percentage,
            rope_theta=self.rope_theta,
            name='rope'
        )

        logger.info(f"GroupedQueryAttention initialized: d_model={d_model}, "
                    f"n_head={n_head}, n_kv_head={n_kv_head}, groups={self.n_group}")

    def _validate_inputs(
        self,
        d_model: int,
        n_head: int,
        n_kv_head: int,
        max_seq_len: int,
        dropout_rate: float,
        rope_percentage: float,
        rope_theta: float
    ) -> None:
        """
        Validate initialization parameters.

        Args:
            d_model: Model dimension to validate.
            n_head: Number of query heads to validate.
            n_kv_head: Number of key-value heads to validate.
            max_seq_len: Maximum sequence length to validate.
            dropout_rate: Dropout rate to validate.
            rope_percentage: RoPE percentage to validate.
            rope_theta: RoPE theta parameter to validate.

        Raises:
            ValueError: If any parameter is invalid.
        """
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if n_head <= 0:
            raise ValueError(f"n_head must be positive, got {n_head}")
        if n_kv_head <= 0:
            raise ValueError(f"n_kv_head must be positive, got {n_kv_head}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        if d_model % n_head != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_head ({n_head})")
        if n_head % n_kv_head != 0:
            raise ValueError(f"n_head ({n_head}) must be divisible by n_kv_head ({n_kv_head})")
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1], got {dropout_rate}")
        if not 0.0 < rope_percentage <= 1.0:
            raise ValueError(f"rope_percentage must be in (0, 1], got {rope_percentage}")
        if rope_theta <= 0:
            raise ValueError(f"rope_theta must be positive, got {rope_theta}")

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
            inputs: Input tensor of shape (batch_size, seq_len, d_model).
            training: Boolean indicating whether the layer should behave in
                training mode (add dropout) or inference mode (no dropout).
            mask: Optional attention mask of shape (batch_size, seq_len, seq_len).
                If provided, positions with True values will be attended to,
                and positions with False values will be masked out.
            return_attention_weights: Boolean, whether to return attention weights
                along with the output. Defaults to False.

        Returns:
            If return_attention_weights=False:
                Output tensor of shape (batch_size, seq_len, d_model).
            If return_attention_weights=True:
                Tuple of (output, attention_weights) where attention_weights has
                shape (batch_size, n_head, seq_len, seq_len).
        """
        batch_size = ops.shape(inputs)[0]
        seq_len = ops.shape(inputs)[1]

        # Project to Q, K, V with different head counts
        q = self.w_q(inputs, training=training)  # (batch, seq_len, n_head * head_dim)
        k = self.w_k(inputs, training=training)  # (batch, seq_len, n_kv_head * head_dim)
        v = self.w_v(inputs, training=training)  # (batch, seq_len, n_kv_head * head_dim)

        # Reshape for multi-head attention
        q = ops.reshape(q, (batch_size, seq_len, self.n_head, self.head_dim))
        k = ops.reshape(k, (batch_size, seq_len, self.n_kv_head, self.head_dim))
        v = ops.reshape(v, (batch_size, seq_len, self.n_kv_head, self.head_dim))

        # Transpose to (batch, num_heads, seq_len, head_dim)
        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))

        # Apply rotary position embeddings to Q and K
        # RoPE expects (batch, num_heads, seq_len, head_dim) which is what we have
        q = self.rope(q, training=training)
        k = self.rope(k, training=training)

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
        output = self.w_o(out, training=training)

        if return_attention_weights:
            return output, attention_weights
        return output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute output shape (same as input shape).

        Args:
            input_shape: Shape tuple of the input tensor.

        Returns:
            Output shape tuple, same as input shape for attention layers.
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "n_head": self.n_head,
            "n_kv_head": self.n_kv_head,
            "max_seq_len": self.max_seq_len,
            "dropout_rate": self.dropout_rate,
            "rope_percentage": self.rope_percentage,
            "rope_theta": self.rope_theta,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------
