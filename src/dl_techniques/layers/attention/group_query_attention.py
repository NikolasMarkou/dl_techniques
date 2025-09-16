"""
Grouped Query Attention (GQA) Implementation with Rotary Position Embeddings

This module implements Grouped Query Attention, an efficient attention mechanism that
reduces the computational and memory requirements of multi-head attention by sharing
key and value projections across multiple query heads while maintaining model quality.

Mathematical Formulation:
    Standard Multi-Head Attention uses:
    - num_heads query projections
    - num_heads key projections
    - num_heads value projections

    Grouped Query Attention uses:
    - num_heads query projections
    - num_kv_heads key projections (where num_kv_heads < num_heads)
    - num_kv_heads value projections

    The key insight is that num_heads % num_kv_heads == 0, creating groups where:
    group_size = num_heads // num_kv_heads

    Each K,V pair is shared across group_size query heads by repeating the K,V
    tensors along the head dimension.

Computational Benefits:
    - Reduces memory usage for K,V caches in autoregressive generation
    - Maintains most of the representational power of full multi-head attention
    - Particularly effective in large language models where KV cache dominates memory
    - Enables longer sequence lengths with same memory budget

Architecture Details:
    1. Project input to Q (num_heads), K (num_kv_heads), V (num_kv_heads)
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
from typing import Optional, Union, Any, Dict, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from ..embedding.rotary_position_embedding import RotaryPositionEmbedding

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class GroupedQueryAttention(keras.layers.Layer):
    """
    Grouped Query Attention layer with rotary position embeddings.

    This layer implements an efficient attention mechanism that reduces computational
    and memory costs by sharing key-value projections across multiple query heads
    while maintaining most of the representational power of full multi-head attention.

    **Intent**: Provide a production-ready grouped query attention implementation
    that achieves significant memory and computational savings over standard
    multi-head attention, particularly beneficial for large language models and
    long sequence processing.

    **Architecture**:
    ```
    Input [B, seq_len, dim]
           ↓
    Q_proj → Q [B, num_heads, seq_len, head_dim]
    K_proj → K [B, num_kv_heads, seq_len, head_dim] → repeat → [B, num_heads, seq_len, head_dim]
    V_proj → V [B, num_kv_heads, seq_len, head_dim] → repeat → [B, num_heads, seq_len, head_dim]
           ↓
    RoPE(Q, K) → Q', K'
           ↓
    Attention(Q', K', V) → [B, num_heads, seq_len, head_dim]
           ↓
    Reshape → [B, seq_len, dim]
           ↓
    Output_proj → Output [B, seq_len, dim]
    ```

    **Mathematical Operations**:
    1. **Projections**: Q = X W_q, K = X W_k, V = X W_v
    2. **Grouping**: K' = repeat(K, group_size, axis=1), V' = repeat(V, group_size, axis=1)
    3. **RoPE**: Q_rope = RoPE(Q), K_rope = RoPE(K')
    4. **Attention**: A = softmax(Q_rope K_rope^T / √d_k) V'
    5. **Output**: O = A W_o

    Args:
        dim: Integer, input/output dimension (embedding size). Must be positive and
            divisible by num_heads.
        num_heads: Integer, number of attention heads for queries. Must be positive.
        num_kv_heads: Integer, number of key-value heads. Must be positive and divide
            num_heads evenly for grouping.
        max_seq_len: Integer, maximum sequence length for positional embeddings.
            Must be positive. Defaults to 2048.
        dropout_rate: Float, dropout rate applied to attention weights.
            Must be between 0.0 and 1.0. Defaults to 0.0.
        rope_percentage: Float, fraction of head dimensions to apply rotary
            embeddings to. Must be between 0.0 (exclusive) and 1.0 (inclusive).
            Defaults to 1.0.
        rope_theta: Float, base frequency for rotary position embeddings.
            Must be positive. Defaults to 10000.0.
        use_bias: Boolean, whether to use bias in linear projections.
            Defaults to False.
        kernel_initializer: String or initializer instance, initializer for
            kernel weights. Defaults to 'glorot_uniform'.
        bias_initializer: String or initializer instance, initializer for
            bias weights. Defaults to 'zeros'.
        kernel_regularizer: Optional regularizer for kernel weights.
        bias_regularizer: Optional regularizer for bias weights.
        **kwargs: Additional keyword arguments for the Layer parent class.

    Input shape:
        3D tensor with shape: `(batch_size, sequence_length, dim)`

    Output shape:
        3D tensor with shape: `(batch_size, sequence_length, dim)`

    Attributes:
        w_q: Dense layer for query projection with shape (dim, num_heads * head_dim).
        w_k: Dense layer for key projection with shape (dim, num_kv_heads * head_dim).
        w_v: Dense layer for value projection with shape (dim, num_kv_heads * head_dim).
        w_o: Dense layer for output projection with shape (num_heads * head_dim, dim).
        dropout: Dropout layer for attention weights.
        rope: Rotary position embedding layer.

    Call arguments:
        inputs: Input tensor of shape `(batch_size, seq_len, dim)`.
        training: Boolean indicating training or inference mode. Affects dropout.
        mask: Optional attention mask of shape `(batch_size, seq_len, seq_len)`
            or `(batch_size, 1, seq_len, seq_len)`. Values should be 1 for
            positions to attend to and 0 for masked positions.
        return_attention_weights: Boolean, whether to return attention weights
            along with the output. Defaults to False.

    Returns:
        If return_attention_weights=False:
            Output tensor of shape `(batch_size, seq_len, dim)`
        If return_attention_weights=True:
            Tuple of (output, attention_weights) where attention_weights
            has shape `(batch_size, num_heads, seq_len, seq_len)`

    Example:
        ```python
        # Basic usage
        gqa = GroupedQueryAttention(
            dim=512,
            num_heads=8,
            num_kv_heads=2,  # 4 query heads per key/value head
            max_seq_len=2048
        )

        inputs = keras.Input(shape=(128, 512))
        outputs = gqa(inputs)

        # Advanced configuration
        gqa = GroupedQueryAttention(
            dim=768,
            num_heads=12,
            num_kv_heads=4,
            dropout_rate=0.1,
            rope_percentage=0.8,
            rope_theta=50000.0,
            kernel_regularizer=keras.regularizers.L2(1e-4)
        )

        # In a transformer decoder
        def decoder_layer(x):
            # Self-attention with GQA
            attn_out = GroupedQueryAttention(
                dim=512, num_heads=8, num_kv_heads=2
            )(x)
            x = x + attn_out

            # Feed-forward network
            ffn_out = FeedForward(...)(keras.layers.LayerNormalization()(x))
            return x + ffn_out
        ```

    Raises:
        ValueError: If dim is not positive or not divisible by num_heads.
        ValueError: If num_heads is not positive or not divisible by num_kv_heads.
        ValueError: If num_kv_heads is not positive.
        ValueError: If dropout_rate is not between 0.0 and 1.0.
        ValueError: If rope_percentage is not between 0.0 (exclusive) and 1.0 (inclusive).
        ValueError: If max_seq_len or rope_theta are not positive.

    Note:
        This implementation follows modern Keras 3 patterns where all sub-layers
        are created in __init__ and explicitly built in build() method for robust
        serialization support.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        max_seq_len: int = 2048,
        dropout_rate: float = 0.0,
        rope_percentage: float = 1.0,
        rope_theta: float = 10000.0,
        use_bias: bool = False,
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        self._validate_inputs(dim, num_heads, num_kv_heads, max_seq_len,
                            dropout_rate, rope_percentage, rope_theta)

        # Store ALL configuration parameters
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
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
        self.head_dim = self.dim // self.num_heads
        self.num_groups = self.num_heads // self.num_kv_heads

        # CREATE all sub-layers in __init__ (they are unbuilt)
        self.w_q = keras.layers.Dense(
            self.num_heads * self.head_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='w_q'
        )

        self.w_k = keras.layers.Dense(
            self.num_kv_heads * self.head_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='w_k'
        )

        self.w_v = keras.layers.Dense(
            self.num_kv_heads * self.head_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='w_v'
        )

        self.w_o = keras.layers.Dense(
            self.dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='w_o'
        )

        # Attention dropout layer
        self.dropout = keras.layers.Dropout(self.dropout_rate, name='attention_dropout')

        # Rotary position embeddings
        self.rope = RotaryPositionEmbedding(
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
            rope_theta=self.rope_theta,
            rope_percentage=self.rope_percentage,
            name='rope'
        )

        logger.info(f"GroupedQueryAttention initialized: dim={dim}, "
                   f"num_heads={num_heads}, num_kv_heads={num_kv_heads}, groups={self.num_groups}")

    def _validate_inputs(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        max_seq_len: int,
        dropout_rate: float,
        rope_percentage: float,
        rope_theta: float
    ) -> None:
        """
        Validate initialization parameters.

        Args:
            dim: Model dimension to validate.
            num_heads: Number of query heads to validate.
            num_kv_heads: Number of key-value heads to validate.
            max_seq_len: Maximum sequence length to validate.
            dropout_rate: Dropout rate to validate.
            rope_percentage: RoPE percentage to validate (must be > 0.0 and <= 1.0).
            rope_theta: RoPE theta parameter to validate.

        Raises:
            ValueError: If any parameter is invalid.
        """
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if num_kv_heads <= 0:
            raise ValueError(f"num_kv_heads must be positive, got {num_kv_heads}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        if rope_theta <= 0:
            raise ValueError(f"rope_theta must be positive, got {rope_theta}")
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        if num_heads % num_kv_heads != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})")
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1], got {dropout_rate}")
        if not 0.0 < rope_percentage <= 1.0:
            raise ValueError(f"rope_percentage must be in (0, 1], got {rope_percentage}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and all its sub-layers.

        This method explicitly builds each sub-layer for robust serialization
        support, ensuring all weight variables exist before weight restoration.

        Args:
            input_shape: Shape tuple of the input tensor.
        """
        # Build sub-layers in computational order
        self.w_q.build(input_shape)
        self.w_k.build(input_shape)
        self.w_v.build(input_shape)

        # Output projection uses the same input shape as it processes the
        # reshaped attention output which has the same dim dimension
        self.w_o.build(input_shape)

        # Dropout doesn't need explicit building as it has no weights
        self.dropout.build(input_shape)

        # Build RoPE layer explicitly for serialization robustness
        # RoPE expects (batch, num_heads, seq_len, head_dim)
        batch_size = input_shape[0] if input_shape[0] is not None else 1
        seq_len = input_shape[1] if input_shape[1] is not None else self.max_seq_len
        rope_input_shape = (batch_size, self.num_heads, seq_len, self.head_dim)
        self.rope.build(rope_input_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
        attention_mask: Optional[keras.KerasTensor] = None,
        return_attention_weights: bool = False
    ) -> Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]]:
        """
        Apply grouped query attention.

        Args:
            inputs: Input tensor of shape (batch_size, seq_len, dim).
            training: Boolean, whether in training mode. Affects dropout behavior.
            attention_mask: Optional attention mask of shape (batch_size, seq_len, seq_len)
                or (batch_size, 1, seq_len, seq_len). Values should be 1 for
                positions to attend to and 0 for masked positions.
            return_attention_weights: Boolean, whether to return attention weights
                along with the output. Defaults to False.

        Returns:
            If return_attention_weights=False:
                Output tensor of shape (batch_size, seq_len, dim)
            If return_attention_weights=True:
                Tuple of (output, attention_weights) where attention_weights
                has shape (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size = ops.shape(inputs)[0]
        seq_len = ops.shape(inputs)[1]

        # Project to Q, K, V with different head counts
        q = self.w_q(inputs)  # (batch, seq_len, num_heads * head_dim)
        k = self.w_k(inputs)  # (batch, seq_len, num_kv_heads * head_dim)
        v = self.w_v(inputs)  # (batch, seq_len, num_kv_heads * head_dim)

        # Reshape for multi-head attention
        q = ops.reshape(q, (batch_size, seq_len, self.num_heads, self.head_dim))
        k = ops.reshape(k, (batch_size, seq_len, self.num_kv_heads, self.head_dim))
        v = ops.reshape(v, (batch_size, seq_len, self.num_kv_heads, self.head_dim))

        # Transpose to (batch, num_heads, seq_len, head_dim) for attention computation
        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))

        # Apply rotary position embeddings to Q and K
        # RoPE expects (batch, num_heads, seq_len, head_dim) which is what we have
        q = self.rope(q, training=training)
        k = self.rope(k, training=training)

        # Key insight: Repeat K,V for each group to match Q head count
        # Each K,V head serves num_groups query heads
        k = ops.repeat(k, self.num_groups, axis=1)  # (batch, num_heads, seq_len, head_dim)
        v = ops.repeat(v, self.num_groups, axis=1)  # (batch, num_heads, seq_len, head_dim)

        # Scaled dot-product attention
        scores = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2)))
        scores = scores / ops.sqrt(ops.cast(self.head_dim, scores.dtype))

        # Apply attention mask if provided
        if attention_mask is not None:
            # Handle different mask shapes
            if len(ops.shape(attention_mask)) == 3:
                # (batch, seq_len, seq_len) -> (batch, num_heads, seq_len, seq_len)
                attention_mask = ops.expand_dims(attention_mask, axis=1)
                attention_mask = ops.repeat(attention_mask, self.num_heads, axis=1)
            elif len(ops.shape(attention_mask)) == 4:
                # Assume (batch, 1, seq_len, seq_len) or (batch, num_heads, seq_len, seq_len)
                if ops.shape(attention_mask)[1] == 1:
                    attention_mask = ops.repeat(attention_mask, self.num_heads, axis=1)

            # Convert mask to additive form: 0 -> -inf, 1 -> 0
            attention_mask = ops.cast(attention_mask, scores.dtype)
            attention_mask = (1.0 - attention_mask) * -1e9
            scores = scores + attention_mask

        # Softmax and dropout
        attention_weights = ops.softmax(scores, axis=-1)
        attention_weights = self.dropout(attention_weights, training=training)

        # Apply attention to values
        out = ops.matmul(attention_weights, v)  # (batch, num_heads, seq_len, head_dim)

        # Transpose back and reshape to original format
        out = ops.transpose(out, (0, 2, 1, 3))  # (batch, seq_len, num_heads, head_dim)
        out = ops.reshape(out, (batch_size, seq_len, self.dim))

        # Final output projection
        output = self.w_o(out, training=training)

        if return_attention_weights:
            return output, attention_weights
        return output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape tuple of the input tensor.

        Returns:
            Output shape tuple, same as input shape for attention layers.
        """
        return tuple(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns ALL configuration parameters passed to __init__ for proper
        serialization and deserialization.

        Returns:
            Dictionary containing the layer configuration with all parameters
            required to recreate this layer.
        """
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'num_heads': self.num_heads,
            'num_kv_heads': self.num_kv_heads,
            'max_seq_len': self.max_seq_len,
            'dropout_rate': self.dropout_rate,
            'rope_percentage': self.rope_percentage,
            'rope_theta': self.rope_theta,
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------
