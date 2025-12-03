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
    2. Apply rotary position embeddings to Q and K (Optional)
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
    Grouped Query Attention layer with optional rotary position embeddings.

    This layer implements an efficient attention mechanism that reduces computational
    and memory costs by sharing key-value projections across multiple query heads.
    It supports both 3D (sequence) and 4D (vision) inputs.

    **Intent**: Provide a production-ready grouped query attention implementation
    that achieves significant memory and computational savings over standard
    multi-head attention.

    **Architecture**:
    ```
    Input [B, seq_len, dim] OR [B, H, W, dim]
           ↓
    Q_proj → Q, K_proj → K, V_proj → V
           ↓
    (Optional) RoPE(Q, K)
           ↓
    Flatten spatial dims if 4D
           ↓
    Repeat K, V to match Q heads (Grouped Broadcast)
           ↓
    Attention(Q, K, V)
           ↓
    Reshape → [B, seq_len, dim] OR [B, H, W, dim]
           ↓
    Output_proj → Output
    ```

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
            embeddings to. If 0.0, RoPE is disabled. Defaults to 1.0.
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
        3D tensor: `(batch_size, sequence_length, dim)`
        4D tensor: `(batch_size, height, width, dim)`

    Output shape:
        Same as input shape.
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

        # CREATE all sub-layers in __init__
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

        # Rotary position embeddings (only if percentage > 0)
        if self.rope_percentage > 0.0:
            self.rope = RotaryPositionEmbedding(
                head_dim=self.head_dim,
                max_seq_len=self.max_seq_len,
                rope_theta=self.rope_theta,
                rope_percentage=self.rope_percentage,
                name='rope'
            )
        else:
            self.rope = None

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
        """Validate initialization parameters."""
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
        if not 0.0 <= rope_percentage <= 1.0:
            raise ValueError(f"rope_percentage must be in [0, 1], got {rope_percentage}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and all its sub-layers.
        Explicitly builds sub-layers for robust serialization.
        """
        # Detect if we need to flatten for 4D inputs during build logic if needed,
        # but Dense layers are generally agnostic to outer dimensions.
        self.w_q.build(input_shape)
        self.w_k.build(input_shape)
        self.w_v.build(input_shape)
        self.w_o.build(input_shape)
        self.dropout.build(input_shape)

        if self.rope is not None:
            # RoPE expects (batch, num_heads, seq_len, head_dim)
            # We estimate shapes based on input rank
            batch_size = input_shape[0] if input_shape[0] is not None else 1
            
            if len(input_shape) == 4:
                # 4D Input: (B, H, W, C) -> seq_len = H*W
                h = input_shape[1] if input_shape[1] is not None else self.max_seq_len
                w = input_shape[2] if input_shape[2] is not None else 1
                seq_len = h * w
            else:
                # 3D Input: (B, S, C)
                seq_len = input_shape[1] if input_shape[1] is not None else self.max_seq_len
                
            rope_input_shape = (batch_size, self.num_heads, seq_len, self.head_dim)
            self.rope.build(rope_input_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
        attention_mask: Optional[keras.KerasTensor] = None,
        return_attention_weights: bool = False
    ) -> Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]]:
        """
        Apply grouped query attention. Supports 3D (B, S, D) and 4D (B, H, W, D) inputs.
        """
        input_shape = ops.shape(inputs)
        rank = len(inputs.shape)
        batch_size = input_shape[0]
        
        # 1. Project to Q, K, V
        # Dense layers broadcast over spatial dims, so this works for 3D and 4D
        q = self.w_q(inputs, training=training)
        k = self.w_k(inputs, training=training)
        v = self.w_v(inputs, training=training)

        # 2. Flatten spatial dimensions if 4D
        if rank == 4:
            height, width = input_shape[1], input_shape[2]
            seq_len = height * width
            q = ops.reshape(q, (batch_size, seq_len, -1))
            k = ops.reshape(k, (batch_size, seq_len, -1))
            v = ops.reshape(v, (batch_size, seq_len, -1))
        else:
            seq_len = input_shape[1]

        # 3. Reshape for Multi-Head Attention
        # Q: (B, S, H, D_h)
        q = ops.reshape(q, (batch_size, seq_len, self.num_heads, self.head_dim))
        k = ops.reshape(k, (batch_size, seq_len, self.num_kv_heads, self.head_dim))
        v = ops.reshape(v, (batch_size, seq_len, self.num_kv_heads, self.head_dim))

        # Transpose to (B, H, S, D_h) for efficient attention
        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))

        # 4. Apply RoPE (Optional)
        if self.rope is not None:
            q = self.rope(q, training=training)
            k = self.rope(k, training=training)

        # 5. Grouping: Repeat K, V to match Q head count
        if self.num_groups > 1:
            k = ops.repeat(k, self.num_groups, axis=1)
            v = ops.repeat(v, self.num_groups, axis=1)

        # 6. Scaled Dot-Product Attention
        # (B, H, S, D_h) @ (B, H, D_h, S) -> (B, H, S, S)
        scores = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2)))
        scores = scores / ops.sqrt(ops.cast(self.head_dim, scores.dtype))

        if attention_mask is not None:
            scores = self._apply_mask(scores, attention_mask)

        attention_weights = ops.softmax(scores, axis=-1)
        attention_weights = self.dropout(attention_weights, training=training)

        # 7. Apply weights to Values
        # (B, H, S, S) @ (B, H, S, D_h) -> (B, H, S, D_h)
        out = ops.matmul(attention_weights, v)

        # 8. Restore Output Shape
        out = ops.transpose(out, (0, 2, 1, 3))  # (B, S, H, D_h)
        out = ops.reshape(out, (batch_size, seq_len, self.dim))  # (B, S, D)

        # Final projection
        output = self.w_o(out, training=training)

        # 9. Reshape back to 4D if input was 4D
        if rank == 4:
            output = ops.reshape(output, (batch_size, height, width, self.dim))

        if return_attention_weights:
            return output, attention_weights
        return output

    def _apply_mask(self, scores, mask):
        """Helper to broadcast and apply attention mask."""
        mask_shape = ops.shape(mask)
        
        # Handle 2D padding mask (B, S)
        if len(mask_shape) == 2:
            mask = ops.reshape(mask, (mask_shape[0], 1, 1, mask_shape[1]))
        # Handle 3D causal/combined mask (B, S, S)
        elif len(mask_shape) == 3:
            mask = ops.expand_dims(mask, axis=1)

        # Broadcast head dim if necessary
        if len(ops.shape(mask)) == 4 and ops.shape(mask)[1] == 1:
            mask = ops.repeat(mask, self.num_heads, axis=1)

        additive_mask = (1.0 - ops.cast(mask, scores.dtype)) * -1e9
        return scores + additive_mask

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        return tuple(input_shape)

    def get_config(self) -> Dict[str, Any]:
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
