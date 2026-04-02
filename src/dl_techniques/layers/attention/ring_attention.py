"""
Exact attention for long sequences via blockwise processing.

This layer implements Ring Attention, a memory-efficient algorithm that
makes it possible to apply transformer attention to sequences of nearly
unlimited length. It overcomes the quadratic memory complexity of standard
attention by computing the attention matrix in smaller, fixed-size blocks,
reducing memory from ``O(N^2)`` to ``O(block_size^2)`` while maintaining
exact mathematical equivalence through online softmax computation.

References:
    - Liu, H., et al. (2023). "Ring Attention with Blockwise Transformers for
      Near-Infinite Context."
    - Milakov, M. & Gimelshein, N. (2018). "Online normalizer calculation
      for softmax."
    - Dao, T., et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact
      Attention with IO-Awareness."
"""

import math
import keras
from keras import ops
from typing import Optional, Union, Any, Dict, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class RingAttention(keras.layers.Layer):
    """Ring Attention with blockwise processing for extremely long sequences.

    Implements the Ring Attention algorithm that partitions the sequence into
    fixed-size blocks and computes attention incrementally using online softmax.
    For each query block ``Q_i``, the algorithm iterates through all key-value
    blocks ``(K_j, V_j)``, computing partial scores
    ``S_ij = Q_i K_j^T / sqrt(d_k)`` and maintaining running statistics:
    ``m_new = max(m_prev, max(S_ij))``,
    ``O_new = O_prev * exp(m_prev - m_new) + exp(S_ij - m_new) V_j``,
    ``l_new = l_prev * exp(m_prev - m_new) + sum(exp(S_ij - m_new))``.
    The final output is ``O / l``, which is mathematically identical to standard
    attention without ever materializing the full ``N x N`` attention matrix.

    **Architecture Overview:**

    .. code-block:: text

        ┌───────────────────────────────┐
        │ Input [B, seq_len, dim]       │
        └──────────────┬────────────────┘
                       ▼
        ┌───────────────────────────────┐
        │ Q = W_q(X), K = W_k(X),       │
        │ V = W_v(X)                    │
        │ [B, heads, seq_len, head_dim] │
        └──────────────┬────────────────┘
                       ▼
        ┌───────────────────────────────┐
        │ Split into blocks of size B_s │
        │ Q_blocks, K_blocks, V_blocks  │
        └──────────────┬────────────────┘
                       ▼
        ┌───────────────────────────────────────┐
        │ For each Q_block_i:                   │
        │ ┌───────────────────────────────────┐ │
        │ │ Init: m=-inf, l=0, O=0            │ │
        │ ├───────────────────────────────────┤ │
        │ │ For each (K_j, V_j):              │ │
        │ │  S = Q_i @ K_j^T * scale          │ │
        │ │  m_new = max(m, max(S))           │ │
        │ │  O = O * exp(m-m_new)             │ │
        │ │      + exp(S-m_new) @ V_j         │ │
        │ │  l = l * exp(m-m_new)             │ │
        │ │      + sum(exp(S-m_new))          │ │
        │ │  m = m_new                        │ │
        │ └───────────────────────────────────┘ │
        │ Output_i = O / l                      │
        └──────────────┬────────────────────────┘
                       ▼
        ┌───────────────────────────────┐
        │ Concatenate block outputs     │
        │ Reshape → [B, seq_len, dim]   │
        └──────────────┬────────────────┘
                       ▼
        ┌───────────────────────────────┐
        │ W_o output projection         │
        └──────────────┬────────────────┘
                       ▼
        ┌───────────────────────────────┐
        │ Output [B, seq_len, dim]      │
        └───────────────────────────────┘

    :param dim: Input/output dimension (embedding size). Must be positive and
        divisible by num_heads.
    :type dim: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param block_size: Sequence block size for processing. Larger blocks use
        more memory but may be more efficient.
    :type block_size: int
    :param dropout_rate: Dropout rate for attention weights, between 0.0 and 1.0.
    :type dropout_rate: float
    :param use_bias: Whether to use bias in linear projections.
    :type use_bias: bool
    :param kernel_initializer: Initializer for kernel weights.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for bias weights.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for kernel weights.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param bias_regularizer: Optional regularizer for bias weights.
    :type bias_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional keyword arguments for the Layer parent class.
    :type kwargs: Any

    :raises ValueError: If dim is not positive or not divisible by num_heads.
    :raises ValueError: If num_heads or block_size are not positive.
    :raises ValueError: If dropout_rate is not between 0.0 and 1.0.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            block_size: int = 512,
            dropout_rate: float = 0.0,
            use_bias: bool = False,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        self._validate_inputs(dim, num_heads, block_size, dropout_rate)

        # Store ALL configuration parameters
        self.dim = dim
        self.num_heads = num_heads
        self.block_size = block_size
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # Derived parameters
        self.head_dim = self.dim // self.num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

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
            self.num_heads * self.head_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='w_k'
        )

        self.w_v = keras.layers.Dense(
            self.num_heads * self.head_dim,
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

        logger.info(f"RingAttention initialized: dim={dim}, "
                    f"num_heads={num_heads}, block_size={block_size}")

    def _validate_inputs(
            self,
            dim: int,
            num_heads: int,
            block_size: int,
            dropout_rate: float
    ) -> None:
        """Validate initialization parameters.

        :param dim: Model dimension to validate.
        :type dim: int
        :param num_heads: Number of attention heads to validate.
        :type num_heads: int
        :param block_size: Block size for sequence processing to validate.
        :type block_size: int
        :param dropout_rate: Dropout rate to validate.
        :type dropout_rate: float

        :raises ValueError: If any parameter is invalid.
        """
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1], got {dropout_rate}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer and all sub-layers for robust serialization.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
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

        # Always call parent build at the end
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None,
            attention_mask: Optional[keras.KerasTensor] = None,
            return_attention_weights: bool = False
    ) -> Union[keras.KerasTensor, Tuple[keras.KerasTensor, None]]:
        """Apply ring attention with blockwise processing.

        :param inputs: Input tensor of shape ``(batch_size, seq_len, dim)``.
        :type inputs: keras.KerasTensor
        :param training: Whether in training mode. Affects dropout behavior.
        :type training: Optional[bool]
        :param attention_mask: Optional attention mask of shape
            ``(batch_size, seq_len, seq_len)`` or
            ``(batch_size, 1, seq_len, seq_len)``.
        :type attention_mask: Optional[keras.KerasTensor]
        :param return_attention_weights: Whether to return attention weights.
            Always returns ``None`` due to blockwise processing.
        :type return_attention_weights: bool

        :return: Output tensor of shape ``(batch_size, seq_len, dim)``.
            If return_attention_weights is ``True``, returns ``(output, None)``.
        :rtype: Union[keras.KerasTensor, Tuple[keras.KerasTensor, None]]
        """
        batch_size = ops.shape(inputs)[0]
        seq_len = ops.shape(inputs)[1]

        # Project to Q, K, V
        q = self.w_q(inputs)  # (batch, seq_len, num_heads * head_dim)
        k = self.w_k(inputs)  # (batch, seq_len, num_heads * head_dim)
        v = self.w_v(inputs)  # (batch, seq_len, num_heads * head_dim)

        # Reshape for multi-head attention
        q = ops.reshape(q, (batch_size, seq_len, self.num_heads, self.head_dim))
        k = ops.reshape(k, (batch_size, seq_len, self.num_heads, self.head_dim))
        v = ops.reshape(v, (batch_size, seq_len, self.num_heads, self.head_dim))

        # Transpose to (batch, num_heads, seq_len, head_dim) for attention computation
        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))

        # Apply scaling
        q = q * self.scale

        # Compute blockwise attention
        attention_output = self._blockwise_attention(
            q, k, v, attention_mask=attention_mask, training=training
        )

        # Transpose back and reshape to original format
        attention_output = ops.transpose(attention_output, (0, 2, 1, 3))  # (batch, seq_len, num_heads, head_dim)
        attention_output = ops.reshape(attention_output, (batch_size, seq_len, self.dim))

        # Final output projection
        output = self.w_o(attention_output, training=training)

        if return_attention_weights:
            # Return None for attention weights as they're not materialized in Ring Attention
            return output, None
        return output

    def _blockwise_attention(
            self,
            queries: keras.KerasTensor,
            keys: keras.KerasTensor,
            values: keras.KerasTensor,
            attention_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Compute blockwise attention with online softmax.

        :param queries: Query tensor of shape ``(batch, num_heads, seq_len, head_dim)``.
        :type queries: keras.KerasTensor
        :param keys: Key tensor of shape ``(batch, num_heads, seq_len, head_dim)``.
        :type keys: keras.KerasTensor
        :param values: Value tensor of shape ``(batch, num_heads, seq_len, head_dim)``.
        :type values: keras.KerasTensor
        :param attention_mask: Optional mask for attention computation.
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Whether in training mode.
        :type training: Optional[bool]

        :return: Attention output of shape ``(batch, num_heads, seq_len, head_dim)``.
        :rtype: keras.KerasTensor
        """
        batch_size = ops.shape(queries)[0]
        num_heads = self.num_heads
        seq_len = ops.shape(queries)[2]
        head_dim = self.head_dim

        # Calculate number of blocks
        num_blocks = (seq_len + self.block_size - 1) // self.block_size

        # Initialize output tensor
        outputs = ops.zeros_like(queries)

        # Process each query block
        for q_block_idx in range(num_blocks):
            # Get query block bounds
            q_start = q_block_idx * self.block_size
            q_end = ops.minimum(q_start + self.block_size, seq_len)
            q_block = queries[:, :, q_start:q_end, :]  # (batch, num_heads, q_block_size, head_dim)

            q_block_size = q_end - q_start

            # Initialize running statistics for online softmax
            # (batch, num_heads, q_block_size)
            running_max = ops.full(
                (batch_size, num_heads, q_block_size),
                -float('inf'),
                dtype=queries.dtype
            )
            running_sum = ops.zeros(
                (batch_size, num_heads, q_block_size),
                dtype=queries.dtype
            )
            # (batch, num_heads, q_block_size, head_dim)
            accumulated_output = ops.zeros_like(q_block)

            # Process each key/value block for this query block
            for kv_block_idx in range(num_blocks):
                # Get key/value block bounds
                kv_start = kv_block_idx * self.block_size
                kv_end = ops.minimum(kv_start + self.block_size, seq_len)
                k_block = keys[:, :, kv_start:kv_end, :]  # (batch, num_heads, kv_block_size, head_dim)
                v_block = values[:, :, kv_start:kv_end, :]

                # Compute attention scores for this block pair
                # (batch, num_heads, q_block_size, head_dim) @ (batch, num_heads, head_dim, kv_block_size)
                # -> (batch, num_heads, q_block_size, kv_block_size)
                scores = ops.matmul(q_block, ops.transpose(k_block, (0, 1, 3, 2)))

                # Apply attention mask if provided
                if attention_mask is not None:
                    # Extract relevant portion of mask for these blocks
                    if len(ops.shape(attention_mask)) == 3:
                        # (batch, seq_len, seq_len) -> extract block region
                        mask_slice = attention_mask[:, q_start:q_end, kv_start:kv_end]
                        # Expand for heads: (batch, 1, q_block_size, kv_block_size)
                        mask_slice = ops.expand_dims(mask_slice, axis=1)
                        mask_slice = ops.repeat(mask_slice, num_heads, axis=1)
                    elif len(ops.shape(attention_mask)) == 4:
                        # (batch, num_heads, seq_len, seq_len) -> extract block region
                        mask_slice = attention_mask[:, :, q_start:q_end, kv_start:kv_end]

                    # Convert mask to additive form: 0 -> -inf, 1 -> 0
                    mask_slice = ops.cast(mask_slice, scores.dtype)
                    mask_slice = (1.0 - mask_slice) * -1e9
                    scores = scores + mask_slice

                # Compute new maximum for safe softmax
                # (batch, num_heads, q_block_size)
                block_max = ops.max(scores, axis=-1)
                new_max = ops.maximum(running_max, block_max)

                # Renormalize previous results
                max_diff = running_max - new_max
                renorm_factor = ops.exp(max_diff)

                # Update running statistics
                running_sum = running_sum * renorm_factor
                accumulated_output = accumulated_output * ops.expand_dims(renorm_factor, axis=-1)

                # Compute new contributions
                # (batch, num_heads, q_block_size, kv_block_size)
                new_scores = ops.exp(scores - ops.expand_dims(new_max, axis=-1))

                # Apply dropout to attention weights
                if training and self.dropout_rate > 0:
                    new_scores = self.dropout(new_scores, training=training)

                # Accumulate new attention output
                # (batch, num_heads, q_block_size, kv_block_size) @ (batch, num_heads, kv_block_size, head_dim)
                # -> (batch, num_heads, q_block_size, head_dim)
                new_output = ops.matmul(new_scores, v_block)
                accumulated_output = accumulated_output + new_output

                # Update running sum
                running_sum = running_sum + ops.sum(new_scores, axis=-1)
                running_max = new_max

            # Normalize final output for this query block
            # (batch, num_heads, q_block_size, head_dim)
            running_sum_expanded = ops.expand_dims(running_sum, axis=-1)
            block_output = accumulated_output / running_sum_expanded

            # Place block output in final tensor using slice assignment
            # Create indices for the slice assignment
            start_indices = [0, 0, q_start, 0]
            outputs = ops.slice_update(
                outputs,
                start_indices=start_indices,
                updates=block_output
            )

        return outputs

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]

        :return: Output shape tuple, same as input shape.
        :rtype: Tuple[Optional[int], ...]
        """
        return tuple(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        :return: Dictionary containing all parameters required to recreate this layer.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'num_heads': self.num_heads,
            'block_size': self.block_size,
            'dropout_rate': self.dropout_rate,
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------
