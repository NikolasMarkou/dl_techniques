"""Computes exact attention for long sequences via blockwise processing.

    This layer implements Ring Attention, a memory-efficient algorithm that
    makes it possible to apply transformer attention to sequences of nearly
    unlimited length. It overcomes the quadratic memory complexity of standard
    attention by computing the attention matrix in smaller, fixed-size blocks.

    Architecture:
        The primary obstacle in scaling attention is the materialization of the
        full attention matrix `A = Q @ K.T`, which has a memory footprint of
        O(N^2) where N is the sequence length. Ring Attention avoids this by
        never constructing the full matrix.

        The core architectural change is to partition the sequence into blocks of a
        fixed `block_size`. The computation is then restructured as a nested
        loop: for each query block `Q_i`, the algorithm iterates through all
        key-value blocks `(K_j, V_j)`. At each step, it computes a partial score
        matrix of size `(block_size, block_size)`, uses it to update a running
        tally of the attention output for `Q_i`, and then discards the partial
        matrix. The memory usage is thus decoupled from the sequence length N
        and instead depends on the `block_size`, reducing memory complexity
        from O(N^2) to O(block_size^2).

    Foundational Mathematics:
        The key challenge is ensuring that this blockwise computation is
        mathematically identical to standard attention. This is achieved using
        an "online softmax" algorithm. The standard softmax function requires a
        normalization term (the sum of all exponentials) that can only be
        computed after seeing all scores.

        The online softmax overcomes this by maintaining running statistics. For
        each query block, as it iterates through key blocks, it tracks:
        1.  `m_i`: The maximum score encountered so far.
        2.  `O_i`: The accumulated, unnormalized attention output.
        3.  `l_i`: The accumulated softmax denominator (sum of exponentials).

        When a new block of scores is computed, the new global maximum `m_{i+1}`
        is found. The previous running statistics `O_i` and `l_i` are then
        rescaled by a factor of `exp(m_i - m_{i+1})` to align them with the new
        maximum. The contributions from the current block are then computed
        using this new maximum and added to the rescaled running statistics.
        This iterative update, particularly the rescaling step, is a numerically
        stable way to compute the exact softmax value without needing all scores
        simultaneously. The final, correctly normalized attention output for the
        query block is obtained only after iterating through all key-value blocks.

    References:
        - The primary algorithm and its application in blockwise transformers:
          Liu, H., et al. (2023). "Ring Attention with Blockwise Transformers for
          Near-Infinite Context."

        - The foundational online softmax algorithm:
          Milakov, M. & Gimelshein, N. (2018). "Online normalizer calculation
          for softmax."

        - A related work that popularized blockwise attention for hardware efficiency:
          Dao, T., et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact
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
    """
    Ring Attention layer with blockwise processing for extremely long sequences.

    This layer implements the Ring Attention algorithm that enables processing of
    very long sequences (millions of tokens) by computing attention in blocks while
    maintaining exact mathematical equivalence to standard attention through online
    softmax computation.

    **Intent**: Provide a production-ready Ring Attention implementation that enables
    transformer models to process extremely long sequences without memory constraints,
    particularly beneficial for document understanding, long-form generation, and
    scientific text processing.

    **Architecture**:
    ```
    Input [B, seq_len, dim]
           ↓
    Q_proj → Q [B, seq_len, num_heads, head_dim]
    K_proj → K [B, seq_len, num_heads, head_dim]
    V_proj → V [B, seq_len, num_heads, head_dim]
           ↓
    Split into blocks: Q_blocks, K_blocks, V_blocks
           ↓
    For each Q_block:
      ├─ Initialize running stats (max, sum, output)
      ├─ For each K_block, V_block:
      │   ├─ Compute scores: S = Q_block @ K_block^T
      │   ├─ Update max and renormalize previous
      │   ├─ Compute attention: exp(S - max) @ V_block
      │   └─ Accumulate to running output
      └─ Normalize final output
           ↓
    Concatenate Q_block outputs → [B, seq_len, num_heads, head_dim]
           ↓
    Reshape → [B, seq_len, dim]
           ↓
    Output_proj → Output [B, seq_len, dim]
    ```

    **Mathematical Operations**:
    1. **Projections**: Q = X W_q, K = X W_k, V = X W_v
    2. **Blocking**: Split along seq_len into blocks of size block_size
    3. **Online Softmax**: For each (Q_i, K_j, V_j) block pair:
       - S_ij = Q_i K_j^T / √d_k
       - m_new = max(m_prev, max(S_ij))
       - prev_renorm = exp(m_prev - m_new)
       - curr_weights = exp(S_ij - m_new)
       - output = prev_output * prev_renorm + curr_weights @ V_j
    4. **Final normalization**: output / sum_exp

    Args:
        dim: Integer, input/output dimension (embedding size). Must be positive and
            divisible by num_heads.
        num_heads: Integer, number of attention heads. Must be positive. Defaults to 8.
        block_size: Integer, sequence block size for processing. Larger blocks use
            more memory but may be more efficient. Must be positive. Defaults to 512.
        dropout_rate: Float, dropout rate applied to attention weights.
            Must be between 0.0 and 1.0. Defaults to 0.0.
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
        w_k: Dense layer for key projection with shape (dim, num_heads * head_dim).
        w_v: Dense layer for value projection with shape (dim, num_heads * head_dim).
        w_o: Dense layer for output projection with shape (num_heads * head_dim, dim).
        dropout: Dropout layer for attention weights.

    Call arguments:
        inputs: Input tensor of shape `(batch_size, seq_len, dim)`.
        training: Boolean indicating training or inference mode. Affects dropout.
        attention_mask: Optional attention mask of shape `(batch_size, seq_len, seq_len)`
            or `(batch_size, 1, seq_len, seq_len)`. Values should be 1 for
            positions to attend to and 0 for masked positions.
        return_attention_weights: Boolean, whether to return attention weights
            along with the output. Defaults to False. Note: Returns None for
            attention_weights due to blockwise computation.

    Returns:
        If return_attention_weights=False:
            Output tensor of shape `(batch_size, seq_len, dim)`
        If return_attention_weights=True:
            Tuple of (output, None) where None is returned for attention_weights
            as full attention matrix is not materialized in blockwise computation.

    Example:
        ```python
        # Basic usage for very long sequences
        ring_attn = RingAttention(
            dim=768,
            num_heads=12,
            block_size=1024,  # Process in 1K token blocks
            dropout_rate=0.1
        )

        # Process 100K token sequence
        long_inputs = keras.Input(shape=(100000, 768))
        outputs = ring_attn(long_inputs)

        # Advanced configuration
        ring_attn = RingAttention(
            dim=1024,
            num_heads=16,
            block_size=512,
            dropout_rate=0.05,
            kernel_regularizer=keras.regularizers.L2(1e-4)
        )

        # In a transformer decoder for long context
        class LongContextTransformerBlock(keras.layers.Layer):
            def __init__(self, dim, **kwargs):
                super().__init__(**kwargs)
                self.attention = RingAttention(
                    dim=dim, num_heads=16, block_size=1024
                )
                self.norm1 = keras.layers.LayerNormalization()
                self.norm2 = keras.layers.LayerNormalization()
                self.ffn = keras.Sequential([
                    keras.layers.Dense(dim * 4, activation='gelu'),
                    keras.layers.Dense(dim)
                ])

            def call(self, inputs):
                # Self-attention with Ring Attention
                normed = self.norm1(inputs)
                attn_out = self.attention(normed)
                x = inputs + attn_out

                # Feed-forward network
                normed = self.norm2(x)
                ffn_out = self.ffn(normed)
                return x + ffn_out

        # Multi-layer long context model
        def create_long_context_model(seq_len=1000000, dim=1024):
            inputs = keras.Input(shape=(seq_len, dim))
            x = inputs
            for i in range(12):  # 12 layers
                x = LongContextTransformerBlock(dim, name=f'layer_{i}')(x)
            return keras.Model(inputs, x)
        ```

    Raises:
        ValueError: If dim is not positive or not divisible by num_heads.
        ValueError: If num_heads or block_size are not positive.
        ValueError: If dropout_rate is not between 0.0 and 1.0.

    Note:
        This implementation focuses on memory efficiency for extremely long sequences.
        The block_size parameter controls the memory-compute tradeoff: larger blocks
        use more memory but may have better computational efficiency.

        For distributed training across multiple devices (as in the original paper),
        additional infrastructure would be needed to handle ring communication
        patterns between devices.

        The attention_weights return is None because Ring Attention doesn't
        materialize the full attention matrix, which is key to its memory efficiency.
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
        """
        Validate initialization parameters.

        Args:
            dim: Model dimension to validate.
            num_heads: Number of attention heads to validate.
            block_size: Block size for sequence processing to validate.
            dropout_rate: Dropout rate to validate.

        Raises:
            ValueError: If any parameter is invalid.
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

        # Always call parent build at the end
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None,
            attention_mask: Optional[keras.KerasTensor] = None,
            return_attention_weights: bool = False
    ) -> Union[keras.KerasTensor, Tuple[keras.KerasTensor, None]]:
        """
        Apply ring attention with blockwise processing.

        Args:
            inputs: Input tensor of shape (batch_size, seq_len, dim).
            training: Boolean, whether in training mode. Affects dropout behavior.
            attention_mask: Optional attention mask of shape (batch_size, seq_len, seq_len)
                or (batch_size, 1, seq_len, seq_len). Values should be 1 for
                positions to attend to and 0 for masked positions.
            return_attention_weights: Boolean, whether to return attention weights.
                Always returns None for attention weights due to blockwise processing.

        Returns:
            If return_attention_weights=False:
                Output tensor of shape (batch_size, seq_len, dim)
            If return_attention_weights=True:
                Tuple of (output, None) - None returned for attention_weights
                as full attention matrix is not materialized.
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
        """
        Core blockwise attention computation with online softmax.

        This implements the Ring Attention algorithm:
        1. Split sequence into blocks
        2. For each query block, process all key/value blocks incrementally
        3. Maintain running softmax statistics for numerical stability
        4. Accumulate attention outputs

        Args:
            queries: Query tensor of shape (batch, num_heads, seq_len, head_dim).
            keys: Key tensor of shape (batch, num_heads, seq_len, head_dim).
            values: Value tensor of shape (batch, num_heads, seq_len, head_dim).
            attention_mask: Optional mask for attention computation.
            training: Boolean indicating training mode.

        Returns:
            Attention output of shape (batch, num_heads, seq_len, head_dim).
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
            'block_size': self.block_size,
            'dropout_rate': self.dropout_rate,
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
        })
        return config