"""
Rotary Position Embedding (RoPE) Layer Implementation.

This module implements the Rotary Position Embedding mechanism introduced in the
RoFormer paper. RoPE provides a way to inject positional information into
transformer models by rotating query and key vectors using trigonometric functions,
allowing the model to naturally encode relative position relationships.

Mathematical Foundation:
=======================

RoPE applies a rotation matrix to each pair of dimensions in the input vector:

For a vector x = [x₀, x₁, x₂, x₃, ...], we group consecutive pairs and apply:

    [x₂ᵢ']   [cos(mθᵢ)  -sin(mθᵢ)] [x₂ᵢ]
    [x₂ᵢ₊₁'] = [sin(mθᵢ)   cos(mθᵢ)] [x₂ᵢ₊₁]

Where:
- m is the position index
- θᵢ = 1 / (10000^(2i/d)) is the frequency for dimension pair i
- d is the total dimensionality

The key insight is that this rotation preserves inner products between vectors
at different positions in a way that depends only on their relative positions,
not their absolute positions.

Implementation Details:
======================

1. **Partial Application**: Only applies RoPE to a fraction of dimensions
   (typically 25-50%) to maintain model stability. The remaining dimensions
   are passed through unchanged.

2. **Efficient Caching**: Pre-computes cos/sin tables for the maximum sequence
   length to avoid recomputation during forward passes.

3. **Complex Number Treatment**: Treats consecutive dimension pairs as complex
   numbers and applies rotation in the complex plane, which is mathematically
   equivalent to the 2D rotation matrix formulation.

4. **Backend Compatibility**: Uses keras.ops for cross-backend compatibility

Key Advantages:
==============

- **Relative Position Encoding**: Naturally encodes relative positions without
  requiring explicit position embeddings
- **Extrapolation**: Can handle sequences longer than those seen during training
- **Computational Efficiency**: Linear complexity with respect to sequence length
- **Translation Invariance**: Attention scores depend only on relative positions

Usage in Attention:
==================

RoPE is typically applied to query and key vectors before computing attention:

```python
# Apply RoPE to queries and keys
rope = RotaryPositionEmbedding(head_dim=64, max_seq_len=512)
q_rotated = rope(queries)  # Shape: (batch, heads, seq_len, head_dim)
k_rotated = rope(keys)     # Shape: (batch, heads, seq_len, head_dim)

# Compute attention with rotated Q and K
attention_scores = keras.ops.matmul(q_rotated, k_rotated, transpose_b=True)
```

References:
===========

1. Su, J., Lu, Y., Pan, S., Wen, B., & Liu, Y. (2021).
   "RoFormer: Enhanced Transformer with Rotary Position Embedding"
   arXiv:2104.09864
   https://arxiv.org/abs/2104.09864

2. Press, O., Smith, N. A., & Lewis, M. (2021).
   "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"
   arXiv:2108.12409
   https://arxiv.org/abs/2108.12409

3. Chen, S., Wong, S., Chen, L., & Tian, Y. (2023).
   "Extending Context Window of Large Language Models via Positional Interpolation"
   arXiv:2306.15595
   https://arxiv.org/abs/2306.15595
"""

import keras
from typing import Optional, Any, Tuple, Dict
from dl_techniques.utils.logger import logger


@keras.saving.register_keras_serializable()
class RotaryPositionEmbedding(keras.layers.Layer):
    """
    Rotary Position Embedding layer for attention mechanisms.

    Rotary Position Embedding (RoPE) integrates positional information by rotating
    query and key vectors in attention mechanisms. This allows the model to naturally
    encode relative position information without requiring explicit positional encodings.

    The layer applies rotary transformations to a portion of the input dimensions,
    typically 25-50%, while leaving the remaining dimensions unchanged for stability.

    Args:
        head_dim: Integer, the dimensionality of each attention head. Must be positive.
        max_seq_len: Integer, maximum sequence length for which to precompute
            rotary embeddings. Must be positive.
        rope_theta: Float, base frequency for rotary embedding computation.
            Default is 10000.0, following the original RoPE paper. Must be positive.
        rope_percentage: Float between 0.0 and 1.0, fraction of head dimensions
            to apply RoPE to. Default is 0.5 (50% of dimensions). Must be in (0, 1].
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        4D tensor with shape: `(batch_size, num_heads, seq_len, head_dim)`

    Output shape:
        4D tensor with shape: `(batch_size, num_heads, seq_len, head_dim)`
        Same as input shape.

    Raises:
        ValueError: If any parameter is invalid or incompatible.

    Example:
        ```python
        # Basic usage
        rope = RotaryPositionEmbedding(head_dim=64, max_seq_len=512)
        x = keras.ops.random.normal([2, 8, 128, 64])  # (batch, heads, seq, dim)
        output = rope(x)
        print(output.shape)  # (2, 8, 128, 64)

        # With custom parameters
        rope = RotaryPositionEmbedding(
            head_dim=128,
            max_seq_len=2048,
            rope_theta=50000.0,
            rope_percentage=0.25  # Apply to only 25% of dimensions
        )

        # In a transformer model
        inputs = keras.Input(shape=(None, 512, 64))  # (batch, heads, seq, head_dim)
        rope_layer = RotaryPositionEmbedding(head_dim=64, max_seq_len=2048)
        rotated = rope_layer(inputs)
        ```

    References:
        RoFormer: Enhanced Transformer with Rotary Position Embedding
        https://arxiv.org/abs/2104.09864

    Note:
        This implementation follows the modern Keras 3 pattern where weights
        are created in build() and configuration is stored in __init__.
        This ensures proper serialization and avoids common build errors.
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int,
        rope_theta: float = 10000.0,
        rope_percentage: float = 0.5,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        if rope_theta <= 0.0:
            raise ValueError(f"rope_theta must be positive, got {rope_theta}")
        if not 0.0 < rope_percentage <= 1.0:
            raise ValueError(f"rope_percentage must be in (0, 1], got {rope_percentage}")
        if head_dim % 2 != 0:
            logger.warning(f"head_dim ({head_dim}) is odd, RoPE works best with even dimensions")

        # Store ALL configuration arguments as instance attributes
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.rope_percentage = rope_percentage

        # Calculate derived parameters
        self.rope_dim = int(head_dim * rope_percentage)
        # Ensure rope_dim is even for proper complex number treatment
        if self.rope_dim % 2 != 0:
            self.rope_dim -= 1
            logger.info(f"Adjusted rope_dim to {self.rope_dim} to ensure even dimension")

        # Initialize weight attributes to None - they'll be created in build()
        self.cos_cached = None
        self.sin_cached = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create the layer's weights.

        This method is called automatically by Keras when the layer is first used.
        It creates the layer's cached cos/sin tables using add_weight().

        Args:
            input_shape: Shape tuple of the input tensor, including batch dimension.
                Expected: (batch_size, num_heads, seq_len, head_dim)

        Raises:
            ValueError: If input_shape is invalid or incompatible.
        """
        # Validate input shape
        if len(input_shape) != 4:
            raise ValueError(
                f"Expected 4D input (batch, heads, seq_len, head_dim), "
                f"got shape with {len(input_shape)} dimensions: {input_shape}"
            )

        input_head_dim = input_shape[-1]
        if input_head_dim is not None and input_head_dim != self.head_dim:
            raise ValueError(
                f"Input head_dim ({input_head_dim}) doesn't match "
                f"layer head_dim ({self.head_dim})"
            )

        # CREATE the layer's own weights using add_weight()
        self._build_rope_cache()

        # Let Keras know the build is complete
        super().build(input_shape)

    def _build_rope_cache(self) -> None:
        """Build and cache the cos/sin tables for rotary embeddings."""
        # Calculate the frequency dimension (half of rope_dim for complex pairs)
        freq_dim = self.rope_dim // 2

        if freq_dim == 0:
            logger.warning("rope_dim is too small, no rotary embedding will be applied")
            # Create dummy weights to avoid serialization issues
            self.cos_cached = self.add_weight(
                name='cos_cached',
                shape=(self.max_seq_len, 1),
                initializer='zeros',
                trainable=False,
                dtype='float32'
            )
            self.sin_cached = self.add_weight(
                name='sin_cached',
                shape=(self.max_seq_len, 1),
                initializer='zeros',
                trainable=False,
                dtype='float32'
            )
            return

        # Create frequency tensor: 1 / (theta ^ (2i / rope_dim)) for i in [0, freq_dim)
        inv_freq = 1.0 / (
            self.rope_theta ** (
                keras.ops.arange(0, freq_dim, dtype='float32') * 2.0 / self.rope_dim
            )
        )

        # Position indices: [0, 1, 2, ..., max_seq_len-1]
        positions = keras.ops.arange(self.max_seq_len, dtype='float32')

        # Outer product to get all position-frequency combinations
        # Shape: (max_seq_len, freq_dim)
        freqs = keras.ops.outer(positions, inv_freq)

        # Create cos and sin tables
        cos_table = keras.ops.cos(freqs)
        sin_table = keras.ops.sin(freqs)

        # Store as non-trainable weights for proper serialization
        self.cos_cached = self.add_weight(
            name='cos_cached',
            shape=cos_table.shape,
            initializer='zeros',
            trainable=False,
            dtype='float32'
        )
        self.sin_cached = self.add_weight(
            name='sin_cached',
            shape=sin_table.shape,
            initializer='zeros',
            trainable=False,
            dtype='float32'
        )

        # Assign the computed values
        self.cos_cached.assign(cos_table)
        self.sin_cached.assign(sin_table)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply rotary position embedding to input tensor.

        Args:
            inputs: Input tensor with shape (batch_size, num_heads, seq_len, head_dim)
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode. Not used in this layer but
                included for consistency.

        Returns:
            Output tensor with same shape as input, with RoPE applied.

        Raises:
            ValueError: If sequence length exceeds max_seq_len.
        """
        # Get sequence length from input
        seq_len = keras.ops.shape(inputs)[2]

        # Early return if no RoPE dimensions
        if self.rope_dim == 0:
            return inputs

        # Ensure sequence length doesn't exceed our cached values
        max_seq_len_tensor = keras.ops.convert_to_tensor(self.max_seq_len, dtype='int32')
        if keras.ops.any(seq_len > max_seq_len_tensor):
            raise ValueError(
                f"Input sequence length exceeds max_seq_len ({self.max_seq_len}). "
                f"Please increase max_seq_len or truncate the input."
            )

        return self._apply_rope(inputs, seq_len)

    def _apply_rope(self, x: keras.KerasTensor, seq_len: keras.KerasTensor) -> keras.KerasTensor:
        """
        Apply rotary position embedding to input tensor.

        Args:
            x: Input tensor with shape (batch_size, num_heads, seq_len, head_dim)
            seq_len: Current sequence length

        Returns:
            Tensor with RoPE applied to the first rope_dim dimensions
        """
        # Split into RoPE and pass-through dimensions
        x_rope = x[..., :self.rope_dim]  # Apply RoPE to these dimensions
        x_pass = x[..., self.rope_dim:]  # Pass these through unchanged

        # Get cached cos/sin values for current sequence length
        cos = self.cos_cached[:seq_len]  # Shape: (seq_len, rope_dim // 2)
        sin = self.sin_cached[:seq_len]  # Shape: (seq_len, rope_dim // 2)

        # Reshape x_rope to separate complex pairs
        # From: (batch, heads, seq_len, rope_dim)
        # To: (batch, heads, seq_len, rope_dim // 2, 2)
        rope_pairs = self.rope_dim // 2

        if rope_pairs == 0:
            # If no pairs to process, return original input
            return x

        # Get the dynamic shape and construct new shape
        input_shape = keras.ops.shape(x_rope)
        batch_size = input_shape[0]
        num_heads = input_shape[1]
        seq_len_dynamic = input_shape[2]

        # Create new shape for reshaping
        new_shape = [batch_size, num_heads, seq_len_dynamic, rope_pairs, 2]
        x_rope_reshaped = keras.ops.reshape(x_rope, new_shape)

        # Extract the two elements of each complex pair
        x1 = x_rope_reshaped[..., 0]  # Real-like component
        x2 = x_rope_reshaped[..., 1]  # Imaginary-like component

        # Apply rotary transformation:
        # [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
        rotated_1 = x1 * cos - x2 * sin
        rotated_2 = x1 * sin + x2 * cos

        # Stack the rotated components back together
        x_rope_rotated = keras.ops.stack([rotated_1, rotated_2], axis=-1)

        # Reshape back to original rope dimensions
        # From: (batch, heads, seq_len, rope_pairs, 2)
        # To: (batch, heads, seq_len, rope_dim)
        x_rope_rotated = keras.ops.reshape(
            x_rope_rotated,
            [batch_size, num_heads, seq_len_dynamic, self.rope_dim]
        )

        # Concatenate rotated and pass-through dimensions
        if self.rope_dim < self.head_dim:
            return keras.ops.concatenate([x_rope_rotated, x_pass], axis=-1)
        else:
            return x_rope_rotated

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape tuple of the input tensor.

        Returns:
            Output shape tuple (same as input shape).
        """
        # RoPE doesn't change the shape, just applies rotational transformations
        return tuple(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Return the layer configuration for serialization.

        This method must return ALL arguments needed to recreate the layer
        via __init__.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            'head_dim': self.head_dim,
            'max_seq_len': self.max_seq_len,
            'rope_theta': self.rope_theta,
            'rope_percentage': self.rope_percentage,
        })
        return config

    # Note: No get_build_config or build_from_config needed.
    # Keras handles the build lifecycle automatically in modern Keras 3.