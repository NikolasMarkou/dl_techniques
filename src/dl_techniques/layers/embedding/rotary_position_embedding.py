"""
Rotary Position Embedding (RoPE) Layer Implementation for Keras 3.x

This module implements the Rotary Position Embedding mechanism introduced in the
RoFormer paper, providing positional information injection through vector rotation
using trigonometric functions for relative position encoding in transformer models.
"""

import keras
from typing import Optional, Any, Tuple, Dict

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class RotaryPositionEmbedding(keras.layers.Layer):
    """
    Rotary Position Embedding layer for transformer attention mechanisms.

    This layer applies rotary transformations to query and key vectors in attention
    mechanisms, encoding relative positional information through trigonometric
    rotation of vector pairs. RoPE enables natural relative position encoding
    without explicit positional embeddings and supports sequence length extrapolation.

    **Intent**: Provide efficient and mathematically sound positional encoding
    for transformer models through rotary embeddings that preserve relative
    position relationships and enable length generalization.

    **Architecture**:
    ```
    Input(shape=[batch, heads, seq_len, head_dim])
           ↓
    Split: RoPE_dims | Pass_dims
           ↓              ↓
    Rotate(cos/sin)   Identity
           ↓              ↓
    Concat: Output(shape=[batch, heads, seq_len, head_dim])
    ```

    **Mathematical Operation**:
        For vector pairs [x₂ᵢ, x₂ᵢ₊₁], applies rotation:
        ```
        [x'₂ᵢ]   [cos(mθᵢ)  -sin(mθᵢ)] [x₂ᵢ]
        [x'₂ᵢ₊₁] = [sin(mθᵢ)   cos(mθᵢ)] [x₂ᵢ₊₁]
        ```
        where m is position index and θᵢ = 1/(10000^(2i/d))

    **Key Features**:
    - Relative position encoding without learned parameters
    - Sequence length extrapolation capability
    - Partial application to maintain model stability
    - Efficient cos/sin table caching
    - Cross-backend compatibility via keras.ops

    Args:
        head_dim: Integer, dimensionality of each attention head. Must be positive.
        max_seq_len: Integer, maximum sequence length for precomputing rotary tables.
            Must be positive.
        rope_theta: Float, base frequency for rotary computation. Higher values
            work better for longer sequences. Defaults to 10000.0. Must be positive.
        rope_percentage: Float between 0.0 and 1.0, fraction of head dimensions
            to apply RoPE to. Defaults to 0.5 for stability. Must be in (0, 1].
        **kwargs: Additional Layer base class arguments.

    Input shape:
        4D tensor with shape: `(batch_size, num_heads, seq_len, head_dim)`

    Output shape:
        4D tensor with shape: `(batch_size, num_heads, seq_len, head_dim)`
        Same as input shape.

    Attributes:
        cos_cached: Non-trainable weight containing cached cosine values.
        sin_cached: Non-trainable weight containing cached sine values.

    Example:
        ```python
        # Basic usage for attention heads
        rope = RotaryPositionEmbedding(head_dim=64, max_seq_len=512)
        queries = keras.random.normal([2, 8, 128, 64])  # (batch, heads, seq, dim)
        keys = keras.random.normal([2, 8, 128, 64])

        q_rotated = rope(queries)
        k_rotated = rope(keys)

        # Use rotated Q/K in attention computation
        attention_scores = keras.ops.matmul(q_rotated, k_rotated, transpose_b=True)

        # Custom configuration for longer sequences
        rope_long = RotaryPositionEmbedding(
            head_dim=128,
            max_seq_len=2048,
            rope_theta=50000.0,      # Higher theta for better extrapolation
            rope_percentage=0.25     # Apply to fewer dimensions
        )
        ```

    Raises:
        ValueError: If head_dim, max_seq_len, or rope_theta are not positive.
        ValueError: If rope_percentage is not in the range (0, 1].
        ValueError: If input sequence length exceeds max_seq_len.

    Note:
        This layer creates cos/sin lookup tables for efficiency. For very long
        sequences, consider position interpolation techniques or increasing
        rope_theta for better extrapolation behavior.

    References:
        RoFormer: Enhanced Transformer with Rotary Position Embedding
        https://arxiv.org/abs/2104.09864
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
        if rope_theta <= 0:
            raise ValueError(f"rope_theta must be positive, got {rope_theta}")
        if not 0.0 < rope_percentage <= 1.0:
            raise ValueError(f"rope_percentage must be in (0, 1], got {rope_percentage}")

        if head_dim % 2 != 0:
            logger.warning(f"head_dim ({head_dim}) is odd, RoPE works best with even dimensions")

        # Store ALL configuration parameters
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.rope_percentage = rope_percentage

        # Calculate RoPE dimensions (ensure even for proper complex pairs)
        self.rope_dim = int(head_dim * rope_percentage)
        if self.rope_dim % 2 != 0:
            self.rope_dim -= 1
            logger.info(f"Adjusted rope_dim to {self.rope_dim} to ensure even dimension")

        # Initialize weight attributes - created in build()
        self.cos_cached = None
        self.sin_cached = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create the layer's cos/sin lookup tables.

        This is called automatically when the layer first processes input.
        Creates efficient cos/sin cache tables for RoPE computation.

        Args:
            input_shape: Expected shape (batch_size, num_heads, seq_len, head_dim)

        Raises:
            ValueError: If input shape is not 4D or head_dim doesn't match.
        """
        # Validate input shape
        if len(input_shape) != 4:
            raise ValueError(
                f"Expected 4D input (batch, heads, seq_len, head_dim), "
                f"got shape with {len(input_shape)} dimensions: {input_shape}"
            )

        input_head_dim = input_shape[-1]
        if input_head_dim != self.head_dim:
            raise ValueError(
                f"Input head_dim ({input_head_dim}) doesn't match "
                f"layer head_dim ({self.head_dim})"
            )

        # Create layer's own weights - cos/sin cache tables
        self._create_rope_cache()

        # Always call parent build at the end
        super().build(input_shape)

    def _create_rope_cache(self) -> None:
        """Create and cache cos/sin lookup tables for rotary embeddings."""
        # Calculate frequency dimension (half of rope_dim for complex pairs)
        freq_dim = self.rope_dim // 2

        if freq_dim == 0:
            logger.warning("rope_dim is too small, no rotary embedding will be applied")
            # Create dummy weights to maintain consistency
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

        # Compute cos and sin tables
        cos_table = keras.ops.cos(freqs)
        sin_table = keras.ops.sin(freqs)

        # Create layer's own weights using add_weight
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
        """Apply rotary position embedding to input tensor.

        Args:
            inputs: Input tensor with shape (batch_size, num_heads, seq_len, head_dim).
            training: Boolean indicating training mode (not used in this layer).

        Returns:
            Output tensor with same shape, RoPE applied to first rope_dim dimensions.

        Raises:
            ValueError: If input sequence length exceeds max_seq_len.
        """
        # Get sequence length from input
        seq_len = keras.ops.shape(inputs)[2]

        # Early return if no RoPE dimensions
        if self.rope_dim == 0:
            return inputs

        # Validate sequence length at build time if known
        seq_len_static = inputs.shape[2]
        if seq_len_static is not None and seq_len_static > self.max_seq_len:
            raise ValueError(
                f"Input sequence length ({seq_len_static}) exceeds max_seq_len ({self.max_seq_len}). "
                f"Please increase max_seq_len or truncate the input."
            )

        return self._apply_rope_rotation(inputs, seq_len)

    def _apply_rope_rotation(
        self,
        x: keras.KerasTensor,
        seq_len: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Apply rotary position embedding transformation.

        Args:
            x: Input tensor with shape (batch_size, num_heads, seq_len, head_dim)
            seq_len: Current sequence length tensor

        Returns:
            Tensor with RoPE applied to the first rope_dim dimensions
        """
        # Split into RoPE and pass-through dimensions
        x_rope = x[..., :self.rope_dim]     # Apply RoPE to these dimensions
        x_pass = x[..., self.rope_dim:]     # Pass these through unchanged

        # Get cached cos/sin values for current sequence length
        cos = self.cos_cached[:seq_len]     # Shape: (seq_len, rope_dim // 2)
        sin = self.sin_cached[:seq_len]     # Shape: (seq_len, rope_dim // 2)

        # Reshape x_rope to separate complex pairs
        # From: (batch, heads, seq_len, rope_dim)
        # To: (batch, heads, seq_len, rope_dim // 2, 2)
        rope_pairs = self.rope_dim // 2

        # Get dynamic shape and construct new shape for reshaping
        input_shape = keras.ops.shape(x_rope)
        batch_size = input_shape[0]
        num_heads = input_shape[1]
        seq_len_dynamic = input_shape[2]

        # Reshape to expose complex pairs
        new_shape = [batch_size, num_heads, seq_len_dynamic, rope_pairs, 2]
        x_rope_reshaped = keras.ops.reshape(x_rope, new_shape)

        # Extract real and imaginary components of each complex pair
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
        """Compute output shape - identical to input shape.

        Args:
            input_shape: Shape of the input tensor.

        Returns:
            Output shape tuple (same as input shape).
        """
        # RoPE preserves tensor shape while applying rotational transformations
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        Returns:
            Dictionary containing ALL __init__ parameters for proper serialization.
        """
        config = super().get_config()
        config.update({
            'head_dim': self.head_dim,
            'max_seq_len': self.max_seq_len,
            'rope_theta': self.rope_theta,
            'rope_percentage': self.rope_percentage,
        })
        return config

# ---------------------------------------------------------------------