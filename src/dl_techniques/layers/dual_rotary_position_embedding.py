"""
Dual Rotary Position Embedding (RoPE) Layer Implementation for Gemma3-style Models.

This module implements the dual RoPE mechanism used in Gemma3 models, where two
different RoPE configurations are maintained:
- Global RoPE: Higher theta_base for full attention (long-range dependencies)
- Local RoPE: Lower theta_base for sliding attention (local dependencies)

The layer pre-computes both sets of cos/sin tables and applies the appropriate
one based on the rope_type parameter during the forward pass.

Mathematical Foundation:
=======================

The dual RoPE applies the same rotary transformation as standard RoPE, but with
two different frequency bases:

For Global RoPE (full attention):
    θᵢ = 1 / (1,000,000^(2i/d)) - slower frequency changes, better for long sequences

For Local RoPE (sliding attention):
    θᵢ = 1 / (10,000^(2i/d)) - faster frequency changes, better for local patterns

The rotation is applied as:
    [x₂ᵢ']   [cos(mθᵢ)  -sin(mθᵢ)] [x₂ᵢ]
    [x₂ᵢ₊₁'] = [sin(mθᵢ)   cos(mθᵢ)] [x₂ᵢ₊₁]

Where m is the position index and i is the dimension pair index.

References:
===========
1. Gemma3 Technical Report: https://storage.googleapis.com/deepmind-media/gemma/gemma-3-report.pdf
2. RoFormer: Enhanced Transformer with Rotary Position Embedding
   https://arxiv.org/abs/2104.09864
"""

import keras
from typing import Optional, Any, Tuple, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

RopeType = Literal['global', 'local']


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class DualRotaryPositionEmbedding(keras.layers.Layer):
    """Dual Rotary Position Embedding layer for Gemma3-style attention mechanisms.

    This layer implements dual RoPE configurations as used in Gemma3 models:
    - Global RoPE: Higher theta_base (typically 1,000,000) for full attention
    - Local RoPE: Lower theta_base (typically 10,000) for sliding attention

    The layer pre-computes both sets of cos/sin tables and applies the appropriate
    one based on the rope_type parameter during the forward pass.

    Args:
        head_dim: Integer, the dimensionality of each attention head. Must be positive and even.
        max_seq_len: Integer, maximum sequence length for which to precompute
            rotary embeddings. Must be positive.
        global_theta_base: Float, base frequency for global RoPE computation.
            Higher values provide better long-range modeling. Default is 1_000_000.0.
        local_theta_base: Float, base frequency for local RoPE computation.
            Lower values provide better local pattern modeling. Default is 10_000.0.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        4D tensor with shape: `(batch_size, num_heads, seq_len, head_dim)`

    Output shape:
        4D tensor with shape: `(batch_size, num_heads, seq_len, head_dim)`
        Same as input shape.

    Example:
        ```python
        # Create dual RoPE layer
        dual_rope = DualRotaryPositionEmbedding(
            head_dim=256,
            max_seq_len=4096
        )

        # Apply global RoPE for full attention
        queries = keras.random.normal([2, 8, 512, 256])
        keys = keras.random.normal([2, 8, 512, 256])

        q_global = dual_rope(queries, rope_type='global')
        k_global = dual_rope(keys, rope_type='global')

        # Apply local RoPE for sliding attention
        q_local = dual_rope(queries, rope_type='local')
        k_local = dual_rope(keys, rope_type='local')
        ```

    Note:
        This implementation follows the modern Keras 3 pattern where weights
        are created in build() and configuration is stored in __init__().
    """

    def __init__(
            self,
            head_dim: int,
            max_seq_len: int,
            global_theta_base: float = 1_000_000.0,
            local_theta_base: float = 10_000.0,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}")
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        if global_theta_base <= 0:
            raise ValueError(f"global_theta_base must be positive, got {global_theta_base}")
        if local_theta_base <= 0:
            raise ValueError(f"local_theta_base must be positive, got {local_theta_base}")

        # Store configuration - NO weights created here
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.global_theta_base = global_theta_base
        self.local_theta_base = local_theta_base

        # Initialize weight attributes - created in build()
        self.cos_global_cached = None
        self.sin_global_cached = None
        self.cos_local_cached = None
        self.sin_local_cached = None

        # Store build shape for serialization
        self._build_input_shape = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer by creating cos/sin tables for both RoPE configurations.

        Args:
            input_shape: Shape tuple indicating the input shape of the layer.
                Expected: (batch_size, num_heads, seq_len, head_dim)
        """
        # Store input shape for serialization
        self._build_input_shape = input_shape

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

        # Build both RoPE caches
        self._build_global_rope_cache()
        self._build_local_rope_cache()

        super().build(input_shape)

    def _build_rope_cache(self, theta_base: float, cache_prefix: str) -> Tuple[Any, Any]:
        """Build cos/sin cache for a given theta_base.

        Args:
            theta_base: Base frequency for RoPE computation.
            cache_prefix: Prefix for the cache weight names ('global' or 'local').

        Returns:
            Tuple of (cos_cache, sin_cache) weights.
        """
        # Calculate the frequency dimension (half of head_dim for complex pairs)
        freq_dim = self.head_dim // 2

        # Create frequency tensor: 1 / (theta_base ^ (2i / head_dim)) for i in [0, freq_dim)
        inv_freq = 1.0 / (
                theta_base ** (
                keras.ops.arange(0, freq_dim, dtype='float32') * 2.0 / self.head_dim
        )
        )

        # Position indices: [0, 1, 2, ..., max_seq_len-1]
        positions = keras.ops.arange(self.max_seq_len, dtype='float32')

        # Outer product to get all position-frequency combinations
        # Shape: (max_seq_len, freq_dim)
        freqs = keras.ops.outer(positions, inv_freq)

        # Duplicate frequencies to match head_dim (as done in original Gemma3)
        # Shape: (max_seq_len, head_dim)
        freqs_full = keras.ops.concatenate([freqs, freqs], axis=1)

        # Create cos and sin tables
        cos_table = keras.ops.cos(freqs_full)
        sin_table = keras.ops.sin(freqs_full)

        # Store as non-trainable weights for proper serialization
        cos_cache = self.add_weight(
            name=f'cos_{cache_prefix}_cached',
            shape=cos_table.shape,
            initializer='zeros',
            trainable=False,
            dtype='float32'
        )
        sin_cache = self.add_weight(
            name=f'sin_{cache_prefix}_cached',
            shape=sin_table.shape,
            initializer='zeros',
            trainable=False,
            dtype='float32'
        )

        # Assign the computed values
        cos_cache.assign(cos_table)
        sin_cache.assign(sin_table)

        return cos_cache, sin_cache

    def _build_global_rope_cache(self) -> None:
        """Build cos/sin cache for global RoPE configuration."""
        self.cos_global_cached, self.sin_global_cached = self._build_rope_cache(
            theta_base=self.global_theta_base,
            cache_prefix='global'
        )

    def _build_local_rope_cache(self) -> None:
        """Build cos/sin cache for local RoPE configuration."""
        self.cos_local_cached, self.sin_local_cached = self._build_rope_cache(
            theta_base=self.local_theta_base,
            cache_prefix='local'
        )

    def call(
            self,
            inputs: Any,
            rope_type: RopeType = 'global',
            training: Optional[bool] = None,
            **kwargs: Any
    ) -> Any:
        """Apply dual rotary position embedding to input tensor.

        Args:
            inputs: Input tensor with shape (batch_size, num_heads, seq_len, head_dim)
            rope_type: Type of RoPE to apply - 'global' for full attention,
                'local' for sliding attention. Defaults to 'global'.
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.

        Returns:
            Output tensor with same shape as input, with appropriate RoPE applied.

        Raises:
            ValueError: If rope_type is not 'global' or 'local', or if sequence
                length exceeds max_seq_len.
        """
        # Validate rope_type
        if rope_type not in ['global', 'local']:
            raise ValueError(f"rope_type must be 'global' or 'local', got {rope_type}")

        # Get sequence length from input
        seq_len = keras.ops.shape(inputs)[2]

        # Ensure sequence length doesn't exceed our cached values
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Input sequence length ({seq_len}) exceeds max_seq_len ({self.max_seq_len}). "
                f"Please increase max_seq_len or truncate the input."
            )

        return self._apply_rope(inputs, seq_len, rope_type)

    def _apply_rope(self, x: Any, seq_len: int, rope_type: RopeType) -> Any:
        """Apply rotary position embedding using the specified RoPE type.

        Args:
            x: Input tensor with shape (batch_size, num_heads, seq_len, head_dim)
            seq_len: Current sequence length
            rope_type: Type of RoPE to apply ('global' or 'local')

        Returns:
            Tensor with RoPE applied using the appropriate configuration
        """
        # Select appropriate cos/sin caches
        if rope_type == 'global':
            cos = self.cos_global_cached[:seq_len]  # Shape: (seq_len, head_dim)
            sin = self.sin_global_cached[:seq_len]  # Shape: (seq_len, head_dim)
        else:  # rope_type == 'local'
            cos = self.cos_local_cached[:seq_len]  # Shape: (seq_len, head_dim)
            sin = self.sin_local_cached[:seq_len]  # Shape: (seq_len, head_dim)

        # Apply RoPE transformation following Gemma3 approach:
        # Split into two halves and apply rotation
        x1 = x[..., : self.head_dim // 2]  # First half
        x2 = x[..., self.head_dim // 2:]  # Second half

        # Expand cos/sin to match input batch and head dimensions
        # From: (seq_len, head_dim) -> (1, 1, seq_len, head_dim)
        cos = keras.ops.expand_dims(keras.ops.expand_dims(cos, 0), 0)
        sin = keras.ops.expand_dims(keras.ops.expand_dims(sin, 0), 0)

        # Apply the rotary transformation: rotation matrix in 2D
        # [x1', x2'] = [x1*cos - x2*sin, x1*sin + x2*cos]
        # But we need to rearrange due to the way frequencies are duplicated

        # Create rotated version: [-x2, x1] (equivalent to 90-degree rotation)
        rotated = keras.ops.concatenate([-x2, x1], axis=-1)

        # Apply final rotation: x*cos + rotated*sin
        x_rotated = (x * cos) + (rotated * sin)

        # Ensure output dtype matches input
        return keras.ops.cast(x_rotated, x.dtype)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input tensor.

        Returns:
            Output shape tuple (same as input shape).
        """
        # Dual RoPE doesn't change the shape, just applies rotational transformations
        return input_shape

    def get_config(self) -> dict[str, Any]:
        """Return the layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            'head_dim': self.head_dim,
            'max_seq_len': self.max_seq_len,
            'global_theta_base': self.global_theta_base,
            'local_theta_base': self.local_theta_base,
        })
        return config

    def get_build_config(self) -> dict[str, Any]:
        """Get the build configuration for proper serialization.

        Returns:
            Dictionary containing the build configuration.
        """
        return {
            'input_shape': self._build_input_shape,
        }

    def build_from_config(self, config: dict[str, Any]) -> None:
        """Build the layer from a config created with get_build_config.

        Args:
            config: Dictionary containing the build configuration.
        """
        if config.get('input_shape') is not None:
            self.build(config['input_shape'])

# ---------------------------------------------------------------------