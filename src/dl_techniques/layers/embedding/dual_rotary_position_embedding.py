"""
A dual-configuration Rotary Position Embedding (RoPE).

This layer provides two distinct, pre-computed Rotary Position Embedding
(RoPE) configurations within a single module. It is specifically designed
for transformer architectures like Gemma3 that employ hybrid attention
strategies, such as combining standard full attention with local or sliding
window attention.

Architecture:
    The core of this layer is not one, but two separate, non-trainable RoPE
    lookup tables (cosine and sine pairs). Each table is generated using a
    different frequency base (`theta_base`), resulting in two distinct sets
    of positional signals optimized for different contextual scales:

    1.  **Global RoPE:** Generated with a large `theta_base` (e.g.,
        1,000,000), this configuration produces low-frequency, long-
        wavelength positional signals. These signals change slowly across
        the sequence, making them ideal for encoding positions in a full,
        all-to-all attention mechanism where capturing long-range
        dependencies is critical.

    2.  **Local RoPE:** Generated with a smaller `theta_base` (e.g.,
        10,000), this configuration produces higher-frequency, shorter-
        wavelength signals. This provides more fine-grained positional
        discrimination for nearby tokens, making it better suited for
        local or sliding window attention where relative positions within a
        small context are most important.

    During the forward pass, the user selects either the 'global' or 'local'
    configuration at runtime. The appropriate pre-computed cosine and sine
    tables are then used to apply the rotational transformation to the input
    query or key vectors.

Foundational Mathematics:
    Rotary Position Embedding encodes the absolute position `m` by applying a
    rotation to a feature vector, with the crucial property that the inner
    product between two rotated vectors depends only on their relative
    position `m - k`. The rotation angle for each feature pair is `m * θ_i`,
    where the frequencies `θ_i` are defined by a geometric progression:

        θ_i = 1 / (theta_base^(2i / d))

    The `theta_base` parameter directly controls the wavelength of the
    positional signal. A larger `theta_base` leads to smaller `θ_i` values
    (lower frequencies), which means the positional signal varies more
    slowly across the sequence. A smaller `theta_base` results in larger
    `θ_i` values (higher frequencies), causing the signal to change more
    rapidly.

    This dual-RoPE layer leverages this principle by maintaining two sets of
    frequencies, `{θ_i}_global` and `{θ_i}_local`, derived from
    `global_theta_base` and `local_theta_base` respectively. This allows
    the model to switch between a positional encoding optimized for stable,
    long-distance relationships and one optimized for precise, local-context
    relationships, aligning the nature of the positional signal with the
    scope of the attention mechanism being used.

References:
    - The dual RoPE mechanism is a key component of the Gemma3 architecture.
      Google (2024). "Gemma 3 Technical Report".

    - The original concept for Rotary Position Embedding:
      Su, J., et al. (2021). "RoFormer: Enhanced Transformer with Rotary
      Position Embedding".
"""

import keras
from typing import Optional, Any, Tuple, Literal, Dict

# ---------------------------------------------------------------------

RopeType = Literal['global', 'local']

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class DualRotaryPositionEmbedding(keras.layers.Layer):
    """Dual-frequency Rotary Position Embedding for hybrid attention architectures.

    Maintains two separate RoPE configurations with different frequency bases
    for Gemma3-style models that combine global full attention with local
    sliding-window attention. Global RoPE uses a large ``theta_base`` (e.g.
    1,000,000) producing low-frequency signals for long-range dependencies,
    while local RoPE uses a smaller ``theta_base`` (e.g. 10,000) for
    fine-grained nearby-token discrimination. At runtime the caller selects
    ``'global'`` or ``'local'`` to pick the appropriate cos/sin tables for
    the rotation ``x' = x * cos(m * theta) + rotate(x) * sin(m * theta)``.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────────────┐
        │  Input (batch, heads, seq_len, head_dim)  │
        └───────────────────┬──────────────────────┘
                            ▼
        ┌───────────────────────────────────────────┐
        │  Select rope_type: 'global' | 'local'     │
        └───────────┬───────────────┬───────────────┘
                    ▼               ▼
        ┌───────────────┐  ┌────────────────┐
        │  Global RoPE  │  │  Local RoPE    │
        │  θ=1/Θ_g^(.)  │  │  θ=1/Θ_l^(.)  │
        │  (low freq)   │  │  (high freq)   │
        └───────┬───────┘  └────────┬───────┘
                └───────────┬───────┘
                            ▼
        ┌──────────────────────────────────────────┐
        │  Split: x1=[:d/2], x2=[d/2:]             │
        │  rotated = [-x2, x1]                      │
        │  output = x * cos + rotated * sin          │
        └───────────────────┬──────────────────────┘
                            ▼
        ┌──────────────────────────────────────────┐
        │  Output (batch, heads, seq_len, head_dim) │
        └──────────────────────────────────────────┘

    :param head_dim: Dimensionality of each attention head. Must be positive
        and even for proper complex pair rotation.
    :type head_dim: int
    :param max_seq_len: Maximum sequence length for precomputing tables. Must
        be positive.
    :type max_seq_len: int
    :param global_theta_base: Base frequency for global RoPE. Higher values
        provide better long-range modeling. Defaults to ``1,000,000.0``.
    :type global_theta_base: float
    :param local_theta_base: Base frequency for local RoPE. Lower values
        provide better local pattern modeling. Defaults to ``10,000.0``.
    :type local_theta_base: float
    :param kwargs: Additional Layer base class arguments.

    :raises ValueError: If ``head_dim`` is not positive or even.
    :raises ValueError: If ``max_seq_len`` is not positive.
    :raises ValueError: If ``theta_base`` values are not positive.
    :raises ValueError: If ``rope_type`` is not ``'global'`` or ``'local'``.
    :raises ValueError: If input sequence length exceeds ``max_seq_len``.
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

        # Store ALL configuration parameters
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.global_theta_base = global_theta_base
        self.local_theta_base = local_theta_base

        # Initialize weight attributes - created in build()
        self.cos_global_cached = None
        self.sin_global_cached = None
        self.cos_local_cached = None
        self.sin_local_cached = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create cos/sin lookup tables for both global and local RoPE.

        :param input_shape: Expected shape
            ``(batch_size, num_heads, seq_len, head_dim)``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If input shape is not 4D or ``head_dim`` does not
            match.
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

        # Create both RoPE cache tables
        self._create_global_rope_cache()
        self._create_local_rope_cache()

        # Always call parent build at the end
        super().build(input_shape)

    def _create_rope_cache_tables(self, theta_base: float, cache_prefix: str) -> Tuple[Any, Any]:
        """Create cos/sin cache tables for a given theta_base configuration.

        :param theta_base: Base frequency for RoPE computation.
        :type theta_base: float
        :param cache_prefix: Prefix for cache weight names
            (``'global'`` or ``'local'``).
        :type cache_prefix: str
        :return: Tuple of ``(cos_cache, sin_cache)`` weights.
        :rtype: Tuple[Any, Any]
        """
        # Calculate frequency dimension (half of head_dim for complex pairs)
        freq_dim = self.head_dim // 2

        # Create frequency tensor: 1 / (theta_base ^ (2i / head_dim))
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

        # Duplicate frequencies to match full head_dim (Gemma3 approach)
        # Shape: (max_seq_len, head_dim)
        freqs_full = keras.ops.concatenate([freqs, freqs], axis=1)

        # Compute cos and sin tables
        cos_table = keras.ops.cos(freqs_full)
        sin_table = keras.ops.sin(freqs_full)

        # Create layer weights using add_weight for proper serialization
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

    def _create_global_rope_cache(self) -> None:
        """Create cos/sin cache for global RoPE configuration."""
        self.cos_global_cached, self.sin_global_cached = self._create_rope_cache_tables(
            theta_base=self.global_theta_base,
            cache_prefix='global'
        )

    def _create_local_rope_cache(self) -> None:
        """Create cos/sin cache for local RoPE configuration."""
        self.cos_local_cached, self.sin_local_cached = self._create_rope_cache_tables(
            theta_base=self.local_theta_base,
            cache_prefix='local'
        )

    def call(
        self,
        inputs: keras.KerasTensor,
        rope_type: RopeType = 'global',
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply dual rotary position embedding to input tensor.

        :param inputs: Input tensor with shape
            ``(batch_size, num_heads, seq_len, head_dim)``.
        :type inputs: keras.KerasTensor
        :param rope_type: Type of RoPE to apply. ``'global'`` for full
            attention with long-range modeling, ``'local'`` for sliding
            attention with local patterns. Defaults to ``'global'``.
        :type rope_type: RopeType
        :param training: Whether in training mode (unused by this layer).
        :type training: Optional[bool]
        :return: Output tensor with same shape, appropriate RoPE transformation
            applied.
        :rtype: keras.KerasTensor
        :raises ValueError: If ``rope_type`` is invalid or sequence length
            exceeds ``max_seq_len``.
        """
        # Validate rope_type
        if rope_type not in ['global', 'local']:
            raise ValueError(f"rope_type must be 'global' or 'local', got '{rope_type}'")

        # Get sequence length from input
        seq_len = keras.ops.shape(inputs)[2]

        # Validate sequence length at build time if known
        seq_len_static = inputs.shape[2]
        if seq_len_static is not None and seq_len_static > self.max_seq_len:
            raise ValueError(
                f"Input sequence length ({seq_len_static}) exceeds max_seq_len ({self.max_seq_len}). "
                f"Please increase max_seq_len or truncate the input."
            )

        return self._apply_dual_rope_rotation(inputs, seq_len, rope_type)

    def _apply_dual_rope_rotation(
        self,
        x: keras.KerasTensor,
        seq_len: keras.KerasTensor,
        rope_type: RopeType
    ) -> keras.KerasTensor:
        """Apply rotary position embedding using the selected RoPE configuration.

        :param x: Input tensor with shape
            ``(batch_size, num_heads, seq_len, head_dim)``.
        :type x: keras.KerasTensor
        :param seq_len: Current sequence length tensor.
        :type seq_len: keras.KerasTensor
        :param rope_type: Type of RoPE to apply (``'global'`` or ``'local'``).
        :type rope_type: RopeType
        :return: Tensor with appropriate RoPE transformation applied.
        :rtype: keras.KerasTensor
        """
        # Select appropriate cos/sin caches based on rope_type
        if rope_type == 'global':
            cos = self.cos_global_cached[:seq_len]  # Shape: (seq_len, head_dim)
            sin = self.sin_global_cached[:seq_len]  # Shape: (seq_len, head_dim)
        else:  # rope_type == 'local'
            cos = self.cos_local_cached[:seq_len]   # Shape: (seq_len, head_dim)
            sin = self.sin_local_cached[:seq_len]   # Shape: (seq_len, head_dim)

        # Apply RoPE transformation following Gemma3 approach
        # Split into two halves for rotation
        half_dim = self.head_dim // 2
        x1 = x[..., :half_dim]     # First half
        x2 = x[..., half_dim:]     # Second half

        # Expand cos/sin to match input batch and head dimensions
        # From: (seq_len, head_dim) -> (1, 1, seq_len, head_dim)
        cos = keras.ops.expand_dims(keras.ops.expand_dims(cos, 0), 0)
        sin = keras.ops.expand_dims(keras.ops.expand_dims(sin, 0), 0)

        # Apply rotary transformation using complex rotation approach
        # Create rotated version: [-x2, x1] (90-degree rotation)
        rotated = keras.ops.concatenate([-x2, x1], axis=-1)

        # Apply final rotation: x*cos + rotated*sin
        x_rotated = (x * cos) + (rotated * sin)

        # Ensure output dtype matches input dtype
        return keras.ops.cast(x_rotated, x.dtype)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape (identical to input shape).

        :param input_shape: Shape of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple (same as input shape).
        :rtype: Tuple[Optional[int], ...]
        """
        # Dual RoPE preserves tensor shape while applying rotational transformations
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        :return: Dictionary containing all ``__init__`` parameters for proper
            serialization.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'head_dim': self.head_dim,
            'max_seq_len': self.max_seq_len,
            'global_theta_base': self.global_theta_base,
            'local_theta_base': self.local_theta_base,
        })
        return config

# ---------------------------------------------------------------------
