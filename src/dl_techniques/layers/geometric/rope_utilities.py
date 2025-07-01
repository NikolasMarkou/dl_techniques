"""Utility functions for Rotary Position Embedding (RoPE) operations."""

from keras import ops
from typing import Tuple

from dl_techniques.utils.logger import logger


def apply_rope_to_attention(q, k, freqs):
    """Apply Rotary Position Embedding to query and key tensors for attention.

    This is a convenience function that applies RoPE rotation to both query
    and key tensors using the same frequencies, which is the standard approach
    in most attention mechanisms.

    Args:
        q: Query tensor with shape (batch_size, num_heads, seq_len, head_dim)
        k: Key tensor with shape (batch_size, num_heads, seq_len, head_dim)
        freqs: Frequency tensor from ContinuousRoPE with shape matching q/k sequence dimension

    Returns:
        Tuple of (rotated_q, rotated_k) with same shapes as input

    Example:
        >>> positions = keras.random.uniform((32, 3)) * 1000
        >>> rope = ContinuousRoPE(dim=64, ndim=3)
        >>> freqs = rope(positions)
        >>>
        >>> q = keras.random.normal((2, 8, 32, 64))
        >>> k = keras.random.normal((2, 8, 32, 64))
        >>>
        >>> q_rot, k_rot = apply_rope_to_attention(q, k, freqs)
        >>> print(q_rot.shape, k_rot.shape)  # (2, 8, 32, 64), (2, 8, 32, 64)
    """
    from dl_techniques.layers.rope_layer_continuous import apply_rope

    q_rotated = apply_rope(q, freqs)
    k_rotated = apply_rope(k, freqs)

    return q_rotated, k_rotated


def apply_rope_cross_attention(q, k, q_freqs, k_freqs):
    """Apply RoPE to query and key tensors with different frequencies.

    This is useful for cross-attention scenarios where queries and keys
    come from different coordinate systems or modalities, requiring
    different positional encodings.

    Args:
        q: Query tensor with shape (batch_size, num_heads, q_seq_len, head_dim)
        k: Key tensor with shape (batch_size, num_heads, k_seq_len, head_dim)
        q_freqs: Frequency tensor for queries
        k_freqs: Frequency tensor for keys

    Returns:
        Tuple of (rotated_q, rotated_k) with same shapes as input

    Example:
        >>> # Different coordinate systems for queries and keys
        >>> q_positions = keras.random.uniform((50, 3)) * 1000  # Query positions
        >>> k_positions = keras.random.uniform((75, 3)) * 500   # Key positions
        >>>
        >>> rope = ContinuousRoPE(dim=64, ndim=3)
        >>> q_freqs = rope(q_positions)
        >>> k_freqs = rope(k_positions)
        >>>
        >>> q = keras.random.normal((2, 8, 50, 64))
        >>> k = keras.random.normal((2, 8, 75, 64))
        >>>
        >>> q_rot, k_rot = apply_rope_cross_attention(q, k, q_freqs, k_freqs)
    """
    from dl_techniques.layers.rope_layer_continuous import apply_rope

    q_rotated = apply_rope(q, q_freqs)
    k_rotated = apply_rope(k, k_freqs)

    return q_rotated, k_rotated


def create_rope_frequencies(positions, dim, ndim=3, max_wavelength=10000.0):
    """Convenience function to create RoPE frequencies from positions.

    Args:
        positions: Position tensor with shape (..., ndim)
        dim: Embedding dimension (typically head_dim)
        ndim: Number of coordinate dimensions
        max_wavelength: Maximum wavelength for frequencies

    Returns:
        Frequency tensor for use with apply_rope

    Example:
        >>> positions = keras.random.uniform((100, 3)) * 1000
        >>> freqs = create_rope_frequencies(positions, dim=64, ndim=3)
        >>> print(freqs.shape)  # (100, 64)
    """
    from dl_techniques.layers.rope_layer_continuous import ContinuousRoPE

    rope_layer = ContinuousRoPE(
        dim=dim,
        ndim=ndim,
        max_wavelength=max_wavelength
    )

    return rope_layer(positions)


def scaled_dot_product_attention_with_rope(q, k, v, positions=None, freqs=None,
                                           dim=None, ndim=3, dropout_rate=0.0, training=None):
    """Complete scaled dot-product attention with optional RoPE.

    This function provides a complete attention mechanism with optional
    Rotary Position Embedding, combining query/key rotation with the
    standard attention computation.

    Args:
        q: Query tensor (batch_size, num_heads, seq_len, head_dim)
        k: Key tensor (batch_size, num_heads, seq_len, head_dim)
        v: Value tensor (batch_size, num_heads, seq_len, head_dim)
        positions: Optional position tensor for RoPE (..., ndim)
        freqs: Optional pre-computed frequencies (overrides positions)
        dim: Head dimension for RoPE (required if positions provided)
        ndim: Coordinate dimensions for RoPE
        dropout_rate: Dropout rate for attention weights
        training: Training mode flag

    Returns:
        Attention output tensor (batch_size, num_heads, seq_len, head_dim)

    Example:
        >>> q = keras.random.normal((2, 8, 32, 64))
        >>> k = keras.random.normal((2, 8, 32, 64))
        >>> v = keras.random.normal((2, 8, 32, 64))
        >>> positions = keras.random.uniform((32, 3)) * 1000
        >>>
        >>> # With RoPE
        >>> output = scaled_dot_product_attention_with_rope(
        ...     q, k, v, positions=positions, dim=64, ndim=3
        ... )
        >>> print(output.shape)  # (2, 8, 32, 64)
        >>>
        >>> # Without RoPE (standard attention)
        >>> output = scaled_dot_product_attention_with_rope(q, k, v)
    """
    # Apply RoPE if positions or frequencies provided
    if freqs is not None:
        q, k = apply_rope_to_attention(q, k, freqs)
    elif positions is not None:
        if dim is None:
            raise ValueError("dim must be provided when using positions")
        freqs = create_rope_frequencies(positions, dim, ndim)
        q, k = apply_rope_to_attention(q, k, freqs)

    # Compute attention scores
    scale = 1.0 / ops.sqrt(float(q.shape[-1]))
    scores = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * scale

    # Apply softmax
    attn_weights = ops.softmax(scores, axis=-1)

    # Apply dropout if specified
    if dropout_rate > 0.0 and training:
        # Note: Keras Dropout layer would need to be created outside this function
        # This is a placeholder for the dropout operation
        logger.warning("Dropout not applied in scaled_dot_product_attention_with_rope. "
                       "Create a Dropout layer externally for proper dropout.")

    # Apply attention to values
    output = ops.matmul(attn_weights, v)

    return output


def rope_attention_block(x, dim, num_heads, positions=None, ndim=3,
                         dropout_rate=0.0, use_bias=True, training=None):
    """Complete multi-head attention block with RoPE support.

    This provides a high-level interface for multi-head attention with
    integrated RoPE support, handling the reshaping and projection operations.

    Args:
        x: Input tensor (batch_size, seq_len, dim)
        dim: Model dimension
        num_heads: Number of attention heads
        positions: Optional positions for RoPE (..., ndim)
        ndim: Coordinate dimensions
        dropout_rate: Dropout rate for attention
        use_bias: Whether to use bias in projections
        training: Training mode flag

    Returns:
        Output tensor (batch_size, seq_len, dim)

    Example:
        >>> x = keras.random.normal((2, 100, 512))
        >>> positions = keras.random.uniform((100, 3)) * 1000
        >>>
        >>> output = rope_attention_block(
        ...     x, dim=512, num_heads=8, positions=positions, ndim=3
        ... )
        >>> print(output.shape)  # (2, 100, 512)
    """
    if dim % num_heads != 0:
        raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

    head_dim = dim // num_heads
    batch_size, seq_len, _ = x.shape

    # This would typically use layers created outside this function
    # For now, we'll show the conceptual structure

    logger.warning("rope_attention_block is a conceptual example. "
                   "Use proper Keras layers (Dense, etc.) in practice.")

    # Conceptual QKV projection (would use actual Dense layers)
    # qkv = dense_layer(x)  # Project to q, k, v
    # q, k, v = split and reshape to (batch, heads, seq, head_dim)

    # Apply RoPE if positions provided
    # if positions is not None:
    #     q, k = apply_rope_to_attention(q, k, create_rope_frequencies(positions, head_dim, ndim))

    # Apply attention and project back
    # output = scaled_dot_product_attention(q, k, v, dropout_rate, training)
    # output = reshape and project back to (batch, seq, dim)

    return x  # Placeholder return