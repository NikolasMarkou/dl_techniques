"""Continuous Rotary Position Embedding (RoPE) for spatial coordinates."""

import keras
from keras import ops
from typing import Optional, Any, Dict, Tuple
import numpy as np

from dl_techniques.utils.logger import logger


@keras.saving.register_keras_serializable()
class ContinuousRoPE(keras.layers.Layer):
    """Continuous Rotary Position Embedding for variable positions.

    This implements RoPE (Rotary Position Embedding) extended to continuous
    coordinates rather than discrete sequence positions. It generates complex
    frequencies that can be used to apply rotational position encoding to
    query and key vectors in attention mechanisms.

    Based on "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    but extended to handle continuous multi-dimensional coordinates.

    Args:
        dim: Integer, dimensionality of the embedding (typically head_dim in attention).
        ndim: Integer, number of coordinate dimensions (e.g., 2 for 2D, 3 for 3D).
        max_wavelength: Float, theta parameter for the embedding. Defaults to 10000.0.
        assert_positive: Boolean, ensures coordinates are positive (useful for
            normalized coordinate systems). Defaults to True.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        - 2D tensor with shape: `(num_positions, ndim)`
        - 3D tensor with shape: `(batch_size, num_positions, ndim)`

    Output shape:
        - 2D tensor with shape: `(num_positions, dim)`
        - 3D tensor with shape: `(batch_size, num_positions, dim)`

    Returns:
        Complex frequencies as two channels (real, imaginary) for applying
        rotational position encoding.

    Example:
        >>> # 3D coordinates for attention
        >>> positions = keras.random.uniform((32, 3)) * 1000
        >>> rope = ContinuousRoPE(dim=64, ndim=3)  # head_dim=64
        >>> freqs = rope(positions)
        >>> print(freqs.shape)  # (32, 64)

        >>> # Apply to query/key vectors in attention
        >>> q = keras.random.normal((1, 8, 32, 64))  # (batch, heads, seq, head_dim)
        >>> rotated_q = apply_rope(q, freqs)

    Notes:
        The output represents complex numbers in polar form that can be used
        to rotate query and key vectors for position-aware attention.
    """

    def __init__(
            self,
            dim: int,
            ndim: int,
            max_wavelength: float = 10000.0,
            assert_positive: bool = True,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.dim = dim
        self.ndim = ndim
        self.max_wavelength = max_wavelength
        self.assert_positive = assert_positive

        # Calculate padding needed if dim is not cleanly divisible by ndim
        self.ndim_padding = dim % ndim
        dim_per_ndim = (dim - self.ndim_padding) // ndim
        self.sincos_padding = dim_per_ndim % 2
        self.padding = self.ndim_padding + self.sincos_padding * ndim

        # Calculate effective dimensions
        effective_dim_per_wave = (self.dim - self.padding) // ndim
        if effective_dim_per_wave <= 0:
            raise ValueError(f"dim ({dim}) too small for ndim ({ndim}). "
                             f"Need at least {ndim * 2} dimensions.")

        # Store build information
        self._build_input_shape = None

        # Will be created in build()
        self.omega = None

    def build(self, input_shape):
        """Build the layer weights."""
        self._build_input_shape = input_shape

        # Validate input shape
        if len(input_shape) < 2:
            raise ValueError(f"Input must be at least 2D, got shape {input_shape}")

        if input_shape[-1] != self.ndim:
            raise ValueError(f"Last dimension of input ({input_shape[-1]}) "
                             f"must match ndim ({self.ndim})")

        # Create frequency weights
        effective_dim_per_wave = (self.dim - self.padding) // self.ndim
        arange_vals = np.arange(0, effective_dim_per_wave, 2, dtype=np.float32)
        omega_vals = 1.0 / (self.max_wavelength ** (arange_vals / effective_dim_per_wave))

        self.omega = self.add_weight(
            name="omega",
            shape=omega_vals.shape,
            initializer="zeros",
            trainable=False,
        )

        # Set the omega values
        self.omega.assign(omega_vals)

        super().build(input_shape)

    def call(self, coords, training=None):
        """Generate continuous RoPE frequencies.

        Args:
            coords: Input tensor of coordinates with shape (..., ndim).
            training: Boolean indicating training mode (unused).

        Returns:
            Complex frequencies tensor with shape (..., dim) representing
            phase angles for rotational position encoding.
        """
        if self.assert_positive:
            # Check if coordinates are positive
            min_coords = ops.min(coords)
            if min_coords < 0:
                logger.warning(f"Negative coordinates detected: min={min_coords}")

        # Ensure float32 precision
        coords = ops.cast(coords, "float32")

        # Expand coordinates for frequency multiplication
        # coords: (..., ndim) -> coords_expanded: (..., ndim, 1)
        coords_expanded = ops.expand_dims(coords, axis=-1)

        # omega: (freq_dim,) -> omega_expanded: (1, ..., 1, freq_dim)
        omega_shape = [1] * (len(coords.shape) - 1) + [self.omega.shape[0]]
        omega_expanded = ops.reshape(self.omega, omega_shape)

        # Compute phase angles: (..., ndim, freq_dim)
        phases = coords_expanded * omega_expanded

        # Flatten coordinate and frequency dimensions
        if len(coords.shape) == 3:  # Batch case
            batch_size, num_points, _ = coords.shape
            phases = ops.reshape(phases, (batch_size, num_points, -1))
        elif len(coords.shape) == 2:  # No batch case
            num_points, _ = coords.shape
            phases = ops.reshape(phases, (num_points, -1))
        else:
            # General case
            new_shape = list(coords.shape[:-1]) + [-1]
            phases = ops.reshape(phases, new_shape)

        # Add padding if necessary
        if self.padding > 0:
            padding_shape = list(phases.shape)
            padding_shape[-1] = self.padding // 2
            padding_zeros = ops.zeros(padding_shape, dtype=phases.dtype)
            phases = ops.concatenate([phases, padding_zeros], axis=-1)

        return phases

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the layer."""
        input_shape_list = list(input_shape)
        return tuple(input_shape_list[:-1] + [self.dim])

    def get_config(self):
        """Returns the layer configuration."""
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "ndim": self.ndim,
            "max_wavelength": self.max_wavelength,
            "assert_positive": self.assert_positive,
        })
        return config

    def get_build_config(self):
        """Get build configuration."""
        return {"input_shape": self._build_input_shape}

    def build_from_config(self, config):
        """Build from configuration."""
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])


def apply_rope(x, freqs):
    """Apply Rotary Position Embedding to input tensor.

    This function applies the continuous RoPE transformation to query or key
    vectors in attention mechanisms using the frequencies generated by ContinuousRoPE.

    Args:
        x: Input tensor with shape (batch_size, num_heads, seq_len, head_dim)
        freqs: Frequency tensor from ContinuousRoPE with shape (batch_size, seq_len, head_dim//2)
           or (seq_len, head_dim//2)

    Returns:
        Rotated tensor with same shape as input x.

    Example:
        >>> positions = keras.random.uniform((32, 3)) * 1000
        >>> rope = ContinuousRoPE(dim=64, ndim=3)
        >>> freqs = rope(positions)
        >>>
        >>> # Apply to attention vectors
        >>> q = keras.random.normal((2, 8, 32, 64))  # (batch, heads, seq, head_dim)
        >>> q_rotated = apply_rope(q, freqs)
        >>> print(q_rotated.shape)  # (2, 8, 32, 64)
    """
    if x.ndim != 4:
        raise ValueError(f"Input x must be 4D (batch, heads, seq, head_dim), got shape {x.shape}")

    batch_size, num_heads, seq_len, head_dim = x.shape

    # Ensure freqs has batch dimension
    if freqs.ndim == 2:
        freqs = ops.expand_dims(freqs, axis=0)  # Add batch dimension
        freqs = ops.repeat(freqs, batch_size, axis=0)

    # Add heads dimension to freqs: (batch, seq, head_dim//2) -> (batch, 1, seq, head_dim//2)
    freqs = ops.expand_dims(freqs, axis=1)

    # Reshape x to separate real and imaginary parts
    # x: (batch, heads, seq, head_dim) -> (batch, heads, seq, head_dim//2, 2)
    x_reshaped = ops.reshape(x, (batch_size, num_heads, seq_len, head_dim // 2, 2))

    # Extract real and imaginary parts
    x_real = x_reshaped[..., 0]  # (batch, heads, seq, head_dim//2)
    x_imag = x_reshaped[..., 1]  # (batch, heads, seq, head_dim//2)

    # Compute cos and sin of frequencies
    cos_freqs = ops.cos(freqs)  # (batch, 1, seq, head_dim//2)
    sin_freqs = ops.sin(freqs)  # (batch, 1, seq, head_dim//2)

    # Apply rotation: (x_real + i*x_imag) * (cos + i*sin) = (x_real*cos - x_imag*sin) + i*(x_real*sin + x_imag*cos)
    rotated_real = x_real * cos_freqs - x_imag * sin_freqs
    rotated_imag = x_real * sin_freqs + x_imag * cos_freqs

    # Combine back into original format
    rotated = ops.stack([rotated_real, rotated_imag], axis=-1)
    rotated = ops.reshape(rotated, (batch_size, num_heads, seq_len, head_dim))

    return ops.cast(rotated, x.dtype)