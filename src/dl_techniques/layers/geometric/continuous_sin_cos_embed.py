"""Continuous sine-cosine positional embedding layer for spatial coordinates."""

import keras
from keras import ops
from typing import Optional, Any, Dict
import numpy as np

from dl_techniques.utils.logger import logger


@keras.saving.register_keras_serializable()
class ContinuousSincosEmbed(keras.layers.Layer):
    """Continuous coordinate embedding using sine and cosine functions.

    This layer embeds continuous coordinates (like 3D positions) using sinusoidal
    functions similar to transformer positional encodings but extended to handle
    arbitrary coordinate dimensions and continuous values.

    Args:
        dim: Integer, dimensionality of the embedded output coordinates.
        ndim: Integer, number of dimensions of the input coordinate space
            (e.g., 2 for 2D coordinates, 3 for 3D coordinates).
        max_wavelength: Float, maximum wavelength for the sinusoidal embedding.
            Controls the frequency range of the embedding. Defaults to 10000.0.
        assert_positive: Boolean, whether to assert that all input coordinates
            are positive. Useful for normalized coordinate systems. Defaults to True.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        - 2D tensor with shape: `(num_points, ndim)`
        - 3D tensor with shape: `(batch_size, num_points, ndim)`

    Output shape:
        - 2D tensor with shape: `(num_points, dim)`
        - 3D tensor with shape: `(batch_size, num_points, dim)`

    Returns:
        Tensor with embedded coordinates using sinusoidal functions.

    Example:
        >>> # 3D coordinates for point cloud
        >>> coords = keras.random.uniform((100, 3)) * 1000  # 100 points in 3D
        >>> embed_layer = ContinuousSincosEmbed(dim=256, ndim=3)
        >>> embedded = embed_layer(coords)
        >>> print(embedded.shape)  # (100, 256)

        >>> # Batch of 2D coordinates
        >>> coords_batch = keras.random.uniform((4, 50, 2)) * 100
        >>> embed_layer_2d = ContinuousSincosEmbed(dim=128, ndim=2)
        >>> embedded_batch = embed_layer_2d(coords_batch)
        >>> print(embedded_batch.shape)  # (4, 50, 128)
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

        # Calculate effective dimensions for wave generation
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
        """Forward computation with sinusoidal embedding.

        Args:
            coords: Input tensor of coordinates with shape (..., ndim).
            training: Boolean indicating training mode (unused).

        Returns:
            Embedded coordinates with shape (..., dim).
        """
        if self.assert_positive:
            # Check if coordinates are positive
            min_coords = ops.min(coords)
            if min_coords < 0:
                logger.warning(f"Negative coordinates detected: min={min_coords}")

        # Ensure float32 precision for numerical stability
        coords = ops.cast(coords, "float32")

        # Expand coordinates for frequency multiplication
        # coords: (..., ndim) -> coords_expanded: (..., ndim, 1)
        coords_expanded = ops.expand_dims(coords, axis=-1)

        # omega: (freq_dim,) -> omega_expanded: (1, ..., 1, freq_dim)
        omega_shape = [1] * (len(coords.shape) - 1) + [self.omega.shape[0]]
        omega_expanded = ops.reshape(self.omega, omega_shape)

        # Compute frequencies: (..., ndim, freq_dim)
        freqs = coords_expanded * omega_expanded

        # Apply sin and cos
        sin_vals = ops.sin(freqs)
        cos_vals = ops.cos(freqs)

        # Concatenate sin and cos: (..., ndim, 2*freq_dim)
        emb = ops.concatenate([sin_vals, cos_vals], axis=-1)

        # Reshape to flatten ndim and frequency dimensions
        if len(coords.shape) == 3:  # Batch case
            batch_size, num_points, _ = coords.shape
            emb = ops.reshape(emb, (batch_size, num_points, -1))
        elif len(coords.shape) == 2:  # No batch case
            num_points, _ = coords.shape
            emb = ops.reshape(emb, (num_points, -1))
        else:
            # General case for arbitrary dimensions
            new_shape = list(coords.shape[:-1]) + [-1]
            emb = ops.reshape(emb, new_shape)

        # Add padding if necessary
        if self.padding > 0:
            padding_shape = list(emb.shape)
            padding_shape[-1] = self.padding
            padding = ops.zeros(padding_shape, dtype=emb.dtype)
            emb = ops.concatenate([emb, padding], axis=-1)

        return emb

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