"""
Generates continuous, multi-dimensional positional embeddings using sinusoids.

This layer implements a technique for encoding continuous spatial coordinates
(e.g., 2D or 3D points) into a high-dimensional vector space. It serves as
a continuous and multi-dimensional generalization of the fixed sinusoidal
positional encodings used in the original Transformer architecture, making it
suitable for tasks involving geometric data like point clouds, images, or
physical simulations.

Architecture:
    The core principle is to map each scalar coordinate value into a vector
    of sine and cosine values at different frequencies. This creates a rich,
    smooth, and periodic representation that allows a neural network to
    easily reason about relative positions and distances.

    The architecture partitions the total embedding dimension (`dim`) among
    the number of input coordinate dimensions (`ndim`). For each coordinate
    `p_k` (e.g., the x, y, or z value of a point), the layer performs the
    following steps:
    1.  It scales the coordinate by a set of fixed, non-learnable
        frequencies that form a geometric progression.
    2.  It applies both the `sin` and `cos` functions to these scaled values.
    3.  The resulting sine and cosine values for all frequencies are
        concatenated to form an embedding for that single coordinate.

    The final output is the concatenation of the embeddings from all input
    coordinate dimensions, resulting in a single vector that encodes the
    full multi-dimensional position.

Foundational Mathematics:
    This method is a direct extension of the positional encoding formula
    from "Attention Is All You Need". For a single continuous coordinate
    `p`, its embedding `E(p)` is a vector where each pair of elements is
    defined by:

        E(p)_{2i}   = sin(p * omega_i)
        E(p)_{2i+1} = cos(p * omega_i)

    The frequencies `omega_i` are fixed and decrease exponentially, forming
    a geometric progression:

        omega_i = 1 / (max_wavelength^(2i / d'))

    where `d'` is the embedding dimension allocated per coordinate. This
    formulation has several key properties:
    -   **Continuity:** The embedding function is smooth, so nearby points in
        coordinate space are mapped to nearby points in the embedding space.
    -   **Relative Positioning:** For any displacement `delta`, the embedding
        `E(p + delta)` can be represented as a linear transformation of
        `E(p)`, making it easy for models like transformers to learn relative
        positional relationships.
    -   **Multi-Frequency Representation:** The use of a spectrum of
        frequencies, from low (`max_wavelength`) to high, allows the model
        to capture both coarse, global positional information and
        fine-grained, local details simultaneously.

References:
    - The core technique is inspired by the original Transformer positional
      encodings:
      Vaswani, A., et al. (2017). "Attention Is All You Need".

    - This style of continuous coordinate embedding is a key component in
      Neural Radiance Fields (NeRF) for representing 3D coordinates:
      Mildenhall, B., et al. (2020). "NeRF: Representing Scenes as Neural
      Radiance Fields for View Synthesis".
"""

import keras
import numpy as np
from keras import ops
from typing import Optional, Any, Dict, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ContinuousSinCosEmbed(keras.layers.Layer):
    """Continuous sinusoidal coordinate embedding for multi-dimensional positions.

    Embeds continuous coordinates (e.g. 2D/3D positions) into a high-dimensional
    vector space using alternating sine and cosine functions at geometrically
    spaced frequencies. For each coordinate ``p_k``, the embedding computes
    ``E(p_k)_{2i} = sin(p_k * omega_i)`` and
    ``E(p_k)_{2i+1} = cos(p_k * omega_i)`` where
    ``omega_i = 1 / (max_wavelength^(2i/d'))``. The per-coordinate embeddings
    are concatenated to produce the final output of dimension ``dim``. This
    creates smooth, continuous representations that preserve spatial
    relationships and enable learning of relative positions.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────┐
        │  Input coords (..., ndim)        │
        └───────────────┬──────────────────┘
                        ▼
        ┌──────────────────────────────────┐
        │  For each coord dimension k:     │
        │    freqs = p_k * omega           │
        │    emb_k = [sin(freqs),          │
        │             cos(freqs)]          │
        └───────────────┬──────────────────┘
                        ▼
        ┌──────────────────────────────────┐
        │  Concatenate [emb_1,...,emb_ndim]│
        └───────────────┬──────────────────┘
                        ▼
        ┌──────────────────────────────────┐
        │  Pad if dim % ndim != 0          │
        └───────────────┬──────────────────┘
                        ▼
        ┌──────────────────────────────────┐
        │  Output (..., dim)               │
        └──────────────────────────────────┘

    :param dim: Dimensionality of the embedded output coordinates. Must be
        positive and should be at least ``2 * ndim``.
    :type dim: int
    :param ndim: Number of dimensions of the input coordinate space (e.g., 2
        for 2D, 3 for 3D). Must be positive.
    :type ndim: int
    :param max_wavelength: Maximum wavelength for the sinusoidal embedding.
        Controls the frequency range. Higher values create more gradual spatial
        variations. Defaults to ``10000.0``.
    :type max_wavelength: float
    :param assert_positive: Whether to check that all input coordinates are
        positive. Useful for normalized coordinate systems. Defaults to
        ``True``.
    :type assert_positive: bool
    :param kwargs: Additional keyword arguments for the Layer base class.

    :raises ValueError: If ``dim`` is too small for the given ``ndim``.
    :raises ValueError: If input parameters are invalid.
    :raises ValueError: If input shape is invalid.
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

        # Validate inputs
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if ndim <= 0:
            raise ValueError(f"ndim must be positive, got {ndim}")
        if max_wavelength <= 0:
            raise ValueError(f"max_wavelength must be positive, got {max_wavelength}")

        # Store ALL configuration parameters
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

        # Store effective dimension for weight creation
        self.effective_dim_per_wave = effective_dim_per_wave

        # Initialize weight attributes - created in build()
        self.omega = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the layer's fixed frequency weights.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If input shape is invalid.
        """
        # Validate input shape
        if len(input_shape) < 2:
            raise ValueError(f"Input must be at least 2D, got shape {input_shape}")

        if input_shape[-1] != self.ndim:
            raise ValueError(f"Last dimension of input ({input_shape[-1]}) "
                             f"must match ndim ({self.ndim})")

        # Create frequency weights
        arange_vals = np.arange(0, self.effective_dim_per_wave, 2, dtype=np.float32)
        omega_vals = 1.0 / (self.max_wavelength ** (arange_vals / self.effective_dim_per_wave))

        # Create layer's own weights
        self.omega = self.add_weight(
            name="omega",
            shape=omega_vals.shape,
            initializer="zeros",
            trainable=False,
        )

        # Set the omega values
        self.omega.assign(omega_vals)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
            self,
            coords: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Embed continuous coordinates using sinusoidal functions.

        :param coords: Input tensor of coordinates with shape ``(..., ndim)``.
        :type coords: keras.KerasTensor
        :param training: Whether in training mode (unused).
        :type training: Optional[bool]
        :return: Embedded coordinates with shape ``(..., dim)`` using
            alternating sine and cosine functions at different frequencies.
        :rtype: keras.KerasTensor
        """
        if self.assert_positive:
            # Check if coordinates are positive
            min_coords = ops.min(coords)
            if ops.convert_to_numpy(min_coords) < 0:
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

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape with last dimension changed to ``dim``.
        :rtype: Tuple[Optional[int], ...]
        """
        input_shape_list = list(input_shape)
        return tuple(input_shape_list[:-1] + [self.dim])

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        :return: Dictionary containing all layer configuration parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "ndim": self.ndim,
            "max_wavelength": self.max_wavelength,
            "assert_positive": self.assert_positive,
        })
        return config

# ---------------------------------------------------------------------
