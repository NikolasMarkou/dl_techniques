"""
Continuous, multi-dimensional rotary position embeddings (RoPE).

This layer extends the concept of Rotary Position Embedding (RoPE),
originally designed for 1D discrete sequences, to handle continuous,
multi-dimensional coordinates. It is designed to inject absolute positional
information into a transformer's attention mechanism in a way that allows
the model to naturally reason about relative positions, which is crucial
for tasks involving spatial data like images, videos, or 3D point clouds.

Architecture:
    Unlike traditional positional embeddings that are added to token
    embeddings, RoPE modifies the query and key vectors directly within the
    attention mechanism by applying a rotation. This layer's role is not to
    produce a final embedding, but to compute the *phase angles* for these
    rotations based on the input coordinates.

    The core architectural idea is to partition the feature dimension (`dim`)
    among the number of spatial dimensions (`ndim`). For each coordinate
    dimension (e.g., x, y, z), the layer computes a corresponding set of
    phase angles by multiplying the coordinate value with a predefined set
    of fixed, non-learnable frequencies. The final output is a single
    vector formed by concatenating the phase angles from all spatial
    dimensions. This vector can then be used in an attention layer to apply
    the N-dimensional rotation to the query and key vectors.

Foundational Mathematics:
    The fundamental principle of RoPE is to encode absolute position `p`
    by applying a rotation matrix `R_p` to a feature vector `x`. The key
    property is that the inner product between two rotated vectors,
    `<R_p * q, R_k * k>`, depends only on their relative displacement, `p - k`.

    In the 1D case, this rotation is equivalent to multiplying a complex
    number representation of the vector by `e^(j * p * theta)`, where `p` is
    the position and `theta` is a frequency. The embedding dimension `d` is
    treated as `d/2` complex numbers, each rotated with a different
    frequency `theta_i` from a geometric progression:

        theta_i = base_freq^(-2i / d)

    This layer generalizes this concept to a continuous N-dimensional
    coordinate vector `P = (p_1, p_2, ..., p_ndim)`. The total embedding
    dimension `d` is split into `ndim` sub-vectors, each of dimension `d'`.
    For each coordinate component `p_k`, a vector of phase angles `phi_k`
    is computed by multiplying the coordinate value with its corresponding
    set of frequencies:

        phi_k = p_k * {theta_0, theta_1, ..., theta_{d'/2 - 1}}

    The final output of this layer is the concatenation of these phase angle
    vectors, `[phi_1, phi_2, ..., phi_ndim]`, which contains all the
    information needed to apply the full N-dimensional rotation to query
    and key vectors in an attention mechanism.

References:
    - The original concept for 1D sequences was introduced in:
      Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021).
      "RoFormer: Enhanced Transformer with Rotary Position Embedding".
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
class ContinuousRoPE(keras.layers.Layer):
    """Continuous multi-dimensional Rotary Position Embedding for spatial data.

    Extends discrete 1D RoPE to continuous, multi-dimensional coordinates by
    partitioning the embedding dimension ``dim`` among ``ndim`` spatial axes.
    For each coordinate component ``p_k``, phase angles are computed as
    ``phi_k = p_k * omega_i`` where ``omega_i = 1 / (max_wavelength^(2i/d'))``
    forms a geometric frequency progression. The concatenated phase angles
    ``[phi_1, ..., phi_ndim]`` can be used to apply N-dimensional rotations to
    query and key vectors in attention, preserving the relative-position
    property ``<R_p q, R_k k> = g(q, k, p-k)``.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────┐
        │  Input coords (..., ndim)        │
        └───────────────┬──────────────────┘
                        ▼
        ┌──────────────────────────────────┐
        │  For each coord dimension k:     │
        │    phi_k = p_k * omega           │
        │    omega_i = 1/Θ^(2i/d')        │
        └───────────────┬──────────────────┘
                        ▼
        ┌──────────────────────────────────┐
        │  Concatenate [phi_1,...,phi_ndim] │
        └───────────────┬──────────────────┘
                        ▼
        ┌──────────────────────────────────┐
        │  Pad if dim % ndim != 0          │
        └───────────────┬──────────────────┘
                        ▼
        ┌──────────────────────────────────┐
        │  Output phase angles (..., dim)  │
        └──────────────────────────────────┘

    :param dim: Dimensionality of the embedding (typically ``head_dim`` in
        attention). Must be positive and should be divisible by ``ndim``.
    :type dim: int
    :param ndim: Number of coordinate dimensions (e.g., 2 for 2D, 3 for 3D).
        Must be positive.
    :type ndim: int
    :param max_wavelength: Theta parameter controlling the frequency range.
        Higher values create lower frequencies. Defaults to ``10000.0``.
    :type max_wavelength: float
    :param assert_positive: Whether to check that coordinates are positive
        (useful for normalized coordinate systems). Defaults to ``True``.
    :type assert_positive: bool
    :param kwargs: Additional keyword arguments for the Layer base class.

    :raises ValueError: If ``dim`` is too small for the given ``ndim``.
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

        # Calculate effective dimensions
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
        """Generate continuous RoPE phase angles from spatial coordinates.

        :param coords: Input tensor of coordinates with shape ``(..., ndim)``.
        :type coords: keras.KerasTensor
        :param training: Whether in training mode (unused).
        :type training: Optional[bool]
        :return: Phase angles tensor with shape ``(..., dim)`` for rotational
            position encoding.
        :rtype: keras.KerasTensor
        """
        if self.assert_positive:
            # Check if coordinates are positive
            min_coords = ops.min(coords)
            if ops.convert_to_numpy(min_coords) < 0:
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
