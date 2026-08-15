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
        │    omega_i = 1/Θ^(2i/d')         │
        └───────────────┬──────────────────┘
                        ▼
        ┌──────────────────────────────────┐
        │  Concatenate [phi_1,...,phi_ndim]│
        └───────────────┬──────────────────┘
                        ▼
        ┌──────────────────────────────────┐
        │  Pad if dim % ndim != 0          │
        └───────────────┬──────────────────┘
                        ▼
        ┌──────────────────────────────────┐
        │  Output phase angles (..., dim/2)│
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
        if self.built:
            return

        # Validate input shape
        if len(input_shape) < 2:
            raise ValueError(f"Input must be at least 2D, got shape {input_shape}")

        if input_shape[-1] != self.ndim:
            raise ValueError(f"Last dimension of input ({input_shape[-1]}) "
                             f"must match ndim ({self.ndim})")

        # Create frequency weights
        arange_vals = np.arange(0, self.effective_dim_per_wave, 2, dtype=np.float32)
        omega_vals = 1.0 / (self.max_wavelength ** (arange_vals / self.effective_dim_per_wave))

        # `omega` comes from an INITIALIZER, never from `add_weight(...)` +
        # `.assign()`: Keras 3 discards an `.assign()` issued inside the
        # `StatelessScope` of the symbolic build pass that runs whenever this layer
        # is first reached from a PARENT's `call()`, which left the table all-zero
        # in every real model (measured on CPU 2026-08-15: direct `.build(...)`
        # gives `omega[0] == 1.0`, through a parent's `call()` it was `0.0`), and an
        # all-zero omega makes every rotary angle 0, i.e. RoPE is the identity.
        # Same defect and same fix as `rotary_position_embedding.py`; the full
        # rationale is at that module's `# DECISION` anchor. See decisions.md D-027.
        # `omega_vals` is NumPy, so closing over it carries no `FuncGraph` tensor.
        self.omega = self.add_weight(
            name="omega",
            shape=omega_vals.shape,
            initializer=lambda shape, dtype=None: keras.ops.cast(
                keras.ops.convert_to_tensor(omega_vals),
                dtype or self.variable_dtype,
            ),
            trainable=False,
        )

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
        :return: Phase angles tensor whose last dimension is the phase width
            (``dim // 2`` for a ``dim`` divisible by ``ndim``); the caller turns
            these phases into a full-width rotation via cos/sin.
        :rtype: keras.KerasTensor
        """
        # NOTE: `assert_positive` is retained as a config-compatible flag but no
        # longer triggers a runtime check. The previous implementation called
        # `ops.convert_to_numpy(ops.min(coords))` here, which is an eager host
        # materialization that breaks `@tf.function` / graph tracing. Removed for
        # graph compatibility (DECISION plan_2026-06-15_9dbb87c1/D-001).

        # DECISION plan-2026-07-31T210633-b63a35aa/D-007
        # Compute the phases in a NEVER-NARROWING working dtype, and cast the
        # frequency table to that same dtype.
        #
        # Do NOT write `ops.cast(coords, "float32")` here. A hard dtype literal is
        # measured against `self.omega`, which Keras AUTOCASTS to the active compute
        # dtype on read -- so the multiply below had mismatched operands and raised
        # `InvalidArgumentError ... cannot compute Mul as input #1 was expected to be
        # a float tensor but is a half/double/bfloat16 tensor [Op:Mul]` under
        # mixed_float16, float64 AND mixed_bfloat16, i.e. this whole layer was dead at
        # 3 of the 4 standard policies.
        #
        # Do NOT "simplify" this to `ops.cast(coords, self.compute_dtype)` either:
        # that NARROWS the phase math to fp16/bf16. The conditional below IS the
        # never-narrow rule, stated rather than inherited from `variable_dtype`.
        #
        # The final `ops.cast(..., self.compute_dtype)` at the return is load-bearing:
        # without it this returns float32 under mixed_float16, a silently wrong output
        # dtype that the consuming attention layer's rotation would have to absorb.
        #
        # This is the FOURTH inline copy of this rule in `src/` and the second in this
        # package -- the sibling `continuous_sin_cos_embedding.py` carries the same
        # block under D-002. Promotion to one shared helper was considered and ruled
        # against (decisions.md D-006/D-007); a FIFTH site should promote instead of
        # copying this again.
        #
        # NOTE this cannot recover fp16/bf16 accuracy at large positions: Keras has
        # already narrowed `coords` at the autocast boundary BEFORE `call()` runs.
        work_dtype = "float64" if self.compute_dtype == "float64" else "float32"
        coords = ops.cast(coords, work_dtype)

        # Expand coordinates for frequency multiplication
        # coords: (..., ndim) -> coords_expanded: (..., ndim, 1)
        coords_expanded = ops.expand_dims(coords, axis=-1)

        # omega: (freq_dim,) -> omega_expanded: (1, ..., 1, freq_dim)
        omega_shape = [1] * (len(coords.shape) - 1) + [self.omega.shape[0]]
        omega_expanded = ops.reshape(ops.cast(self.omega, work_dtype), omega_shape)

        # Compute phase angles: (..., ndim, freq_dim)
        phases = coords_expanded * omega_expanded

        # Flatten the (ndim, freq_dim) trailing axes into one. Use the dynamic
        # leading dims from ops.shape (runtime values, never None) plus a single
        # inferred -1 so this is graph-safe for any rank / dynamic batch dim.
        phases = ops.reshape(phases, (*ops.shape(coords)[:-1], -1))

        # Add padding if necessary. Build the pad shape from dynamic leading
        # dims (ops.shape -> runtime values, never None) + a static last dim,
        # so ops.zeros is graph-safe under symbolic tracing.
        if self.padding > 0:
            padding_zeros = ops.zeros(
                (*ops.shape(phases)[:-1], self.padding // 2), dtype=phases.dtype
            )
            phases = ops.concatenate([phases, padding_zeros], axis=-1)

        # Back to the caller's compute dtype (see D-007 above -- not optional).
        return ops.cast(phases, self.compute_dtype)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        The output carries the per-position PHASE width, which is half of the
        requested ``dim`` (these phases are later turned into a full-width
        rotation by the caller applying cos/sin). For a ``dim`` cleanly
        divisible by ``ndim`` this equals ``dim // 2``.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape with last dimension set to the phase width.
        :rtype: Tuple[Optional[int], ...]
        """
        # DECISION plan_2026-06-15_9dbb87c1/D-003: report the ACTUAL phase width
        # (= dim/2 for divisible dim), not ``dim``. The prior code returned
        # ``dim`` (2x too large); a wrong compute_output_shape is worse than none.
        phase_width = self.ndim * (self.effective_dim_per_wave // 2) + self.padding // 2
        input_shape_list = list(input_shape)
        return tuple(input_shape_list[:-1] + [phase_width])

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
