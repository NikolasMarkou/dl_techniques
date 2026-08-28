"""
Continuous, multi-dimensional rotary position embeddings (RoPE).

This layer generalizes RoPE from a 1D discrete sequence to CONTINUOUS
N-dimensional coordinates. It suits spatial data: images, video, point
clouds. It does not produce an embedding. It produces the PHASE ANGLES a
downstream attention layer needs to rotate its query and key vectors.

Architecture:
    The feature width ``dim`` is partitioned across ``ndim`` coordinate axes.
    Each axis gets its own slice of the width and a shared geometric
    frequency ladder. For coordinate component ``p_k`` the layer computes
    ``phi_k = p_k * omega``, and the output is the concatenation
    ``[phi_1, ..., phi_ndim]``.

    The output width is the PHASE width, not ``dim``. There is one phase per
    adjacent channel pair, so a ``dim`` cleanly divisible by ``ndim`` gives
    ``dim // 2`` phases. The caller turns each phase into a cos/sin pair and
    rotates the matching channel pair.

    When ``dim`` is not cleanly divisible by ``ndim``, or the per-axis slice
    is odd, the layer pads. ``padding // 2`` zero columns are appended, which
    are phases of zero, i.e. no rotation for those channel pairs.

Foundational Mathematics:
    RoPE encodes an absolute position ``p`` by rotating a feature vector by
    ``R_p``. The inner product ``<R_p q, R_k k>`` then depends only on
    ``p - k``. In 1D the rotation is a multiply by ``e^(j * p * theta)``, the
    ``d``-dimensional vector being read as ``d/2`` complex numbers with
    frequencies from a geometric progression::

        theta_i = base_freq^(-2i / d)

    Here the position is a continuous vector ``P = (p_1, ..., p_ndim)``. The
    width ``d`` is split into ``ndim`` sub-vectors of width ``d'``, and each
    component contributes its own phases::

        phi_k = p_k * {theta_0, theta_1, ..., theta_{d'/2 - 1}}

    The output is ``[phi_1, ..., phi_ndim]``, which carries everything the
    caller needs to apply the full N-dimensional rotation.

References:
    - Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021).
      "RoFormer: Enhanced Transformer with Rotary Position Embedding".
"""

import keras
import numpy as np
from keras import ops
from typing import Optional, Any, Dict, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ContinuousRoPE(keras.layers.Layer):
    """Turn continuous N-D coordinates into RoPE phase angles.

    Takes a coordinate tensor of shape ``(..., ndim)`` and returns phase
    angles. It does NOT rotate anything and it does not emit an embedding.
    The consuming attention layer takes cos and sin of these phases and
    applies the rotation to its own query and key vectors.

    The width ``dim`` is partitioned across the ``ndim`` axes. Each axis
    contributes ``phi_k = p_k * omega`` with the shared ladder
    ``omega_i = 1 / (max_wavelength^(2i/d'))``, and the results are
    concatenated. The output width is the PHASE width, one phase per adjacent
    channel pair, which is ``dim // 2`` for a ``dim`` divisible by ``ndim``.

    **Architecture Overview:**

    .. code-block:: text

        coords  (..., ndim)          omega  (freq_dim,)
              │                      omega_i = 1 / Theta^(2i/d')
              │                      non-trainable weight
              ▼                              │
        expand to (..., ndim, 1)             │
              │                              │
              └──────── multiply ────────────┘
                        │
                        ▼
        phases  (..., ndim, freq_dim)
                        │
                flatten the last two axes
                        ▼
        phases  (..., ndim * freq_dim)
                        │
                append padding // 2 zero columns when
                dim is not cleanly divisible by ndim
                        ▼
        output  (..., ndim * (d'//2) + padding//2)

        Each output column is ONE phase, matching ONE adjacent channel
        pair in the consumer. A zero column means that pair is not
        rotated.

    :param dim: Feature width the phases are meant for, usually the attention
        ``head_dim``. Must be positive. It should divide cleanly by ``ndim``;
        the remainder is handled by padding.
    :type dim: int
    :param ndim: Number of coordinate axes, so 2 for 2D and 3 for 3D. Must be
        positive and must equal the input's last dimension.
    :type ndim: int
    :param max_wavelength: Base of the frequency ladder. Larger values give
        lower frequencies and longer wavelengths. Defaults to ``10000.0``.
    :type max_wavelength: float
    :param assert_positive: Kept for config compatibility only. It triggers
        NO runtime check; see the note in :meth:`call`. Defaults to ``True``.
    :type assert_positive: bool
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    :ivar padding: Total channels that could not be split evenly. The output
        gains ``padding // 2`` zero phase columns.
    :vartype padding: int
    :ivar effective_dim_per_wave: Channels per axis after padding is removed,
        written ``d'`` above.
    :vartype effective_dim_per_wave: int
    :ivar omega: Non-trainable frequency vector of shape ``(d' // 2,)``.
        ``None`` until ``build()`` runs.
    :vartype omega: keras.Variable or None

    Input shape:
        At least 2D, with the last dimension equal to ``ndim``:
        ``(..., ndim)``.

    Output shape:
        The input shape with the last dimension replaced by
        ``ndim * (d' // 2) + padding // 2``.

    :raises ValueError: If ``dim``, ``ndim`` or ``max_wavelength`` is not
        positive, or if ``dim`` is too small to give each axis at least two
        channels. Raised from ``__init__``.
    :raises ValueError: If the input is less than 2D, or if its last
        dimension is not ``ndim``. Raised from ``build()``.

    Example:

    .. code-block:: python

        import keras
        from dl_techniques.layers.embedding import (
            create_embedding_layer,
        )

        rope = create_embedding_layer(
            "continuous_rope", dim=64, ndim=2,
        )
        coords = keras.random.uniform((2, 100, 2))
        rope(coords).shape  # (2, 100, 32)
    """

    def __init__(
            self,
            dim: int,
            ndim: int,
            max_wavelength: float = 10000.0,
            assert_positive: bool = True,
            **kwargs: Any
    ) -> None:
        """Validate the configuration and derive the padding split.

        No weight is created here; ``omega`` is built in :meth:`build`.

        :param dim: Feature width the phases are meant for.
        :type dim: int
        :param ndim: Number of coordinate axes.
        :type ndim: int
        :param max_wavelength: Base of the frequency ladder.
        :type max_wavelength: float
        :param assert_positive: Config-compatible flag; no runtime effect.
        :type assert_positive: bool
        :param kwargs: Additional keyword arguments for the ``Layer`` base
            class.
        :type kwargs: Any
        :raises ValueError: If ``dim``, ``ndim`` or ``max_wavelength`` is not
            positive, or if ``dim`` leaves an axis with no channels.
        """
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
        """Create the non-trainable ``omega`` frequency vector.

        ``omega`` has shape ``(effective_dim_per_wave // 2,)`` and is shared
        by every coordinate axis.

        :param input_shape: Shape of the coordinate tensor, ``(..., ndim)``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the input is less than 2D, or if its last
            dimension is not ``ndim``.
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

        # `omega` comes from an INITIALIZER, never `add_weight(...)` +
        # `.assign()`. Keras 3 discards an `.assign()` issued inside the
        # `StatelessScope` of the symbolic build pass, which runs whenever this
        # layer is first reached from a parent's `call()`. Measured on CPU
        # 2026-08-15: a direct `.build(...)` gives `omega[0] == 1.0`, through a
        # parent's `call()` it was `0.0`. An all-zero omega makes every angle 0,
        # so RoPE becomes the identity. Same defect and fix as
        # `rotary_position_embedding.py`; the full rationale is at that module's
        # `# DECISION` anchor. `omega_vals` is NumPy, so closing over it carries
        # no `FuncGraph` tensor. See decisions.md D-027 of
        # plan-2026-08-14T233721-d4f9beb2.
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
        :return: Phase angles whose last dimension is the phase width, which
            is ``dim // 2`` for a ``dim`` divisible by ``ndim``. The caller
            turns each phase into cos/sin and rotates one channel pair with
            it.
        :rtype: keras.KerasTensor
        """
        # `assert_positive` is retained as a config-compatible flag but no
        # longer triggers a runtime check. The previous implementation called
        # `ops.convert_to_numpy(ops.min(coords))` here, which is an eager host
        # materialization that breaks `@tf.function` / graph tracing. Removed for
        # graph compatibility (DECISION plan_2026-06-15_9dbb87c1/D-001).
        # That plan directory no longer exists, so this comment is the only
        # surviving record of the change. Do not delete it.

        # DECISION plan-2026-07-31T210633-b63a35aa/D-007
        # Compute the phases in a NEVER-NARROWING work dtype and cast the
        # frequency table to the same dtype. Do NOT write
        # `ops.cast(coords, "float32")`: `self.omega` is a VARIABLE that Keras
        # autocasts to the compute dtype on read, so a hard literal left the
        # multiply with mismatched operands and raised `InvalidArgumentError ...
        # cannot compute Mul as input #1 was expected to be a float tensor but
        # is a half/double/bfloat16 tensor [Op:Mul]` under mixed_float16,
        # float64 and mixed_bfloat16 alike -- the layer was dead at 3 of the 4
        # standard policies. Do NOT use `ops.cast(coords, self.compute_dtype)`
        # either; that narrows the phase math to fp16/bf16. The final
        # `ops.cast(..., self.compute_dtype)` at the return is required: without
        # it the layer returns float32 under mixed_float16. This cannot recover
        # fp16/bf16 accuracy at large positions, because Keras narrowed `coords`
        # at the autocast boundary before `call()` ran. A FIFTH copy of this
        # rule in `src/` should promote a shared helper instead of copying
        # again. See decisions.md D-007.
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

        # Back to the caller's compute dtype. Required, not cosmetic: see the
        # D-007 anchor above.
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
        # That plan directory no longer exists, so this comment is the only
        # surviving record of the change. Do not delete it.
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
