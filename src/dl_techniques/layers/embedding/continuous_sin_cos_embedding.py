"""
Embed continuous multi-dimensional coordinates with sine and cosine.

This module provides :class:`ContinuousSinCosEmbed`, which maps a point in
``ndim``-dimensional continuous space to a ``dim``-wide feature vector. It
generalizes the Transformer's fixed sinusoidal encoding in two ways: the
position may be a real number rather than an integer index, and there may be
several coordinate axes rather than one. That makes it usable for point
clouds, geometric data and physical simulations.

Architecture:
    The output width ``dim`` is divided among the ``ndim`` coordinate axes.
    Every axis gets the SAME frequency ladder, a geometric progression of
    fixed, non-trainable values. Each coordinate is multiplied by the ladder,
    sine and cosine are taken, and the per-axis blocks are concatenated.

    The division rarely comes out even, so the layer computes a per-axis
    width that divides cleanly and both halves of the sin/cos pair, then
    pads the remainder with zeros. Those pad channels are literally zero and
    carry no position information. The class docstring's diagram gives the
    exact arithmetic.

Foundational Mathematics:
    For a single coordinate ``p`` the embedding pairs are::

        E(p)_{2i}     = sin(p * omega_i)
        E(p)_{2i + 1} = cos(p * omega_i)

    with a ladder that decays geometrically::

        omega_i = 1 / max_wavelength ** (2i / d')

    where ``d'`` is the width allocated to one axis. Three properties
    follow. The map is smooth, so nearby coordinates land nearby. For any
    displacement ``delta``, ``E(p + delta)`` is a linear map of ``E(p)``, so
    a model can learn to reason about relative position. And the spread of
    frequencies carries coarse and fine position at the same time.

Accuracy:
    This layer is sensitive to the MAGNITUDE of its coordinates under
    reduced precision, badly so. The class docstring carries the measured
    figures and the remedies that were tried. Read them before using it
    inside a ``mixed_float16`` or ``mixed_bfloat16`` region.

References:
    - Vaswani, A., et al. (2017). "Attention Is All You Need". The source of
      the sinusoidal encoding this layer generalizes.
    - Mildenhall, B., et al. (2020). "NeRF: Representing Scenes as Neural
      Radiance Fields for View Synthesis". Uses this style of continuous
      coordinate embedding for 3D points.
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
class ContinuousSinCosEmbed(keras.layers.Layer):
    """Embed an ``ndim``-dimensional continuous coordinate as sin/cos.

    Multiplies every coordinate axis by a shared geometric frequency ladder,
    takes sine and cosine, and concatenates the per-axis blocks. The result
    is smooth in the coordinate, so nearby points embed to nearby vectors.
    The ladder is a fixed, non-trainable weight.

    The output is always ``dim`` wide, but a variable number of its trailing
    channels can be exactly zero. Those are pad channels that carry no
    position. How many there are is set at construction, and it is NOT
    ``dim % ndim``. The diagram below gives the arithmetic and three worked
    cases. A caller that slices this output, or sizes a downstream layer by
    counting informative channels, needs that number.

    ``assert_positive`` is accepted and serialized but performs no check.
    See the note in ``call()``.

    **Architecture Overview:**

    .. code-block:: text

        ┌────────────────────────────────────────────┐
        │  coords  (..., ndim)                       │
        └────────────────────────────────────────────┘
                              │
                              ▼
        ┌────────────────────────────────────────────┐
        │  cast to work_dtype                        │
        │  float64 under a float64 policy,           │
        │  float32 at every other policy             │
        └────────────────────────────────────────────┘
                              │
                              ▼
                (..., ndim, 1)          omega  (freq_dim,)
                                        omega_i =
                                          1 / wl**(2i/eff)
                       │                      │
                       └──────────┬───────────┘
                                  ▼
        ┌────────────────────────────────────────────┐
        │  freqs = coord * omega                     │
        │  (..., ndim, freq_dim)                     │
        └────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
                sin(freqs)                  cos(freqs)
                    └─────────────┬─────────────┘
                                  ▼
        ┌────────────────────────────────────────────┐
        │  concat on the last axis                   │
        │  (..., ndim, 2 * freq_dim)                 │
        │  then flatten the last two axes            │
        │  (..., ndim * 2 * freq_dim)                │
        └────────────────────────────────────────────┘
                              │
                              ▼
        ┌────────────────────────────────────────────┐
        │  append `padding` zero channels            │
        │  skipped when padding == 0                 │
        └────────────────────────────────────────────┘
                              │
                              ▼
        ┌────────────────────────────────────────────┐
        │  cast to compute_dtype                     │
        │  output  (..., dim)                        │
        └────────────────────────────────────────────┘

        Width arithmetic, all fixed at construction:

            ndim_padding   = dim % ndim
            dim_per_ndim   = (dim - ndim_padding) // ndim
            sincos_padding = dim_per_ndim % 2
            padding        = ndim_padding
                             + sincos_padding * ndim
            eff            = (dim - padding) // ndim
            freq_dim       = eff // 2

            ndim * 2 * freq_dim + padding == dim, always.

        So the output is ALWAYS `dim` wide, but its last
        `padding` channels are exactly zero and encode
        nothing. `padding` is not `dim % ndim`: at dim=10,
        ndim=2 that term is 0 and `padding` is still 2,
        because a per-axis width of 5 cannot hold sin/cos
        pairs. At dim=32, ndim=3 padding is 2; at dim=30,
        ndim=4 it is 6; at dim=12, ndim=2 it is 0.

    :param dim: Width of the output. Must be positive, and at least
        ``2 * ndim`` for the per-axis width to hold one sin/cos pair.
    :type dim: int
    :param ndim: Number of coordinate axes, so 2 for a 2D point and 3 for a
        3D one. Must be positive and must equal the input's last dimension.
    :type ndim: int
    :param max_wavelength: Base of the frequency ladder. Larger values make
        the embedding vary more slowly with position. Must be positive.
        Defaults to ``10000.0``.
    :type max_wavelength: float
    :param assert_positive: Retained for config compatibility. It performs
        NO check. Defaults to ``True``.
    :type assert_positive: bool
    :param kwargs: Additional keyword arguments for the Layer base class.

    :ivar padding: Number of trailing zero channels in the output.
    :vartype padding: int
    :ivar effective_dim_per_wave: Width allocated to one coordinate axis,
        always even and at least 2.
    :vartype effective_dim_per_wave: int
    :ivar omega: The frequency ladder, shape
        ``(effective_dim_per_wave // 2,)``. ``None`` until build.
    :vartype omega: Optional[keras.Variable]

    Input shape:
        At least 2D, with a last dimension of ``ndim``, for example
        ``(batch, ndim)`` or ``(batch, num_points, ndim)``.

    Output shape:
        The input shape with the last dimension replaced by ``dim``.

    :raises ValueError: If ``dim``, ``ndim`` or ``max_wavelength`` is not
        positive, or if ``dim`` is too small to give each axis one sin/cos
        pair. Raised from ``__init__``.
    :raises ValueError: If the input is rank 1, or if its last dimension is
        not ``ndim``. Raised from ``build()``.

    Example:

    .. code-block:: python

        import numpy as np
        from dl_techniques.layers.embedding \
            import continuous_sin_cos_embedding as csc

        emb = csc.ContinuousSinCosEmbed(dim=32, ndim=3)
        pts = np.random.rand(4, 3).astype("float32")
        emb(pts).shape  # (4, 32)

    .. note::
       **Large coordinates are inaccurate under reduced precision.**
       The layer RUNS at ``float32``, ``mixed_float16``, ``float64`` and
       ``mixed_bfloat16``. Running is not the same as being right.

       *The measurement.* On CPU against a float64 oracle, with
       ``dim=64, ndim=3, max_wavelength=10000`` and coordinates drawn from
       ``[0, 2000)``, the output is wrong by up to **0.47** at
       ``mixed_float16`` and up to **1.99** at ``mixed_bfloat16``. An
       8-seed sweep gave 0.45-0.48 and 1.96-2.00. The output ranges over
       ``[-1, 1]``, so at bfloat16 the error is nearly the full range and
       the embedding carries essentially no position.

       *Why ``call()`` cannot fix it.* Keras narrows ``coords`` at the
       autocast boundary, BEFORE ``call()`` runs. A coordinate of
       ``1934.448`` was measured arriving as ``1934.0``. The ``omega``
       read is narrowed the same way. The widened working dtype inside
       ``call()`` therefore starts from operands that already lost the
       precision. The error tracks the MAGNITUDE of the coordinates, not
       the layer: the same configuration over ``[0, 64)`` is roughly 40x
       more accurate at both policies. Those per-policy figures live with
       the tolerance table of the test that uses them.

       *Remedies, as measured. They do not all work.* Handing in float32
       ``coords`` does NOT help; the error stays at 0.47 at fp16, because
       Keras narrows at the boundary whatever the caller passes.
       Constructing with ``autocast=False`` and float32 coordinates helps
       without fixing it, 0.47 to 0.16 at fp16 and 1.99 to 0.56 at bf16,
       because the ``omega`` read is still narrowed. An fp16 consumer
       should keep this layer OUT of the reduced-precision region, running
       it under float32 and casting the result. That path is UNTESTED here.

       *float64 is exact.* ``build()`` computes the frequency table at
       float64 when the layer's ``variable_dtype`` is float64, and at
       float32 otherwise. Under a float64 policy the layer is then float64
       accurate end to end: against a float64 oracle at ``dim=64, ndim=3``
       with coordinates in ``[0, 64)``, the max abs error is exactly
       ``0.0``. It was ``4.36e-07`` while the table was float32 at every
       policy. The widening is CONDITIONAL on purpose, and why is recorded
       at the anchor in ``build()``.
    """

    def __init__(
            self,
            dim: int,
            ndim: int,
            max_wavelength: float = 10000.0,
            assert_positive: bool = True,
            **kwargs: Any
    ) -> None:
        """Validate the configuration and fix the width arithmetic.

        Every padding and per-axis width is decided here, from ``dim`` and
        ``ndim`` alone. The frequency ladder is created in :meth:`build`.

        :param dim: Width of the output.
        :type dim: int
        :param ndim: Number of coordinate axes.
        :type ndim: int
        :param max_wavelength: Base of the frequency ladder.
        :type max_wavelength: float
        :param assert_positive: Retained for config compatibility, performs
            no check.
        :type assert_positive: bool
        :param kwargs: Additional keyword arguments for the Layer base class.
        :type kwargs: Any
        :raises ValueError: If ``dim``, ``ndim`` or ``max_wavelength`` is not
            positive, or if ``dim`` is too small to give each axis one
            sin/cos pair.
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

        # Two separate padding terms. The first covers a dim that does not
        # divide by ndim. The second covers a per-axis width that is odd and
        # so cannot hold whole sin/cos pairs. See the class docstring.
        self.ndim_padding = dim % ndim
        dim_per_ndim = (dim - self.ndim_padding) // ndim
        self.sincos_padding = dim_per_ndim % 2
        self.padding = self.ndim_padding + self.sincos_padding * ndim

        # Width allocated to one coordinate axis. Always even.
        effective_dim_per_wave = (self.dim - self.padding) // ndim
        if effective_dim_per_wave <= 0:
            raise ValueError(f"dim ({dim}) too small for ndim ({ndim}). "
                             f"Need at least {ndim * 2} dimensions.")

        self.effective_dim_per_wave = effective_dim_per_wave

        # The frequency ladder is created in build().
        self.omega = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the fixed frequency ladder and check the input shape.

        :param input_shape: Shape of the coordinate input, ending in
            ``ndim``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the input is rank 1, or if its last dimension
            is not ``ndim``.
        """
        if self.built:
            return

        # Validate input shape
        if len(input_shape) < 2:
            raise ValueError(f"Input must be at least 2D, got shape {input_shape}")

        if input_shape[-1] != self.ndim:
            raise ValueError(f"Last dimension of input ({input_shape[-1]}) "
                             f"must match ndim ({self.ndim})")

        # DECISION plan-2026-07-31T210633-b63a35aa/D-009
        # Widen the frequency table to float64 ONLY when the weight can hold
        # float64. Do NOT make it unconditional: a float64 frequency rounded
        # to the nearest float32 is not the float32-computed frequency, and
        # that moved the FLOAT32 output at 10 of 10 corpora swept (measured,
        # D-008), breaking a hard bit-identity invariant. Do NOT key it off
        # `compute_dtype` either: under mixed policies the compute dtype is
        # narrow while `variable_dtype` is float32, and this table is a
        # WEIGHT. Measured on CPU against a float64 oracle at coords in
        # [0, 64), the float64-policy error goes 4.364258e-07 -> exactly 0.0.
        # Pinned by `test_float64_policy_accuracy_floor_is_machine_precision`.
        # See decisions.md D-009.
        arange_dtype = np.float64 if self.variable_dtype == "float64" else np.float32
        arange_vals = np.arange(0, self.effective_dim_per_wave, 2, dtype=arange_dtype)
        omega_vals = 1.0 / (self.max_wavelength ** (arange_vals / self.effective_dim_per_wave))

        # DECISION plan-2026-08-14T233721-d4f9beb2/D-027
        # `omega` is materialized by an INITIALIZER. Do NOT restore
        #     self.omega = self.add_weight(..., initializer="zeros")
        #     self.omega.assign(omega_vals)
        # Keras 3 runs the symbolic build inside a `StatelessScope` whenever a
        # sublayer is first reached from a parent's `call()`, which is every
        # real model, and that scope records the `.assign()` then discards it.
        # The table stays at its "zeros" initializer and the embedding
        # collapses to one constant vector.
        # Measured on CPU 2026-08-15: a direct `.build(...)` gives
        # `omega[0] == 1.0`; the same layer reached through a parent layer's
        # `call()` gave `omega[0] == 0.0` with the whole table all-zero.
        # Initializers run at variable-CREATION time and survive the scope.
        # Same defect and same fix as `rotary_position_embedding.py`, D-021.
        # `omega_vals` is a NumPy array, so closing over it is safe. A tensor
        # computed in `build()` would belong to the symbolic pass's scratch
        # FuncGraph and raise "out of scope" on the eager pass. The cast keeps
        # the float64 invariant above: `omega_vals` is already float64 exactly
        # when `variable_dtype` is. See decisions.md D-027.
        self.omega = self.add_weight(
            name="omega",
            shape=omega_vals.shape,
            initializer=lambda shape, dtype=None: keras.ops.cast(
                keras.ops.convert_to_tensor(omega_vals),
                dtype or self.variable_dtype,
            ),
            trainable=False,
        )

        super().build(input_shape)

    def call(
            self,
            coords: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Map coordinates onto the sine and cosine basis.

        :param coords: Coordinates of shape ``(..., ndim)``.
        :type coords: keras.KerasTensor
        :param training: Accepted for the Keras call signature and unused.
            This layer behaves the same in both modes.
        :type training: Optional[bool]
        :return: Embedding of shape ``(..., dim)``, in ``compute_dtype``.
            Its last ``padding`` channels are zero.
        :rtype: keras.KerasTensor
        """
        # `assert_positive` is retained as a config-compatible flag and no
        # longer triggers a runtime check. The previous implementation called
        # `ops.convert_to_numpy(ops.min(coords))` here, an eager host
        # materialization that breaks `@tf.function` and graph tracing. Do not
        # restore it. The originating plan directory is gone, so this comment
        # is the only record of why the flag is inert. Removed for
        # graph compatibility (DECISION plan_2026-06-15_9dbb87c1/D-001).

        # DECISION plan-2026-07-31T210633-b63a35aa/D-002
        # Compute the sinusoids in a never-narrowing working dtype, and cast
        # the frequency table to that same dtype. Three prohibitions, all
        # specific to THIS site.
        #
        # 1. Do NOT write `ops.cast(coords, "float32")`. A hard literal is
        #    matched against `self.omega`, which Keras autocasts to the active
        #    compute dtype on read, so the multiply below got mismatched
        #    operands and raised `InvalidArgumentError ... cannot compute Mul
        #    as input #1 was expected to be a float tensor but is a
        #    half/double/bfloat16 tensor [Op:Mul]` under mixed_float16,
        #    float64 and mixed_bfloat16. That killed this layer and every
        #    consumer of it (the embedding factory, TextEncoder, TextDecoder,
        #    SupernodePooling) at 3 of the 4 standard policies.
        # 2. Do NOT simplify to `ops.cast(coords, self.compute_dtype)`. That
        #    narrows the sinusoid math to fp16 or bf16. The measured cost is
        #    in the class docstring's note and is pinned by
        #    `test_fp16_large_position_error_bound_is_pinned`. The conditional
        #    below states the never-narrow rule rather than inheriting it from
        #    `variable_dtype`.
        # 3. Do NOT drop the `ops.cast(..., self.compute_dtype)` at the
        #    return. Without it this layer returns float32 under
        #    mixed_float16, a wrong output dtype every downstream `add` and
        #    `concatenate` has to absorb.
        #
        # None of this recovers fp16 or bf16 accuracy at large positions.
        # Keras has already narrowed `coords` at the autocast boundary before
        # `call()` runs. See decisions.md D-002.
        work_dtype = "float64" if self.compute_dtype == "float64" else "float32"
        coords = ops.cast(coords, work_dtype)

        # (..., ndim) -> (..., ndim, 1), so the ladder broadcasts.
        coords_expanded = ops.expand_dims(coords, axis=-1)

        # (freq_dim,) -> (1, ..., 1, freq_dim).
        omega_shape = [1] * (len(coords.shape) - 1) + [self.omega.shape[0]]
        omega_expanded = ops.reshape(ops.cast(self.omega, work_dtype), omega_shape)

        # (..., ndim, freq_dim).
        freqs = coords_expanded * omega_expanded

        sin_vals = ops.sin(freqs)
        cos_vals = ops.cos(freqs)

        # (..., ndim, 2 * freq_dim).
        emb = ops.concatenate([sin_vals, cos_vals], axis=-1)

        # Flatten the trailing two axes into one. Take the leading dims from
        # ops.shape, which gives runtime values and never None, and let a
        # single -1 infer the rest. Graph-safe at any rank and with a dynamic
        # batch axis.
        emb = ops.reshape(emb, (*ops.shape(coords)[:-1], -1))

        # Build the pad shape the same way: dynamic leading dims from
        # ops.shape plus a static last dim, so ops.zeros stays graph-safe
        # under symbolic tracing.
        if self.padding > 0:
            padding = ops.zeros(
                (*ops.shape(emb)[:-1], self.padding), dtype=emb.dtype
            )
            emb = ops.concatenate([emb, padding], axis=-1)

        # Back to the caller's compute dtype. Required, see D-002 above.
        return ops.cast(emb, self.compute_dtype)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Report the output shape, with ``dim`` as the last dimension.

        :param input_shape: Shape of the coordinate input.
        :type input_shape: Tuple[Optional[int], ...]
        :return: The same shape with the last dimension replaced by ``dim``.
        :rtype: Tuple[Optional[int], ...]
        """
        input_shape_list = list(input_shape)
        return tuple(input_shape_list[:-1] + [self.dim])

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor arguments for serialization.

        :return: Configuration dictionary carrying every ``__init__``
            argument, including the inert ``assert_positive``.
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
