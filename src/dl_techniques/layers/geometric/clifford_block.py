"""
CliffordNet block and constituent primitives.

.. warning::

   **HIGH blast radius. Any edit to this module must run EVERY importer's test
   suite, not just ``tests/test_layers/test_geometric/``.**

   The importers are not local and the diff size does not predict the risk.
   Re-derive the list before editing::

       grep -rln "geometric.clifford_block" src/ tests/

   Measured 2026-08-10 (9 source modules across 4 packages, plus 3 test
   modules): ``models/cliffordnet/{lm,model}.py``,
   ``models/clip/clifford_clip.py``,
   ``models/video_jepa/{encoder,predictor}.py``,
   ``train/cliffordnet/{infer_cliffordnet_nlp,train_cliffordnet_nlp,
   train_downsampling_techniques}.py``. The corresponding suites are
   ``tests/test_layers/test_geometric/``, ``tests/test_models/test_cliffordnet/``,
   ``tests/test_models/test_clip/`` and ``tests/test_models/test_video_jepa/``.

   This warning exists because it was learned the expensive way: an edit to this
   file (plan-2026-08-10-3649c19e/iter-1/step-2) was scored ``radius:MED`` on
   diff size, so ``test_clip`` and ``test_video_jepa`` were never run — and the
   adversarial review found a red test in the first unrun one it opened. See
   decisions.md D-033.

Implements the geometric-algebra vision block from
    Zhongping Ji, "CliffordNet: All You Need is Geometric Algebra",
    arXiv:2601.06793v2 (2026).  Reference code: github.com/ParaMind2025/CAN
    (``CliffordInteraction_PyTorch`` / ``CliffordAlgebraBlock`` in ``model.py``).

================================================================================
Theory -- geometric algebra as a single, unified interaction primitive
================================================================================

Motivation
----------
Modern vision backbones factor each block into two engineered stages: a spatial
token mixer (self-attention or convolution) and a channel mixer (a heavy
Feed-Forward Network / MLP).

CliffordNet rejects that decomposition.  It argues
that ONE algebraic operation, the Clifford *geometric product* can carry
both roles simultaneously, at strictly linear cost, so no FFN is required.  The
guiding thesis is that "global understanding is an emergent property of rigorous
local processing": dense, algebraically-complete local interaction stands in for
both global attention and channel-mixing MLPs.

The geometric product
---------------------
For two multivectors (here, per-pixel channel vectors) ``u`` and ``v`` the
geometric product splits into a symmetric and an antisymmetric part::

    u v  =  u . v      +      u ^ v
            (inner)          (wedge / exterior)
            "coherence"      "structure"

* The **inner product** ``u . v`` is the generalized dot product.  It measures
  feature *coherence* / alignment -- how strongly the detail stream agrees with
  its local context.  CliffordNet realizes it as a gated Hadamard term,
  ``SiLU(u * v)``, i.e. an alignment-controlled diffusion / gate.

* The **wedge product** ``u ^ v`` is the antisymmetric bivector -- the oriented
  plane (area element) spanned by ``u`` and ``v``.  It measures *structural
  variation*: orthogonality, orientation, a "geometric torque / vorticity" that
  fires exactly on the edges and texture where local context diverges from the
  center.  Ordinary dot-product attention keeps only the symmetric (inner) part
  and DISCARDS this bivector; retaining it is what the paper calls
  **algebraic completeness**.

Dual-stream geometric block
---------------------------
Each block derives two streams from the normalized input ``X_norm``:

* **Detail stream** (high frequency, no spatial mixing):
  ``Z_det = Linear(X_norm)`` -- a 1x1 pointwise projection.
* **Context stream** (local aggregation):
  ``Z_ctx = act(Norm(DWConv(DWConv(X_norm))))`` -- two stacked depthwise
  convolutions aggregating local structure.  With the default
  ``context_kernel_size=3`` the effective receptive field is 5x5.  (The paper
  states 7x7 in Sec. 5.3; that is an error -- two stacked KxK convolutions give
  (2K-1)x(2K-1).  The paper also describes the two convolutions as "separated
  by non-linear activation" in Sec. 3.4, but neither Algorithm 1 nor the
  reference ``get_context_local`` has an activation between them, so there is
  none here either.)

An optional *differential* coupling sharpens the interaction::

    ctx_mode="diff":  Z_ctx <- Z_ctx - Z_det
    ctx_mode="abs" :  Z_ctx                        (pure aggregation)

The two streams then interact through the geometric product to produce the
geometric feature ``G_feat`` that drives the state update.

Sparse rolling geometric product (linear complexity)
----------------------------------------------------
A full channel-pairwise product is O(D^2).  CliffordNet samples only a few
diagonals of that interaction matrix via cyclic channel shifts (rolls), giving
O(N . D . |shifts|) -- linear in both tokens ``N`` and channels ``D``.  For each
offset ``s`` in ``shifts`` (rolling ``Z_ctx`` by ``s`` along the channel axis)::

    dot_s[c]   = act( Z_det[c] * Z_ctx[(c-s) mod D] )        # inner / coherence
    wedge_s[c] = Z_det[c] * Z_ctx[(c-s) mod D]
               - Z_ctx[c] * Z_det[(c-s) mod D]               # wedge / bivector

The per-shift dot and wedge tensors are concatenated and projected back to ``D``
channels by a learnable Dense ``P``.  Exponentially spaced shifts (1, 2, 4, 8,
...) impose a ring topology with logarithmic mixing range.

Note on the shift direction: Eq. 11 of the paper indexes the context at
``(c + s) % D``, while both Algorithm 1 and the reference implementation use
``roll(C, s)``, i.e. ``(c - s) % D``.  We follow the code, not Eq. 11.  The two
differ only by a relabelling of the shift set (and a sign on the bivector),
which the learnable projection ``P`` absorbs.

Gated Geometric Residual (GGR) -- an Euler step of a feature ODE
---------------------------------------------------------------
The block treats depth as time and takes a first-order Euler step of a
continuous geometric evolution ``dH/dt = f(H, G_feat)``::

    H_out = H_prev + gamma * ( SiLU(H_norm) + alpha * G_feat )

* ``gamma`` -- LayerScale, a per-channel scale initialized ~ 0 so the block
  starts near identity (stable very deep stacks).
* ``SiLU(H_norm)`` -- conditions the identity / state path.
* ``alpha = sigmoid(Gate([H_norm, G_feat]))`` -- a learned gate blending the
  identity path with the injected geometric interaction.

Eq. 13 of the paper writes the conditioning term as ``SiLU(H_{l-1})`` (the
*un-normalized* previous state), while Algorithm 1 line 36 and the reference
implementation both use ``SiLU(X_ln)`` (the *normalized* state).  We follow
Algorithm 1 / the reference; do not "fix" this against Eq. 13.

Global context branch (optional)
--------------------------------
A whole-image summary ``C_glo = GlobalAvgPool(X_norm)`` runs the same geometric
product (hardcoded ``shifts=[1, 2]``, ``cli_mode="full"``, differential context)
and is superposed onto the local ``G_feat``, adding multi-scale awareness when
enabled.

Efficiency
----------
With ZERO FFN blocks, CliffordNet sets a new parameter-efficiency Pareto
frontier on CIFAR-100: ~1.4M params -> 77.82% (vs ResNet-18's 76.75% at ~8x
more params); ~2.6M -> 79.05% (beating MobileNetV2 and ViT-Tiny at similar
size); larger variants surpass ResNet-50 / DenseNet-121 at a fraction of the
parameters.

================================================================================
Implementation notes, known behaviours and deviations
================================================================================

1. ``GatedGeometricResidual`` returns ONLY the gamma-scaled term
   ``gamma * (SiLU(H_norm) + alpha * G_feat)``; the residual add ``H_prev + ...``
   and any stochastic-depth (drop-path) are performed EXTERNALLY by the caller /
   model, so the computation graph is explicit and manually inspectable.  The
   reference does ``x = shortcut + drop_path(gamma * x_mixed)`` inside the
   block; the split is deliberate.  Do not re-inline either op here.

2. ``ctx_mode`` has NO EFFECT on the wedge branch.  With
   ``W(u, v) = u * T_s(v) - v * T_s(u)`` and ``v = Z_ctx - Z_det``, the
   self-terms cancel exactly by antisymmetry::

       W(det, ctx - det) = det*T(ctx) - det*T(det) - ctx*T(det) + det*T(det)
                         = det*T(ctx) - ctx*T(det) = W(det, ctx)

   Consequences: ``cli_mode="wedge"`` makes ``ctx_mode`` completely inert (the
   two settings produce bit-identical models), and in ``cli_mode="full"`` the
   differential context only reaches the inner/dot term.  This is inherited
   from the reference implementation, so the paper's Table 4 "Wedge-Only
   (Differential Mode)" row is the same model as its absolute-mode counterpart,
   and the ~1.4% diff-vs-abs gap in Table 3 is attributable to the inner term
   alone.  The constructor warns for the fully-inert combination.

3. Global branch superposition weight.  ``G_feat + G_glo`` uses an implicit
   beta = 1, matching ``CliffordAlgebraBlock`` in the reference.  The repo's
   other variant (``gffn.py``, ``gffn_mode="h"``) instead learns
   ``beta = nn.Parameter([0.5])``, and Eq. 7 of the paper carries beta
   explicitly.  If a learnable superposition weight is wanted, add it as a new
   opt-in kwarg rather than changing the default.

4. Weight initialization does not reproduce the paper.  The reference applies
   ``trunc_normal_(std=0.02)`` to every Conv2d/Linear and zeroes biases; the
   default here is Keras' ``glorot_uniform``.  Because this block's output is
   quadratic in its activations, init scale matters more than usual.  Pass
   ``kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02)`` for
   reproduction runs.  The default is left alone to avoid silently changing
   existing checkpoints' training dynamics.

5. Memory.  For ``cli_mode="full"`` the concatenation materialises a
   ``(B, H, W, 2*|shifts|*D)`` tensor that is kept for the backward pass (10x
   activation blowup at ``|shifts|=5``).  Because
   ``proj(concat(c_1..c_k)) == sum_i c_i @ W_i``, the projection could be split
   along its input axis and accumulated per component, removing the concat
   entirely.  Not done here: it changes float summation order, so it is a
   behavioural (if numerically tiny) change and belongs behind a flag with a
   tolerance-based parity test.

6. ``@keras.saving.register_keras_serializable()`` is intentionally left without
   a ``package=`` argument.  Adding one would change the registered name from
   ``Custom>ClassName`` and break by-name loading of existing ``.keras`` files.

Key primitives
--------------
- :class:`SparseRollingGeometricProduct`  -- shifted dot + wedge interaction
- :class:`GatedGeometricResidual`         -- GGR update with LayerScale
- :class:`CliffordNetBlock`               -- full isotropic block (no FFN)
- :class:`CausalCliffordNetBlock`         -- autoregressive variant

Removed surface
---------------
``CliffordNetBlockDSv2`` and ``CausalCliffordNetBlockDSv2`` -- the strided
downsampling design-space siblings (decoupled stream/skip pools, pyramid-diff
context) -- were deleted together with their entire consumer closure by
plan-2026-08-10-3649c19e/iter-1/step-2. Do NOT re-add them: the experiment they
served was declared dead by the owner, and their two model consumers
(``cliffordnet/embedding_unet.py``, ``cliffordnet/lmunet.py``) plus four
trainers no longer exist. See decisions.md D-005/D-006. ``.keras`` checkpoints
that serialized either registered class name can no longer be loaded.
"""

from __future__ import annotations

import numbers
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
)

import keras
from keras import initializers, regularizers

# ---------------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from ..norms.factory import create_normalization_layer

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

CliMode = Literal["inner", "wedge", "full"]
CtxMode = Literal["diff", "abs"]
InputMode = Literal["image", "sequence"]

_CLI_MODES: Tuple[str, ...] = ("inner", "wedge", "full")
_CTX_MODES: Tuple[str, ...] = ("diff", "abs")
_INPUT_MODES: Tuple[str, ...] = ("image", "sequence")

# Global-branch constants matching the original implementation.
_GLOBAL_SHIFTS: List[int] = [1, 2]
_GLOBAL_CLI_MODE: CliMode = "full"
# SparseRollingGeometricProduct drops shifts with ``s >= channels``, so the
# global branch needs strictly more channels than its largest shift for all of
# _GLOBAL_SHIFTS to survive.
_MIN_GLOBAL_CHANNELS: int = max(_GLOBAL_SHIFTS) + 1

# Normalization types measured to reduce over the SEQUENCE axis, i.e. every
# position's output depends on all positions. Selecting one of these on a causal
# block silently leaks the future into the past.
_SEQUENCE_REDUCING_NORMS: FrozenSet[str] = frozenset({
    "batch_norm",              # reduces over (B, H, W) at training=True
    "bias_free_batch_norm",    # same reduction-axis logic, at training=True
    "global_response_norm",    # reduces over (H, W) UNCONDITIONALLY - not training-gated
})


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _activation_spec(activation: Any) -> Any:
    """Canonicalise an activation spec for storage on the layer.

    Returns the spec in a form that :func:`_serialize_activation` can round-trip:
    ``None`` and strings pass through; a serialized dict (as produced by
    deserialization of a saved config) is turned back into a callable; anything
    else is returned unchanged.

    :param activation: String name, ``None``, serialized dict, or callable.
    :return: Canonical activation spec.
    """
    if activation is None or isinstance(activation, str):
        return activation
    if isinstance(activation, dict):
        return keras.activations.deserialize(activation)
    return activation


def _resolve_activation(activation: Any) -> Callable[[Any], Any]:
    """Resolve an activation spec to a callable.

    Strings are resolved via ``keras.activations.get``; ``None`` maps to
    identity (linear); callables are returned as-is.  Stateful activation
    *layers* are rejected: they would create their weights during ``call()``
    instead of ``build()``, which breaks ``.keras`` weight loading.

    :param activation: String name, ``None``, serialized dict, or callable.
    :return: A callable applying the activation.
    :raises ValueError: If ``activation`` is a ``keras.layers.Layer``.
    """
    if isinstance(activation, keras.layers.Layer):
        raise ValueError(
            "Activation must be a string name or a plain callable, not a "
            f"keras Layer instance ({type(activation).__name__}). Layer "
            "activations may own weights, which would be created during "
            "call() rather than build() and would not survive a .keras "
            "round-trip. Use e.g. 'leaky_relu' or keras.activations.silu."
        )
    if activation is None:
        return keras.activations.linear
    if isinstance(activation, str):
        return keras.activations.get(activation)
    if isinstance(activation, dict):
        return keras.activations.deserialize(activation)
    return activation


def _serialize_activation(activation: Any) -> Any:
    """Serialize an activation spec for ``get_config``.

    ``None`` and strings pass through unchanged; callables are serialized via
    ``keras.saving.serialize_keras_object`` so that a config containing a raw
    function object is still JSON-serialisable.

    :param activation: Canonical activation spec.
    :return: JSON-serialisable representation.
    """
    if activation is None or isinstance(activation, str):
        return activation
    return keras.saving.serialize_keras_object(activation)


def _validate_shifts(shifts: Any) -> List[int]:
    """Validate and normalise a shift-offset list.

    Rejects ``s <= 0``: ``s = 0`` makes the wedge term identically zero and
    wastes a slot in the projection input; negative shifts are accepted by
    ``keras.ops.roll`` but are almost certainly unintended.  Accepts any
    integral type (including numpy integers) but not ``bool``.

    :param shifts: Candidate sequence of shift offsets.
    :return: Shifts as a list of Python ints.
    :raises ValueError: If ``shifts`` is empty or contains a non-positive-int.
    """
    if isinstance(shifts, (str, bytes)) or not isinstance(shifts, Sequence):
        raise ValueError(
            f"shifts must be a sequence of ints >= 1; got {shifts!r}"
        )
    if not shifts:
        raise ValueError("shifts must be a non-empty sequence")
    normalised: List[int] = []
    for s in shifts:
        if isinstance(s, bool) or not isinstance(s, numbers.Integral) or s < 1:
            raise ValueError(
                f"shifts must be a sequence of ints >= 1; got {list(shifts)!r}"
            )
        normalised.append(int(s))
    return normalised


def _left_padded_shape(
    shape: Tuple[Optional[int], ...], pad: int
) -> Tuple[Optional[int], ...]:
    """Return ``shape`` with the W axis (index 2) grown by ``pad``.

    A ``None`` (dynamic) W axis stays ``None`` rather than collapsing to
    ``pad``, so downstream shape consumers see the truth.

    :param shape: 4-D shape ``(B, H, W, D)``.
    :param pad: Number of positions added to the W axis.
    :return: Shape with W increased by ``pad``.
    """
    w = shape[2]
    return (shape[0], shape[1], None if w is None else w + pad, shape[3])


# ---------------------------------------------------------------------------
# SparseRollingGeometricProduct
# ---------------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class SparseRollingGeometricProduct(keras.layers.Layer):
    """Sparse rolling realisation of the Clifford geometric product.

    For each shift offset *s* in ``shifts`` (filtered to ``s < channels``),
    computes element-wise scalar (dot) and/or bivector (wedge) interaction
    terms between a detail stream Z_det and a context stream Z_ctx, then
    projects the concatenated result back to ``channels``. The dot component
    is ``D_s[c] = dot_activation(Z_det[c] * Z_ctx[(c-s) % D])`` and the wedge
    component is
    ``W_s[c] = Z_det[c] * Z_ctx[(c-s) % D] - Z_ctx[c] * Z_det[(c-s) % D]``.

    This layer performs no context differencing of its own; the caller decides
    what ``Z_ctx`` is (the reference implementation folds ``ctx_mode`` into the
    interaction module, this port applies it in
    :class:`CliffordNetBlock`).

    **Architecture Overview:**

    .. code-block:: text

        ┌────────────────┐  ┌────────────────┐
        │ Z_det [B,H,W,D]│  │ Z_ctx [B,H,W,D]│
        └───────┬────────┘  └───────┬────────┘
                │                   │
                └────────┬──────────┘
                         ▼
        ┌────────────────────────────────────┐
        │  For each shift s:                 │
        │  ├─ Wedge: Z_det·roll(Z_ctx,s)     │
        │  │         - Z_ctx·roll(Z_det,s)   │
        │  └─ Dot:  SiLU(Z_det·roll(Z_ctx,s))│
        └───────────────┬────────────────────┘
                        ▼
        ┌────────────────────────────────────┐
        │  Concatenate all components        │
        └───────────────┬────────────────────┘
                        ▼
        ┌────────────────────────────────────┐
        │  Dense projection → [B,H,W,D]      │
        └────────────────────────────────────┘

    :param channels: Feature dimensionality D.
    :type channels: int
    :param shifts: Cyclic channel offsets; values ``>= channels`` are filtered.
    :type shifts: Sequence[int]
    :param cli_mode: Components to retain
        (``"inner"``, ``"wedge"``, ``"full"``). Defaults to ``"full"``.
    :type cli_mode: CliMode
    :param use_bias: Whether the projection Dense uses a bias.
    :type use_bias: bool
    :param dot_activation: Activation applied to the inner/dot term. Defaults
        to ``"silu"``, which reproduces the reference implementation. Must be a
        string name, ``None`` (identity), or a stateless callable.
    :type dot_activation: Any
    :param kernel_initializer: Initializer for the projection kernel.
    :type kernel_initializer: Any
    :param bias_initializer: Initializer for the projection bias.
    :type bias_initializer: Any
    :param kernel_regularizer: Regularizer for the projection kernel.
    :type kernel_regularizer: Optional[Any]
    :param bias_regularizer: Regularizer for the projection bias.
    :type bias_regularizer: Optional[Any]
    :param kwargs: Passed to ``keras.layers.Layer``.
    """

    def __init__(
        self,
        channels: int,
        shifts: Sequence[int],
        cli_mode: CliMode = "full",
        use_bias: bool = True,
        dot_activation: Any = "silu",
        kernel_initializer: Any = "glorot_uniform",
        bias_initializer: Any = "zeros",
        kernel_regularizer: Optional[Any] = None,
        bias_regularizer: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if cli_mode not in _CLI_MODES:
            raise ValueError(
                f"cli_mode must be one of {_CLI_MODES}, got {cli_mode!r}"
            )
        requested_shifts = _validate_shifts(shifts)

        self.channels = channels
        # Filter out offsets >= channels: a full cyclic roll contributes no new
        # information (matches CliffordInteraction_PyTorch).
        self.shifts = [s for s in requested_shifts if s < channels]
        if not self.shifts:
            raise ValueError(
                f"All provided shifts {requested_shifts} are >= channels "
                f"({channels}). No valid shifts remain after filtering."
            )
        dropped = [s for s in requested_shifts if s >= channels]
        if dropped:
            logger.warning(
                "SparseRollingGeometricProduct dropping shifts %s "
                "(>= channels=%d); kept shifts=%s",
                dropped, channels, self.shifts,
            )

        self.cli_mode = cli_mode
        self.use_bias = use_bias
        # The default "silu" reproduces the reference implementation exactly.
        self.dot_activation = _activation_spec(dot_activation)
        self._dot_activation_fn = _resolve_activation(self.dot_activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        # Number of concatenated channels before projection.
        multiplier = 2 if cli_mode == "full" else 1
        self._proj_input_dim = multiplier * len(self.shifts) * channels

        self.proj = keras.layers.Dense(
            channels,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name="proj",
        )

        self._input_shape_for_build: Optional[Tuple] = None

    # ------------------------------------------------------------------

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the projection layer.

        Keras passes the shape of the FIRST positional ``call`` argument here
        (``z_det``), because ``build`` declares a single parameter; both streams
        are required to have the same shape.

        :param input_shape: Shape of a *single* input tensor ``(B, H, W, D)``.
        """
        self._input_shape_for_build = tuple(input_shape)
        self.proj.build((*input_shape[:-1], self._proj_input_dim))
        super().build(input_shape)

    def get_build_config(self) -> Dict[str, Any]:
        """Return the shape needed to rebuild this layer.

        Explicit because ``call`` takes two positional tensors; relying on
        Keras' auto-captured shapes dict here is version-sensitive.
        """
        if self._input_shape_for_build is not None:
            return {"input_shape": self._input_shape_for_build}
        return {}

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Rebuild from :meth:`get_build_config` output."""
        if "input_shape" in config:
            self.build(tuple(config["input_shape"]))

    # ------------------------------------------------------------------

    def call(
        self,
        z_det: keras.KerasTensor,
        z_ctx: keras.KerasTensor,
        **kwargs: Any,
    ) -> keras.KerasTensor:
        """Compute sparse geometric product and project.

        :param z_det: Detail stream  ``(B, H, W, D)``.
        :param z_ctx: Context stream ``(B, H, W, D)``.
        :return: Projected interaction tensor ``(B, H, W, channels)``.
        """
        components: List[keras.KerasTensor] = []

        for s in self.shifts:
            z_ctx_s = keras.ops.roll(z_ctx, shift=s, axis=-1)

            if self.cli_mode in ("wedge", "full"):
                # Bivector: anti-symmetric cross-term. z_det_s is only needed
                # for the wedge branch; skip it for cli_mode='inner'.
                z_det_s = keras.ops.roll(z_det, shift=s, axis=-1)
                components.append(z_det * z_ctx_s - z_ctx * z_det_s)

            if self.cli_mode in ("inner", "full"):
                # Scalar: gated inner product.
                components.append(self._dot_activation_fn(z_det * z_ctx_s))

        # See module docstring note 5: this concat is the memory hot spot and
        # is equivalent to a sum of per-component matmuls.
        g_raw = keras.ops.concatenate(components, axis=-1)
        return self.proj(g_raw)

    # ------------------------------------------------------------------

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape.

        :param input_shape: Shape of one input stream ``(B, H, W, D)``.
        :return: Output shape ``(B, H, W, channels)``.
        """
        return (*input_shape[:-1], self.channels)

    # ------------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        """Return serialisable configuration.

        :return: Dictionary with all constructor arguments.
        """
        config = super().get_config()
        config.update(
            {
                "channels": self.channels,
                # Stores the post-filter shift list (s < channels). Round-trip
                # serialisation is idempotent for a fixed `channels`; if the
                # caller reconstructs with a different `channels`, any shifts
                # the original constructor dropped are not recoverable.
                "shifts": list(self.shifts),
                "cli_mode": self.cli_mode,
                "use_bias": self.use_bias,
                "dot_activation": _serialize_activation(self.dot_activation),
                "kernel_initializer": initializers.serialize(self.kernel_initializer),
                "bias_initializer": initializers.serialize(self.bias_initializer),
                "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
                "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            }
        )
        return config


# ---------------------------------------------------------------------------
# GatedGeometricResidual
# ---------------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class GatedGeometricResidual(keras.layers.Layer):
    """Gated Geometric Residual (GGR) update.

    Implements the Euler-discretised ODE step
    ``H_out = H_prev + gamma * (SiLU(H_norm) + alpha * G_feat)``, where
    ``alpha`` is a learned sigmoid gate on ``concat(H_norm, G_feat)`` and
    ``gamma`` is a LayerScale vector initialised near zero. This layer returns
    ONLY the LayerScale-gated term; the residual add and any stochastic-depth
    op are external, model-level operations.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────┐   ┌──────────────────┐
        │ H_norm [B,H,W,D] │   │ G_feat [B,H,W,D] │
        └────────┬─────────┘   └─────────┬────────┘
                 │                       │
                 │   concat(H_norm, G_feat)
                 │        ┌────────────┐ │
                 │        │ Gate:      │ │
                 │        │ α=sigmoid()│ │
                 │        └─────┬──────┘ │
                 ▼              ▼        ▼
          ┌───────────┐  ┌───────────────────┐
          │  SiLU(H)  │  │     α · G_feat    │
          └─────┬─────┘  └─────────┬─────────┘
                └───────┬──────────┘
                        ▼
                ┌───────────────┐
                │  γ · ( sum )  │   LayerScale
                └───────┬───────┘
                        ▼
          Output: residual TERM  [B, H, W, D]
          (residual add + drop-path are EXTERNAL)

    :param channels: Feature dimensionality D.
    :type channels: int
    :param layer_scale_init: Initial LayerScale gamma. Defaults to 1e-5.
    :type layer_scale_init: float
    :param use_bias: Whether the gate Dense uses an additive bias. Defaults to
        ``True`` (matches the reference ``gate_fc``).
    :type use_bias: bool
    :param gate_activation: Activation producing ``alpha``. Defaults to
        ``"sigmoid"``.
    :type gate_activation: Any
    :param feature_activation: Activation on the identity/state path. Defaults
        to ``"silu"``.
    :type feature_activation: Any
    :param use_gate: Whether to apply the multiplicative ``alpha * G_feat``
        gate. ``False`` drops it (``feat + G_feat``), which the degree-1
        homogeneous / bias-free denoiser path requires. The ``gate_dense``
        sub-layer is still constructed and saved either way, so the ``.keras``
        weight layout is identical at both settings. Defaults to ``True``.
    :type use_gate: bool
    :param kernel_initializer: Initializer for the gate kernel.
    :type kernel_initializer: Any
    :param bias_initializer: Initializer for the gate bias.
    :type bias_initializer: Any
    :param kernel_regularizer: Regularizer for the gate kernel.
    :type kernel_regularizer: Optional[Any]
    :param bias_regularizer: Regularizer for the gate bias.
    :type bias_regularizer: Optional[Any]
    :param kwargs: Passed to ``keras.layers.Layer``.
    """

    def __init__(
        self,
        channels: int,
        layer_scale_init: float = 1e-5,
        use_bias: bool = True,
        gate_activation: Any = "sigmoid",
        feature_activation: Any = "silu",
        use_gate: bool = True,
        kernel_initializer: Any = "glorot_uniform",
        bias_initializer: Any = "zeros",
        kernel_regularizer: Optional[Any] = None,
        bias_regularizer: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")

        self.channels = channels
        self.layer_scale_init = layer_scale_init
        self.use_bias = use_bias
        # Defaults ("sigmoid"/"silu", use_gate=True) reproduce the reference
        # GGR update.
        self.gate_activation = _activation_spec(gate_activation)
        self.feature_activation = _activation_spec(feature_activation)
        self.use_gate = use_gate
        self._gate_activation_fn = _resolve_activation(self.gate_activation)
        self._feature_activation_fn = _resolve_activation(self.feature_activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        # Learned gate: Dense(2C -> C) followed by the gate activation.
        # NOTE: built unconditionally (even when use_gate=False) so a
        # use_gate=False model (e.g. the homogeneous Clifford denoiser) keeps a
        # stable weight layout for .keras checkpoint round-trips. When use_gate
        # is False the layer is inert (never referenced in call()).
        self.gate_dense = keras.layers.Dense(
            channels,
            use_bias=self.use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name="gate_dense",
        )
        # DECISION plan-2026-08-10T130454-3649c19e/D-008: `use_gate` is a real,
        # live, tested constructor kwarg and STAYS. Do NOT delete it "to follow
        # the rewrite's intent" (the reading recorded in decisions.md D-007):
        # that plan entry rested on a measured-false premise -- that this class
        # had lost `use_gate` and that `clifford_rnn.py` therefore raised a
        # construction-time TypeError. Measured 2026-08-10: `CliffordRNN(units=8)`
        # constructs and runs. Removing the flag would strip a documented
        # checkpoint-stability pattern and its regression tests
        # (TestInertGateNotTrainable) from working code.
        # Do NOT re-introduce a `_LEGACY_CONFIG_KEYS`-style shim listing it
        # either: a module-level tuple with no use site and no `from_config`
        # override documents a compatibility promise the code does not keep.
        #
        # DECISION plan-2026-07-22T090932-e433f233/D-001: when use_gate=False the
        # gate is inert, so its kernel/bias receive no gradient and Keras emits a
        # "Gradients do not exist for variables [...gate_dense...]" UserWarning
        # once per training run (42 entries for the bias-free Clifford U-Net).
        # Marking the inert sub-layer non-trainable removes those variables from
        # model.trainable_variables, so the optimizer never sees them and
        # _filter_empty_gradients() never warns.
        #
        # Do NOT "simplify" this by deleting the sub-layer or building it
        # conditionally: `weights` is independent of `trainable`
        # (keras/src/layers/layer.py:632-652), so the variables are still SAVED
        # and the .keras weight layout stays byte-identical — which is the whole
        # point of building it unconditionally above.
        #
        # DECISION plan-2026-07-22T090932-e433f233/D-003: placement is
        # load-bearing but subtle. This works PRE-build because
        # Layer._track_variable() (keras/src/layers/layer.py:1316-1322) applies
        # `if not self.trainable: variable.trainable = False` to every variable
        # as it is created, so the flag propagates to weights that do not exist
        # yet. The `trainable` SETTER alone (layer.py:564-582) would not — it
        # only walks variables that already exist.
        #
        # DECISION plan-2026-07-22T090932-e433f233/D-006: known, accepted
        # consequence — `model.trainable = True` (the standard unfreeze idiom,
        # used at src/train/bfunet/variance_probe.py:177) RE-ENABLES gate_dense
        # and brings the warning back: the setter recurses into `_layers`
        # (layer.py:581-582) and has no knowledge of `use_gate`. Re-apply this
        # guard manually after any global unfreeze.
        if not self.use_gate:
            self.gate_dense.trainable = False

        self.gamma: Optional[keras.Variable] = None

    # ------------------------------------------------------------------

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build LayerScale and the gate projection.

        :param input_shape: Shape of a single input stream ``(B, H, W, D)``.
        """
        if input_shape[-1] is not None and input_shape[-1] != self.channels:
            raise ValueError(
                f"{type(self).__name__} expected last dim == channels="
                f"{self.channels}, got input_shape[-1]={input_shape[-1]}."
            )
        self.gamma = self.add_weight(
            name="gamma",
            shape=(self.channels,),
            initializer=initializers.Constant(self.layer_scale_init),
            trainable=True,
        )
        self.gate_dense.build((*input_shape[:-1], 2 * self.channels))
        super().build(input_shape)

    # ------------------------------------------------------------------

    def call(
        self,
        h_norm: keras.KerasTensor,
        g_feat: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Apply GGR update.

        :param h_norm: Normalised input features ``(B, H, W, D)``.
        :param g_feat: Geometric interaction features ``(B, H, W, D)``.
        :param training: Whether in training mode (unused; kept so callers can
            pass it uniformly).
        :return: Scaled residual term ``(B, H, W, D)``; caller adds to H_prev.
        """
        feat = self._feature_activation_fn(h_norm)
        if self.use_gate:
            gate_input = keras.ops.concatenate([h_norm, g_feat], axis=-1)
            alpha = self._gate_activation_fn(self.gate_dense(gate_input))
            h_mix = feat + alpha * g_feat
        else:
            # The multiplicative alpha*g_feat gate is degree-2 in the input and
            # breaks strict degree-1 homogeneity (Miyasawa). Do NOT keep it
            # here — use g_feat directly on the homogeneous path.
            h_mix = feat + g_feat
        return h_mix * self.gamma

    # ------------------------------------------------------------------

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape.

        :param input_shape: Shape of a single input stream ``(B, H, W, D)``.
        :return: Output shape ``(B, H, W, channels)``.
        """
        return (*input_shape[:-1], self.channels)

    # ------------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        """Return serialisable configuration.

        :return: Dictionary with all constructor arguments.
        """
        config = super().get_config()
        config.update(
            {
                "channels": self.channels,
                "layer_scale_init": self.layer_scale_init,
                "use_bias": self.use_bias,
                "gate_activation": _serialize_activation(self.gate_activation),
                "feature_activation": _serialize_activation(
                    self.feature_activation
                ),
                "use_gate": self.use_gate,
                "kernel_initializer": initializers.serialize(self.kernel_initializer),
                "bias_initializer": initializers.serialize(self.bias_initializer),
                "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
                "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            }
        )
        return config


# ---------------------------------------------------------------------------
# CliffordNetBlock
# ---------------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class CliffordNetBlock(keras.layers.Layer):
    """Full isotropic CliffordNet block (no FFN).

    Implements the geometric-algebra vision block from arXiv:2601.06793v2.
    A dual-stream architecture generates detail ``Z_det = Linear(X_norm)`` and
    context ``Z_ctx = act(Norm(DWConv(DWConv(X_norm))))`` streams, optionally
    subtracting the detail stream from the context (``ctx_mode="diff"``). The
    streams interact via a sparse rolling geometric product and are combined
    through a Gated Geometric Residual (GGR) update. ``call()`` returns ONLY
    this transformed term (``h_mix``); the residual add is an external,
    model-level op.

    .. note::

        ``ctx_mode`` does not affect the wedge branch at all: the self-terms
        cancel by antisymmetry, so ``cli_mode="wedge"`` makes ``ctx_mode``
        inert and ``cli_mode="full"`` only differences the inner term. See
        note 2 in the module docstring.

    .. note::

        When ``use_global_context=True``, the global branch uses fixed
        ``shifts=[1, 2]``, ``cli_mode="full"``, and differential context
        regardless of the caller's ``shifts`` / ``cli_mode`` / ``ctx_mode``
        settings, and superposes with an implicit weight of 1. The global
        branch is a compact whole-image summary and deliberately decouples its
        hyperparameters from the local branch.

    **Architecture Overview:**

    .. code-block:: text

        ┌────────────────────────────────┐
        │  X_prev  [B, H, W, D]          │
        └───────────────┬────────────────┘
                        ▼
        ┌────────────────────────────────┐
        │  LayerNorm → X_norm            │
        └───────┬───────────────┬────────┘
                ▼               ▼
        ┌──────────────┐ ┌──────────────────┐
        │ Detail       │ │ Context          │
        │ Z_det=       │ │ DWConv→DWConv→   │
        │  Linear(X)   │ │ BN→SiLU→Z_ctx    │
        └──────┬───────┘ └────────┬─────────┘
               │    (diff: Z_ctx -= Z_det)
               ├──────────┬───────┘
               ▼          ▼
        ┌────────────────────────────────┐
        │ Local Sparse Geometric Product │
        │ → G_feat                       │
        └───────────────┬────────────────┘
                        │  (+ optional global branch)
                        ▼
        ┌────────────────────────────────┐
        │ GGR(X_norm, G_feat) → H_mix    │
        └───────────────┬────────────────┘
                        ▼
        ┌────────────────────────────────┐
        │ return H_mix  (transform only; │
        │ residual added externally)     │
        │ [B, H, W, D]                   │
        └────────────────────────────────┘

    :param channels: Feature dimensionality D (constant throughout).
    :type channels: int
    :param shifts: Channel-shift offsets for the local interaction.
    :type shifts: Sequence[int]
    :param cli_mode: Algebraic components for the local interaction
        (``"inner"``, ``"wedge"``, ``"full"``). Defaults to ``"full"``.
    :type cli_mode: CliMode
    :param ctx_mode: Context mode (``"diff"`` or ``"abs"``). Defaults to
        ``"diff"``. Affects the inner term only; see the note above.
    :type ctx_mode: CtxMode
    :param use_global_context: Whether to add a global-average-pool branch.
        Requires ``channels >= 3``. Defaults to ``False``.
    :type use_global_context: bool
    :param causal: Sequence-safe mode for autoregressive use. IMPLIES
        ``input_mode="sequence"`` (passing ``input_mode="image"`` alongside it
        raises); uses ``(1, K)`` valid depthwise convolutions with left-only
        padding and a causal cumulative mean for the global context. Accepts
        ``(B, seq_len, D)`` or ``(B, 1, seq_len, D)``. Defaults to ``False``.
    :type causal: bool
    :param input_mode: Input contract. ``"image"`` accepts rank-4
        ``(B, H, W, D)`` only and uses ``(K, K)`` same-padded depthwise
        convolutions. ``"sequence"`` accepts BOTH rank-3 ``(B, L, D)`` and
        rank-4 ``(B, 1, L, D)`` (the sequence axis is axis 2; axis 1 must be a
        singleton) and uses ``(1, K)`` depthwise convolutions — same-padded and
        therefore bidirectional when ``causal=False``, valid-padded with an
        explicit left pad when ``causal=True``. ``None`` (the default) resolves
        to ``"sequence"`` when ``causal=True`` and to ``"image"`` otherwise; the
        RESOLVED value is what ``get_config`` serializes. This is a construction
        -time choice, not an inference from the input rank, because the two
        geometries register DIFFERENT depthwise kernel shapes — ``(1, K, D, 1)``
        vs ``(K, K, D, 1)`` — so an inferred mode would make the saved weight
        layout a function of the first tensor seen.
    :type input_mode: Optional[InputMode]
    :param layer_scale_init: Initial LayerScale value. Defaults to 1e-5.
    :type layer_scale_init: float
    :param use_bias: Whether the detail/projection Dense layers use a bias.
        Defaults to ``True``. Not forwarded to the GGR gate: the reference
        block leaves the gate at bias=True regardless, and forwarding would
        change behaviour for existing ``use_bias=False`` checkpoints.
    :type use_bias: bool
    :param activation: Context-stream activation applied after the context
        normalization. Defaults to ``"silu"``.
    :type activation: Any
    :param dot_activation: Inner-term activation inside the geometric products.
        Defaults to ``"silu"``.
    :type dot_activation: Any
    :param gate_activation: GGR gate activation. Defaults to ``"sigmoid"``.
    :type gate_activation: Any
    :param feature_activation: GGR identity-path activation. Defaults to
        ``"silu"``.
    :type feature_activation: Any
    :param use_gate: Forwarded to the internal
        :class:`GatedGeometricResidual`; ``False`` drops the multiplicative
        gate for degree-1-homogeneous (bias-free) consumers. Defaults to
        ``True``.
    :type use_gate: bool
    :param context_kernel_size: Depthwise kernel size K for the two context
        convolutions; effective receptive field is (2K-1). Defaults to 3.
    :type context_kernel_size: int
    :param kernel_initializer: Kernel initializer for Dense layers. See note 4
        in the module docstring regarding reproduction of the paper.
    :type kernel_initializer: Any
    :param bias_initializer: Bias initializer for Dense layers.
    :type bias_initializer: Any
    :param kernel_regularizer: Kernel regularizer for Dense layers.
    :type kernel_regularizer: Optional[Any]
    :param bias_regularizer: Bias regularizer for Dense layers.
    :type bias_regularizer: Optional[Any]
    :param normalization_type: Normalization applied to the context stream,
        resolved by ``create_normalization_layer``. ``None`` (the default)
        resolves per MODE: ``"batch_norm"`` in image mode (matching the
        reference ``BatchNorm2d``) and ``"zero_centered_rms_norm"`` in sequence
        mode, causal or not. The RESOLVED value is what ``get_config``
        serializes, so this default is checkpoint-safe. Passing one of
        ``_SEQUENCE_REDUCING_NORMS`` (``"batch_norm"``,
        ``"bias_free_batch_norm"``, ``"global_response_norm"``) explicitly
        together with ``causal=True`` raises ``ValueError``: those types reduce
        over the sequence axis, so the future leaks into the past. The check is
        not gated on ``training`` because ``"global_response_norm"`` reduces
        unconditionally. Non-causal sequence mode is deliberately NOT
        restricted — a bidirectional encoder is allowed to mix across
        positions.
    :type normalization_type: Optional[str]
    :param normalization_kwargs: Extra kwargs for the context normalization.
    :type normalization_kwargs: Optional[Dict[str, Any]]
    :param input_normalization_type: Normalization applied to the block input.
        ``None`` (default) uses ``LayerNormalization(epsilon=1e-6)``, matching
        the reference ``LayerNorm2d``.
    :type input_normalization_type: Optional[str]
    :param input_normalization_kwargs: Extra kwargs for the input
        normalization.
    :type input_normalization_kwargs: Optional[Dict[str, Any]]
    :param kwargs: Passed to ``keras.layers.Layer``.
    """

    def __init__(
        self,
        channels: int,
        shifts: Sequence[int],
        cli_mode: CliMode = "full",
        ctx_mode: CtxMode = "diff",
        use_global_context: bool = False,
        causal: bool = False,
        input_mode: Optional[InputMode] = None,
        layer_scale_init: float = 1e-5,
        use_bias: bool = True,
        activation: Any = "silu",
        dot_activation: Any = "silu",
        gate_activation: Any = "sigmoid",
        feature_activation: Any = "silu",
        use_gate: bool = True,
        context_kernel_size: int = 3,
        kernel_initializer: Any = "glorot_uniform",
        bias_initializer: Any = "zeros",
        kernel_regularizer: Optional[Any] = None,
        bias_regularizer: Optional[Any] = None,
        normalization_type: Optional[str] = None,
        normalization_kwargs: Optional[Dict[str, Any]] = None,
        input_normalization_type: Optional[str] = None,
        input_normalization_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if cli_mode not in _CLI_MODES:
            raise ValueError(
                f"cli_mode must be one of {_CLI_MODES}, got {cli_mode!r}"
            )
        if ctx_mode not in _CTX_MODES:
            raise ValueError(
                f"ctx_mode must be one of {_CTX_MODES}, got {ctx_mode!r}"
            )
        if input_mode is not None and input_mode not in _INPUT_MODES:
            raise ValueError(
                f"input_mode must be one of {_INPUT_MODES}, got {input_mode!r}"
            )
        # `causal` IMPLIES sequence mode. An unspecified input_mode (None)
        # resolves silently; an explicit "image" is an incoherent request (a
        # causal block convolves along axis 2 with left-only padding, which is
        # meaningless for a real H>1 image) and is rejected loudly rather than
        # overridden behind the caller's back.
        if causal:
            if input_mode == "image":
                raise ValueError(
                    "input_mode='image' is incompatible with causal=True: "
                    "causal blocks are sequence blocks (they left-pad and "
                    "convolve along axis 2 with a (1, K) kernel). Pass "
                    "input_mode='sequence' or omit it."
                )
            resolved_input_mode: str = "sequence"
        else:
            resolved_input_mode = input_mode if input_mode is not None else "image"
        # The context-norm DEFAULT is mode-derived: image mode keeps the
        # reference `BatchNorm2d` equivalent, sequence mode (causal or not) gets
        # a per-position norm. Only an EXPLICIT caller choice can be unsafe, so
        # the raise below is reachable only from a caller-supplied value and the
        # resolved default can never trip it.
        if normalization_type is None:
            resolved_normalization_type: str = (
                "zero_centered_rms_norm"
                if resolved_input_mode == "sequence"
                else "batch_norm"
            )
        else:
            # DECISION plan-2026-08-11T110821-54118fdd/D-002
            # A causal block reshapes its sequence to (B, 1, L, D), so axis 2 IS
            # the sequence axis -- and `batch_norm`/`bias_free_batch_norm`
            # reduce over (B, H, W) at training=True while
            # `global_response_norm` reduces over (H, W) UNCONDITIONALLY (it
            # leaks at inference too, which is why this raise is NOT gated on
            # `training`). Measured on a 4x1x12x8 probe with a fresh NON-DC
            # perturbation of positions 8..11: max |delta| over positions 0..7
            # was 1.067 on a signal of scale 4.95, i.e. ~22% of the signal
            # flowing backwards in time. Do NOT "fix" this by probing with a DC
            # shift -- that probe is VACUOUS (measured leak 1.9e-06) because the
            # input LayerNorm removes a per-position DC offset before the
            # context stream ever sees it. Do NOT extend this raise to
            # non-causal sequence mode (a bidirectional encoder is ALLOWED to
            # mix across positions) and do NOT change image mode's `batch_norm`
            # default: it is load-bearing for the strict xfail at
            # test_video_jepa.py::test_predictor_graph_mode_dropout_zero and for
            # the batch_size >= 2 guards in the video_jepa data path. See
            # decisions.md D-002.
            if causal and normalization_type in _SEQUENCE_REDUCING_NORMS:
                raise ValueError(
                    f"normalization_type={normalization_type!r} is incompatible "
                    f"with causal=True: it reduces over the sequence axis, so "
                    f"every position's context normalization sees the whole "
                    f"sequence and the future leaks into the past (measured max "
                    f"|delta| 1.067 on a 4.95-scale signal). "
                    f"'batch_norm' and 'bias_free_batch_norm' leak at "
                    f"training=True; 'global_response_norm' reduces "
                    f"unconditionally and leaks at inference too. Sequence-axis"
                    f"-reducing types are "
                    f"{sorted(_SEQUENCE_REDUCING_NORMS)}; pass a per-position "
                    f"type such as 'zero_centered_rms_norm' (the causal "
                    f"default) or omit normalization_type."
                )
            resolved_normalization_type = normalization_type
        if (
            not isinstance(context_kernel_size, numbers.Integral)
            or isinstance(context_kernel_size, bool)
            or context_kernel_size <= 0
        ):
            raise ValueError(
                f"context_kernel_size must be a positive int, "
                f"got {context_kernel_size!r}"
            )
        # The global branch hardcodes shifts=[1, 2]; with fewer than 3 channels
        # the inner SparseRollingGeometricProduct would silently drop shift=2
        # (channels=2) or reject the layer entirely (channels=1). Fail up front
        # rather than building a quietly different global branch.
        if use_global_context and channels < _MIN_GLOBAL_CHANNELS:
            raise ValueError(
                f"use_global_context=True requires channels >= "
                f"{_MIN_GLOBAL_CHANNELS} (global branch uses "
                f"shifts={_GLOBAL_SHIFTS}); got channels={channels}"
            )
        if cli_mode == "wedge" and ctx_mode == "diff":
            logger.warning(
                "CliffordNetBlock: ctx_mode='diff' has NO effect when "
                "cli_mode='wedge' (the self-terms cancel by antisymmetry, so "
                "W(det, ctx - det) == W(det, ctx)). This model is identical "
                "to ctx_mode='abs'."
            )

        self.channels = channels
        # Pre-filter list, kept verbatim so get_config round-trips the caller's
        # intent; SparseRollingGeometricProduct re-applies the s < channels
        # filter on reconstruction.
        self.shifts = _validate_shifts(shifts)
        self.cli_mode = cli_mode
        self.ctx_mode = ctx_mode
        self.use_global_context = use_global_context
        self.causal = causal
        self.input_mode = resolved_input_mode
        self.layer_scale_init = layer_scale_init
        self.use_bias = use_bias
        self.activation = _activation_spec(activation)
        self.dot_activation = _activation_spec(dot_activation)
        self.gate_activation = _activation_spec(gate_activation)
        self.feature_activation = _activation_spec(feature_activation)
        self._activation_fn = _resolve_activation(self.activation)
        self.use_gate = use_gate
        self.context_kernel_size = int(context_kernel_size)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        # The RESOLVED string, never the None sentinel: get_config() serializes
        # this attribute, so a future change to the mode-derived default cannot
        # silently re-type an existing checkpoint's context norm.
        self.normalization_type = resolved_normalization_type
        self.normalization_kwargs = dict(normalization_kwargs or {})
        self.input_normalization_type = input_normalization_type
        self.input_normalization_kwargs = dict(input_normalization_kwargs or {})

        # Image mode is rank-4-only. Sequence mode accepts BOTH rank-3
        # ``(B, L, D)`` and rank-4 ``(B, 1, L, D)``; ``InputSpec`` cannot
        # express "ndim in {3, 4} AND axis 1 == 1 when ndim == 4" (measured on
        # Keras 3.8: combining ``axes={1: 1}`` with ``min_ndim=3`` rejects every
        # rank-3 input), so the singleton-H constraint is enforced in build().
        if self.input_mode == "sequence":
            self.input_spec = keras.layers.InputSpec(min_ndim=3, max_ndim=4)
        else:
            self.input_spec = keras.layers.InputSpec(ndim=4)

        # Rank of the tensor the layer was built on. Rank 3 means call() must
        # expand to the internal 4-D representation and squeeze on the way out.
        self._input_rank: Optional[int] = None

        _dense_kwargs: Dict[str, Any] = dict(
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )

        # --- Step 1: Input norm ---
        # input_normalization_type=None reproduces the reference LayerNorm2d
        # (channel-wise LayerNormalization, epsilon=1e-6).
        if self.input_normalization_type is None:
            self.input_norm = keras.layers.LayerNormalization(
                epsilon=1e-6, name="input_norm"
            )
        else:
            self.input_norm = create_normalization_layer(
                self.input_normalization_type,
                name="input_norm",
                **self.input_normalization_kwargs,
            )

        # --- Step 2a: Detail stream (1x1 pointwise) ---
        self.linear_det = keras.layers.Dense(
            channels, name="linear_det", **_dense_kwargs
        )

        # --- Step 2b: Context stream ---
        # Two stacked KxK depthwise convolutions (K=context_kernel_size,
        # default 3 -> effective 5x5 RF), a single normalization layer after
        # both, then `activation` in call(). No activation between the two
        # convolutions, matching the reference `get_context_local`.
        #
        # The `causal` flag gates the four (and only four) behavioural
        # differences between the vision and sequence variants: (a) the
        # depthwise geometry here ((1,K)/valid + explicit left-pad vs
        # (K,K)/same); (b) the build() shapes (left-padded); (c) the call()
        # padding before each DWConv; (d) the global-context statistic (causal
        # cumulative mean vs full-image GAP). CausalCliffordNetBlock is a thin
        # subclass; do not duplicate the block body into it, and do not merge
        # these two convolution paths.
        # DECISION plan-2026-08-11T110821-54118fdd/D-001
        # The internal representation is ALWAYS 4-D ``(B, H, W, D)``, even when
        # the public contract is a rank-3 ``(B, L, D)`` sequence. Do NOT replace
        # these DepthwiseConv2D layers with a native ``DepthwiseConv1D``: that
        # would change the registered kernels from ``(1, K, D, 1)`` to
        # ``(K, D, 1)`` and break by-name ``.keras`` loading for EVERY existing
        # causal checkpoint (cliffordnet/lm.py, the CLIP text tower, the
        # video_jepa causal blocks). Rank-3 support is therefore a call()-level
        # expand/squeeze around an unchanged 4-D body.
        # Image mode deliberately keeps the ``(K, K)`` kernel: on an H=1 input a
        # ``(K, K)`` same-padded convolution is provably bit-identical to the
        # ``(1, K)`` middle-row one (measured: max abs diff 0.0), but 2/3 of its
        # taps then multiply against zero padding and are dead weight -- so
        # sequence mode uses ``(1, K)`` while image mode must keep ``(K, K)`` to
        # stay weight-compatible with existing vision checkpoints. This is also
        # why input_mode is an explicit constructor choice and is never inferred
        # from the input rank: inferring it would make the saved kernel SHAPE a
        # function of the first tensor the layer ever saw. See decisions.md
        # D-001.
        if self.input_mode == "sequence":
            _dw_kernel: Any = (1, self.context_kernel_size)
            # Causal: "valid" + the explicit left-pad in call(). Non-causal
            # sequence: "same", i.e. a bidirectional 1-D context window.
            _dw_padding = "valid" if self.causal else "same"
        else:
            _dw_kernel = self.context_kernel_size
            _dw_padding = "same"
        self.dw_conv = keras.layers.DepthwiseConv2D(
            kernel_size=_dw_kernel,
            padding=_dw_padding,
            use_bias=False,
            name="dw_conv",
        )
        self.dw_conv2 = keras.layers.DepthwiseConv2D(
            kernel_size=_dw_kernel,
            padding=_dw_padding,
            use_bias=False,
            name="dw_conv2",
        )
        # Name kept as "ctx_bn" for checkpoint compatibility even though the
        # layer type is configurable.
        self.ctx_norm = create_normalization_layer(
            self.normalization_type,
            name="ctx_bn",
            **self.normalization_kwargs,
        )

        # --- Step 3: Local sparse rolling product ---
        self.local_geo_prod = SparseRollingGeometricProduct(
            channels=channels,
            shifts=self.shifts,
            cli_mode=cli_mode,
            use_bias=use_bias,
            dot_activation=dot_activation,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name="local_geo_prod",
        )

        # --- Optional global context branch (gFFN-G) ---
        # Always hardcoded to shifts=[1,2] and cli_mode='full', matching the
        # reference CliffordAlgebraBlock.
        if use_global_context:
            self.global_geo_prod = SparseRollingGeometricProduct(
                channels=channels,
                shifts=_GLOBAL_SHIFTS,
                cli_mode=_GLOBAL_CLI_MODE,
                use_bias=use_bias,
                dot_activation=dot_activation,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name="global_geo_prod",
            )
        else:
            self.global_geo_prod = None

        # --- Step 4 / 5: GGR ---
        # use_bias is intentionally NOT forwarded (see the class docstring).
        self.ggr = GatedGeometricResidual(
            channels=channels,
            layer_scale_init=layer_scale_init,
            gate_activation=gate_activation,
            feature_activation=feature_activation,
            use_gate=use_gate,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name="ggr",
        )

    # ------------------------------------------------------------------

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build all sub-layers in dependency order.

        :param input_shape: ``(B, H, W, D)`` in image mode; ``(B, L, D)`` or
            ``(B, 1, L, D)`` in sequence mode.
        :raises ValueError: If the rank is unsupported for the configured
            ``input_mode``, if a rank-4 sequence input has ``H != 1``, or if the
            channel axis does not equal ``channels``.
        """
        # This block is isotropic in channels. A mismatched D would otherwise
        # produce a cryptic broadcast error at the external residual add.
        if input_shape[-1] is not None and input_shape[-1] != self.channels:
            raise ValueError(
                f"{type(self).__name__} is isotropic: expected last dim == "
                f"channels={self.channels}, got input_shape[-1]={input_shape[-1]}. "
                f"Project the input before the block (e.g. with a 1x1 Conv) "
                f"or rebuild the block with channels={input_shape[-1]}."
            )

        rank = len(input_shape)
        self._input_rank = rank
        if self.input_mode == "sequence":
            if rank not in (3, 4):
                raise ValueError(
                    f"{type(self).__name__}(input_mode='sequence') expects a "
                    f"rank-3 (B, L, D) or rank-4 (B, 1, L, D) input, got "
                    f"rank-{rank} shape {tuple(input_shape)}."
                )
            if rank == 4 and input_shape[1] is not None and input_shape[1] != 1:
                raise ValueError(
                    f"{type(self).__name__}(input_mode='sequence') expects a "
                    f"singleton axis 1 for rank-4 input (B, 1, L, D): the "
                    f"sequence axis is axis 2. Got input_shape[1]="
                    f"{input_shape[1]} in {tuple(input_shape)}."
                )
            # Internal representation is always 4-D (D-001).
            if rank == 3:
                spatial_shape = (
                    input_shape[0],
                    1,
                    input_shape[1],
                    input_shape[2],
                )
            else:
                spatial_shape = tuple(input_shape)
        else:
            if rank != 4:
                raise ValueError(
                    f"{type(self).__name__}(input_mode='image') expects a "
                    f"rank-4 (B, H, W, D) input, got rank-{rank} shape "
                    f"{tuple(input_shape)}."
                )
            spatial_shape = tuple(input_shape)

        # Step 1: norm
        self.input_norm.build(spatial_shape)

        # Step 2a: detail linear
        self.linear_det.build(spatial_shape)
        stream_shape = self.linear_det.compute_output_shape(spatial_shape)

        # Step 2b: context -- two DWConvs, then a single normalization.
        if self.causal:
            pad = self.context_kernel_size - 1
            # The valid convolutions are built on LEFT-PADDED shapes so that,
            # after the explicit left-pad in call(), they preserve the
            # sequence length. Both therefore output `spatial_shape`.
            self.dw_conv.build(_left_padded_shape(spatial_shape, pad))
            self.dw_conv2.build(_left_padded_shape(spatial_shape, pad))
            ctx_out = spatial_shape
        else:
            self.dw_conv.build(spatial_shape)
            dw1_out = self.dw_conv.compute_output_shape(spatial_shape)
            self.dw_conv2.build(dw1_out)
            ctx_out = self.dw_conv2.compute_output_shape(dw1_out)
        self.ctx_norm.build(ctx_out)

        # Step 3: local product
        self.local_geo_prod.build(stream_shape)

        # Optional global branch
        if self.global_geo_prod is not None:
            self.global_geo_prod.build(stream_shape)

        # GGR
        self.ggr.build(stream_shape)

        super().build(input_shape)

    # ------------------------------------------------------------------

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Forward pass.

        :param inputs: Feature tensor ``(B, H, W, D)``; in sequence mode also
            ``(B, L, D)`` or ``(B, 1, L, D)``.
        :param training: Whether in training mode.
        :return: Residual term with the SAME rank and shape as ``inputs``; the
            caller adds it to ``inputs``.
        """
        # Rank-3 sequence input is lifted to the internal 4-D representation
        # (D-001) and squeezed back at the end; the body below always sees
        # (B, H, W, D). The branch reads the rank of THIS call's input rather
        # than the rank seen at build time, so a layer built on (B, L, D) still
        # behaves correctly if the equivalent (B, 1, L, D) is fed later (the
        # sub-layers are built on the internal 4-D shape either way).
        call_rank = len(inputs.shape)
        if call_rank == 3:
            x_prev = keras.ops.expand_dims(inputs, axis=1)
        else:
            x_prev = inputs

        # --- Step 1: Normalise ---
        # `training` is passed explicitly: a configurable input normalization
        # may be a batch-norm variant. Keras drops the kwarg for layers whose
        # call() does not accept it.
        x_norm = self.input_norm(x_prev, training=training)

        # --- Step 2: Dual-stream generation ---
        z_det = self.linear_det(x_norm)

        # Two stacked depthwise convolutions -> norm -> activation. On the
        # causal path each valid convolution is preceded by a left-only pad so
        # position i sees only positions <= i; norm and activation are shared
        # with the vision path.
        if self.causal:
            z_ctx = self._causal_pad(x_norm, self.context_kernel_size)
            z_ctx = self.dw_conv(z_ctx)
            z_ctx = self._causal_pad(z_ctx, self.context_kernel_size)
            z_ctx = self.dw_conv2(z_ctx)
        else:
            z_ctx = self.dw_conv(x_norm)
            z_ctx = self.dw_conv2(z_ctx)
        z_ctx = self._activation_fn(self.ctx_norm(z_ctx, training=training))

        if self.ctx_mode == "diff":
            # Differential ("Laplacian-like") context. Note this is C_loc minus
            # the *projected* detail stream, not the raw state, so it is only a
            # discrete Laplacian to the extent that linear_det stays near
            # identity. Affects the inner term only; the wedge term is
            # invariant to this subtraction (module docstring note 2).
            z_ctx = z_ctx - z_det

        # --- Step 3: Local sparse geometric interaction ---
        g_feat = self.local_geo_prod(z_det, z_ctx)

        # --- Step 4: Optional global context branch ---
        # The global branch always uses differential context and is
        # independent of the local ctx_mode setting.
        if self.global_geo_prod is not None:
            # GAP keeps spatial dims as 1; let the subtraction broadcast to
            # (B,H,W,D) rather than materialising a redundant intermediate.
            # The causal variant substitutes a cumulative mean so position i
            # only ever summarises positions 0..i.
            if self.causal:
                c_glo = self._causal_cumulative_mean(x_norm)
            else:
                c_glo = keras.ops.mean(x_norm, axis=[1, 2], keepdims=True)
            c_glo = c_glo - z_det
            g_glo = self.global_geo_prod(z_det, c_glo)
            # Implicit superposition weight beta = 1 (module docstring note 3).
            g_feat = g_feat + g_glo

        # --- Step 5: GGR (transform only; residual add is external) ---
        h_mix = self.ggr(x_norm, g_feat, training=training)
        if call_rank == 3:
            h_mix = keras.ops.squeeze(h_mix, axis=1)
        return h_mix

    # ------------------------------------------------------------------

    @staticmethod
    def _causal_pad(
        x: keras.KerasTensor, kernel_size: int
    ) -> keras.KerasTensor:
        """Apply left-only (causal) zero-padding along the W axis.

        For ``(B, 1, W, D)``, pads ``kernel_size - 1`` zeros on the left of the
        W dimension so that a ``"valid"`` convolution preserves the sequence
        length and each position only sees past/current positions. Used only on
        the ``causal=True`` path.

        :param x: Input tensor ``(B, 1, W, D)``.
        :param kernel_size: Depthwise kernel extent along W.
        :return: Left-padded tensor ``(B, 1, W + kernel_size - 1, D)``.
        """
        pad_w = kernel_size - 1
        # pad format: [[B_lo, B_hi], [H_lo, H_hi], [W_lo, W_hi], [D_lo, D_hi]]
        return keras.ops.pad(x, [[0, 0], [0, 0], [pad_w, 0], [0, 0]])

    # ------------------------------------------------------------------

    @staticmethod
    def _causal_cumulative_mean(
        x: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """Causal global context: cumulative mean along the W (sequence) axis.

        For input ``(B, 1, seq_len, D)``, position *i* receives the mean of
        positions ``0..i``.  This preserves autoregressive causality while
        still providing each position with a growing global summary. Used only
        on the ``causal=True`` path.

        The accumulation and the division are performed in ``float32`` when the
        input dtype is narrower than ``float32`` (see the DECISION anchor
        below); ``float32`` and ``float64`` inputs are computed in their own
        dtype and are bit-unchanged.
        """
        # DECISION plan-2026-08-11T110821-54118fdd/D-005
        # Do NOT write the obvious `keras.ops.cast(keras.ops.arange(...),
        # x.dtype)` here. Under `mixed_float16` the compute dtype is float16,
        # and float16 cannot represent the integers 1..seq_len exactly past
        # 2048: the divisor vector alone was MEASURED at 2.44e-04 relative
        # error at the endpoint 4097 (4.88e-04 max over 1..4096, where the
        # float16 spacing is 2) and at exactly 0.0 for seq_len == 2048. The
        # float16 `cumsum` is a LARGER error still, because a running sum of
        # thousands of terms is accumulated at 11-bit precision -- measured
        # end-to-end against a float64 reference on the float16-rounded input,
        # max relative error fell from 17.65 -> 7.7e-03 (L=2048), 17.65 ->
        # 3.9e-02 (L=4096) and 458.8 -> 1.4e-01 (L=8192) when this widening
        # was introduced, i.e. 449x to 3255x. Both the accumulation and the
        # divisors are therefore widened to float32, and the result is cast
        # back so the layer's public dtype contract is unchanged (float32
        # output verified BIT-IDENTICAL, `np.array_equal` True).
        # Widen only NARROWER-than-float32 dtypes: a float64 input must stay
        # float64, so it is computed in float64, never downcast to float32.
        input_dtype = keras.backend.standardize_dtype(x.dtype)
        compute_dtype = (
            "float32" if input_dtype in ("float16", "bfloat16") else input_dtype
        )

        # x shape: (B, 1, seq_len, D)
        x_c = keras.ops.cast(x, compute_dtype)
        cumsum = keras.ops.cumsum(x_c, axis=2)  # (B, 1, seq_len, D)
        # `keras.ops.shape(x)[2]` keeps this graph-safe under a dynamic
        # sequence length (measured working at L=13 and L=29 through a
        # functional model with `Input(shape=(1, None, D))`).
        seq_len = keras.ops.shape(x)[2]
        # divisors: [1, 2, 3, ..., seq_len] reshaped to (1, 1, seq_len, 1)
        divisors = keras.ops.cast(
            keras.ops.arange(1, seq_len + 1), compute_dtype
        )
        divisors = keras.ops.reshape(divisors, (1, 1, -1, 1))
        return keras.ops.cast(cumsum / divisors, input_dtype)

    # ------------------------------------------------------------------

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape.

        Rank-preserving by construction: a rank-3 ``(B, L, D)`` sequence input
        yields a rank-3 output and a rank-4 input yields a rank-4 output, since
        the internal H=1 axis is added and removed inside ``call()``.

        :param input_shape: Input shape ``(B, H, W, D)`` or ``(B, L, D)``.
        :return: Same as input shape.
        """
        return input_shape

    # ------------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        """Return serialisable configuration.

        :return: Dictionary with all constructor arguments.
        """
        config = super().get_config()
        config.update(
            {
                "channels": self.channels,
                "shifts": list(self.shifts),
                "cli_mode": self.cli_mode,
                "ctx_mode": self.ctx_mode,
                "use_global_context": self.use_global_context,
                "causal": self.causal,
                # The RESOLVED mode, never the None sentinel, so a reloaded
                # config reproduces the DWConv geometry exactly.
                "input_mode": self.input_mode,
                "layer_scale_init": self.layer_scale_init,
                "use_bias": self.use_bias,
                "activation": _serialize_activation(self.activation),
                "dot_activation": _serialize_activation(self.dot_activation),
                "gate_activation": _serialize_activation(self.gate_activation),
                "use_gate": self.use_gate,
                "feature_activation": _serialize_activation(
                    self.feature_activation
                ),
                "context_kernel_size": self.context_kernel_size,
                "kernel_initializer": initializers.serialize(self.kernel_initializer),
                "bias_initializer": initializers.serialize(self.bias_initializer),
                "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
                "bias_regularizer": regularizers.serialize(self.bias_regularizer),
                "normalization_type": self.normalization_type,
                "normalization_kwargs": dict(self.normalization_kwargs),
                "input_normalization_type": self.input_normalization_type,
                "input_normalization_kwargs": dict(
                    self.input_normalization_kwargs
                ),
            }
        )
        return config


# ---------------------------------------------------------------------------
# CausalCliffordNetBlock -- sequence-safe variant for autoregressive LMs
# ---------------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class CausalCliffordNetBlock(CliffordNetBlock):
    """Causal (autoregressive) variant of :class:`CliffordNetBlock`.

    Equivalent to ``CliffordNetBlock(causal=True)`` with a
    ``"zero_centered_rms_norm"`` context-norm default: the two context
    ``DepthwiseConv2D`` layers use ``kernel=(1, K)`` / ``padding="valid"``
    with explicit left-only zero-padding, and the optional global-context
    branch uses a causal cumulative mean, so position *i* only sees positions
    ``<= i``. Expects 4-D input ``(B, 1, seq_len, D)`` (sequence reshaped for
    2-D convolutions with ``H = 1``).

    Kept as a registered subclass (rather than folded away) purely for
    checkpoint / back-compat: the registered class name and the legacy Keras
    auto-name ``causal_clifford_net_block`` are preserved so existing weights
    load by name. All behaviour lives in :class:`CliffordNetBlock` gated by
    ``causal=True``.

    :param kwargs: Forwarded to :class:`CliffordNetBlock`; ``causal`` is forced
        to ``True`` and ``normalization_type`` defaults to
        ``"zero_centered_rms_norm"``.
    """

    def __init__(self, **kwargs: Any) -> None:
        # Preserve the causal-specific context-norm default while letting the
        # caller override it explicitly. Redundant since the base class resolves
        # the same value for sequence mode, but kept so this subclass keeps
        # OWNING its documented default rather than inheriting it silently: if
        # the base default ever moves again, this class does not move with it.
        kwargs.setdefault("normalization_type", "zero_centered_rms_norm")
        # Force causal=True even if a from_config dict carries a `causal` key,
        # so this subclass can never be built non-causal (and never raises a
        # duplicate-kwarg TypeError).
        kwargs["causal"] = True
        super().__init__(**kwargs)


# DECISION plan-2026-08-10T130454-3649c19e/D-006
# `CliffordNetBlockDSv2` and `CausalCliffordNetBlockDSv2` used to live below this
# line (~700 lines, plus `_make_pool_v2`, `_make_ctx_norm`, `_make_causal_pool`
# and the `SkipPoolV2` / `CtxModeV2` / `CtxNormType` aliases). They were DELETED
# together with their whole consumer closure, not merely orphaned.
#
# Do NOT restore them, and do NOT add a `strides`/`out_channels` downsampling
# path onto `CliffordNetBlock` as a stand-in. The owner declared the DSv2
# downsampling experiment dead; the two model modules that consumed it
# (`models/cliffordnet/embedding_unet.py`, `models/cliffordnet/lmunet.py`), four
# trainers, and three test suites were removed in the same commit. Re-adding the
# classes here would re-create exactly the half-deleted state this cleanup
# exists to repair. See decisions.md D-005/D-006.
