"""
CliffordRNN — recurrent geometric-algebra layer, drop-in for ``keras.layers.RNN``.

Adapts the CliffordNet vision block (Zhongping Ji, "CliffordNet: All You Need is
Geometric Algebra", arXiv:2601.06793v2) into a *recurrent* primitive by replacing
the block's spatial locality with temporal locality.  Provides:

- :class:`CliffordRNNCell` -- a cell usable directly in ``keras.layers.RNN``,
  ``keras.layers.Bidirectional`` and ``keras.layers.RNN([cell_a, cell_b])``.
- :class:`CliffordRNN`     -- an ``RNN`` subclass with the same public surface as
  ``keras.layers.GRU`` / ``LSTM`` (``units`` + ``return_sequences``,
  ``return_state``, ``go_backwards``, ``stateful``, ``unroll``,
  ``zero_output_for_mask``, ``dropout``, ``recurrent_dropout``, masking).

================================================================================
Theory — from spatial locality to temporal locality
================================================================================

The vision block is isotropic in channels and derives two streams from one
normalized tensor: a *detail* stream (1x1 pointwise, no mixing) and a *context*
stream (stacked depthwise convs, i.e. aggregation over spatial neighbours).  A
recurrent cell has no spatial axis, but it has something better: the hidden
state is already an aggregate of the entire past.  The mapping is therefore
exact rather than analogical::

    vision block                     CliffordRNNCell
    ------------------------------   -------------------------------------------
    X_prev (feature map)             h_{t-1}                (state = residual path)
    X_norm = Norm(X_prev)            h_norm = Norm(h_{t-1})
    Z_det  = Linear(X_norm)          Z_det  = Linear(x_t)   PRESENT observation
    Z_ctx  = act(N(DW(DW(X_norm))))  Z_ctx  = act(N(W_h h_norm))   PAST aggregate
    ctx_mode="diff": Z_ctx -= Z_det  identical -> TEMPORAL Laplacian / surprise
    GAP(X_norm) global branch        causal cumulative mean (or EMA) of Z_det
    depth as ODE time (Euler step)   sequence position as ODE time (Euler step)

Because ``Z_det`` is a function of ``x_t`` only and ``Z_ctx`` of ``h_{t-1}``
only, the two grades of the geometric product acquire clean temporal semantics:

* ``dot   = act(Z_det ⊙ roll(Z_ctx, s))`` -- *coherence*: how strongly the new
  observation agrees with the accumulated past.  With ``ctx_mode="diff"`` this is
  a gated prediction-error magnitude.
* ``wedge = Z_det ⊙ roll(Z_ctx,s) − Z_ctx ⊙ roll(Z_det,s)`` -- the *bivector*:
  oriented area swept between past and present, i.e. a **change-point /
  temporal-torque detector**.  It is exactly the antisymmetric part that
  dot-product attention and every classical gate (LSTM/GRU) discard.  Note
  ``u ∧ u = 0``: at ``t = 1`` with a zero state and ``ctx_mode="diff"`` the two
  streams are collinear and the wedge vanishes identically -- correct algebra,
  and the reason the ``dot`` grade must stay enabled (``cli_mode="full"``).

Cost is ``O(D · |shifts|)`` per step, linear in width, with no FFN anywhere.

State update (stability)
------------------------
The block returns only the LayerScale-gated term and lets the model add the
residual.  A recurrent cell cannot outsource that, and a pure residual carry
``h_t = h_{t-1} + term`` drifts without bound over long sequences, so the cell
owns the update and offers three modes:

    "gated"    h_t = σ(W_f [x_t, h_norm] + b_f) ⊙ h_{t-1} + term   (default)
    "decay"    h_t = σ(λ) ⊙ h_{t-1} + term        (input-independent, SSM-like)
    "residual" h_t = h_{t-1} + term               (faithful to the block; pair
                                                   with a downstream norm)

``σ`` is hardcoded sigmoid for the carry gate (it must be a bounded contraction);
``gate_activation`` still configures the GGR's ``alpha``.  With ``σ ≈ 0.73`` at
``b_f = 1`` and an ``O(1)`` update, the steady-state magnitude is bounded by
``≈ γ / (1 − σ)``.

Deliberate deviations from the vision block
-------------------------------------------
1. ``layer_scale_init`` defaults to **1.0**, not ``1e-5``.  In a deep stack,
   γ≈0 means "start at identity"; in a *recurrence*, γ≈0 means the state never
   moves and no gradient reaches the geometric machinery.
2. ``normalization_type`` / ``state_normalization_type`` default to ``None`` ->
   ``LayerNormalization(epsilon=1e-6)``.  BatchNorm is unsound inside a
   recurrence (statistics shared across timesteps); any string is still routed
   through ``create_normalization_layer`` (e.g. ``"zero_centered_rms_norm"``,
   ``"bias_free_batch_norm"``).
3. The context stream is a single dense recurrent projection (``use_bias=False``,
   orthogonal init) instead of two depthwise convs: the state already carries the
   receptive field, so extra intra-step mixing would only duplicate ``Z_det``.
4. ``use_bias`` IS forwarded to the GGR here (the vision block pins GGR to
   ``use_bias=True`` for byte-identity with its own history).  This is new code,
   so a fully bias-free / degree-1-homogeneous configuration is reachable:
   ``use_bias=False, use_gate=False, state_update="decay",
   activation="leaky_relu", dot_activation="leaky_relu",
   feature_activation="leaky_relu", normalization_type="zero_centered_rms_norm",
   state_normalization_type="zero_centered_rms_norm"``.
5. ``include_vector_grade=True`` adds ``Z_det`` to ``G_feat`` -- the grade-1
   (vector) part of the multivector, alongside grade-0 (dot) and grade-2 (wedge).
   ``G_feat`` is otherwise purely multiplicative in the input; enable this first
   if training stalls on tasks needing a direct additive input path.

Global context branch
---------------------
``use_global_context=True`` reproduces the causal global branch: a running
summary of ``Z_det`` over the prefix, differenced against the present
(``C_glo = c_t − Z_det``) and pushed through a second geometric product with
hardcoded ``shifts=[1, 2]`` and ``cli_mode="full"``.  ``"cumulative_mean"`` is
the exact prefix mean (adds two states: the mean and a step counter);
``"ema"`` uses a learnable per-channel decay (adds one state).

Recurrent dropout applies to the state used to *compute* the streams, never to
the carry path -- dropping the carry would destroy the state itself.

Usage
-----
.. code-block:: python

    import keras
    from clifford_rnn import CliffordRNN, CliffordRNNCell

    # Drop-in for GRU/LSTM/RNN
    x = keras.Input((None, 32))
    y = CliffordRNN(64, return_sequences=True, dropout=0.1)(x)

    # Or as a cell, with every keras.layers.RNN feature
    y = keras.layers.RNN(CliffordRNNCell(64), return_sequences=True)(x)
    y = keras.layers.Bidirectional(CliffordRNN(64, return_sequences=True))(x)
    y = keras.layers.RNN([CliffordRNNCell(64), CliffordRNNCell(64)])(x)
"""

from __future__ import annotations

import keras
from keras import initializers, regularizers
from typing import Any, Dict, List, Optional, Sequence, Tuple

# DropoutRNNCell is a private-but-stable Keras mixin. It is inherited purely so
# that ``keras.layers.RNN._maybe_config_dropout_masks`` recognises this cell and
# pre-populates the masks OUTSIDE the scan (required for a stateless JAX loop).
# All mask methods are also implemented locally, so behaviour is identical if the
# import ever fails.
from keras.src.layers.rnn.dropout_rnn_cell import (  # type: ignore
    DropoutRNNCell as _DropoutRNNCellMixin,
)

# ---------------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------------

from .clifford_block import (
    SparseRollingGeometricProduct,
    GatedGeometricResidual,
    # DECISION plan-2026-08-10T130454-3649c19e/D-010
    # The activation helper trio is IMPORTED from the sibling, not re-copied.
    # Do NOT reinstate a local copy: this file previously carried a stale fork
    # of ``_resolve_activation`` that pre-dated the sibling's hardening, so a
    # keras.layers.Layer passed as an activation was silently accepted here and
    # rejected there, and ``get_config`` emitted raw callables that are not
    # JSON-serialisable. Do NOT extract these into a new shared module either:
    # two call sites is not an earned abstraction and the sibling owns them.
    # See decisions.md D-010.
    _activation_spec,
    _resolve_activation,
    _serialize_activation,
)
from ..norms.factory import create_normalization_layer  # type: ignore
from ...utils.logger import logger  # type: ignore

__all__ = ["CliffordRNNCell", "CliffordRNN"]

# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

# Global-branch constants, matching the vision block exactly.
_GLOBAL_SHIFTS: List[int] = [1, 2]
_GLOBAL_CLI_MODE: str = "full"

_CLI_MODES = ("inner", "wedge", "full")
_CTX_MODES = ("diff", "abs")
_STATE_UPDATES = ("gated", "decay", "residual")
_GLOBAL_MODES = ("cumulative_mean", "ema")


# ---------------------------------------------------------------------------
# CliffordRNNCell
# ---------------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class CliffordRNNCell(_DropoutRNNCellMixin, keras.layers.Layer):
    """Recurrent cell driven by a sparse Clifford geometric product.

    One timestep:

    .. code-block:: text

        x_t [B,F]                         h_{t-1} [B,D]
           │                                 │  (recurrent dropout -> h_used)
           │                                 ▼
           │                        ┌──────────────────┐
           │                        │ state_norm       │→ h_norm
           │                        └────────┬─────────┘
           ▼                                 ▼
        ┌──────────────┐          ┌────────────────────────┐
        │ Z_det =      │          │ Z_ctx = act(Norm(W_h · │
        │  Dense(x_t)  │          │        h_norm))        │
        │  (PRESENT)   │          │        (PAST)          │
        └──────┬───────┘          └───────────┬────────────┘
               │      (ctx_mode="diff": Z_ctx -= Z_det)
               ├──────────────┬───────────────┘
               ▼              ▼
        ┌────────────────────────────────────────┐
        │ SparseRollingGeometricProduct          │
        │   dot   -> coherence      (grade 0)    │
        │   wedge -> change / torque (grade 2)   │
        └───────────────┬────────────────────────┘
                        │ + optional global branch (prefix mean of Z_det)
                        │ + optional Z_det          (grade 1)
                        ▼  G_feat
        ┌────────────────────────────────────────┐
        │ GGR: term = γ ⊙ (act(h_norm) + α⊙G)    │
        └───────────────┬────────────────────────┘
                        ▼
        h_t = carry(h_{t-1}) + term        →  output = h_t

    :param units: State / output dimensionality D.
    :type units: int
    :param shifts: Cyclic channel offsets for the local product; exponentially
        spaced values give logarithmic channel-mixing range. Offsets
        ``>= units`` are dropped by the inner product layer.
    :type shifts: Sequence[int]
    :param cli_mode: Algebraic grades retained (``"inner"``, ``"wedge"``,
        ``"full"``). Defaults to ``"full"``.
    :type cli_mode: str
    :param ctx_mode: ``"diff"`` subtracts ``Z_det`` from ``Z_ctx`` (temporal
        Laplacian / prediction error); ``"abs"`` keeps the raw aggregate.
    :type ctx_mode: str
    :param use_global_context: Add the prefix-summary geometric branch.
    :type use_global_context: bool
    :param global_context_mode: ``"cumulative_mean"`` (exact prefix mean, +2
        states) or ``"ema"`` (learnable per-channel decay, +1 state).
    :type global_context_mode: str
    :param state_update: Carry rule: ``"gated"``, ``"decay"`` or ``"residual"``.
    :type state_update: str
    :param include_vector_grade: Add ``Z_det`` (grade 1) to ``G_feat``.
    :type include_vector_grade: bool
    :param layer_scale_init: Initial GGR LayerScale γ. Defaults to ``1.0``
        (NOT the block's ``1e-5``; see module docstring).
    :type layer_scale_init: float
    :param forget_bias_init: Bias/logit init for the carry gate; larger values
        push the initial carry towards identity (long-range retention).
    :type forget_bias_init: float
    :param activation: Context-stream activation. Defaults to ``"silu"``.
    :type activation: Any
    :param dot_activation: Gate on the inner-product grade. Defaults ``"silu"``.
    :type dot_activation: Any
    :param gate_activation: GGR ``alpha`` activation. Defaults ``"sigmoid"``.
    :type gate_activation: Any
    :param feature_activation: GGR identity-path activation. Defaults ``"silu"``.
    :type feature_activation: Any
    :param use_gate: Keep the GGR multiplicative ``alpha`` path. ``False`` gives
        a degree-1-homogeneous update.
    :type use_gate: bool
    :param use_bias: Bias on every Dense in the cell (detail projection,
        product projection, GGR gate, carry gate).
    :type use_bias: bool
    :param normalization_type: Context-stream norm; ``None`` ->
        ``LayerNormalization(epsilon=1e-6)``.
    :type normalization_type: Optional[str]
    :param normalization_kwargs: Kwargs for the context-stream norm.
    :type normalization_kwargs: Optional[Dict[str, Any]]
    :param state_normalization_type: Norm applied to ``h_{t-1}``; ``None`` ->
        ``LayerNormalization(epsilon=1e-6)``.
    :type state_normalization_type: Optional[str]
    :param state_normalization_kwargs: Kwargs for the state norm.
    :type state_normalization_kwargs: Optional[Dict[str, Any]]
    :param kernel_initializer: Init for input-facing kernels.
    :type kernel_initializer: Any
    :param recurrent_initializer: Init for the recurrent context kernel.
    :type recurrent_initializer: Any
    :param bias_initializer: Init for biases (carry-gate bias excepted).
    :type bias_initializer: Any
    :param kernel_regularizer: Regularizer for input-facing kernels.
    :type kernel_regularizer: Optional[Any]
    :param recurrent_regularizer: Regularizer for the recurrent kernel.
    :type recurrent_regularizer: Optional[Any]
    :param bias_regularizer: Regularizer for biases.
    :type bias_regularizer: Optional[Any]
    :param dropout_rate: Input dropout rate (one mask reused across timesteps).
    :type dropout_rate: float
    :param recurrent_dropout_rate: State dropout rate; applied to the stream inputs
        only, never to the carry path.
    :type recurrent_dropout_rate: float
    :param seed: Seed for the dropout ``SeedGenerator``.
    :type seed: Optional[int]
    :param kwargs: Passed to ``keras.layers.Layer``.
    """

    def __init__(
        self,
        units: int,
        shifts: Sequence[int] = (1, 2, 4),
        cli_mode: str = "full",
        ctx_mode: str = "diff",
        use_global_context: bool = False,
        global_context_mode: str = "cumulative_mean",
        state_update: str = "gated",
        include_vector_grade: bool = False,
        layer_scale_init: float = 1.0,
        forget_bias_init: float = 1.0,
        activation: Any = "silu",
        dot_activation: Any = "silu",
        gate_activation: Any = "sigmoid",
        feature_activation: Any = "silu",
        use_gate: bool = True,
        use_bias: bool = True,
        normalization_type: Optional[str] = None,
        normalization_kwargs: Optional[Dict[str, Any]] = None,
        state_normalization_type: Optional[str] = None,
        state_normalization_kwargs: Optional[Dict[str, Any]] = None,
        kernel_initializer: Any = "glorot_uniform",
        recurrent_initializer: Any = "orthogonal",
        bias_initializer: Any = "zeros",
        kernel_regularizer: Optional[Any] = None,
        recurrent_regularizer: Optional[Any] = None,
        bias_regularizer: Optional[Any] = None,
        dropout_rate: float = 0.0,
        recurrent_dropout_rate: float = 0.0,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # --- validation (fail loudly at construction, not mid-scan) ---
        if not isinstance(units, int) or isinstance(units, bool) or units <= 0:
            raise ValueError(f"units must be a positive int, got {units!r}")
        if cli_mode not in _CLI_MODES:
            raise ValueError(
                f"cli_mode must be one of {_CLI_MODES}, got {cli_mode!r}"
            )
        if ctx_mode not in _CTX_MODES:
            raise ValueError(
                f"ctx_mode must be one of {_CTX_MODES}, got {ctx_mode!r}"
            )
        if state_update not in _STATE_UPDATES:
            raise ValueError(
                f"state_update must be one of {_STATE_UPDATES}, "
                f"got {state_update!r}"
            )
        if global_context_mode not in _GLOBAL_MODES:
            raise ValueError(
                f"global_context_mode must be one of {_GLOBAL_MODES}, "
                f"got {global_context_mode!r}"
            )
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout_rate}")
        if not 0.0 <= recurrent_dropout_rate < 1.0:
            raise ValueError(
                f"recurrent_dropout must be in [0, 1), got {recurrent_dropout_rate}"
            )
        # The global branch hardcodes shifts=[1, 2]; units < 2 would either
        # silently drop a shift or make the inner layer raise. Fail up front.
        if use_global_context and units < 2:
            raise ValueError(
                f"use_global_context=True requires units >= 2 "
                f"(global branch uses shifts={_GLOBAL_SHIFTS}); got units={units}"
            )
        if state_update == "residual":
            logger.warning(
                "CliffordRNNCell(state_update='residual'): the carry is an "
                "unbounded sum over timesteps (faithful to the vision block, "
                "where the model owns the residual). Prefer 'gated'/'decay' for "
                "long sequences, or normalize the output downstream."
            )

        self.units = units
        self.shifts = list(shifts)
        self.cli_mode = cli_mode
        self.ctx_mode = ctx_mode
        self.use_global_context = use_global_context
        self.global_context_mode = global_context_mode
        self.state_update = state_update
        self.include_vector_grade = include_vector_grade
        self.layer_scale_init = layer_scale_init
        self.forget_bias_init = forget_bias_init
        self.activation = _activation_spec(activation)
        self.dot_activation = _activation_spec(dot_activation)
        self.gate_activation = _activation_spec(gate_activation)
        self.feature_activation = _activation_spec(feature_activation)
        # Resolve eagerly so a keras.layers.Layer activation is rejected at
        # construction (the sibling's contract) rather than mid-scan.
        self._activation_fn = _resolve_activation(self.activation)
        self.use_gate = use_gate
        self.use_bias = use_bias
        self.normalization_type = normalization_type
        self.normalization_kwargs = dict(normalization_kwargs or {})
        self.state_normalization_type = state_normalization_type
        self.state_normalization_kwargs = dict(state_normalization_kwargs or {})
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.dropout_rate = dropout_rate
        self.recurrent_dropout_rate = recurrent_dropout_rate
        self.seed = seed
        self.seed_generator = keras.random.SeedGenerator(seed=seed)

        # --- Step 1: state norm (analogue of the block's input_norm) ---
        # None reproduces the block's hardcoded LayerNormalization(eps=1e-6).
        if state_normalization_type is None:
            self.state_norm = keras.layers.LayerNormalization(
                epsilon=1e-6, name="state_norm"
            )
        else:
            self.state_norm = create_normalization_layer(
                state_normalization_type,
                name="state_norm",
                **self.state_normalization_kwargs,
            )

        # --- Step 2a: detail stream, present timestep only (no mixing) ---
        self.linear_det = keras.layers.Dense(
            units,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name="linear_det",
        )

        # --- Step 2b: context stream, past aggregate ---
        # use_bias=False mirrors the block's DepthwiseConv2D(use_bias=False):
        # the following normalization makes an additive bias redundant.
        self.linear_ctx = keras.layers.Dense(
            units,
            use_bias=False,
            kernel_initializer=recurrent_initializer,
            kernel_regularizer=recurrent_regularizer,
            name="linear_ctx",
        )
        if normalization_type is None:
            self.ctx_norm = keras.layers.LayerNormalization(
                epsilon=1e-6, name="ctx_norm"
            )
        else:
            self.ctx_norm = create_normalization_layer(
                normalization_type,
                name="ctx_norm",
                **self.normalization_kwargs,
            )

        # --- Step 3: local sparse rolling geometric product ---
        self.local_geo_prod = SparseRollingGeometricProduct(
            channels=units,
            shifts=self.shifts,
            cli_mode=cli_mode,
            use_bias=use_bias,
            dot_activation=self.dot_activation,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name="local_geo_prod",
        )

        # --- Optional global (prefix-summary) branch ---
        # shifts / cli_mode hardcoded, exactly as in the vision block.
        if use_global_context:
            self.global_geo_prod = SparseRollingGeometricProduct(
                channels=units,
                shifts=_GLOBAL_SHIFTS,
                cli_mode=_GLOBAL_CLI_MODE,
                use_bias=use_bias,
                dot_activation=self.dot_activation,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name="global_geo_prod",
            )
        else:
            self.global_geo_prod = None

        # --- Step 4: GGR (returns the γ-scaled term only) ---
        # use_bias IS forwarded here, unlike the vision block (see module
        # docstring, deviation 4) so a fully bias-free cell is reachable.
        self.ggr = GatedGeometricResidual(
            channels=units,
            layer_scale_init=layer_scale_init,
            use_bias=use_bias,
            gate_activation=self.gate_activation,
            feature_activation=self.feature_activation,
            use_gate=use_gate,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name="ggr",
        )

        # --- Step 5: carry ---
        if state_update == "gated":
            self.carry_dense = keras.layers.Dense(
                units,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=(
                    initializers.Constant(forget_bias_init)
                    if use_bias
                    else bias_initializer
                ),
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name="carry_dense",
            )
        else:
            self.carry_dense = None

    # ------------------------------------------------------------------
    # RNN cell protocol
    # ------------------------------------------------------------------

    @property
    def state_size(self) -> List[int]:
        """State sizes: ``[h]`` (+ prefix summary, + step counter)."""
        if not self.use_global_context:
            return [self.units]
        if self.global_context_mode == "ema":
            return [self.units, self.units]
        return [self.units, self.units, 1]

    @property
    def output_size(self) -> int:
        """Output dimensionality (the state itself)."""
        return self.units

    def get_initial_state(
        self, batch_size: Optional[int] = None
    ) -> List[keras.KerasTensor]:
        """Zero-initialised state list matching :attr:`state_size`.

        :param batch_size: Batch dimension.
        :return: List of zero tensors.
        """
        states = [
            keras.ops.zeros((batch_size, self.units), dtype=self.compute_dtype)
        ]
        if self.use_global_context:
            states.append(
                keras.ops.zeros(
                    (batch_size, self.units), dtype=self.compute_dtype
                )
            )
            if self.global_context_mode == "cumulative_mean":
                states.append(
                    keras.ops.zeros((batch_size, 1), dtype=self.compute_dtype)
                )
        return states

    # ------------------------------------------------------------------
    # Dropout masks
    # ------------------------------------------------------------------
    # Implemented locally so the cell is correct even without the Keras mixin;
    # ``keras.layers.RNN`` calls these once per sequence, before the scan, so a
    # single mask is shared across timesteps (variational dropout).

    def _make_mask(self, step_input: keras.KerasTensor, rate: float):
        return keras.random.dropout(
            keras.ops.ones_like(step_input),
            rate=rate,
            seed=self.seed_generator,
        )

    def get_dropout_mask(self, step_input: keras.KerasTensor):
        if getattr(self, "_dropout_mask", None) is None and self.dropout_rate > 0:
            self._dropout_mask = self._make_mask(step_input, self.dropout_rate)
        return getattr(self, "_dropout_mask", None)

    def get_recurrent_dropout_mask(self, step_input: keras.KerasTensor):
        if (
            getattr(self, "_recurrent_dropout_mask", None) is None
            and self.recurrent_dropout_rate > 0
        ):
            self._recurrent_dropout_mask = self._make_mask(
                step_input, self.recurrent_dropout_rate
            )
        return getattr(self, "_recurrent_dropout_mask", None)

    def reset_dropout_mask(self) -> None:
        self._dropout_mask = None

    def reset_recurrent_dropout_mask(self) -> None:
        self._recurrent_dropout_mask = None

    # ------------------------------------------------------------------

    def build(self, input_shape: Tuple) -> None:
        """Build sub-layers in dependency order.

        :param input_shape: Per-timestep input shape ``(B, F)``.
        :type input_shape: Tuple
        """
        if len(input_shape) != 2:
            raise ValueError(
                f"{type(self).__name__} expects a per-timestep input of rank 2 "
                f"(batch, features); got input_shape={input_shape}. Wrap the "
                "cell in keras.layers.RNN, or use CliffordRNN directly."
            )
        input_dim = input_shape[-1]
        batch = input_shape[0]
        state_shape = (batch, self.units)

        self.state_norm.build(state_shape)

        self.linear_det.build(input_shape)
        self.linear_ctx.build(state_shape)
        self.ctx_norm.build(state_shape)

        self.local_geo_prod.build(state_shape)
        if self.global_geo_prod is not None:
            self.global_geo_prod.build(state_shape)
        self.ggr.build(state_shape)

        if self.state_update == "gated":
            self.carry_dense.build((batch, (input_dim or 0) + self.units))
        elif self.state_update == "decay":
            # Input-independent per-channel decay (SSM-style). sigmoid(logit)
            # keeps the carry a contraction for any learned value.
            self.decay_logit = self.add_weight(
                name="decay_logit",
                shape=(self.units,),
                initializer=initializers.Constant(self.forget_bias_init),
                trainable=True,
            )

        if self.use_global_context and self.global_context_mode == "ema":
            self.ema_logit = self.add_weight(
                name="ema_logit",
                shape=(self.units,),
                initializer=initializers.Constant(2.0),
                trainable=True,
            )

        super().build(input_shape)

    # ------------------------------------------------------------------

    def call(
        self,
        inputs: keras.KerasTensor,
        states: Any,
        training: Optional[bool] = None,
    ) -> Tuple[keras.KerasTensor, List[keras.KerasTensor]]:
        """Advance one timestep.

        :param inputs: Input at time *t*, ``(B, F)``.
        :type inputs: keras.KerasTensor
        :param states: State list ``[h]`` (+ ``c``, + ``n``).
        :type states: Any
        :param training: Training mode flag (drives dropout).
        :type training: Optional[bool]
        :return: ``(output, new_states)`` with ``output == new_states[0]``.
        :rtype: Tuple[keras.KerasTensor, List[keras.KerasTensor]]
        """
        states = list(states) if isinstance(states, (list, tuple)) else [states]
        h_prev = states[0]

        # --- dropout -------------------------------------------------
        if training and self.dropout_rate > 0.0:
            mask = self.get_dropout_mask(inputs)
            if mask is not None:
                inputs = inputs * mask
        h_used = h_prev
        if training and self.recurrent_dropout_rate > 0.0:
            rmask = self.get_recurrent_dropout_mask(h_prev)
            if rmask is not None:
                # Streams only. The carry below uses the UNDROPPED h_prev:
                # zeroing the carry would erase the state, not regularise it.
                h_used = h_prev * rmask

        # --- Step 1/2: dual-stream generation ------------------------
        h_norm = self.state_norm(h_used)
        z_det = self.linear_det(inputs)  # PRESENT (grade-1 detail)
        z_ctx = self._activation_fn(
            self.ctx_norm(self.linear_ctx(h_norm), training=training)
        )  # PAST (aggregate)

        if self.ctx_mode == "diff":
            # Temporal Laplacian: how the accumulated past differs from the
            # present observation, i.e. a prediction-error / surprise signal.
            z_ctx = z_ctx - z_det

        # --- Step 3: local geometric interaction ---------------------
        g_feat = self.local_geo_prod(z_det, z_ctx)

        # --- Step 4: optional global (prefix) branch ------------------
        new_states: List[keras.KerasTensor] = []
        if self.global_geo_prod is not None:
            c_prev = states[1]
            if self.global_context_mode == "cumulative_mean":
                n_new = states[2] + 1.0
                # Exact running mean of z_det over the prefix; n starts at 0 so
                # the first step is a plain assignment (no division by zero).
                c_new = c_prev + (z_det - c_prev) / n_new
                new_states = [c_new, n_new]
            else:
                rho = keras.ops.sigmoid(self.ema_logit)
                c_new = rho * c_prev + (1.0 - rho) * z_det
                new_states = [c_new]
            # Hardcoded differential context, as in the vision block.
            g_feat = g_feat + self.global_geo_prod(z_det, c_new - z_det)

        if self.include_vector_grade:
            # Grade-1 (vector) part of the multivector: a direct additive path
            # for the present observation. Off by default (purity).
            g_feat = g_feat + z_det

        # --- Step 5: GGR term + carry --------------------------------
        term = self.ggr(h_norm, g_feat, training=training)

        if self.state_update == "gated":
            gate_in = keras.ops.concatenate([inputs, h_norm], axis=-1)
            # Sigmoid is hardcoded here (unlike gate_activation, which feeds the
            # GGR alpha): the carry must be a bounded contraction or the state
            # diverges over long sequences.
            carry = keras.ops.sigmoid(self.carry_dense(gate_in))
            h_new = carry * h_prev + term
        elif self.state_update == "decay":
            h_new = keras.ops.sigmoid(self.decay_logit) * h_prev + term
        else:  # "residual" -- faithful to the block's external residual add
            h_new = h_prev + term

        return h_new, [h_new] + new_states

    # ------------------------------------------------------------------

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Per-timestep output shape ``(B, units)``."""
        return (*input_shape[:-1], self.units)

    # ------------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        """Return serialisable configuration."""
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "shifts": list(self.shifts),
                "cli_mode": self.cli_mode,
                "ctx_mode": self.ctx_mode,
                "use_global_context": self.use_global_context,
                "global_context_mode": self.global_context_mode,
                "state_update": self.state_update,
                "include_vector_grade": self.include_vector_grade,
                "layer_scale_init": self.layer_scale_init,
                "forget_bias_init": self.forget_bias_init,
                "activation": _serialize_activation(self.activation),
                "dot_activation": _serialize_activation(self.dot_activation),
                "gate_activation": _serialize_activation(self.gate_activation),
                "feature_activation": _serialize_activation(
                    self.feature_activation
                ),
                "use_gate": self.use_gate,
                "use_bias": self.use_bias,
                "normalization_type": self.normalization_type,
                "normalization_kwargs": dict(self.normalization_kwargs),
                "state_normalization_type": self.state_normalization_type,
                "state_normalization_kwargs": dict(
                    self.state_normalization_kwargs
                ),
                "kernel_initializer": initializers.serialize(
                    self.kernel_initializer
                ),
                "recurrent_initializer": initializers.serialize(
                    self.recurrent_initializer
                ),
                "bias_initializer": initializers.serialize(
                    self.bias_initializer
                ),
                "kernel_regularizer": regularizers.serialize(
                    self.kernel_regularizer
                ),
                "recurrent_regularizer": regularizers.serialize(
                    self.recurrent_regularizer
                ),
                "bias_regularizer": regularizers.serialize(
                    self.bias_regularizer
                ),
                "dropout_rate": self.dropout_rate,
                "recurrent_dropout_rate": self.recurrent_dropout_rate,
                "seed": self.seed,
            }
        )
        return config


# ---------------------------------------------------------------------------
# CliffordRNN
# ---------------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class CliffordRNN(keras.layers.RNN):
    """Clifford geometric-algebra recurrent layer.

    Wraps :class:`CliffordRNNCell` in ``keras.layers.RNN``, exposing the same
    call/constructor surface as ``keras.layers.GRU`` / ``keras.layers.LSTM``:
    swapping ``GRU(64)`` for ``CliffordRNN(64)`` requires no other change.
    Masking, ``stateful``, ``unroll``, ``go_backwards``, ``return_state``,
    ``zero_output_for_mask``, ``reset_state()`` and ``Bidirectional`` all come
    from the base class unmodified.

    Call arguments:
        sequences: 3-D tensor ``(batch, timesteps, features)``.
        initial_state: Optional list of initial state tensors.
        mask: Optional binary mask ``(batch, timesteps)``.
        training: Whether the layer is in training mode.

    Output shape:
        ``(batch, timesteps, units)`` if ``return_sequences`` else
        ``(batch, units)``; plus the state tensors if ``return_state``.

    :param units: State / output dimensionality.
    :type units: int
    :param kwargs: Cell arguments (see :class:`CliffordRNNCell`) and the
        standard ``keras.layers.RNN`` arguments.

    .. code-block:: python

        x = keras.Input((None, 32))
        y = CliffordRNN(64, shifts=[1, 2, 4], return_sequences=True)(x)
    """

    def __init__(
        self,
        units: int,
        shifts: Sequence[int] = (1, 2, 4),
        cli_mode: str = "full",
        ctx_mode: str = "diff",
        use_global_context: bool = False,
        global_context_mode: str = "cumulative_mean",
        state_update: str = "gated",
        include_vector_grade: bool = False,
        layer_scale_init: float = 1.0,
        forget_bias_init: float = 1.0,
        activation: Any = "silu",
        dot_activation: Any = "silu",
        gate_activation: Any = "sigmoid",
        feature_activation: Any = "silu",
        use_gate: bool = True,
        use_bias: bool = True,
        normalization_type: Optional[str] = None,
        normalization_kwargs: Optional[Dict[str, Any]] = None,
        state_normalization_type: Optional[str] = None,
        state_normalization_kwargs: Optional[Dict[str, Any]] = None,
        kernel_initializer: Any = "glorot_uniform",
        recurrent_initializer: Any = "orthogonal",
        bias_initializer: Any = "zeros",
        kernel_regularizer: Optional[Any] = None,
        recurrent_regularizer: Optional[Any] = None,
        bias_regularizer: Optional[Any] = None,
        dropout_rate: float = 0.0,
        recurrent_dropout_rate: float = 0.0,
        seed: Optional[int] = None,
        return_sequences: bool = False,
        return_state: bool = False,
        go_backwards: bool = False,
        stateful: bool = False,
        unroll: bool = False,
        zero_output_for_mask: bool = False,
        **kwargs: Any,
    ) -> None:
        cell = CliffordRNNCell(
            units=units,
            shifts=shifts,
            cli_mode=cli_mode,
            ctx_mode=ctx_mode,
            use_global_context=use_global_context,
            global_context_mode=global_context_mode,
            state_update=state_update,
            include_vector_grade=include_vector_grade,
            layer_scale_init=layer_scale_init,
            forget_bias_init=forget_bias_init,
            activation=activation,
            dot_activation=dot_activation,
            gate_activation=gate_activation,
            feature_activation=feature_activation,
            use_gate=use_gate,
            use_bias=use_bias,
            normalization_type=normalization_type,
            normalization_kwargs=normalization_kwargs,
            state_normalization_type=state_normalization_type,
            state_normalization_kwargs=state_normalization_kwargs,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            dropout_rate=dropout_rate,
            recurrent_dropout_rate=recurrent_dropout_rate,
            seed=seed,
            name="clifford_rnn_cell",
            dtype=kwargs.get("dtype", None),
            trainable=kwargs.get("trainable", True),
        )
        super().__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            zero_output_for_mask=zero_output_for_mask,
            **kwargs,
        )
        self.input_spec = keras.layers.InputSpec(ndim=3)

    # ------------------------------------------------------------------

    def call(
        self,
        sequences: keras.KerasTensor,
        initial_state: Optional[Any] = None,
        mask: Optional[keras.KerasTensor] = None,
        training: bool = False,
    ) -> Any:
        """Run the recurrence (delegated to ``keras.layers.RNN``)."""
        return super().call(
            sequences,
            initial_state=initial_state,
            mask=mask,
            training=training,
        )

    # ------------------------------------------------------------------
    # Convenience pass-throughs (parity with GRU / LSTM)
    # ------------------------------------------------------------------

    @property
    def units(self) -> int:
        return self.cell.units

    @property
    def shifts(self) -> List[int]:
        return list(self.cell.shifts)

    @property
    def cli_mode(self) -> str:
        return self.cell.cli_mode

    @property
    def ctx_mode(self) -> str:
        return self.cell.ctx_mode

    @property
    def state_update(self) -> str:
        return self.cell.state_update

    @property
    def dropout(self) -> float:
        return self.cell.dropout_rate

    @property
    def recurrent_dropout(self) -> float:
        return self.cell.recurrent_dropout_rate

    # ------------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        """Flatten the cell config into the layer config (as GRU/LSTM do)."""
        cell = self.cell
        config = {
            "units": cell.units,
            "shifts": list(cell.shifts),
            "cli_mode": cell.cli_mode,
            "ctx_mode": cell.ctx_mode,
            "use_global_context": cell.use_global_context,
            "global_context_mode": cell.global_context_mode,
            "state_update": cell.state_update,
            "include_vector_grade": cell.include_vector_grade,
            "layer_scale_init": cell.layer_scale_init,
            "forget_bias_init": cell.forget_bias_init,
            "activation": _serialize_activation(cell.activation),
            "dot_activation": _serialize_activation(cell.dot_activation),
            "gate_activation": _serialize_activation(cell.gate_activation),
            "feature_activation": _serialize_activation(
                cell.feature_activation
            ),
            "use_gate": cell.use_gate,
            "use_bias": cell.use_bias,
            "normalization_type": cell.normalization_type,
            "normalization_kwargs": dict(cell.normalization_kwargs),
            "state_normalization_type": cell.state_normalization_type,
            "state_normalization_kwargs": dict(
                cell.state_normalization_kwargs
            ),
            "kernel_initializer": initializers.serialize(
                cell.kernel_initializer
            ),
            "recurrent_initializer": initializers.serialize(
                cell.recurrent_initializer
            ),
            "bias_initializer": initializers.serialize(cell.bias_initializer),
            "kernel_regularizer": regularizers.serialize(
                cell.kernel_regularizer
            ),
            "recurrent_regularizer": regularizers.serialize(
                cell.recurrent_regularizer
            ),
            "bias_regularizer": regularizers.serialize(cell.bias_regularizer),
            "dropout_rate": cell.dropout_rate,
            "recurrent_dropout_rate": cell.recurrent_dropout_rate,
            "seed": cell.seed,
        }
        base_config = super().get_config()
        # The cell is rebuilt from the flattened kwargs above; keeping the
        # serialised cell would duplicate it and break `cls(**config)`.
        del base_config["cell"]
        return {**base_config, **config}

    @classmethod
    def from_config(
        cls, config: Dict[str, Any], custom_objects: Optional[Any] = None
    ) -> "CliffordRNN":
        """Rebuild from a flattened config (overrides ``RNN.from_config``)."""
        return cls(**config)


# ---------------------------------------------------------------------------