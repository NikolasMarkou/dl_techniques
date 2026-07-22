"""
CliffordNet block and constituent primitives.

Implements the geometric-algebra vision block from
    Zhongping Ji, "CliffordNet: All You Need is Geometric Algebra",
    arXiv:2601.06793v2 (2026).  Reference code: github.com/ParaMind2025/CAN

================================================================================
Theory — geometric algebra as a single, unified interaction primitive
================================================================================

Motivation
----------
Modern vision backbones factor each block into two engineered stages: a spatial
token mixer (self-attention or convolution) and a channel mixer (a heavy
Feed-Forward Network / MLP).  CliffordNet rejects that decomposition.  It argues
that ONE algebraic operation — the Clifford *geometric product* — can carry both
roles simultaneously, at strictly linear cost, so no FFN is required.  The
guiding thesis is that "global understanding is an emergent property of rigorous
local processing": dense, algebraically-complete local interaction stands in for
both global attention and channel-mixing MLPs.

The geometric product
---------------------
For two multivectors (here, per-pixel channel vectors) ``u`` and ``v`` the
geometric product splits into a symmetric and an antisymmetric part::

    u v  =  u · v      +      u ∧ v
            (inner)          (wedge / exterior)
            "coherence"      "structure"

* The **inner product** ``u · v`` is the generalized dot product.  It measures
  feature *coherence* / alignment — how strongly the detail stream agrees with
  its local context.  CliffordNet realizes it as a gated Hadamard term,
  ``SiLU(u ⊙ v)``, i.e. an alignment-controlled diffusion / gate.

* The **wedge product** ``u ∧ v`` is the antisymmetric bivector — the oriented
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
  ``Z_det = Linear(X_norm)`` — a 1×1 pointwise projection.
* **Context stream** (local aggregation):
  ``Z_ctx = act(Norm(DWConv(DWConv(X_norm))))`` — two stacked depthwise
  convolutions (~5×5 effective receptive field) aggregating local structure.

An optional *differential* (Laplacian) coupling sharpens the interaction::

    ctx_mode="diff":  Z_ctx <- Z_ctx − Z_det     (discrete Laplacian Δ)
    ctx_mode="abs" :  Z_ctx                        (pure aggregation)

The two streams then interact through the geometric product to produce the
geometric feature ``G_feat`` that drives the state update.

Sparse rolling geometric product (linear complexity)
----------------------------------------------------
A full channel-pairwise product is O(D²).  CliffordNet samples only a few
diagonals of that interaction matrix via cyclic channel shifts (rolls), giving
O(N · D · |shifts|) — linear in both tokens ``N`` and channels ``D``.  For each
offset ``s`` in ``shifts`` (rolling ``Z_ctx`` by ``s`` along the channel axis)::

    dot_s[c]   = act( Z_det[c] · Z_ctx[(c−s) mod D] )        # inner / coherence
    wedge_s[c] = Z_det[c] · Z_ctx[(c−s) mod D]
               − Z_ctx[c] · Z_det[(c−s) mod D]               # wedge / bivector

The per-shift dot and wedge tensors are concatenated and projected back to ``D``
channels by a learnable Dense ``P``.  Exponentially spaced shifts (1, 2, 4, 8, …)
impose a ring topology with logarithmic mixing range.

Gated Geometric Residual (GGR) — an Euler step of a feature ODE
---------------------------------------------------------------
The block treats depth as time and takes a first-order Euler step of a
continuous geometric evolution ``∂H/∂t = f(H, G_feat)``::

    H_out = H_prev + γ ⊙ ( SiLU(H_norm) + α ⊙ G_feat )

* ``γ`` — LayerScale, a per-channel scale initialized ≈ 0 so the block starts
  near identity (stable very deep stacks).
* ``SiLU(H_norm)`` — conditions the identity / state path.
* ``α = sigmoid(Gate([H_norm, G_feat]))`` — a learned gate blending the identity
  path with the injected geometric interaction.

Global context branch (optional)
--------------------------------
A whole-image summary ``C_glo = GlobalAvgPool(X_norm)`` runs the same geometric
product (hardcoded ``shifts=[1, 2]``, ``cli_mode="full"``, differential context)
and is superposed onto the local ``G_feat``, adding multi-scale awareness when
enabled.

Efficiency
----------
With ZERO FFN blocks, CliffordNet sets a new parameter-efficiency Pareto frontier
on CIFAR-100: ~1.4M params -> 77.82% (matching ResNet-18's 76.75% at ~8× fewer
params); ~2.6M -> 79.05% (beating MobileNetV2 and ViT-Tiny at similar size);
larger variants surpass ResNet-50 / DenseNet-121 at a fraction of the parameters.

Implementation note (this file)
-------------------------------
:class:`GatedGeometricResidual` returns ONLY the γ-scaled term
``γ ⊙ (SiLU(H_norm) + α ⊙ G_feat)``; the residual add ``H_prev + …`` and any
stochastic-depth (drop-path) are performed EXTERNALLY by the caller / model, so
the computation graph is explicit and manually inspectable.  A bias-free,
degree-1-homogeneous configuration (linear final projection, gate removed,
variance-only norms) is used by the Clifford denoiser (Miyasawa-compliant).

Key primitives
--------------
- :class:`SparseRollingGeometricProduct`  -- shifted dot + wedge interaction
- :class:`GatedGeometricResidual`         -- GGR update with LayerScale
- :class:`CliffordNetBlock`               -- full isotropic block (no FFN)

Fixes vs previous version
--------------------------
1. ``SparseRollingGeometricProduct``: filter ``shifts`` to keep only
   ``s < channels`` (matches ``CliffordInteraction_PyTorch`` behaviour).
2. ``CliffordNetBlock`` context stream: two stacked ``DepthwiseConv2D``
   (effective 5×5 RF) with a single ``BatchNormalization`` after both,
   matching the original ``get_context_local`` sequential.
3. ``CliffordNetBlock`` global branch: shifts hardcoded to ``[1, 2]``,
   ``cli_mode`` hardcoded to ``"full"`` (original always uses these for
   the global interaction regardless of block settings).
4. ``CliffordNetBlock.call``: differential context (``c_glo -= z_det``)
   is applied unconditionally to the global branch, matching the original's
   hardcoded ``ctx_mode='diff'`` inside the global interaction.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import keras
from keras import initializers, regularizers

from ..norms.factory import create_normalization_layer
from ...utils.logger import logger

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

CliMode = Literal["inner", "wedge", "full"]
CtxMode = Literal["diff", "abs"]
SkipPool = Literal["avg", "max"]

# Global-branch constants matching the original implementation
_GLOBAL_SHIFTS: List[int] = [1, 2]
_GLOBAL_CLI_MODE: CliMode = "full"


def _resolve_activation(activation: Any) -> Any:
    """Resolve an activation spec to a callable.

    Matches the ``ctx_activation`` idiom already used in this module
    (``keras.activations.get``). Strings are resolved via
    ``keras.activations.get``; ``None`` maps to identity (linear);
    callables / keras layer instances are returned as-is so they can be
    invoked directly.

    :param activation: String name, ``None``, or a callable/layer instance.
    :return: A callable applying the activation.
    """
    if activation is None:
        return keras.activations.linear
    if isinstance(activation, str):
        return keras.activations.get(activation)
    return activation


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
    is D_s[c] = SiLU(Z_det[c] * Z_ctx[(c-s) % D]) and the wedge component is
    W_s[c] = Z_det[c] * Z_ctx[(c-s)%D] - Z_ctx[c] * Z_det[(c-s)%D].

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
    :type shifts: List[int]
    :param cli_mode: Components to retain
        (``"inner"``, ``"wedge"``, ``"full"``). Defaults to ``"full"``.
    :type cli_mode: CliMode
    :param use_bias: Whether the projection Dense uses a bias.
    :type use_bias: bool
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
        shifts: List[int],
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
        if not shifts:
            raise ValueError("shifts must be a non-empty list")
        if cli_mode not in ("inner", "wedge", "full"):
            raise ValueError(
                f"cli_mode must be 'inner', 'wedge', or 'full', got {cli_mode!r}"
            )
        # Reject shifts <= 0. s=0 makes the wedge term
        # identically zero and wastes a slot in the proj input; negative
        # shifts are accepted by keras.ops.roll but are almost certainly
        # unintended.
        for _s in shifts:
            if not isinstance(_s, (int,)) or isinstance(_s, bool) or _s < 1:
                raise ValueError(
                    f"shifts must be a list of ints >= 1; got {shifts!r}"
                )

        self.channels = channels
        # Filter out offsets >= channels: a full cyclic roll contributes
        # no new information (matches CliffordInteraction_PyTorch behaviour).
        self.shifts = [s for s in shifts if s < channels]
        if not self.shifts:
            raise ValueError(
                f"All provided shifts {shifts} are >= channels ({channels}). "
                "No valid shifts remain after filtering."
            )
        _dropped = [s for s in shifts if s >= channels]
        if _dropped:
            logger.warning(
                "SparseRollingGeometricProduct dropping shifts %s "
                "(>= channels=%d); kept shifts=%s",
                _dropped, channels, self.shifts,
            )
        self.cli_mode = cli_mode
        self.use_bias = use_bias
        # DECISION plan_2026-07-01_6dc255c1/D-001: dot activation is now a
        # ctor kwarg. Default "silu" reproduces the previously-hardcoded
        # keras.activations.silu byte-identically. Do NOT re-hardcode SiLU in
        # call(); a homogeneous denoiser selects e.g. "leaky_relu" here.
        self.dot_activation = dot_activation
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        # Number of concatenated channels before projection
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

    # ------------------------------------------------------------------

    def build(self, input_shape: Tuple) -> None:
        """Build the projection layer.

        :param input_shape: Shape of a *single* input tensor ``(B, H, W, D)``.
        :type input_shape: Tuple
        """
        self._input_shape_for_build = input_shape
        self.proj.build((*input_shape[:-1], self._proj_input_dim))
        super().build(input_shape)

    def get_build_config(self) -> Dict[str, Any]:
        if hasattr(self, "_input_shape_for_build"):
            return {"input_shape": self._input_shape_for_build}
        return {}

    def build_from_config(self, config: Dict[str, Any]) -> None:
        if "input_shape" in config:
            self.build(config["input_shape"])

    # ------------------------------------------------------------------

    def call(
        self,
        z_det: keras.KerasTensor,
        z_ctx: keras.KerasTensor,
        **kwargs: Any,
    ) -> keras.KerasTensor:
        """Compute sparse geometric product and project.

        :param z_det: Detail stream  ``(B, H, W, D)``.
        :type z_det: keras.KerasTensor
        :param z_ctx: Context stream ``(B, H, W, D)``.
        :type z_ctx: keras.KerasTensor
        :return: Projected interaction tensor ``(B, H, W, channels)``.
        :rtype: keras.KerasTensor
        """
        components: List[keras.KerasTensor] = []

        for s in self.shifts:
            z_ctx_s = keras.ops.roll(z_ctx, shift=s, axis=-1)

            if self.cli_mode in ("wedge", "full"):
                # Bivector: anti-symmetric cross-term. z_det_s is only
                # needed for the wedge branch; skip it for cli_mode='inner'.
                z_det_s = keras.ops.roll(z_det, shift=s, axis=-1)
                wedge = z_det * z_ctx_s - z_ctx * z_det_s
                components.append(wedge)

            if self.cli_mode in ("inner", "full"):
                # Scalar: gated inner product.
                # DECISION plan_2026-07-01_6dc255c1/D-001: resolve from
                # self.dot_activation (default "silu" == old hardcoded
                # keras.activations.silu). Do NOT re-hardcode SiLU here —
                # existing consumers rely on the byte-identical default while a
                # homogeneous denoiser needs a degree-1 activation (LeakyReLU).
                dot = _resolve_activation(self.dot_activation)(z_det * z_ctx_s)
                components.append(dot)

        g_raw = keras.ops.concatenate(components, axis=-1)
        return self.proj(g_raw)

    # ------------------------------------------------------------------

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape.

        :param input_shape: Shape of one input stream ``(B, H, W, D)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape ``(B, H, W, channels)``.
        :rtype: Tuple[Optional[int], ...]
        """
        return (*input_shape[:-1], self.channels)

    # ------------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        """Return serialisable configuration.

        :return: Dictionary with all constructor arguments.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "channels": self.channels,
                # Stores the post-filter shift list (s < channels). Round-trip
                # serialisation is idempotent for a fixed `channels`; if the
                # caller reconstructs with a different `channels`, any shifts
                # the original constructor dropped are not recoverable.
                "shifts": self.shifts,
                "cli_mode": self.cli_mode,
                "use_bias": self.use_bias,
                "dot_activation": self.dot_activation,
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
    H_out = H_prev + gamma * (SiLU(H_norm) + alpha * G_feat), where alpha
    is a learned sigmoid gate on concat(H_norm, G_feat) and gamma is a
    LayerScale scalar initialised near zero. This layer returns ONLY the
    LayerScale-gated term; the residual add and any stochastic-depth op are
    external, model-level operations.

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
          │  SiLU(H)  │  │     α · G_feat     │
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
        True. Set to False for bias-free (Miyasawa-compliant) denoising blocks.
    :type use_bias: bool
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
        # DECISION plan_2026-07-01_6dc255c1/D-001: gate/feature activations and
        # the gate itself are now ctor kwargs. Defaults ("sigmoid", "silu",
        # use_gate=True) reproduce the previously-hardcoded GGR update
        # byte-identically. Do NOT re-hardcode sigmoid/SiLU. use_gate=False
        # drops the multiplicative alpha*g_feat path (degree-2 in the input),
        # which the homogeneous denoiser requires — see D-001.
        self.gate_activation = gate_activation
        self.feature_activation = feature_activation
        self.use_gate = use_gate
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        # Learned gate: Dense(2C -> C) followed by sigmoid.
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
        # Two known, accepted consequences (both verified, neither a regression
        # relative to the pre-fix behaviour):
        #   1. `model.trainable = True` (the standard unfreeze idiom, used at
        #      src/train/bfunet/variance_probe.py:177) RE-ENABLES gate_dense and
        #      brings the warning back — the setter recurses into `_layers`
        #      (layer.py:581-582) and has no knowledge of `use_gate`. Re-apply
        #      this guard manually after any global unfreeze.
        #   2. Resuming from a `.keras` saved BEFORE this change, WITH optimizer
        #      state, skips optimizer loading entirely (24 saved vars vs 20
        #      expected) — including `iterations`, so the LR schedule restarts.
        #      Keras only warns; it does not error. No such checkpoint exists in
        #      results/ today, and common.py:2030 resumes weights-only, so live
        #      exposure is zero. Delete any pre-fix optimizer state rather than
        #      trusting a resume from it.
        if not self.use_gate:
            self.gate_dense.trainable = False
        # DECISION plan_2026-07-03_eb53492e/D-001: GGR no longer owns a
        # stochastic-depth op — the residual add AND the stochastic-depth layer
        # are now external, model-level ops (x = x + SD(rate)(block(x))).
        # Do NOT re-inline a stochastic-depth layer or a per-block rate kwarg
        # here; see decisions.md D-001.

    # ------------------------------------------------------------------

    def build(self, input_shape: Tuple) -> None:
        """Build LayerScale and the gate projection.

        :param input_shape: Shape of a single input stream ``(B, H, W, D)``.
        :type input_shape: Tuple
        """
        self.gamma = self.add_weight(
            name="gamma",
            shape=(self.channels,),
            initializer=initializers.Constant(self.layer_scale_init),
            trainable=True,
        )
        gate_input_shape = (*input_shape[:-1], 2 * self.channels)
        self.gate_dense.build(gate_input_shape)
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
        :type h_norm: keras.KerasTensor
        :param g_feat: Geometric interaction features ``(B, H, W, D)``.
        :type g_feat: keras.KerasTensor
        :param training: Whether in training mode.
        :type training: Optional[bool]
        :return: Scaled residual term ``(B, H, W, D)``; caller adds to H_prev.
        :rtype: keras.KerasTensor
        """
        # DECISION plan_2026-07-01_6dc255c1/D-001: resolve gate/feature
        # activations from ctor kwargs. Do NOT re-hardcode sigmoid/SiLU — the
        # defaults ("sigmoid"/"silu", use_gate=True) reproduce the old update
        # byte-identically; existing consumers depend on that.
        feat = _resolve_activation(self.feature_activation)(h_norm)
        if self.use_gate:
            gate_input = keras.ops.concatenate([h_norm, g_feat], axis=-1)
            alpha = _resolve_activation(self.gate_activation)(
                self.gate_dense(gate_input)
            )
            h_mix = feat + alpha * g_feat
        else:
            # DECISION plan_2026-07-01_6dc255c1/D-001: the multiplicative
            # alpha*g_feat gate is degree-2 in the input and breaks strict
            # degree-1 homogeneity (Miyasawa). Do NOT keep it here — use g_feat
            # directly on the homogeneous path.
            h_mix = feat + g_feat
        h_mix = h_mix * self.gamma

        return h_mix

    # ------------------------------------------------------------------

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape.

        :param input_shape: Shape of a single input stream ``(B, H, W, D)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape ``(B, H, W, channels)``.
        :rtype: Tuple[Optional[int], ...]
        """
        return (*input_shape[:-1], self.channels)

    # ------------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        """Return serialisable configuration.

        :return: Dictionary with all constructor arguments.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "channels": self.channels,
                "layer_scale_init": self.layer_scale_init,
                "use_bias": self.use_bias,
                "gate_activation": self.gate_activation,
                "feature_activation": self.feature_activation,
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

    Implements the geometric-algebra vision block from arXiv:2601.06793v2 par. 9.
    A dual-stream architecture generates detail Z_det = Linear(X_norm) and
    context Z_ctx = SiLU(BN(DWConv(DWConv(X_norm)))) streams, optionally
    applying a discrete Laplacian (Z_ctx -= Z_det). The streams interact via
    a sparse rolling geometric product, and are combined through a Gated
    Geometric Residual (GGR) update. ``call()`` returns ONLY this transformed
    term (``h_mix``); the residual add is an external, model-level op. An
    optional global branch uses GAP-based context with hardcoded shifts=[1,2]
    and cli_mode='full'.

    .. note::

        When ``use_global_context=True``, the global branch uses fixed
        ``shifts=[1, 2]``, ``cli_mode='full'``, and differential context
        regardless of the caller's ``shifts`` / ``cli_mode`` / ``ctx_mode``
        settings. The global branch is a compact whole-image summary and
        deliberately decouples its hyperparameters from the local branch.

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
    :type shifts: List[int]
    :param cli_mode: Algebraic components for the local interaction
        (``"inner"``, ``"wedge"``, ``"full"``). Defaults to ``"full"``.
    :type cli_mode: CliMode
    :param ctx_mode: Context mode (``"diff"`` or ``"abs"``).
        Defaults to ``"diff"``.
    :type ctx_mode: CtxMode
    :param use_global_context: Whether to add a global-average-pool branch.
        Defaults to ``False``.
    :type use_global_context: bool
    :param layer_scale_init: Initial LayerScale value. Defaults to 1e-5.
    :type layer_scale_init: float
    :param use_bias: Whether Dense layers use bias. Defaults to ``True``.
    :type use_bias: bool
    :param kernel_initializer: Kernel initializer for Dense layers.
    :type kernel_initializer: Any
    :param bias_initializer: Bias initializer for Dense layers.
    :type bias_initializer: Any
    :param kernel_regularizer: Kernel regularizer for Dense layers.
    :type kernel_regularizer: Optional[Any]
    :param bias_regularizer: Bias regularizer for Dense layers.
    :type bias_regularizer: Optional[Any]
    :param kwargs: Passed to ``keras.layers.Layer``.
    """

    def __init__(
        self,
        channels: int,
        shifts: List[int],
        cli_mode: CliMode = "full",
        ctx_mode: CtxMode = "diff",
        use_global_context: bool = False,
        causal: bool = False,
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
        normalization_type: str = "batch_norm",
        normalization_kwargs: Optional[Dict[str, Any]] = None,
        input_normalization_type: Optional[str] = None,
        input_normalization_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if ctx_mode not in ("diff", "abs"):
            raise ValueError(f"ctx_mode must be 'diff' or 'abs', got {ctx_mode!r}")
        if not isinstance(context_kernel_size, int) or isinstance(
            context_kernel_size, bool
        ) or context_kernel_size <= 0:
            raise ValueError(
                f"context_kernel_size must be a positive int, "
                f"got {context_kernel_size!r}"
            )
        # The global branch hardcodes shifts=[1, 2]; with
        # channels < 2 the inner SRGP filter would either silently drop
        # shifts (channels=2 -> only shift=1 remains, warning) or reject
        # the layer entirely (channels=1 -> ValueError). Fail up front.
        if use_global_context and channels < 2:
            raise ValueError(
                f"use_global_context=True requires channels >= 2 "
                f"(global branch uses shifts=[1, 2]); got channels={channels}"
            )

        self.channels = channels
        self.shifts = list(shifts)
        self.cli_mode = cli_mode
        self.ctx_mode = ctx_mode
        self.use_global_context = use_global_context
        self.causal = causal
        self.layer_scale_init = layer_scale_init
        self.use_bias = use_bias
        # DECISION plan_2026-07-01_6dc255c1/D-001: context-stream activation and
        # the geometric-product / gate activations are now ctor kwargs threaded
        # to the internal SRGP and GGR. Defaults ("silu"/"silu"/"sigmoid"/
        # "silu", use_gate=True) reproduce today's behavior byte-identically.
        # Do NOT re-hardcode SiLU/sigmoid.
        self.activation = activation
        self.dot_activation = dot_activation
        self.gate_activation = gate_activation
        self.feature_activation = feature_activation
        self.use_gate = use_gate
        self.context_kernel_size = context_kernel_size
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.normalization_type = normalization_type
        self.normalization_kwargs = dict(normalization_kwargs or {})
        self.input_normalization_type = input_normalization_type
        self.input_normalization_kwargs = dict(input_normalization_kwargs or {})

        _dense_kwargs: Dict[str, Any] = dict(
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )

        # --- Step 1: Input norm ---
        # DECISION plan_2026-07-01_6dc255c1/D-001: input_normalization_type=None
        # reproduces the previously-hardcoded center=True LayerNormalization
        # (epsilon=1e-6, name="input_norm") byte-identically. Do NOT drop that
        # default branch — existing Clifford consumers depend on it. A non-None
        # value routes through create_normalization_layer (e.g.
        # "bias_free_batch_norm" for the homogeneous denoiser).
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

        # --- Step 2a: Detail stream (1×1 pointwise) ---
        self.linear_det = keras.layers.Dense(
            channels, name="linear_det", **_dense_kwargs
        )

        # --- Step 2b: Context stream ---
        # Two stacked KxK depthwise convolutions (K=context_kernel_size, default
        # 3 -> effective (2K-1)x(2K-1) = 5×5 RF), one configurable normalization
        # layer after both, then SiLU in call(). Default normalization_type is
        # "batch_norm" (matches the original CliffordBlock's BatchNormalization);
        # pass normalization_type="zero_centered_rms_norm" for the bias-free /
        # degree-1-homogeneous denoiser configuration.
        # DECISION plan_2026-07-04_d2ac2f68/D-001: the vision and causal Clifford
        # blocks are unified via this single `causal` flag. It gates the four (and
        # only four) behavioral differences: (a) context depthwise-conv geometry
        # ((1,K)/valid + explicit left-pad vs (K,K)/same) here; (b) the build()
        # shapes (left-padded); (c) the call() causal padding before each DWConv;
        # (d) the global-context statistic (causal cumulative mean vs full-image
        # GAP). CausalCliffordNetBlock is now a THIN subclass that only forces
        # causal=True + a norm default. Do NOT re-duplicate the block body into the
        # subclass, and do NOT collapse these two paths — the causal path relies on
        # `H=1` sequence tensors and left-only padding along axis=2.
        if self.causal:
            self.dw_conv = keras.layers.DepthwiseConv2D(
                kernel_size=(1, self.context_kernel_size),
                padding="valid",
                use_bias=False,
                name="dw_conv",
            )
            self.dw_conv2 = keras.layers.DepthwiseConv2D(
                kernel_size=(1, self.context_kernel_size),
                padding="valid",
                use_bias=False,
                name="dw_conv2",
            )
        else:
            self.dw_conv = keras.layers.DepthwiseConv2D(
                kernel_size=self.context_kernel_size,
                padding="same",
                use_bias=False,
                name="dw_conv",
            )
            self.dw_conv2 = keras.layers.DepthwiseConv2D(
                kernel_size=self.context_kernel_size,
                padding="same",
                use_bias=False,
                name="dw_conv2",
            )
        self.ctx_norm = create_normalization_layer(
            self.normalization_type,
            name="ctx_bn",
            **self.normalization_kwargs,
        )

        # --- Step 3: Local sparse rolling product ---
        self.local_geo_prod = SparseRollingGeometricProduct(
            channels=channels,
            shifts=shifts,
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
        # Always hardcoded to shifts=[1,2] and cli_mode='full',
        # matching the original CliffordAlgebraBlock.
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
        # NOTE: use_bias is intentionally NOT forwarded here — the original
        # block left GGR at its use_bias=True default regardless of the block's
        # use_bias. Forwarding it would change behavior for existing
        # use_bias=False consumers (byte-identity violation). With use_gate=False
        # the gate_dense is unused, so its bias is irrelevant to the denoiser.
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

    def build(self, input_shape: Tuple) -> None:
        """Build all sub-layers in dependency order.

        :param input_shape: ``(B, H, W, D)``
        :type input_shape: Tuple
        """
        # B7: this block is isotropic in channels. Mismatched D produces a
        # cryptic broadcast error at the residual addition; reject early.
        if input_shape[-1] is not None and input_shape[-1] != self.channels:
            raise ValueError(
                f"{type(self).__name__} is isotropic: expected last dim == "
                f"channels={self.channels}, got input_shape[-1]={input_shape[-1]}. "
                f"Project the input before the block (e.g. with a 1x1 Conv) "
                f"or rebuild the block with channels={input_shape[-1]}."
            )
        spatial_shape = input_shape

        # Step 1: norm
        self.input_norm.build(spatial_shape)

        # Step 2a: detail linear
        self.linear_det.build(spatial_shape)
        stream_shape = self.linear_det.compute_output_shape(spatial_shape)

        # Step 2b: context -- two DWConvs, then single BN
        # DECISION plan_2026-07-04_d2ac2f68/D-001: causal builds the valid convs on
        # LEFT-PADDED shapes (W += K-1) so that, after the explicit left-pad in
        # call(), the "valid" conv preserves the sequence length.
        if self.causal:
            _pad = self.context_kernel_size - 1
            padded_shape = (*input_shape[:2],
                            (input_shape[2] or 0) + _pad, input_shape[3])
            self.dw_conv.build(padded_shape)
            # After valid conv on padded input, output W = original W
            dw1_out = input_shape
            padded_shape2 = (*dw1_out[:2],
                             (dw1_out[2] or 0) + _pad, dw1_out[3])
            self.dw_conv2.build(padded_shape2)
            self.ctx_norm.build(dw1_out)
        else:
            self.dw_conv.build(spatial_shape)
            dw1_out = self.dw_conv.compute_output_shape(spatial_shape)
            self.dw_conv2.build(dw1_out)
            dw2_out = self.dw_conv2.compute_output_shape(dw1_out)
            self.ctx_norm.build(dw2_out)

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

        :param inputs: Feature tensor ``(B, H, W, D)``.
        :type inputs: keras.KerasTensor
        :param training: Whether in training mode.
        :type training: Optional[bool]
        :return: Updated feature tensor ``(B, H, W, D)``.
        :rtype: keras.KerasTensor
        """
        x_prev = inputs

        # --- Step 1: Normalise ---
        x_norm = self.input_norm(x_prev)

        # --- Step 2: Dual-stream generation ---
        z_det = self.linear_det(x_norm)

        # Two stacked depthwise convolutions -> configurable norm -> activation
        # DECISION plan_2026-07-01_6dc255c1/D-001: context activation resolved
        # from self.activation (default "silu" == old hardcoded SiLU). Do NOT
        # re-hardcode SiLU — a homogeneous denoiser selects "leaky_relu".
        # DECISION plan_2026-07-04_d2ac2f68/D-001: causal inserts a left-only pad
        # before each valid DWConv (position i sees only positions <= i); the
        # subsequent norm + activation is SHARED with the vision path. Do NOT
        # re-hardcode SiLU on the causal branch — self.activation default "silu"
        # is byte-identical to the previous hardcoded keras.activations.silu.
        if self.causal:
            z_ctx = self._causal_pad(x_norm, self.context_kernel_size)
            z_ctx = self.dw_conv(z_ctx)
            z_ctx = self._causal_pad(z_ctx, self.context_kernel_size)
            z_ctx = self.dw_conv2(z_ctx)
        else:
            z_ctx = self.dw_conv(x_norm)
            z_ctx = self.dw_conv2(z_ctx)
        z_ctx = _resolve_activation(self.activation)(
            self.ctx_norm(z_ctx, training=training)
        )

        if self.ctx_mode == "diff":
            z_ctx = z_ctx - z_det  # discrete Laplacian approximation

        # --- Step 3: Local sparse geometric interaction ---
        g_feat = self.local_geo_prod(z_det, z_ctx)

        # --- Step 4: Optional global context branch ---
        # The global branch always uses differential context and is
        # independent of the local ctx_mode setting.
        if self.global_geo_prod is not None:
            # GAP keeps spatial dims as 1; let the subtraction broadcast
            # to (B,H,W,D) — materialising the broadcast via broadcast_to
            # would just allocate a redundant intermediate.
            # DECISION plan_2026-07-04_d2ac2f68/D-001: causal uses a cumulative mean
            # along the sequence axis (position i sees mean of 0..i) instead of the
            # full-image GAP, preserving autoregressive causality.
            if self.causal:
                c_glo = self._causal_cumulative_mean(x_norm)
            else:
                c_glo = keras.ops.mean(x_norm, axis=[1, 2], keepdims=True)
            # Hardcoded differential: C_glo = GAP(X_norm) - Z_det
            c_glo = c_glo - z_det
            g_glo = self.global_geo_prod(z_det, c_glo)
            g_feat = g_feat + g_glo

        # --- Step 5: GGR (transform only; residual add is external) ---
        h_mix = self.ggr(x_norm, g_feat, training=training)
        return h_mix

    # ------------------------------------------------------------------

    @staticmethod
    def _causal_pad(x: keras.KerasTensor, kernel_size: int = 3) -> keras.KerasTensor:
        """Apply left-only (causal) zero-padding along the W axis.

        For ``(B, H, W, D)`` with ``H = 1``, pads ``kernel_size - 1`` zeros
        on the left of the W dimension so that a ``"valid"`` convolution
        preserves the sequence length and each position only sees past/current.
        Used only on the ``causal=True`` path.
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
        """
        # x shape: (B, 1, seq_len, D)
        cumsum = keras.ops.cumsum(x, axis=2)  # (B, 1, seq_len, D)
        seq_len = keras.ops.shape(x)[2]
        # divisors: [1, 2, 3, ..., seq_len] reshaped to (1, 1, seq_len, 1)
        divisors = keras.ops.cast(
            keras.ops.arange(1, seq_len + 1), x.dtype
        )
        divisors = keras.ops.reshape(divisors, (1, 1, -1, 1))
        return cumsum / divisors

    # ------------------------------------------------------------------

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape.

        :param input_shape: Input shape ``(B, H, W, D)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Same as input shape.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    # ------------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        """Return serialisable configuration.

        :return: Dictionary with all constructor arguments.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "channels": self.channels,
                "shifts": self.shifts,
                "cli_mode": self.cli_mode,
                "ctx_mode": self.ctx_mode,
                "use_global_context": self.use_global_context,
                "causal": self.causal,
                "layer_scale_init": self.layer_scale_init,
                "use_bias": self.use_bias,
                "activation": self.activation,
                "dot_activation": self.dot_activation,
                "gate_activation": self.gate_activation,
                "feature_activation": self.feature_activation,
                "use_gate": self.use_gate,
                "context_kernel_size": self.context_kernel_size,
                "kernel_initializer": initializers.serialize(self.kernel_initializer),
                "bias_initializer": initializers.serialize(self.bias_initializer),
                "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
                "bias_regularizer": regularizers.serialize(self.bias_regularizer),
                "normalization_type": self.normalization_type,
                "normalization_kwargs": dict(self.normalization_kwargs),
                "input_normalization_type": self.input_normalization_type,
                "input_normalization_kwargs": dict(self.input_normalization_kwargs),
            }
        )
        return config


# ---------------------------------------------------------------------------
# CausalCliffordNetBlock — sequence-safe variant for autoregressive LMs
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
    auto-name ``causal_clifford_net_block`` are preserved so existing
    weights load by-name. All behavior lives in :class:`CliffordNetBlock`
    gated by ``causal=True`` (see DECISION plan_2026-07-04_d2ac2f68/D-001).

    :param kwargs: Forwarded to :class:`CliffordNetBlock`; ``causal`` is forced
        to ``True`` and ``normalization_type`` defaults to
        ``"zero_centered_rms_norm"``.
    """

    def __init__(self, **kwargs: Any) -> None:
        # Preserve the causal-specific context-norm default while letting the
        # caller override it explicitly.
        kwargs.setdefault("normalization_type", "zero_centered_rms_norm")
        # Force causal=True even if a from_config dict carries a `causal` key,
        # so this subclass can never be built non-causal (and never raises a
        # duplicate-kwarg TypeError). See DECISION plan_2026-07-04_d2ac2f68/D-001.
        kwargs["causal"] = True
        super().__init__(**kwargs)


# ---------------------------------------------------------------------------
# CliffordNetBlockDSv2 — design-space sibling for downsampling experiments
# ---------------------------------------------------------------------------

# Type aliases for v2 (additive — do not modify the v1 aliases above).
SkipPoolV2 = Literal[
    "avg", "max", "blur", "gaussian_dw", "pixel_unshuffle", "resnetd"
]
CtxModeV2 = Literal["diff", "abs", "pyramid_diff"]
CtxNormType = Literal["bn", "gn", "ln", "none"]


def _make_pool_v2(
    kind: str,
    channels: int,
    strides: int,
    name: str,
) -> keras.layers.Layer:
    """Build a single stride-``s`` downsampling layer that maps
    ``(B, H, W, C) -> (B, H/s, W/s, channels)``.

    Used by ``CliffordNetBlockDSv2`` for both the stream and skip paths.
    All paths target ``channels`` output channels so downstream layers
    (SRGP, GGR, residual sum) see a uniform channel dim. Internal
    channel expansion (axis E) happens after the residual sum via a
    separate 1x1 projection on the block's output.
    """
    from ..blur_pool import BlurPool2D
    from ..pixel_unshuffle import PixelUnshuffle2D

    # All pool kinds collapse to Identity at strides=1 by design. This
    # lets a hierarchical model use a single ``stream_pool="blur"``
    # configuration across every stage; only the strided stages actually
    # apply the kind-specific transform, and the others are pass-through.
    # Validating the kind name happens upstream in __init__.
    if strides == 1:
        return keras.layers.Identity(name=name)
    if kind == "avg":
        return keras.layers.AveragePooling2D(
            pool_size=strides, strides=strides, padding="same", name=name
        )
    if kind == "max":
        return keras.layers.MaxPooling2D(
            pool_size=strides, strides=strides, padding="same", name=name
        )
    if kind == "blur":
        return BlurPool2D(strides=strides, padding="same", name=name)
    if kind == "gaussian_dw":
        from ...utils.tensors import depthwise_gaussian_kernel
        k = max(5, 2 * strides + 1)
        gauss_np = depthwise_gaussian_kernel(
            channels=channels,
            kernel_size=(k, k),
            nsig=(2.0, 2.0),
            dtype="float32",
        )
        return keras.layers.DepthwiseConv2D(
            kernel_size=k,
            strides=strides,
            padding="same",
            use_bias=False,
            depthwise_initializer=keras.initializers.Constant(gauss_np),
            name=name,
        )
    if kind == "pixel_unshuffle":
        return PixelUnshuffle2D(
            scale=strides, out_channels=channels, name=name
        )
    if kind == "resnetd":
        return keras.Sequential(
            [
                keras.layers.AveragePooling2D(
                    pool_size=strides, strides=strides, padding="same"
                ),
                keras.layers.Conv2D(
                    channels, kernel_size=1, padding="same", use_bias=True
                ),
            ],
            name=name,
        )
    raise ValueError(f"Unknown pool kind: {kind!r}")


def _make_ctx_norm(
    norm_type: str, channels: int, name: str
) -> Optional[keras.layers.Layer]:
    """Build the context-stream normalisation layer."""
    if norm_type == "bn":
        return keras.layers.BatchNormalization(name=name)
    if norm_type == "gn":
        groups = max(1, min(32, channels // 4))
        while channels % groups != 0:
            groups -= 1
        return keras.layers.GroupNormalization(groups=groups, name=name)
    if norm_type == "ln":
        return keras.layers.LayerNormalization(epsilon=1e-6, name=name)
    if norm_type == "none":
        return None
    raise ValueError(f"Unknown ctx_norm_type: {norm_type!r}")

# ---------------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class CliffordNetBlockDSv2(keras.layers.Layer):
    """Design-space downsampling block for the experiments described in
    ``analyses/analysis_2026-04-30_41b5e415/summary.md``.

    Relative to a plain single-7x7-DW context block:

    1. **Decoupled stream/skip pool kinds (axes A and B).** ``stream_pool``
       and ``skip_pool`` are independent and accept ``avg | max | blur |
       gaussian_dw | pixel_unshuffle | resnetd``.
    2. **Internal channel expansion (axis E).** Optional ``out_channels``
       triggers a 1x1 projection at the END of the block (after the
       residual sum), so the block output has ``out_channels`` channels.
       SRGP / GGR still operate at ``channels``.
    3. **Configurable context-stream norm (axis G).** ``ctx_norm_type``
       selects ``bn | gn | ln | none`` — the input LayerNorm is unchanged.
    4. **Pyramid-diff context mode (axis D).** ``ctx_mode="pyramid_diff"``
       at ``strides>1`` applies a Laplacian-pyramid level subtraction
       (``z_ctx -= upsample(avg_pool(z_ctx, s))``). At ``strides=1`` it
       falls back to plain ``"diff"``.

    Defaults reflect the empirical winner of the 11-variant downsampling
    sweep documented in ``src/train/cliffordnet/DOWNSAMPLING.md``: V1
    (``stream_pool="blur"``, ``skip_pool="blur"``). Pass
    ``stream_pool="avg", skip_pool="avg"`` explicitly to reproduce the
    legacy single-7x7-DW baseline (the V0 baseline of the sweep).
    """

    def __init__(
        self,
        channels: int,
        shifts: List[int],
        cli_mode: CliMode = "full",
        ctx_mode: CtxModeV2 = "diff",
        use_global_context: bool = False,
        kernel_size: int = 7,
        strides: int = 1,
        stream_pool: SkipPoolV2 = "blur",
        skip_pool: SkipPoolV2 = "blur",
        out_channels: Optional[int] = None,
        ctx_norm_type: CtxNormType = "bn",
        ctx_activation: Optional[str] = "silu",
        layer_scale_init: float = 1e-5,
        use_bias: bool = True,
        kernel_initializer: Any = "glorot_uniform",
        bias_initializer: Any = "zeros",
        kernel_regularizer: Optional[Any] = None,
        bias_regularizer: Optional[Any] = None,
        causal: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        # DECISION plan_2026-07-04_5e774fa6/D-001: the causal DSv2 variant is a
        # `causal` FLAG on this class, not a separate implementation
        # (CausalCliffordNetBlockDSv2 is a thin subclass that forces
        # `causal=True`). The one non-clean-boolean spot is the context conv:
        # it is a DIFFERENT Keras layer CLASS per branch (DepthwiseConv2D vs
        # Conv2D(groups=channels)), resolved ONCE here at construction — NEVER
        # per-call (that would break Keras weight naming/serialization). Do NOT
        # re-duplicate the causal body into a second class, and do NOT collapse
        # the conv-class choice into a runtime `if` inside call(). See
        # decisions.md D-001/D-002.
        if causal:
            # 'pyramid_diff' is excluded for the causal variant: the bilinear
            # UpSampling2D in the DSv2 pyramid path mixes neighbouring
            # positions, leaking future info along the W (sequence) axis.
            if ctx_mode not in ("diff", "abs"):
                raise ValueError(
                    f"ctx_mode must be 'diff' or 'abs' for the causal variant "
                    f"(pyramid_diff is non-causal), got {ctx_mode!r}"
                )
        else:
            if ctx_mode not in ("diff", "abs", "pyramid_diff"):
                raise ValueError(
                    f"ctx_mode must be 'diff'|'abs'|'pyramid_diff', got "
                    f"{ctx_mode!r}"
                )
        if not isinstance(kernel_size, int) or kernel_size <= 0:
            raise ValueError(
                f"kernel_size must be a positive int, got {kernel_size!r}"
            )
        if not isinstance(strides, int) or strides < 1:
            raise ValueError(f"strides must be int>=1, got {strides!r}")
        if causal:
            # avg/max only: blur/gaussian_dw use symmetric kernels that cross
            # the temporal boundary on the right edge (future leak);
            # pixel_unshuffle changes channel count; resnetd bundles an
            # unneeded 1x1 conv. See _make_causal_pool.
            valid_pools = {"avg", "max"}
            if stream_pool not in valid_pools:
                raise ValueError(
                    f"stream_pool must be one of {sorted(valid_pools)} "
                    f"(causal variant), got {stream_pool!r}"
                )
            if skip_pool not in valid_pools:
                raise ValueError(
                    f"skip_pool must be one of {sorted(valid_pools)} "
                    f"(causal variant), got {skip_pool!r}"
                )
        else:
            valid_pools = {
                "avg", "max", "blur", "gaussian_dw", "pixel_unshuffle",
                "resnetd",
            }
            if stream_pool not in valid_pools:
                raise ValueError(
                    f"stream_pool must be one of {sorted(valid_pools)}, "
                    f"got {stream_pool!r}"
                )
            if skip_pool not in valid_pools:
                raise ValueError(
                    f"skip_pool must be one of {sorted(valid_pools)}, "
                    f"got {skip_pool!r}"
                )
        if out_channels is not None and out_channels <= 0:
            raise ValueError(
                f"out_channels must be positive or None, got {out_channels!r}"
            )
        if ctx_norm_type not in ("bn", "gn", "ln", "none"):
            raise ValueError(
                f"ctx_norm_type must be 'bn'|'gn'|'ln'|'none', got "
                f"{ctx_norm_type!r}"
            )
        # See D-002: global branch needs channels >= 2.
        if use_global_context and channels < 2:
            raise ValueError(
                f"use_global_context=True requires channels >= 2 "
                f"(global branch uses shifts=[1, 2]); got channels={channels}"
            )

        self.channels = channels
        self.shifts = list(shifts)
        self.cli_mode = cli_mode
        self.ctx_mode = ctx_mode
        self.use_global_context = use_global_context
        self.kernel_size = kernel_size
        self.strides = strides
        self.stream_pool_kind = stream_pool
        self.skip_pool_kind = skip_pool
        self.out_channels = out_channels
        self.ctx_norm_type = ctx_norm_type
        self.ctx_activation = ctx_activation
        self.layer_scale_init = layer_scale_init
        self.use_bias = use_bias
        self.causal = causal
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.input_norm = keras.layers.LayerNormalization(
            epsilon=1e-6, name="input_norm"
        )
        self.linear_det = keras.layers.Dense(
            channels,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name="linear_det",
        )
        # DECISION plan_2026-07-04_5e774fa6/D-001: context conv CLASS resolved
        # once here. TF/CUDA DepthwiseConv2D requires equal row/col strides;
        # the causal path needs asymmetric strides=(1, s), so it uses
        # Conv2D(groups=channels) (algebraically depthwise) + a manual left-pad
        # (see call/_causal_pad_w). Do NOT swap this to a per-call branch.
        if causal:
            self.dw_conv = keras.layers.Conv2D(
                filters=channels,
                kernel_size=(1, kernel_size),
                strides=(1, strides),
                padding="valid",
                groups=channels,
                use_bias=(ctx_norm_type == "none"),
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name="dw_conv",
            )
        else:
            self.dw_conv = keras.layers.DepthwiseConv2D(
                kernel_size=kernel_size,
                strides=strides,
                padding="same",
                use_bias=(ctx_norm_type == "none"),
                name="dw_conv",
            )
        self.ctx_norm = _make_ctx_norm(
            ctx_norm_type, channels, name="ctx_norm"
        )

        # Pool builder differs per branch: _make_causal_pool has NO `channels`
        # arg (avg/max only, pools along W) vs the 6-kind _make_pool_v2.
        if causal:
            self.stream_pool = _make_causal_pool(
                stream_pool, strides, name="stream_pool"
            )
            self.skip_pool = _make_causal_pool(
                skip_pool, strides, name="skip_pool"
            )
        else:
            self.stream_pool = _make_pool_v2(
                stream_pool, channels, strides, name="stream_pool"
            )
            self.skip_pool = _make_pool_v2(
                skip_pool, channels, strides, name="skip_pool"
            )

        # Pyramid-diff path is non-causal only (bilinear upsample leaks future).
        self._pyr_pool: Optional[keras.layers.Layer] = None
        self._pyr_up: Optional[keras.layers.Layer] = None
        if not causal and ctx_mode == "pyramid_diff" and strides > 1:
            self._pyr_pool = keras.layers.AveragePooling2D(
                pool_size=strides, strides=strides, padding="same",
                name="pyr_pool",
            )
            self._pyr_up = keras.layers.UpSampling2D(
                size=strides, interpolation="bilinear", name="pyr_up",
            )

        self.local_geo_prod = SparseRollingGeometricProduct(
            channels=channels,
            shifts=shifts,
            cli_mode=cli_mode,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name="local_geo_prod",
        )

        if use_global_context:
            self.global_geo_prod = SparseRollingGeometricProduct(
                channels=channels,
                shifts=_GLOBAL_SHIFTS,
                cli_mode=_GLOBAL_CLI_MODE,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name="global_geo_prod",
            )
        else:
            self.global_geo_prod = None

        self.ggr = GatedGeometricResidual(
            channels=channels,
            layer_scale_init=layer_scale_init,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name="ggr",
        )

        self.out_proj: Optional[keras.layers.Conv2D] = None
        if out_channels is not None and out_channels != channels:
            self.out_proj = keras.layers.Conv2D(
                filters=out_channels,
                kernel_size=1,
                padding="same",
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name="out_proj",
            )

    @staticmethod
    def _ceildiv(a: Optional[int], b: int) -> Optional[int]:
        return None if a is None else -(-a // b)

    @staticmethod
    def _causal_pad_w(
        x: keras.KerasTensor, kernel_size: int
    ) -> keras.KerasTensor:
        """Left-only zero-pad along W by ``kernel_size - 1``.

        For ``(B, H, W, D)`` with ``H = 1``, pads ``kernel_size - 1`` zeros
        on the LEFT of the W dimension so that a ``valid`` conv with
        ``kernel_size=(1, kernel_size)`` preserves causality (each output
        position only sees past/current input positions). Used only by the
        ``causal=True`` context path.
        """
        pad_w = kernel_size - 1
        return keras.ops.pad(x, [[0, 0], [0, 0], [pad_w, 0], [0, 0]])

    @staticmethod
    def _causal_cumulative_mean_w(
        x: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """Causal cumulative mean along W (axis=2).

        For input ``(B, 1, W, C)`` returns the same shape where pooled
        position ``j`` is the average of pooled positions ``0..j``. Used
        by the causal global-context branch to preserve causality (vs the
        non-causal full-spatial mean which would mix future into past).
        """
        cumsum = keras.ops.cumsum(x, axis=2)
        seq_len = keras.ops.shape(x)[2]
        divisors = keras.ops.cast(
            keras.ops.arange(1, seq_len + 1), x.dtype
        )
        divisors = keras.ops.reshape(divisors, (1, 1, -1, 1))
        return cumsum / divisors

    def build(self, input_shape: Tuple) -> None:
        """Explicitly build every sub-layer in dependency order.

        Required for clean save/load: Keras serialisation records the
        built-state of each sub-layer at the moment of saving, and
        complains on load if any layer was implicitly built only via
        ``call``-time tracing.

        When ``causal=True`` this additionally validates the H==1 layout and
        builds ``dw_conv`` on the LEFT-PADDED shape (W' = W + kernel_size - 1)
        because the causal conv uses ``padding="valid"`` on a pre-padded input.
        """
        # B7: isotropic core (the optional out_proj at the end is the only
        # place channels may change).
        if input_shape[-1] is not None and input_shape[-1] != self.channels:
            raise ValueError(
                f"{type(self).__name__} expects input last dim == "
                f"channels={self.channels}, got input_shape[-1]={input_shape[-1]}."
            )
        # Causal path operates on (B, 1, seq_len, D): H must be 1.
        if self.causal and input_shape[1] is not None and input_shape[1] != 1:
            raise ValueError(
                f"{type(self).__name__} expects H == 1 (input shape "
                f"(B, 1, seq_len, D)), got input_shape[1]={input_shape[1]}."
            )
        b, h, w, _ = input_shape

        self.input_norm.build(input_shape)

        # Pool layers consume full-resolution input.
        self.stream_pool.build(input_shape)
        self.skip_pool.build(input_shape)

        # Stream-side shape after stream pool.
        pooled_h = self._ceildiv(h, self.strides)
        pooled_w = self._ceildiv(w, self.strides)
        if self.causal:
            # H stays at 1; only W is pooled.
            stream_shape = (b, h, pooled_w, self.channels)
        else:
            stream_shape = (b, pooled_h, pooled_w, self.channels)

        self.linear_det.build(stream_shape)

        if self.causal:
            # DW conv consumes the LEFT-PADDED full-resolution input.
            # padded_w = w + kernel_size - 1 (None-safe).
            padded_w = None if w is None else (w + self.kernel_size - 1)
            self.dw_conv.build((b, h, padded_w, self.channels))
        else:
            # DW conv consumes full-resolution input; its output spatial
            # dims equal the pooled shape (same ceil semantics for
            # stride+"same" padding).
            self.dw_conv.build(input_shape)
        if self.ctx_norm is not None:
            self.ctx_norm.build(stream_shape)

        # Pyramid-diff helpers (only present when ctx_mode='pyramid_diff'
        # AND strides>1).
        if self._pyr_pool is not None:
            self._pyr_pool.build(stream_shape)
        if self._pyr_up is not None:
            # AveragePooling2D output:
            pyr_lo_shape = (
                b,
                self._ceildiv(pooled_h, self.strides),
                self._ceildiv(pooled_w, self.strides),
                self.channels,
            )
            self._pyr_up.build(pyr_lo_shape)

        self.local_geo_prod.build(stream_shape)
        if self.global_geo_prod is not None:
            self.global_geo_prod.build(stream_shape)
        self.ggr.build(stream_shape)

        if self.out_proj is not None:
            self.out_proj.build(stream_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        x_prev = inputs

        x_norm = self.input_norm(x_prev)
        x_norm_p = self.stream_pool(x_norm)
        z_det = self.linear_det(x_norm_p)

        if self.causal:
            # Left-pad along W, then valid conv with stride (strict causality).
            z_ctx = self.dw_conv(
                self._causal_pad_w(x_norm, self.kernel_size)
            )
        else:
            z_ctx = self.dw_conv(x_norm)
        if self.ctx_norm is not None:
            z_ctx = self.ctx_norm(z_ctx, training=training)
        if self.ctx_activation is not None:
            z_ctx = keras.activations.get(self.ctx_activation)(z_ctx)

        if self.ctx_mode == "diff":
            z_ctx = z_ctx - z_det
        elif self.ctx_mode == "abs":
            pass
        elif self.ctx_mode == "pyramid_diff":
            if self._pyr_pool is not None and self._pyr_up is not None:
                # pyramid_diff at strides>1.
                # AveragePooling2D(padding="same", pool_size=s) on a tensor
                # of spatial shape (H/s, W/s) produces ceil((H/s)/s) rows;
                # UpSampling2D(size=s) then expands those by an exact factor
                # of s. When (H/s) is not divisible by s, the upsample is
                # strictly larger than z_ctx and the subtraction broadcasts
                # incorrectly (or errors). Crop to z_ctx's spatial extent
                # so the Laplacian-pyramid level subtraction is well-defined
                # for arbitrary input dims.
                z_lo = self._pyr_pool(z_ctx)
                z_lo_up = self._pyr_up(z_lo)
                target_h = keras.ops.shape(z_ctx)[1]
                target_w = keras.ops.shape(z_ctx)[2]
                z_lo_up = z_lo_up[:, :target_h, :target_w, :]
                z_ctx = z_ctx - z_lo_up
            else:
                z_ctx = z_ctx - z_det

        g_feat = self.local_geo_prod(z_det, z_ctx)

        if self.global_geo_prod is not None:
            if self.causal:
                # Causal global branch: cumulative mean over the pooled W axis
                # (each position sees past+current pooled positions only).
                c_glo = self._causal_cumulative_mean_w(x_norm_p)
            else:
                # P2 mirror: drop redundant broadcast_to.
                c_glo = keras.ops.mean(x_norm_p, axis=[1, 2], keepdims=True)
            c_glo = c_glo - z_det
            g_glo = self.global_geo_prod(z_det, c_glo)
            g_feat = g_feat + g_glo

        # DECISION plan_2026-07-03_eb53492e/D-002: returns the pooled transform
        # only (h_mix at `channels`). `skip_pool` and `out_proj` are kept as
        # public block-owned sub-layers so the model orchestrates the exact
        # POST-SUM projection: out_proj(skip_pool(x) + SD(block(x))). out_proj
        # has use_bias=True (non-distributive), so it must run on the SUM, not
        # on h_mix alone. Do NOT re-inline the skip pool / add / projection
        # here; see decisions.md D-002.
        h_mix = self.ggr(x_norm_p, g_feat, training=training)
        return h_mix

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        # Transform-only return: pooled spatial (per `strides`) at `channels`
        # width. `out_proj` (channels -> out_channels) now runs at model level,
        # so out_channels is NOT reflected here (D-002).
        b, h, w, _ = input_shape
        new_h = None if h is None else self._ceildiv(h, self.strides)
        new_w = None if w is None else self._ceildiv(w, self.strides)
        return (b, new_h, new_w, self.channels)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "channels": self.channels,
                "shifts": self.shifts,
                "cli_mode": self.cli_mode,
                "ctx_mode": self.ctx_mode,
                "use_global_context": self.use_global_context,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "stream_pool": self.stream_pool_kind,
                "skip_pool": self.skip_pool_kind,
                "out_channels": self.out_channels,
                "ctx_norm_type": self.ctx_norm_type,
                "ctx_activation": self.ctx_activation,
                "layer_scale_init": self.layer_scale_init,
                "use_bias": self.use_bias,
                "causal": self.causal,
                "kernel_initializer": initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": initializers.serialize(
                    self.bias_initializer
                ),
                "kernel_regularizer": regularizers.serialize(
                    self.kernel_regularizer
                ),
                "bias_regularizer": regularizers.serialize(
                    self.bias_regularizer
                ),
            }
        )
        return config


# ---------------------------------------------------------------------------
# CausalCliffordNetBlockDSv2 — causal sibling of CliffordNetBlockDSv2
# ---------------------------------------------------------------------------


# Type aliases scoped to the causal DSv2 surface (narrower than DSv2's).
CausalCtxModeV2 = Literal["diff", "abs"]
CausalSkipPoolV2 = Literal["avg", "max"]


def _make_causal_pool(
    kind: str,
    strides: int,
    name: str,
) -> keras.layers.Layer:
    """Build a causal-safe stride-``s`` downsampling pool along W.

    Restricted to ``"avg"`` and ``"max"``: with ``padding="same"`` and a
    ``(1, strides)`` kernel/stride along W, both pad with zero / ``-inf``
    (resp.) on the right edge. Padding values carry no information about
    future input, so position ``i`` in the output only depends on real
    input positions ``<= i*strides + (strides-1)``. Other DSv2 pool kinds
    (``blur``, ``gaussian_dw``, ``pixel_unshuffle``, ``resnetd``) are
    excluded — see DECISION D-001 in ``CausalCliffordNetBlockDSv2``.

    At ``strides == 1`` the pool collapses to ``Identity`` (matches the
    DSv2 ``_make_pool_v2`` convention).
    """
    if strides == 1:
        return keras.layers.Identity(name=name)
    if kind == "avg":
        return keras.layers.AveragePooling2D(
            pool_size=(1, strides),
            strides=(1, strides),
            padding="same",
            name=name,
        )
    if kind == "max":
        return keras.layers.MaxPooling2D(
            pool_size=(1, strides),
            strides=(1, strides),
            padding="same",
            name=name,
        )
    raise ValueError(
        f"Unknown causal pool kind: {kind!r} (expected 'avg' or 'max')."
    )

# ---------------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class CausalCliffordNetBlockDSv2(CliffordNetBlockDSv2):
    """Causal (autoregressive) sibling of :class:`CliffordNetBlockDSv2`.

    Thin subclass that forces ``causal=True`` on the unified
    :class:`CliffordNetBlockDSv2`. See DECISION plan_2026-07-04_5e774fa6/D-001
    in that class: the causal variant is a construction-time flag, not a
    separate implementation. The causal branch uses a
    ``Conv2D(groups=channels)`` context conv with an explicit left-pad along W
    (strictly causal), pool kinds restricted to ``{"avg", "max"}`` via
    ``_make_causal_pool``, ``ctx_mode`` restricted to ``{"diff", "abs"}`` (no
    ``pyramid_diff``), and a causal cumulative-mean global-context statistic.
    Expects 4-D input ``(B, 1, seq_len, D)`` (H == 1).

    The registered class name + ``@register_keras_serializable`` are preserved
    (Keras registration is per concrete class, not inherited) so the trained
    NLP-UNet checkpoint keeps loading by name.
    """

    def __init__(self, **kwargs: Any) -> None:
        # Pool defaults differ from the base (base default "blur"); the causal
        # variant defaults to "avg" for both. An explicit caller value wins.
        kwargs.setdefault("stream_pool", "avg")
        kwargs.setdefault("skip_pool", "avg")
        # Force the flag (overwrite, not setdefault) so a from_config dict
        # carrying causal=... still resolves to True and a foot-gun
        # CausalCliffordNetBlockDSv2(causal=False) is impossible.
        kwargs["causal"] = True
        super().__init__(**kwargs)

# ---------------------------------------------------------------------------
