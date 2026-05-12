"""
CliffordNet block and constituent primitives.

Implements the geometric-algebra-based vision block from:
    "CliffordNet: All You Need is Geometric Algebra"  arXiv:2601.06793v2

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
   (effective 7×7 RF) with a single ``BatchNormalization`` after both,
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

from ..stochastic_depth import StochasticDepth
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
    is D_s[c] = SiLU(Z_det[c] * Z_ctx[(c+s) % D]) and the wedge component is
    W_s[c] = Z_det[c] * Z_ctx[(c+s)%D] - Z_ctx[c] * Z_det[(c+s)%D].

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
        # DECISION D-001: reject shifts <= 0. s=0 makes the wedge term
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
                # Scalar: gated inner product
                dot = keras.activations.silu(z_det * z_ctx_s)
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
    LayerScale scalar initialised near zero. DropPath is applied to the
    combined term before residual addition when ``drop_path_rate > 0``.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────┐  ┌──────────────────┐
        │ H_norm [B,H,W,D] │  │ G_feat [B,H,W,D] │
        └────────┬─────────┘  └────────┬─────────┘
                 │                     │
                 ├─────────┬───────────┤
                 │         ▼           │
                 │  ┌─────────────┐    │
                 │  │ Concat→Gate │    │
                 │  │ α=sigmoid() │    │
                 │  └──────┬──────┘    │
                 │         ▼           │
                 ▼    ┌──────────┐     ▼
        ┌────────┐    │ α·G_feat │   (add)
        │SiLU(H) │    └────┬─────┘     │
        └───┬────┘         │           │
            └──────┬───────┘           │
                   ▼                   │
            ┌─────────────┐            │
            │ γ · (sum)   │            │
            └──────┬──────┘            │
                   ▼                   │
            ┌─────────────┐            │
            │  DropPath   │            │
            └──────┬──────┘            │
                   ▼                   │
            Output (residual term)     │
                   [B, H, W, D]        │
        ───────────────────────────────┘

    :param channels: Feature dimensionality D.
    :type channels: int
    :param layer_scale_init: Initial LayerScale gamma. Defaults to 1e-5.
    :type layer_scale_init: float
    :param drop_path_rate: Stochastic-depth probability. Defaults to 0.0.
    :type drop_path_rate: float
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
        drop_path_rate: float = 0.0,
        kernel_initializer: Any = "glorot_uniform",
        bias_initializer: Any = "zeros",
        kernel_regularizer: Optional[Any] = None,
        bias_regularizer: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if not (0.0 <= drop_path_rate < 1.0):
            raise ValueError(
                f"drop_path_rate must be in [0, 1), got {drop_path_rate}"
            )

        self.channels = channels
        self.layer_scale_init = layer_scale_init
        self.drop_path_rate = drop_path_rate
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        # Learned gate: Dense(2C -> C) followed by sigmoid
        self.gate_dense = keras.layers.Dense(
            channels,
            use_bias=True,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name="gate_dense",
        )

        self.drop_path = (
            StochasticDepth(drop_path_rate=drop_path_rate, name="drop_path")
            if drop_path_rate > 0.0
            else None
        )

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
        :param training: Whether in training mode (affects DropPath).
        :type training: Optional[bool]
        :return: Scaled residual term ``(B, H, W, D)``; caller adds to H_prev.
        :rtype: keras.KerasTensor
        """
        gate_input = keras.ops.concatenate([h_norm, g_feat], axis=-1)
        alpha = keras.activations.sigmoid(self.gate_dense(gate_input))

        h_mix = keras.activations.silu(h_norm) + alpha * g_feat
        h_mix = h_mix * self.gamma

        if self.drop_path is not None:
            h_mix = self.drop_path(h_mix, training=training)

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
                "drop_path_rate": self.drop_path_rate,
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
    a sparse rolling geometric product, are combined through a Gated Geometric
    Residual (GGR) update, and added back as a residual. An optional global
    branch uses GAP-based context with hardcoded shifts=[1,2] and cli_mode='full'.

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
        │ X_out = X_prev + H_mix         │
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
    :param drop_path_rate: DropPath probability. Defaults to 0.0.
    :type drop_path_rate: float
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
        layer_scale_init: float = 1e-5,
        drop_path_rate: float = 0.0,
        use_bias: bool = True,
        kernel_initializer: Any = "glorot_uniform",
        bias_initializer: Any = "zeros",
        kernel_regularizer: Optional[Any] = None,
        bias_regularizer: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if ctx_mode not in ("diff", "abs"):
            raise ValueError(f"ctx_mode must be 'diff' or 'abs', got {ctx_mode!r}")
        # DECISION D-002: the global branch hardcodes shifts=[1, 2]; with
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
        self.layer_scale_init = layer_scale_init
        self.drop_path_rate = drop_path_rate
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        _dense_kwargs: Dict[str, Any] = dict(
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )

        # --- Step 1: Input norm ---
        self.input_norm = keras.layers.LayerNormalization(
            epsilon=1e-6, name="input_norm"
        )

        # --- Step 2a: Detail stream (1×1 pointwise) ---
        self.linear_det = keras.layers.Dense(
            channels, name="linear_det", **_dense_kwargs
        )

        # --- Step 2b: Context stream ---
        # Two stacked 3×3 depthwise convolutions (effective 7×7 RF),
        # one BatchNormalization after both, then SiLU in call().
        self.dw_conv = keras.layers.DepthwiseConv2D(
            kernel_size=3,
            padding="same",
            use_bias=False,
            name="dw_conv",
        )
        self.dw_conv2 = keras.layers.DepthwiseConv2D(
            kernel_size=3,
            padding="same",
            use_bias=False,
            name="dw_conv2",
        )
        self.ctx_bn = keras.layers.BatchNormalization(name="ctx_bn")

        # --- Step 3: Local sparse rolling product ---
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

        # --- Optional global context branch (gFFN-G) ---
        # Always hardcoded to shifts=[1,2] and cli_mode='full',
        # matching the original CliffordAlgebraBlock.
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

        # --- Step 4 / 5: GGR ---
        self.ggr = GatedGeometricResidual(
            channels=channels,
            layer_scale_init=layer_scale_init,
            drop_path_rate=drop_path_rate,
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
                f"CliffordNetBlock is isotropic: expected last dim == "
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
        self.dw_conv.build(spatial_shape)
        dw1_out = self.dw_conv.compute_output_shape(spatial_shape)
        self.dw_conv2.build(dw1_out)
        dw2_out = self.dw_conv2.compute_output_shape(dw1_out)
        self.ctx_bn.build(dw2_out)

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

        # Two stacked depthwise convolutions -> single BN -> SiLU
        z_ctx = self.dw_conv(x_norm)
        z_ctx = self.dw_conv2(z_ctx)
        z_ctx = keras.activations.silu(self.ctx_bn(z_ctx, training=training))

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
            c_glo = keras.ops.mean(x_norm, axis=[1, 2], keepdims=True)
            # Hardcoded differential: C_glo = GAP(X_norm) - Z_det
            c_glo = c_glo - z_det
            g_glo = self.global_geo_prod(z_det, c_glo)
            g_feat = g_feat + g_glo

        # --- Step 5 & 6: GGR + residual ---
        h_mix = self.ggr(x_norm, g_feat, training=training)
        return x_prev + h_mix

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
                "layer_scale_init": self.layer_scale_init,
                "drop_path_rate": self.drop_path_rate,
                "use_bias": self.use_bias,
                "kernel_initializer": initializers.serialize(self.kernel_initializer),
                "bias_initializer": initializers.serialize(self.bias_initializer),
                "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
                "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            }
        )
        return config


# ---------------------------------------------------------------------------
# CausalCliffordNetBlock — sequence-safe variant for autoregressive LMs
# ---------------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class CausalCliffordNetBlock(keras.layers.Layer):
    """CliffordNetBlock variant with causal (left-only) padded convolutions.

    Designed for autoregressive language modeling where information must not
    flow from future to past positions.  The only change from
    :class:`CliffordNetBlock` is that the two ``DepthwiseConv2D`` layers in
    the context stream use ``padding="valid"`` with explicit left-only
    zero-padding so that position *i* can only see positions ``<= i``.

    Expects 4-D input ``(B, 1, seq_len, D)`` (sequence reshaped for 2-D
    convolutions with ``H = 1``).

    All other components (normalisation, detail stream, geometric products,
    GGR, global context branch) are identical to the vision block.

    :param channels: Feature dimensionality D.
    :param shifts: Channel-shift offsets for the sparse rolling product.
    :param cli_mode: Algebraic components (``"inner"``, ``"wedge"``, ``"full"``).
    :param ctx_mode: Context mode (``"diff"`` or ``"abs"``).
    :param use_global_context: Add global-average-pool context branch.
    :param layer_scale_init: Initial LayerScale value.
    :param drop_path_rate: DropPath probability.
    :param use_bias: Whether Dense layers use bias.
    :param kernel_initializer: Kernel initializer.
    :param bias_initializer: Bias initializer.
    :param kernel_regularizer: Kernel regularizer.
    :param bias_regularizer: Bias regularizer.
    """

    def __init__(
        self,
        channels: int,
        shifts: List[int],
        cli_mode: CliMode = "full",
        ctx_mode: CtxMode = "diff",
        use_global_context: bool = False,
        layer_scale_init: float = 1e-5,
        drop_path_rate: float = 0.0,
        use_bias: bool = True,
        kernel_initializer: Any = "glorot_uniform",
        bias_initializer: Any = "zeros",
        kernel_regularizer: Optional[Any] = None,
        bias_regularizer: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if ctx_mode not in ("diff", "abs"):
            raise ValueError(f"ctx_mode must be 'diff' or 'abs', got {ctx_mode!r}")
        # See D-002 (CliffordNetBlock): global branch needs channels >= 2.
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
        self.layer_scale_init = layer_scale_init
        self.drop_path_rate = drop_path_rate
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        _dense_kwargs: Dict[str, Any] = dict(
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )

        # --- Step 1: Input norm ---
        self.input_norm = keras.layers.LayerNormalization(
            epsilon=1e-6, name="input_norm"
        )

        # --- Step 2a: Detail stream (1×1 pointwise) ---
        self.linear_det = keras.layers.Dense(
            channels, name="linear_det", **_dense_kwargs
        )

        # --- Step 2b: Causal context stream ---
        # DepthwiseConv2D with kernel (1, 3) and padding="valid"; explicit
        # left-only padding is applied along W in call() so position i sees
        # only positions <= i.  H dimension uses kernel=1 (no spatial mixing
        # along H, which is 1 for sequences reshaped to (B, 1, seq_len, D)).
        self._ctx_kernel_w = 3
        self.dw_conv = keras.layers.DepthwiseConv2D(
            kernel_size=(1, self._ctx_kernel_w),
            padding="valid",
            use_bias=False,
            name="dw_conv",
        )
        self.dw_conv2 = keras.layers.DepthwiseConv2D(
            kernel_size=(1, self._ctx_kernel_w),
            padding="valid",
            use_bias=False,
            name="dw_conv2",
        )
        self.ctx_bn = keras.layers.BatchNormalization(name="ctx_bn")

        # --- Step 3: Local sparse rolling product ---
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

        # --- Optional global context branch ---
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

        # --- GGR ---
        self.ggr = GatedGeometricResidual(
            channels=channels,
            layer_scale_init=layer_scale_init,
            drop_path_rate=drop_path_rate,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name="ggr",
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _causal_pad(x: keras.KerasTensor, kernel_size: int = 3) -> keras.KerasTensor:
        """Apply left-only (causal) zero-padding along the W axis.

        For ``(B, H, W, D)`` with ``H = 1``, pads ``kernel_size - 1`` zeros
        on the left of the W dimension so that a ``"valid"`` convolution
        preserves the sequence length and each position only sees past/current.
        """
        pad_w = kernel_size - 1
        # pad format: [[B_lo, B_hi], [H_lo, H_hi], [W_lo, W_hi], [D_lo, D_hi]]
        return keras.ops.pad(x, [[0, 0], [0, 0], [pad_w, 0], [0, 0]])

    # ------------------------------------------------------------------

    def build(self, input_shape: Tuple) -> None:
        """Build all sub-layers."""
        spatial_shape = input_shape

        self.input_norm.build(spatial_shape)
        self.linear_det.build(spatial_shape)
        stream_shape = self.linear_det.compute_output_shape(spatial_shape)

        # Causal conv: input gets left-padded by (kernel_size-1) before valid conv
        padded_shape = (*input_shape[:2],
                        (input_shape[2] or 0) + 2, input_shape[3])
        self.dw_conv.build(padded_shape)
        # After valid conv on padded input, output W = original W
        dw1_out = input_shape
        padded_shape2 = (*dw1_out[:2],
                         (dw1_out[2] or 0) + 2, dw1_out[3])
        self.dw_conv2.build(padded_shape2)
        self.ctx_bn.build(dw1_out)

        self.local_geo_prod.build(stream_shape)
        if self.global_geo_prod is not None:
            self.global_geo_prod.build(stream_shape)
        self.ggr.build(stream_shape)

        super().build(input_shape)

    # ------------------------------------------------------------------

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Forward pass with causal convolutions.

        :param inputs: Feature tensor ``(B, 1, seq_len, D)``.
        :param training: Whether in training mode.
        :return: Updated feature tensor ``(B, 1, seq_len, D)``.
        """
        x_prev = inputs

        # --- Step 1: Normalise ---
        x_norm = self.input_norm(x_prev)

        # --- Step 2: Dual-stream generation ---
        z_det = self.linear_det(x_norm)

        # Causal context stream: left-pad then valid conv (×2) -> BN -> SiLU
        z_ctx = self._causal_pad(x_norm)
        z_ctx = self.dw_conv(z_ctx)
        z_ctx = self._causal_pad(z_ctx)
        z_ctx = self.dw_conv2(z_ctx)
        z_ctx = keras.activations.silu(self.ctx_bn(z_ctx, training=training))

        if self.ctx_mode == "diff":
            z_ctx = z_ctx - z_det

        # --- Step 3: Local sparse geometric interaction ---
        g_feat = self.local_geo_prod(z_det, z_ctx)

        # --- Step 4: Optional global context branch (causal) ---
        # Uses cumulative mean along the sequence axis so position i only
        # sees the average of positions 0..i (preserves causality).
        if self.global_geo_prod is not None:
            c_glo = self._causal_cumulative_mean(x_norm)
            c_glo = c_glo - z_det
            g_glo = self.global_geo_prod(z_det, c_glo)
            g_feat = g_feat + g_glo

        # --- Step 5 & 6: GGR + residual ---
        h_mix = self.ggr(x_norm, g_feat, training=training)
        return x_prev + h_mix

    # ------------------------------------------------------------------

    @staticmethod
    def _causal_cumulative_mean(
        x: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """Causal global context: cumulative mean along the W (sequence) axis.

        For input ``(B, 1, seq_len, D)``, position *i* receives the mean of
        positions ``0..i``.  This preserves autoregressive causality while
        still providing each position with a growing global summary.
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
        return input_shape

    # ------------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "channels": self.channels,
            "shifts": self.shifts,
            "cli_mode": self.cli_mode,
            "ctx_mode": self.ctx_mode,
            "use_global_context": self.use_global_context,
            "layer_scale_init": self.layer_scale_init,
            "drop_path_rate": self.drop_path_rate,
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
        })
        return config


# ---------------------------------------------------------------------------
# CliffordNetBlockDS — single-7x7-DW context with optional stride downsampling
# ---------------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class CliffordNetBlockDS(keras.layers.Layer):
    """CliffordNetBlock variant with single 7x7 depthwise context conv and
    optional stride-based downsampling.

    Architectural differences vs :class:`CliffordNetBlock`:

    1. The context stream uses a *single* ``DepthwiseConv2D`` with
       ``kernel_size=7`` (configurable) instead of two stacked 3x3
       depthwise convolutions. The 7x7 DW captures the same effective
       receptive field but with a single kernel and BN.
    2. The depthwise conv may use ``strides > 1`` to spatially
       downsample the feature map. When ``strides > 1``:

       - ``x_norm`` is pooled (avg or max, selected by ``skip_pool``)
         before being split into the detail and context streams, so
         both streams share spatial dims for the element-wise
         geometric product.
       - The residual ``x_prev`` is pooled with the same pool type
         and stride, so the residual addition shape-matches.

       Channel dimensionality is preserved through the block. To
       change channels at a stage boundary, place a 1x1 projection
       outside the block (matches the existing CliffordNet hierarchy
       conventions).

    With ``strides == 1`` the block behaves like a thin
    single-7x7-DW variant of :class:`CliffordNetBlock` (no pooling
    layers built or applied).

    **Architecture Overview** (with ``strides=s``):

    .. code-block:: text

        ┌────────────────────────────────┐
        │  X_prev  [B, H, W, D]          │
        └───────────────┬────────────────┘
                        ▼
        ┌────────────────────────────────┐
        │  LayerNorm → X_norm            │
        └───────────────┬────────────────┘
                        ▼
            (if s>1) Pool(stream)
                        │
                        ▼
        ┌──────────────────────────────────┐
        │  X_norm_p  [B, H/s, W/s, D]      │
        └────────┬───────────────┬─────────┘
                 ▼               ▼
        ┌──────────────┐ ┌──────────────────┐
        │ Detail       │ │ Context          │
        │ Z_det=       │ │ DWConv7x7(s)→    │
        │  Linear(Xp)  │ │ BN→SiLU→Z_ctx    │
        └──────┬───────┘ └────────┬─────────┘
               │  (diff: Z_ctx -= Z_det)
               ├──────────┬───────┘
               ▼          ▼
        ┌────────────────────────────────┐
        │ Local Sparse Geometric Product │
        │ → G_feat                       │
        └───────────────┬────────────────┘
                        │  (+ optional global branch)
                        ▼
        ┌────────────────────────────────┐
        │ GGR(X_norm_p, G_feat) → H_mix  │
        └───────────────┬────────────────┘
                        ▼
            (if s>1) Pool(skip) on X_prev
                        │
                        ▼
        ┌────────────────────────────────┐
        │ X_out = X_skip + H_mix         │
        │ [B, H/s, W/s, D]               │
        └────────────────────────────────┘

    :param channels: Feature dimensionality D (constant throughout).
    :type channels: int
    :param shifts: Channel-shift offsets for the local interaction.
    :type shifts: List[int]
    :param cli_mode: Algebraic components (``"inner"``, ``"wedge"``,
        ``"full"``). Defaults to ``"full"``.
    :type cli_mode: CliMode
    :param ctx_mode: Context mode (``"diff"`` or ``"abs"``).
        Defaults to ``"diff"``.
    :type ctx_mode: CtxMode
    :param use_global_context: Whether to add a global-average-pool
        branch. Defaults to ``False``.
    :type use_global_context: bool
    :param kernel_size: Depthwise context kernel size. Defaults to ``7``.
    :type kernel_size: int
    :param strides: Spatial stride for the depthwise context conv and
        the residual / stream-split pool. ``1`` (default) preserves
        spatial dims; ``2`` halves H and W.
    :type strides: int
    :param skip_pool: Pool type used both to downsample ``x_norm``
        before the stream split and to downsample ``x_prev`` for the
        residual when ``strides > 1``. ``"avg"`` (default) or ``"max"``.
        Has no effect when ``strides == 1``.
    :type skip_pool: SkipPool
    :param layer_scale_init: Initial LayerScale value. Defaults to 1e-5.
    :type layer_scale_init: float
    :param drop_path_rate: DropPath probability. Defaults to 0.0.
    :type drop_path_rate: float
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
        kernel_size: int = 7,
        strides: int = 1,
        skip_pool: SkipPool = "avg",
        use_ctx_bn: bool = True,
        ctx_activation: Optional[str] = "silu",
        layer_scale_init: float = 1e-5,
        drop_path_rate: float = 0.0,
        use_bias: bool = True,
        kernel_initializer: Any = "glorot_uniform",
        bias_initializer: Any = "zeros",
        kernel_regularizer: Optional[Any] = None,
        bias_regularizer: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if ctx_mode not in ("diff", "abs"):
            raise ValueError(
                f"ctx_mode must be 'diff' or 'abs', got {ctx_mode!r}"
            )
        if not isinstance(kernel_size, int) or kernel_size <= 0:
            raise ValueError(
                f"kernel_size must be a positive int, got {kernel_size!r}"
            )
        if not isinstance(strides, int) or strides < 1:
            raise ValueError(
                f"strides must be an int >= 1, got {strides!r}"
            )
        if skip_pool not in ("avg", "max"):
            raise ValueError(
                f"skip_pool must be 'avg' or 'max', got {skip_pool!r}"
            )
        # See D-002: global branch uses shifts=[1, 2].
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
        self.skip_pool = skip_pool
        self.use_ctx_bn = use_ctx_bn
        self.ctx_activation = ctx_activation
        self.layer_scale_init = layer_scale_init
        self.drop_path_rate = drop_path_rate
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        _dense_kwargs: Dict[str, Any] = dict(
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )

        # --- Step 1: Input norm ---
        self.input_norm = keras.layers.LayerNormalization(
            epsilon=1e-6, name="input_norm"
        )

        # --- Step 2a: Detail stream (1x1 pointwise) ---
        self.linear_det = keras.layers.Dense(
            channels, name="linear_det", **_dense_kwargs
        )

        # --- Step 2b: Context stream — single (kernel_size x kernel_size) DW conv with optional stride ---
        # When ``use_ctx_bn`` is False the conv carries a learnable bias.
        # Note: a conv bias only restores the SHIFT half of BN's affine
        # transform, not the per-channel SCALE. If a calibrated scale is
        # needed without BN, use a LayerScale layer in addition.
        self.dw_conv = keras.layers.DepthwiseConv2D(
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            use_bias=not use_ctx_bn,
            name="dw_conv",
        )
        self.ctx_bn = (
            keras.layers.BatchNormalization(name="ctx_bn")
            if use_ctx_bn
            else None
        )

        # --- Pool layers (only when strides > 1) ---
        # Two separate instances: one used to pool x_norm before the
        # stream split, the other to pool x_prev for the residual skip.
        # Pools have no trainable state but separate instances keep
        # serialization and naming clean.
        if strides > 1:
            pool_cls = (
                keras.layers.AveragePooling2D
                if skip_pool == "avg"
                else keras.layers.MaxPooling2D
            )
            self.stream_pool = pool_cls(
                pool_size=strides,
                strides=strides,
                padding="same",
                name="stream_pool",
            )
            self.skip_pool_layer = pool_cls(
                pool_size=strides,
                strides=strides,
                padding="same",
                name="skip_pool",
            )
        else:
            self.stream_pool = None
            self.skip_pool_layer = None

        # --- Step 3: Local sparse rolling product ---
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

        # --- Optional global context branch ---
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

        # --- Step 4 / 5: GGR ---
        self.ggr = GatedGeometricResidual(
            channels=channels,
            layer_scale_init=layer_scale_init,
            drop_path_rate=drop_path_rate,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name="ggr",
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _ceildiv(a: Optional[int], b: int) -> Optional[int]:
        """Ceiling division that propagates ``None`` (dynamic dim)."""
        if a is None:
            return None
        return -(-a // b)

    def _pooled_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Return shape after stream/skip pool given current ``strides``."""
        if self.strides == 1:
            return input_shape
        b, h, w, d = input_shape
        return (b, self._ceildiv(h, self.strides),
                self._ceildiv(w, self.strides), d)

    # ------------------------------------------------------------------

    def build(self, input_shape: Tuple) -> None:
        """Build all sub-layers in dependency order.

        :param input_shape: ``(B, H, W, D)``
        :type input_shape: Tuple
        """
        # B7: isotropic in channels (see CliffordNetBlock.build).
        if input_shape[-1] is not None and input_shape[-1] != self.channels:
            raise ValueError(
                f"CliffordNetBlockDS is isotropic: expected last dim == "
                f"channels={self.channels}, got input_shape[-1]={input_shape[-1]}."
            )
        # Step 1: norm operates on full-resolution input
        self.input_norm.build(input_shape)

        # Pool layers operate on full-resolution input
        if self.stream_pool is not None:
            self.stream_pool.build(input_shape)
        if self.skip_pool_layer is not None:
            self.skip_pool_layer.build(input_shape)

        # Stream-side shapes (after optional stream pool)
        pooled_shape = self._pooled_shape(input_shape)

        # Step 2a: detail linear (operates on pooled shape)
        self.linear_det.build(pooled_shape)
        stream_shape = self.linear_det.compute_output_shape(pooled_shape)

        # Step 2b: depthwise context conv operates on full-res input.
        # Its output spatial dims equal the pooled shape (same ceil
        # semantics for stride+"same" padding).
        self.dw_conv.build(input_shape)
        if self.ctx_bn is not None:
            self.ctx_bn.build(stream_shape)

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
        :return: Updated feature tensor
            ``(B, H/strides, W/strides, D)``.
        :rtype: keras.KerasTensor
        """
        x_prev = inputs

        # --- Step 1: Normalise (full resolution) ---
        x_norm = self.input_norm(x_prev)

        # --- Optional spatial downsample of x_norm before stream split ---
        if self.stream_pool is not None:
            x_norm_p = self.stream_pool(x_norm)
        else:
            x_norm_p = x_norm

        # --- Step 2a: Detail stream on (possibly pooled) x_norm ---
        z_det = self.linear_det(x_norm_p)

        # --- Step 2b: Context stream — single (kxk) DW conv on full-res ---
        # When strides>1, the strided conv directly produces the
        # downsampled spatial map (matches x_norm_p shape).
        z_ctx = self.dw_conv(x_norm)
        if self.ctx_bn is not None:
            z_ctx = self.ctx_bn(z_ctx, training=training)
        if self.ctx_activation is not None:
            z_ctx = keras.activations.get(self.ctx_activation)(z_ctx)

        if self.ctx_mode == "diff":
            z_ctx = z_ctx - z_det  # discrete Laplacian approximation

        # --- Step 3: Local sparse geometric interaction ---
        g_feat = self.local_geo_prod(z_det, z_ctx)

        # --- Step 4: Optional global context branch ---
        if self.global_geo_prod is not None:
            # Drop redundant broadcast_to; subtraction broadcasts (B,1,1,D)
            # against (B,H,W,D) without materialising the full tensor.
            c_glo = keras.ops.mean(x_norm_p, axis=[1, 2], keepdims=True)
            c_glo = c_glo - z_det
            g_glo = self.global_geo_prod(z_det, c_glo)
            g_feat = g_feat + g_glo

        # --- Step 5 & 6: GGR + residual (with skip pool when strides>1) ---
        h_mix = self.ggr(x_norm_p, g_feat, training=training)
        if self.skip_pool_layer is not None:
            x_skip = self.skip_pool_layer(x_prev)
        else:
            x_skip = x_prev
        return x_skip + h_mix

    # ------------------------------------------------------------------

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape.

        :param input_shape: Input shape ``(B, H, W, D)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape with H, W divided by ``strides``
            (ceiling semantics for ``padding="same"``).
        :rtype: Tuple[Optional[int], ...]
        """
        return self._pooled_shape(input_shape)

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
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "skip_pool": self.skip_pool,
                "use_ctx_bn": self.use_ctx_bn,
                "ctx_activation": self.ctx_activation,
                "layer_scale_init": self.layer_scale_init,
                "drop_path_rate": self.drop_path_rate,
                "use_bias": self.use_bias,
                "kernel_initializer": initializers.serialize(self.kernel_initializer),
                "bias_initializer": initializers.serialize(self.bias_initializer),
                "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
                "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            }
        )
        return config


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
    """Design-space sibling of :class:`CliffordNetBlockDS` for the
    downsampling experiments described in
    ``analyses/analysis_2026-04-30_41b5e415/summary.md``.

    Differences vs :class:`CliffordNetBlockDS`:

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
    legacy ``CliffordNetBlockDS`` behaviour (the V0 baseline of the sweep).
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
        drop_path_rate: float = 0.0,
        use_bias: bool = True,
        kernel_initializer: Any = "glorot_uniform",
        bias_initializer: Any = "zeros",
        kernel_regularizer: Optional[Any] = None,
        bias_regularizer: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
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
        valid_pools = {
            "avg", "max", "blur", "gaussian_dw", "pixel_unshuffle", "resnetd"
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
        self.drop_path_rate = drop_path_rate
        self.use_bias = use_bias
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

        self.stream_pool = _make_pool_v2(
            stream_pool, channels, strides, name="stream_pool"
        )
        self.skip_pool = _make_pool_v2(
            skip_pool, channels, strides, name="skip_pool"
        )

        self._pyr_pool: Optional[keras.layers.Layer] = None
        self._pyr_up: Optional[keras.layers.Layer] = None
        if ctx_mode == "pyramid_diff" and strides > 1:
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
            drop_path_rate=drop_path_rate,
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

    def build(self, input_shape: Tuple) -> None:
        """Explicitly build every sub-layer in dependency order.

        Required for clean save/load: Keras serialisation records the
        built-state of each sub-layer at the moment of saving, and
        complains on load if any layer was implicitly built only via
        ``call``-time tracing. Mirrors :class:`CliffordNetBlockDS.build`.
        """
        # B7: isotropic core (the optional out_proj at the end is the only
        # place channels may change).
        if input_shape[-1] is not None and input_shape[-1] != self.channels:
            raise ValueError(
                f"CliffordNetBlockDSv2 expects input last dim == "
                f"channels={self.channels}, got input_shape[-1]={input_shape[-1]}."
            )
        b, h, w, _ = input_shape

        self.input_norm.build(input_shape)

        # Pool layers consume full-resolution input.
        self.stream_pool.build(input_shape)
        self.skip_pool.build(input_shape)

        # Stream-side shape after stream pool.
        pooled_h = self._ceildiv(h, self.strides)
        pooled_w = self._ceildiv(w, self.strides)
        stream_shape = (b, pooled_h, pooled_w, self.channels)

        self.linear_det.build(stream_shape)

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
                # DECISION D-003: pyramid_diff at strides>1.
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
            # P2 mirror: drop redundant broadcast_to.
            c_glo = keras.ops.mean(x_norm_p, axis=[1, 2], keepdims=True)
            c_glo = c_glo - z_det
            g_glo = self.global_geo_prod(z_det, c_glo)
            g_feat = g_feat + g_glo

        h_mix = self.ggr(x_norm_p, g_feat, training=training)
        x_skip = self.skip_pool(x_prev)
        out = x_skip + h_mix

        if self.out_proj is not None:
            out = self.out_proj(out)
        return out

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        b, h, w, _ = input_shape
        new_h = None if h is None else self._ceildiv(h, self.strides)
        new_w = None if w is None else self._ceildiv(w, self.strides)
        out_c = (
            self.out_channels if self.out_channels is not None
            else self.channels
        )
        return (b, new_h, new_w, out_c)

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
                "drop_path_rate": self.drop_path_rate,
                "use_bias": self.use_bias,
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
class CausalCliffordNetBlockDSv2(keras.layers.Layer):
    """Causal sibling of :class:`CliffordNetBlockDSv2` for autoregressive
    sequence modeling.

    Combines the V2 design space of :class:`CliffordNetBlockDSv2` (dual
    stream + ``local_geo_prod`` + GGR + skip + optional global branch +
    optional ``out_channels`` projection + configurable ctx norm/activation)
    with the causal padding paradigm of :class:`CausalCliffordNetBlock`.

    Expects 4-D input ``(B, 1, seq_len, D)`` (sequence reshaped for 2-D
    convolutions with ``H = 1``). The block is **strictly causal along W**:
    output at position ``i`` depends only on inputs at positions
    ``<= i*strides + (strides-1)`` (the input chunk that pools to output
    ``i``).

    Architectural differences vs :class:`CliffordNetBlockDSv2`:

    1. **Causal DW context conv.** The depthwise conv uses
       ``kernel_size=(1, kernel_size)``, ``padding="valid"`` and a manual
       left-only zero-pad of width ``kernel_size-1`` along W (mirrors
       :class:`CausalCliffordNetBlock`). Same output spatial shape as DSv2
       (``ceil(seq_len/strides)``).
    2. **Restricted ctx_mode.** ``"diff"`` and ``"abs"`` only —
       ``"pyramid_diff"`` is forbidden because the bilinear upsample mixes
       neighbours, breaking causality.
    3. **Restricted pool kinds.** ``stream_pool`` and ``skip_pool`` are
       restricted to ``{"avg", "max"}`` — ``blur``, ``gaussian_dw``,
       ``pixel_unshuffle`` and ``resnetd`` either leak future information or
       change channel semantics (see DECISION D-001 below).
    4. **Causal global context.** When ``use_global_context=True`` the
       global branch uses a *cumulative* mean along W (each pooled position
       sees the average of past+current pooled positions only), not the
       full spatial mean used by DSv2.

    All other components — ``input_norm`` (LayerNorm), ``linear_det``
    (1x1 Dense), SRGP local product, GGR, optional ``out_channels``
    projection — are unchanged from DSv2.

    :param channels: Feature dimensionality D. ``input_shape[-1]`` must
        equal ``channels`` (B7 contract).
    :param shifts: Channel-shift offsets for the sparse rolling product.
    :param cli_mode: Algebraic components (``"inner"``, ``"wedge"``,
        ``"full"``).
    :param ctx_mode: ``"diff"`` or ``"abs"``.
    :param use_global_context: Add causal cumulative-mean global branch.
    :param kernel_size: DW conv kernel size along W.
    :param strides: Downsampling factor along W. ``1`` preserves length;
        ``>1`` reduces W to ``ceil(seq_len/strides)``.
    :param stream_pool: ``"avg"`` or ``"max"``.
    :param skip_pool: ``"avg"`` or ``"max"``.
    :param out_channels: Optional 1x1 projection at the end of the block.
        ``None`` (default) preserves channel count.
    :param ctx_norm_type: ``"bn"``, ``"gn"``, ``"ln"``, or ``"none"``.
    :param ctx_activation: Optional activation after ctx_norm
        (default ``"silu"``).
    :param layer_scale_init: Initial LayerScale value for GGR.
    :param drop_path_rate: DropPath probability for GGR.
    :param use_bias: Whether Dense layers use bias.
    :param kernel_initializer: Kernel initializer.
    :param bias_initializer: Bias initializer.
    :param kernel_regularizer: Kernel regularizer.
    :param bias_regularizer: Bias regularizer.
    """

    def __init__(
        self,
        channels: int,
        shifts: List[int],
        cli_mode: CliMode = "full",
        ctx_mode: CausalCtxModeV2 = "diff",
        use_global_context: bool = False,
        kernel_size: int = 7,
        strides: int = 1,
        stream_pool: CausalSkipPoolV2 = "avg",
        skip_pool: CausalSkipPoolV2 = "avg",
        out_channels: Optional[int] = None,
        ctx_norm_type: CtxNormType = "bn",
        ctx_activation: Optional[str] = "silu",
        layer_scale_init: float = 1e-5,
        drop_path_rate: float = 0.0,
        use_bias: bool = True,
        kernel_initializer: Any = "glorot_uniform",
        bias_initializer: Any = "zeros",
        kernel_regularizer: Optional[Any] = None,
        bias_regularizer: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        # DECISION D-001: ctx_mode is restricted to 'diff'|'abs' for the
        # causal variant. 'pyramid_diff' is excluded because the
        # bilinear UpSampling2D used by the DSv2 pyramid path mixes
        # neighbouring positions, which would leak future information
        # along the W (sequence) axis. Anyone needing pyramid-diff must
        # use the non-causal CliffordNetBlockDSv2.
        if ctx_mode not in ("diff", "abs"):
            raise ValueError(
                f"ctx_mode must be 'diff' or 'abs' for the causal variant "
                f"(pyramid_diff is non-causal), got {ctx_mode!r}"
            )
        if not isinstance(kernel_size, int) or kernel_size <= 0:
            raise ValueError(
                f"kernel_size must be a positive int, got {kernel_size!r}"
            )
        if not isinstance(strides, int) or strides < 1:
            raise ValueError(f"strides must be int>=1, got {strides!r}")
        # DECISION D-001 (cont.): stream_pool / skip_pool restricted to
        # avg/max. blur and gaussian_dw use symmetric kernels that cross
        # the temporal boundary on the right edge -> future leak.
        # pixel_unshuffle changes channel count (incompatible with the
        # H=1 layout). resnetd's pool has the same constraint as a plain
        # pool but bundles a 1x1 Conv that we don't need here.
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
        if out_channels is not None and out_channels <= 0:
            raise ValueError(
                f"out_channels must be positive or None, got {out_channels!r}"
            )
        if ctx_norm_type not in ("bn", "gn", "ln", "none"):
            raise ValueError(
                f"ctx_norm_type must be 'bn'|'gn'|'ln'|'none', got "
                f"{ctx_norm_type!r}"
            )
        # See D-002 (CliffordNetBlock): global branch needs channels >= 2.
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
        self.drop_path_rate = drop_path_rate
        self.use_bias = use_bias
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

        # DECISION D-002: causal DW conv = (1, kernel_size) + valid +
        # manual left-pad of (kernel_size - 1) along W. Same pattern as
        # CausalCliffordNetBlock but generalised to arbitrary kernel_size
        # and stride. With stride s, output W = ceil(W/s) (verified in
        # findings/dsv2-merge-points.md): floor((W + k - 1 - k)/s) + 1 =
        # floor((W-1)/s) + 1 = ceil(W/s).
        # Implementation note: TF/CUDA's DepthwiseConv2D requires equal
        # row/col strides, which we cannot satisfy here (we need
        # strides=(1, s)). We use Conv2D with groups=channels instead
        # — algebraically identical to depthwise but supports asymmetric
        # strides on GPU.
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
        self.ctx_norm = _make_ctx_norm(
            ctx_norm_type, channels, name="ctx_norm"
        )

        self.stream_pool = _make_causal_pool(
            stream_pool, strides, name="stream_pool"
        )
        self.skip_pool = _make_causal_pool(
            skip_pool, strides, name="skip_pool"
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
            drop_path_rate=drop_path_rate,
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
        position only sees past/current input positions).
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
        by the global-context branch to preserve causality (vs DSv2's
        full-spatial mean which would mix future into past).
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

        Validates the H=1 contract and ``input_shape[-1] == channels``.
        Mirrors :class:`CliffordNetBlockDSv2.build` with two differences:
        (1) accepts and validates the H=1 layout, (2) builds ``dw_conv``
        on the LEFT-PADDED shape (W' = W + kernel_size - 1).
        """
        if input_shape[-1] is not None and input_shape[-1] != self.channels:
            raise ValueError(
                f"CausalCliffordNetBlockDSv2 expects input last dim == "
                f"channels={self.channels}, got "
                f"input_shape[-1]={input_shape[-1]} (isotropic core)."
            )
        if input_shape[1] is not None and input_shape[1] != 1:
            raise ValueError(
                f"CausalCliffordNetBlockDSv2 expects H == 1 (input shape "
                f"(B, 1, seq_len, D)), got input_shape[1]={input_shape[1]}."
            )
        b, h, w, _ = input_shape

        self.input_norm.build(input_shape)

        # Pool layers consume full-resolution input.
        self.stream_pool.build(input_shape)
        self.skip_pool.build(input_shape)

        # Stream-side shape after stream pool (H stays at 1).
        pooled_w = self._ceildiv(w, self.strides)
        stream_shape = (b, h, pooled_w, self.channels)

        self.linear_det.build(stream_shape)

        # DW conv consumes the LEFT-PADDED full-resolution input.
        # padded_w = w + kernel_size - 1 (None-safe).
        padded_w = None if w is None else (w + self.kernel_size - 1)
        padded_shape = (b, h, padded_w, self.channels)
        self.dw_conv.build(padded_shape)
        if self.ctx_norm is not None:
            self.ctx_norm.build(stream_shape)

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
        """Forward pass with strict causal mixing along W.

        :param inputs: Tensor ``(B, 1, seq_len, D)``.
        :param training: Whether in training mode.
        :return: Tensor ``(B, 1, ceil(seq_len/strides), out_channels)``
            where ``out_channels`` defaults to ``channels``.
        """
        x_prev = inputs

        x_norm = self.input_norm(x_prev)
        x_norm_p = self.stream_pool(x_norm)
        z_det = self.linear_det(x_norm_p)

        # Causal context: left-pad along W, then valid conv with stride.
        x_padded = self._causal_pad_w(x_norm, self.kernel_size)
        z_ctx = self.dw_conv(x_padded)
        if self.ctx_norm is not None:
            z_ctx = self.ctx_norm(z_ctx, training=training)
        if self.ctx_activation is not None:
            z_ctx = keras.activations.get(self.ctx_activation)(z_ctx)

        if self.ctx_mode == "diff":
            z_ctx = z_ctx - z_det
        # 'abs' is a no-op pass-through.

        g_feat = self.local_geo_prod(z_det, z_ctx)

        if self.global_geo_prod is not None:
            # Causal global branch: cumulative mean over the pooled W axis.
            c_glo = self._causal_cumulative_mean_w(x_norm_p)
            c_glo = c_glo - z_det
            g_glo = self.global_geo_prod(z_det, c_glo)
            g_feat = g_feat + g_glo

        h_mix = self.ggr(x_norm_p, g_feat, training=training)
        x_skip = self.skip_pool(x_prev)
        out = x_skip + h_mix

        if self.out_proj is not None:
            out = self.out_proj(out)
        return out

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        b, h, w, _ = input_shape
        new_w = None if w is None else self._ceildiv(w, self.strides)
        out_c = (
            self.out_channels if self.out_channels is not None
            else self.channels
        )
        return (b, h, new_w, out_c)

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
                "drop_path_rate": self.drop_path_rate,
                "use_bias": self.use_bias,
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
