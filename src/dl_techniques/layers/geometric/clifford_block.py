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

from typing import Any, Dict, List, Literal, Optional, Tuple

import keras
from keras import initializers, regularizers

from ..stochastic_depth import StochasticDepth
from ...utils.logger import logger

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

CliMode = Literal["inner", "wedge", "full"]
CtxMode = Literal["diff", "abs"]

# Global-branch constants matching the original implementation
_GLOBAL_SHIFTS: List[int] = [1, 2]
_GLOBAL_CLI_MODE: CliMode = "full"


# ===========================================================================
# SparseRollingGeometricProduct
# ===========================================================================


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
            z_det_s = keras.ops.roll(z_det, shift=s, axis=-1)
            z_ctx_s = keras.ops.roll(z_ctx, shift=s, axis=-1)

            if self.cli_mode in ("wedge", "full"):
                # Bivector: anti-symmetric cross-term
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
                # Store the original (unfiltered) shifts so that round-trip
                # serialisation always re-applies the same filter.
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


# ===========================================================================
# GatedGeometricResidual
# ===========================================================================


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


# ===========================================================================
# CliffordNetBlock
# ===========================================================================


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
        │  X_prev  [B, H, W, D]         │
        └───────────────┬────────────────┘
                        ▼
        ┌────────────────────────────────┐
        │  LayerNorm → X_norm            │
        └───────┬───────────────┬────────┘
                ▼               ▼
        ┌──────────────┐ ┌──────────────────┐
        │ Detail       │ │ Context           │
        │ Z_det=       │ │ DWConv→DWConv→    │
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
        │ GGR(X_norm, G_feat) → H_mix   │
        └───────────────┬────────────────┘
                        ▼
        ┌────────────────────────────────┐
        │ X_out = X_prev + H_mix        │
        │ [B, H, W, D]                  │
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
            c_glo = keras.ops.mean(x_norm, axis=[1, 2], keepdims=True)
            c_glo = keras.ops.broadcast_to(c_glo, keras.ops.shape(z_det))
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


# ===========================================================================
# CausalCliffordNetBlock — sequence-safe variant for autoregressive LMs
# ===========================================================================


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