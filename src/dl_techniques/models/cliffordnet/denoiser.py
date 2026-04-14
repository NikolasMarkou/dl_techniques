"""
CliffordNet bias-free image denoiser.

A bias-free denoising model built on CliffordNet geometric-algebra blocks,
designed to satisfy the requirements of Miyasawa's theorem (1961) for
optimal denoising and implicit score function estimation.

Architecture: Conv2D stem (bias-free) -> L x CliffordNetBlock(use_bias=False)
-> LayerNorm -> Conv2D 1x1 (bias-free, linear) -> residual skip connection.

Bias-free design principles (from Miyasawa's theorem)
------------------------------------------------------
1. **No additive bias** in any Conv2D or Dense layer (``use_bias=False``).
2. **Normalization without centering**: ``BatchNormalization(center=False)``,
   ``LayerNormalization(center=False)`` — scale only, no shift.
3. **Linear final output**: no activation on the output projection.
4. **Residual learning**: ``output = input + f(input)`` so the network
   learns the noise residual (proportional to the score function).
5. **Zero-centered inputs**: data normalised to ``[-1, +1]``.

Pre-defined variants
--------------------
- ``CliffordNetDenoiser.tiny``  -- channels=64,  depth=6,  shifts=[1,2]
- ``CliffordNetDenoiser.small`` -- channels=96,  depth=8,  shifts=[1,2,4]
- ``CliffordNetDenoiser.base``  -- channels=128, depth=12, shifts=[1,2,4,8]
- ``CliffordNetDenoiser.large`` -- channels=128, depth=16, shifts=[1,2,4,8,16]
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import keras
from keras import initializers, regularizers

from dl_techniques.layers.geometric.clifford_block import (
    CliMode,
    CtxMode,
    SparseRollingGeometricProduct,
    GatedGeometricResidual,
)
from dl_techniques.layers.stochastic_depth import StochasticDepth
from dl_techniques.utils.logger import logger

# Match CliffordNet reference: trunc_normal_(std=0.02)
_DEFAULT_KERNEL_INIT = initializers.TruncatedNormal(stddev=0.02)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _linear_drop_path_rates(num_blocks: int, max_rate: float) -> List[float]:
    """Linearly spaced drop-path rates from 0 to *max_rate*."""
    if num_blocks <= 1:
        return [0.0] * num_blocks
    step = max_rate / (num_blocks - 1)
    return [round(i * step, 6) for i in range(num_blocks)]


# ===========================================================================
# BiasFreeClifordNetBlock
# ===========================================================================


@keras.saving.register_keras_serializable()
class BiasFreeClifordNetBlock(keras.layers.Layer):
    """Bias-free CliffordNet block for denoising.

    Identical architecture to :class:`CliffordNetBlock` but with all
    additive bias/centering terms removed to satisfy Miyasawa's theorem:

    - ``LayerNormalization(center=False)`` (scale only)
    - ``BatchNormalization(center=False)`` (scale only)
    - All Dense / projection layers with ``use_bias=False``
    - GGR gate Dense with ``use_bias=False``

    :param channels: Feature dimensionality D.
    :param shifts: Channel-shift offsets for sparse rolling product.
    :param cli_mode: ``"inner"`` | ``"wedge"`` | ``"full"``.
    :param ctx_mode: ``"diff"`` | ``"abs"``.
    :param use_global_context: Add global-average-pool branch.
    :param layer_scale_init: Initial LayerScale value.
    :param drop_path_rate: DropPath probability.
    :param kernel_initializer: Kernel initializer.
    :param kernel_regularizer: Kernel regularizer.
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
        kernel_initializer: Any = "glorot_uniform",
        kernel_regularizer: Optional[Any] = None,
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
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

        _dense_kwargs: Dict[str, Any] = dict(
            use_bias=False,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
        )

        # --- Step 1: Input norm (bias-free: center=False) ---
        self.input_norm = keras.layers.LayerNormalization(
            epsilon=1e-6, center=False, name="input_norm"
        )

        # --- Step 2a: Detail stream (1x1 pointwise, bias-free) ---
        self.linear_det = keras.layers.Dense(
            channels, name="linear_det", **_dense_kwargs
        )

        # --- Step 2b: Context stream (two DWConv + BN bias-free) ---
        self.dw_conv = keras.layers.DepthwiseConv2D(
            kernel_size=3, padding="same", use_bias=False, name="dw_conv",
        )
        self.dw_conv2 = keras.layers.DepthwiseConv2D(
            kernel_size=3, padding="same", use_bias=False, name="dw_conv2",
        )
        self.ctx_bn = keras.layers.BatchNormalization(
            center=False, name="ctx_bn"
        )

        # --- Step 3: Local sparse rolling product (bias-free) ---
        self.local_geo_prod = SparseRollingGeometricProduct(
            channels=channels, shifts=shifts, cli_mode=cli_mode,
            use_bias=False,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name="local_geo_prod",
        )

        # --- Optional global context branch ---
        if use_global_context:
            self.global_geo_prod = SparseRollingGeometricProduct(
                channels=channels, shifts=[1, 2], cli_mode="full",
                use_bias=False,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name="global_geo_prod",
            )
        else:
            self.global_geo_prod = None

        # --- Step 4: Bias-free GGR ---
        self.ggr = BiasFreeGatedGeometricResidual(
            channels=channels,
            layer_scale_init=layer_scale_init,
            drop_path_rate=drop_path_rate,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name="ggr",
        )

    def build(self, input_shape: Tuple) -> None:
        spatial_shape = input_shape
        self.input_norm.build(spatial_shape)
        self.linear_det.build(spatial_shape)
        stream_shape = self.linear_det.compute_output_shape(spatial_shape)
        self.dw_conv.build(spatial_shape)
        dw1_out = self.dw_conv.compute_output_shape(spatial_shape)
        self.dw_conv2.build(dw1_out)
        dw2_out = self.dw_conv2.compute_output_shape(dw1_out)
        self.ctx_bn.build(dw2_out)
        self.local_geo_prod.build(stream_shape)
        if self.global_geo_prod is not None:
            self.global_geo_prod.build(stream_shape)
        self.ggr.build(stream_shape)
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        x_prev = inputs
        x_norm = self.input_norm(x_prev)

        # Dual-stream
        z_det = self.linear_det(x_norm)
        z_ctx = self.dw_conv(x_norm)
        z_ctx = self.dw_conv2(z_ctx)
        z_ctx = keras.activations.silu(self.ctx_bn(z_ctx, training=training))

        if self.ctx_mode == "diff":
            z_ctx = z_ctx - z_det

        # Local sparse geometric interaction
        g_feat = self.local_geo_prod(z_det, z_ctx)

        # Optional global context branch
        if self.global_geo_prod is not None:
            c_glo = keras.ops.mean(x_norm, axis=[1, 2], keepdims=True)
            c_glo = keras.ops.broadcast_to(c_glo, keras.ops.shape(z_det))
            c_glo = c_glo - z_det
            g_glo = self.global_geo_prod(z_det, c_glo)
            g_feat = g_feat + g_glo

        # GGR + residual
        h_mix = self.ggr(x_norm, g_feat, training=training)
        return x_prev + h_mix

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        return input_shape

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
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
        })
        return config


# ===========================================================================
# BiasFreeGatedGeometricResidual
# ===========================================================================


@keras.saving.register_keras_serializable()
class BiasFreeGatedGeometricResidual(keras.layers.Layer):
    """Bias-free variant of GatedGeometricResidual.

    Same architecture as :class:`GatedGeometricResidual` but with
    ``use_bias=False`` on the gate Dense layer.

    :param channels: Feature dimensionality D.
    :param layer_scale_init: Initial LayerScale gamma.
    :param drop_path_rate: Stochastic-depth probability.
    :param kernel_initializer: Initializer for the gate kernel.
    :param kernel_regularizer: Regularizer for the gate kernel.
    """

    def __init__(
        self,
        channels: int,
        layer_scale_init: float = 1e-5,
        drop_path_rate: float = 0.0,
        kernel_initializer: Any = "glorot_uniform",
        kernel_regularizer: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if not (0.0 <= drop_path_rate < 1.0):
            raise ValueError(f"drop_path_rate must be in [0, 1), got {drop_path_rate}")

        self.channels = channels
        self.layer_scale_init = layer_scale_init
        self.drop_path_rate = drop_path_rate
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

        # Bias-free gate: Dense(2C -> C, use_bias=False) + sigmoid
        self.gate_dense = keras.layers.Dense(
            channels, use_bias=False,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name="gate_dense",
        )

        self.drop_path = (
            StochasticDepth(drop_path_rate=drop_path_rate, name="drop_path")
            if drop_path_rate > 0.0 else None
        )

    def build(self, input_shape: Tuple) -> None:
        self.gamma = self.add_weight(
            name="gamma",
            shape=(self.channels,),
            initializer=initializers.Constant(self.layer_scale_init),
            trainable=True,
        )
        gate_input_shape = (*input_shape[:-1], 2 * self.channels)
        self.gate_dense.build(gate_input_shape)
        super().build(input_shape)

    def call(
        self,
        h_norm: keras.KerasTensor,
        g_feat: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        gate_input = keras.ops.concatenate([h_norm, g_feat], axis=-1)
        alpha = keras.activations.sigmoid(self.gate_dense(gate_input))
        h_mix = keras.activations.silu(h_norm) + alpha * g_feat
        h_mix = h_mix * self.gamma
        if self.drop_path is not None:
            h_mix = self.drop_path(h_mix, training=training)
        return h_mix

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        return (*input_shape[:-1], self.channels)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "channels": self.channels,
            "layer_scale_init": self.layer_scale_init,
            "drop_path_rate": self.drop_path_rate,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
        })
        return config


# ===========================================================================
# CliffordNetDenoiser
# ===========================================================================


@keras.saving.register_keras_serializable()
class CliffordNetDenoiser(keras.Model):
    """Bias-free CliffordNet image denoiser.

    Satisfies the requirements of Miyasawa's theorem for optimal
    least-squares denoising and implicit score function estimation:
    all layers are bias-free, normalizations use scale-only (no centering),
    and the output is a residual connection with linear projection.

    The learned residual is proportional to the score function:
    ``residual = sigma^2 * nabla_y log p(y)``

    Architecture::

        Input (B, H, W, C)
            |
        Conv2D stem (bias-free) + BN(center=False)
            |
        L x BiasFreeClifordNetBlock
            |
        LayerNorm(center=False)
            |
        Conv2D 1x1 (bias-free, linear)  -->  residual
            |
        Output = Input + residual

    :param channels: Internal feature dimensionality D.
    :param depth: Number of CliffordNet blocks L.
    :param in_channels: Number of input image channels (1 for grayscale,
        3 for RGB). Defaults to 1.
    :param shifts: Channel-shift offsets for sparse rolling product.
    :param cli_mode: ``"inner"`` | ``"wedge"`` | ``"full"``.
    :param ctx_mode: ``"diff"`` | ``"abs"``.
    :param use_global_context: Add global-average-pool gFFN-G branch.
    :param layer_scale_init: Initial LayerScale value.
    :param stochastic_depth_rate: Maximum DropPath rate (linearly scheduled).
    :param kernel_initializer: Kernel initializer.
    :param kernel_regularizer: Kernel regularizer.
    """

    LAYERNORM_EPSILON: float = 1e-6

    def __init__(
        self,
        channels: int = 64,
        depth: int = 6,
        in_channels: int = 1,
        shifts: Optional[List[int]] = None,
        cli_mode: CliMode = "full",
        ctx_mode: CtxMode = "diff",
        use_global_context: bool = False,
        layer_scale_init: float = 1e-5,
        stochastic_depth_rate: float = 0.0,
        kernel_initializer: Any = "glorot_uniform",
        kernel_regularizer: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")

        self.channels = channels
        self.depth = depth
        self.in_channels = in_channels
        self.shifts = shifts if shifts is not None else [1, 2]
        self.cli_mode = cli_mode
        self.ctx_mode = ctx_mode
        self.use_global_context = use_global_context
        self.layer_scale_init = layer_scale_init
        self.stochastic_depth_rate = stochastic_depth_rate
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

        self._build_stem()
        self._build_blocks()
        self._build_head()

        logger.info(
            f"Created CliffordNetDenoiser (channels={channels}, depth={depth}, "
            f"in_channels={in_channels}, shifts={self.shifts}, "
            f"cli_mode={cli_mode}, ctx_mode={ctx_mode}, "
            f"use_global_context={use_global_context}, bias_free=True)"
        )

    # ------------------------------------------------------------------
    # Private builders
    # ------------------------------------------------------------------

    def _build_stem(self) -> None:
        """Bias-free stem: Conv2D + BN(center=False)."""
        self.stem_conv = keras.layers.Conv2D(
            filters=self.channels,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="stem_conv",
        )
        self.stem_norm = keras.layers.BatchNormalization(
            center=False, name="stem_norm"
        )

    def _build_blocks(self) -> None:
        """Build L bias-free CliffordNet blocks with linear drop-path."""
        drop_rates = _linear_drop_path_rates(
            self.depth, self.stochastic_depth_rate
        )
        self.blocks_list: List[BiasFreeClifordNetBlock] = []
        for i in range(self.depth):
            block = BiasFreeClifordNetBlock(
                channels=self.channels,
                shifts=self.shifts,
                cli_mode=self.cli_mode,
                ctx_mode=self.ctx_mode,
                use_global_context=self.use_global_context,
                layer_scale_init=self.layer_scale_init,
                drop_path_rate=drop_rates[i],
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"bf_clifford_block_{i}",
            )
            self.blocks_list.append(block)

    def _build_head(self) -> None:
        """Bias-free output head: LayerNorm + 1x1 Conv (linear)."""
        self.head_norm = keras.layers.LayerNormalization(
            epsilon=self.LAYERNORM_EPSILON,
            center=False,
            name="head_norm",
        )
        # 1x1 projection back to input channels -- linear (no activation)
        self.output_proj = keras.layers.Conv2D(
            filters=self.in_channels,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="output_proj",
        )

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        super().build(input_shape)
        if len(input_shape) == 3:
            build_shape = (None,) + tuple(input_shape)
        else:
            build_shape = tuple(input_shape)
        dummy = keras.KerasTensor(build_shape)
        _ = self.call(dummy)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Forward pass: learns noise residual, adds back to input.

        :param inputs: Noisy image ``(B, H, W, C)`` in ``[-1, +1]``.
        :param training: Whether in training mode.
        :return: Denoised image ``(B, H, W, C)``.
        """
        # Stem
        x = self.stem_conv(inputs)
        x = self.stem_norm(x, training=training)

        # Backbone blocks
        for block in self.blocks_list:
            x = block(x, training=training)

        # Head: LayerNorm + linear 1x1 projection
        x = self.head_norm(x)
        residual = self.output_proj(x)

        # Residual learning: denoised = noisy + learned_residual
        return inputs + residual

    # ------------------------------------------------------------------
    # Shape inference
    # ------------------------------------------------------------------

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        return input_shape

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "channels": self.channels,
            "depth": self.depth,
            "in_channels": self.in_channels,
            "shifts": self.shifts,
            "cli_mode": self.cli_mode,
            "ctx_mode": self.ctx_mode,
            "use_global_context": self.use_global_context,
            "layer_scale_init": self.layer_scale_init,
            "stochastic_depth_rate": self.stochastic_depth_rate,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CliffordNetDenoiser":
        for key in ("kernel_regularizer",):
            if config.get(key) and isinstance(config[key], dict):
                config[key] = regularizers.deserialize(config[key])
        return cls(**config)

    # ------------------------------------------------------------------
    # Summary override
    # ------------------------------------------------------------------

    def summary(self, **kwargs: Any) -> None:
        if not self.built:
            logger.warning("Model not built; calling build() with symbolic input.")
            dummy = keras.KerasTensor((None, None, None, self.in_channels))
            self.build(dummy.shape)
        super().summary(**kwargs)
        logger.info("CliffordNetDenoiser configuration:")
        logger.info(f"  channels            : {self.channels}")
        logger.info(f"  depth               : {self.depth}")
        logger.info(f"  in_channels         : {self.in_channels}")
        logger.info(f"  shifts              : {self.shifts}")
        logger.info(f"  cli_mode            : {self.cli_mode}")
        logger.info(f"  ctx_mode            : {self.ctx_mode}")
        logger.info(f"  use_global_context  : {self.use_global_context}")
        logger.info(f"  stochastic_depth    : {self.stochastic_depth_rate}")
        logger.info(f"  bias_free           : True")

    # ------------------------------------------------------------------
    # Factory class methods
    # ------------------------------------------------------------------

    MODEL_VARIANTS: Dict[str, Dict[str, Any]] = {
        "tiny": dict(
            channels=64,
            depth=6,
            shifts=[1, 2],
            cli_mode="full",
            ctx_mode="diff",
            use_global_context=False,
            layer_scale_init=1e-5,
            kernel_initializer=_DEFAULT_KERNEL_INIT,
        ),
        "small": dict(
            channels=96,
            depth=8,
            shifts=[1, 2, 4],
            cli_mode="full",
            ctx_mode="diff",
            use_global_context=False,
            layer_scale_init=1e-5,
            kernel_initializer=_DEFAULT_KERNEL_INIT,
        ),
        "base": dict(
            channels=128,
            depth=12,
            shifts=[1, 2, 4, 8],
            cli_mode="full",
            ctx_mode="diff",
            use_global_context=False,
            layer_scale_init=1e-5,
            kernel_initializer=_DEFAULT_KERNEL_INIT,
        ),
        "large": dict(
            channels=128,
            depth=16,
            shifts=[1, 2, 4, 8, 16],
            cli_mode="full",
            ctx_mode="diff",
            use_global_context=True,
            layer_scale_init=1e-5,
            kernel_initializer=_DEFAULT_KERNEL_INIT,
        ),
    }

    @classmethod
    def from_variant(
        cls,
        variant: str,
        in_channels: int = 1,
        **kwargs: Any,
    ) -> "CliffordNetDenoiser":
        """Create a denoiser from a predefined variant.

        :param variant: One of ``"tiny"``, ``"small"``, ``"base"``, ``"large"``.
        :param in_channels: Number of input channels (1 or 3).
        :param kwargs: Override any default hyperparameter.
        :return: Configured :class:`CliffordNetDenoiser` instance.
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                f"Available: {list(cls.MODEL_VARIANTS.keys())}"
            )
        defaults = dict(cls.MODEL_VARIANTS[variant])
        defaults.update(kwargs)
        logger.info(f"Creating CliffordNetDenoiser-{variant.upper()}")
        return cls(in_channels=in_channels, **defaults)

    @classmethod
    def tiny(cls, in_channels: int = 1, **kwargs: Any) -> "CliffordNetDenoiser":
        """CliffordNetDenoiser-Tiny: channels=64, depth=6, shifts=[1,2]."""
        return cls.from_variant("tiny", in_channels=in_channels, **kwargs)

    @classmethod
    def small(cls, in_channels: int = 1, **kwargs: Any) -> "CliffordNetDenoiser":
        """CliffordNetDenoiser-Small: channels=96, depth=8, shifts=[1,2,4]."""
        return cls.from_variant("small", in_channels=in_channels, **kwargs)

    @classmethod
    def base(cls, in_channels: int = 1, **kwargs: Any) -> "CliffordNetDenoiser":
        """CliffordNetDenoiser-Base: channels=128, depth=12, shifts=[1,2,4,8]."""
        return cls.from_variant("base", in_channels=in_channels, **kwargs)

    @classmethod
    def large(cls, in_channels: int = 1, **kwargs: Any) -> "CliffordNetDenoiser":
        """CliffordNetDenoiser-Large: channels=128, depth=16, shifts=[1,2,4,8,16]."""
        return cls.from_variant("large", in_channels=in_channels, **kwargs)
