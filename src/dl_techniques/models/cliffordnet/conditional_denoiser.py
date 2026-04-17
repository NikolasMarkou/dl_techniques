"""
CliffordNet conditional denoiser with Miyasawa's theorem compliance.

Implements a multi-scale U-Net denoiser built on bias-free CliffordNet
geometric-algebra blocks, supporting multi-modal conditioning:

- **Dense conditioning**: Spatial feature maps (e.g., RGB images → depth)
  injected via bias-free FiLM modulation on the context stream.
- **Discrete conditioning**: Class embeddings injected via spatial
  broadcast addition.
- **Hybrid conditioning**: Both simultaneously.

All layers are strictly bias-free to satisfy Miyasawa's theorem (1961)
for optimal least-squares denoising and implicit score function
estimation:

    hat{x}(y, c) = E[x|y, c] = y + sigma^2 * nabla_y log p(y|c)

Architecture::

    Inputs: noisy_target (B,H,W,C_target)
          + dense_condition (B,H,W,C_cond) [optional]
          + class_label (B,1) [optional]
        |
    Bias-free stem → (B, H/2, W/2, ch[0])
        |
    Encoder levels 0..N-1:
        [BiasFreeConditionedCliffordBlock × blocks_per_level] + downsample
        |
    Bottleneck:
        [BiasFreeConditionedCliffordBlock × blocks_per_level]
        |
    Decoder levels N-1..0:
        upsample + skip concat + [BiasFreeConditionedCliffordBlock × blocks_per_level]
        |
    Bias-free head: LayerNorm(center=False) → Conv2D 1×1 (linear, no bias)
        |
    Output = noisy_target + residual

Pre-defined variants
--------------------
- ``CliffordNetConditionalDenoiser.tiny``  -- 3 levels, ~2M params
- ``CliffordNetConditionalDenoiser.small`` -- 3 levels, ~5M params
- ``CliffordNetConditionalDenoiser.base``  -- 4 levels, ~12M params
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import keras
from keras import initializers, regularizers

from dl_techniques.layers.geometric.clifford_block import (
    CliMode,
    CtxMode,
    SparseRollingGeometricProduct,
)
from dl_techniques.layers.stochastic_depth import StochasticDepth
from dl_techniques.utils.logger import logger

# Match CliffordNet reference: trunc_normal_(std=0.02)
_DEFAULT_KERNEL_INIT = initializers.TruncatedNormal(stddev=0.02)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _linear_drop_path_rates(num_blocks: int, max_rate: float) -> List[float]:
    """Linearly spaced drop-path rates from 0 to *max_rate*."""
    if num_blocks <= 1:
        return [0.0] * num_blocks
    step = max_rate / (num_blocks - 1)
    return [round(i * step, 6) for i in range(num_blocks)]


# ===========================================================================
# BiasFreeConditionedGGR
# ===========================================================================


@keras.saving.register_keras_serializable()
class BiasFreeConditionedGGR(keras.layers.Layer):
    """Bias-free Gated Geometric Residual with conditioning support.

    Same as BiasFreeGatedGeometricResidual but the gate also receives
    a conditioning vector when available, enabling the gate to modulate
    the geometric features based on external context.

    :param channels: Feature dimensionality D.
    :param layer_scale_init: Initial LayerScale gamma.
    :param drop_path_rate: Stochastic-depth probability.
    :param kernel_initializer: Kernel initializer.
    :param kernel_regularizer: Kernel regularizer.
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

        self.channels = channels
        self.layer_scale_init = layer_scale_init
        self.drop_path_rate = drop_path_rate
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

        self.gate_dense = keras.layers.Dense(
            channels,
            use_bias=False,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name="gate_dense",
        )
        self.drop_path = (
            StochasticDepth(drop_path_rate=drop_path_rate, name="drop_path")
            if drop_path_rate > 0.0
            else None
        )

    def build(self, input_shape: Tuple) -> None:
        self._input_shape_for_build = input_shape
        self.gamma = self.add_weight(
            name="gamma",
            shape=(self.channels,),
            initializer=initializers.Constant(self.layer_scale_init),
            trainable=True,
        )
        gate_input_shape = (*input_shape[:-1], 2 * self.channels)
        self.gate_dense.build(gate_input_shape)
        super().build(input_shape)

    def get_build_config(self) -> Dict[str, Any]:
        if hasattr(self, "_input_shape_for_build"):
            return {"input_shape": self._input_shape_for_build}
        return {}

    def build_from_config(self, config: Dict[str, Any]) -> None:
        if "input_shape" in config:
            self.build(config["input_shape"])

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
            "kernel_initializer": initializers.serialize(
                self.kernel_initializer
            ),
            "kernel_regularizer": regularizers.serialize(
                self.kernel_regularizer
            ),
        })
        return config


# ===========================================================================
# BiasFreeConditionedCliffordBlock
# ===========================================================================


@keras.saving.register_keras_serializable()
class BiasFreeConditionedCliffordBlock(keras.layers.Layer):
    """Bias-free CliffordNet block with multi-modal conditioning.

    Extends the BiasFreeClifordNetBlock with conditioning injection into
    the context stream via bias-free FiLM (multiplicative modulation).
    Dense conditioning modulates the context stream before the geometric
    product. Discrete conditioning is added to the normalized input
    before the dual-stream split.

    The conditioning injection preserves the bias-free property required
    by Miyasawa's theorem: all modulations are multiplicative (no beta
    shift) and all projections use ``use_bias=False``.

    :param channels: Feature dimensionality D.
    :param shifts: Channel-shift offsets for sparse rolling product.
    :param cli_mode: ``"inner"`` | ``"wedge"`` | ``"full"``.
    :param ctx_mode: ``"diff"`` | ``"abs"``.
    :param use_global_context: Add global-average-pool branch.
    :param layer_scale_init: Initial LayerScale value.
    :param drop_path_rate: DropPath probability.
    :param enable_dense_conditioning: Accept dense spatial conditioning.
    :param enable_discrete_conditioning: Accept discrete class conditioning.
    :param class_embedding_dim: Dimension of discrete class embedding.
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
        enable_dense_conditioning: bool = False,
        enable_discrete_conditioning: bool = False,
        class_embedding_dim: int = 128,
        kernel_initializer: Any = "glorot_uniform",
        kernel_regularizer: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")

        self.channels = channels
        self.shifts = list(shifts)
        self.cli_mode = cli_mode
        self.ctx_mode = ctx_mode
        self.use_global_context = use_global_context
        self.layer_scale_init = layer_scale_init
        self.drop_path_rate = drop_path_rate
        self.enable_dense_conditioning = enable_dense_conditioning
        self.enable_discrete_conditioning = enable_discrete_conditioning
        self.class_embedding_dim = class_embedding_dim
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

        _dense_kwargs: Dict[str, Any] = dict(
            use_bias=False,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
        )

        # --- Input norm (bias-free: center=False) ---
        self.input_norm = keras.layers.LayerNormalization(
            epsilon=1e-6, center=False, name="input_norm"
        )

        # --- Detail stream (pointwise, bias-free) ---
        self.linear_det = keras.layers.Dense(
            channels, name="linear_det", **_dense_kwargs
        )

        # --- Context stream (DWConv + BN, bias-free) ---
        self.dw_conv = keras.layers.DepthwiseConv2D(
            kernel_size=3, padding="same", use_bias=False, name="dw_conv",
        )
        self.dw_conv2 = keras.layers.DepthwiseConv2D(
            kernel_size=3, padding="same", use_bias=False, name="dw_conv2",
        )
        self.ctx_bn = keras.layers.BatchNormalization(
            center=False, name="ctx_bn"
        )

        # --- Dense conditioning: bias-free FiLM on context stream ---
        if enable_dense_conditioning:
            # Learns multiplicative gamma from conditioning features
            self.cond_gamma_proj = keras.layers.Dense(
                channels,
                use_bias=False,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name="cond_gamma_proj",
            )

        # --- Discrete conditioning: projection for spatial broadcast ---
        if enable_discrete_conditioning:
            self.class_proj = keras.layers.Dense(
                channels,
                use_bias=False,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name="class_proj",
            )

        # --- Local sparse rolling geometric product (bias-free) ---
        self.local_geo_prod = SparseRollingGeometricProduct(
            channels=channels,
            shifts=shifts,
            cli_mode=cli_mode,
            use_bias=False,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name="local_geo_prod",
        )

        # --- Optional global context branch ---
        if use_global_context:
            self.global_geo_prod = SparseRollingGeometricProduct(
                channels=channels,
                shifts=[1, 2],
                cli_mode="full",
                use_bias=False,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name="global_geo_prod",
            )
        else:
            self.global_geo_prod = None

        # --- GGR gate (bias-free) ---
        self.ggr = BiasFreeConditionedGGR(
            channels=channels,
            layer_scale_init=layer_scale_init,
            drop_path_rate=drop_path_rate,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name="ggr",
        )

    def build(self, input_shape: Tuple) -> None:
        self._input_shape_for_build = input_shape
        spatial_shape = input_shape
        self.input_norm.build(spatial_shape)
        self.linear_det.build(spatial_shape)
        stream_shape = self.linear_det.compute_output_shape(spatial_shape)
        self.dw_conv.build(spatial_shape)
        dw1_out = self.dw_conv.compute_output_shape(spatial_shape)
        self.dw_conv2.build(dw1_out)
        dw2_out = self.dw_conv2.compute_output_shape(dw1_out)
        self.ctx_bn.build(dw2_out)

        if self.enable_dense_conditioning:
            self.cond_gamma_proj.build(spatial_shape)
        if self.enable_discrete_conditioning:
            cls_shape = (spatial_shape[0], self.class_embedding_dim)
            self.class_proj.build(cls_shape)

        self.local_geo_prod.build(stream_shape)
        if self.global_geo_prod is not None:
            self.global_geo_prod.build(stream_shape)
        self.ggr.build(stream_shape)
        super().build(input_shape)

    def get_build_config(self) -> Dict[str, Any]:
        if hasattr(self, "_input_shape_for_build"):
            return {"input_shape": self._input_shape_for_build}
        return {}

    def build_from_config(self, config: Dict[str, Any]) -> None:
        if "input_shape" in config:
            self.build(config["input_shape"])

    def call(
        self,
        inputs: keras.KerasTensor,
        dense_cond: Optional[keras.KerasTensor] = None,
        discrete_cond: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Forward pass with optional conditioning.

        :param inputs: Feature tensor ``(B, H, W, channels)``.
        :param dense_cond: Dense conditioning features
            ``(B, H, W, C_cond)``, or ``None``.
        :param discrete_cond: Class embedding vector ``(B, D_emb)``,
            or ``None``.
        :param training: Whether in training mode.
        :return: Output tensor ``(B, H, W, channels)``.
        """
        x_prev = inputs
        x_norm = self.input_norm(x_prev)

        # Inject discrete conditioning before dual-stream split
        if self.enable_discrete_conditioning and discrete_cond is not None:
            # Project and broadcast: (B, D_emb) → (B, 1, 1, channels)
            cls_feat = self.class_proj(discrete_cond)
            cls_feat = keras.ops.expand_dims(
                keras.ops.expand_dims(cls_feat, axis=1), axis=1
            )
            x_norm = x_norm + cls_feat

        # Dual-stream
        z_det = self.linear_det(x_norm)

        z_ctx = self.dw_conv(x_norm)
        z_ctx = self.dw_conv2(z_ctx)
        z_ctx = keras.activations.silu(
            self.ctx_bn(z_ctx, training=training)
        )

        # Inject dense conditioning via bias-free FiLM on context stream
        if self.enable_dense_conditioning and dense_cond is not None:
            # Multiplicative modulation: z_ctx = z_ctx * (1 + gamma)
            gamma = keras.activations.tanh(self.cond_gamma_proj(dense_cond))
            z_ctx = z_ctx * (1.0 + gamma)

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
            "enable_dense_conditioning": self.enable_dense_conditioning,
            "enable_discrete_conditioning": self.enable_discrete_conditioning,
            "class_embedding_dim": self.class_embedding_dim,
            "kernel_initializer": initializers.serialize(
                self.kernel_initializer
            ),
            "kernel_regularizer": regularizers.serialize(
                self.kernel_regularizer
            ),
        })
        return config


# ===========================================================================
# CliffordNetConditionalDenoiser
# ===========================================================================


@keras.saving.register_keras_serializable()
class CliffordNetConditionalDenoiser(keras.Model):
    """Bias-free multi-scale conditional CliffordNet denoiser.

    Implements a U-Net-style encoder-decoder with CliffordNet
    geometric-algebra blocks at each resolution level. Supports
    dense conditioning (spatial feature maps), discrete conditioning
    (class embeddings), or both simultaneously.

    All layers satisfy Miyasawa's theorem:
    - ``use_bias=False`` on all Conv2D and Dense layers
    - ``center=False`` on all LayerNorm and BatchNorm layers
    - Linear output projection (no activation)
    - Residual learning: ``output = input + residual``

    The learned residual is proportional to the conditional score:
    ``residual = sigma^2 * nabla_y log p(y|c)``

    :param in_channels: Number of input/output channels (e.g., 1 for
        depth, 3 for RGB).
    :param level_channels: Channel count at each encoder level, e.g.,
        ``[64, 128, 256]`` for 3 levels + bottleneck at 256.
    :param level_blocks: Number of CliffordNet blocks per level.
    :param level_shifts: Shift patterns per level (list of lists).
    :param cli_mode: Geometric product mode.
    :param ctx_mode: Context mode.
    :param use_global_context: Add global branch at each level.
    :param layer_scale_init: Initial LayerScale value.
    :param stochastic_depth_rate: Maximum drop-path rate.
    :param enable_dense_conditioning: Accept dense conditioning input.
    :param dense_cond_channels: Channel count of dense conditioning
        features. Required if ``enable_dense_conditioning=True``.
    :param enable_discrete_conditioning: Accept discrete conditioning.
    :param num_classes: Number of classes for discrete conditioning.
    :param class_embedding_dim: Dimension of class embedding.
    :param kernel_initializer: Kernel initializer.
    :param kernel_regularizer: Kernel regularizer.
    """

    LAYERNORM_EPSILON: float = 1e-6

    def __init__(
        self,
        in_channels: int = 1,
        level_channels: Optional[List[int]] = None,
        level_blocks: Optional[List[int]] = None,
        level_shifts: Optional[List[List[int]]] = None,
        cli_mode: CliMode = "full",
        ctx_mode: CtxMode = "diff",
        use_global_context: bool = False,
        layer_scale_init: float = 1e-5,
        stochastic_depth_rate: float = 0.0,
        enable_dense_conditioning: bool = False,
        dense_cond_channels: Optional[int] = None,
        enable_discrete_conditioning: bool = False,
        num_classes: int = 0,
        class_embedding_dim: int = 128,
        kernel_initializer: Any = "glorot_uniform",
        kernel_regularizer: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if in_channels <= 0:
            raise ValueError(
                f"in_channels must be positive, got {in_channels}"
            )

        # Defaults
        if level_channels is None:
            level_channels = [64, 128, 256]
        if level_blocks is None:
            level_blocks = [2, 2, 4]
        if level_shifts is None:
            level_shifts = [[1, 2], [1, 2, 4], [1, 2, 4, 8]]

        if len(level_channels) != len(level_blocks):
            raise ValueError(
                f"level_channels ({len(level_channels)}) and "
                f"level_blocks ({len(level_blocks)}) must have same length"
            )
        if len(level_shifts) != len(level_channels):
            raise ValueError(
                f"level_shifts ({len(level_shifts)}) and "
                f"level_channels ({len(level_channels)}) must have same length"
            )

        self.in_channels = in_channels
        self.level_channels = list(level_channels)
        self.level_blocks = list(level_blocks)
        self.level_shifts = [list(s) for s in level_shifts]
        self.cli_mode = cli_mode
        self.ctx_mode = ctx_mode
        self.use_global_context = use_global_context
        self.layer_scale_init = layer_scale_init
        self.stochastic_depth_rate = stochastic_depth_rate
        self.enable_dense_conditioning = enable_dense_conditioning
        self.dense_cond_channels = dense_cond_channels
        self.enable_discrete_conditioning = enable_discrete_conditioning
        self.num_classes = num_classes
        self.class_embedding_dim = class_embedding_dim
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

        self.num_levels = len(level_channels)
        # Bottleneck uses same config as last encoder level
        self.bottleneck_channels = level_channels[-1]

        self._build_conditioning()
        self._build_stem()
        self._build_encoder()
        self._build_bottleneck()
        self._build_decoder()
        self._build_head()

        logger.info(
            f"Created CliffordNetConditionalDenoiser "
            f"(levels={self.num_levels}, "
            f"channels={level_channels}, "
            f"blocks={level_blocks}, "
            f"shifts={level_shifts}, "
            f"dense_cond={enable_dense_conditioning}, "
            f"discrete_cond={enable_discrete_conditioning})"
        )

    # ------------------------------------------------------------------
    # Private builders
    # ------------------------------------------------------------------

    def _common_dense_kwargs(self) -> Dict[str, Any]:
        return dict(
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
        )

    def _common_conv_kwargs(self) -> Dict[str, Any]:
        return dict(
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
        )

    def _build_conditioning(self) -> None:
        """Build conditioning pathways."""
        # Dense conditioning encoder: per-level projection to match channels
        if self.enable_dense_conditioning:
            if self.dense_cond_channels is None:
                raise ValueError(
                    "dense_cond_channels required when "
                    "enable_dense_conditioning=True"
                )
            self.cond_projections = []
            for i in range(self.num_levels):
                proj = keras.layers.Conv2D(
                    filters=self.level_channels[i],
                    kernel_size=1,
                    padding="same",
                    name=f"cond_proj_level_{i}",
                    **self._common_conv_kwargs(),
                )
                self.cond_projections.append(proj)
            # Bottleneck projection
            self.cond_proj_bottleneck = keras.layers.Conv2D(
                filters=self.bottleneck_channels,
                kernel_size=1,
                padding="same",
                name="cond_proj_bottleneck",
                **self._common_conv_kwargs(),
            )

        # Discrete conditioning: embedding + projection
        if self.enable_discrete_conditioning:
            if self.num_classes <= 0:
                raise ValueError(
                    "num_classes must be positive when "
                    "enable_discrete_conditioning=True"
                )
            self.class_embedding = keras.layers.Embedding(
                input_dim=self.num_classes,
                output_dim=self.class_embedding_dim,
                embeddings_initializer="he_normal",
                name="class_embedding",
            )

    def _build_stem(self) -> None:
        """Bias-free stem: Conv2D stride-1 + BN(center=False)."""
        self.stem_conv = keras.layers.Conv2D(
            filters=self.level_channels[0],
            kernel_size=3,
            strides=1,
            padding="same",
            name="stem_conv",
            **self._common_conv_kwargs(),
        )
        self.stem_norm = keras.layers.BatchNormalization(
            center=False, name="stem_norm"
        )

    def _build_encoder(self) -> None:
        """Build encoder levels with conditioned CliffordNet blocks."""
        total_blocks = sum(self.level_blocks)
        drop_rates = _linear_drop_path_rates(
            total_blocks, self.stochastic_depth_rate
        )

        self.encoder_levels = []
        block_idx = 0

        for level in range(self.num_levels):
            ch = self.level_channels[level]
            shifts = self.level_shifts[level]
            n_blocks = self.level_blocks[level]

            # Channel transition conv (if not first level)
            if level > 0:
                transition = keras.layers.Conv2D(
                    filters=ch,
                    kernel_size=1,
                    padding="same",
                    name=f"enc_transition_{level}",
                    **self._common_conv_kwargs(),
                )
            else:
                transition = None

            # Blocks at this level
            blocks = []
            for b in range(n_blocks):
                block = BiasFreeConditionedCliffordBlock(
                    channels=ch,
                    shifts=shifts,
                    cli_mode=self.cli_mode,
                    ctx_mode=self.ctx_mode,
                    use_global_context=self.use_global_context,
                    layer_scale_init=self.layer_scale_init,
                    drop_path_rate=drop_rates[block_idx],
                    enable_dense_conditioning=self.enable_dense_conditioning,
                    enable_discrete_conditioning=(
                        self.enable_discrete_conditioning
                    ),
                    class_embedding_dim=self.class_embedding_dim,
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f"enc_block_{level}_{b}",
                )
                blocks.append(block)
                block_idx += 1

            # Downsample (except at last level — bottleneck handles it)
            if level < self.num_levels - 1:
                downsample = keras.layers.Conv2D(
                    filters=ch,
                    kernel_size=3,
                    strides=2,
                    padding="same",
                    name=f"enc_downsample_{level}",
                    **self._common_conv_kwargs(),
                )
            else:
                downsample = None

            self.encoder_levels.append({
                "transition": transition,
                "blocks": blocks,
                "downsample": downsample,
            })

    def _build_bottleneck(self) -> None:
        """Build bottleneck blocks at lowest resolution."""
        ch = self.bottleneck_channels
        shifts = self.level_shifts[-1]
        n_blocks = self.level_blocks[-1]
        total_blocks = sum(self.level_blocks)
        max_drop = self.stochastic_depth_rate

        self.bottleneck_blocks = []
        for b in range(n_blocks):
            rate = max_drop * (total_blocks - 1 + b) / max(
                total_blocks + n_blocks - 2, 1
            )
            block = BiasFreeConditionedCliffordBlock(
                channels=ch,
                shifts=shifts,
                cli_mode=self.cli_mode,
                ctx_mode=self.ctx_mode,
                use_global_context=True,
                layer_scale_init=self.layer_scale_init,
                drop_path_rate=rate,
                enable_dense_conditioning=self.enable_dense_conditioning,
                enable_discrete_conditioning=(
                    self.enable_discrete_conditioning
                ),
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name=f"bottleneck_block_{b}",
            )
            self.bottleneck_blocks.append(block)

    def _build_decoder(self) -> None:
        """Build decoder levels with skip connections."""
        self.decoder_levels = []

        for level in range(self.num_levels - 1, -1, -1):
            ch = self.level_channels[level]
            shifts = self.level_shifts[level]
            n_blocks = self.level_blocks[level]

            # Upsample
            if level < self.num_levels - 1:
                upsample = keras.layers.UpSampling2D(
                    size=2, interpolation="bilinear", name=f"dec_upsample_{level}"
                )
            else:
                upsample = None

            # Skip connection projection (concat encoder + decoder → ch)
            # After concat, channels = enc_ch + dec_ch
            if level < self.num_levels - 1:
                dec_ch_in = self.level_channels[level + 1] if level < self.num_levels - 1 else self.bottleneck_channels
            else:
                dec_ch_in = self.bottleneck_channels
            skip_proj = keras.layers.Conv2D(
                filters=ch,
                kernel_size=1,
                padding="same",
                name=f"dec_skip_proj_{level}",
                **self._common_conv_kwargs(),
            )

            # Blocks at this level
            blocks = []
            for b in range(n_blocks):
                block = BiasFreeConditionedCliffordBlock(
                    channels=ch,
                    shifts=shifts,
                    cli_mode=self.cli_mode,
                    ctx_mode=self.ctx_mode,
                    use_global_context=self.use_global_context,
                    layer_scale_init=self.layer_scale_init,
                    drop_path_rate=0.0,
                    enable_dense_conditioning=self.enable_dense_conditioning,
                    enable_discrete_conditioning=(
                        self.enable_discrete_conditioning
                    ),
                    class_embedding_dim=self.class_embedding_dim,
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f"dec_block_{level}_{b}",
                )
                blocks.append(block)

            self.decoder_levels.append({
                "upsample": upsample,
                "skip_proj": skip_proj,
                "blocks": blocks,
                "level": level,
            })

    def _build_head(self) -> None:
        """Bias-free output head: LayerNorm + 1x1 Conv (linear)."""
        self.head_norm = keras.layers.LayerNormalization(
            epsilon=self.LAYERNORM_EPSILON,
            center=False,
            name="head_norm",
        )
        self.output_proj = keras.layers.Conv2D(
            filters=self.in_channels,
            kernel_size=1,
            strides=1,
            padding="same",
            name="output_proj",
            **self._common_conv_kwargs(),
        )

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, input_shape) -> None:
        """Build model via symbolic forward pass.

        :param input_shape: Shape(s) of the input tensor(s).
        """
        self._build_input_shape = input_shape

        if isinstance(input_shape, list):
            target_shape = input_shape[0]
        else:
            target_shape = input_shape

        if len(target_shape) == 3:
            target_shape = (None,) + tuple(target_shape)
        else:
            target_shape = tuple(target_shape)

        inputs = [keras.KerasTensor(target_shape)]

        if self.enable_dense_conditioning:
            cond_shape = (
                target_shape[0],
                target_shape[1],
                target_shape[2],
                self.dense_cond_channels,
            )
            inputs.append(keras.KerasTensor(cond_shape))

        if self.enable_discrete_conditioning:
            inputs.append(keras.KerasTensor((target_shape[0], 1)))

        _ = self.call(inputs)
        super().build(input_shape)

    def get_build_config(self) -> Dict[str, Any]:
        if hasattr(self, "_build_input_shape"):
            shape = self._build_input_shape
            # Serialize: convert list of tuples for JSON compatibility
            if isinstance(shape, list):
                return {"input_shape": [list(s) for s in shape]}
            return {"input_shape": list(shape)}
        return {}

    def build_from_config(self, config: Dict[str, Any]) -> None:
        if "input_shape" in config:
            shape = config["input_shape"]
            if isinstance(shape, list) and shape and isinstance(shape[0], list):
                shape = [tuple(s) for s in shape]
            else:
                shape = tuple(shape)
            self.build(shape)

    # ------------------------------------------------------------------
    # Conditioning helpers
    # ------------------------------------------------------------------

    def _get_dense_cond_at_level(
        self,
        dense_cond: keras.KerasTensor,
        level: int,
        is_bottleneck: bool = False,
    ) -> keras.KerasTensor:
        """Downsample and project dense conditioning to match level."""
        # Downsample conditioning to match spatial resolution.
        # Stem is stride-1, so level 0 is at full res. Each encoder
        # level applies one stride-2 downsample, so we need `level`
        # pooling operations to match.
        cond = dense_cond
        for _ in range(level):
            cond = keras.layers.AveragePooling2D(
                pool_size=2, padding="same"
            )(cond)

        if is_bottleneck:
            return self.cond_proj_bottleneck(cond)
        return self.cond_projections[level](cond)

    def _get_discrete_cond(
        self, class_label: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Embed class label to conditioning vector."""
        emb = self.class_embedding(class_label)
        # Flatten from (B, 1, D) to (B, D)
        return keras.ops.squeeze(emb, axis=1)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def call(
        self,
        inputs: Union[keras.KerasTensor, List[keras.KerasTensor]],
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Forward pass: learns conditional noise residual.

        :param inputs: Either a single tensor (unconditional) or a list:
            - ``[noisy_target]``
            - ``[noisy_target, dense_cond]``
            - ``[noisy_target, class_label]`` (if only discrete)
            - ``[noisy_target, dense_cond, class_label]``
        :param training: Whether in training mode.
        :return: Denoised target ``(B, H, W, in_channels)``.
        """
        # Parse inputs
        if isinstance(inputs, (list, tuple)):
            noisy_target = inputs[0]
            idx = 1
            dense_cond = None
            class_label = None
            if self.enable_dense_conditioning and idx < len(inputs):
                dense_cond = inputs[idx]
                idx += 1
            if self.enable_discrete_conditioning and idx < len(inputs):
                class_label = inputs[idx]
                idx += 1
        else:
            noisy_target = inputs
            dense_cond = None
            class_label = None

        # Prepare conditioning
        discrete_emb = None
        if self.enable_discrete_conditioning and class_label is not None:
            discrete_emb = self._get_discrete_cond(class_label)

        # Stem: (B,H,W,C_in) → (B,H/2,W/2,ch[0])
        x = self.stem_conv(noisy_target)
        x = self.stem_norm(x, training=training)

        # Encoder
        skip_features = []
        for level_info in self.encoder_levels:
            level = self.encoder_levels.index(level_info)

            # Channel transition
            if level_info["transition"] is not None:
                x = level_info["transition"](x)

            # Get conditioning at this level
            level_dense_cond = None
            if (
                self.enable_dense_conditioning
                and dense_cond is not None
            ):
                level_dense_cond = self._get_dense_cond_at_level(
                    dense_cond, level
                )

            # Apply blocks
            for block in level_info["blocks"]:
                x = block(
                    x,
                    dense_cond=level_dense_cond,
                    discrete_cond=discrete_emb,
                    training=training,
                )

            # Save for skip connection
            skip_features.append(x)

            # Downsample
            if level_info["downsample"] is not None:
                x = level_info["downsample"](x)

        # Bottleneck
        bottle_dense_cond = None
        if self.enable_dense_conditioning and dense_cond is not None:
            bottle_dense_cond = self._get_dense_cond_at_level(
                dense_cond, self.num_levels - 1, is_bottleneck=True
            )
        for block in self.bottleneck_blocks:
            x = block(
                x,
                dense_cond=bottle_dense_cond,
                discrete_cond=discrete_emb,
                training=training,
            )

        # Decoder
        for dec_info in self.decoder_levels:
            level = dec_info["level"]

            # Upsample
            if dec_info["upsample"] is not None:
                x = dec_info["upsample"](x)

            # Skip connection: concatenate + project
            skip = skip_features[level]
            x = keras.ops.concatenate([x, skip], axis=-1)
            x = dec_info["skip_proj"](x)

            # Get conditioning at this level
            level_dense_cond = None
            if (
                self.enable_dense_conditioning
                and dense_cond is not None
            ):
                level_dense_cond = self._get_dense_cond_at_level(
                    dense_cond, level
                )

            # Apply blocks
            for block in dec_info["blocks"]:
                x = block(
                    x,
                    dense_cond=level_dense_cond,
                    discrete_cond=discrete_emb,
                    training=training,
                )

        # Head: LayerNorm + linear 1x1 projection
        x = self.head_norm(x)
        residual = self.output_proj(x)

        # Residual learning: denoised = noisy + learned_residual
        return noisy_target + residual

    # ------------------------------------------------------------------
    # Shape inference
    # ------------------------------------------------------------------

    def compute_output_shape(
        self, input_shape
    ) -> Tuple[Optional[int], ...]:
        if isinstance(input_shape, list):
            return input_shape[0]
        return input_shape

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "in_channels": self.in_channels,
            "level_channels": self.level_channels,
            "level_blocks": self.level_blocks,
            "level_shifts": self.level_shifts,
            "cli_mode": self.cli_mode,
            "ctx_mode": self.ctx_mode,
            "use_global_context": self.use_global_context,
            "layer_scale_init": self.layer_scale_init,
            "stochastic_depth_rate": self.stochastic_depth_rate,
            "enable_dense_conditioning": self.enable_dense_conditioning,
            "dense_cond_channels": self.dense_cond_channels,
            "enable_discrete_conditioning": (
                self.enable_discrete_conditioning
            ),
            "num_classes": self.num_classes,
            "class_embedding_dim": self.class_embedding_dim,
            "kernel_initializer": initializers.serialize(
                self.kernel_initializer
            ),
            "kernel_regularizer": regularizers.serialize(
                self.kernel_regularizer
            ),
        })
        return config

    @classmethod
    def from_config(
        cls, config: Dict[str, Any]
    ) -> "CliffordNetConditionalDenoiser":
        for key in ("kernel_regularizer",):
            if config.get(key) and isinstance(config[key], dict):
                config[key] = regularizers.deserialize(config[key])
        return cls(**config)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self, **kwargs: Any) -> None:
        if not self.built:
            logger.warning(
                "Model not built; call build() with input shape first."
            )
        super().summary(**kwargs)
        logger.info("CliffordNetConditionalDenoiser configuration:")
        logger.info(f"  in_channels         : {self.in_channels}")
        logger.info(f"  level_channels      : {self.level_channels}")
        logger.info(f"  level_blocks        : {self.level_blocks}")
        logger.info(f"  level_shifts        : {self.level_shifts}")
        logger.info(f"  cli_mode            : {self.cli_mode}")
        logger.info(f"  ctx_mode            : {self.ctx_mode}")
        logger.info(f"  use_global_context  : {self.use_global_context}")
        logger.info(f"  dense_conditioning  : {self.enable_dense_conditioning}")
        logger.info(
            f"  discrete_conditioning: {self.enable_discrete_conditioning}"
        )
        logger.info(f"  bias_free           : True")

    # ------------------------------------------------------------------
    # Factory class methods
    # ------------------------------------------------------------------

    MODEL_VARIANTS: Dict[str, Dict[str, Any]] = {
        "tiny": dict(
            level_channels=[64, 96, 128],
            level_blocks=[2, 2, 2],
            level_shifts=[[1, 2], [1, 2, 4], [1, 2, 4]],
            cli_mode="full",
            ctx_mode="diff",
            use_global_context=False,
            layer_scale_init=1e-5,
            kernel_initializer=_DEFAULT_KERNEL_INIT,
        ),
        "small": dict(
            level_channels=[64, 128, 192],
            level_blocks=[2, 3, 3],
            level_shifts=[[1, 2], [1, 2, 4], [1, 2, 4, 8]],
            cli_mode="full",
            ctx_mode="diff",
            use_global_context=False,
            layer_scale_init=1e-5,
            kernel_initializer=_DEFAULT_KERNEL_INIT,
        ),
        "base": dict(
            level_channels=[96, 192, 256, 256],
            level_blocks=[2, 3, 4, 2],
            level_shifts=[[1, 2], [1, 2, 4], [1, 2, 4, 8], [1, 2, 4]],
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
        enable_dense_conditioning: bool = False,
        dense_cond_channels: Optional[int] = None,
        enable_discrete_conditioning: bool = False,
        num_classes: int = 0,
        class_embedding_dim: int = 128,
        **kwargs: Any,
    ) -> "CliffordNetConditionalDenoiser":
        """Create from a predefined variant.

        :param variant: One of ``"tiny"``, ``"small"``, ``"base"``.
        :param in_channels: Number of input channels.
        :param enable_dense_conditioning: Enable dense conditioning.
        :param dense_cond_channels: Dense conditioning channel count.
        :param enable_discrete_conditioning: Enable discrete conditioning.
        :param num_classes: Number of classes for discrete conditioning.
        :param class_embedding_dim: Class embedding dimension.
        :param kwargs: Override any default hyperparameter.
        :return: Configured model instance.
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                f"Available: {list(cls.MODEL_VARIANTS.keys())}"
            )
        defaults = dict(cls.MODEL_VARIANTS[variant])
        defaults.update(kwargs)
        logger.info(
            f"Creating CliffordNetConditionalDenoiser-{variant.upper()}"
        )
        return cls(
            in_channels=in_channels,
            enable_dense_conditioning=enable_dense_conditioning,
            dense_cond_channels=dense_cond_channels,
            enable_discrete_conditioning=enable_discrete_conditioning,
            num_classes=num_classes,
            class_embedding_dim=class_embedding_dim,
            **defaults,
        )

    @classmethod
    def tiny(
        cls, in_channels: int = 1, **kwargs: Any
    ) -> "CliffordNetConditionalDenoiser":
        """Tiny variant: 3 levels, ~2M params."""
        return cls.from_variant("tiny", in_channels=in_channels, **kwargs)

    @classmethod
    def small(
        cls, in_channels: int = 1, **kwargs: Any
    ) -> "CliffordNetConditionalDenoiser":
        """Small variant: 3 levels, ~5M params."""
        return cls.from_variant("small", in_channels=in_channels, **kwargs)

    @classmethod
    def base(
        cls, in_channels: int = 1, **kwargs: Any
    ) -> "CliffordNetConditionalDenoiser":
        """Base variant: 4 levels, ~12M params."""
        return cls.from_variant("base", in_channels=in_channels, **kwargs)
