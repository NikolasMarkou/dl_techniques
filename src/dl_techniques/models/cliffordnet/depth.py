"""
CliffordNet monocular depth estimation model.

A U-Net encoder-decoder for monocular depth estimation built on
CliffordNet geometric-algebra blocks.  Unlike the conditional denoiser,
this model uses **standard (with-bias)** CliffordNet blocks and predicts
depth directly from RGB — no denoising formulation, no conditioning.

Architecture::

    Input: RGB (B, H, W, 3)
        |
    Stem: Conv2D(3→ch[0], k=3, s=1) + BN
        |
    Encoder levels 0..N-1:
        [CliffordNetBlock × blocks_per_level] + Conv2D(s=2) downsample
        |
    Bottleneck:
        [CliffordNetBlock × blocks_per_level] (use_global_context=True)
        |
    Decoder levels N-1..0:
        UpSampling2D + skip concat + Conv2D 1×1 proj
        + [CliffordNetBlock × blocks_per_level]
        |
    Head: LayerNorm → Conv2D 1×1 (linear, with bias)
        |
    Output: depth (B, H, W, 1)

Pre-defined variants
--------------------
- ``CliffordNetDepthEstimator.tiny``   -- 3 levels, ~2M params
- ``CliffordNetDepthEstimator.small``  -- 3 levels, ~5M params
- ``CliffordNetDepthEstimator.base``   -- 4 levels, ~12M params
- ``CliffordNetDepthEstimator.large``  -- 4 levels, ~40M params
- ``CliffordNetDepthEstimator.xlarge`` -- 4 levels, ~120M params
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import keras
from keras import initializers, regularizers

from dl_techniques.layers.geometric.clifford_block import (
    CliMode,
    CtxMode,
    CliffordNetBlock,
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
# CliffordNetDepthEstimator
# ===========================================================================


@keras.saving.register_keras_serializable()
class CliffordNetDepthEstimator(keras.Model):
    """U-Net depth estimator built on CliffordNet geometric-algebra blocks.

    All layers use bias (``use_bias=True``) and standard normalization
    (LayerNorm/BatchNorm with centering).  This is a direct regression
    model — no denoising formulation, no residual output add.

    :param in_channels: Number of input channels (3 for RGB).
    :param out_channels: Number of output channels (1 for depth).
    :param level_channels: Channel count at each encoder level.
    :param level_blocks: Number of CliffordNet blocks per level.
    :param level_shifts: Shift patterns per level (list of lists).
    :param cli_mode: Geometric product mode.
    :param ctx_mode: Context mode.
    :param use_global_context: Add global branch at each level.
    :param layer_scale_init: Initial LayerScale value.
    :param stochastic_depth_rate: Maximum drop-path rate.
    :param upsample_interpolation: Interpolation for UpSampling2D.
    :param kernel_initializer: Kernel initializer.
    :param kernel_regularizer: Kernel regularizer.
    """

    LAYERNORM_EPSILON: float = 1e-6

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        level_channels: Optional[List[int]] = None,
        level_blocks: Optional[List[int]] = None,
        level_shifts: Optional[List[List[int]]] = None,
        cli_mode: CliMode = "full",
        ctx_mode: CtxMode = "diff",
        use_global_context: bool = False,
        layer_scale_init: float = 1e-5,
        stochastic_depth_rate: float = 0.0,
        upsample_interpolation: str = "nearest",
        kernel_initializer: Any = "glorot_uniform",
        kernel_regularizer: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if in_channels <= 0:
            raise ValueError(
                f"in_channels must be positive, got {in_channels}"
            )
        if out_channels <= 0:
            raise ValueError(
                f"out_channels must be positive, got {out_channels}"
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
        self.out_channels = out_channels
        self.level_channels = list(level_channels)
        self.level_blocks = list(level_blocks)
        self.level_shifts = [list(s) for s in level_shifts]
        self.cli_mode = cli_mode
        self.ctx_mode = ctx_mode
        self.use_global_context = use_global_context
        self.layer_scale_init = layer_scale_init
        self.stochastic_depth_rate = stochastic_depth_rate
        self.upsample_interpolation = upsample_interpolation
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

        self.num_levels = len(level_channels)
        self.bottleneck_channels = level_channels[-1]

        self._build_stem()
        self._build_encoder()
        self._build_bottleneck()
        self._build_decoder()
        self._build_head()

        logger.info(
            f"Created CliffordNetDepthEstimator "
            f"(levels={self.num_levels}, "
            f"channels={level_channels}, "
            f"blocks={level_blocks}, "
            f"shifts={level_shifts})"
        )

    # ------------------------------------------------------------------
    # Private builders
    # ------------------------------------------------------------------

    def _common_conv_kwargs(self) -> Dict[str, Any]:
        return dict(
            use_bias=True,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
        )

    def _build_stem(self) -> None:
        """Stem: Conv2D stride-1 + BatchNorm."""
        self.stem_conv = keras.layers.Conv2D(
            filters=self.level_channels[0],
            kernel_size=3,
            strides=1,
            padding="same",
            name="stem_conv",
            **self._common_conv_kwargs(),
        )
        self.stem_norm = keras.layers.BatchNormalization(name="stem_norm")

    def _build_encoder(self) -> None:
        """Build encoder levels with CliffordNet blocks."""
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

            # CliffordNet blocks at this level
            blocks = []
            for b in range(n_blocks):
                block = CliffordNetBlock(
                    channels=ch,
                    shifts=shifts,
                    cli_mode=self.cli_mode,
                    ctx_mode=self.ctx_mode,
                    use_global_context=self.use_global_context,
                    layer_scale_init=self.layer_scale_init,
                    drop_path_rate=drop_rates[block_idx],
                    use_bias=True,
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
            block = CliffordNetBlock(
                channels=ch,
                shifts=shifts,
                cli_mode=self.cli_mode,
                ctx_mode=self.ctx_mode,
                use_global_context=True,  # Always global at bottleneck
                layer_scale_init=self.layer_scale_init,
                drop_path_rate=rate,
                use_bias=True,
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

            # Upsample (except at last/highest level)
            if level < self.num_levels - 1:
                upsample = keras.layers.UpSampling2D(
                    size=2,
                    interpolation=self.upsample_interpolation,
                    name=f"dec_upsample_{level}",
                )
            else:
                upsample = None

            # Skip connection projection (concat encoder + decoder → ch)
            skip_proj = keras.layers.Conv2D(
                filters=ch,
                kernel_size=1,
                padding="same",
                name=f"dec_skip_proj_{level}",
                **self._common_conv_kwargs(),
            )

            # CliffordNet blocks at this level
            blocks = []
            for b in range(n_blocks):
                block = CliffordNetBlock(
                    channels=ch,
                    shifts=shifts,
                    cli_mode=self.cli_mode,
                    ctx_mode=self.ctx_mode,
                    use_global_context=self.use_global_context,
                    layer_scale_init=self.layer_scale_init,
                    drop_path_rate=0.0,  # No drop-path in decoder
                    use_bias=True,
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
        """Output head: LayerNorm + 1x1 Conv (linear, with bias)."""
        self.head_norm = keras.layers.LayerNormalization(
            epsilon=self.LAYERNORM_EPSILON,
            name="head_norm",
        )
        self.output_proj = keras.layers.Conv2D(
            filters=self.out_channels,
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

        :param input_shape: Shape of the input tensor ``(B, H, W, C)``.
        """
        self._build_input_shape = input_shape

        if isinstance(input_shape, (list, tuple)) and isinstance(
            input_shape[0], (list, tuple)
        ):
            target_shape = input_shape[0]
        else:
            target_shape = input_shape

        if len(target_shape) == 3:
            target_shape = (None,) + tuple(target_shape)
        else:
            target_shape = tuple(target_shape)

        inputs = keras.KerasTensor(target_shape)
        _ = self.call(inputs)
        super().build(input_shape)

    def get_build_config(self) -> Dict[str, Any]:
        if hasattr(self, "_build_input_shape"):
            shape = self._build_input_shape
            if isinstance(shape, list):
                return {"input_shape": [list(s) for s in shape]}
            return {"input_shape": list(shape)}
        return {}

    def build_from_config(self, config: Dict[str, Any]) -> None:
        if "input_shape" in config:
            shape = config["input_shape"]
            if isinstance(shape, list) and shape and isinstance(
                shape[0], list
            ):
                shape = [tuple(s) for s in shape]
            else:
                shape = tuple(shape)
            self.build(shape)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Forward pass: RGB → depth.

        :param inputs: RGB tensor ``(B, H, W, in_channels)``.
        :param training: Whether in training mode.
        :return: Depth map ``(B, H, W, out_channels)``.
        """
        # Stem: (B,H,W,3) → (B,H,W,ch[0])
        x = self.stem_conv(inputs)
        x = self.stem_norm(x, training=training)

        # Encoder
        skip_features = []
        for level, level_info in enumerate(self.encoder_levels):
            # Channel transition
            if level_info["transition"] is not None:
                x = level_info["transition"](x)

            # Apply CliffordNet blocks
            for block in level_info["blocks"]:
                x = block(x, training=training)

            # Save for skip connection
            skip_features.append(x)

            # Downsample
            if level_info["downsample"] is not None:
                x = level_info["downsample"](x)

        # Bottleneck
        for block in self.bottleneck_blocks:
            x = block(x, training=training)

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

            # Apply CliffordNet blocks
            for block in dec_info["blocks"]:
                x = block(x, training=training)

        # Head: LayerNorm + linear 1x1 projection → depth
        x = self.head_norm(x)
        depth = self.output_proj(x)

        return depth

    # ------------------------------------------------------------------
    # Shape inference
    # ------------------------------------------------------------------

    def compute_output_shape(
        self, input_shape
    ) -> Tuple[Optional[int], ...]:
        if isinstance(input_shape, list):
            s = input_shape[0]
        else:
            s = input_shape
        return s[:-1] + (self.out_channels,)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "level_channels": self.level_channels,
            "level_blocks": self.level_blocks,
            "level_shifts": self.level_shifts,
            "cli_mode": self.cli_mode,
            "ctx_mode": self.ctx_mode,
            "use_global_context": self.use_global_context,
            "layer_scale_init": self.layer_scale_init,
            "stochastic_depth_rate": self.stochastic_depth_rate,
            "upsample_interpolation": self.upsample_interpolation,
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
    ) -> "CliffordNetDepthEstimator":
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
        logger.info("CliffordNetDepthEstimator configuration:")
        logger.info(f"  in_channels         : {self.in_channels}")
        logger.info(f"  out_channels        : {self.out_channels}")
        logger.info(f"  level_channels      : {self.level_channels}")
        logger.info(f"  level_blocks        : {self.level_blocks}")
        logger.info(f"  level_shifts        : {self.level_shifts}")
        logger.info(f"  cli_mode            : {self.cli_mode}")
        logger.info(f"  ctx_mode            : {self.ctx_mode}")
        logger.info(f"  use_global_context  : {self.use_global_context}")

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
            level_channels=[64, 128, 256, 256],
            level_blocks=[2, 2, 2, 2],
            level_shifts=[[1, 2], [1, 2, 4], [1, 2, 4, 8], [1, 2, 4]],
            cli_mode="full",
            ctx_mode="diff",
            use_global_context=True,
            layer_scale_init=1e-5,
            kernel_initializer=_DEFAULT_KERNEL_INIT,
        ),
        "large": dict(
            level_channels=[96, 192, 320, 384],
            level_blocks=[3, 4, 5, 3],
            level_shifts=[[1, 2], [1, 2, 4], [1, 2, 4, 8], [1, 2, 4, 8]],
            cli_mode="full",
            ctx_mode="diff",
            use_global_context=True,
            layer_scale_init=1e-5,
            stochastic_depth_rate=0.1,
            kernel_initializer=_DEFAULT_KERNEL_INIT,
        ),
        "xlarge": dict(
            level_channels=[160, 320, 448, 640],
            level_blocks=[2, 4, 6, 3],
            level_shifts=[
                [1, 2], [1, 2, 4], [1, 2, 4, 8], [1, 2, 4, 8, 16],
            ],
            cli_mode="full",
            ctx_mode="diff",
            use_global_context=True,
            layer_scale_init=1e-5,
            stochastic_depth_rate=0.15,
            kernel_initializer=_DEFAULT_KERNEL_INIT,
        ),
    }

    @classmethod
    def from_variant(
        cls,
        variant: str,
        in_channels: int = 3,
        out_channels: int = 1,
        **kwargs: Any,
    ) -> "CliffordNetDepthEstimator":
        """Create from a predefined variant.

        :param variant: One of ``"tiny"``, ``"small"``, ``"base"``,
            ``"large"``, ``"xlarge"``.
        :param in_channels: Number of input channels (default 3 for RGB).
        :param out_channels: Number of output channels (default 1 for
            depth).
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
            f"Creating CliffordNetDepthEstimator-{variant.upper()}"
        )
        return cls(
            in_channels=in_channels,
            out_channels=out_channels,
            **defaults,
        )

    @classmethod
    def tiny(
        cls, in_channels: int = 3, out_channels: int = 1, **kwargs: Any
    ) -> "CliffordNetDepthEstimator":
        """Tiny variant: 3 levels, ~2M params."""
        return cls.from_variant(
            "tiny", in_channels=in_channels,
            out_channels=out_channels, **kwargs,
        )

    @classmethod
    def small(
        cls, in_channels: int = 3, out_channels: int = 1, **kwargs: Any
    ) -> "CliffordNetDepthEstimator":
        """Small variant: 3 levels, ~5M params."""
        return cls.from_variant(
            "small", in_channels=in_channels,
            out_channels=out_channels, **kwargs,
        )

    @classmethod
    def base(
        cls, in_channels: int = 3, out_channels: int = 1, **kwargs: Any
    ) -> "CliffordNetDepthEstimator":
        """Base variant: 4 levels, ~12M params."""
        return cls.from_variant(
            "base", in_channels=in_channels,
            out_channels=out_channels, **kwargs,
        )

    @classmethod
    def large(
        cls, in_channels: int = 3, out_channels: int = 1, **kwargs: Any
    ) -> "CliffordNetDepthEstimator":
        """Large variant: 4 levels, ~40M params."""
        return cls.from_variant(
            "large", in_channels=in_channels,
            out_channels=out_channels, **kwargs,
        )

    @classmethod
    def xlarge(
        cls, in_channels: int = 3, out_channels: int = 1, **kwargs: Any
    ) -> "CliffordNetDepthEstimator":
        """Extra-large variant: 4 levels, ~120M params."""
        return cls.from_variant(
            "xlarge", in_channels=in_channels,
            out_channels=out_channels, **kwargs,
        )
