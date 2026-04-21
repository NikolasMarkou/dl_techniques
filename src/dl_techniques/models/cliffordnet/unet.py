"""
CliffordNetUNet — generic multi-head U-Net built on CliffordNet geometric-algebra blocks.

A single backbone (stem → encoder → bottleneck → decoder) with a declarative
head specification.  Heads are attached at configurable *taps* (decoder levels
or the bottleneck).  Dense/spatial heads support always-on deep supervision:
when enabled, one auxiliary head of the same type is attached at every decoder
level deeper than the primary tap.

Tap semantics
-------------
- ``tap = k`` (int, ``0 <= k < N``): decoder output at level ``k``.  Level 0 is
  full-resolution; level ``N-1`` is the coarsest decoder output.
- ``tap = "bottleneck"``: post-bottleneck features (same spatial resolution as
  level ``N-1`` but before any decoder block).  Primarily for classification.

Head specification
------------------
::

    head_configs = {
        "classification": {"type": "classification", "tap": "bottleneck",
                           "num_classes": 80},
        "segmentation":   {"type": "segmentation", "tap": 0,
                           "num_classes": 81, "deep_supervision": True},
    }

Forward pass returns ``Dict[str, KerasTensor]`` keyed by head name with
``{name}_aux_{level}`` entries for each aux head when deep supervision is on.

Use cases
---------
- Depth estimation — :func:`create_cliffordnet_depth`.
- COCO multi-task (classification + semantic segmentation) — see
  ``src/train/cliffordnet/train_coco_multitask.py``.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple, Union

import keras
from keras import initializers, regularizers

from dl_techniques.layers.geometric.clifford_block import (
    CliMode,
    CtxMode,
    CliffordNetBlock,
)
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------------
# Constants / types
# ---------------------------------------------------------------------------

Tap = Union[int, str, List[int]]
"""int = decoder level; ``"bottleneck"`` = post-bottleneck features; ``list[int]`` = multi-tap (FPN-style, for detection)."""

_BOTTLENECK_TAP: str = "bottleneck"
_SPATIAL_HEAD_TYPES: Tuple[str, ...] = ("segmentation", "depth", "spatial")
_POOLED_HEAD_TYPES: Tuple[str, ...] = ("classification",)
_DETECTION_HEAD_TYPES: Tuple[str, ...] = ("detection",)

# Match CliffordNet reference: trunc_normal_(std=0.02)
_DEFAULT_KERNEL_INIT = initializers.TruncatedNormal(stddev=0.02)


def _linear_drop_path_rates(num_blocks: int, max_rate: float) -> List[float]:
    """Linearly spaced drop-path rates from 0 to *max_rate*."""
    if num_blocks <= 1:
        return [0.0] * num_blocks
    step = max_rate / (num_blocks - 1)
    return [round(i * step, 6) for i in range(num_blocks)]


# ===========================================================================
# Inline head blocks
# ===========================================================================
#
# DECISION D-001 — Minimal inline heads rather than reusing
# ``dl_techniques.layers.vision_heads.factory``.  The factory heads are
# transformer-oriented (optional attention/FFN stacks) and ``MultiTaskHead``
# stores children in a plain ``dict`` that Keras does not track for weight
# serialization.  A small, focused head keeps parity with the existing depth
# head (LayerNorm + 1×1 Conv) and keeps serialization straightforward.


@keras.saving.register_keras_serializable()
class _ClassificationHeadBlock(keras.layers.Layer):
    """Pooled classification head: GAP → LayerNorm → (Dropout) → (Dense hidden) → Dense logits.

    Returns **logits** — activation (softmax / sigmoid) is applied by the loss.

    :param num_classes: Number of output classes.
    :param dropout: Dropout applied after norm. 0.0 disables.
    :param hidden_dim: Optional hidden Dense + GELU before logits. None disables.
    :param kernel_initializer: Kernel initializer for Dense layers.
    :param kernel_regularizer: Kernel regularizer for Dense layers.
    :param use_bias: Whether Dense layers use bias.
    """

    def __init__(
        self,
        num_classes: int,
        dropout: float = 0.0,
        hidden_dim: Optional[int] = None,
        kernel_initializer: Any = None,
        kernel_regularizer: Any = None,
        use_bias: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")

        self.num_classes = num_classes
        self.dropout_rate = float(dropout)
        self.hidden_dim = hidden_dim
        self.kernel_initializer = (
            initializers.get(kernel_initializer)
            if kernel_initializer is not None
            else _DEFAULT_KERNEL_INIT
        )
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.use_bias = use_bias

        self.gap = keras.layers.GlobalAveragePooling2D(name="gap")
        self.norm = keras.layers.LayerNormalization(epsilon=1e-6, name="norm")
        self.drop = (
            keras.layers.Dropout(self.dropout_rate, name="dropout")
            if self.dropout_rate > 0.0
            else None
        )
        if hidden_dim is not None and hidden_dim > 0:
            self.hidden = keras.layers.Dense(
                hidden_dim,
                activation="gelu",
                use_bias=use_bias,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name="hidden",
            )
        else:
            self.hidden = None
        self.logits = keras.layers.Dense(
            num_classes,
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="logits",
        )

    def call(self, x: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        x = self.gap(x)
        x = self.norm(x)
        if self.drop is not None:
            x = self.drop(x, training=training)
        if self.hidden is not None:
            x = self.hidden(x)
        return self.logits(x)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "dropout": self.dropout_rate,
                "hidden_dim": self.hidden_dim,
                "kernel_initializer": initializers.serialize(self.kernel_initializer),
                "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
                "use_bias": self.use_bias,
            }
        )
        return config


@keras.saving.register_keras_serializable()
class _SpatialHeadBlock(keras.layers.Layer):
    """Dense-prediction head: LayerNorm → (optional Conv3×3 GELU) → Conv1×1(out_channels).

    Returns **raw values** — activation (softmax / sigmoid / linear depth) is
    applied by the loss / caller.  Matches the existing depth head style.

    :param out_channels: Number of output channels (1 for depth, N for segmentation).
    :param hidden_dim: Optional hidden Conv3×3 + GELU.  None → LayerNorm + Conv1×1 only.
    :param kernel_initializer: Kernel initializer for Conv layers.
    :param kernel_regularizer: Kernel regularizer for Conv layers.
    :param use_bias: Whether Conv layers use bias.
    """

    def __init__(
        self,
        out_channels: int,
        hidden_dim: Optional[int] = None,
        kernel_initializer: Any = None,
        kernel_regularizer: Any = None,
        use_bias: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}")

        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.kernel_initializer = (
            initializers.get(kernel_initializer)
            if kernel_initializer is not None
            else _DEFAULT_KERNEL_INIT
        )
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.use_bias = use_bias

        self.norm = keras.layers.LayerNormalization(epsilon=1e-6, name="norm")
        if hidden_dim is not None and hidden_dim > 0:
            self.hidden = keras.layers.Conv2D(
                filters=hidden_dim,
                kernel_size=3,
                padding="same",
                activation="gelu",
                use_bias=use_bias,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name="hidden",
            )
        else:
            self.hidden = None
        self.proj = keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=1,
            padding="same",
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="proj",
        )

    def call(self, x: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        x = self.norm(x)
        if self.hidden is not None:
            x = self.hidden(x)
        return self.proj(x)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "out_channels": self.out_channels,
                "hidden_dim": self.hidden_dim,
                "kernel_initializer": initializers.serialize(self.kernel_initializer),
                "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
                "use_bias": self.use_bias,
            }
        )
        return config


# ---------------------------------------------------------------------------
# Detection head block — thin wrapper around YOLOv12DetectionHead
# ---------------------------------------------------------------------------
#
# Accepts a ``List[KerasTensor]`` ordered shallow-to-deep (smallest-stride
# first, matching YOLO FPN convention P3→P4→P5).  Delegates entirely to
# :class:`YOLOv12DetectionHead` — we do not reimplement DFL / head convs.
# Output: ``(B, total_anchors, 4*reg_max + num_classes)`` rank-3 logits.
#
# Stride-agnostic — anchor layout is the loss's concern, not the head's.


@keras.saving.register_keras_serializable()
class _DetectionHeadBlock(keras.layers.Layer):
    """FPN-style detection head wrapping :class:`YOLOv12DetectionHead`.

    :param num_classes: Number of detection classes (80 for COCO).
    :param reg_max: DFL regression bins per box edge (YOLO default 16).
    :param kernel_initializer: Forwarded to the inner head's conv blocks.
    :param kernel_regularizer: Forwarded to the inner head's conv blocks.
    :param use_bias: Present for API parity with sibling heads; not used by YOLOv12DetectionHead.
    """

    def __init__(
        self,
        num_classes: int,
        reg_max: int = 16,
        kernel_initializer: Any = None,
        kernel_regularizer: Any = None,
        use_bias: bool = True,  # noqa: ARG002 — kept for API parity
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if reg_max <= 0:
            raise ValueError(f"reg_max must be positive, got {reg_max}")
        self.num_classes = int(num_classes)
        self.reg_max = int(reg_max)
        self.kernel_initializer = (
            initializers.get(kernel_initializer)
            if kernel_initializer is not None
            else _DEFAULT_KERNEL_INIT
        )
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

        # Lazy import to avoid pulling yolo12 deps unless detection is used.
        from dl_techniques.layers.yolo12_heads import YOLOv12DetectionHead

        self._det_head = YOLOv12DetectionHead(
            num_classes=self.num_classes,
            reg_max=self.reg_max,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="det_head",
        )

    def call(
        self, inputs: List[keras.KerasTensor], training: Optional[bool] = None
    ) -> keras.KerasTensor:
        if not isinstance(inputs, (list, tuple)):
            raise ValueError(
                f"_DetectionHeadBlock expects a list/tuple of feature tensors, "
                f"got {type(inputs).__name__}"
            )
        return self._det_head(list(inputs), training=training)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "reg_max": self.reg_max,
                "kernel_initializer": initializers.serialize(self.kernel_initializer),
                "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            }
        )
        return config


# ===========================================================================
# CliffordNetUNet
# ===========================================================================


@keras.saving.register_keras_serializable()
class CliffordNetUNet(keras.Model):
    """Generic multi-head U-Net built on CliffordNet geometric-algebra blocks.

    Backbone: stem (Conv 7×7 + BN) → encoder (N levels of CliffordNetBlocks
    with stride-2 downsample) → bottleneck (CliffordNetBlocks with global
    context) → decoder (UpSample + skip concat + 1×1 proj + CliffordNetBlocks).

    Heads are attached at taps declared by ``head_configs``.  When
    ``head_configs`` is empty, :meth:`call` returns raw per-level features:
    ``{"level_0": ..., ..., "level_{N-1}": ..., "bottleneck": ...}``.

    :param in_channels: Input channels (3 for RGB).
    :param level_channels: Channel count per encoder/decoder level.
    :param level_blocks: Number of CliffordNet blocks per level.
    :param level_shifts: Shift patterns per level (one list per level).
    :param cli_mode: Geometric product mode.
    :param ctx_mode: Context mode.
    :param use_global_context: Global branch at each non-bottleneck level.
    :param layer_scale_init: LayerScale init value inside CliffordNetBlock.
    :param stochastic_depth_rate: Max linearly-scaled drop-path rate across encoder + bottleneck.
    :param upsample_interpolation: Interpolation for decoder UpSampling2D.
    :param kernel_initializer: Kernel initializer for Conv / Dense.
    :param kernel_regularizer: Kernel regularizer for Conv / Dense.
    :param head_configs: Declarative head specification.  See module docstring.
        Keys are head names; each value is a dict with ``type``, ``tap``, and
        type-specific fields (``num_classes`` or ``out_channels``,
        ``deep_supervision``, ``dropout``, ``hidden_dim``).
    """

    LAYERNORM_EPSILON: float = 1e-6

    def __init__(
        self,
        in_channels: int = 3,
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
        head_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")

        if level_channels is None:
            level_channels = [64, 128, 256]
        if level_blocks is None:
            level_blocks = [2, 2, 4]
        if level_shifts is None:
            level_shifts = [[1, 2], [1, 2, 4], [1, 2, 4, 8]]

        if len(level_channels) != len(level_blocks):
            raise ValueError(
                f"level_channels ({len(level_channels)}) and level_blocks "
                f"({len(level_blocks)}) must have the same length"
            )
        if len(level_shifts) != len(level_channels):
            raise ValueError(
                f"level_shifts ({len(level_shifts)}) and level_channels "
                f"({len(level_channels)}) must have the same length"
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
        self.upsample_interpolation = upsample_interpolation
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

        self.num_levels = len(level_channels)
        self.bottleneck_channels = level_channels[-1]

        # Validate and normalize head_configs
        self.head_configs: Dict[str, Dict[str, Any]] = self._normalize_head_configs(
            head_configs
        )

        self._build_stem()
        self._build_encoder()
        self._build_bottleneck()
        self._build_decoder()
        self._build_heads()

        logger.info(
            f"Created CliffordNetUNet "
            f"(levels={self.num_levels}, channels={level_channels}, "
            f"blocks={level_blocks}, heads={list(self.head_configs.keys())})"
        )

    # ------------------------------------------------------------------
    # Head config validation
    # ------------------------------------------------------------------

    def _normalize_head_configs(
        self, head_configs: Optional[Dict[str, Dict[str, Any]]]
    ) -> Dict[str, Dict[str, Any]]:
        if head_configs is None:
            return {}
        if not isinstance(head_configs, dict):
            raise TypeError(
                f"head_configs must be Dict[str, Dict[str, Any]], "
                f"got {type(head_configs).__name__}"
            )
        normalized: Dict[str, Dict[str, Any]] = {}
        for name, spec in head_configs.items():
            if not isinstance(spec, dict):
                raise TypeError(
                    f"head_configs['{name}'] must be a dict, "
                    f"got {type(spec).__name__}"
                )
            spec = dict(spec)
            htype = spec.get("type")
            _ALL_HEAD_TYPES = (
                _SPATIAL_HEAD_TYPES + _POOLED_HEAD_TYPES + _DETECTION_HEAD_TYPES
            )
            if htype not in _ALL_HEAD_TYPES:
                raise ValueError(
                    f"head_configs['{name}'].type must be one of "
                    f"{_ALL_HEAD_TYPES}, got {htype!r}"
                )
            tap = spec.get("tap")
            if isinstance(tap, int):
                if not 0 <= tap < self.num_levels:
                    raise ValueError(
                        f"head_configs['{name}'].tap={tap} out of range "
                        f"[0, {self.num_levels})"
                    )
            elif isinstance(tap, list):
                if len(tap) < 2:
                    raise ValueError(
                        f"head_configs['{name}'].tap list must have >= 2 elements, "
                        f"got {tap}"
                    )
                for t in tap:
                    if not isinstance(t, int) or not 0 <= t < self.num_levels:
                        raise ValueError(
                            f"head_configs['{name}'].tap list element {t!r} "
                            f"is not an int in [0, {self.num_levels})"
                        )
                if len(set(tap)) != len(tap):
                    raise ValueError(
                        f"head_configs['{name}'].tap list has duplicate levels: {tap}"
                    )
            elif tap == _BOTTLENECK_TAP:
                pass
            else:
                raise ValueError(
                    f"head_configs['{name}'].tap must be int in "
                    f"[0, {self.num_levels}), {_BOTTLENECK_TAP!r}, or list[int], got {tap!r}"
                )
            if htype in _POOLED_HEAD_TYPES and tap != _BOTTLENECK_TAP:
                logger.warning(
                    f"head_configs['{name}']: pooled head at tap={tap!r} is "
                    f"unusual; bottleneck is the recommended tap."
                )
            if htype in _DETECTION_HEAD_TYPES:
                if not isinstance(tap, list):
                    raise ValueError(
                        f"head_configs['{name}']: detection heads require "
                        f"tap=list[int] (FPN-style multi-tap), got {tap!r}"
                    )
                if len(tap) != 3:
                    raise ValueError(
                        f"head_configs['{name}']: detection heads require "
                        f"exactly 3 taps (YOLOv12DetectionHead constraint), "
                        f"got {len(tap)}"
                    )
            ds = bool(spec.get("deep_supervision", False))
            if ds and (
                htype in _POOLED_HEAD_TYPES
                or tap == _BOTTLENECK_TAP
                or isinstance(tap, list)
            ):
                raise ValueError(
                    f"head_configs['{name}']: deep_supervision requires a "
                    f"spatial head with a single integer decoder-level tap "
                    f"(got type={htype!r}, tap={tap!r})"
                )
            if htype in _POOLED_HEAD_TYPES and "num_classes" not in spec:
                raise ValueError(
                    f"head_configs['{name}']: pooled heads require 'num_classes'"
                )
            if htype in _DETECTION_HEAD_TYPES and "num_classes" not in spec:
                raise ValueError(
                    f"head_configs['{name}']: detection heads require 'num_classes'"
                )
            if htype in _SPATIAL_HEAD_TYPES:
                if "out_channels" not in spec and "num_classes" not in spec:
                    raise ValueError(
                        f"head_configs['{name}']: spatial heads require "
                        f"'out_channels' (or 'num_classes')"
                    )
                if "out_channels" not in spec:
                    spec["out_channels"] = spec["num_classes"]
            spec["deep_supervision"] = ds
            normalized[name] = spec
        return normalized

    # ------------------------------------------------------------------
    # Private builders — backbone (preserved from depth.py for layer-name stability)
    # ------------------------------------------------------------------

    def _common_conv_kwargs(self) -> Dict[str, Any]:
        return dict(
            use_bias=True,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
        )

    def _build_stem(self) -> None:
        self.stem_conv = keras.layers.Conv2D(
            filters=self.level_channels[0],
            kernel_size=7,
            strides=1,
            padding="same",
            name="stem_conv",
            **self._common_conv_kwargs(),
        )
        self.stem_norm = keras.layers.BatchNormalization(name="stem_norm")

    def _build_encoder(self) -> None:
        total_blocks = sum(self.level_blocks)
        drop_rates = _linear_drop_path_rates(total_blocks, self.stochastic_depth_rate)

        self.encoder_levels: List[Dict[str, Any]] = []
        block_idx = 0

        for level in range(self.num_levels):
            ch = self.level_channels[level]
            shifts = self.level_shifts[level]
            n_blocks = self.level_blocks[level]

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

            self.encoder_levels.append(
                {"transition": transition, "blocks": blocks, "downsample": downsample}
            )

    def _build_bottleneck(self) -> None:
        ch = self.bottleneck_channels
        shifts = self.level_shifts[-1]
        n_blocks = self.level_blocks[-1]
        total_blocks = sum(self.level_blocks)
        max_drop = self.stochastic_depth_rate

        self.bottleneck_blocks: List[CliffordNetBlock] = []
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
        self.decoder_levels: List[Dict[str, Any]] = []

        for level in range(self.num_levels - 1, -1, -1):
            ch = self.level_channels[level]
            shifts = self.level_shifts[level]
            n_blocks = self.level_blocks[level]

            if level < self.num_levels - 1:
                upsample = keras.layers.UpSampling2D(
                    size=2,
                    interpolation=self.upsample_interpolation,
                    name=f"dec_upsample_{level}",
                )
            else:
                upsample = None

            skip_proj = keras.layers.Conv2D(
                filters=ch,
                kernel_size=1,
                padding="same",
                name=f"dec_skip_proj_{level}",
                **self._common_conv_kwargs(),
            )

            blocks = []
            for b in range(n_blocks):
                block = CliffordNetBlock(
                    channels=ch,
                    shifts=shifts,
                    cli_mode=self.cli_mode,
                    ctx_mode=self.ctx_mode,
                    use_global_context=self.use_global_context,
                    layer_scale_init=self.layer_scale_init,
                    drop_path_rate=0.0,
                    use_bias=True,
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f"dec_block_{level}_{b}",
                )
                blocks.append(block)

            self.decoder_levels.append(
                {
                    "upsample": upsample,
                    "skip_proj": skip_proj,
                    "blocks": blocks,
                    "level": level,
                }
            )

    # ------------------------------------------------------------------
    # Head builders
    # ------------------------------------------------------------------

    def _make_head(
        self, name: str, spec: Dict[str, Any], suffix: str = ""
    ) -> keras.layers.Layer:
        """Instantiate a single head layer from a spec."""
        htype = spec["type"]
        layer_name = f"head_{name}{suffix}"
        shared_kwargs: Dict[str, Any] = dict(
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            use_bias=True,
            name=layer_name,
        )
        if htype in _POOLED_HEAD_TYPES:
            return _ClassificationHeadBlock(
                num_classes=int(spec["num_classes"]),
                dropout=float(spec.get("dropout", 0.0)),
                hidden_dim=spec.get("hidden_dim"),
                **shared_kwargs,
            )
        if htype in _SPATIAL_HEAD_TYPES:
            return _SpatialHeadBlock(
                out_channels=int(spec["out_channels"]),
                hidden_dim=spec.get("hidden_dim"),
                **shared_kwargs,
            )
        if htype in _DETECTION_HEAD_TYPES:
            det_kwargs = {
                k: v for k, v in shared_kwargs.items() if k != "use_bias"
            }
            return _DetectionHeadBlock(
                num_classes=int(spec["num_classes"]),
                reg_max=int(spec.get("reg_max", 16)),
                **det_kwargs,
            )
        raise ValueError(f"Unknown head type {htype!r}")

    def _build_heads(self) -> None:
        """Build primary and auxiliary heads from head_configs.

        Head layers are stored in two parallel structures so Keras tracks them:
          - ``self._primary_heads: Dict[str, Layer]`` — one entry per head name.
          - ``self._aux_heads: Dict[str, Dict[int, Layer]]`` — per-level aux heads.
        Each layer is also bound as a distinct attribute (``self._head_<slug>``)
        so it appears in ``self.weights`` and ``model.summary()``.
        """
        self._primary_heads: Dict[str, keras.layers.Layer] = {}
        self._aux_heads: Dict[str, Dict[int, keras.layers.Layer]] = {}
        # Map head name → primary tap for call()
        self._head_taps: Dict[str, Tap] = {}

        for name, spec in self.head_configs.items():
            head = self._make_head(name, spec)
            self._primary_heads[name] = head
            setattr(self, f"_head_primary_{name}", head)
            self._head_taps[name] = spec["tap"]

            if spec["deep_supervision"]:
                primary_level = int(spec["tap"])  # validated earlier
                aux_levels = [
                    L for L in range(primary_level + 1, self.num_levels)
                ]
                aux_map: Dict[int, keras.layers.Layer] = {}
                for L in aux_levels:
                    aux_head = self._make_head(name, spec, suffix=f"_aux_{L}")
                    aux_map[L] = aux_head
                    setattr(self, f"_head_aux_{name}_{L}", aux_head)
                self._aux_heads[name] = aux_map

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, input_shape: Any) -> None:
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
            if isinstance(shape, list) and shape and isinstance(shape[0], list):
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
    ) -> Dict[str, keras.KerasTensor]:
        """Forward: RGB → per-head outputs.

        :param inputs: ``(B, H, W, in_channels)``.
        :param training: training-mode flag.
        :return: ``Dict[str, KerasTensor]`` — one entry per head name, plus
            ``{name}_aux_{level}`` entries for each aux head. When
            ``head_configs`` is empty, returns raw per-level features keyed
            ``"level_0"`` ... ``"level_{N-1}"`` and ``"bottleneck"``.
        """
        # Stem
        x = self.stem_conv(inputs)
        x = self.stem_norm(x, training=training)

        # Encoder
        skip_features: List[keras.KerasTensor] = []
        for level_info in self.encoder_levels:
            if level_info["transition"] is not None:
                x = level_info["transition"](x)
            for block in level_info["blocks"]:
                x = block(x, training=training)
            skip_features.append(x)
            if level_info["downsample"] is not None:
                x = level_info["downsample"](x)

        # Bottleneck
        for block in self.bottleneck_blocks:
            x = block(x, training=training)
        bottleneck_features = x

        # Decoder — collect per-level outputs
        decoder_features: Dict[int, keras.KerasTensor] = {}
        for dec_info in self.decoder_levels:
            level = dec_info["level"]
            if dec_info["upsample"] is not None:
                x = dec_info["upsample"](x)
            x = keras.ops.concatenate([x, skip_features[level]], axis=-1)
            x = dec_info["skip_proj"](x)
            for block in dec_info["blocks"]:
                x = block(x, training=training)
            decoder_features[level] = x

        # No heads → return raw feature pyramid
        if not self.head_configs:
            out: Dict[str, keras.KerasTensor] = {
                f"level_{L}": decoder_features[L] for L in decoder_features
            }
            out["bottleneck"] = bottleneck_features
            return out

        # Heads
        outputs: Dict[str, keras.KerasTensor] = {}
        for name, head in self._primary_heads.items():
            tap = self._head_taps[name]
            if isinstance(tap, list):
                feats = [decoder_features[t] for t in tap]
                outputs[name] = head(feats, training=training)
            elif tap == _BOTTLENECK_TAP:
                outputs[name] = head(bottleneck_features, training=training)
            else:
                outputs[name] = head(decoder_features[int(tap)], training=training)
            if name in self._aux_heads:
                for L, aux_head in self._aux_heads[name].items():
                    outputs[f"{name}_aux_{L}"] = aux_head(
                        decoder_features[L], training=training
                    )
        return outputs

    # ------------------------------------------------------------------
    # Shape inference
    # ------------------------------------------------------------------

    def compute_output_shape(self, input_shape: Any) -> Any:
        if isinstance(input_shape, list):
            s = input_shape[0]
        else:
            s = input_shape
        s = tuple(s)

        def _level_hw(level: int) -> Tuple[Optional[int], Optional[int]]:
            h = s[1] // (2 ** level) if s[1] is not None else None
            w = s[2] // (2 ** level) if s[2] is not None else None
            return h, w

        if not self.head_configs:
            out: Dict[str, Any] = {}
            for L in range(self.num_levels):
                h, w = _level_hw(L)
                out[f"level_{L}"] = (s[0], h, w, self.level_channels[L])
            h, w = _level_hw(self.num_levels - 1)
            out["bottleneck"] = (s[0], h, w, self.bottleneck_channels)
            return out

        shapes: Dict[str, Any] = {}
        for name, spec in self.head_configs.items():
            tap = spec["tap"]
            if spec["type"] in _POOLED_HEAD_TYPES:
                shapes[name] = (s[0], spec["num_classes"])
            elif spec["type"] in _DETECTION_HEAD_TYPES:
                reg_max = int(spec.get("reg_max", 16))
                out_dim = 4 * reg_max + int(spec["num_classes"])
                shapes[name] = (s[0], None, out_dim)
            else:
                level = int(tap) if tap != _BOTTLENECK_TAP else (self.num_levels - 1)
                h, w = _level_hw(level)
                shapes[name] = (s[0], h, w, spec["out_channels"])
                if spec["deep_supervision"]:
                    for L in range(int(tap) + 1, self.num_levels):
                        h_l, w_l = _level_hw(L)
                        shapes[f"{name}_aux_{L}"] = (
                            s[0], h_l, w_l, spec["out_channels"]
                        )
        return shapes

    # ------------------------------------------------------------------
    # Serialization (DECISION D-007 — head_configs as JSON for robust round-trip)
    # ------------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "in_channels": self.in_channels,
                "level_channels": self.level_channels,
                "level_blocks": self.level_blocks,
                "level_shifts": self.level_shifts,
                "cli_mode": self.cli_mode,
                "ctx_mode": self.ctx_mode,
                "use_global_context": self.use_global_context,
                "layer_scale_init": self.layer_scale_init,
                "stochastic_depth_rate": self.stochastic_depth_rate,
                "upsample_interpolation": self.upsample_interpolation,
                "kernel_initializer": initializers.serialize(self.kernel_initializer),
                "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
                "head_configs_json": json.dumps(self.head_configs, default=str),
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CliffordNetUNet":
        config = dict(config)
        hc_json = config.pop("head_configs_json", None)
        if hc_json is not None:
            config["head_configs"] = json.loads(hc_json)
        if isinstance(config.get("kernel_regularizer"), dict):
            config["kernel_regularizer"] = regularizers.deserialize(
                config["kernel_regularizer"]
            )
        if isinstance(config.get("kernel_initializer"), dict):
            config["kernel_initializer"] = initializers.deserialize(
                config["kernel_initializer"]
            )
        return cls(**config)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self, **kwargs: Any) -> None:
        if not self.built:
            logger.warning("Model not built; call build() with input shape first.")
        super().summary(**kwargs)
        logger.info("CliffordNetUNet configuration:")
        logger.info(f"  in_channels         : {self.in_channels}")
        logger.info(f"  level_channels      : {self.level_channels}")
        logger.info(f"  level_blocks        : {self.level_blocks}")
        logger.info(f"  level_shifts        : {self.level_shifts}")
        logger.info(f"  cli_mode            : {self.cli_mode}")
        logger.info(f"  ctx_mode            : {self.ctx_mode}")
        logger.info(f"  use_global_context  : {self.use_global_context}")
        logger.info(f"  heads               : {list(self.head_configs.keys())}")

    # ------------------------------------------------------------------
    # Variants
    # ------------------------------------------------------------------

    MODEL_VARIANTS: Dict[str, Dict[str, Any]] = {
        "tiny": dict(
            level_channels=[32, 64, 128],
            level_blocks=[2, 2, 2],
            level_shifts=[[1, 2], [1, 2, 4], [1, 2, 4]],
            cli_mode="full",
            ctx_mode="diff",
            use_global_context=False,
            layer_scale_init=1e-5,
            kernel_initializer=_DEFAULT_KERNEL_INIT,
        ),
        "small": dict(
            level_channels=[32, 64, 128],
            level_blocks=[2, 3, 3],
            level_shifts=[[1, 2], [1, 2, 4], [1, 2, 4, 8]],
            cli_mode="full",
            ctx_mode="diff",
            use_global_context=False,
            layer_scale_init=1e-5,
            kernel_initializer=_DEFAULT_KERNEL_INIT,
        ),
        "base": dict(
            level_channels=[32, 64, 128, 256],
            level_blocks=[2, 2, 2, 2],
            level_shifts=[[1, 2], [1, 2, 4], [1, 2, 4, 8], [1, 2, 4]],
            cli_mode="full",
            ctx_mode="diff",
            use_global_context=True,
            layer_scale_init=1e-5,
            kernel_initializer=_DEFAULT_KERNEL_INIT,
        ),
        "medium": dict(
            level_channels=[64, 128, 256, 256],
            level_blocks=[2, 3, 3, 2],
            level_shifts=[[1, 2], [1, 2, 4], [1, 2, 4, 8], [1, 2, 4, 8]],
            cli_mode="full",
            ctx_mode="diff",
            use_global_context=True,
            layer_scale_init=1e-5,
            stochastic_depth_rate=0.05,
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
        head_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> "CliffordNetUNet":
        """Create a :class:`CliffordNetUNet` from a named variant.

        :param variant: One of ``tiny``, ``small``, ``base``, ``medium``, ``large``, ``xlarge``.
        :param in_channels: Input channels.
        :param head_configs: Optional head specification (see class docstring).
        :param kwargs: Override any default hyperparameter.
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant {variant!r}. Available: {list(cls.MODEL_VARIANTS)}"
            )
        defaults = dict(cls.MODEL_VARIANTS[variant])
        defaults.update(kwargs)
        logger.info(f"Creating CliffordNetUNet-{variant.upper()}")
        return cls(in_channels=in_channels, head_configs=head_configs, **defaults)


# ===========================================================================
# Depth factory (replaces CliffordNetDepthEstimator)
# ===========================================================================


def create_cliffordnet_depth(
    variant: str = "tiny",
    in_channels: int = 3,
    out_channels: int = 1,
    enable_deep_supervision: bool = True,
    head_hidden_dim: Optional[int] = None,
    **kwargs: Any,
) -> CliffordNetUNet:
    """Create a CliffordNet U-Net configured for monocular depth estimation.

    Exact replacement for the previous ``CliffordNetDepthEstimator.from_variant``.
    Attaches a single spatial head at decoder level 0 and — when
    ``enable_deep_supervision`` is ``True`` (default) — auxiliary heads at every
    deeper decoder level.  The head itself matches the original depth head
    (LayerNorm + Conv 1×1) when ``head_hidden_dim`` is ``None``.

    :param variant: Model variant (``tiny``, ``small``, ``base``, ``medium``, ``large``, ``xlarge``).
    :param in_channels: Input channels (3 for RGB).
    :param out_channels: Number of depth channels (1 for scalar depth).
    :param enable_deep_supervision: Attach aux heads at deeper decoder levels.
    :param head_hidden_dim: Optional hidden Conv3×3 before the output Conv1×1.
        ``None`` matches the legacy depth head (LayerNorm + Conv1×1 only).
    :param kwargs: Override any backbone hyperparameter.
    """
    head_configs = {
        "depth": {
            "type": "depth",
            "tap": 0,
            "out_channels": out_channels,
            "deep_supervision": enable_deep_supervision,
            "hidden_dim": head_hidden_dim,
        }
    }
    logger.info(
        f"Creating CliffordNet depth estimator "
        f"(variant={variant}, out_channels={out_channels}, "
        f"deep_supervision={enable_deep_supervision})"
    )
    return CliffordNetUNet.from_variant(
        variant,
        in_channels=in_channels,
        head_configs=head_configs,
        **kwargs,
    )
