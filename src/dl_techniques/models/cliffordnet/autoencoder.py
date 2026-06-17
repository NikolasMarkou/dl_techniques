"""
CliffordNet explicit-Laplacian-pyramid image autoencoder.

This module builds a deterministic image autoencoder that fuses an EXPLICIT,
inspectable Laplacian pyramid with CliffordNet geometric-algebra feature
extraction. The defining design choice is that the multi-scale split/merge is a
standalone, serializable helper Layer (``LaplacianPyramidLevel``) operating at
the RAW signal channel count, completely decoupled from any learned Clifford
processing. This makes the high/low frequency decomposition auditable and makes
the reconstruction identity exact to float precision.

Channel-bookkeeping scheme (LOCKED)
-----------------------------------
The low/high decomposition is deliberately *channel-preserving* so the Laplacian
reconstruction identity holds exactly in raw signal space:

- ``low_i  = down(blur(x_i))``  via ``GaussianFilter(strides=(1,1), padding="same")``
  (preserves C) then ``BlurPool2D(strides=2)`` (preserves C). Result has shape
  ``(B, H/2, W/2, C_i)`` -- same channel count as ``x_i``. Both stages are
  anti-aliased.
- ``up`` via ``UpSampling2D(bilinear)`` -- preserves channels, doubles H, W.
- ``high_i = x_i - up(low_i)``: shape ``(B, H, W, C_i)``, same channels as ``x_i``.
  Since ``high_i`` is DEFINED as the residual ``x_i - up(low_i)``, the merge
  ``high_i + up(low_i)`` recovers ``x_i`` exactly (up to float rounding),
  independent of blur quality.

The split is purely signal-level (NO learnable channel change): this is what
makes the reconstruction identity hold. ``PixelUnshuffle2D`` is deliberately NOT
used for the low path -- its 4x channel multiplier breaks the additive identity.
It is reserved as a documented alternative only if a future lossless variant is
wanted; the default path is Gaussian + BlurPool down / bilinear up.

The model class (``CliffordLaplacianUNet``) widens channels via external 1x1
``Conv2D`` before each isotropic ``CliffordNetBlock`` group; the Laplacian SIGNAL
pyramid itself stays at the raw channel count. Clifford blocks process widened
feature copies that contribute learned refinements on the decoder side; they
never sit inside the invertibility path of this helper. The ``LaplacianPyramidLevel``
helper now lives in ``dl_techniques.layers.laplacian_filter``; this module defines
the model (``CliffordLaplacianUNet``) and imports the helper from there.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import keras

from dl_techniques.layers.gaussian_filter import GaussianFilter
from dl_techniques.layers.blur_pool import BlurPool2D
from dl_techniques.layers.laplacian_filter import LaplacianPyramidLevel
from dl_techniques.layers.geometric.clifford_block import CliffordNetBlock
from dl_techniques.utils.logger import logger


# ===========================================================================
# CliffordLaplacianUNet
# ===========================================================================


@keras.saving.register_keras_serializable()
class CliffordLaplacianUNet(keras.Model):
    """Deterministic image autoencoder: explicit learnable Laplacian pyramid + Clifford U-Net.

    A feature-space Laplacian U-Net with two decoupled tracks:

    - **Track A (signal pyramid)**: one :class:`LaplacianPyramidLevel` per encoder
      level produces ``(low_i, high_i)`` at the NATIVE channel count. ``low_i``
      (H/2) feeds the next level; ``high_i`` (full res) is the high-frequency band
      kept at level ``i``. This track is weightless except for the (optionally
      learnable) blur and is decoupled from any learned channel change.
    - **Track B (learned features)**: :class:`CliffordNetBlock` is ISOTROPIC, so
      EVERY channel change is an external 1x1 ``keras.layers.Conv2D``. The high
      band of each level is projected up to ``level_channels[i]``, refined by
      Clifford blocks, and used as the decoder skip. The coarsest ``low`` feeds a
      bottleneck. The decoder collapses bottom-up via a FEATURE-SPACE Laplacian
      merge (``keras.ops.add`` of upsampled coarser features and the level skip),
      each step refined by Clifford blocks. A final 1x1 conv projects back to
      ``in_channels``.

    The model returns a single-output dict ``{"reconstruction": y}`` with ``y`` of
    identical shape to the input.

    Divisibility contract: input H, W must be divisible by ``2**num_levels``
    (each encoder level halves the spatial resolution via :class:`BlurPool2D`).

    :param in_channels: Number of input/output image channels (e.g. 3 for RGB).
    :type in_channels: int
    :param level_channels: Per-level working width for the Clifford feature track.
        Its length defines ``num_levels``.
    :type level_channels: Tuple[int, ...]
    :param level_blocks: Number of Clifford blocks per level (encoder and decoder);
        the bottleneck reuses ``level_blocks[-1]``. Must match ``num_levels``.
    :type level_blocks: Tuple[int, ...]
    :param level_shifts: Per-level channel-shift offsets for the Clifford blocks.
        ``None`` defaults to ``[[1, 2]] * num_levels``.
    :type level_shifts: Optional[List[List[int]]]
    :param cli_mode: Algebraic components for the local interaction.
    :type cli_mode: str
    :param ctx_mode: Context mode for the Clifford blocks.
    :type ctx_mode: str
    :param use_global_context: Whether Clifford blocks use the global-context
        branch (requires per-block ``channels >= 2``).
    :type use_global_context: bool
    :param blur_kernel_size: Gaussian blur kernel size for the pyramid split.
    :type blur_kernel_size: Tuple[int, int]
    :param blur_sigma: Gaussian sigma; ``-1`` derives it from the kernel size.
    :type blur_sigma: float
    :param blur_trainable: Whether the pyramid blur kernels are learnable.
    :type blur_trainable: bool
    :param kernel_initializer: Initializer for all 1x1 ``Conv2D`` projections.
    :type kernel_initializer: Any
    :param kernel_regularizer: Regularizer for all 1x1 ``Conv2D`` projections.
    :type kernel_regularizer: Optional[Any]
    :param kwargs: Additional keyword arguments for :class:`keras.Model`.
    """

    def __init__(
        self,
        in_channels: int = 3,
        level_channels: Tuple[int, ...] = (32, 64, 128),
        level_blocks: Tuple[int, ...] = (2, 2, 2),
        level_shifts: Optional[List[List[int]]] = None,
        cli_mode: str = "full",
        ctx_mode: str = "diff",
        use_global_context: bool = True,
        blur_kernel_size: Tuple[int, int] = (5, 5),
        blur_sigma: float = -1,
        blur_trainable: bool = False,
        kernel_initializer: Any = "glorot_uniform",
        kernel_regularizer: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.num_levels = len(level_channels)
        if len(level_blocks) != self.num_levels:
            raise ValueError(
                f"len(level_blocks)={len(level_blocks)} must equal "
                f"num_levels=len(level_channels)={self.num_levels}"
            )
        if level_shifts is None:
            # [1, 2] is valid for SparseRollingGeometricProduct (ints >= 1) and
            # matches the hardcoded global-branch shifts.
            level_shifts = [[1, 2] for _ in range(self.num_levels)]
        if len(level_shifts) != self.num_levels:
            raise ValueError(
                f"len(level_shifts)={len(level_shifts)} must equal "
                f"num_levels={self.num_levels}"
            )

        # Store all ctor args verbatim (as given) for get_config.
        self.in_channels = in_channels
        self.level_channels = level_channels
        self.level_blocks = level_blocks
        self.level_shifts = level_shifts
        self.cli_mode = cli_mode
        self.ctx_mode = ctx_mode
        self.use_global_context = use_global_context
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
        self.blur_trainable = blur_trainable
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer

        _conv_kwargs: Dict[str, Any] = dict(
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
        )

        def _make_blocks(n: int, channels: int, shifts: List[int]):
            return [
                CliffordNetBlock(
                    channels=channels,
                    shifts=shifts,
                    cli_mode=cli_mode,
                    ctx_mode=ctx_mode,
                    use_global_context=use_global_context,
                )
                for _ in range(n)
            ]

        # --- Signal pyramid (Track A): one per level, native channels ---
        self.pyramid_levels = [
            LaplacianPyramidLevel(blur_kernel_size, blur_sigma, blur_trainable)
            for _ in range(self.num_levels)
        ]

        # --- Encoder (Track B): project native high band -> level width, refine ---
        self.enc_proj = [
            keras.layers.Conv2D(level_channels[i], 1, **_conv_kwargs)
            for i in range(self.num_levels)
        ]
        self.enc_blocks = [
            _make_blocks(level_blocks[i], level_channels[i], level_shifts[i])
            for i in range(self.num_levels)
        ]

        # --- Bottleneck on coarsest low band (native channels) ---
        self.bottleneck_proj = keras.layers.Conv2D(
            level_channels[-1], 1, **_conv_kwargs
        )
        self.bottleneck_blocks = _make_blocks(
            level_blocks[-1], level_channels[-1], level_shifts[-1]
        )

        # --- Decoder: upsample coarser features, align channels, merge + refine ---
        self.dec_up = [
            keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")
            for _ in range(self.num_levels)
        ]
        self.dec_align = [
            keras.layers.Conv2D(level_channels[i], 1, **_conv_kwargs)
            for i in range(self.num_levels)
        ]
        self.dec_blocks = [
            _make_blocks(level_blocks[i], level_channels[i], level_shifts[i])
            for i in range(self.num_levels)
        ]

        # --- Output head: decoder level-0 features -> image channels ---
        self.out_conv = keras.layers.Conv2D(in_channels, 1, **_conv_kwargs)

    # ------------------------------------------------------------------

    def build(self, input_shape) -> None:
        """Build all sublayers in computational order; ``super().build()`` LAST.

        Enforces the divisibility contract: static H, W (if known) must be
        divisible by ``2**num_levels``.

        :param input_shape: Input shape ``(B, H, W, C)``.
        """
        batch, h, w, c = input_shape
        divisor = 2 ** self.num_levels
        for dim, name in ((h, "height"), (w, "width")):
            if dim is not None and dim % divisor != 0:
                raise ValueError(
                    f"Input {name}={dim} must be divisible by "
                    f"2**num_levels={divisor} (the model halves spatial "
                    f"resolution once per encoder level)."
                )

        # --- Encoder: walk down the signal pyramid, build feature track ---
        cur = input_shape  # native-channel signal at current level
        highs_shapes: List[Tuple] = []
        for i in range(self.num_levels):
            self.pyramid_levels[i].build(cur)
            low_shape, high_shape = self.pyramid_levels[i].compute_output_shape(cur)

            # Project native high band -> level width, then refine.
            self.enc_proj[i].build(high_shape)
            feat_shape = self.enc_proj[i].compute_output_shape(high_shape)
            for blk in self.enc_blocks[i]:
                blk.build(feat_shape)
                feat_shape = blk.compute_output_shape(feat_shape)
            highs_shapes.append(feat_shape)

            cur = low_shape  # native-channel low band feeds the next level

        # --- Bottleneck on coarsest low band (native channels) ---
        self.bottleneck_proj.build(cur)
        b_shape = self.bottleneck_proj.compute_output_shape(cur)
        for blk in self.bottleneck_blocks:
            blk.build(b_shape)
            b_shape = blk.compute_output_shape(b_shape)

        # --- Decoder: collapse bottom-up ---
        feat_shape = b_shape
        for i in range(self.num_levels - 1, -1, -1):
            self.dec_up[i].build(feat_shape)
            up_shape = self.dec_up[i].compute_output_shape(feat_shape)
            self.dec_align[i].build(up_shape)
            up_shape = self.dec_align[i].compute_output_shape(up_shape)
            # merged has the same shape as the aligned upsampled features
            # (== highs_shapes[i] by construction); refine in place.
            merged_shape = up_shape
            for blk in self.dec_blocks[i]:
                blk.build(merged_shape)
                merged_shape = blk.compute_output_shape(merged_shape)
            feat_shape = merged_shape

        # --- Output head ---
        self.out_conv.build(feat_shape)

        super().build(input_shape)

    # ------------------------------------------------------------------

    def call(self, inputs, training=None) -> Dict[str, Any]:
        """Forward pass.

        :param inputs: Image tensor ``(B, H, W, C)``.
        :param training: Whether in training mode.
        :return: ``{"reconstruction": y}`` with ``y`` of shape ``(B, H, W, C)``.
        """
        # --- Encoder ---
        x = inputs
        highs: List[Any] = []
        for i in range(self.num_levels):
            low, high = self.pyramid_levels[i].split(x)
            h = self.enc_proj[i](high)
            for blk in self.enc_blocks[i]:
                h = blk(h, training=training)
            highs.append(h)
            x = low

        # --- Bottleneck ---
        b = self.bottleneck_proj(x)
        for blk in self.bottleneck_blocks:
            b = blk(b, training=training)

        # --- Decoder (feature-space Laplacian merge) ---
        feat = b
        for i in range(self.num_levels - 1, -1, -1):
            up = self.dec_up[i](feat)
            up = self.dec_align[i](up)
            merged = keras.ops.add(up, highs[i])
            for blk in self.dec_blocks[i]:
                merged = blk(merged, training=training)
            feat = merged

        y = self.out_conv(feat)
        return {"reconstruction": y}

    # ------------------------------------------------------------------

    def compute_output_shape(self, input_shape):
        return {"reconstruction": tuple(input_shape)}

    # ------------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "in_channels": self.in_channels,
                "level_channels": list(self.level_channels),
                "level_blocks": list(self.level_blocks),
                "level_shifts": [list(s) for s in self.level_shifts],
                "cli_mode": self.cli_mode,
                "ctx_mode": self.ctx_mode,
                "use_global_context": self.use_global_context,
                "blur_kernel_size": self.blur_kernel_size,
                "blur_sigma": self.blur_sigma,
                "blur_trainable": self.blur_trainable,
                "kernel_initializer": self.kernel_initializer,
                "kernel_regularizer": self.kernel_regularizer,
            }
        )
        return config

    # ------------------------------------------------------------------
    # Variants
    # ------------------------------------------------------------------

    MODEL_VARIANTS: Dict[str, Dict[str, Any]] = {
        "small": dict(
            level_channels=[32, 64, 128],
            level_blocks=[2, 2, 2],
        ),
        "base": dict(
            level_channels=[48, 96, 192, 384],
            level_blocks=[2, 2, 2, 2],
        ),
        "large": dict(
            level_channels=[64, 128, 256, 512],
            level_blocks=[3, 3, 3, 3],
        ),
    }

    @classmethod
    def from_variant(
        cls,
        variant: str,
        in_channels: int = 3,
        **kwargs: Any,
    ) -> "CliffordLaplacianUNet":
        """Create a :class:`CliffordLaplacianUNet` from a named variant.

        :param variant: One of ``small``, ``base``, ``large``.
        :param in_channels: Number of input/output image channels (3 for RGB).
        :param kwargs: Override any default hyperparameter (kwargs win over the
            variant defaults).
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant {variant!r}. Available: {list(cls.MODEL_VARIANTS)}"
            )
        defaults = dict(cls.MODEL_VARIANTS[variant])
        defaults.update(kwargs)
        logger.info(f"Creating CliffordLaplacianUNet-{variant.upper()}")
        return cls(in_channels=in_channels, **defaults)


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------


def create_clifford_laplacian_unet(
    variant: str = "small",
    in_channels: int = 3,
    **overrides: Any,
) -> CliffordLaplacianUNet:
    """Create a :class:`CliffordLaplacianUNet` deterministic image autoencoder.

    Thin convenience delegate to :meth:`CliffordLaplacianUNet.from_variant`.

    :param variant: Named model variant; one of ``small``, ``base``, ``large``.
        ``small`` is a 3-level pyramid, ``base``/``large`` are 4-level (inputs
        must be divisible by ``2**num_levels`` -> 8 for ``small``, 16 for the
        4-level variants).
    :type variant: str
    :param in_channels: Number of input/output image channels (3 for RGB).
    :type in_channels: int
    :param overrides: Keyword overrides forwarded to the constructor; these take
        precedence over the variant defaults.
    :return: A configured (un-built) :class:`CliffordLaplacianUNet`.
    :rtype: CliffordLaplacianUNet
    """
    return CliffordLaplacianUNet.from_variant(
        variant, in_channels=in_channels, **overrides
    )
