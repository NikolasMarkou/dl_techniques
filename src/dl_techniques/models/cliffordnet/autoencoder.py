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

import keras
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.laplacian_filter import LaplacianPyramidLevel
from dl_techniques.layers.geometric.clifford_block import CliffordNetBlock


# ---------------------------------------------------------------------------
# CliffordLaplacianUNet
# ---------------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class CliffordLaplacianUNet(keras.Model):
    """Deterministic image autoencoder: explicit learnable Laplacian pyramid + Clifford U-Net.

    **Intent**: Provide a feature-space Laplacian U-Net that fuses an EXPLICIT,
    inspectable signal pyramid with isotropic CliffordNet geometric-algebra
    feature extraction. The multi-scale split/merge is a standalone, serializable
    helper layer (:class:`LaplacianPyramidLevel`) operating at the RAW signal
    channel count, fully decoupled from any learned Clifford processing, so the
    high/low frequency decomposition stays auditable and the reconstruction
    identity exact to float precision. The model returns a single-output dict
    ``{"reconstruction": y}`` with ``y`` of identical shape to the input.

    The architecture has two decoupled tracks:

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

    **Architecture**:
    ```
    Input(B, H, W, C_in)
           |
      +----v---------------------------------------------------+
      | Encoder (per level i = 0 .. num_levels-1)              |
      |   low_i, high_i = LaplacianPyramidLevel[i].split(x)   |  Track A (signal)
      |   h_i  = enc_proj[i](high_i)        # 1x1 -> ch[i]    |  Track B (features)
      |   h_i  = enc_blocks[i](h_i)         # Clifford x N    |
      |   x    = low_i                      # feeds level i+1 |
      +----v---------------------------------------------------+
           |  (coarsest low band, native channels)
      +----v---------------------------------------------------+
      | Bottleneck                                            |
      |   b = bottleneck_proj(x)            # 1x1 -> ch[-1]   |
      |   b = bottleneck_blocks(b)          # Clifford x N    |
      +----v---------------------------------------------------+
           |
      +----v---------------------------------------------------+
      | Decoder (per level i = num_levels-1 .. 0)             |
      |   up     = dec_up[i](feat)          # bilinear x2     |
      |   up     = dec_align[i](up)         # 1x1 -> ch[i]    |
      |   merged = add(up, h_i)             # Laplacian merge |
      |   feat   = dec_blocks[i](merged)    # Clifford x N    |
      +----v---------------------------------------------------+
           |
      out_conv(feat)                        # 1x1 -> C_in
           |
    {"reconstruction": (B, H, W, C_in)}
    ```

    Divisibility contract: input H, W must be divisible by ``2**num_levels``
    (each encoder level halves the spatial resolution via :class:`BlurPool2D`).

    Args:
        in_channels: Number of input/output image channels (e.g. 3 for RGB).
            Must be positive. Defaults to ``3``.
        level_channels: Per-level working width for the Clifford feature track.
            Its length defines ``num_levels``. Defaults to ``(32, 64, 128)``.
        level_blocks: Number of Clifford blocks per level (encoder and decoder);
            the bottleneck reuses ``level_blocks[-1]``. Must match ``num_levels``.
            Defaults to ``(2, 2, 2)``.
        level_shifts: Per-level channel-shift offsets for the Clifford blocks.
            ``None`` defaults to ``[[1, 2]] * num_levels``. Defaults to ``None``.
        cli_mode: Algebraic components for the local interaction; one of
            ``"inner"``, ``"wedge"``, ``"full"``. Defaults to ``"full"``.
        ctx_mode: Context mode for the Clifford blocks; one of ``"diff"``,
            ``"abs"``. Defaults to ``"diff"``.
        use_global_context: Whether Clifford blocks use the global-context branch
            (requires per-block ``channels >= 2``). Defaults to ``True``.
        blur_kernel_size: Gaussian blur kernel size (length-2 tuple of positive
            ints) for the pyramid split. Defaults to ``(5, 5)``.
        blur_sigma: Gaussian sigma; ``-1`` derives it from the kernel size.
            Defaults to ``-1``.
        blur_trainable: Whether the pyramid blur kernels are learnable. Defaults
            to ``False``.
        kernel_initializer: Initializer for all 1x1 ``Conv2D`` projections.
            Defaults to ``"glorot_uniform"``.
        kernel_regularizer: Regularizer for all 1x1 ``Conv2D`` projections.
            Defaults to ``None``.
        **kwargs: Additional keyword arguments for :class:`keras.Model`.

    Input shape:
        4D tensor ``(batch_size, height, width, in_channels)``. ``height`` and
        ``width`` must be divisible by ``2 ** num_levels``.

    Output shape:
        A dict ``{"reconstruction": tensor}`` whose tensor has the same shape as
        the input: ``(batch_size, height, width, in_channels)``.

    Raises:
        ValueError: If ``in_channels <= 0``; if any ``level_channels`` or
            ``level_blocks`` value is non-positive; if ``len(level_channels) !=
            len(level_blocks)``; if ``cli_mode``/``ctx_mode`` is not in its valid
            set; or if ``blur_kernel_size`` is not a length-2 tuple of positive
            ints.

    Example:
        ```python
        import numpy as np

        # Direct construction (tiny, CPU-friendly).
        model = CliffordLaplacianUNet(
            level_channels=(8, 16),
            level_blocks=(1, 1),
        )
        out = model(np.zeros((1, 32, 32, 3), "float32"))
        recon = out["reconstruction"]  # shape (1, 32, 32, 3)

        # Named variant.
        model = CliffordLaplacianUNet.from_variant("small", in_channels=3)

        # Factory convenience.
        model = create_clifford_laplacian_unet("small", in_channels=3)
        ```
    """

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
        if level_shifts is None:
            # [1, 2] is valid for SparseRollingGeometricProduct (ints >= 1) and
            # matches the hardcoded global-branch shifts.
            level_shifts = [[1, 2] for _ in range(self.num_levels)]

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
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        # Validate config BEFORE building any sublayers.
        self._validate_config()

        # Build sublayers via private helpers (same attrs/order/args as inline).
        self._build_encoder()
        self._build_decoder()
        self._build_output()

        logger.info(
            f"Created CliffordLaplacianUNet: {self.num_levels} levels, "
            f"channels={list(self.level_channels)}, "
            f"blocks={list(self.level_blocks)}"
        )

    # ------------------------------------------------------------------

    def _validate_config(self) -> None:
        """Validate constructor arguments; raise ``ValueError`` on bad config.

        Defense-in-depth model-scoped validation that fails at construction
        (before any sublayer is built) with a clear message. The valid
        ``cli_mode``/``ctx_mode`` literals mirror the sets enforced by
        :class:`CliffordNetBlock` (``clifford_block.py``).
        """
        if self.in_channels <= 0:
            raise ValueError(
                f"CliffordLaplacianUNet: in_channels must be positive, "
                f"got {self.in_channels}"
            )

        if len(self.level_blocks) != self.num_levels:
            raise ValueError(
                f"CliffordLaplacianUNet: len(level_blocks)="
                f"{len(self.level_blocks)} must equal "
                f"num_levels=len(level_channels)={self.num_levels}"
            )

        for i, ch in enumerate(self.level_channels):
            if ch <= 0:
                raise ValueError(
                    f"CliffordLaplacianUNet: level_channels[{i}] must be "
                    f"positive, got {ch}"
                )

        for i, nb in enumerate(self.level_blocks):
            if nb <= 0:
                raise ValueError(
                    f"CliffordLaplacianUNet: level_blocks[{i}] must be "
                    f"positive, got {nb}"
                )

        if len(self.level_shifts) != self.num_levels:
            raise ValueError(
                f"CliffordLaplacianUNet: len(level_shifts)="
                f"{len(self.level_shifts)} must equal "
                f"num_levels={self.num_levels}"
            )

        if self.cli_mode not in ("inner", "wedge", "full"):
            raise ValueError(
                f"CliffordLaplacianUNet: cli_mode must be 'inner', 'wedge', "
                f"or 'full', got {self.cli_mode!r}"
            )

        if self.ctx_mode not in ("diff", "abs"):
            raise ValueError(
                f"CliffordLaplacianUNet: ctx_mode must be 'diff' or 'abs', "
                f"got {self.ctx_mode!r}"
            )

        ks = self.blur_kernel_size
        if (
            not isinstance(ks, (tuple, list))
            or len(ks) != 2
            or any(
                (not isinstance(k, int)) or isinstance(k, bool) or k <= 0
                for k in ks
            )
        ):
            raise ValueError(
                f"CliffordLaplacianUNet: blur_kernel_size must be a length-2 "
                f"sequence of positive ints, got {ks!r}"
            )

    # ------------------------------------------------------------------

    def _make_blocks(
        self, n: int, channels: int, shifts: List[int]
    ) -> List[CliffordNetBlock]:
        """Build ``n`` isotropic CliffordNetBlock(s) at ``channels`` width."""
        return [
            CliffordNetBlock(
                channels=channels,
                shifts=shifts,
                cli_mode=self.cli_mode,
                ctx_mode=self.ctx_mode,
                use_global_context=self.use_global_context,
            )
            for _ in range(n)
        ]

    # ------------------------------------------------------------------

    def _build_encoder(self) -> None:
        """Build the encoder feature track and the per-split-stage signal pyramid.

        Clifford encodes the full incoming signal at each scale (``enc_proj`` then
        ``enc_blocks``); the deepest stage (index ``L-1``) IS the bottleneck
        refinement, so there is one pyramid level per SPLIT stage (``L-1`` total).
        """
        _conv_kwargs: Dict[str, Any] = dict(
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
        )

        # --- Signal pyramid: one per SPLIT stage (indices 0 .. L-2) ---
        self.pyramid_levels = [
            LaplacianPyramidLevel(
                self.blur_kernel_size, self.blur_sigma, self.blur_trainable
            )
            for _ in range(self.num_levels - 1)
        ]

        # --- Encoder feature track: project to level width, refine (incl. deepest) ---
        self.enc_proj = [
            keras.layers.Conv2D(self.level_channels[i], 1, **_conv_kwargs)
            for i in range(self.num_levels)
        ]
        self.enc_blocks = [
            self._make_blocks(
                self.level_blocks[i], self.level_channels[i], self.level_shifts[i]
            )
            for i in range(self.num_levels)
        ]

    # ------------------------------------------------------------------

    def _build_decoder(self) -> None:
        """Build the decoder: upsample, align channels, merge + refine."""
        _conv_kwargs: Dict[str, Any] = dict(
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
        )

        self.dec_up = [
            keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")
            for _ in range(self.num_levels - 1)
        ]
        self.dec_align = [
            keras.layers.Conv2D(self.level_channels[i], 1, **_conv_kwargs)
            for i in range(self.num_levels - 1)
        ]
        self.dec_blocks = [
            self._make_blocks(
                self.level_blocks[i], self.level_channels[i], self.level_shifts[i]
            )
            for i in range(self.num_levels - 1)
        ]

    # ------------------------------------------------------------------

    def _build_output(self) -> None:
        """Build the output head: decoder level-0 features -> image channels."""
        _conv_kwargs: Dict[str, Any] = dict(
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
        )
        self.out_conv = keras.layers.Conv2D(self.in_channels, 1, **_conv_kwargs)

    # ------------------------------------------------------------------

    def build(self, input_shape) -> None:
        """Build all sublayers in computational order; ``super().build()`` LAST.

        Enforces the divisibility contract: static H, W (if known) must be
        divisible by ``2**num_levels``.

        Args:
            input_shape: Input shape ``(B, H, W, C)``. A 3D shape ``(H, W, C)``
                (e.g. from ``summary()``) is accepted and a dummy batch dim is
                prepended.

        Raises:
            ValueError: If a known static ``height`` or ``width`` is not divisible
                by ``2 ** num_levels``.
        """
        # summary() may call build with a 3D shape (no batch dim); prepend a
        # dummy batch dim so the unpack below is safe.
        if len(input_shape) == 3:
            input_shape = (None,) + tuple(input_shape)
        batch, h, w, c = input_shape
        divisor = 2 ** (self.num_levels - 1)
        for dim, name in ((h, "height"), (w, "width")):
            if dim is not None and dim % divisor != 0:
                raise ValueError(
                    f"Input {name}={dim} must be divisible by "
                    f"2**(num_levels-1)={divisor} (the model halves spatial "
                    f"resolution once per SPLIT stage)."
                )

        L = self.num_levels

        # --- Encoder: encode the full signal at each scale, THEN split ---
        cur = input_shape  # signal at the current scale
        bottleneck_shape = None
        for i in range(L):
            self.enc_proj[i].build(cur)
            feat_shape = self.enc_proj[i].compute_output_shape(cur)
            for blk in self.enc_blocks[i]:
                blk.build(feat_shape)
                feat_shape = blk.compute_output_shape(feat_shape)

            if i < L - 1:
                # Split stage: split ENCODED features; low band feeds next stage.
                self.pyramid_levels[i].build(feat_shape)
                low_shape, high_shape = self.pyramid_levels[i].compute_output_shape(
                    feat_shape
                )
                cur = low_shape  # high_shape == feat_shape is the decoder skip
            else:
                # Deepest stage IS the bottleneck (no split).
                bottleneck_shape = feat_shape

        # --- Decoder: collapse bottom-up over the L-1 split stages ---
        feat_shape = bottleneck_shape
        for i in range(L - 2, -1, -1):
            self.dec_up[i].build(feat_shape)
            up_shape = self.dec_up[i].compute_output_shape(feat_shape)
            self.dec_align[i].build(up_shape)
            up_shape = self.dec_align[i].compute_output_shape(up_shape)
            # merged has the same shape as the aligned upsampled features
            # (== the level-i high-band skip by construction); refine in place.
            merged_shape = up_shape
            for blk in self.dec_blocks[i]:
                blk.build(merged_shape)
                merged_shape = blk.compute_output_shape(merged_shape)
            feat_shape = merged_shape

        # --- Output head ---
        self.out_conv.build(feat_shape)

        super().build(input_shape)

    # ------------------------------------------------------------------

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> Dict[str, keras.KerasTensor]:
        """Forward pass.

        Args:
            inputs: Image tensor ``(B, H, W, C)``.
            training: Whether the call is in training mode.

        Returns:
            A dict ``{"reconstruction": y}`` with ``y`` of shape ``(B, H, W, C)``.
        """
        # DECISION plan_2026-06-18_959c992a/D-001: encode-then-split hierarchical flow.
        # Clifford ENCODES the full incoming signal at each scale FIRST, THEN the
        # Laplacian pyramid splits the ENCODED features (high -> skip, low -> next stage);
        # the deepest stage is the bottleneck (no split). Do NOT revert to the prior
        # "split raw signal first, Clifford only on the high band" topology: it left the
        # descending low path unencoded (no hierarchical feature composition) and the
        # "raw-signal exact-reconstruction" framing never held for the full model anyway
        # (the decoder merges in feature space via keras.ops.add, not LaplacianPyramidLevel.merge).
        x = inputs
        skips: List[Any] = []
        feat = None
        for i in range(self.num_levels):
            e = self.enc_proj[i](x)
            for blk in self.enc_blocks[i]:
                e = blk(e, training=training)
            if i < self.num_levels - 1:
                low, high = self.pyramid_levels[i].split(e)
                skips.append(high)
                x = low
            else:
                feat = e

        for i in range(self.num_levels - 2, -1, -1):
            up = self.dec_up[i](feat)
            up = self.dec_align[i](up)
            merged = keras.ops.add(up, skips[i])
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
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "kernel_regularizer": keras.regularizers.serialize(
                    self.kernel_regularizer
                ),
            }
        )
        return config

    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CliffordLaplacianUNet":
        config = dict(config)
        if config.get("kernel_initializer") is not None:
            config["kernel_initializer"] = keras.initializers.deserialize(
                config["kernel_initializer"]
            )
        if config.get("kernel_regularizer") is not None:
            config["kernel_regularizer"] = keras.regularizers.deserialize(
                config["kernel_regularizer"]
            )
        return cls(**config)

    # ------------------------------------------------------------------
    # Variants
    # ------------------------------------------------------------------

    @classmethod
    def from_variant(
        cls,
        variant: str,
        in_channels: int = 3,
        **kwargs: Any,
    ) -> "CliffordLaplacianUNet":
        """Create a :class:`CliffordLaplacianUNet` from a named variant.

        Args:
            variant: One of ``"small"``, ``"base"``, ``"large"``.
            in_channels: Number of input/output image channels (3 for RGB).
                Defaults to ``3``.
            **kwargs: Override any default hyperparameter (kwargs win over the
                variant defaults).

        Returns:
            A configured (un-built) :class:`CliffordLaplacianUNet`.

        Raises:
            ValueError: If ``variant`` is not a key of ``MODEL_VARIANTS``.

        Example:
            ```python
            model = CliffordLaplacianUNet.from_variant("base", in_channels=1)
            ```
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant {variant!r}. Available: {list(cls.MODEL_VARIANTS)}"
            )
        defaults = dict(cls.MODEL_VARIANTS[variant])
        defaults.update(kwargs)
        logger.info(f"Creating CliffordLaplacianUNet-{variant.upper()}")
        logger.info(
            f"from_variant received in_channels={in_channels}, "
            f"overrides={kwargs}"
        )
        return cls(in_channels=in_channels, **defaults)

    # ------------------------------------------------------------------

    def summary(self, **kwargs) -> None:
        """Print the Keras model summary plus a config line.

        Args:
            **kwargs: Forwarded verbatim to :meth:`keras.Model.summary`.
        """
        super().summary(**kwargs)
        logger.info(
            f"CliffordLaplacianUNet config: levels={self.num_levels}, "
            f"channels={list(self.level_channels)}, "
            f"blocks={list(self.level_blocks)}, "
            f"cli_mode={self.cli_mode}, ctx_mode={self.ctx_mode}"
        )


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------


def create_clifford_laplacian_unet(
    variant: str = "small",
    in_channels: int = 3,
    **kwargs: Any,
) -> "CliffordLaplacianUNet":
    """Create a :class:`CliffordLaplacianUNet` deterministic image autoencoder.

    **Intent**: Thin convenience delegate to
    :meth:`CliffordLaplacianUNet.from_variant` so callers can build a named
    variant in one call without importing the class directly.

    Args:
        variant: Named model variant; one of ``"small"``, ``"base"``,
            ``"large"``. ``"small"`` is a 3-level pyramid; ``"base"``/``"large"``
            are 4-level (inputs must be divisible by ``2 ** num_levels`` -> 8 for
            ``"small"``, 16 for the 4-level variants). Defaults to ``"small"``.
        in_channels: Number of input/output image channels (3 for RGB). Defaults
            to ``3``.
        **kwargs: Keyword overrides forwarded to the constructor; these take
            precedence over the variant defaults.

    Returns:
        A configured (un-built) :class:`CliffordLaplacianUNet`.

    Example:
        ```python
        model = create_clifford_laplacian_unet("small", in_channels=3)

        # Override a variant default.
        model = create_clifford_laplacian_unet("base", blur_trainable=True)
        ```
    """
    return CliffordLaplacianUNet.from_variant(
        variant, in_channels=in_channels, **kwargs
    )

# ---------------------------------------------------------------------
