"""
CliffordNet hierarchical Laplacian U-Net image autoencoder.

This module builds a deterministic image autoencoder that fuses an EXPLICIT,
inspectable Laplacian split/merge with CliffordNet geometric-algebra feature
extraction in a standard hierarchical (encode-then-split) U-Net topology. At
each encoder stage a ``CliffordNetBlock`` group ENCODES the full incoming signal
at the current scale; a :class:`LaplacianPyramidLevel` then splits the ENCODED
FEATURES into a high band (kept as the decoder skip) and a low band (fed to the
next, coarser stage). The deepest stage does not split -- it serves directly as
the bottleneck. The decoder collapses bottom-up, merging upsampled coarser
features with the level skip in FEATURE space (``keras.ops.add``) and refining
each merge with Clifford blocks. The model returns ``{"reconstruction": y}`` with
``y`` of identical shape to the input.

What this is (and is NOT)
-------------------------
The Laplacian split here operates on ENCODED FEATURE tensors, not on the raw
input signal. The pyramid is therefore a *learnable multi-scale high/low
decomposition* baked into the U-Net, NOT an invertible raw-signal codec: there
is NO raw-signal exact-reconstruction property. (The full model never delivered
one anyway -- the decoder merges in feature space via ``keras.ops.add``, not via
``LaplacianPyramidLevel.merge``.) ``LaplacianPyramidLevel.split`` remains
algebraically invertible per call (``high + up(low) == x`` up to float rounding,
on whatever tensor it is given), but the surrounding model is a learned
autoencoder whose reconstruction quality is trained, not guaranteed by algebra.

Channel-bookkeeping
-------------------
The low/high split is *channel-preserving*, so it applies cleanly to the encoded
``level_channels[i]``-wide feature tensor at each split stage:

- ``low  = down(blur(e))``  via ``GaussianFilter(strides=(1,1), padding="same")``
  then ``BlurPool2D(strides=2)`` (both preserve C). Result ``(B, H/2, W/2, C_i)``.
- ``up`` via ``UpSampling2D(bilinear)`` -- preserves channels, doubles H, W.
- ``high = e - up(low)``: shape ``(B, H, W, C_i)``, same channels as ``e``; this
  high band is used directly as the decoder skip (no extra Clifford on it).

``CliffordNetBlock`` is ISOTROPIC, so every channel change is an external 1x1
``keras.layers.Conv2D`` (``enc_proj`` on the encoder side, ``dec_align`` on the
decoder side). The ``LaplacianPyramidLevel`` helper lives in
``dl_techniques.layers.laplacian_filter``; this module defines the model
(``CliffordLaplacianUNet``) and imports the helper from there.
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
    """Deterministic image autoencoder: hierarchical Laplacian U-Net + Clifford blocks.

    **Intent**: Provide a hierarchical (encode-then-split) Laplacian U-Net that
    fuses an EXPLICIT, inspectable high/low split/merge with isotropic CliffordNet
    geometric-algebra feature extraction. At each encoder stage a Clifford block
    group ENCODES the full incoming signal at that scale; a
    :class:`LaplacianPyramidLevel` then splits the ENCODED FEATURES into a high
    band (the decoder skip) and a low band (the next, coarser stage's input). The
    deepest stage does not split -- it IS the bottleneck. The split operates on
    feature tensors, so there is NO raw-signal exact-reconstruction property; the
    pyramid is a learnable multi-scale decomposition, and the model is a trained
    autoencoder (see the module docstring). It returns a single-output dict
    ``{"reconstruction": y}`` with ``y`` of identical shape to the input.

    Per-stage flow (``L = num_levels``):

    - **Encoder, split stages** (``i = 0 .. L-2``): ``e = enc_proj[i](x)`` (1x1
      ``Conv2D`` ``C_in``/``ch[i-1]`` -> ``ch[i]``), then ``e = enc_blocks[i](e)``
      (Clifford x N, isotropic) encodes the FULL signal at this scale.
      ``low, high = pyramid_levels[i].split(e)``; ``high`` (full res, ``ch[i]``)
      is appended as the decoder skip WITH NO extra Clifford on it; ``low`` (H/2)
      feeds stage ``i+1``.
    - **Encoder, deepest stage** (``i = L-1``): ``e = enc_proj[L-1](x)``; ``e =
      enc_blocks[L-1](e)``; NO split. This ``e`` is the bottleneck feature.
    - **Decoder** (``i = L-2 .. 0``): ``up = dec_up[i](feat)`` (bilinear x2);
      ``up = dec_align[i](up)`` (1x1 -> ``ch[i]``); ``merged = add(up, skip_i)``
      (FEATURE-SPACE merge); ``feat = dec_blocks[i](merged)`` (Clifford x N).
    - **Head**: ``out_conv(feat)`` (1x1 -> ``in_channels``).

    **Architecture**:
    ```
    Input(B, H, W, C_in)
           |
      +----v---------------------------------------------------+
      | Encoder split stages (i = 0 .. L-2)                   |
      |   e         = enc_proj[i](x)        # 1x1 -> ch[i]    |
      |   e         = enc_blocks[i](e)      # Clifford x N    |
      |   low, high = pyramid_levels[i].split(e)              |
      |   skip_i    = high                  # decoder skip    |
      |   x         = low                   # feeds stage i+1 |
      +----v---------------------------------------------------+
           |  (coarsest low band, ch[L-2])
      +----v---------------------------------------------------+
      | Deepest stage = bottleneck (i = L-1, NO split)        |
      |   e    = enc_proj[L-1](x)           # 1x1 -> ch[L-1]  |
      |   feat = enc_blocks[L-1](e)         # Clifford x N    |
      +----v---------------------------------------------------+
           |
      +----v---------------------------------------------------+
      | Decoder (i = L-2 .. 0)                                |
      |   up     = dec_up[i](feat)          # bilinear x2     |
      |   up     = dec_align[i](up)         # 1x1 -> ch[i]    |
      |   merged = add(up, skip_i)          # feature merge   |
      |   feat   = dec_blocks[i](merged)    # Clifford x N    |
      +----v---------------------------------------------------+
           |
      out_conv(feat)                        # 1x1 -> C_in
           |
    {"reconstruction": (B, H, W, C_in)}
    ```

    Divisibility contract: input H, W must be divisible by ``2**(num_levels-1)``.
    Each SPLIT stage halves the spatial resolution via :class:`BlurPool2D`, and
    there are ``num_levels-1`` split stages (the deepest stage is the bottleneck
    and does not downsample).

    Args:
        in_channels: Number of input/output image channels (e.g. 3 for RGB).
            Must be positive. Defaults to ``3``.
        level_channels: Per-level working width for the Clifford feature track.
            Its length defines ``num_levels``. Defaults to ``(32, 64, 128)``.
        level_blocks: Number of Clifford blocks per level (encoder and decoder);
            the deepest stage (which serves as the bottleneck) uses
            ``level_blocks[-1]``. Must match ``num_levels``. Defaults to
            ``(2, 2, 2)``.
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
        ``width`` must be divisible by ``2 ** (num_levels - 1)``.

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
        divisible by ``2**(num_levels-1)`` (one halving per SPLIT stage; there
        are ``num_levels-1`` split stages, the deepest stage being the
        non-downsampling bottleneck).

        Args:
            input_shape: Input shape ``(B, H, W, C)``. A 3D shape ``(H, W, C)``
                (e.g. from ``summary()``) is accepted and a dummy batch dim is
                prepended.

        Raises:
            ValueError: If a known static ``height`` or ``width`` is not divisible
                by ``2 ** (num_levels - 1)``.
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
            ``"large"``. ``"small"`` is a 3-level model; ``"base"``/``"large"``
            are 4-level. Inputs must be divisible by ``2 ** (num_levels - 1)``:
            4 for the 3-level ``"small"``, 8 for the 4-level ``"base"``/``"large"``
            variants. Defaults to ``"small"``.
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
