"""DocScanner's rectification-stage parts, and the width table the whole port reads.

This module will hold the RAFT-lineage building blocks of DocScanner's second
stage -- the ``align_corners`` coordinate adapter, ``coords_grid``, the convex
``8x`` upsample, the instance-norm helper, the residual block, the feature
encoder, ``SepConvGRU``, the motion encoder, the flow/mask heads and the update
block. The sampling primitives themselves (``coords_grid``,
``sample_at_pixel_coords``, ``convex_upsample``) live in the sibling ``warp.py``
-- see its docstring for why the port's one sampling convention is a module of
its own. At this point this module carries the module-level CONSTANTS, the
single ``_VARIANT_SPEC`` width table, the instance-norm stand-in, and the
feature encoder with its residual unit.

Why the width table lives HERE and not in ``model.py``
------------------------------------------------------
The package's import graph runs ``model.py -> components.py -> (nothing in this
package)``. ``components.py`` needs every channel width, because the encoder,
the GRU and the motion encoder are all built here; if the table lived in
``model.py`` those widths would have to travel back down the import edge, which
is a cycle. So the table is defined at the bottom of the graph and ``model.py``
imports it, rather than the reverse. Nothing about the table is private to
``model.py``: it is the port's single source of every channel count, which is
exactly why it has to sit where every consumer can reach it.

Registration convention
-----------------------
Classes added to this module register under
``dl_techniques.models.doc_scanner.components`` -- the key strips BOTH the
``vision`` family and the ``image_restoration`` subfamily, per the repo-wide
convention. It is NOT the full import path. ``DocScannerResidualBlock`` and
``DocScannerFeatureEncoder`` are registered that way below.

The upstream reference
----------------------
Every constant below cites the line of the upstream PyTorch release it was read
from, at ``/media/arxwn/data_fast/repositories/DocScanner`` (commit as cloned
2026-09-10). One constant -- :data:`SEQUENCE_LOSS_GAMMA` -- has no upstream line
to cite, because the release ships inference code only and contains no training
loop at all; it is cited to the paper instead, and labelled as such.

References:
    - Feng et al., 2021. DocScanner: Robust Document Image Rectification with
      Progressive Learning. (https://arxiv.org/abs/2110.14968), v2.
    - Upstream release: https://github.com/fh2019ustc/DocScanner --
      ``model.py``, ``update.py``, ``extractor.py``, ``seg.py``,
      ``inference.py``.
"""

import keras
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------
# Constants.
#
# Every value below is transcribed from the upstream release, and the citation
# beside it is the line that USES the value, never a constructor default that
# no call site ever reaches. See the `_VARIANT_SPEC` anchor at the bottom of
# this file for why that distinction is load-bearing here.
# ---------------------------------------------------------------------

# `extractor.py:21` and `:93` construct `nn.InstanceNorm2d(...)`, whose torch
# default is `eps=1e-5`. The Keras stand-in for per-sample instance norm is
# `keras.layers.GroupNormalization(groups=C)`, whose OWN default epsilon is
# `1e-3` -- 100x larger, with no shape symptom and no warning. Every
# construction site in this port therefore passes this value explicitly.
# `norms/factory.py`'s 18 registry keys contain neither instance nor group
# norm, so the layer is constructed directly rather than through
# `create_normalization_layer`; `layers/CLAUDE.md` requires exactly this when a
# normalization layer is built by hand -- state the epsilon, cite the source.
INSTANCE_NORM_EPSILON: float = 1e-5

# The rectifier runs its refinement loop on a 1/8-resolution coordinate field
# and convex-upsamples back. `model.py:49-50` builds `coords_grid(N, H // 8,
# W // 8)`; `model.py:65` reshapes the upsampled flow to `(N, 2, 8 * H, 8 * W)`.
# The feature encoder's total stride is the same 8 (stem stride 2 at
# `extractor.py:96`, then `layer2`/`layer3` at stride 2, `extractor.py:100-101`).
SPATIAL_DIVISOR: int = 8

# The number of GRU refinement iterations. `model.py:67` declares `iters=12` as
# the forward-pass default AND `inference.py:28` passes `iters=12` explicitly at
# the only call site, so the two agree -- this is not a dead default.
REFINE_ITERATIONS: int = 12

# The exponential decay of the K-iteration sequence loss,
# `L = sum_{k=1..K} gamma^(K-k) * L^(k)`, so the LAST iteration carries weight
# exactly 1 and earlier ones decay backwards.
#
# NOT FROM THE CODE. The upstream release ships inference only -- it contains no
# training loop, no loss and no optimizer -- so there is no file:line to cite.
# The value is the paper's: arXiv:2110.14968v2, Eq. 9 (`gamma = 0.85`, `K = 12`).
# Anyone reconciling this port against the upstream checkout will not find it
# there, and that is expected rather than a transcription error.
SEQUENCE_LOSS_GAMMA: float = 0.85

# The composite pipeline binarizes the segmenter's confidence map before using
# it as a multiplicative background mask: `inference.py:25`,
# `msk = (msk > 0.5).float()`.
SEG_MASK_THRESHOLD: float = 0.5

# The backward map the rectifier emits is in PIXEL units; the upstream
# inference wrapper rescales it to the `[-1, 1]` range its sampler wants with
# `bm = (2 * (bm / 286.8) - 1) * 0.99` -- `inference.py:29`, one line carrying
# both numbers. The divisor is NOT the 288 training resolution: it is 286.8,
# and the trailing 0.99 shrinks the range slightly inside the image domain.
# Neither number is explained upstream or in the paper; both are transcribed
# exactly rather than "corrected" to 288 / 1.0.
BM_CALIBRATION_DIVISOR: float = 286.8
BM_CALIBRATION_SCALE: float = 0.99

# ---------------------------------------------------------------------


# DECISION plan-2026-09-10T065432-05fcb6dd/D-006: read these widths from the CALL
# SITES (`model.py:35-39`, `update.py:88`). Do NOT read them from `update.py:18,36`'s
# `hidden_dim=128, input_dim=192+128` defaults, which no call site reaches: the GRU is
# width-agnostic, so a 128-wide hidden state passes EVERY shape test. Worse, `192+128`
# equals the live 320 (D-007), so the wrong reading looks half-confirmed. No width
# literal belongs anywhere else in this package. See decisions.md D-006, D-007.
_VARIANT_SPEC: Dict[str, Dict[str, Any]] = {
    "docscanner-l": {
        # `model.py:35-36`: `self.hidden_dim = hdim = 160`, `self.context_dim = 160`.
        # The encoder's 320 output channels are split in half into the GRU's
        # hidden state (`tanh`) and its context input (`relu`), `model.py:74-76`,
        # which is why these two are equal and sum to `fnet_output_dim`.
        "hidden_dim": 160,
        "context_dim": 160,
        # `update.py:88`: `SepConvGRU(hidden_dim=hidden_dim, input_dim=160+160)`.
        # This overrides `update.py:36`'s dead `input_dim=192+128` default.
        "gru_input_dim": 320,
        # `model.py:38`: `BasicEncoder(output_dim=320, norm_fn='instance')`, and
        # `extractor.py:104`'s `conv2` maps 240 -> `output_dim` with a 1x1.
        "fnet_output_dim": 320,
        # `extractor.py:96`: the 7x7 stride-2 stem emits 80 channels. The
        # reference then declares `norm1 = nn.InstanceNorm2d(64)` at
        # `extractor.py:93` and applies it to those 80 channels -- inert under
        # torch's `affine=False` (no parameters are allocated), but a Keras
        # affine normalization allocates by shape and would CRASH. 80 is the
        # true width; 64 is a latent upstream bug.
        "encoder_stem_channels": 80,
        # `extractor.py:98-101`: `in_planes = 80`, then `_make_layer(80,
        # stride=1)`, `_make_layer(160, stride=2)`, `_make_layer(240, stride=2)`.
        "encoder_stage_channels": (80, 160, 240),
    },
}

# The motion encoder's, flow head's and mask head's internal widths
# (`update.py:66-70`, `:89`, `:91-94`) are DERIVED from the row above rather
# than independent -- e.g. the motion encoder's 158 is `hidden_dim - 2`, and the
# mask head's 576 is `SPATIAL_DIVISOR ** 2 * 9`. They are added to this table by
# the step that builds those blocks, expressed as derivations, so that no width
# literal ever appears at a construction site.

# ---------------------------------------------------------------------
# The instance-norm stand-in, and the initializer the reference uses.
# ---------------------------------------------------------------------


# DECISION plan-2026-09-10T065432-05fcb6dd/D-011: this is a NON-AFFINE norm --
# `center=False, scale=False` -- because torch's `nn.InstanceNorm2d` defaults to
# `affine=False` and allocates NO per-channel scale/offset at all
# (`extractor.py:21-24`, `:93`). Do NOT "helpfully" turn the affine on: it is
# precisely the absence of parameters that makes the reference's
# `InstanceNorm2d(64)`-fed-80-channels line inert upstream, so an affine port
# would be a divergence dressed as a fix. Do NOT route this through
# `create_normalization_layer` either -- the norms factory's 18 keys contain
# neither instance nor group norm (F-07), so this is a direct construction by
# necessity, not by preference. And do NOT drop the explicit `epsilon`: Keras'
# GroupNormalization defaults to 1e-3 against torch's 1e-5, a silent 100x with
# no shape symptom. See decisions.md D-011.
def _instance_norm(channels: int, name: str) -> keras.layers.GroupNormalization:
    """Build the Keras equivalent of ``nn.InstanceNorm2d(channels)``.

    ``GroupNormalization`` with ``groups == channels`` normalizes each channel
    of each sample independently over the spatial axes -- which IS instance
    normalization. The two precedents in this repo that do the same are
    ``models/vision_language/ideogram4/vae.py:52-64`` and
    ``models/vision_language/sam/sam3/maskformer_segmentation.py:172-183``.

    :param channels: The number of channels the layer will see. It becomes the
        group count, so it must equal the trailing dimension of the tensor this
        layer is applied to -- a mismatch is a hard ``ValueError`` from Keras
        when ``channels`` does not divide the real width, and a SILENT
        mis-grouping when it does.
    :type channels: int
    :param name: Sub-layer name.
    :type name: str
    :return: A non-affine, ``epsilon=1e-5`` per-channel normalization layer.
    :rtype: keras.layers.GroupNormalization
    """
    return keras.layers.GroupNormalization(
        groups=channels,
        axis=-1,
        epsilon=INSTANCE_NORM_EPSILON,
        center=False,
        scale=False,
        name=name,
    )


def _kaiming_fan_out() -> keras.initializers.Initializer:
    """A FRESH ``kaiming_normal_(mode='fan_out', nonlinearity='relu')``.

    `extractor.py:106-108` re-initializes every ``nn.Conv2d`` this way after
    construction. Keras' ``"he_normal"`` is fan_IN, so it is not the same
    distribution; ``VarianceScaling(scale=2.0, mode="fan_out",
    distribution="untruncated_normal")`` is exactly torch's.

    A new instance is returned per call rather than a shared module-level one:
    a single ``Initializer`` object handed to many layers is a known
    aliasing hazard in this repo.

    :return: A newly constructed initializer.
    :rtype: keras.initializers.Initializer
    """
    return keras.initializers.VarianceScaling(
        scale=2.0, mode="fan_out", distribution="untruncated_normal"
    )


# `extractor.py:115-118`: `_make_layer` builds exactly TWO `ResidualBlock`s per
# stage, the first carrying the stage's stride and the second always stride 1.
_ENCODER_BLOCKS_PER_STAGE: int = 2


def _ceil_div(value: Optional[int], divisor: int) -> Optional[int]:
    """``ceil(value / divisor)``, propagating ``None`` for an unknown axis."""
    if value is None:
        return None
    return -(-value // divisor)


# ---------------------------------------------------------------------
# The residual unit and the feature encoder.
# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.doc_scanner.components")
class DocScannerResidualBlock(keras.layers.Layer):
    """The feature encoder's residual unit (``extractor.py:4-40``).

    Architecture:

    .. code-block:: text

        Input [B, H, W, C_in]
          |                                   \\
        Conv2D 3x3 stride=s -> InstanceNorm     Conv2D 1x1 stride=s   (only
          -> ReLU                               -> InstanceNorm        if s != 1)
        Conv2D 3x3 stride=1 -> InstanceNorm      |
          -> ReLU                                |
          \\____________________ Add ____________/
                                 |
                                ReLU
                                 |
                        Output [B, ceil(H/s), ceil(W/s), filters]

    Two details of the reference are load-bearing and easy to lose in
    translation:

    1. **The second ReLU is applied BEFORE the add, and a third after it.**
       ``extractor.py:33-39`` reads ``y = relu(norm2(conv2(y)))`` and then
       ``return relu(x + y)``. That is not the usual ResNet ordering (which
       adds the un-activated branch), and it is why
       :class:`~dl_techniques.layers.standard_blocks.BasicBlock` is not reused
       here -- see the class-level note in
       :class:`DocScannerFeatureEncoder`.
    2. **The shortcut is an identity ONLY when ``stride == 1``**, and the
       reference then also requires ``C_in == filters``, because it adds the
       raw input. This class raises rather than broadcasting.

    :param filters: Output channel count; also the width of both convolutions.
    :type filters: int
    :param stride: Stride of the first convolution and of the projection
        shortcut. ``1`` selects the identity shortcut.
    :type stride: int
    :param kwargs: Forwarded to ``keras.layers.Layer``.
    :type kwargs: Any
    :raises ValueError: If ``filters`` or ``stride`` is not positive.
    """

    def __init__(self, filters: int, stride: int = 1, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        if filters <= 0:
            raise ValueError(f"filters must be positive, got {filters}")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")

        self.filters = filters
        self.stride = stride

        # DECISION plan-2026-09-10T065432-05fcb6dd/D-012: a stride-2 3x3 conv is
        # padded EXPLICITLY (ZeroPadding2D(1) + "valid"), never with "same".
        # Keras/TF "same" at stride 2 puts pad_before=0, pad_after=1 and so
        # samples input centres at 2j+1; torch's `padding=1` pads 1/1 and
        # samples 2j. Both emit the SAME output shape, so no shape test can see
        # the one-pixel shift, and this port's whole coordinate contract
        # (warp.py, invariant #1) is stated in absolute pixels. At stride 1,
        # "same" IS torch's `padding=1` exactly, so the padding layer is not
        # created there. See decisions.md D-012.
        self.pad1: Optional[keras.layers.ZeroPadding2D] = None
        conv1_padding = "same"
        if stride != 1:
            self.pad1 = keras.layers.ZeroPadding2D(padding=1, name="pad1")
            conv1_padding = "valid"

        self.conv1 = keras.layers.Conv2D(
            filters=filters,
            kernel_size=3,
            strides=stride,
            padding=conv1_padding,
            use_bias=True,
            kernel_initializer=_kaiming_fan_out(),
            name="conv1",
        )
        self.norm1 = _instance_norm(filters, name="norm1")
        self.conv2 = keras.layers.Conv2D(
            filters=filters,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=True,
            kernel_initializer=_kaiming_fan_out(),
            name="conv2",
        )
        self.norm2 = _instance_norm(filters, name="norm2")

        # `extractor.py:26-30`: the projection exists ONLY for stride != 1.
        # A 1x1 conv at stride 2 with "valid" padding reads input index 2j,
        # which is what torch's zero-padding 1x1 does, so no explicit pad is
        # needed on this branch.
        self.downsample_conv: Optional[keras.layers.Conv2D] = None
        self.norm3: Optional[keras.layers.GroupNormalization] = None
        if stride != 1:
            self.downsample_conv = keras.layers.Conv2D(
                filters=filters,
                kernel_size=1,
                strides=stride,
                padding="valid",
                use_bias=True,
                kernel_initializer=_kaiming_fan_out(),
                name="downsample_conv",
            )
            self.norm3 = _instance_norm(filters, name="norm3")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build every sub-layer explicitly, propagating shapes by hand.

        :param input_shape: ``(batch, height, width, channels)``.
        :type input_shape: tuple
        :raises ValueError: If the input is not rank 4, or if ``stride == 1``
            and the input channel count differs from ``filters`` (the identity
            shortcut would then be an illegal add).
        """
        if self.built:
            return

        if len(input_shape) != 4:
            raise ValueError(
                f"Expected 4D input shape (batch, height, width, channels), "
                f"got {len(input_shape)}D: {input_shape}"
            )

        in_channels = input_shape[-1]
        if (
            self.stride == 1
            and in_channels is not None
            and in_channels != self.filters
        ):
            raise ValueError(
                f"DocScannerResidualBlock with stride=1 uses an IDENTITY "
                f"shortcut, so its input channel count ({in_channels}) must "
                f"equal filters ({self.filters}). The reference only ever "
                f"builds a projection shortcut when stride != 1 "
                f"(extractor.py:26-30); use stride=2 or match the widths."
            )

        main_shape = tuple(input_shape)
        if self.pad1 is not None:
            self.pad1.build(main_shape)
            main_shape = self.pad1.compute_output_shape(main_shape)

        self.conv1.build(main_shape)
        mid_shape = self.conv1.compute_output_shape(main_shape)
        self.norm1.build(mid_shape)
        self.conv2.build(mid_shape)
        self.norm2.build(mid_shape)

        if self.downsample_conv is not None:
            self.downsample_conv.build(tuple(input_shape))
            self.norm3.build(
                self.downsample_conv.compute_output_shape(tuple(input_shape))
            )

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Run the two-conv branch and add the shortcut.

        :param inputs: Tensor of shape ``(batch, height, width, channels)``.
        :type inputs: keras.KerasTensor
        :param training: Standard Keras training flag, forwarded to the norms.
        :type training: Optional[bool]
        :return: Tensor of shape
            ``(batch, ceil(height / stride), ceil(width / stride), filters)``.
        :rtype: keras.KerasTensor
        """
        branch = inputs
        if self.pad1 is not None:
            branch = self.pad1(branch)
        branch = keras.ops.relu(
            self.norm1(self.conv1(branch), training=training)
        )
        # The ReLU here is INSIDE the branch, before the add (extractor.py:34).
        branch = keras.ops.relu(
            self.norm2(self.conv2(branch), training=training)
        )

        shortcut = inputs
        if self.downsample_conv is not None:
            shortcut = self.norm3(
                self.downsample_conv(inputs), training=training
            )

        return keras.ops.relu(shortcut + branch)

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Spatial axes divide by ``stride`` (rounding up); channels become ``filters``.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple
        :return: The output shape tuple.
        :rtype: tuple
        """
        batch, height, width, _ = input_shape
        return (
            batch,
            _ceil_div(height, self.stride),
            _ceil_div(width, self.stride),
            self.filters,
        )

    def get_config(self) -> Dict[str, Any]:
        """Return every constructor argument.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "stride": self.stride,
        })
        return config


@register_dl_technique("dl_techniques.models.doc_scanner.components")
class DocScannerFeatureEncoder(keras.layers.Layer):
    """DocScanner's stride-8 feature encoder (``extractor.py:84-134``).

    Architecture, at the ``docscanner-l`` widths:

    .. code-block:: text

        Input [B, H, W, 3]
          -> ZeroPadding2D(3) -> Conv2D 7x7 stride=2 -> 80ch    [B, H/2, W/2, 80]
          -> InstanceNorm(80) -> ReLU
          -> stage 1: 2 x ResidualBlock(80,  stride 1, 1)       [B, H/2, W/2, 80]
          -> stage 2: 2 x ResidualBlock(160, stride 2, 1)       [B, H/4, W/4, 160]
          -> stage 3: 2 x ResidualBlock(240, stride 2, 1)       [B, H/8, W/8, 240]
          -> Conv2D 1x1 -> 320ch, NO norm and NO activation     [B, H/8, W/8, 320]

    The trailing ``1x1`` is bare on purpose: ``extractor.py:104`` and
    ``:131`` have nothing after it, and the rectifier splits its 320 channels
    into a ``tanh``-ed hidden state and a ``relu``-ed context, so a ReLU here
    would half-rectify the hidden state before ``tanh`` ever saw it.

    Why this is not
    :class:`~dl_techniques.layers.standard_blocks.BasicBlock`
    ---------------------------------------------------------
    ``BasicBlock`` is close -- two 3x3 convs, an optional 1x1 projection, an
    add -- but it differs on three points that all matter here, and the first
    is decisive: its normalization is chosen by ``normalization_type`` and
    built through ``create_normalization_layer``, whose registry contains
    neither instance nor group norm (F-07), so it CANNOT build the norm this
    encoder needs. It also applies its second activation only AFTER the add,
    where the reference applies one on the branch and another after
    (``extractor.py:34,39``), and it builds its convolutions bias-free where
    torch's ``nn.Conv2d`` defaults to ``bias=True``. Reusing it would mean
    changing all three in a shared layer with other callers.

    :param stem_channels: Width of the 7x7 stride-2 stem, and therefore the
        group count of ``norm1``. This is **80**, not the reference's 64 --
        see the note at the construction site.
    :type stem_channels: int
    :param stage_channels: Output width of each residual stage, outermost
        first. The first stage runs at stride 1 and every later stage at
        stride 2 (``extractor.py:99-101``).
    :type stage_channels: Sequence[int]
    :param output_dim: Channel count of the trailing 1x1 convolution.
    :type output_dim: int
    :param kwargs: Forwarded to ``keras.layers.Layer``.
    :type kwargs: Any
    :raises ValueError: If any width is not positive, if ``stage_channels`` is
        empty, or if the resulting total stride is not
        :data:`SPATIAL_DIVISOR`.
    """

    def __init__(
            self,
            stem_channels: int,
            stage_channels: Sequence[int],
            output_dim: int,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if stem_channels <= 0:
            raise ValueError(
                f"stem_channels must be positive, got {stem_channels}"
            )
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        stage_channels = tuple(int(width) for width in stage_channels)
        if not stage_channels:
            raise ValueError("stage_channels must name at least one stage")
        if any(width <= 0 for width in stage_channels):
            raise ValueError(
                f"every stage width must be positive, got {stage_channels}"
            )

        # Stem stride 2, then stride 2 for every stage after the first.
        total_stride = 2 ** len(stage_channels)
        if total_stride != SPATIAL_DIVISOR:
            raise ValueError(
                f"{len(stage_channels)} stages give a total stride of "
                f"{total_stride}, but the rectifier refines its coordinate "
                f"field at 1 / {SPATIAL_DIVISOR} resolution and convex-"
                f"upsamples by exactly that factor. The two must agree."
            )

        self.stem_channels = stem_channels
        self.stage_channels = stage_channels
        self.output_dim = output_dim

        # `extractor.py:95`: Conv2d(3, 80, kernel_size=7, stride=2, padding=3).
        # D-012 applies here too: the explicit pad reproduces torch's 3/3 split
        # instead of TF "same"'s 2/3.
        self.pad1 = keras.layers.ZeroPadding2D(padding=3, name="pad1")
        self.conv1 = keras.layers.Conv2D(
            filters=stem_channels,
            kernel_size=7,
            strides=2,
            padding="valid",
            use_bias=True,
            kernel_initializer=_kaiming_fan_out(),
            name="conv1",
        )

        # DECISION plan-2026-09-10T065432-05fcb6dd/D-011: `norm1` is built at
        # `stem_channels` (80), NOT at the reference's literal 64. The upstream
        # `nn.InstanceNorm2d(64)` at `extractor.py:93` (and `BatchNorm2d(64)` at
        # `:90`) is applied to the 80-channel output of `conv1`
        # (`extractor.py:95`, `:124-125`) -- a latent upstream bug that is inert
        # ONLY because torch's `affine=False` default allocates no per-channel
        # parameters, so the channel argument is never used. Keras
        # `GroupNormalization` uses it as the GROUP COUNT: 64 does not divide 80
        # and raises, and a divisor that did (16, say) would silently normalize
        # in 5-channel groups. Do NOT "restore fidelity" by writing 64 here.
        # See decisions.md D-011; guarded by TestTheStemNormIsEightyWideNotSixtyFour.
        self.norm1 = _instance_norm(stem_channels, name="norm1")

        # Flat list, never List[List[Layer]]: a nested sub-layer list silently
        # restores fresh kernels on a `.keras` round trip.
        self.blocks: List[DocScannerResidualBlock] = []
        for stage_index, width in enumerate(stage_channels):
            stage_stride = 1 if stage_index == 0 else 2
            for block_index in range(_ENCODER_BLOCKS_PER_STAGE):
                self.blocks.append(DocScannerResidualBlock(
                    filters=width,
                    stride=stage_stride if block_index == 0 else 1,
                    name=f"layer{stage_index + 1}_block{block_index}",
                ))

        # `extractor.py:104`: Conv2d(240, output_dim, kernel_size=1). Nothing
        # follows it.
        self.conv2 = keras.layers.Conv2D(
            filters=output_dim,
            kernel_size=1,
            strides=1,
            padding="valid",
            use_bias=True,
            kernel_initializer=_kaiming_fan_out(),
            name="conv2",
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the stem, every residual block and the output 1x1 explicitly.

        :param input_shape: ``(batch, height, width, channels)``.
        :type input_shape: tuple
        :raises ValueError: If the input is not rank 4.
        """
        if self.built:
            return

        if len(input_shape) != 4:
            raise ValueError(
                f"Expected 4D input shape (batch, height, width, channels), "
                f"got {len(input_shape)}D: {input_shape}"
            )

        shape = tuple(input_shape)
        self.pad1.build(shape)
        shape = self.pad1.compute_output_shape(shape)
        self.conv1.build(shape)
        shape = self.conv1.compute_output_shape(shape)
        self.norm1.build(shape)

        for block in self.blocks:
            block.build(shape)
            shape = block.compute_output_shape(shape)

        self.conv2.build(shape)

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Run stem, stages and the output projection.

        :param inputs: Tensor of shape ``(batch, height, width, 3)``.
        :type inputs: keras.KerasTensor
        :param training: Standard Keras training flag.
        :type training: Optional[bool]
        :return: Tensor of shape ``(batch, ceil(height / 8), ceil(width / 8),
            output_dim)``.
        :rtype: keras.KerasTensor
        """
        x = self.conv1(self.pad1(inputs))
        x = keras.ops.relu(self.norm1(x, training=training))

        for block in self.blocks:
            x = block(x, training=training)

        return self.conv2(x)

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Both spatial axes divide by :data:`SPATIAL_DIVISOR` (rounding up).

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple
        :return: The output shape tuple.
        :rtype: tuple
        """
        batch, height, width, _ = input_shape
        for _ in range(len(self.stage_channels)):
            height = _ceil_div(height, 2)
            width = _ceil_div(width, 2)
        return (batch, height, width, self.output_dim)

    def get_config(self) -> Dict[str, Any]:
        """Return every constructor argument.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "stem_channels": self.stem_channels,
            "stage_channels": list(self.stage_channels),
            "output_dim": self.output_dim,
        })
        return config


__all__: List[str] = [
    "INSTANCE_NORM_EPSILON",
    "SPATIAL_DIVISOR",
    "REFINE_ITERATIONS",
    "SEQUENCE_LOSS_GAMMA",
    "SEG_MASK_THRESHOLD",
    "BM_CALIBRATION_DIVISOR",
    "BM_CALIBRATION_SCALE",
    "DocScannerResidualBlock",
    "DocScannerFeatureEncoder",
]
