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
feature encoder with its residual unit, and the separable recurrent update
core ``SepConvGRU``.

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

# The width of a coordinate/flow field: one x channel and one y channel.
# `model.py:26-28` stacks exactly two coordinate planes, `update.py:69` reads a
# 2-channel flow into the motion encoder's flow branch, and `update.py:10`
# emits 2 channels from the flow head. It is structural, not a tunable width,
# so it is a module constant rather than a `_VARIANT_SPEC` row.
FLOW_CHANNELS: int = 2

# DECISION plan-2026-09-10T065432-05fcb6dd/D-015: the size of the convex-upsample
# neighbourhood -- a 3x3 window, so 9 candidate source pixels per destination
# sub-pixel (`model.py:58`'s `F.unfold(..., [3, 3], padding=1)`). It is DEFINED
# here, in the constants module, and imported by `warp.py`; do NOT move it back
# to `warp.py`, where step 3 first put it. The import graph runs
# `warp.py -> components.py` (warp already reads `SPATIAL_DIVISOR` from here),
# so defining it in `warp.py` and importing it here for the mask head's
# `SPATIAL_DIVISOR ** 2 * CONVEX_NEIGHBOURS` would close an import CYCLE whose
# failure mode depends on which module is imported first -- i.e. it works from
# the tests and breaks from a different entry point. `warp.py` re-exports the
# name, so `from ...warp import CONVEX_NEIGHBOURS` keeps working unchanged.
# See decisions.md D-015.
CONVEX_NEIGHBOURS: int = 9

# The scalar the update block applies to its mask head's output before handing
# it to `convex_upsample`: `update.py:104`, `mask = .25 * self.mask(net)`.
#
# DECISION plan-2026-09-10T065432-05fcb6dd/D-016: this belongs to the CALL SITE,
# not to the mask head and not to `convex_upsample`. Upstream applies it in
# `BasicUpdateBlock.forward`, strictly between the head and the upsample, and
# `convex_upsample` is documented and guarded as taking RAW logits
# (`TestConvexUpsampleTakesRawLogits`). Do NOT fold it into the head's final
# convolution and do NOT re-apply it downstream: the mask is softmaxed over the
# 9-neighbour axis, so this factor is a softmax TEMPERATURE (0.25 flattens the
# distribution 4x). Dropping it or applying it twice changes every weight the
# upsample produces, is invisible to every shape, dtype and finiteness test, and
# leaves a model that still trains. See decisions.md D-016.
MASK_LOGIT_SCALE: float = 0.25

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
        # --- the update block's internal widths -------------------------
        # `update.py:67-68`: the warped-feature branch is `Conv2d(320, 240, 1)`
        # then `Conv2d(240, 160, 3)`. Its INPUT 320 is `fnet_output_dim` and is
        # inferred from the tensor, not declared. 240 and 160 are RAFT-lineage
        # literals: 240 coincides with the encoder's last stage width and 160
        # with `hidden_dim`, but upstream states no such relationship, so they
        # are transcribed as their own entries rather than "derived" from a
        # coincidence. A derivation nobody wrote down is a fabrication.
        "motion_corr_hidden": 240,
        "motion_corr_out": 160,
        # `update.py:69-70`: the flow branch is `Conv2d(2, 160, 7)` then
        # `Conv2d(160, 80, 3)`. Same reading: not derived, transcribed.
        "motion_flow_hidden": 160,
        "motion_flow_out": 80,
        # `update.py:89`: `FlowHead(hidden_dim, hidden_dim=320)` -- the CALL
        # SITE. NOT `update.py:7`'s `hidden_dim=256` constructor default, which
        # no call site reaches; that is exactly the D-006 dead-default trap, and
        # a 256-wide flow head passes every shape test this port has.
        "flow_head_hidden": 320,
        # `update.py:92`: `nn.Conv2d(hidden_dim, 288, 3, padding=1)`. A genuine
        # upstream literal with no derivation. It is NOT the 288 training
        # resolution; the collision is a coincidence and must not be turned into
        # a link between a channel count and an image size.
        "mask_head_hidden": 288,
        # --- the SEGMENTATION stage's widths ----------------------------
        #
        # These two rows belong to the OTHER stage. `seg.py:456-478` builds
        # every one of the U2NET-P's eleven RSU stages as
        # `RSU*(in_ch, 16, 64)` -- `stage1 = RSU7(in_ch, 16, 64)`,
        # `stage2 = RSU6(64, 16, 64)`, ... `stage1d = RSU7(128, 16, 64)`. The
        # "P" (pruned) variant is defined by that uniformity: the full U2NET at
        # `seg.py:344-450` varies mid/out per stage (`RSU7(3, 32, 64)`, ...),
        # and the port must not borrow ITS numbers.
        #
        # They are CALL-SITE readings, per D-006: `RSU7.__init__`'s signature
        # default is `mid_ch=12, out_ch=3` and nothing ever constructs an RSU
        # that way. A 12-wide RSU passes every shape test in this package,
        # because the outer projection fixes the block's output width and the
        # ladder is width-agnostic in between.
        #
        # They are INDEPENDENT of every rectifier row above. `seg_out_channels`
        # is 64 and `encoder_stem_channels` is 80; the two stages were sized
        # separately by two different papers and there is no relationship to
        # express. Do NOT derive one from the other, and do NOT "notice" that
        # 64 also appears in `SPATIAL_DIVISOR ** 2`.
        "seg_mid_channels": 16,
        "seg_out_channels": 64,
    },
}


def _derived_update_block_widths(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Widths that are consequences of the row above, not independent choices.

    Kept as a derivation rather than three more transcribed literals so that
    changing ``hidden_dim`` in one place moves every dependent width with it.

    :param spec: One ``_VARIANT_SPEC`` row, already carrying its base widths.
    :type spec: Dict[str, Any]
    :return: The derived widths, ready to merge into that row.
    :rtype: Dict[str, Any]
    :raises ValueError: If the row's ``gru_input_dim`` is not what the update
        block will actually feed the GRU.
    """
    hidden_dim = spec["hidden_dim"]
    context_dim = spec["context_dim"]
    gru_input_dim = spec["gru_input_dim"]

    # THE CONTEXT-WIDTH COMPOSITION. `update.py:98` builds the GRU's input as
    # `cat([inp, motion_features])` at the CALL SITE -- the GRU does not
    # assemble it and cannot see either half. `update.py:88` then declares that
    # width as `160 + 160`. If either half drifts, the GRU's declared input
    # width silently disagrees with what it receives, and the failure surfaces
    # (if at all) as a confusing convolution build error far from the cause.
    # Checked here, at import time, once.
    if gru_input_dim != context_dim + hidden_dim:
        raise ValueError(
            f"gru_input_dim ({gru_input_dim}) must equal context_dim "
            f"({context_dim}) + the motion encoder's output width "
            f"({hidden_dim}); the update block assembles the GRU's input as "
            f"concat([inp, motion_features]) (update.py:98) and update.py:88 "
            f"declares that sum as 160 + 160."
        )

    return {
        # `update.py:81`: the motion encoder returns `cat([out, flow])`, and
        # `update.py:88` counts that as the second 160 of `input_dim=160+160`.
        #
        # The fusion convolution's own width -- `update.py:71`'s
        # `Conv2d(160+80, 160-2, ...)`, i.e. 158 -- is deliberately NOT a row
        # here. It is `output_dim - FLOW_CHANNELS`, and
        # `DocScannerMotionEncoder` derives it from the `output_dim` it is
        # given. Stating it in both places would be a rule kept in lockstep by
        # hand, which is a defect waiting to happen rather than a single source.
        "motion_output_dim": hidden_dim,
        # `update.py:94`: `nn.Conv2d(288, 64*9, 1)`. The 64 is
        # `SPATIAL_DIVISOR ** 2` (one weight per destination sub-pixel of the 8x
        # upsample) and the 9 is `CONVEX_NEIGHBOURS`. Neither 9 nor 576 is
        # restated here.
        "mask_head_output_channels": SPATIAL_DIVISOR ** 2 * CONVEX_NEIGHBOURS,
    }


for _variant_row in _VARIANT_SPEC.values():
    _variant_row.update(_derived_update_block_widths(_variant_row))

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


# ---------------------------------------------------------------------
# The separable recurrent update core.
# ---------------------------------------------------------------------

# `update.py:37-39`: the FIRST pass' three convolutions are `(1, 5)` kernels
# with `padding=(0, 2)`. In torch's `(kH, kW)` ordering that is a kernel one
# pixel tall and five wide, so it mixes along the WIDTH axis only. Keras'
# `kernel_size` is `(kH, kW)` as well -- the tuple transfers verbatim, and the
# axis it acts on does NOT change with the channels-last layout, because the
# layout moves the CHANNEL axis, not the spatial pair.
GRU_HORIZONTAL_KERNEL_SIZE: Tuple[int, int] = (1, 5)

# `update.py:41-43`: the SECOND pass' kernels are `(5, 1)` with
# `padding=(2, 0)` -- five tall, one wide, mixing along HEIGHT only.
GRU_VERTICAL_KERNEL_SIZE: Tuple[int, int] = (5, 1)


# DECISION plan-2026-09-10T065432-05fcb6dd/D-014: three orderings in this class
# are silent when wrong, i.e. every shape, dtype, gradient and serialization test
# stays green under the mutation.
#   (1) The two passes are SEQUENTIAL, not parallel and not commutative: the
#       vertical pass consumes the `h` the horizontal pass just produced
#       (`update.py:50` feeds `update.py:53`), and the SAME `x` both times.
#       Do NOT compute both from the entry `h` and combine them.
#   (2) Horizontal is FIRST. The order is observable only on an input whose
#       horizontal and vertical structure differ, on a NON-SQUARE fixture.
#   (3) `q` is convolved over `concat(r * h, x)`, NOT over `concat(h, x)`
#       (`update.py:49`). Dropping the reset gate leaves a working GRU that
#       simply cannot forget, which no shape or finiteness probe can see.
# Also: these are STRIDE-1 convolutions, so Keras `"same"` and torch's symmetric
# `padding=(0, 2)` / `(2, 0)` agree exactly -- D-012's stride-2 asymmetry does
# NOT apply here, and that equivalence is asserted rather than assumed
# (`TestTheStrideOnePaddingMatchesTorchsSymmetricPadding`). See decisions.md D-014.
@register_dl_technique("dl_techniques.models.doc_scanner.components")
class SepConvGRU(keras.layers.Layer):
    """The rectifier's recurrent update core (``update.py:35-61``).

    A convolutional GRU whose 5x5 receptive field is factorized into two
    sequential 1-D passes, after RAFT. Each pass is a complete GRU step --
    update gate ``z``, reset gate ``r``, candidate ``q`` -- over the
    concatenation of the hidden state and the (unchanging) context input:

    .. code-block:: text

        pass 1, kernel (1, 5), mixes along WIDTH
            hx = concat([h, x])
            z  = sigmoid(convz1(hx))
            r  = sigmoid(convr1(hx))
            q  = tanh(convq1(concat([r * h, x])))
            h  = (1 - z) * h + z * q

        pass 2, kernel (5, 1), mixes along HEIGHT, consuming the h ABOVE
            hx = concat([h, x])
            z  = sigmoid(convz2(hx))
            r  = sigmoid(convr2(hx))
            q  = tanh(convq2(concat([r * h, x])))
            h  = (1 - z) * h + z * q

    Gate polarity is the reference's and is worth stating because it is the
    opposite of some GRU write-ups: ``z == 1`` means "take the candidate
    entirely" (``h_new == q``) and ``z == 0`` means "keep the old state"
    (``h_new == h``). An inverted convention still trains and still produces
    finite output of the right shape.

    Widths. Every convolution reads ``hidden_dim + input_dim`` channels and
    emits ``hidden_dim``; at the one shipped variant that is ``160 + 320 ->
    160``. Both numbers come from ``_VARIANT_SPEC``, which reads them from
    ``update.py:88``'s CALL SITE and never from ``update.py:36``'s dead
    ``hidden_dim=128, input_dim=192+128`` default -- see the ``_VARIANT_SPEC``
    anchor and decisions.md D-006 / D-007.

    Initialization. The reference does NOT re-initialize these convolutions
    (``extractor.py:106-108`` re-inits the ENCODER's, and ``update.py`` has no
    equivalent), so upstream they carry torch's ``Conv2d`` default. This port
    leaves Keras' ``glorot_uniform`` default in place rather than transcribing
    torch's ``kaiming_uniform(a=sqrt(5))``: the model is trained from scratch,
    no checkpoint is transferred, and inventing an initializer the reference
    never states would be a divergence dressed as fidelity.

    :param hidden_dim: Width of the hidden state, and of every convolution's
        output. ``_VARIANT_SPEC["docscanner-l"]["hidden_dim"]`` at the shipped
        variant.
    :type hidden_dim: int
    :param input_dim: Width of the context input ``x``.
        ``_VARIANT_SPEC["docscanner-l"]["gru_input_dim"]`` at the shipped
        variant.
    :type input_dim: int
    :param kwargs: Forwarded to ``keras.layers.Layer``.
    :type kwargs: Any
    :raises ValueError: If either width is not positive.

    Example:

    .. code-block:: python

        spec = _VARIANT_SPEC["docscanner-l"]
        gru = SepConvGRU(
            hidden_dim=spec["hidden_dim"], input_dim=spec["gru_input_dim"])
        net = gru([hidden_state, context])   # (B, H, W, hidden_dim)
    """

    def __init__(
            self,
            hidden_dim: int,
            input_dim: int,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        # All six sub-layers are created UNCONDITIONALLY here; none is built
        # lazily inside `call`, and none is created behind a flag.
        self.convz1 = self._gate_conv(GRU_HORIZONTAL_KERNEL_SIZE, "convz1")
        self.convr1 = self._gate_conv(GRU_HORIZONTAL_KERNEL_SIZE, "convr1")
        self.convq1 = self._gate_conv(GRU_HORIZONTAL_KERNEL_SIZE, "convq1")

        self.convz2 = self._gate_conv(GRU_VERTICAL_KERNEL_SIZE, "convz2")
        self.convr2 = self._gate_conv(GRU_VERTICAL_KERNEL_SIZE, "convr2")
        self.convq2 = self._gate_conv(GRU_VERTICAL_KERNEL_SIZE, "convq2")

    def _gate_conv(
            self,
            kernel_size: Tuple[int, int],
            name: str
    ) -> keras.layers.Conv2D:
        """One of the six gate convolutions.

        ``padding="same"`` at stride 1 is torch's symmetric ``padding=(0, 2)``
        / ``(2, 0)`` EXACTLY -- the odd kernel extents 5 and 1 both need an even
        total pad, which TF splits evenly, so no side is favoured. D-012's
        stride-2 asymmetry cannot arise here.

        :param kernel_size: ``(kh, kw)``; ``(1, 5)`` mixes width, ``(5, 1)``
            mixes height.
        :type kernel_size: Tuple[int, int]
        :param name: Sub-layer name.
        :type name: str
        :return: An unbuilt convolution emitting ``hidden_dim`` channels.
        :rtype: keras.layers.Conv2D
        """
        return keras.layers.Conv2D(
            filters=self.hidden_dim,
            kernel_size=kernel_size,
            strides=1,
            padding="same",
            use_bias=True,
            name=name,
        )

    def build(self, input_shape: Sequence[Tuple[Optional[int], ...]]) -> None:
        """Build all six convolutions on the concatenated ``[h, x]`` width.

        :param input_shape: A two-element sequence
            ``[hidden_shape, input_shape]``, each ``(batch, height, width,
            channels)``.
        :type input_shape: Sequence[tuple]
        :raises ValueError: If two shapes were not supplied, if either is not
            rank 4, or if either channel count disagrees with the width this
            layer was constructed for.
        """
        if self.built:
            return

        if len(input_shape) != 2:
            raise ValueError(
                f"SepConvGRU is called on a two-element sequence "
                f"[hidden_state, inputs]; got {len(input_shape)} element(s): "
                f"{input_shape}"
            )

        hidden_shape, context_shape = (tuple(s) for s in input_shape)

        for label, shape in (
                ("hidden_state", hidden_shape), ("inputs", context_shape)):
            if len(shape) != 4:
                raise ValueError(
                    f"Expected a 4D {label} shape (batch, height, width, "
                    f"channels), got {len(shape)}D: {shape}"
                )

        if hidden_shape[-1] is not None and hidden_shape[-1] != self.hidden_dim:
            raise ValueError(
                f"hidden_state has {hidden_shape[-1]} channels but this "
                f"SepConvGRU was built for hidden_dim={self.hidden_dim}"
            )
        if context_shape[-1] is not None and context_shape[-1] != self.input_dim:
            raise ValueError(
                f"inputs has {context_shape[-1]} channels but this SepConvGRU "
                f"was built for input_dim={self.input_dim}"
            )

        # Every one of the six convolutions sees the SAME concatenated width:
        # `concat([h, x])` for z and r, `concat([r * h, x])` for q, and `r * h`
        # has the same width as `h`.
        concat_shape = (
            hidden_shape[0],
            hidden_shape[1],
            hidden_shape[2],
            self.hidden_dim + self.input_dim,
        )

        for conv in (
                self.convz1, self.convr1, self.convq1,
                self.convz2, self.convr2, self.convq2,
        ):
            conv.build(concat_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: Sequence[keras.KerasTensor],
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Run the horizontal pass, then the vertical pass on its output.

        :param inputs: ``[hidden_state, context]``. ``hidden_state`` is
            ``(batch, height, width, hidden_dim)``; ``context`` is
            ``(batch, height, width, input_dim)`` and is the SAME tensor for
            both passes.
        :type inputs: Sequence[keras.KerasTensor]
        :param training: Unused -- this layer holds no training-dependent
            sub-layer -- and accepted only so Keras may forward it.
        :type training: Optional[bool]
        :return: The updated hidden state, ``(batch, height, width,
            hidden_dim)``.
        :rtype: keras.KerasTensor
        """
        hidden_state, context = inputs

        # --- pass 1: (1, 5) kernels, mixing along WIDTH -------------------
        hidden_state = self._gru_step(
            hidden_state, context, self.convz1, self.convr1, self.convq1)

        # --- pass 2: (5, 1) kernels, mixing along HEIGHT -------------------
        # It consumes the hidden state pass 1 JUST produced, and the same
        # `context`. Swapping these two blocks is shape-preserving.
        hidden_state = self._gru_step(
            hidden_state, context, self.convz2, self.convr2, self.convq2)

        return hidden_state

    @staticmethod
    def _gru_step(
            hidden_state: keras.KerasTensor,
            context: keras.KerasTensor,
            conv_z: keras.layers.Conv2D,
            conv_r: keras.layers.Conv2D,
            conv_q: keras.layers.Conv2D,
    ) -> keras.KerasTensor:
        """One complete gated step (``update.py:46-50``), for one pass.

        :param hidden_state: ``(batch, height, width, hidden_dim)``.
        :type hidden_state: keras.KerasTensor
        :param context: ``(batch, height, width, input_dim)``.
        :type context: keras.KerasTensor
        :param conv_z: The update-gate convolution of this pass.
        :type conv_z: keras.layers.Conv2D
        :param conv_r: The reset-gate convolution of this pass.
        :type conv_r: keras.layers.Conv2D
        :param conv_q: The candidate convolution of this pass.
        :type conv_q: keras.layers.Conv2D
        :return: The updated hidden state, same shape as ``hidden_state``.
        :rtype: keras.KerasTensor
        """
        gate_input = keras.ops.concatenate([hidden_state, context], axis=-1)

        update_gate = keras.ops.sigmoid(conv_z(gate_input))
        reset_gate = keras.ops.sigmoid(conv_r(gate_input))

        # The candidate reads the RESET hidden state, not the raw one.
        candidate = keras.ops.tanh(conv_q(keras.ops.concatenate(
            [reset_gate * hidden_state, context], axis=-1)))

        # z == 1 -> take the candidate; z == 0 -> keep the state.
        return (1.0 - update_gate) * hidden_state + update_gate * candidate

    def compute_output_shape(
            self,
            input_shape: Sequence[Tuple[Optional[int], ...]]
    ) -> Tuple[Optional[int], ...]:
        """The hidden state's shape, unchanged -- both passes are stride 1.

        :param input_shape: ``[hidden_shape, context_shape]``.
        :type input_shape: Sequence[tuple]
        :return: The output shape tuple.
        :rtype: tuple
        """
        hidden_shape = tuple(input_shape[0])
        return (
            hidden_shape[0],
            hidden_shape[1],
            hidden_shape[2],
            self.hidden_dim,
        )

    def get_config(self) -> Dict[str, Any]:
        """Return every constructor argument.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "input_dim": self.input_dim,
        })
        return config


# ---------------------------------------------------------------------
# The update block and its two heads.
# ---------------------------------------------------------------------


# DECISION plan-2026-09-10T065432-05fcb6dd/D-016: the tail of this layer's output
# is the RAW flow, re-appended unchanged -- `cat([out, flow])`, `update.py:81`.
# Do NOT re-append the flow BRANCH (`flo`, the 80-channel `convf2` output): that
# mutation is shape-breaking here and so is not the live hazard. The live hazard
# is a SCALED or otherwise transformed flow -- normalising it, dividing it by the
# spatial size, or re-using `convf1`'s input after an in-place edit. Any of those
# keeps the width at exactly `motion_output_dim` and stays green under every
# shape, dtype, gradient and serialization test, while removing the update
# block's only direct, un-convolved view of where the coordinate field currently
# is. See decisions.md D-016.
@register_dl_technique("dl_techniques.models.doc_scanner.components")
class DocScannerMotionEncoder(keras.layers.Layer):
    """The update block's input encoder (``update.py:64-81``).

    Two independent convolutional branches -- one over the resampled features,
    one over the current flow -- concatenated, fused by a single convolution,
    and finally concatenated with the RAW flow again:

    .. code-block:: text

        warped_features [B, H, W, 320]      flow [B, H, W, 2]
              |                                   |
        Conv2D 1x1 -> 240 -> ReLU           Conv2D 7x7 -> 160 -> ReLU
        Conv2D 3x3 -> 160 -> ReLU           Conv2D 3x3 ->  80 -> ReLU
              \\_____________ concat -> 240 _______/
                                |
                        Conv2D 3x3 -> 158 -> ReLU
                                |
                    concat([out, flow]) -> [B, H, W, 160]

    **There is no correlation volume.** Upstream names this layer's second
    parameter ``corr`` and the RAFT ancestry makes that read like a 4D cost
    volume, but ``model.py:91`` passes ``warpfea`` -- a bilinear resample of the
    encoder's own ``fmap1`` at the current coordinates (F-01, plan invariant
    #2). Nothing in this port constructs a cost volume, so the parameter is
    named ``warped_features`` here rather than transcribing a misnomer that
    would invite one to be added.

    Initialization follows :class:`SepConvGRU`: ``extractor.py:106-108``
    re-initializes the ENCODER's convolutions and ``update.py`` has no
    equivalent, so Keras' ``glorot_uniform`` default is left in place rather
    than inventing an initializer the reference never states.

    :param output_dim: Width of the returned tensor, i.e. the second half of
        the GRU's input width. ``_VARIANT_SPEC[...]["motion_output_dim"]``.
        The fusion convolution emits ``output_dim - FLOW_CHANNELS`` channels so
        that re-appending the flow lands back on exactly this number.
    :type output_dim: int
    :param corr_hidden: Width of the warped-feature branch's 1x1 convolution
        (``update.py:67``).
    :type corr_hidden: int
    :param corr_out: Width of the warped-feature branch's 3x3 convolution
        (``update.py:68``).
    :type corr_out: int
    :param flow_hidden: Width of the flow branch's 7x7 convolution
        (``update.py:69``).
    :type flow_hidden: int
    :param flow_out: Width of the flow branch's 3x3 convolution
        (``update.py:70``).
    :type flow_out: int
    :param kwargs: Forwarded to ``keras.layers.Layer``.
    :type kwargs: Any
    :raises ValueError: If any width is not positive, or if ``output_dim`` does
        not exceed :data:`FLOW_CHANNELS` (the fusion convolution would then have
        a non-positive width).

    Example:

    .. code-block:: python

        spec = _VARIANT_SPEC["docscanner-l"]
        encoder = DocScannerMotionEncoder(
            output_dim=spec["motion_output_dim"],
            corr_hidden=spec["motion_corr_hidden"],
            corr_out=spec["motion_corr_out"],
            flow_hidden=spec["motion_flow_hidden"],
            flow_out=spec["motion_flow_out"],
        )
        motion = encoder([flow, warped_features])   # (B, H, W, 160)
    """

    def __init__(
            self,
            output_dim: int,
            corr_hidden: int,
            corr_out: int,
            flow_hidden: int,
            flow_out: int,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        for label, value in (
                ("output_dim", output_dim),
                ("corr_hidden", corr_hidden),
                ("corr_out", corr_out),
                ("flow_hidden", flow_hidden),
                ("flow_out", flow_out),
        ):
            if value <= 0:
                raise ValueError(f"{label} must be positive, got {value}")

        if output_dim <= FLOW_CHANNELS:
            raise ValueError(
                f"output_dim ({output_dim}) must exceed FLOW_CHANNELS "
                f"({FLOW_CHANNELS}): the fusion convolution emits "
                f"output_dim - FLOW_CHANNELS channels (update.py:71's "
                f"`160-2`) and the raw flow is concatenated back on."
            )

        self.output_dim = output_dim
        self.corr_hidden = corr_hidden
        self.corr_out = corr_out
        self.flow_hidden = flow_hidden
        self.flow_out = flow_out

        # `update.py:71`: `Conv2d(160+80, 160-2, ...)`. Derived, never written.
        self.fusion_channels = output_dim - FLOW_CHANNELS

        # All five sub-layers are created UNCONDITIONALLY; none is behind a flag
        # and none is created inside `call`. Every convolution below is STRIDE 1
        # with an odd kernel, so Keras `"same"` is torch's symmetric
        # `padding=k//2` exactly -- D-012's stride-2 asymmetry cannot arise
        # here. The 1x1 needs no padding at all.
        self.convc1 = keras.layers.Conv2D(
            filters=corr_hidden,
            kernel_size=1,
            strides=1,
            padding="valid",
            use_bias=True,
            name="convc1",
        )
        self.convc2 = keras.layers.Conv2D(
            filters=corr_out,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=True,
            name="convc2",
        )
        self.convf1 = keras.layers.Conv2D(
            filters=flow_hidden,
            kernel_size=7,
            strides=1,
            padding="same",
            use_bias=True,
            name="convf1",
        )
        self.convf2 = keras.layers.Conv2D(
            filters=flow_out,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=True,
            name="convf2",
        )
        self.conv = keras.layers.Conv2D(
            filters=self.fusion_channels,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=True,
            name="conv",
        )

    def build(self, input_shape: Sequence[Tuple[Optional[int], ...]]) -> None:
        """Build both branches and the fusion convolution.

        :param input_shape: A two-element sequence
            ``[flow_shape, warped_features_shape]``, each
            ``(batch, height, width, channels)`` -- in that order, matching
            ``update.py:73``'s ``forward(self, flow, corr)``.
        :type input_shape: Sequence[tuple]
        :raises ValueError: If two shapes were not supplied, if either is not
            rank 4, or if the flow does not have :data:`FLOW_CHANNELS` channels.
        """
        if self.built:
            return

        if len(input_shape) != 2:
            raise ValueError(
                f"DocScannerMotionEncoder is called on a two-element sequence "
                f"[flow, warped_features]; got {len(input_shape)} element(s): "
                f"{input_shape}"
            )

        flow_shape, warped_shape = (tuple(s) for s in input_shape)

        for label, shape in (
                ("flow", flow_shape), ("warped_features", warped_shape)):
            if len(shape) != 4:
                raise ValueError(
                    f"Expected a 4D {label} shape (batch, height, width, "
                    f"channels), got {len(shape)}D: {shape}"
                )

        if flow_shape[-1] is not None and flow_shape[-1] != FLOW_CHANNELS:
            raise ValueError(
                f"flow must have exactly {FLOW_CHANNELS} channels (x, y), got "
                f"{flow_shape[-1]}. Its width is what makes the fusion "
                f"convolution's output_dim - {FLOW_CHANNELS} land back on "
                f"output_dim after the raw flow is concatenated on."
            )

        self.convc1.build(warped_shape)
        corr_shape = self.convc1.compute_output_shape(warped_shape)
        self.convc2.build(corr_shape)
        corr_shape = self.convc2.compute_output_shape(corr_shape)

        self.convf1.build(flow_shape)
        flo_shape = self.convf1.compute_output_shape(flow_shape)
        self.convf2.build(flo_shape)
        flo_shape = self.convf2.compute_output_shape(flo_shape)

        fused_shape = (
            corr_shape[0],
            corr_shape[1],
            corr_shape[2],
            corr_shape[-1] + flo_shape[-1],
        )
        self.conv.build(fused_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: Sequence[keras.KerasTensor],
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Encode the resampled features and the current flow jointly.

        :param inputs: ``[flow, warped_features]``. ``flow`` is
            ``(batch, height, width, FLOW_CHANNELS)``; ``warped_features`` is
            ``(batch, height, width, fnet_output_dim)`` and is a RESAMPLE of the
            feature map, not a correlation volume.
        :type inputs: Sequence[keras.KerasTensor]
        :param training: Unused -- this layer holds no training-dependent
            sub-layer -- and accepted only so Keras may forward it.
        :type training: Optional[bool]
        :return: ``(batch, height, width, output_dim)``, whose LAST
            :data:`FLOW_CHANNELS` channels are ``flow`` itself, unchanged.
        :rtype: keras.KerasTensor
        """
        flow, warped_features = inputs

        corr = keras.ops.relu(self.convc1(warped_features))
        corr = keras.ops.relu(self.convc2(corr))

        flo = keras.ops.relu(self.convf1(flow))
        flo = keras.ops.relu(self.convf2(flo))

        fused = keras.ops.relu(
            self.conv(keras.ops.concatenate([corr, flo], axis=-1))
        )

        # The RAW flow, not `flo` and not a rescaling of it (see the anchor
        # above the class). Guarded by
        # `TestTheMotionEncoderTailIsTheRawFlow`.
        return keras.ops.concatenate([fused, flow], axis=-1)

    def compute_output_shape(
            self,
            input_shape: Sequence[Tuple[Optional[int], ...]]
    ) -> Tuple[Optional[int], ...]:
        """Spatial axes are unchanged (every convolution is stride 1).

        :param input_shape: ``[flow_shape, warped_features_shape]``.
        :type input_shape: Sequence[tuple]
        :return: The output shape tuple.
        :rtype: tuple
        """
        flow_shape = tuple(input_shape[0])
        return (flow_shape[0], flow_shape[1], flow_shape[2], self.output_dim)

    def get_config(self) -> Dict[str, Any]:
        """Return every constructor argument.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "corr_hidden": self.corr_hidden,
            "corr_out": self.corr_out,
            "flow_hidden": self.flow_hidden,
            "flow_out": self.flow_out,
        })
        return config


# DECISION plan-2026-09-10T065432-05fcb6dd/D-016: this head has NO output
# activation (`update.py:14`, `conv2(relu(conv1(x)))` -- the ReLU is between the
# two convolutions, never after the second). Do NOT add one. The value it emits
# is a residual added to a coordinate field (`model.py:88`,
# `coords1 = coords1 + delta_flow`), so it MUST be able to be negative: a page
# corner that has to move left or up cannot be reached otherwise. A ReLU or a
# tanh here leaves a model that trains, produces finite output of exactly the
# right shape, and can only ever push coordinates one way. See decisions.md
# D-016.
@register_dl_technique("dl_techniques.models.doc_scanner.components")
class FlowHead(keras.layers.Layer):
    """The residual-flow head (``update.py:6-14``).

    ``Conv2D 3x3 -> hidden_dim -> ReLU -> Conv2D 3x3 -> FLOW_CHANNELS``, raw.

    ``hidden_dim`` here is the MIDDLE width, not the input width -- upstream's
    signature is ``FlowHead(input_dim, hidden_dim)`` and it is easy to read the
    two the wrong way round. This port infers the input width from the tensor,
    so the one declared width is unambiguous. Its value comes from
    ``update.py:89``'s call site, ``FlowHead(hidden_dim, hidden_dim=320)``, and
    NOT from ``update.py:7``'s dead ``hidden_dim=256`` default (D-006).

    :param hidden_dim: Width of the intermediate convolution.
        ``_VARIANT_SPEC[...]["flow_head_hidden"]``.
    :type hidden_dim: int
    :param kwargs: Forwarded to ``keras.layers.Layer``.
    :type kwargs: Any
    :raises ValueError: If ``hidden_dim`` is not positive.

    Example:

    .. code-block:: python

        head = FlowHead(hidden_dim=_VARIANT_SPEC["docscanner-l"]["flow_head_hidden"])
        delta_flow = head(net)   # (B, H, W, 2), unbounded, signed
    """

    def __init__(self, hidden_dim: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")

        self.hidden_dim = hidden_dim

        # Both stride 1 with odd kernels: `"same"` is torch's `padding=1`.
        self.conv1 = keras.layers.Conv2D(
            filters=hidden_dim,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=True,
            name="conv1",
        )
        self.conv2 = keras.layers.Conv2D(
            filters=FLOW_CHANNELS,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=True,
            name="conv2",
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build both convolutions explicitly.

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
        self.conv1.build(shape)
        self.conv2.build(self.conv1.compute_output_shape(shape))

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Emit the raw, signed, unbounded coordinate residual.

        :param inputs: ``(batch, height, width, channels)`` -- the GRU's hidden
            state.
        :type inputs: keras.KerasTensor
        :param training: Unused; accepted only so Keras may forward it.
        :type training: Optional[bool]
        :return: ``(batch, height, width, FLOW_CHANNELS)``. NOT activated.
        :rtype: keras.KerasTensor
        """
        return self.conv2(keras.ops.relu(self.conv1(inputs)))

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Spatial axes unchanged; the channel axis becomes ``FLOW_CHANNELS``.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple
        :return: The output shape tuple.
        :rtype: tuple
        """
        batch, height, width, _ = input_shape
        return (batch, height, width, FLOW_CHANNELS)

    def get_config(self) -> Dict[str, Any]:
        """Return every constructor argument.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({"hidden_dim": self.hidden_dim})
        return config


@register_dl_technique("dl_techniques.models.doc_scanner.components")
class DocScannerUpdateBlock(keras.layers.Layer):
    """One refinement iteration (``update.py:84-106``).

    .. code-block:: text

        net, inp, warped_features, flow
              |
        motion_features = motion_encoder([flow, warped_features])   -> 160
        gru_input       = concat([inp, motion_features])            -> 320
        net             = gru([net, gru_input])                     -> 160
              |
        delta_flow = flow_head(net)                    -> 2   (raw, signed)
        mask       = MASK_LOGIT_SCALE * mask_logits(net)  -> 576 (raw logits)
              |
        return (net, mask, delta_flow)

    Three things here are silent when wrong -- no shape, dtype, finiteness or
    serialization test can see any of them -- and each has a guard named beside
    it:

    1. **The GRU's input is ``concat([inp, motion_features])``, context FIRST**
       (``update.py:98``). Both halves are ``context_dim == motion_output_dim``
       wide at the shipped variant, so swapping them is perfectly shape-legal
       and merely permutes which learned filters see which half. Guarded by
       ``TestTheGRUInputIsContextThenMotion``.
    2. **The ``0.25`` is applied HERE, exactly once** (``update.py:104``), not
       inside the mask head and not inside ``convex_upsample`` -- see the
       :data:`MASK_LOGIT_SCALE` anchor. Guarded by
       ``TestTheMaskScaleIsAppliedExactlyOnce``.
    3. **The return order is ``(net, mask, delta_flow)``** (``update.py:106``).
       All three are float tensors of the same spatial size; at the shipped
       widths their channel counts differ, but a caller that unpacks them in the
       wrong order gets a coherent-looking model either way. Guarded BY VALUE,
       not by shape, in ``TestTheReturnOrderIsNetMaskDeltaFlow``.

    :param hidden_dim: Width of the recurrent state ``net``, and the flow head's
        input. ``_VARIANT_SPEC[...]["hidden_dim"]``.
    :type hidden_dim: int
    :param context_dim: Width of the per-iteration context ``inp``.
    :type context_dim: int
    :param gru_input_dim: Width the GRU is built for. Must equal
        ``context_dim + motion_output_dim``; see the composition check below.
    :type gru_input_dim: int
    :param motion_output_dim: Width the motion encoder returns.
    :type motion_output_dim: int
    :param motion_corr_hidden: ``update.py:67``.
    :type motion_corr_hidden: int
    :param motion_corr_out: ``update.py:68``.
    :type motion_corr_out: int
    :param motion_flow_hidden: ``update.py:69``.
    :type motion_flow_hidden: int
    :param motion_flow_out: ``update.py:70``.
    :type motion_flow_out: int
    :param flow_head_hidden: The flow head's middle width (``update.py:89``).
    :type flow_head_hidden: int
    :param mask_hidden: The mask head's middle width (``update.py:92``).
    :type mask_hidden: int
    :param kwargs: Forwarded to ``keras.layers.Layer``.
    :type kwargs: Any
    :raises ValueError: If any width is not positive, or if ``gru_input_dim``
        does not equal ``context_dim + motion_output_dim``.

    Example:

    .. code-block:: python

        spec = _VARIANT_SPEC["docscanner-l"]
        block = DocScannerUpdateBlock(
            hidden_dim=spec["hidden_dim"],
            context_dim=spec["context_dim"],
            gru_input_dim=spec["gru_input_dim"],
            motion_output_dim=spec["motion_output_dim"],
            motion_corr_hidden=spec["motion_corr_hidden"],
            motion_corr_out=spec["motion_corr_out"],
            motion_flow_hidden=spec["motion_flow_hidden"],
            motion_flow_out=spec["motion_flow_out"],
            flow_head_hidden=spec["flow_head_hidden"],
            mask_hidden=spec["mask_head_hidden"],
        )
        net, mask, delta_flow = block([net, inp, warped_features, flow])
    """

    def __init__(
            self,
            hidden_dim: int,
            context_dim: int,
            gru_input_dim: int,
            motion_output_dim: int,
            motion_corr_hidden: int,
            motion_corr_out: int,
            motion_flow_hidden: int,
            motion_flow_out: int,
            flow_head_hidden: int,
            mask_hidden: int,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        for label, value in (
                ("hidden_dim", hidden_dim),
                ("context_dim", context_dim),
                ("gru_input_dim", gru_input_dim),
                ("motion_output_dim", motion_output_dim),
                ("motion_corr_hidden", motion_corr_hidden),
                ("motion_corr_out", motion_corr_out),
                ("motion_flow_hidden", motion_flow_hidden),
                ("motion_flow_out", motion_flow_out),
                ("flow_head_hidden", flow_head_hidden),
                ("mask_hidden", mask_hidden),
        ):
            if value <= 0:
                raise ValueError(f"{label} must be positive, got {value}")

        # THE CONTEXT-WIDTH COMPOSITION, asserted at the site that performs it.
        # `update.py:98` assembles the GRU's input as `cat([inp,
        # motion_features])`; `update.py:88` separately declares the GRU's
        # `input_dim` as `160 + 160`. NOTHING inside the GRU can see either
        # half, so if one drifts the two disagree silently until a convolution
        # somewhere else fails to build. `_derived_update_block_widths` makes
        # the same check against `_VARIANT_SPEC`; this one also covers a caller
        # that passes widths by hand.
        if gru_input_dim != context_dim + motion_output_dim:
            raise ValueError(
                f"gru_input_dim ({gru_input_dim}) must equal context_dim "
                f"({context_dim}) + motion_output_dim ({motion_output_dim}) = "
                f"{context_dim + motion_output_dim}: this block builds the "
                f"GRU's input itself, as concat([inp, motion_features]) "
                f"(update.py:98), and the GRU cannot see either half."
            )

        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.gru_input_dim = gru_input_dim
        self.motion_output_dim = motion_output_dim
        self.motion_corr_hidden = motion_corr_hidden
        self.motion_corr_out = motion_corr_out
        self.motion_flow_hidden = motion_flow_hidden
        self.motion_flow_out = motion_flow_out
        self.flow_head_hidden = flow_head_hidden
        self.mask_hidden = mask_hidden

        # `update.py:94`: `Conv2d(288, 64*9, 1)`. Structural, so it is derived
        # rather than exposed as a knob -- it is fixed by the upsample factor
        # and the neighbourhood size, both of which are properties of
        # `convex_upsample`, not of the variant.
        self.mask_output_channels = SPATIAL_DIVISOR ** 2 * CONVEX_NEIGHBOURS

        # All sub-layers unconditional; none created in `call`.
        self.motion_encoder = DocScannerMotionEncoder(
            output_dim=motion_output_dim,
            corr_hidden=motion_corr_hidden,
            corr_out=motion_corr_out,
            flow_hidden=motion_flow_hidden,
            flow_out=motion_flow_out,
            name="motion_encoder",
        )
        self.gru = SepConvGRU(
            hidden_dim=hidden_dim,
            input_dim=gru_input_dim,
            name="gru",
        )
        self.flow_head = FlowHead(
            hidden_dim=flow_head_hidden,
            name="flow_head",
        )

        # The mask head is upstream's `nn.Sequential(Conv2d(160, 288, 3, pad=1),
        # ReLU, Conv2d(288, 576, 1))` (`update.py:91-94`), written out as two
        # named convolutions rather than a `keras.Sequential` so that `build()`
        # can propagate shapes explicitly and so that `mask_logits` below has
        # something to expose. It is NOT a separate registered class: it has one
        # call site, and promoting it would be a Complexity-Budget charge with
        # no payoff.
        self.mask_conv1 = keras.layers.Conv2D(
            filters=mask_hidden,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=True,
            name="mask_conv1",
        )
        self.mask_conv2 = keras.layers.Conv2D(
            filters=self.mask_output_channels,
            kernel_size=1,
            strides=1,
            padding="valid",
            use_bias=True,
            name="mask_conv2",
        )

    def mask_logits(self, net: keras.KerasTensor) -> keras.KerasTensor:
        """The mask head's output, **UNSCALED**.

        Interface contract, because this method has two callers -- :meth:`call`
        and the guard that pins the scale factor:

        * Parameter: ``net``, ``(batch, height, width, hidden_dim)``.
        * Returns: ``(batch, height, width, SPATIAL_DIVISOR ** 2 *
          CONVEX_NEIGHBOURS)``, i.e. 576 channels of raw logits with
          :data:`MASK_LOGIT_SCALE` **not** applied.
        * Failure mode: raises whatever Keras raises if this layer has not been
          built, or if ``net``'s width differs from ``hidden_dim``.

        :meth:`call` is the only place the ``0.25`` is applied; a caller that
        wants the value the upsample consumes must multiply it in.

        :param net: The GRU's hidden state.
        :type net: keras.KerasTensor
        :return: Raw, unscaled mask logits.
        :rtype: keras.KerasTensor
        """
        return self.mask_conv2(keras.ops.relu(self.mask_conv1(net)))

    def build(self, input_shape: Sequence[Tuple[Optional[int], ...]]) -> None:
        """Build the motion encoder, the GRU and both heads explicitly.

        :param input_shape: A four-element sequence
            ``[net_shape, inp_shape, warped_features_shape, flow_shape]``,
            matching ``update.py:96``'s ``forward(self, net, inp, corr, flow)``.
        :type input_shape: Sequence[tuple]
        :raises ValueError: If four shapes were not supplied, if any is not rank
            4, or if ``net`` / ``inp`` disagree with the widths this block was
            constructed for.
        """
        if self.built:
            return

        if len(input_shape) != 4:
            raise ValueError(
                f"DocScannerUpdateBlock is called on a four-element sequence "
                f"[net, inp, warped_features, flow]; got {len(input_shape)} "
                f"element(s): {input_shape}"
            )

        net_shape, inp_shape, warped_shape, flow_shape = (
            tuple(s) for s in input_shape
        )

        for label, shape in (
                ("net", net_shape),
                ("inp", inp_shape),
                ("warped_features", warped_shape),
                ("flow", flow_shape),
        ):
            if len(shape) != 4:
                raise ValueError(
                    f"Expected a 4D {label} shape (batch, height, width, "
                    f"channels), got {len(shape)}D: {shape}"
                )

        if net_shape[-1] is not None and net_shape[-1] != self.hidden_dim:
            raise ValueError(
                f"net has {net_shape[-1]} channels but this update block was "
                f"built for hidden_dim={self.hidden_dim}"
            )
        if inp_shape[-1] is not None and inp_shape[-1] != self.context_dim:
            raise ValueError(
                f"inp has {inp_shape[-1]} channels but this update block was "
                f"built for context_dim={self.context_dim}. That width is one "
                f"half of the GRU's input; the motion encoder supplies the "
                f"other (update.py:98)."
            )

        self.motion_encoder.build([flow_shape, warped_shape])
        motion_shape = self.motion_encoder.compute_output_shape(
            [flow_shape, warped_shape]
        )

        gru_input_shape = (
            net_shape[0],
            net_shape[1],
            net_shape[2],
            self.context_dim + motion_shape[-1],
        )
        self.gru.build([net_shape, gru_input_shape])

        self.flow_head.build(net_shape)

        self.mask_conv1.build(net_shape)
        self.mask_conv2.build(self.mask_conv1.compute_output_shape(net_shape))

        super().build(input_shape)

    def call(
            self,
            inputs: Sequence[keras.KerasTensor],
            training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]:
        """Run one refinement iteration.

        :param inputs: ``[net, inp, warped_features, flow]``. ``net`` is the
            recurrent state ``(B, H, W, hidden_dim)``; ``inp`` is the
            per-iteration context ``(B, H, W, context_dim)``, constant across
            iterations upstream; ``warped_features`` is the feature map
            RESAMPLED at the current coordinates (not a correlation volume);
            ``flow`` is the current coordinate residual
            ``(B, H, W, FLOW_CHANNELS)``.
        :type inputs: Sequence[keras.KerasTensor]
        :param training: Forwarded to the sub-layers.
        :type training: Optional[bool]
        :return: ``(net, mask, delta_flow)`` -- in that order. ``net`` is the
            updated state ``(B, H, W, hidden_dim)``; ``mask`` is
            ``(B, H, W, 576)`` of ``MASK_LOGIT_SCALE``-scaled logits, ready for
            ``convex_upsample``, which softmaxes them; ``delta_flow`` is the
            raw signed residual ``(B, H, W, FLOW_CHANNELS)``.
        :rtype: tuple
        """
        net, inp, warped_features, flow = inputs

        motion_features = self.motion_encoder(
            [flow, warped_features], training=training)

        # Context FIRST, motion second (update.py:98). Both halves are the same
        # width at the shipped variant, so a swap is shape-legal.
        gru_input = keras.ops.concatenate([inp, motion_features], axis=-1)

        net = self.gru([net, gru_input], training=training)

        delta_flow = self.flow_head(net, training=training)

        # The 0.25, applied HERE and exactly once -- see the MASK_LOGIT_SCALE
        # anchor. `convex_upsample` takes it from here as raw logits and
        # softmaxes them; a second application would flatten the softmax
        # another 4x, and none of that is visible in a shape.
        mask = MASK_LOGIT_SCALE * self.mask_logits(net)

        return net, mask, delta_flow

    def compute_output_shape(
            self,
            input_shape: Sequence[Tuple[Optional[int], ...]]
    ) -> Tuple[Tuple[Optional[int], ...], ...]:
        """Three shapes, in the same order :meth:`call` returns their tensors.

        :param input_shape: ``[net_shape, inp_shape, warped_shape, flow_shape]``.
        :type input_shape: Sequence[tuple]
        :return: ``(net_shape, mask_shape, delta_flow_shape)``.
        :rtype: tuple
        """
        net_shape = tuple(input_shape[0])
        batch, height, width = net_shape[0], net_shape[1], net_shape[2]
        return (
            (batch, height, width, self.hidden_dim),
            (batch, height, width, self.mask_output_channels),
            (batch, height, width, FLOW_CHANNELS),
        )

    def get_config(self) -> Dict[str, Any]:
        """Return every constructor argument.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "context_dim": self.context_dim,
            "gru_input_dim": self.gru_input_dim,
            "motion_output_dim": self.motion_output_dim,
            "motion_corr_hidden": self.motion_corr_hidden,
            "motion_corr_out": self.motion_corr_out,
            "motion_flow_hidden": self.motion_flow_hidden,
            "motion_flow_out": self.motion_flow_out,
            "flow_head_hidden": self.flow_head_hidden,
            "mask_hidden": self.mask_hidden,
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
    "SepConvGRU",
    "GRU_HORIZONTAL_KERNEL_SIZE",
    "GRU_VERTICAL_KERNEL_SIZE",
    "FLOW_CHANNELS",
    "CONVEX_NEIGHBOURS",
    "MASK_LOGIT_SCALE",
    "DocScannerMotionEncoder",
    "FlowHead",
    "DocScannerUpdateBlock",
]
