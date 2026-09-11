"""DocScanner's model classes: the rectifier and the segmenter.

The two stages are trained INDEPENDENTLY (paper §4.3) and are two separate
``keras.Model`` subclasses here, not one model with a flag. They share this
module because they share the variant table they are both projected from --
``components._VARIANT_SPEC`` is ONE table describing a TWO-stage model, and the
two projections are deliberately opposite rules (see the D-023 anchor above
:func:`_segmenter_variant_rows`).

:class:`DocScannerSegmenter` is stage 1: the pruned U2-Net that predicts the
document/background confidence map. Its own class docstring carries its ladder,
its seven-output contract and why it constrains no input size. Everything
structural it uses lives in :mod:`.u2net_blocks`.

:class:`DocScannerRectifier` is stage 2 of DocScanner -- the RAFT-lineage
progressive refiner that turns a (already background-masked) page photograph
into a full-resolution BACKWARD map in pixel units. It is a direct port of
``model.py:31-99`` of the upstream release, and its whole content is one loop:

.. code-block:: text

    fmap1   = fnet(image)                                  (B, H/8, W/8, 320)
    net     = tanh(fmap1[..., :160])       the recurrent state
    inp     = relu(fmap1[..., 160:])       the per-iteration context
    warpfea = fmap1                        the FIRST iteration reads it raw

    coodslar = coords_grid(B, H,   W  )    full-res identity field
    coords0  = coords_grid(B, H/8, W/8)    the fixed origin of the flow
    coords1  = coords_grid(B, H/8, W/8)    the field being refined

    repeat `iters` times:
        coords1 = stop_gradient(coords1)
        flow    = coords1 - coords0
        net, mask, delta_flow = update_block([net, inp, warpfea, flow])
        coords1 = coords1 + delta_flow
        flow_up = convex_upsample(coords1 - coords0, mask)
        bm_up   = coodslar + flow_up                       <- one prediction
        warpfea = sample_at_pixel_coords(fmap1, coords1)

There is NO correlation volume, despite the RAFT lineage. ``warpfea`` is a
bilinear resample of the encoder's own 320-channel feature map at the current
coordinates. See ``findings.md`` F-01 and the ``__init__.py`` docstring.

The output contract
-------------------
``training=True`` returns the whole sequence, stacked as
``(B, iters, H, W, 2)``; anything else returns only the last map,
``(B, H, W, 2)``. That asymmetry is not a convenience -- it is what lets the
12-iteration exponentially-weighted sequence loss be computed by a stock
``compile(loss=...)``. This repo forbids a custom ``train_step`` (H-5), so the
sequence has to arrive at the loss as a model OUTPUT or not at all. Upstream
makes the same split with a ``test_mode`` flag (``model.py:95-99``); here the
flag is Keras' own ``training``, so ``fit()`` and ``predict()`` each get the
form they need without the caller choosing.

Units
-----
``bm_up`` is in ABSOLUTE FULL-RESOLUTION PIXEL coordinates -- channel 0 is a
column index over ``[0, W - 1]``, channel 1 a row index over ``[0, H - 1]``,
the order :func:`~.warp.coords_grid` emits. It is NOT normalized. The
``(2 * bm / 286.8 - 1) * 0.99`` calibration of ``inference.py:29`` belongs to
the composite pipeline that feeds a sampler, not to this class, and applying it
here would double-apply it there.

:class:`DocScanner` is the two of them wired together as ``inference.py:17-31``
wires them -- segment, threshold at 0.5, mask MULTIPLICATIVELY, rectify,
calibrate. It is an INFERENCE assembly and cannot be trained end to end: the
threshold has zero gradient almost everywhere, which is exactly right, because
the paper trains the two modules INDEPENDENTLY (§4.3). See its own class
docstring and the D-025/D-026/D-027 anchors in its ``call``.

Why height and width must be statically known
---------------------------------------------
:func:`~.warp.coords_grid` needs Python ints: it materializes an ``arange`` per
axis. So this model cannot be built at ``(None, None, None, 3)`` the way the
sibling ``doc_res`` can. :meth:`DocScannerRectifier.build` says so explicitly
rather than failing later inside ``keras.ops.arange`` with a message about
neither this model nor coordinates. Build it at a concrete
``(None, 288, 288, 3)``; a first eager call always supplies concrete extents
anyway.

Registration convention
-----------------------
Classes here register under ``dl_techniques.models.doc_scanner.model`` -- the
key strips BOTH the ``vision`` family and the ``image_restoration`` subfamily,
per the repo-wide convention (H-3). It is NOT the full import path.

References:
    - Upstream release: https://github.com/fh2019ustc/DocScanner --
      ``model.py:31-99`` (the rectifier), ``model.py:45-51``
      (``initialize_flow``), ``seg.py:451-543`` (the segmenter),
      ``inference.py:17-31`` (both call sites).
    - Feng et al., 2021. DocScanner: Robust Document Image Rectification with
      Progressive Learning. (https://arxiv.org/abs/2110.14968), v2.
    - Teed & Deng, 2020. RAFT: Recurrent All-Pairs Field Transforms for Optical
      Flow. ECCV 2020. (https://arxiv.org/abs/2003.12039).
    - Qin et al., 2020. U2-Net: Going Deeper with Nested U-Structure for
      Salient Object Detection. Pattern Recognition.
      (https://arxiv.org/abs/2005.09007) -- the segmentation backbone and the
      deep-supervision scheme its seven outputs exist for.
"""

import keras
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.utils.model_build import materialize_sublayers

from .components import (
    BM_CALIBRATION_DIVISOR,
    BM_CALIBRATION_SCALE,
    FLOW_CHANNELS,
    REFINE_ITERATIONS,
    SEG_MASK_THRESHOLD,
    SEG_OUTPUT_CHANNELS,
    SPATIAL_DIVISOR,
    DocScannerFeatureEncoder,
    DocScannerUpdateBlock,
    _VARIANT_SPEC,
)
# The three private names are imported rather than restated. `_POOL_PADDING` is
# D-019's MEASURED `ceil_mode=True` remedy and `_upsample_like` is D-020's
# half-pixel resize; writing `padding="same"` or a second resize helper here
# would be a second copy of a decision that was settled by measurement, i.e. a
# rule kept in lockstep by hand. They are private to the package, not to the
# module.
from .u2net_blocks import (
    RSU4,
    RSU4F,
    RSU5,
    RSU6,
    RSU7,
    _POOL_PADDING,
    _POOL_SIZE,
    _upsample_like,
)
from .warp import convex_upsample, coords_grid, sample_at_pixel_coords

# ---------------------------------------------------------------------
# The variant table.
#
# `MODEL_VARIANTS` is DERIVED from `_VARIANT_SPEC` rather than restated. The
# spec is this port's single source of every channel count (D-006); a second
# hand-maintained copy of the same fifteen numbers is a lockstep invariant, i.e.
# a defect waiting for the first edit that touches only one of them.
#
# Two keys of a spec row are deliberately NOT constructor arguments of the
# rectifier and are dropped here:
#
# * `mask_head_output_channels` -- structural, fixed by `SPATIAL_DIVISOR ** 2 *
#   CONVEX_NEIGHBOURS`, and derived inside `DocScannerUpdateBlock` itself.
# * `seg_mid_channels` / `seg_out_channels` -- they belong to the OTHER stage.
#   `_VARIANT_SPEC` is the port's single width table and describes the whole
#   two-stage model, so from step 8 it carries the U2NET-P's widths too; the
#   rectifier's constructor has never taken them. This exclusion list is
#   therefore "not a RECTIFIER argument", not "not a real width".
# * every remaining key IS a constructor argument, which is what makes
#   `from_variant` a splat rather than a translation table.
# ---------------------------------------------------------------------

#: Spec keys that are not rectifier constructor arguments.
_SPEC_KEYS_NOT_CONSTRUCTOR_ARGS: Tuple[str, ...] = (
    "mask_head_output_channels",
    "seg_mid_channels",
    "seg_out_channels",
)

#: One line per variant, kept beside the table it annotates rather than inside
#: `_VARIANT_SPEC` (which is a WIDTH table and carries no prose).
_VARIANT_DESCRIPTIONS: Dict[str, str] = {
    "docscanner-l": (
        "The released, executable DocScanner architecture -- the one the "
        "upstream checkpoint name (inference.py:104, "
        "'./model_pretrained/DocScanner-L.pth') and the online demo both "
        "refer to. Widths read from the CALL SITES model.py:35-39 and "
        "update.py:88, never from update.py:18,36's dead defaults. "
        "DocScanner-T and DocScanner-B have no row: the paper publishes no "
        "architectural split for -T at all, and -B's only published trace "
        "(Table 1) gives the encoder column and nothing for the motion "
        "encoder or the GRU."
    ),
}


def _variant_rows() -> Dict[str, Dict[str, Any]]:
    """Project ``_VARIANT_SPEC`` onto the rectifier's constructor signature.

    Interface contract -- this has two readers, :attr:`MODEL_VARIANTS` and the
    architecture-facts guard that checks the two tables cannot drift:

    * Parameters: none.
    * Returns: ``{variant: {constructor_kwarg: value, ..., "iters": int,
      "description": str}}``. Sequence-valued widths are returned as ``list``
      so a row is JSON-serializable as-is.
    * Failure mode: raises ``KeyError`` if a spec row has no description --
      deliberately, because an undocumented variant row is how an invented
      interpolation gets shipped.

    :return: The variant table.
    :rtype: Dict[str, Dict[str, Any]]
    """
    rows: Dict[str, Dict[str, Any]] = {}
    for name, spec in _VARIANT_SPEC.items():
        row: Dict[str, Any] = {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in spec.items()
            if key not in _SPEC_KEYS_NOT_CONSTRUCTOR_ARGS
        }
        # `model.py:67` declares `iters=12` as the forward default AND
        # `inference.py:28` passes it explicitly at the only call site, so the
        # two agree -- this is not a dead default (see the REFINE_ITERATIONS
        # citation in components.py).
        row["iters"] = REFINE_ITERATIONS
        row["description"] = _VARIANT_DESCRIPTIONS[name]
        rows[name] = row
    return rows


# DECISION plan-2026-09-10T065432-05fcb6dd/D-023: the SEGMENTER's variant row is
# built by INCLUDING every `seg_`-prefixed spec key (prefix stripped to give the
# constructor argument name), while the rectifier's row above is built by
# EXCLUDING a hard-coded list. The two rules are deliberately opposite and must
# stay that way. Do NOT "make them consistent" by giving the segmenter an
# exclusion list too: `_VARIANT_SPEC` is ONE table describing a TWO-stage model
# (D-006), so a row projected by exclusion silently absorbs every key added for
# the other stage. That is not hypothetical -- step 8 added `seg_mid_channels`
# and `seg_out_channels` to the spec and three of the rectifier's step-7 tests
# went RED for exactly that reason. An inclusion rule cannot fail that way: a
# key added for the rectifier is invisible here by construction.
#
# The prefix, not a hand-written map, is what makes the rule STATED rather than
# maintained. Guarded from both sides by `TestTheSegmenterVariantTable`: every
# `seg_`-prefixed spec key must reach the row, and no un-prefixed key may.
# See decisions.md D-023.
_SEG_SPEC_KEY_PREFIX: str = "seg_"

#: One line per variant, as :data:`_VARIANT_DESCRIPTIONS` is for the rectifier.
#: Separate because the two stages are cited to two different files and were
#: sized by two different papers; a shared string would have to be vague about
#: both.
_SEGMENTER_VARIANT_DESCRIPTIONS: Dict[str, str] = {
    "docscanner-l": (
        "The U2NET-P ('P' for pruned) localization backbone that DocScanner's "
        "released inference wrapper constructs as U2NETP(3, 1) "
        "(inference.py:20). Every one of its eleven RSU stages is built at the "
        "uniform mid/out widths 16/64 (seg.py:456-478) -- read from those CALL "
        "SITES, never from RSU7.__init__'s dead mid_ch=12, out_ch=3 defaults. "
        "The FULL U2NET at seg.py:344-450 varies mid/out per stage and this "
        "port must not borrow its numbers. There is one row for the same "
        "reason the rectifier has one row: -L is the released, executable "
        "architecture."
    ),
}


def _segmenter_variant_rows() -> Dict[str, Dict[str, Any]]:
    """Project ``_VARIANT_SPEC``'s ``seg_*`` keys onto the segmenter signature.

    Interface contract -- two readers, :attr:`DocScannerSegmenter.MODEL_VARIANTS`
    and the guard that pins the projection rule:

    * Parameters: none.
    * Returns: ``{variant: {constructor_kwarg: value, ...,
      "output_channels": int, "description": str}}``, where each
      ``constructor_kwarg`` is a spec key with :data:`_SEG_SPEC_KEY_PREFIX`
      stripped.
    * Failure mode: raises ``KeyError`` if a spec row has no description --
      deliberately, for the reason :func:`_variant_rows` gives.

    See the D-023 anchor above for why this INCLUDES by prefix where
    :func:`_variant_rows` EXCLUDES by list.

    :return: The segmenter's variant table.
    :rtype: Dict[str, Dict[str, Any]]
    """
    rows: Dict[str, Dict[str, Any]] = {}
    for name, spec in _VARIANT_SPEC.items():
        row: Dict[str, Any] = {
            key[len(_SEG_SPEC_KEY_PREFIX):]: value
            for key, value in spec.items()
            if key.startswith(_SEG_SPEC_KEY_PREFIX)
        }
        row["output_channels"] = SEG_OUTPUT_CHANNELS
        row["description"] = _SEGMENTER_VARIANT_DESCRIPTIONS[name]
        rows[name] = row
    return rows


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.doc_scanner.model")
class DocScannerRectifier(keras.Model):
    """DocScanner's progressive rectification stage (``model.py:31-99``).

    See the module docstring for the loop and for the output contract. This
    class documents the knobs.

    Every width is a constructor argument with NO default. That is deliberate
    and is this plan's D-006 rule made structural: upstream carries dead
    constructor defaults (``update.py:18,36``'s ``hidden_dim=128,
    input_dim=192+128``) that no call site reaches, and a reader who trusts a
    signature over its call site silently builds a 128-wide GRU that passes
    every shape test this port has. A signature with no defaults cannot be
    read that way. Use :meth:`from_variant` or :func:`create_doc_scanner_rectifier`.

    :param hidden_dim: Width of the recurrent state ``net``. Half of
        ``fnet_output_dim`` (``model.py:35``, ``:74``).
    :type hidden_dim: int
    :param context_dim: Width of the per-iteration context ``inp``. The other
        half of ``fnet_output_dim`` (``model.py:36``, ``:74``).
    :type context_dim: int
    :param gru_input_dim: Width the GRU is built for; must equal
        ``context_dim + motion_output_dim`` (``update.py:88``).
    :type gru_input_dim: int
    :param fnet_output_dim: Channels the feature encoder emits; must equal
        ``hidden_dim + context_dim``, because the refiner SPLITS it into the
        two (``model.py:74``).
    :type fnet_output_dim: int
    :param encoder_stem_channels: Width of the encoder's 7x7 stride-2 stem.
        **80**, not the reference's inert 64 -- see the D-011 anchor in
        ``components.py``.
    :type encoder_stem_channels: int
    :param encoder_stage_channels: Per-stage encoder widths, outermost first.
        Exactly three stages, so the total stride is
        :data:`~.components.SPATIAL_DIVISOR`.
    :type encoder_stage_channels: Sequence[int]
    :param motion_output_dim: Width the motion encoder returns
        (``update.py:81``).
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
    :param mask_head_hidden: The mask head's middle width (``update.py:92``).
    :type mask_head_hidden: int
    :param iters: Refinement iterations. The ONE update block is applied this
        many times -- it is not ``iters`` separate blocks, which is why this is
        a value knob and not a structural one.
    :type iters: int
    :param kwargs: Additional keyword arguments for ``keras.Model``.
    :type kwargs: Any

    :raises ValueError: If any width or ``iters`` is not positive, if
        ``fnet_output_dim != hidden_dim + context_dim``, or if
        ``gru_input_dim != context_dim + motion_output_dim``.

    Example:
        .. code-block:: python

            model = DocScannerRectifier.from_variant("docscanner-l")
            bm = model(page, training=False)          # (B, 288, 288, 2)
            seq = model(page, training=True)          # (B, 12, 288, 288, 2)
    """

    #: Derived from ``_VARIANT_SPEC``, never restated. One row, and the reason
    #: there is only one is in that row's own ``description``.
    MODEL_VARIANTS: Dict[str, Dict[str, Any]] = _variant_rows()

    def __init__(
            self,
            hidden_dim: int,
            context_dim: int,
            gru_input_dim: int,
            fnet_output_dim: int,
            encoder_stem_channels: int,
            encoder_stage_channels: Sequence[int],
            motion_output_dim: int,
            motion_corr_hidden: int,
            motion_corr_out: int,
            motion_flow_hidden: int,
            motion_flow_out: int,
            flow_head_hidden: int,
            mask_head_hidden: int,
            iters: int = REFINE_ITERATIONS,
            **kwargs: Any
    ) -> None:
        """Validate the configuration and create both sub-layers."""
        super().__init__(**kwargs)

        for label, value in (
                ("hidden_dim", hidden_dim),
                ("context_dim", context_dim),
                ("gru_input_dim", gru_input_dim),
                ("fnet_output_dim", fnet_output_dim),
                ("encoder_stem_channels", encoder_stem_channels),
                ("motion_output_dim", motion_output_dim),
                ("motion_corr_hidden", motion_corr_hidden),
                ("motion_corr_out", motion_corr_out),
                ("motion_flow_hidden", motion_flow_hidden),
                ("motion_flow_out", motion_flow_out),
                ("flow_head_hidden", flow_head_hidden),
                ("mask_head_hidden", mask_head_hidden),
                ("iters", iters),
        ):
            if value <= 0:
                raise ValueError(f"{label} must be positive, got {value}")

        # THE SPLIT, asserted at the class that performs it. `model.py:74`
        # reads `torch.split(fmap1, [160, 160], dim=1)` -- the encoder's whole
        # output is consumed, nothing is dropped and nothing is reused. Neither
        # the encoder nor the update block can see this relationship: the
        # encoder is told only its own `output_dim`, and the update block is
        # told only the two halves. If it drifts, the slice below silently
        # discards channels (too wide an encoder) or wraps around into an empty
        # tensor (too narrow), and the second is the only one that raises.
        if fnet_output_dim != hidden_dim + context_dim:
            raise ValueError(
                f"fnet_output_dim ({fnet_output_dim}) must equal hidden_dim "
                f"({hidden_dim}) + context_dim ({context_dim}) = "
                f"{hidden_dim + context_dim}: the refiner SPLITS the encoder's "
                f"output into the tanh'd recurrent state and the relu'd "
                f"context (model.py:74), consuming all of it."
            )

        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.gru_input_dim = gru_input_dim
        self.fnet_output_dim = fnet_output_dim
        self.encoder_stem_channels = encoder_stem_channels
        self.encoder_stage_channels = tuple(
            int(width) for width in encoder_stage_channels)
        self.motion_output_dim = motion_output_dim
        self.motion_corr_hidden = motion_corr_hidden
        self.motion_corr_out = motion_corr_out
        self.motion_flow_hidden = motion_flow_hidden
        self.motion_flow_out = motion_flow_out
        self.flow_head_hidden = flow_head_hidden
        self.mask_head_hidden = mask_head_hidden
        self.iters = iters

        self.fnet = DocScannerFeatureEncoder(
            stem_channels=encoder_stem_channels,
            stage_channels=self.encoder_stage_channels,
            output_dim=fnet_output_dim,
            name="fnet",
        )

        # ONE update block, applied `iters` times -- `model.py:38` constructs a
        # single `BasicUpdateBlock` and `model.py:89` calls it inside the loop.
        # The recurrence is in the state, not in the parameters. Do NOT build a
        # list of `iters` blocks: it would train, it would serialize, and it
        # would be a different (12x larger) architecture.
        self.update_block = DocScannerUpdateBlock(
            hidden_dim=hidden_dim,
            context_dim=context_dim,
            gru_input_dim=gru_input_dim,
            motion_output_dim=motion_output_dim,
            motion_corr_hidden=motion_corr_hidden,
            motion_corr_out=motion_corr_out,
            motion_flow_hidden=motion_flow_hidden,
            motion_flow_out=motion_flow_out,
            flow_head_hidden=flow_head_hidden,
            mask_hidden=mask_head_hidden,
            name="update_block",
        )

    # -----------------------------------------------------------------

    # DECISION plan-2026-09-10T065432-05fcb6dd/D-017: the `stop_gradient` of
    # `model.py:86` lives in a METHOD OF ITS OWN purely so that it has a seam a
    # test can patch. It is not a generalisation and it has exactly one
    # production caller. Do NOT inline it back into `call()` and do NOT
    # "simplify" it away. Dropping the detach is INVISIBLE at inference -- the
    # forward values are bit-identical, so every shape, value, round-trip,
    # finiteness, serialization and knob test stays green -- and changes only
    # the gradients, which no ordinary assertion looks at; without it the
    # accumulation `coords1 = coords1 + delta_flow` carries gradient through all
    # `iters` steps and the network is trained through a 12-deep coordinate
    # recurrence it was never meant to see. The guard wraps THIS method and
    # asserts, on the tape, that no gradient crosses it, so an inlined
    # `keras.ops.stop_gradient` would leave the guard wrapping dead code and
    # silently passing. See decisions.md D-017; guarded by
    # `TestTheCoordinateFieldIsDetachedEachIteration`.
    def _detach_coordinate_field(
            self,
            coords: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """Cut the coordinate field off the backward graph. One loop step.

        Interface contract, because this method has two callers -- :meth:`call`
        and the guard that pins it:

        * Parameter: ``coords``, ``(B, H/8, W/8, FLOW_CHANNELS)``.
        * Returns: the same values, detached from the tape.
        * Failure mode: none; it is one op.

        See the D-017 anchor immediately above this method for why it is a
        method at all, and what must not be done to it.

        :param coords: The coordinate field at the top of an iteration.
        :type coords: keras.KerasTensor
        :return: The same field, with no gradient path to earlier iterations.
        :rtype: keras.KerasTensor
        """
        return keras.ops.stop_gradient(coords)

    def _require_static_multiple_of_the_stride(
            self,
            height: Optional[int],
            width: Optional[int],
    ) -> Tuple[int, int]:
        """Return ``(height, width)`` or raise, naming the offending extent.

        Interface contract -- two callers, :meth:`build` and :meth:`call`:

        * Parameters: the static height and width, either of which may be
          ``None``.
        * Returns: both as Python ``int``.
        * Failure mode: ``ValueError`` naming the axis and the value.

        Unlike the sibling ``doc_res``, ``None`` is NOT tolerated here. That
        model can be built symbolically because every op in it reads its extents
        off a tensor; this one materializes an identity coordinate field with
        :func:`~.warp.coords_grid`, which needs Python ints. Raising here, with
        the reason, is the alternative to failing several frames deeper inside
        ``keras.ops.arange``.

        :param height: Static height, or ``None``.
        :type height: Optional[int]
        :param width: Static width, or ``None``.
        :type width: Optional[int]
        :return: ``(height, width)``.
        :rtype: Tuple[int, int]
        :raises ValueError: If either extent is unknown or is not a multiple of
            :data:`~.components.SPATIAL_DIVISOR`.
        """
        unknown = [
            axis for axis, extent in (("height", height), ("width", width))
            if extent is None
        ]
        if unknown:
            raise ValueError(
                f"DocScannerRectifier needs statically-known spatial extents, "
                f"but {' and '.join(unknown)} is None. It materializes a "
                f"full-resolution identity coordinate field (model.py:47), "
                f"which is an arange per axis and cannot be built from an "
                f"unknown size. Build at a concrete shape, e.g. "
                f"(None, 288, 288, 3)."
            )

        offenders = [
            (axis, extent)
            for axis, extent in (("height", height), ("width", width))
            if extent % SPATIAL_DIVISOR != 0
        ]
        if offenders:
            described = ", ".join(
                f"{axis}={extent}" for axis, extent in offenders)
            raise ValueError(
                f"DocScannerRectifier requires spatial dimensions divisible by "
                f"{SPATIAL_DIVISOR} (the refinement loop runs at 1 / "
                f"{SPATIAL_DIVISOR} resolution and convex-upsamples by exactly "
                f"that factor), got {described}. Pad the input to a multiple "
                f"of {SPATIAL_DIVISOR} before calling and crop the prediction "
                f"back afterwards; this model does not pad internally."
            )

        return int(height), int(width)

    def _batched_coords_grid(
            self,
            like: keras.KerasTensor,
            height: int,
            width: int,
    ) -> keras.KerasTensor:
        """An identity coordinate field carrying ``like``'s batch axis.

        Interface contract -- two callers in :meth:`call`, one at full
        resolution and one at ``1 / SPATIAL_DIVISOR``:

        * Parameters: ``like``, any tensor whose batch axis and dtype the field
          must match; ``height`` / ``width``, the field's own extents, which
          need NOT be ``like``'s.
        * Returns: ``(batch, height, width, FLOW_CHANNELS)``.
        * Failure mode: raises whatever the backend raises if ``like`` is not
          rank 4.

        :func:`~.warp.coords_grid` takes an explicit batch count, but the batch
        axis is the ONE extent this model does not know statically -- a
        symbolic build hands ``call`` a ``None`` there, and
        ``keras.ops.repeat(..., None)`` is not a thing. So the field is built
        once at batch 1 and broadcast by adding a zero column sliced off
        ``like``. That is a broadcast, not a tile: it costs no materialized
        copy, it is graph-safe under an unknown batch, and it carries the dtype
        across for free, which a hard-coded ``"float32"`` would not do under
        mixed precision.

        :param like: Tensor to take the batch axis and dtype from.
        :type like: keras.KerasTensor
        :param height: Field height in pixels.
        :type height: int
        :param width: Field width in pixels.
        :type width: int
        :return: The identity field, in ``(x, y)`` channel order.
        :rtype: keras.KerasTensor
        """
        field = coords_grid(1, height, width, dtype=like.dtype)
        return field + keras.ops.zeros_like(like[..., :1])

    # -----------------------------------------------------------------

    def build(self, input_shape: Any) -> None:
        """Materialize both sub-layers by tracing ``call`` symbolically.

        Without this, a subclassed model inherits ``Layer.build``, which marks
        the model built while every sub-layer is still unbuilt -- so a
        ``.keras`` reload would restore nothing and the first forward pass
        would create fresh random weights with nothing raising (H-6).

        :param input_shape: ``(batch, height, width, 3)``. Height and width
            must be concrete and divisible by
            :data:`~.components.SPATIAL_DIVISOR`.
        :type input_shape: Any
        :raises ValueError: If the shape is not rank 4, or if a spatial extent
            is unknown or not a multiple of the stride.
        """
        if self.built:
            return

        if len(input_shape) != 4:
            raise ValueError(
                f"Expected 4D input shape (batch, height, width, channels), "
                f"got {len(input_shape)}D: {input_shape}"
            )
        self._require_static_multiple_of_the_stride(
            input_shape[1], input_shape[2])

        materialize_sublayers(self, input_shape)
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Refine a backward map over :attr:`iters` iterations.

        :param inputs: ``(batch, height, width, 3)``, channels-last RGB.
            Height and width must be divisible by
            :data:`~.components.SPATIAL_DIVISOR`.
        :type inputs: keras.KerasTensor
        :param training: Standard Keras training flag. It also SELECTS THE
            OUTPUT FORM -- see below and the module docstring.
        :type training: Optional[bool]
        :return: ``(batch, iters, height, width, FLOW_CHANNELS)`` when
            ``training`` is true, i.e. the whole refinement sequence stacked
            along a new axis 1, oldest first; otherwise
            ``(batch, height, width, FLOW_CHANNELS)``, the last iteration only.
            Either way the values are ABSOLUTE FULL-RESOLUTION PIXEL
            coordinates in ``(x, y)`` channel order, not normalized.
        :rtype: keras.KerasTensor
        :raises ValueError: If a statically-known spatial extent is unknown or
            not divisible by the stride. Eager calls always know their extents,
            so this fires on every call, not only the first.
        """
        static_shape = inputs.shape
        height, width = self._require_static_multiple_of_the_stride(
            static_shape[1], static_shape[2])

        fmap1 = self.fnet(inputs, training=training)

        # `model.py:74-76`: split 320 into two 160s, tanh the state and relu
        # the context. The split is CONTIGUOUS and in that order -- state
        # first. Swapping the two halves is shape-legal at the shipped variant
        # (both are 160 wide) and merely feeds tanh the channels relu was
        # trained for.
        net = keras.ops.tanh(fmap1[..., :self.hidden_dim])
        inp = keras.ops.relu(fmap1[..., self.hidden_dim:])

        # DECISION plan-2026-09-10T065432-05fcb6dd/D-017: `warpfea` starts as
        # `fmap1` ITSELF (`model.py:72`, before the loop) and is RE-SAMPLED FROM
        # `fmap1` at the end of every iteration (`model.py:92`) -- never from
        # the previous `warpfea`. Do NOT write `warpfea = sample(warpfea,
        # coords1)`: chaining is shape-identical, finite, trainable and
        # serializable, and is wrong in a way that only appears from the THIRD
        # iteration on (at iteration 1 `warpfea` still is `fmap1`, and at
        # iteration 2 the chained and unchained forms coincide because the
        # first sample was taken at the identity grid). It would also compound
        # bilinear smoothing 12 times, quietly low-passing the features the
        # motion encoder reads. Guarded by
        # ``TestWarpedFeaturesAreResampledFromTheEncoderOutput``, which spies on
        # every call and requires all of them to receive the SAME array.
        #
        # DECISION plan-2026-09-10T065432-05fcb6dd/D-055: the STATEMENT ORDER
        # inside the loop below is part of that claim. ``coords1 = coords1 +
        # delta_flow`` comes FIRST and the resample LAST. Do NOT hoist
        # ``warpfea = sample_at_pixel_coords(fmap1, coords1)`` above the
        # coordinate update "to group the sampling with the flow": that reads
        # the SAME ``fmap1`` on every call (so the spy above stays green) and
        # merely lags the features by one iteration -- shape-identical, finite,
        # trainable, serializable, and MEASURED to move the emitted map by
        # ``max|delta| = 0.916`` px. Guarded by the same class's
        # ``test_the_first_sample_is_taken_AFTER_the_first_coordinate_update``,
        # which inspects the sampler's SECOND argument.
        warpfea = fmap1

        # `model.py:46-51`. `coodslar` is FULL resolution and is what makes the
        # emitted map absolute; `coords0` and `coords1` are the 1/8-resolution
        # field, identical at entry, so the first `flow` is exactly zero.
        coodslar = self._batched_coords_grid(inputs, height, width)
        coords0 = self._batched_coords_grid(
            fmap1,
            height // SPATIAL_DIVISOR,
            width // SPATIAL_DIVISOR,
        )
        coords1 = coords0

        predictions: List[keras.KerasTensor] = []
        for _ in range(self.iters):
            coords1 = self._detach_coordinate_field(coords1)
            flow = coords1 - coords0

            net, mask, delta_flow = self.update_block(
                [net, inp, warpfea, flow], training=training)

            coords1 = coords1 + delta_flow

            # The mask arrives already scaled by MASK_LOGIT_SCALE (the 0.25 of
            # `update.py:104`); `convex_upsample` documents and is guarded as
            # taking RAW logits, so it is NOT scaled again here.
            flow_up = convex_upsample(coords1 - coords0, mask)

            # ABSOLUTE pixel coordinates. The `(2 * bm / 286.8 - 1) * 0.99`
            # calibration of `inference.py:29` belongs to the composite
            # pipeline that hands this to a sampler, not here.
            predictions.append(coodslar + flow_up)

            warpfea = sample_at_pixel_coords(fmap1, coords1)

        if training:
            return keras.ops.stack(predictions, axis=1)
        return predictions[-1]

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """The INFERENCE shape. The training shape carries an extra axis.

        Keras' ``compute_output_shape`` takes no ``training`` flag, and the
        forward's output rank depends on it (see :meth:`call`). The inference
        form is reported because that is the one a functional graph, a
        ``predict()`` and every downstream consumer of a backward map see; the
        training form is ``(batch, iters, height, width, FLOW_CHANNELS)`` and is
        stated in :meth:`call`'s docstring rather than guessed at here.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple
        :return: ``(batch, height, width, FLOW_CHANNELS)``.
        :rtype: tuple
        """
        batch, height, width, _ = input_shape
        return (batch, height, width, FLOW_CHANNELS)

    def get_config(self) -> Dict[str, Any]:
        """Return every constructor argument needed to recreate this model.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "context_dim": self.context_dim,
            "gru_input_dim": self.gru_input_dim,
            "fnet_output_dim": self.fnet_output_dim,
            "encoder_stem_channels": self.encoder_stem_channels,
            "encoder_stage_channels": list(self.encoder_stage_channels),
            "motion_output_dim": self.motion_output_dim,
            "motion_corr_hidden": self.motion_corr_hidden,
            "motion_corr_out": self.motion_corr_out,
            "motion_flow_hidden": self.motion_flow_hidden,
            "motion_flow_out": self.motion_flow_out,
            "flow_head_hidden": self.flow_head_hidden,
            "mask_head_hidden": self.mask_head_hidden,
            "iters": self.iters,
        })
        return config

    @classmethod
    def from_variant(
            cls,
            variant: str,
            pretrained: bool = False,
            **kwargs: Any
    ) -> "DocScannerRectifier":
        """Create a rectifier from the one cited configuration.

        :param variant: A key of :attr:`MODEL_VARIANTS`. Currently only
            ``"docscanner-l"``; the row's ``description`` says why there is no
            ``-T`` or ``-B``.
        :type variant: str
        :param pretrained: Must be ``False``.
        :type pretrained: bool
        :param kwargs: Constructor overrides applied on top of the variant's
            configuration.
        :type kwargs: Any
        :return: The constructed, unbuilt model.
        :rtype: DocScannerRectifier
        :raises ValueError: If ``variant`` is not a known key; the message
            lists the available ones.
        :raises NotImplementedError: If ``pretrained`` is ``True``.
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown DocScanner variant '{variant}'. Available variants: "
                f"{sorted(cls.MODEL_VARIANTS.keys())}"
            )

        if pretrained:
            raise NotImplementedError(
                f"No pretrained weights are distributed for DocScanner "
                f"variant '{variant}', and the upstream PyTorch checkpoint "
                f"cannot be transferred into this port. Two structural "
                f"divergences make a transferred checkpoint silently wrong "
                f"rather than merely untested: (1) the feature encoder's "
                f"`norm1` is built at {cls.MODEL_VARIANTS[variant]['encoder_stem_channels']} "
                f"channels here, against the reference's literal 64 "
                f"(extractor.py:93), which is inert upstream only because "
                f"torch's InstanceNorm2d defaults to affine=False -- see the "
                f"D-011 anchor in components.py; and (2) every stride-2 "
                f"convolution pads asymmetrically to reproduce torch's "
                f"symmetric padding, which Keras 'same' does not do -- see the "
                f"D-012 anchor. Train from scratch with "
                f"`src/train/doc_scanner/`, or restore your own checkpoint "
                f"with `model.load_weights(path)`."
            )

        config = dict(cls.MODEL_VARIANTS[variant])
        config.pop("description", None)
        config.update(kwargs)
        return cls(**config)


# ---------------------------------------------------------------------


def create_doc_scanner_rectifier(
        variant: str = "docscanner-l",
        pretrained: bool = False,
        **kwargs: Any
) -> DocScannerRectifier:
    """Create a DocScanner rectifier. The module-level entry point.

    :param variant: A key of :attr:`DocScannerRectifier.MODEL_VARIANTS`.
        Defaults to ``"docscanner-l"``.
    :type variant: str
    :param pretrained: Must be ``False``; see
        :meth:`DocScannerRectifier.from_variant`.
    :type pretrained: bool
    :param kwargs: Constructor overrides forwarded unchanged.
    :type kwargs: Any
    :return: The constructed, unbuilt model.
    :rtype: DocScannerRectifier
    """
    return DocScannerRectifier.from_variant(
        variant, pretrained=pretrained, **kwargs)


# ---------------------------------------------------------------------
# Stage 1: the U2NET-P segmenter.
# ---------------------------------------------------------------------

#: The six encoder stages of ``U2NETP``, outermost first (``seg.py:456-470``).
#: The two innermost are the pooling-FREE ``RSU4F``: by 1/16 and 1/32 of the
#: input there is nothing left to pool, so the block grows its receptive field
#: by dilation instead. This tuple's LENGTH is the ladder depth; it is not a
#: knob, because each entry is a different class.
_ENCODER_STAGE_CLASSES: Tuple[type, ...] = (RSU7, RSU6, RSU5, RSU4, RSU4F, RSU4F)

#: The five decoder stages, DEEPEST FIRST -- ``stage5d, stage4d, stage3d,
#: stage2d, stage1d`` (``seg.py:473-477``). DERIVED from the encoder tuple
#: rather than written out again: the U2NET-P is a symmetric U, so the decoder
#: stage that consumes ``hxN``'s skip is the same CLASS as the encoder stage
#: that produced it (``stage5d``/``stage5`` are both ``RSU4F``,
#: ``stage1d``/``stage1`` both ``RSU7``). A second hand-written tuple would be
#: that symmetry maintained by hand.
_DECODER_STAGE_CLASSES: Tuple[type, ...] = tuple(
    reversed(_ENCODER_STAGE_CLASSES[:-1]))

#: The number of 3x3 side-supervision heads (``seg.py:479-484``). SIX, against
#: FIVE decoder stages -- see the D-022 anchor inside
#: :meth:`DocScannerSegmenter.call`.
_SIDE_HEAD_COUNT: int = len(_DECODER_STAGE_CLASSES) + 1

#: ``side1..side6 = nn.Conv2d(64, out_ch, 3, padding=1)``, ``seg.py:479-484``.
_SIDE_HEAD_KERNEL_SIZE: int = 3

#: ``outconv = nn.Conv2d(6, out_ch, 1)``, ``seg.py:486`` -- a 1x1 FUSION over
#: the six stacked side maps. Its input width is ``_SIDE_HEAD_COUNT *
#: output_channels``, which is where upstream's literal 6 comes from.
_FUSION_KERNEL_SIZE: int = 1


@register_dl_technique("dl_techniques.models.doc_scanner.model")
class DocScannerSegmenter(keras.Model):
    """DocScanner's localization stage: a pruned U2-Net (``seg.py:451-543``).

    A six-stage encoder ladder of ReSidual U-blocks, a five-stage decoder that
    mirrors it, six 3x3 side-supervision heads resized onto the outermost head's
    resolution, and a 1x1 convolution that FUSES those six into the map anyone
    actually uses. Everything structural is in :mod:`.u2net_blocks`; this class
    is the wiring.

    The output contract: SEVEN maps
    -------------------------------
    ``call`` returns a ``list`` of seven tensors, each
    ``(batch, height, width, output_channels)`` and each already through a
    ``sigmoid``, in upstream's order ``[d0, d1, d2, d3, d4, d5, d6]``
    (``seg.py:549-550``).

    * ``d0`` is the FUSION and the only one the pipeline consumes --
      ``inference.py:24`` unpacks ``msk, _1, ..., _6 = self.msk(x)`` and drops
      the rest.
    * ``d1..d6`` exist for DEEP SUPERVISION: U2-Net's loss is the sum of the
      BCE of all seven, which is why they are outputs at all and not debug
      state. They are returned rather than hidden for the same reason
      :class:`DocScannerRectifier` returns its whole iteration sequence -- this
      repo forbids a custom ``train_step`` (H-5), so anything the loss needs has
      to arrive as a model OUTPUT.

    All seven are at the INPUT resolution. That is not a coincidence of 288
    being divisible by 32: ``d1``'s source ``hx1d`` is the outermost decoder
    stage, which never left the input resolution, and every other head is
    resized onto ``d1``.

    Any input size works
    --------------------
    Unlike :class:`DocScannerRectifier`, which needs extents divisible by 8 and
    statically known, this model imposes NOTHING on height and width. The
    ``ceil_mode`` pooling remedy (D-019) and the resize-onto-the-skip decoder
    (D-020) between them absorb a ragged ladder: at height 37 the encoder levels
    run ``37, 19, 10, 5, 3, 2`` and every concatenation still meets at the
    skip's own size.

    :param mid_channels: Upstream's ``mid_ch``, the width of every RSU block's
        interior. 16 for the pruned variant; read from
        ``_VARIANT_SPEC["seg_mid_channels"]``, never as a literal.
    :type mid_channels: int
    :param out_channels: Upstream's ``out_ch``, the width every RSU block emits
        and therefore the width of every skip. 64 for the pruned variant.
    :type out_channels: int
    :param output_channels: Channels each of the seven maps carries. One
        confidence plane (``inference.py:20``, ``U2NETP(3, 1)``).
    :type output_channels: int
    :param kwargs: Additional keyword arguments for ``keras.Model``.
    :type kwargs: Any

    :raises ValueError: If any of the three widths is not positive.

    Example:
        .. code-block:: python

            model = DocScannerSegmenter.from_variant("docscanner-l")
            maps = model(page, training=False)        # 7 x (B, 288, 288, 1)
            mask = keras.ops.cast(maps[0] > 0.5, page.dtype)
    """

    #: Derived from ``_VARIANT_SPEC``'s ``seg_*`` rows, never restated. See the
    #: D-023 anchor above :func:`_segmenter_variant_rows`.
    MODEL_VARIANTS: Dict[str, Dict[str, Any]] = _segmenter_variant_rows()

    def __init__(
            self,
            mid_channels: int,
            out_channels: int,
            output_channels: int = SEG_OUTPUT_CHANNELS,
            **kwargs: Any
    ) -> None:
        """Validate the widths and create every sub-layer."""
        super().__init__(**kwargs)

        for label, value in (
                ("mid_channels", mid_channels),
                ("out_channels", out_channels),
                ("output_channels", output_channels),
        ):
            if value <= 0:
                raise ValueError(f"{label} must be positive, got {value}")

        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.output_channels = output_channels

        # Every sub-layer is created HERE, unconditionally -- never in `build`
        # and never on first call. `build` only materializes them (H-6).
        self.encoder_stages: List[keras.layers.Layer] = [
            stage_class(
                mid_channels=mid_channels,
                out_channels=out_channels,
                name=f"stage{index + 1}",
            )
            for index, stage_class in enumerate(_ENCODER_STAGE_CLASSES)
        ]

        # One fewer pool than encoder stages: `seg.py` pools after every stage
        # except the deepest (`stage6` has no pool).
        self.pools: List[keras.layers.MaxPooling2D] = [
            keras.layers.MaxPooling2D(
                pool_size=_POOL_SIZE,
                strides=_POOL_SIZE,
                padding=_POOL_PADDING,
                name=f"pool{index + 1}{index + 2}",
            )
            for index in range(len(_ENCODER_STAGE_CLASSES) - 1)
        ]

        # Deepest first, so `decoder_stages[0]` is `stage5d`. Each consumes
        # `concat(upsampled_deeper, skip)`; both halves are `out_channels` wide,
        # so the declared input is `2 * out_channels` -- upstream's literal 128
        # DERIVED, not transcribed (`seg.py:473-477`, `RSU4F(128, 16, 64)`). The
        # RSU blocks infer their input width from the tensor, so that derivation
        # is not passed anywhere and is instead asserted on the built weights by
        # `TestTheDecoderConsumesTwiceTheSkipWidth`.
        self.decoder_stages: List[keras.layers.Layer] = [
            stage_class(
                mid_channels=mid_channels,
                out_channels=out_channels,
                name=f"stage{len(_DECODER_STAGE_CLASSES) - index}d",
            )
            for index, stage_class in enumerate(_DECODER_STAGE_CLASSES)
        ]

        self.side_convs: List[keras.layers.Conv2D] = [
            keras.layers.Conv2D(
                filters=output_channels,
                kernel_size=_SIDE_HEAD_KERNEL_SIZE,
                padding="same",
                name=f"side{index + 1}",
            )
            for index in range(_SIDE_HEAD_COUNT)
        ]

        self.outconv = keras.layers.Conv2D(
            filters=output_channels,
            kernel_size=_FUSION_KERNEL_SIZE,
            padding="same",
            name="outconv",
        )

    # -----------------------------------------------------------------

    def build(self, input_shape: Any) -> None:
        """Materialize every sub-layer by tracing ``call`` symbolically.

        Without this, a subclassed model inherits ``Layer.build``, which marks
        the model built while every sub-layer is still unbuilt -- so a
        ``.keras`` reload would restore nothing and the first forward pass would
        create fresh random weights with nothing raising (H-6).

        :param input_shape: ``(batch, height, width, channels)``. No constraint
            on height or width; see the class docstring.
        :type input_shape: Any
        :raises ValueError: If the shape is not rank 4.
        """
        if self.built:
            return

        if len(input_shape) != 4:
            raise ValueError(
                f"DocScannerSegmenter expects a 4D input shape "
                f"(batch, height, width, channels), got {len(input_shape)}D: "
                f"{input_shape}"
            )

        materialize_sublayers(self, input_shape)
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> List[keras.KerasTensor]:
        """Run the nested U and emit seven confidence maps.

        :param inputs: ``(batch, height, width, channels)``, channels-last.
            Upstream feeds raw 3-channel RGB (``inference.py:20``); the
            ``sobel_net`` edge channel defined at ``seg.py:8-31`` is DEAD CODE
            upstream and is deliberately not ported.
        :type inputs: keras.KerasTensor
        :param training: Forwarded to every block; the RSU blocks contain
            ``BatchNormalization``, so it is load-bearing here in a way it is
            not for the rectifier.
        :type training: Optional[bool]
        :return: Seven tensors ``[d0, d1, ..., d6]``, each
            ``(batch, height, width, output_channels)`` and each in ``[0, 1]``.
            ``d0`` is the fusion; see the class docstring.
        :rtype: List[keras.KerasTensor]
        """
        # Down: `hx1 .. hx6`, pooling between stages but not after the last.
        encoder_outputs: List[keras.KerasTensor] = []
        level = inputs
        for index, stage in enumerate(self.encoder_stages):
            level = stage(level, training=training)
            encoder_outputs.append(level)
            if index < len(self.pools):
                level = self.pools[index](level)

        # Up: `hx5d .. hx1d`, collected deepest-first to match
        # `self.decoder_stages`. The deepest decoder's partner is `hx6` itself
        # (`seg.py:515-517`: `hx6up = _upsample_like(hx6, hx5)` then
        # `stage5d(cat((hx6up, hx5)))`), so unlike the ladder INSIDE an RSU
        # block every decoder here resizes -- there is no unpooled bottom.
        decoder_outputs: List[keras.KerasTensor] = []
        deeper = encoder_outputs[-1]
        for index, stage in enumerate(self.decoder_stages):
            skip = encoder_outputs[-2 - index]
            # ORDER: deeper FIRST, skip second -- `torch.cat((hx6up, hx5), 1)`.
            # Both halves are `out_channels` wide, so a swap is entirely
            # shape-preserving. Guarded by `TestTheDecoderConcatenationOrder`.
            merged = keras.ops.concatenate(
                [_upsample_like(deeper, skip), skip], axis=-1)
            deeper = stage(merged, training=training)
            decoder_outputs.append(deeper)

        # DECISION plan-2026-09-10T065432-05fcb6dd/D-022: `side6` reads
        # `encoder_outputs[-1]` -- the ENCODER's deepest output `hx6` -- and NOT
        # a decoder output. There are SIX side heads and only FIVE decoder
        # stages, because the U2NET-P supervises the bottom of the U where the
        # decoder has not started yet: `seg.py:546` is `d6 = self.side6(hx6)`,
        # against `d1..d5 = sideN(hx{N}d)`. Do NOT "fix" the asymmetry by
        # appending a sixth decoder stage, and do NOT feed `side6` the deepest
        # decoder output `hx5d`: `hx5d` and `hx6` are both `out_channels` wide
        # and both are resized onto `d1` anyway, so either error is entirely
        # shape-preserving -- it still yields seven finite maps in [0, 1] that
        # train and serialize. Guarded by
        # `TestTheSixthSideHeadReadsTheEncoder`. See decisions.md D-022.
        side_sources: List[keras.KerasTensor] = list(reversed(decoder_outputs))
        side_sources.append(encoder_outputs[-1])

        side_logits = [
            conv(source)
            for conv, source in zip(self.side_convs, side_sources)
        ]

        # `d1` is the reference resolution -- it comes off `hx1d`, which never
        # left the input resolution -- and upstream applies no resize to it at
        # all (`seg.py:534`). Every other head is resized ONTO it.
        reference = side_logits[0]
        resized_logits = [reference] + [
            _upsample_like(logit, reference) for logit in side_logits[1:]
        ]

        # The FUSION. `d0` is a learned 1x1 combination of all six, not a copy
        # of any one of them: `outconv(cat((d1, ..., d6), 1))`, `seg.py:548`.
        fused = self.outconv(keras.ops.concatenate(resized_logits, axis=-1))

        return [
            keras.ops.sigmoid(logit) for logit in [fused] + resized_logits
        ]

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> List[Tuple[Optional[int], ...]]:
        """Seven identically-shaped maps at the INPUT resolution.

        :param input_shape: ``(batch, height, width, channels)``.
        :type input_shape: tuple
        :return: A list of seven ``(batch, height, width, output_channels)``.
        :rtype: List[tuple]
        """
        batch, height, width, _ = input_shape
        return [
            (batch, height, width, self.output_channels)
        ] * (_SIDE_HEAD_COUNT + 1)

    def get_config(self) -> Dict[str, Any]:
        """Return every constructor argument needed to recreate this model.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "mid_channels": self.mid_channels,
            "out_channels": self.out_channels,
            "output_channels": self.output_channels,
        })
        return config

    @classmethod
    def from_variant(
            cls,
            variant: str,
            pretrained: bool = False,
            **kwargs: Any
    ) -> "DocScannerSegmenter":
        """Create a segmenter from the one cited configuration.

        :param variant: A key of :attr:`MODEL_VARIANTS`. Currently only
            ``"docscanner-l"``; the row's ``description`` says why.
        :type variant: str
        :param pretrained: Must be ``False``.
        :type pretrained: bool
        :param kwargs: Constructor overrides applied on top of the variant's
            configuration.
        :type kwargs: Any
        :return: The constructed, unbuilt model.
        :rtype: DocScannerSegmenter
        :raises ValueError: If ``variant`` is not a known key; the message lists
            the available ones.
        :raises NotImplementedError: If ``pretrained`` is ``True``.
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown DocScanner variant '{variant}'. Available variants: "
                f"{sorted(cls.MODEL_VARIANTS.keys())}"
            )

        if pretrained:
            raise NotImplementedError(
                f"No pretrained weights are distributed for DocScanner "
                f"segmentation variant '{variant}'. There is no Keras "
                f"checkpoint, and the upstream PyTorch one cannot be "
                f"transferred because IT DOES NOT EXIST IN THE REFERENCE "
                f"CHECKOUT: inference.py:103 loads './model_pretrained/seg.pth' "
                f"and no such file is present there -- only an external "
                f"download link in the upstream README. That absence is also "
                f"why the one clue to the checkpoint's key layout cannot be "
                f"acted on: inference.py:40's `reload_seg_model` keeps "
                f"`{{k[6:]: v ...}}`, i.e. it strips a SIX-CHARACTER key "
                f"prefix, and which prefix that is -- and whether every key "
                f"carries it -- is unverifiable without the file. A conversion "
                f"script written against that guess would fail SILENTLY for "
                f"every key it mismatched, because `reload_seg_model` filters "
                f"non-matching keys out rather than raising. Train from "
                f"scratch with `src/train/doc_scanner/`, or restore your own "
                f"checkpoint with `model.load_weights(path)`."
            )

        config = dict(cls.MODEL_VARIANTS[variant])
        config.pop("description", None)
        config.update(kwargs)
        return cls(**config)


# ---------------------------------------------------------------------


def create_doc_scanner_segmenter(
        variant: str = "docscanner-l",
        pretrained: bool = False,
        **kwargs: Any
) -> DocScannerSegmenter:
    """Create a DocScanner segmenter. The module-level entry point.

    :param variant: A key of :attr:`DocScannerSegmenter.MODEL_VARIANTS`.
        Defaults to ``"docscanner-l"``.
    :type variant: str
    :param pretrained: Must be ``False``; see
        :meth:`DocScannerSegmenter.from_variant`.
    :type pretrained: bool
    :param kwargs: Constructor overrides forwarded unchanged.
    :type kwargs: Any
    :return: The constructed, unbuilt model.
    :rtype: DocScannerSegmenter
    """
    return DocScannerSegmenter.from_variant(
        variant, pretrained=pretrained, **kwargs)


# ---------------------------------------------------------------------
# The composite: stage 1 -> threshold -> multiplicative mask -> stage 2 ->
# calibration. `inference.py:17-31`.
# ---------------------------------------------------------------------

#: Where a composite variant row keeps the segmenter's own row.
_COMPOSITE_SEGMENTER_KEY: str = "segmenter_config"

#: Where a composite variant row keeps the rectifier's own row.
_COMPOSITE_RECTIFIER_KEY: str = "rectifier_config"

#: One line per variant, as for each stage separately. It says only what is
#: true of the PAIR; the two stages' own descriptions are not repeated here and
#: are reachable through :attr:`DocScannerSegmenter.MODEL_VARIANTS` and
#: :attr:`DocScannerRectifier.MODEL_VARIANTS`.
_COMPOSITE_VARIANT_DESCRIPTIONS: Dict[str, str] = {
    "docscanner-l": (
        "Both released stages wired as `inference.py:17-31` wires them: "
        "U2NETP -> threshold at 0.5 -> multiplicative background mask -> the "
        "12-iteration rectifier -> the (2 * bm / 286.8 - 1) * 0.99 "
        "calibration. There is one row because each stage has one row, and "
        "the reason each has one is in that stage's own description."
    ),
}


def _composite_variant_rows() -> Dict[str, Dict[str, Any]]:
    """Pair each stage's variant row under one key.

    Interface contract -- two readers, :attr:`DocScanner.MODEL_VARIANTS` and
    the guard that pins the pairing:

    * Parameters: none.
    * Returns: ``{variant: {"segmenter_config": {...},
      "rectifier_config": {...}, "description": str}}``, where the two nested
      dicts are exactly what :meth:`DocScannerSegmenter.from_variant` and
      :meth:`DocScannerRectifier.from_variant` would construct from, minus
      their own ``description``.
    * Failure mode: raises ``KeyError`` if a spec row has no composite
      description, for the reason :func:`_variant_rows` gives.

    It CALLS the two stage projections rather than re-projecting
    ``_VARIANT_SPEC`` a third time. A third projection would be a third
    encoding of the D-006/D-023 rules, kept in lockstep by hand.

    :return: The composite's variant table.
    :rtype: Dict[str, Dict[str, Any]]
    """
    segmenter_rows = _segmenter_variant_rows()
    rectifier_rows = _variant_rows()

    rows: Dict[str, Dict[str, Any]] = {}
    for name in _VARIANT_SPEC:
        segmenter_config = dict(segmenter_rows[name])
        segmenter_config.pop("description", None)
        rectifier_config = dict(rectifier_rows[name])
        rectifier_config.pop("description", None)
        rows[name] = {
            _COMPOSITE_SEGMENTER_KEY: segmenter_config,
            _COMPOSITE_RECTIFIER_KEY: rectifier_config,
            "description": _COMPOSITE_VARIANT_DESCRIPTIONS[name],
        }
    return rows


@register_dl_technique("dl_techniques.models.doc_scanner.model")
class DocScanner(keras.Model):
    """The two stages wired together. An INFERENCE assembly (``inference.py:17-31``).

    .. code-block:: text

        msk, _1, ..., _6 = segmenter(x)      seven maps; only the first is used
        msk              = (msk > 0.5)       BINARY, not soft
        x                = msk * x           MULTIPLICATIVE, not concatenated
        bm               = rectifier(x)      (B, H, W, 2), absolute pixel units
        bm               = (2 * bm / 286.8 - 1) * 0.99

    This class cannot be trained end to end, and that is not a limitation of
    the port
    -------------------------------------------------------------------------
    The threshold ``(msk > 0.5)`` has zero gradient almost everywhere, so no
    gradient reaches the segmenter through it -- ever. That matches the paper,
    which trains the two modules INDEPENDENTLY (§4.3, finding F-15), and it is
    why there are two trainers under ``src/train/doc_scanner/`` and not one.
    Train :class:`DocScannerSegmenter` and :class:`DocScannerRectifier`
    separately and assemble them here for inference. The property is pinned by
    ``TestNoGradientReachesTheSegmenter`` rather than left as a claim; see the
    D-025 anchor in :meth:`call`.

    The output units
    ----------------
    :class:`DocScannerRectifier` emits ABSOLUTE full-resolution pixel
    coordinates. This class emits the CALIBRATED map, nominally
    ``[-0.99, 0.99]`` -- the range a normalized sampler wants. The two are not
    interchangeable; see the D-027 anchor in :meth:`call` for what ``286.8`` is
    and, more importantly, what it is not.

    **That range is NOT enforced, and an untrained model leaves it** (D-054).
    The rectifier's map is an unconstrained regression, nothing clamps it to
    the image domain, and the calibration is affine, so it inherits whatever
    the rectifier emits. MEASURED on a freshly initialized ``docscanner-l`` at
    288x288, seeds 0/1/2: the rectifier spans ``[-61.0, +329.4]`` px and this
    class ``[-1.411, +1.284]``. Read ``[-0.99, 0.99]`` as a property of a
    CONVERGED model, not as a contract. The failure is SILENT -- the downstream
    sampler is edge-clamped (D-056), so an out-of-domain map degrades the
    gather instead of raising. README section 2 carries the same caveat.

    :param segmenter_config: Constructor keyword arguments for
        :class:`DocScannerSegmenter`.
    :type segmenter_config: Dict[str, Any]
    :param rectifier_config: Constructor keyword arguments for
        :class:`DocScannerRectifier`.
    :type rectifier_config: Dict[str, Any]
    :param kwargs: Additional keyword arguments for ``keras.Model``.
    :type kwargs: Any

    :raises ValueError: If either configuration is not a non-empty mapping, or
        if either stage rejects its own arguments.

    Example:
        .. code-block:: python

            scanner = create_doc_scanner("docscanner-l")
            scanner.build((None, 288, 288, 3))
            bm = scanner(page, training=False)        # (B, 288, 288, 2)
    """

    #: Derived by PAIRING the two stages' own tables, never restated.
    MODEL_VARIANTS: Dict[str, Dict[str, Any]] = _composite_variant_rows()

    def __init__(
            self,
            segmenter_config: Dict[str, Any],
            rectifier_config: Dict[str, Any],
            **kwargs: Any
    ) -> None:
        """Validate both configurations and create both stages."""
        super().__init__(**kwargs)

        for label, config in (
                (_COMPOSITE_SEGMENTER_KEY, segmenter_config),
                (_COMPOSITE_RECTIFIER_KEY, rectifier_config),
        ):
            if not isinstance(config, dict) or not config:
                raise ValueError(
                    f"{label} must be a non-empty dict of constructor keyword "
                    f"arguments for that stage, got {config!r}. Use "
                    f"`DocScanner.from_variant('docscanner-l')` or "
                    f"`create_doc_scanner()`; neither stage has constructor "
                    f"defaults, deliberately (see the D-006 anchor in "
                    f"components.py)."
                )

        self.segmenter_config = dict(segmenter_config)
        self.rectifier_config = dict(rectifier_config)

        # Both stages are created HERE, unconditionally. Each validates its own
        # widths in its own constructor, so this class re-validates nothing:
        # a duplicated width check is a rule kept in lockstep by hand.
        self.segmenter = DocScannerSegmenter(
            **self.segmenter_config, name="segmenter")
        self.rectifier = DocScannerRectifier(
            **self.rectifier_config, name="rectifier")

    # -----------------------------------------------------------------

    def build(self, input_shape: Any) -> None:
        """Materialize both stages by tracing ``call`` symbolically.

        :param input_shape: ``(batch, height, width, 3)``. Height and width
            must be concrete and divisible by
            :data:`~.components.SPATIAL_DIVISOR` -- the RECTIFIER's constraint,
            checked here so the message names this model's own stage rather
            than arriving from two frames down.
        :type input_shape: Any
        :raises ValueError: If the shape is not rank 4, or if a spatial extent
            is unknown or not a multiple of the stride.
        """
        if self.built:
            return

        if len(input_shape) != 4:
            raise ValueError(
                f"DocScanner expects a 4D input shape "
                f"(batch, height, width, channels), got {len(input_shape)}D: "
                f"{input_shape}"
            )

        # The rectifier's own check, CALLED rather than copied. The segmenter
        # imposes nothing, so the composite's contract is exactly the
        # rectifier's and any second copy of the divisibility rule would be a
        # second thing to update.
        self.rectifier._require_static_multiple_of_the_stride(
            input_shape[1], input_shape[2])

        materialize_sublayers(self, input_shape)
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Segment, mask, rectify, calibrate.

        :param inputs: ``(batch, height, width, 3)``, channels-last RGB in
            ``[0, 1]``. Height and width must be divisible by
            :data:`~.components.SPATIAL_DIVISOR`.
        :type inputs: keras.KerasTensor
        :param training: Forwarded to the SEGMENTER only, whose RSU blocks
            carry ``BatchNormalization``. See the D-026 anchor below for why
            the rectifier is called with ``training=False`` unconditionally.
        :type training: Optional[bool]
        :return: ``(batch, height, width, FLOW_CHANNELS)`` -- the CALIBRATED
            backward map, in ``(x, y)`` channel order, NOMINALLY
            ``[-BM_CALIBRATION_SCALE, +BM_CALIBRATION_SCALE]``. That bound is
            not enforced and an untrained model leaves it (D-054: measured
            ``[-1.411, +1.284]`` on a fresh ``docscanner-l``); see the class
            docstring and README section 2.
        :rtype: keras.KerasTensor
        """
        # `inference.py:24`: `msk, _1,_2,_3,_4,_5,_6 = self.msk(x)`. Six of the
        # seven maps are DEEP-SUPERVISION outputs and are dropped here; `d0`,
        # the fusion, is the only one the pipeline consumes.
        confidence = self.segmenter(inputs, training=training)[0]

        # DECISION plan-2026-09-10T065432-05fcb6dd/D-025: `inference.py:25` is
        # `msk = (msk > 0.5).float()` and then `x = msk * x` -- a HARD threshold
        # and a MULTIPLICATIVE mask. Both halves are load-bearing and both have
        # a tempting wrong variant:
        #
        # * Do NOT replace the threshold with the raw sigmoid (a "soft mask"),
        #   and do NOT add a straight-through estimator or a temperature to
        #   make it differentiable. It would train, it would be finite, it
        #   would keep every shape -- and it would be a mechanism the reference
        #   does not have. The non-differentiability is not a defect to route
        #   around: the paper trains the two modules INDEPENDENTLY (§4.3), so
        #   nothing is meant to flow from here back into the segmenter. This
        #   class is an INFERENCE assembly.
        # * Do NOT concatenate the mask onto the image. That is the other
        #   common way to feed a mask to a downstream network, and here it
        #   would silently change the rectifier's input width from 3 to 4 --
        #   which the encoder infers from the tensor, so it builds happily.
        #
        # Guarded by `TestTheMaskIsAppliedByMultiplicationNotConcatenation`,
        # `TestTheConfidenceMapIsThresholdedNotUsedRaw` and
        # `TestNoGradientReachesTheSegmenter`. See decisions.md D-025.
        mask = keras.ops.cast(
            confidence > SEG_MASK_THRESHOLD, dtype=inputs.dtype)
        masked = mask * inputs

        # DECISION plan-2026-09-10T065432-05fcb6dd/D-026: the rectifier is
        # called with `training=False` UNCONDITIONALLY, not with this call's
        # `training`. Its flag does not select a normalization mode -- it has
        # no `BatchNormalization` and no dropout anywhere, only non-affine
        # GroupNormalization, which behaves identically either way -- it
        # selects the OUTPUT RANK: `training=True` returns the whole
        # `(B, 12, H, W, 2)` refinement sequence instead of one map. Forwarding
        # it would make this model's output rank depend on a flag Keras sets
        # implicitly inside `fit()`, on a model that CANNOT be fit (see D-025).
        # Do NOT "fix" this by forwarding the flag for symmetry. Guarded by
        # `TestTheOutputFormDoesNotDependOnTheTrainingFlag`. See decisions.md
        # D-026.
        backward_map = self.rectifier(masked, training=False)

        # DECISION plan-2026-09-10T065432-05fcb6dd/D-027: `inference.py:29`'s
        # `bm = (2 * (bm / 286.8) - 1) * 0.99`, transcribed exactly and applied
        # EXACTLY ONCE, here and nowhere else in the port. The rectifier
        # deliberately emits absolute pixel coordinates and says so; applying
        # this there as well would double-apply it.
        #
        # What 286.8 is: an empirical constant of the RELEASED DocScanner-L
        # checkpoint at its 288 training resolution. What it is NOT: a
        # calibration for THIS port. No released checkpoint is ever loaded here
        # -- `pretrained=True` raises on all three classes -- so the constant is
        # carried for FIDELITY to the reference, not because it is calibrated
        # for a from-scratch model. Anyone training this from scratch may well
        # need a different divisor, and 288 is the obvious candidate. Do NOT
        # silently "correct" it to 288 or drop the 0.99: that would be a
        # different pipeline wearing the reference's name. Change it
        # deliberately, and say so. Guarded by
        # `TestTheCalibrationIsAppliedExactlyOnce`. See decisions.md D-027.
        return (
            2.0 * (backward_map / BM_CALIBRATION_DIVISOR) - 1.0
        ) * BM_CALIBRATION_SCALE

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """One calibrated backward map, at the input resolution.

        Unlike :meth:`DocScannerRectifier.compute_output_shape` this is the
        WHOLE contract, not just the inference half: the composite's output
        form does not depend on ``training`` (D-026).

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple
        :return: ``(batch, height, width, FLOW_CHANNELS)``.
        :rtype: tuple
        """
        batch, height, width, _ = input_shape
        return (batch, height, width, FLOW_CHANNELS)

    def get_config(self) -> Dict[str, Any]:
        """Return every constructor argument needed to recreate this model.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            _COMPOSITE_SEGMENTER_KEY: dict(self.segmenter_config),
            _COMPOSITE_RECTIFIER_KEY: dict(self.rectifier_config),
        })
        return config

    @classmethod
    def from_variant(
            cls,
            variant: str,
            pretrained: bool = False,
            **kwargs: Any
    ) -> "DocScanner":
        """Create the composite from the one cited configuration.

        :param variant: A key of :attr:`MODEL_VARIANTS`. Currently only
            ``"docscanner-l"``; the row's ``description`` says why.
        :type variant: str
        :param pretrained: Must be ``False``.
        :type pretrained: bool
        :param kwargs: Constructor overrides applied on top of the variant's
            configuration. To change one stage's widths, pass a whole
            ``segmenter_config`` / ``rectifier_config`` -- they are replaced,
            not merged, because a partial merge would silently mix two
            variants' widths.
        :type kwargs: Any
        :return: The constructed, unbuilt model.
        :rtype: DocScanner
        :raises ValueError: If ``variant`` is not a known key; the message
            lists the available ones.
        :raises NotImplementedError: If ``pretrained`` is ``True``.
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown DocScanner variant '{variant}'. Available variants: "
                f"{sorted(cls.MODEL_VARIANTS.keys())}"
            )

        if pretrained:
            raise NotImplementedError(
                f"No pretrained weights are distributed for DocScanner "
                f"variant '{variant}', and BOTH stages refuse for their own, "
                f"different reasons -- neither of which this composite can "
                f"repair. The segmentation checkpoint DOES NOT EXIST IN THE "
                f"REFERENCE CHECKOUT (inference.py:103 loads "
                f"'./model_pretrained/seg.pth' and no such file is there), and "
                f"the rectification checkpoint, even if obtained, cannot be "
                f"transferred into this port because the feature encoder's "
                f"`norm1` width and every stride-2 padding differ (the D-011 "
                f"and D-012 anchors in components.py). Call "
                f"`DocScannerSegmenter.from_variant(..., pretrained=True)` or "
                f"`DocScannerRectifier.from_variant(..., pretrained=True)` for "
                f"the full statement of either. Train both stages from scratch "
                f"with `src/train/doc_scanner/` -- they train INDEPENDENTLY -- "
                f"then assemble them here."
            )

        config = dict(cls.MODEL_VARIANTS[variant])
        config.pop("description", None)
        config.update(kwargs)
        return cls(**config)


# ---------------------------------------------------------------------


def create_doc_scanner(
        variant: str = "docscanner-l",
        pretrained: bool = False,
        **kwargs: Any
) -> DocScanner:
    """Create the assembled two-stage DocScanner. The module-level entry point.

    :param variant: A key of :attr:`DocScanner.MODEL_VARIANTS`. Defaults to
        ``"docscanner-l"``.
    :type variant: str
    :param pretrained: Must be ``False``; see :meth:`DocScanner.from_variant`.
    :type pretrained: bool
    :param kwargs: Constructor overrides forwarded unchanged.
    :type kwargs: Any
    :return: The constructed, unbuilt model.
    :rtype: DocScanner
    """
    return DocScanner.from_variant(variant, pretrained=pretrained, **kwargs)


__all__: List[str] = [
    "DocScanner",
    "DocScannerRectifier",
    "DocScannerSegmenter",
    "create_doc_scanner",
    "create_doc_scanner_rectifier",
    "create_doc_scanner_segmenter",
]
