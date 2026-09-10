"""DocScanner's model classes. At this step: the rectifier.

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
      ``model.py:31-99`` (this class), ``model.py:45-51``
      (``initialize_flow``), ``inference.py:28-29`` (the call site).
    - Feng et al., 2021. DocScanner: Robust Document Image Rectification with
      Progressive Learning. (https://arxiv.org/abs/2110.14968), v2.
    - Teed & Deng, 2020. RAFT: Recurrent All-Pairs Field Transforms for Optical
      Flow. ECCV 2020. (https://arxiv.org/abs/2003.12039).
"""

import keras
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.utils.model_build import materialize_sublayers

from .components import (
    FLOW_CHANNELS,
    REFINE_ITERATIONS,
    SPATIAL_DIVISOR,
    DocScannerFeatureEncoder,
    DocScannerUpdateBlock,
    _VARIANT_SPEC,
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
# * nothing else; every remaining key IS a constructor argument, which is what
#   makes `from_variant` a splat rather than a translation table.
# ---------------------------------------------------------------------

#: Spec keys that are not rectifier constructor arguments.
_SPEC_KEYS_NOT_CONSTRUCTOR_ARGS: Tuple[str, ...] = (
    "mask_head_output_channels",
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


__all__: List[str] = [
    "DocScannerRectifier",
    "create_doc_scanner_rectifier",
]
