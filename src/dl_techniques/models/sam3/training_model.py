"""``Sam3TrainingModel`` -- the packed-tensor training wrapper for SAM 3.

Why a wrapper exists at all: TARGET PACKING, and nothing else
--------------------------------------------------------------
SAM 1 needed a separate ``SAMTrainingModel`` because ``SAM.call``'s
``postprocess_masks`` runs ``ops.image.resize``, which raises under ``fit()``'s
graph mode. **That reason has NO analogue here and is deliberately not reused.**
It was CHECKED, not assumed: a plain ``keras.Model`` wrapping :class:`Sam3Image`
at ``jit_compile=False`` completed a real ``fit()`` step (loss ``0.0467``), so
``Sam3Image.call`` traces fine. Carrying SAM 1's forcing reason forward would be
a GHOST constraint, and the ghost is recorded here so nobody re-derives it and
then over-scopes this wrapper into driving submodules directly the way SAM 1's
must.

The real -- and only -- reason is that the SAM 3 detection loss is **JOINT**:
one Hungarian assignment is shared across the classification, box, presence and
mask terms, so all four have to be seen by ONE
:class:`~dl_techniques.losses.sam3_detection_loss.Sam3DetectionLoss` object. A
single ``Loss`` object handed a dict ``y_pred`` breaks:
``CompileLoss.build`` broadcasts that one object across every leaf of the
structure via ``tree.map_structure`` and then ``KeyError``s. Splitting
supervision into per-output-key dict losses would compute a different (or no)
assignment per term, which is exactly the property the joint matcher exists to
provide. The sanctioned precedent is DINO's D-024: **emit ONE packed tensor and
let one ``Loss`` split it.** That is all this wrapper does.

There is no custom ``train_step`` here, and there must never be one.

The layout is NOT defined here
------------------------------
``losses/sam3_detection_loss.py`` is the layout's single home: the ``PACKED_*``
and ``META_*`` channel constants, :func:`packed_channel_count`,
``unpack_predictions``, ``unpack_targets`` and ``derive_keep_loss`` all live
there and are IMPORTED below. Nothing in this module re-spells a channel index
or re-derives ``C``; the pack functions here place their fields BY the imported
constants (see :func:`_pack_rows`), so a layout change moves one file. A count
restated in two places is a hand-maintained lockstep invariant, i.e. a latent
defect, and the pack/unpack pair is pinned value-exactly by
``tests/test_models/test_sam3/test_training_model.py``.

Note that the meta row's channel ``2`` is ``is_exhaustive``, not the reserved
zero the plan's prose described; see decisions.md D-010. It is imported as
``META_IS_EXHAUSTIVE`` rather than treated as spare.

Importing the TensorFlow package is FORBIDDEN in this file
-----------------------------------------------------------
This module lives under ``models/sam3/``, whose ``keras.ops`` purity is a
close-out gate checked by GREP -- so the two literal tokens that gate greps for
are deliberately NOT spelled anywhere in this file, not even in prose. (Writing
one of them here erodes the instrument, a failure already measured three times
in this repository; it was measured a fourth time on this very docstring, which
is why it now reads the way it does.) The packing below is pure ``keras.ops``.
The loss module's own TensorFlow dependency is sanctioned because it is
training-only and never traced in a forward path; this file inherits no such
exemption and takes none.

``jit_compile=False`` is MANDATORY, and has ONE home
-----------------------------------------------------
:func:`compile_sam3_trainer` is the single compile site, and it sets
``jit_compile=False`` by ``setdefault`` so the invariant holds by construction
rather than by remembering. The constraint is doubly forced: this model family
already pins it, and the matcher crosses an eager ``py_function`` boundary for
which no ``EagerPyFunc`` XLA kernel exists, so ``jit_compile=True`` fails hard.

``training=`` is forwarded EXPLICITLY at every call site
---------------------------------------------------------
``training=None`` is NOT inference at a non-zero ``drop_path_rate``: this
repository's shared ``StochasticDepth`` short-circuits on ``training is False``
ONLY, so the ``None`` a plain ``model(inputs)`` passes down DROPS PATHS
(D-123). :meth:`Sam3TrainingModel.call` therefore threads its own ``training``
argument into :class:`Sam3Image` explicitly, and the ``.keras`` round-trip test
compares values at ``training=False`` -- at ``training=None`` a correct round
trip measures deltas of 0.2-2.2 that look exactly like reinitialized weights.
"""

from typing import Any, Dict, Optional, Tuple

import keras
from keras import ops

from ...losses.sam3_detection_loss import (
    META_IS_EXHAUSTIVE,
    META_KEEP_LOSS,
    META_NUM_BOXES,
    PACKED_BOX_START,
    PACKED_MASK_START,
    PACKED_SCORE_CHANNEL,
    Sam3DetectionLoss,
    derive_keep_loss,
    packed_channel_count,
)
from ...utils.logger import logger
from .sam3_image import Sam3Image

# ---------------------------------------------------------------------
# Widths DERIVED from the imported constants -- never restated. `_BOX_WIDTH`
# is 4 and `_META_WIDTH` is 3 today; both follow the layout automatically.
# ---------------------------------------------------------------------
_BOX_WIDTH: int = PACKED_MASK_START - PACKED_BOX_START
_META_WIDTH: int = max(META_KEEP_LOSS, META_NUM_BOXES, META_IS_EXHAUSTIVE) + 1


def _zero_block(reference: Any, width: int) -> Any:
    """Return a zero tensor shaped like ``reference`` with a trailing ``width``.

    Interface contract: ``reference`` is any float tensor; the return has
    ``reference``'s shape plus one trailing axis of length ``width``, is filled
    with zeros of ``reference``'s dtype, and carries ``reference``'s DYNAMIC
    batch axis (which is why this is built by broadcast rather than by
    ``ops.zeros``, whose shape must be static). Never raises.

    :param reference: Tensor whose shape and dtype are copied.
    :type reference: Any
    :param width: Length of the trailing axis.
    :type width: int
    :return: A zero tensor of shape ``reference.shape + (width,)``.
    :rtype: Any
    """
    zero = ops.expand_dims(ops.zeros_like(reference), axis=-1)
    repeats = (1,) * len(zero.shape[:-1]) + (width,)
    return ops.tile(zero, repeats)


def _pack_rows(score: Any, boxes: Optional[Any], masks: Optional[Any],
               mask_size: int) -> Any:
    """Assemble ``(B, R, C)`` rows by PLACING fields at the layout's channels.

    Interface contract: ``score`` is ``(B, R)``, ``boxes`` is ``(B, R, 4)`` or
    ``None`` (zero-filled), ``masks`` is ``(B, R, mask_size)`` or ``None``
    (zero-filled); the return is ``(B, R, packed_channel_count(mask_size))``.
    Never raises -- a wrongly shaped input fails in ``ops`` rather than here.

    Every field is written to ``columns[<IMPORTED CONSTANT>]``, so this function
    contains no channel literal at all: it is correct for any assignment of the
    ``PACKED_*`` constants, not just today's ``(0, 1, 5)``.

    :param score: Per-row scalar channel -- class logit, presence logit or the
        GT validity flag depending on which side is being packed.
    :type score: Any
    :param boxes: ``cxcywh`` boxes, or ``None`` to zero-fill the box block.
    :type boxes: Optional[Any]
    :param masks: Flattened masks, or ``None`` to zero-fill the mask block.
    :type masks: Optional[Any]
    :param mask_size: Flattened mask length; ``0`` when masks are off.
    :type mask_size: int
    :return: The packed rows.
    :rtype: Any
    """
    # DECISION plan-2026-08-05T124709-6c4fac48/D-015
    # This is a column SCATTER on purpose. Do NOT "simplify" it to the obvious
    # `ops.concatenate([score[..., None], boxes, masks], axis=-1)`: that
    # spelling encodes the field ORDER as source-code order, which is a channel
    # index restated in a second place with nothing keeping it in step with the
    # `PACKED_*` constants it is supposed to follow. The scatter below contains
    # no channel literal at all and stays correct under any reassignment of
    # those constants. It costs nothing measurable -- the stack is over
    # `PACKED_MASK_START` (5) columns and the mask block is concatenated whole,
    # never scattered. See decisions.md D-015.
    zero = ops.zeros_like(score)
    columns = [zero] * PACKED_MASK_START
    columns[PACKED_SCORE_CHANNEL] = score
    if boxes is not None:
        for offset in range(_BOX_WIDTH):
            columns[PACKED_BOX_START + offset] = boxes[..., offset]
    packed = ops.stack(columns, axis=-1)
    if mask_size > 0:
        block = masks if masks is not None else _zero_block(score, mask_size)
        packed = ops.concatenate([packed, block], axis=-1)
    return packed


def _pack_meta_row(keep_loss: Any, num_boxes: Any, is_exhaustive: Any,
                   channels: int) -> Any:
    """Assemble the target tensor's LAST row, ``(B, 1, C)``.

    Interface contract: the three inputs are ``(B,)`` float tensors; the return
    is ``(B, 1, channels)`` with each field at its imported ``META_*`` channel
    and every other channel exactly ``0.0``. Never raises.

    :param keep_loss: Per-image presence target.
    :type keep_loss: Any
    :param num_boxes: Per-image GT box count.
    :type num_boxes: Any
    :param is_exhaustive: Per-image exhaustive-annotation flag (read only on
        the ``weak_loss=True`` path; decisions.md D-010).
    :type is_exhaustive: Any
    :param channels: The packed channel width ``C``.
    :type channels: int
    :return: The meta row.
    :rtype: Any
    """
    zero = ops.zeros_like(keep_loss)
    columns = [zero] * _META_WIDTH
    columns[META_KEEP_LOSS] = keep_loss
    columns[META_NUM_BOXES] = num_boxes
    columns[META_IS_EXHAUSTIVE] = is_exhaustive
    row = ops.stack(columns, axis=-1)
    if channels > _META_WIDTH:
        row = ops.concatenate(
            [row, _zero_block(keep_loss, channels - _META_WIDTH)], axis=-1)
    return ops.expand_dims(row, axis=1)


def pack_predictions(outputs: Dict[str, Any],
                     include_masks: bool = False) -> Any:
    """Pack :class:`Sam3Image`'s five-key output dict into ONE tensor.

    Interface contract: ``outputs`` is exactly what ``Sam3Image.call`` returns
    (``pred_logits`` ``(B, Q, 1)``, ``pred_boxes`` ``(B, Q, 4)``,
    ``pred_masks`` ``(B, Q, H, W)``, ``presence_logit`` ``(B, 1)``,
    ``semantic_seg`` -- the last DELIBERATELY unpacked, phase 2 leaves it
    unsupervised). The return is ``(B, Q + 1, C)``: rows ``0..Q-1`` are the
    queries, row ``Q`` is the presence row whose non-score channels are exactly
    zero. This is the exact inverse of ``unpack_predictions``, pinned
    value-exactly by test. Raises only on a missing required key.

    :param outputs: ``Sam3Image``'s output dict.
    :type outputs: Dict[str, Any]
    :param include_masks: Whether to pack the flattened mask block.
    :type include_masks: bool
    :return: The packed prediction tensor.
    :rtype: Any
    :raises KeyError: If a required output key is absent.
    """
    logits = outputs["pred_logits"]
    boxes = outputs["pred_boxes"]
    presence = outputs["presence_logit"]

    masks = None
    mask_size = 0
    if include_masks:
        raw = outputs["pred_masks"]
        height, width = int(raw.shape[-2]), int(raw.shape[-1])
        mask_size = height * width
        masks = ops.reshape(
            raw, (-1, int(raw.shape[1]), mask_size))

    queries = _pack_rows(
        logits[..., PACKED_SCORE_CHANNEL], boxes, masks, mask_size)
    presence_row = _pack_rows(presence, None, None, mask_size)
    return ops.concatenate([queries, presence_row], axis=1)


def pack_targets(target_boxes: Any,
                 target_valid: Any,
                 target_masks: Optional[Any] = None,
                 num_boxes: Optional[Any] = None,
                 is_exhaustive: Optional[Any] = None,
                 include_masks: bool = False) -> Any:
    """Pack padded ground truth into ONE tensor the detection loss consumes.

    Interface contract: ``target_boxes`` is ``(B, N_max, 4)`` in normalized
    ``cxcywh``, ``target_valid`` is ``(B, N_max)`` (``1`` real, ``0`` padding),
    ``target_masks`` is ``(B, N_max, H, W)`` or ``(B, N_max, P)``. The return is
    ``(B, N_max + 1, C)`` whose last row is the meta row. ``num_boxes`` defaults
    to the per-image count of valid rows and ``is_exhaustive`` to all-ones;
    ``keep_loss`` is ALWAYS derived by the loss module's ``derive_keep_loss``,
    so the presence target has one formula in the repository, not two.

    The **zero-GT image is an ordinary member of this layout**, not an edge
    case: it is simply an image whose validity column is all zero, which makes
    ``keep_loss`` ``0.0`` and ``num_boxes`` ``0`` (the loss clamps the batch
    divisor to a minimum of 1). ``N_max > Q`` is likewise ordinary -- the
    matcher assigns ``min(Q, N)`` pairs and the surplus GT contributes nothing.

    :param target_boxes: ``(B, N_max, 4)`` padded GT boxes, ``cxcywh``.
    :type target_boxes: Any
    :param target_valid: ``(B, N_max)`` validity flags.
    :type target_valid: Any
    :param target_masks: Padded GT masks, required iff ``include_masks``.
    :type target_masks: Optional[Any]
    :param num_boxes: ``(B,)`` per-image GT count; defaults to the valid count.
    :type num_boxes: Optional[Any]
    :param is_exhaustive: ``(B,)`` exhaustive-annotation flag; defaults to 1.
    :type is_exhaustive: Optional[Any]
    :param include_masks: Whether the layout carries a mask block.
    :type include_masks: bool
    :return: The packed target tensor.
    :rtype: Any
    :raises ValueError: If ``include_masks`` disagrees with whether
        ``target_masks`` was supplied. This is the ONE leg of the three-way
        width contract that can be caught at pack time, and it is caught loudly
        rather than left to slice garbage.
    """
    if include_masks and target_masks is None:
        raise ValueError(
            "pack_targets: include_masks=True requires target_masks. Packing "
            "without them would emit C=5 while the loss slices C=5+P, which "
            "silently mis-slices rather than raising.")
    if not include_masks and target_masks is not None:
        raise ValueError(
            "pack_targets: target_masks was supplied but include_masks=False, "
            "so they would be dropped silently. Set include_masks=True or "
            "stop passing masks.")

    boxes = ops.cast(target_boxes, "float32")
    valid = ops.cast(target_valid, "float32")

    masks = None
    mask_size = 0
    if include_masks:
        masks = ops.cast(target_masks, "float32")
        if len(masks.shape) == 4:
            mask_size = int(masks.shape[-2]) * int(masks.shape[-1])
            masks = ops.reshape(masks, (-1, int(masks.shape[1]), mask_size))
        else:
            mask_size = int(masks.shape[-1])

    if num_boxes is None:
        counts = ops.sum(ops.cast(valid > 0.0, "float32"), axis=-1)
    else:
        counts = ops.cast(num_boxes, "float32")
    if is_exhaustive is None:
        exhaustive = ops.ones_like(counts)
    else:
        exhaustive = ops.cast(is_exhaustive, "float32")

    keep = ops.squeeze(derive_keep_loss(boxes, valid), axis=-1)
    rows = _pack_rows(valid, boxes, masks, mask_size)
    meta = _pack_meta_row(
        keep, counts, exhaustive, packed_channel_count(mask_size))
    return ops.concatenate([rows, meta], axis=1)


@keras.saving.register_keras_serializable()
class Sam3TrainingModel(keras.Model):
    """A :class:`Sam3Image` whose output is ONE packed supervision tensor.

    Interface contract: ``call(inputs, training=...)`` takes exactly the input
    dict :class:`Sam3Image` takes (``image`` / ``token_ids`` /
    optional ``token_padding_mask``) and returns a single ``(B, Q + 1, C)``
    tensor in the packed layout, ready for a single
    :class:`~dl_techniques.losses.sam3_detection_loss.Sam3DetectionLoss` under
    stock ``fit()``. It adds no parameters of its own -- every trainable
    variable belongs to the wrapped model -- and it neither reduces nor rescales
    anything. It raises only in the constructor.

    ``include_masks`` is a THREE-WAY contract, and two of the three legs raise
    ---------------------------------------------------------------------------
    The training model, the loss and the data pipeline must agree on the packed
    channel width, because a silent width mismatch SLICES GARBAGE rather than
    raising -- ``unpack_*`` are pure slices with no validation, by design.
    Enforcement, in order of strength:

    1. :func:`compile_sam3_trainer` **raises** if the loss it is handed
       disagrees with ``model.include_masks``. That is the model-vs-loss leg,
       caught at compile time.
    2. :func:`pack_targets` **raises** if ``include_masks`` disagrees with
       whether masks were actually supplied. That is the pipeline-vs-itself
       leg, caught at pack time.
    3. The pipeline-vs-model leg has no construction-time hook (the pipeline
       does not hold a model reference), so it is pinned by a test that asserts
       all three land on the same :func:`packed_channel_count`. This module
       exposes :attr:`packed_channels` and :meth:`packed_target_spec` precisely
       so the pipeline can DERIVE that width instead of restating it.

    :param sam3: The wrapped model, or its serialized config dict.
    :type sam3: Any
    :param include_masks: Whether the packed layout carries the flattened mask
        block. Default ``False``, matching
        :class:`Sam3DetectionLoss`'s own default and the reference's one
        shipped training config (decisions.md D-009).
    :type include_masks: bool
    :param kwargs: Forwarded to :class:`keras.Model`.
    :raises ValueError: If ``sam3`` is neither a :class:`Sam3Image` nor a
        deserializable config.
    """

    def __init__(self, sam3: Any, include_masks: bool = False,
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if isinstance(sam3, dict):
            sam3 = keras.saving.deserialize_keras_object(sam3)
        if not isinstance(sam3, Sam3Image):
            raise ValueError(
                "Sam3TrainingModel: `sam3` must be a Sam3Image (or its "
                f"serialized config), got {type(sam3).__name__}.")
        self.sam3 = sam3
        self.include_masks = bool(include_masks)

        shapes = self.sam3.compute_output_shape()
        self.num_queries = int(shapes["pred_logits"][1])
        self.mask_grid: Tuple[int, int] = (
            int(shapes["pred_masks"][2]), int(shapes["pred_masks"][3]))
        #: Flattened mask length ``P``, or ``0`` when masks are off.
        self.mask_size = (self.mask_grid[0] * self.mask_grid[1]
                          if self.include_masks else 0)
        #: The packed channel width ``C``. Derived, never restated.
        self.packed_channels = packed_channel_count(self.mask_size)

        logger.info(
            "Sam3TrainingModel: Q=%d, masks=%s, packed width C=%d",
            self.num_queries, self.include_masks, self.packed_channels)

    # -----------------------------------------------------------------
    # build / forward
    # -----------------------------------------------------------------

    def build(self, input_shape: Optional[Any] = None) -> None:
        """Build the wrapped model explicitly.

        :param input_shape: Forwarded to :meth:`Sam3Image.build`, which ignores
            it and derives every shape from its own components.
        :type input_shape: Optional[Any]
        """
        if self.built:
            return
        if not self.sam3.built:
            self.sam3.build(input_shape)
        super().build(input_shape)

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build before Keras restores weights, so every variable exists.

        A subclassed model whose sub-layers build lazily restores an INCOMPLETE
        weight set from a ``.keras`` file with no exception and no shape
        symptom, which is why this mirrors :class:`Sam3Image`'s own hook.

        :param config: Ignored; every shape comes from stored configuration.
        :type config: Dict[str, Any]
        """
        del config
        if not self.built:
            self.build(None)

    def call(self, inputs: Dict[str, Any],
             training: Optional[bool] = None) -> Any:
        """Run SAM 3 and pack its outputs into one supervision tensor.

        :param inputs: ``{'image': ..., 'token_ids': ...}`` plus the optional
            ``'token_padding_mask'`` -- exactly :meth:`Sam3Image.call`'s
            contract.
        :type inputs: Dict[str, Any]
        :param training: Keras training flag. Forwarded EXPLICITLY: at a
            non-zero ``drop_path_rate`` the default ``None`` drops paths and is
            NOT inference (D-123).
        :type training: Optional[bool]
        :return: ``(B, Q + 1, C)`` packed predictions.
        :rtype: Any
        """
        outputs = self.sam3(inputs, training=training)
        return pack_predictions(outputs, include_masks=self.include_masks)

    # -----------------------------------------------------------------
    # shapes / config
    # -----------------------------------------------------------------

    def packed_target_spec(self, max_instances: int) -> Tuple[int, int]:
        """Return the ``(N_max + 1, C)`` shape a packed target must have.

        The data pipeline calls this instead of computing ``C`` itself, which
        is what makes the three-way width contract derivable rather than
        hand-maintained.

        :param max_instances: ``N_max``, the padded GT slot count.
        :type max_instances: int
        :return: The per-sample packed target shape.
        :rtype: Tuple[int, int]
        """
        return (int(max_instances) + 1, self.packed_channels)

    def compute_output_shape(
            self, input_shape: Optional[Any] = None
    ) -> Tuple[Optional[int], int, int]:
        """Return the packed prediction shape, from stored config only.

        :param input_shape: Ignored; the batch axis is reported as ``None``.
        :type input_shape: Optional[Any]
        :return: ``(None, Q + 1, C)``.
        :rtype: Tuple[Optional[int], int, int]
        """
        del input_shape
        return (None, self.num_queries + 1, self.packed_channels)

    def get_config(self) -> Dict[str, Any]:
        """Serialize the wrapper, including the whole wrapped SAM 3.

        :return: Configuration consumable by :meth:`from_config`.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "sam3": keras.saving.serialize_keras_object(self.sam3),
            "include_masks": self.include_masks,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Sam3TrainingModel":
        """Rebuild a wrapper (and its SAM 3) from :meth:`get_config` output.

        :param config: A dict produced by :meth:`get_config`.
        :type config: Dict[str, Any]
        :return: The reconstructed wrapper.
        :rtype: Sam3TrainingModel
        """
        config = dict(config)
        config["sam3"] = keras.saving.deserialize_keras_object(config["sam3"])
        return cls(**config)


def compile_sam3_trainer(model: Sam3TrainingModel,
                         optimizer: Any = "adam",
                         loss: Optional[Sam3DetectionLoss] = None,
                         **compile_kwargs: Any) -> None:
    """Compile ``model`` with the joint SAM 3 detection loss. ONE compile site.

    This function exists so that two invariants hold BY CONSTRUCTION rather than
    by anyone remembering them:

    * ``jit_compile=False``. Keras 3.8's ``fit()`` defaults to
      ``jit_compile='auto'``, which selects XLA on a GPU, and the matcher is a
      eager ``py_function`` boundary for which no ``EagerPyFunc`` XLA kernel
      exists -- so XLA does not degrade, it fails hard at the first step.
    * ``model.include_masks == loss.include_masks``. A disagreement here does
      NOT raise anywhere downstream: ``unpack_*`` are pure slices, so a
      too-narrow tensor is sliced to garbage and trains on it. This is the
      model-vs-loss leg of the three-way width contract, and it is checked here
      because this is the one place that sees both objects.

    :param model: The wrapper to compile.
    :type model: Sam3TrainingModel
    :param optimizer: Any Keras optimizer or its string name.
    :type optimizer: Any
    :param loss: The joint loss. ``None`` constructs a
        :class:`Sam3DetectionLoss` agreeing with ``model.include_masks``, which
        is the failure-proof path.
    :type loss: Optional[Sam3DetectionLoss]
    :param compile_kwargs: Forwarded to ``keras.Model.compile``. Passing
        ``jit_compile`` here overrides the mandatory ``False`` and WILL make
        the first ``fit()`` step raise.
    :type compile_kwargs: Any
    :raises ValueError: If ``loss.include_masks`` disagrees with the model's.
    """
    # DECISION plan-2026-08-05T124709-6c4fac48/D-015
    # ONE compile site, and the width check lives here. Do NOT inline
    # `model.compile(...)` in a trainer or a test: `jit_compile` would then be
    # `'auto'` by default (XLA on a GPU, which the matcher's eager
    # `py_function` boundary cannot run) and the model-vs-loss `include_masks`
    # agreement would have no place to be checked at all --
    # `unpack_predictions` / `unpack_targets` are
    # pure slices with no validation, so a width mismatch trains on garbage
    # instead of raising. Do NOT "helpfully" coerce a mismatched loss to the
    # model's flag either: silently rewriting a caller's explicit configuration
    # hides the pipeline's own disagreement, which is the leg this check cannot
    # see. See decisions.md D-015.
    if loss is None:
        loss = Sam3DetectionLoss(include_masks=model.include_masks)
    elif bool(getattr(loss, "include_masks", False)) != model.include_masks:
        expected = ("{}+P".format(packed_channel_count(0))
                    if loss.include_masks else str(packed_channel_count(0)))
        raise ValueError(
            "compile_sam3_trainer: include_masks disagrees between the "
            f"training model ({model.include_masks}, packing C="
            f"{model.packed_channels}) and the loss ({loss.include_masks}, "
            f"slicing C={expected}). A width mismatch slices garbage rather "
            "than raising.")
    compile_kwargs.setdefault("jit_compile", False)
    model.compile(optimizer=optimizer, loss=loss, **compile_kwargs)


__all__ = [
    "Sam3TrainingModel",
    "compile_sam3_trainer",
    "pack_predictions",
    "pack_targets",
]
