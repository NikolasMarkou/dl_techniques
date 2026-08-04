"""Ground-truth-gated mask supervision for the SAM 2 video trainer.

What upstream does, and why a second SAM-loss home exists
---------------------------------------------------------
Upstream SAM 2 multiplies its mask, dice and IoU terms by
``target_obj = any(target_masks > 0)`` -- a per-frame flag derived from the
GROUND TRUTH, never from the model's own ``object_score_logits``
(``training/loss_fns.py:219-297``). A fully-occluded frame therefore
contributes exactly zero mask gradient while its object-score BCE term stays
live and pushes the score negative.

:class:`SAM2GatedMaskLoss` is that gate, and it lives in a NEW module rather
than as an option on :class:`~dl_techniques.losses.sam_mask_loss.SAMMaskLoss`
because ``sam_mask_loss.py`` is protected by SAM 1's 357-test gate and must
stay byte-unchanged. Everything that module owns -- ``match_mask_axis``,
``to_dice_layout``, ``to_focal_layout``, the two ``SegmentationWrapperLoss``
instances and their layout refusals -- is INHERITED or imported here, never
re-derived: H-17 records that a 1-channel focal is bit-identically blind to
negatives and that raw-layout dice reduces ``(num_masks, height)`` instead of
``(height, width)`` and returns a plausible wrong number.

Why the gate zeroes PROBABILITIES, and what it costs
----------------------------------------------------
The gate replaces the predicted probabilities of an absent row with zeros
through ``ops.where``, which passes **exactly zero** gradient to the suppressed
branch -- the same semantics as upstream's ``loss_mask * target_obj``. On such
a row:

* **dice becomes exactly 0**: ``1 - (0 + s) / (0 + 0 + s) = 0``, with no
  residual at all. MEASURED ``0.0`` at three shapes.
* **focal does NOT become exactly 0**. The row degenerates to the
  perfectly-predicted all-background case, and ``SegmentationLosses.focal_loss``
  clips its probabilities to ``[1e-7, 1 - 1e-7]`` before taking the log, so the
  background channel still contributes
  ``alpha * (1 - p)^gamma * -log(p)`` at ``p = 1 - 1e-7``.

That residual is **MEASURED, not assumed** (plan assumption A-6, which predicted
"negligible but not zero" and is confirmed). On this stack, float32, at
``(1,1,16,16)``, ``(2,3,16,16)`` and ``(2,4,64,64)`` alike -- it is a per-pixel
constant, so it does not vary with the mask grid::

    focal residual per fully-absent batch : 4.235165241142481e-22
    dice  residual per fully-absent batch : 0.0
    combined at focal_weight=20           : 8.4703304822849621e-21

Against a representative present-row loss of ``3.834`` that is ``2.2e-21``
relative -- fifteen orders of magnitude inside the plan's own
``> 1e-6``-of-the-ungated-term stop trigger, and far below float32's own
accumulation quantum at a loss of order 1 (``~6e-8``). :data:`ABSENT_ROW_FOCAL_RESIDUAL`
carries the measured number so a guard bound can be DERIVED from it rather than
guessed, and so a future change to ``SegmentationLosses``' clipping is loud.

What is deliberately NOT here
-----------------------------
* **No object-score loss class.** Upstream's ``loss_class`` is
  ``sigmoid_focal_loss(..., focal_gamma_obj_score=0.0,
  focal_alpha_obj_score=-1.0)`` at weight 1 -- i.e. a plain BCE. It is stock
  ``keras.losses.BinaryCrossentropy(from_logits=True)``; a bespoke class with
  one call site and no behaviour of its own would be an unearned abstraction.
* **No IoU loss class.** ``SAMIoULoss`` is reused unchanged; the MODEL zeroes
  both channels of its ``iou_supervision`` output on absent rows, which makes
  both the squared error and its gradient exactly 0 under a mean reduction.

Both of those choices, and the ``compile()`` dict that assembles all three, live
in ``models/sam2/training_model.py::compile_sam2_video_trainer`` -- the site a
reader assembling a training run actually reaches. See ``decisions.md`` D-052.
"""

from typing import Any, Dict

import keras
from keras import ops

from .sam_mask_loss import (
    DICE_CHANNELS,
    FOCAL_CHANNELS,
    SAMMaskLoss,
    _require_channels_last,
    _static_mask_stack_shape,
    match_mask_axis,
    to_dice_layout,
    to_focal_layout,
)

#: MEASURED focal contribution of one fully-gated (all-zero GT, all-zero
#: probability) mask row, float32, on this Keras 3.8 / TF 2.18 stack. It is a
#: per-pixel constant of ``SegmentationLosses.focal_loss``'s ``1e-7`` clip and
#: is therefore INDEPENDENT of the mask grid and of the batch: the same value
#: was measured at ``(1, 1, 16, 16)``, ``(2, 3, 16, 16)`` and ``(2, 4, 64, 64)``.
#: The dice residual is exactly ``0.0`` and needs no constant.
ABSENT_ROW_FOCAL_RESIDUAL = 4.235165241142481e-22


def mask_presence_gate(masks: Any) -> Any:
    """Derive per-row object presence from a ``(B, M, h, w)`` mask stack.

    **This is the single home of "presence".** Upstream computes
    ``target_obj = any(target_masks[:, 0] > 0)`` and gates its mask, dice and
    IoU terms with it; this repository needs the same flag in two places -- the
    loss (from its ``y_true``) and :class:`SAM2TrainingModel` (from its
    ``gt_masks`` input, for the IoU supervision it packs). A second derivation
    would be a second definition of occlusion that can disagree with the first
    with no shape, dtype or finiteness symptom, so both call sites import this.

    Interface contract: the argument is a rank-4 binary or probability mask
    stack; the return is a **boolean** tensor of shape ``(B, M, 1, 1)``, ready
    to broadcast against the stack it came from. It never raises -- validation
    of the rank belongs to the caller's own layout guard, which runs first.

    :param masks: ``(B, M, h, w)`` ground-truth masks. Values above ``0`` mark
        foreground; the threshold is ``> 0`` rather than ``>= 0.5`` because
        that is what upstream uses and because a downsampled binary mask is
        exactly ``0`` or exactly ``1``.
    :type masks: Any
    :return: ``(B, M, 1, 1)`` bool, ``True`` where that row has any foreground.
    :rtype: Any
    """
    return ops.max(masks, axis=(-2, -1), keepdims=True) > 0.0


@keras.saving.register_keras_serializable()
class SAM2GatedMaskLoss(SAMMaskLoss):
    """:class:`SAMMaskLoss` with upstream's ground-truth presence gate.

    Rows whose ground truth is entirely empty contribute exactly zero gradient
    and (up to :data:`ABSENT_ROW_FOCAL_RESIDUAL`) exactly zero loss. Every other
    row is scored exactly as :class:`SAMMaskLoss` scores it -- the gate is the
    only difference, and it is applied to the PROBABILITIES, after the sigmoid
    and before either layout adapter.

    :param kwargs: Everything :class:`SAMMaskLoss` accepts
        (``focal_weight``, ``dice_weight``, ``from_logits``, ``config``,
        ``name``, ``reduction``).

    Call args:

    * ``y_true``: binary ground-truth mask stack ``(B, M, h, w)``. For the
      video trainer the frame axis IS the mask axis (D-051), so ``M == T``.
    * ``y_pred``: predicted mask logits (or probabilities) ``(B, M, h, w)``.

    :raises ValueError: if either tensor is not a rank-4 mask stack with static
        spatial extents, via ``sam_mask_loss``'s own guards.

    .. warning::

       Do NOT gate on ``y_pred`` -- on the model's own predicted mask or its
       ``object_score_logits``. That gate is self-fulfilling: a head that
       predicts "absent" switches off the very loss that would correct it, and
       the resulting model trains to a constant "nothing is here" with a
       falling loss and no other symptom. The gate is GROUND TRUTH, always.

    Example:

    .. code-block:: python

        loss = SAM2GatedMaskLoss()
        model.compile(loss={"low_res_logits": loss, ...}, jit_compile=False)
    """

    def __init__(self, name: str = "sam2_gated_mask_loss", **kwargs: Any) -> None:
        super().__init__(name=name, **kwargs)

    def call(self, y_true: Any, y_pred: Any) -> Any:
        """Compute the gated ``focal_weight * focal + dice_weight * dice``.

        :param y_true: ``(B, M, h, w)`` binary ground truth.
        :type y_true: Any
        :param y_pred: ``(B, M, h, w)`` mask logits or probabilities.
        :type y_pred: Any
        :return: Scalar loss.
        :rtype: Any
        """
        pred_shape = _static_mask_stack_shape(y_pred, "SAM2GatedMaskLoss(y_pred)")
        true_shape = _static_mask_stack_shape(y_true, "SAM2GatedMaskLoss(y_true)")

        probabilities = ops.sigmoid(y_pred) if self.from_logits else y_pred
        truth = ops.cast(y_true, probabilities.dtype)
        truth = match_mask_axis(truth, true_shape, pred_shape)

        # DECISION plan-2026-08-04T044628-4c240b4c/D-052
        # `ops.where`, and it gates the PROBABILITIES -- not the logits, and not
        # the loss afterwards. Three tempting variants are all wrong:
        #   * gating the LOGITS with 0.0 gives sigmoid(0) = 0.5, i.e. a
        #     maximally-uncertain prediction scored against an empty mask --
        #     the largest possible dice term, not the smallest;
        #   * gating them with a large negative constant is an APPROXIMATION
        #     with a guessed epsilon, where `ops.where` on the probabilities is
        #     exact (dice becomes 0 identically);
        #   * multiplying by a float gate zeroes this row's gradient too, but
        #     it also multiplies the PRESENT rows by 1.0, which is only
        #     gradient-neutral because 1.0 happens to be the identity -- it
        #     stops being so the moment anyone makes the gate soft.
        # `ops.where` passes exactly zero gradient to the suppressed branch,
        # which is precisely upstream's `loss_mask * target_obj`. The gate is
        # derived from `truth`, i.e. from GROUND TRUTH; deriving it from
        # `probabilities` would be self-fulfilling. See decisions.md D-052.
        present = mask_presence_gate(truth)
        gated = ops.where(present, probabilities, ops.zeros_like(probabilities))

        # The adapters and their refusals are `sam_mask_loss`'s, unchanged: a
        # 1-channel focal is bit-identically blind to negatives and raw-layout
        # dice reduces the wrong axes (H-17). This class adds a gate, not a
        # second opinion about layout.
        dice_pred = to_dice_layout(gated)
        dice_true = to_dice_layout(truth)
        _require_channels_last(dice_pred, DICE_CHANNELS, "SAM2GatedMaskLoss dice path")

        focal_pred = to_focal_layout(gated)
        focal_true = to_focal_layout(truth)
        _require_channels_last(
            focal_pred, FOCAL_CHANNELS, "SAM2GatedMaskLoss focal path")

        focal_term = self._focal(focal_true, focal_pred)
        dice_term = self._dice(dice_true, dice_pred)
        return self.focal_weight * focal_term + self.dice_weight * dice_term

    def get_config(self) -> Dict[str, Any]:
        """Full serialization config, identical in shape to the parent's.

        :return: Configuration dict consumable by
            :meth:`SAMMaskLoss.from_config`.
        :rtype: Dict[str, Any]
        """
        return super().get_config()
