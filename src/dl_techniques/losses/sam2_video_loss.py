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

The focal term is computed FROM LOGITS, and that is not a style choice
----------------------------------------------------------------------
``SegmentationLosses.focal_loss`` works in PROBABILITY space and opens with
``y_pred = ops.clip(y_pred, 1e-7, 1 - 1e-7)`` (``segmentation_loss.py:255``).
``ops.clip`` has an **exactly zero** derivative outside its range, so every
pixel whose sigmoid saturates past ``1e-7`` receives exactly zero focal gradient
-- permanently, and in both directions. That is not a corner case here: it is
what killed the first SAM 2 video training run.

MEASURED on the 60-epoch toy checkpoint (``results/sam2_toy_overfit``), one
batch, ``training=False``, per-loss-term gradient norms over every trainable
variable:

===========================  =============
term                         ``|grad|``
===========================  =============
gated mask (this class)      ``3.1751e-13``
object-score BCE             ``4.4969e+00``
IoU regression               ``1.4062e-01``
===========================  =============

Thirteen orders below the score term, while the mask loss VALUE read a healthy
``3.937``. The mechanism: the trained model's GT-present mask logits ran
``min -353.88 / max +67.66 / mean -71.69``, with **95.46 %** of pixels at
``sigmoid < 1e-7`` -- i.e. inside the clip's dead zone. A loss value cannot see
this and neither can any shape, dtype or finiteness check.

Upstream is immune for exactly one reason: it never leaves logit space.
``sigmoid_focal_loss`` (``training/loss_fns.py:77-85``) is::

    prob   = inputs.sigmoid()
    ce     = F.binary_cross_entropy_with_logits(inputs, targets)
    p_t    = prob * targets + (1 - prob) * (1 - targets)
    loss   = ce * ((1 - p_t) ** gamma)
    alpha_t= alpha * targets + (1 - alpha) * (1 - targets)
    loss   = alpha_t * loss

No clip anywhere, and ``d(ce)/d(logit) = p - t`` never reaches zero. This class
reproduces that shape with :func:`_focal_from_logits`, using the stable
``softplus(-x) + x * (1 - t)`` form of BCE-with-logits (the textbook
``max(x, 0) - x*t + log1p(exp(-|x|))`` was tried first and MEASURED to autodiff
to the wrong value -- a sign flip -- at exactly ``x = 0``; see the function's
own anchor).

**It is not a re-derivation of the repository's focal semantics.** For a BINARY
target the two-channel one-hot that ``sam_mask_loss``'s ``to_focal_layout``
builds reduces algebraically to upstream's expression::

    sum_c alpha * (1 - p_c)^gamma * (-t_c log p_c)
        = alpha * (1 - p_t)^gamma * ce        for t in {0, 1}

so on unsaturated logits the new term is the OLD term to float error -- pinned
by ``test_a_clip_with_no_absent_row_matches_the_ungated_loss``, which still
compares this class against plain :class:`SAMMaskLoss` by value. The clip is
the only thing that was removed.

``losses/segmentation_loss.py`` and ``losses/sam_mask_loss.py`` are NOT touched:
both are inside SAM 1's 357-test gate. SAM 1's own mask supervision therefore
still carries this clip; see ``decisions.md`` D-072 for why that is recorded and
not fixed here.

Why the gate zeroes PROBABILITIES for dice and the LOSS for focal
------------------------------------------------------------------
On a GT-absent row:

* **dice**: the gate replaces the predicted probabilities with zeros through
  ``ops.where``, so the row becomes ``1 - (0 + s) / (0 + 0 + s) = 0`` -- exactly
  zero, no residual. MEASURED ``0.0`` at three shapes. Unchanged.
* **focal**: probabilities are no longer the focal term's input, so the gate is
  applied to the PER-PIXEL LOSS instead -- which is literally upstream's
  ``loss_mask * target_obj``. ``ops.where`` again passes exactly zero gradient
  to the suppressed branch, and now the absent row's VALUE is exactly zero too.

That closes plan assumption A-6 rather than bounding it.
:data:`ABSENT_ROW_FOCAL_RESIDUAL` is retained as a named constant (guards derive
bounds from it) and is now **0.0**; the ``4.235165241142481e-22`` it used to
carry was the clip's own per-pixel floor, and its disappearance is one of the
RED proofs that the formulation actually changed.

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
in ``models/SAM/SAM2/training_model.py::compile_sam2_video_trainer`` -- the site a
reader assembling a training run actually reaches. See ``decisions.md`` D-052.
"""

from typing import Any, Dict

import keras
from keras import ops

from .sam_mask_loss import (
    DICE_CHANNELS,
    SAMMaskLoss,
    _require_channels_last,
    _static_mask_stack_shape,
    match_mask_axis,
    to_dice_layout,
)

#: Focal contribution of one fully-gated (all-zero GT) mask row. **Exactly
#: zero** since the focal term moved to logit space: the gate is now applied to
#: the per-pixel loss, so an absent row is removed from the numerator outright.
#:
#: It used to be ``4.235165241142481e-22`` -- the per-pixel floor of
#: ``SegmentationLosses.focal_loss``'s ``ops.clip(y_pred, 1e-7, 1 - 1e-7)``,
#: measured identically at ``(1, 1, 16, 16)``, ``(2, 3, 16, 16)`` and
#: ``(2, 4, 64, 64)`` because it is grid-independent. The constant is KEPT
#: rather than deleted: guards derive bounds from it, and a future change that
#: reintroduced a clipped probability-space focal would push it off zero and be
#: caught by name instead of silently.
ABSENT_ROW_FOCAL_RESIDUAL = 0.0


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


def _focal_from_logits(truth: Any, logits: Any, gamma: float, alpha: float) -> Any:
    """Per-pixel binary focal loss computed in LOGIT space, no clipping.

    Interface contract: ``truth`` and ``logits`` are broadcast-compatible
    tensors of the same dtype carrying, respectively, values in ``{0, 1}`` and
    unbounded real logits; the return has their broadcast shape and is NOT
    reduced -- the caller owns the gate and the reduction. It never raises and
    never allocates a channel axis.

    # DECISION plan-2026-08-04T044628-4c240b4c/D-072
    # Do NOT "simplify" this back to `ops.sigmoid(logits)` fed into
    # `SegmentationLosses.focal_loss`, and do NOT route it through
    # `to_focal_layout`'s two-channel one-hot. That path opens with
    # `ops.clip(y_pred, 1e-7, 1 - 1e-7)` on PROBABILITIES
    # (`segmentation_loss.py:255`), and `ops.clip` has an exactly zero
    # derivative outside its range. MEASURED consequence on the shipped
    # toy checkpoint: 95.46 % of GT-present mask pixels sat at
    # `sigmoid < 1e-7` and the whole mask term's gradient norm was
    # 3.1751e-13 against 4.4969e+00 for the object-score BCE -- the mask
    # head could not learn, while the loss VALUE read a healthy 3.937.
    # Equally, do NOT reach for `ops.log(ops.sigmoid(x))`: that underflows
    # to `-inf` at the `-1024` sentinel D-043 provably emits.
    #
    # The BCE-with-logits below is spelled `softplus(-x) + x*(1-t)`, NOT the
    # textbook `max(x, 0) - x*t + log1p(exp(-|x|))`. The two have identical
    # VALUES at every logit tested (-1024, -70, -8, +/-1e-8, 8, 70, 1024,
    # both targets) but the textbook form's AUTODIFF is wrong at exactly
    # `x = 0`: `abs` has no derivative there and `ops.maximum(x, 0)` breaks
    # its tie towards `x`, giving measured `dce/dx = +1.0` at `t=0` and
    # `0.0` at `t=1` where the true values are `+0.5` and `-0.5` -- a SIGN
    # FLIP on the foreground branch. `softplus` is smooth, so the measured
    # gradient is exact at all nine points. An exactly-zero logit is not
    # hypothetical in this pipeline; `ops.where` gates emit exact zeros.
    # See decisions.md D-072.

    :param truth: Binary ground truth.
    :type truth: Any
    :param logits: Predicted mask logits.
    :type logits: Any
    :param gamma: Focusing exponent, ``LossConfig.focal_gamma``.
    :type gamma: float
    :param alpha: Balancing scale, ``LossConfig.focal_alpha``. Applied as a
        SCALAR to both classes, which is what
        ``SegmentationLosses.focal_loss`` does and therefore what SAM 1's
        :class:`SAMMaskLoss` does; upstream instead uses
        ``alpha_t = alpha*t + (1-alpha)*(1-t)``. That divergence is
        pre-existing, is NOT the gradient-death mechanism, and adopting it
        here would down-weight foreground 3x relative to background -- see
        decisions.md D-073.
    :type alpha: float
    :return: Per-pixel focal loss, unreduced.
    :rtype: Any
    """
    cross_entropy = ops.softplus(-logits) + logits * (1.0 - truth)
    probabilities = ops.sigmoid(logits)
    p_t = probabilities * truth + (1.0 - probabilities) * (1.0 - truth)
    modulator = ops.power(1.0 - p_t, gamma)
    return alpha * modulator * cross_entropy


@keras.saving.register_keras_serializable()
class SAM2GatedMaskLoss(SAMMaskLoss):
    """:class:`SAMMaskLoss` with upstream's ground-truth presence gate.

    Rows whose ground truth is entirely empty contribute exactly zero gradient
    and exactly zero loss (:data:`ABSENT_ROW_FOCAL_RESIDUAL` is ``0.0``). Every
    other row is scored as :class:`SAMMaskLoss` scores it **up to float error,
    on unsaturated logits** -- the dice term is identical, and the focal term is
    the same expression evaluated in LOGIT space rather than through
    ``SegmentationLosses.focal_loss``'s ``1e-7`` probability clip. That clip is
    what made the shipped mask head untrainable (module docstring, D-072), so
    the two formulations DIVERGE by design once ``|logit|`` passes ~16: there
    the parent's gradient is exactly ``0`` and this class's is not.

    The gate is applied to the probabilities on the dice path and to the
    per-pixel loss on the focal path; both are exactly upstream's
    ``loss_* * target_obj``.

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

        if self.from_logits:
            logits = y_pred
            probabilities = ops.sigmoid(logits)
        else:
            # Only the `from_logits=True` path is saturation-immune. Recovering
            # logits from probabilities needs a clip to keep `log` finite, so a
            # caller who hands over probabilities has already destroyed the
            # information the fix depends on. The trainer never takes this path
            # (`compile_sam2_video_trainer` passes raw `low_res_logits`); it
            # exists only so the inherited constructor argument stays honest.
            probabilities = y_pred
            safe = ops.clip(probabilities, 1e-7, 1.0 - 1e-7)
            logits = ops.log(safe) - ops.log1p(-safe)

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

        # DICE: unchanged. `sam_mask_loss`'s adapter and its refusal are reused,
        # not re-derived -- raw-layout dice reduces (num_masks, height) instead
        # of (height, width) and returns a plausible number without raising
        # (H-17). Dice IS still saturation-limited through its own
        # `d(sigmoid)/dx = p(1-p)` chain factor, but so is upstream's
        # (`loss_fns.py:53` sigmoids first too), so that is faithfulness, not a
        # divergence. Only the focal clip was a divergence.
        dice_pred = to_dice_layout(gated)
        dice_true = to_dice_layout(truth)
        _require_channels_last(dice_pred, DICE_CHANNELS, "SAM2GatedMaskLoss dice path")

        # FOCAL: logit space, and the gate moves from the probabilities to the
        # per-pixel loss (D-072). `to_focal_layout`'s two-channel one-hot is
        # deliberately NOT used here -- it exists to make
        # `SegmentationLosses.focal_loss` see negatives at all, and the binary
        # logit form sees them natively. `_require_channels_last` on the focal
        # path goes with it; the layout it guarded no longer exists on this
        # side. The dice guard above is untouched, so H-17's actual trap (the
        # silent wrong reduction) is still refused.
        per_pixel_focal = _focal_from_logits(
            truth, logits, self.config.focal_gamma, self.config.focal_alpha)
        focal_term = ops.mean(
            ops.where(present, per_pixel_focal,
                      ops.zeros_like(per_pixel_focal)))

        dice_term = self._dice(dice_true, dice_pred)
        return self.focal_weight * focal_term + self.dice_weight * dice_term

    def get_config(self) -> Dict[str, Any]:
        """Full serialization config, identical in shape to the parent's.

        :return: Configuration dict consumable by
            :meth:`SAMMaskLoss.from_config`.
        :rtype: Dict[str, Any]
        """
        return super().get_config()
