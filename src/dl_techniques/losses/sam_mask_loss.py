"""
SAM training losses: `SAMMaskLoss` (focal + dice) and `SAMIoULoss` (MSE).

The reuse path is MEASURED BROKEN, and this module is the containment
---------------------------------------------------------------------
``SegmentationLosses`` cannot be pointed at SAM's mask stack as-is. Two
independent defects, both silent:

1. **``focal_loss`` is bit-identically blind to negatives on a 1-channel map.**
   Its cross-entropy term is ``-y_true * log(y_pred)`` summed over the last
   axis, so with a single channel every ``y_true == 0`` pixel contributes
   exactly zero. Setting **every** negative pixel's prediction to ``0.99``
   (maximally wrong) left the loss at ``0.041383`` -- identical to six
   decimals. Feeding a 2-channel one-hot ``concat([1-p, p], -1)`` restores the
   complementary term and the loss reacts (``0.137 -> 0.835``).
2. **``dice_loss`` reduces the wrong axes on the raw ``(B, M, h, w)`` layout.**
   It sums over ``axis=[1, 2]``, which on that layout is ``(M, h)`` rather than
   ``(h, w)``. It does not raise; it returns a plausible ``0.624``. Reshaping to
   ``(B*M, h, w, 1)`` puts the spatial axes where the reduction expects them.

``SegmentationLosses`` itself is deliberately NOT modified: it is shared with
other consumers and changing its reduction semantics would move their numbers.
The adapters live here, the raw layout is REFUSED rather than silently reduced,
and every path ships with a destroy-negatives and a destroy-positives probe.

Layouts
-------
* SAM's mask stack: ``(B, M, h, w)`` -- batch, mask, height, width. This is what
  ``SAMTrainingModel`` emits under ``low_res_logits`` and what the GT is
  broadcast to.
* Dice wants ``(B*M, h, w, 1)``; focal wants ``(B*M, h, w, 2)``.
"""

from typing import Any, Dict, Optional, Tuple

import keras
from keras import ops

from .segmentation_loss import LossConfig
from .segmentation_wrapper_loss import SegmentationWrapperLoss

#: Number of channels each adapted layout must carry.
DICE_CHANNELS = 1
FOCAL_CHANNELS = 2


# ---------------------------------------------------------------------------
# Layout adapters -- the whole point of this module
# ---------------------------------------------------------------------------
def _static_mask_stack_shape(tensor: Any, who: str) -> Tuple[Optional[int], ...]:
    """
    Validate that ``tensor`` is a ``(B, M, h, w)`` mask stack and return its shape.

    Args:
        tensor: The tensor to check.
        who: Name used in the error message.

    Returns:
        The static shape tuple.

    Raises:
        ValueError: if the rank is not 4, or if either spatial extent is
            unknown at trace time (the adapters reshape against them).
    """
    shape = tuple(tensor.shape)
    if len(shape) != 4:
        raise ValueError(
            f"{who} expects a (batch, num_masks, height, width) mask stack; "
            f"got shape {shape} (rank {len(shape)})."
        )
    if shape[2] is None or shape[3] is None:
        raise ValueError(
            f"{who} needs static spatial extents to reshape the mask stack; "
            f"got shape {shape}."
        )
    return shape


def _require_channels_last(tensor: Any, channels: int, who: str) -> None:
    """
    Refuse a tensor that is not ``(N, h, w, channels)``.

    This is the guard that turns ``dice_loss``'s silent wrong answer into a
    loud one. ``SegmentationLosses.dice_loss`` reduces ``axis=[1, 2]``; handed a
    raw ``(B, M, h, w)`` mask stack it reduces ``(M, h)`` instead of ``(h, w)``
    and returns a plausible number. A raw mask stack never has a trailing axis
    of size 1 or 2 for any real mask width, so this check discriminates.

    Args:
        tensor: Candidate tensor.
        channels: Required size of the trailing axis.
        who: Name used in the error message.

    Raises:
        ValueError: if the rank is not 4 or the trailing axis is not
            ``channels``.
    """
    shape = tuple(tensor.shape)
    if len(shape) != 4 or shape[-1] != channels:
        raise ValueError(
            f"{who} requires a channels-last (N, height, width, {channels}) "
            f"tensor; got shape {shape}. Passing SAM's raw (batch, num_masks, "
            f"height, width) mask stack straight into SegmentationLosses is "
            f"the measured trap: dice_loss reduces axis=[1, 2], which on that "
            f"layout is (num_masks, height) rather than (height, width), and "
            f"it returns a plausible number instead of raising. Use "
            f"`to_dice_layout` / `to_focal_layout`."
        )


def _match_mask_axis(
    truth: Any,
    true_shape: Tuple[Optional[int], ...],
    pred_shape: Tuple[Optional[int], ...],
) -> Any:
    """
    Repeat a single-instance GT stack across a multi-mask prediction stack.

    ``SAMTrainingModel`` concatenates every refinement round's logits on the
    mask axis, so ``y_pred`` is ``(B, M*R, h, w)`` while the pipeline's
    ``y_true`` stays ``(B, M, h, w)`` -- and MUST stay that way, or every data
    source would have to know the model's round count.

    Args:
        truth: The GT mask stack, already cast to the prediction's dtype.
        true_shape: Static shape of ``truth``.
        pred_shape: Static shape of the prediction.

    Returns:
        ``truth`` unchanged when the mask axes already agree, else ``truth``
        tiled to the prediction's mask axis.

    Raises:
        ValueError: if the mask axes disagree and the prediction's is not an
            exact multiple of the truth's. Tiling a partial multiple would score
            some rounds against the wrong instance while every rank and spatial
            assertion still passed.
    """
    true_masks, pred_masks = true_shape[1], pred_shape[1]
    if true_masks is None or pred_masks is None or true_masks == pred_masks:
        return truth
    if pred_masks % true_masks != 0:
        raise ValueError(
            f"SAMMaskLoss cannot align y_true's mask axis ({true_masks}) with "
            f"y_pred's ({pred_masks}): {pred_masks} is not a whole multiple of "
            f"{true_masks}. y_pred carries num_masks * num_refinement_rounds "
            f"masks concatenated round-major; y_true must carry either the same "
            f"number or exactly num_masks."
        )
    # `concatenate`, not `repeat` and not `tile`. Not `repeat` because
    # `SAMTrainingModel` concatenates rounds round-major ([r0 masks..., r1
    # masks...]) and `repeat` would interleave, scoring round r against mask r's
    # ground truth. Not `tile` because -- MEASURED -- `ops.tile` erases the
    # static shape under `fit()`'s trace: a `(None, 1, 64, 64)` y_true came back
    # `(None, None, None, None)` and the very next `to_dice_layout` raised
    # `needs static spatial extents`. `concatenate` keeps them.
    return ops.concatenate([truth] * (pred_masks // true_masks), axis=1)


def to_dice_layout(mask_stack: Any) -> Any:
    """
    Reshape a ``(B, M, h, w)`` mask stack to dice's ``(B*M, h, w, 1)`` layout.

    Args:
        mask_stack: Probabilities or binary labels, ``(B, M, h, w)``.

    Returns:
        ``(B*M, h, w, 1)``.

    Raises:
        ValueError: via :func:`_static_mask_stack_shape`.
    """
    shape = _static_mask_stack_shape(mask_stack, "to_dice_layout")
    return ops.reshape(mask_stack, (-1, shape[2], shape[3], DICE_CHANNELS))


def to_focal_layout(mask_stack: Any) -> Any:
    """
    Reshape a ``(B, M, h, w)`` mask stack to focal's ``(B*M, h, w, 2)`` one-hot.

    The complementary channel is what makes the loss see negatives at all.

    Args:
        mask_stack: Probabilities or binary labels in ``[0, 1]``, ``(B, M, h, w)``.

    Returns:
        ``(B*M, h, w, 2)`` = ``concat([1 - p, p], axis=-1)``.

    Raises:
        ValueError: via :func:`_static_mask_stack_shape`.
    """
    single = to_dice_layout(mask_stack)
    return ops.concatenate([1.0 - single, single], axis=-1)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
@keras.saving.register_keras_serializable()
class SAMMaskLoss(keras.losses.Loss):
    """
    Focal + dice on SAM's ``low_res_logits``, through the two adapters.

    Args:
        focal_weight: Weight of the focal term.
        dice_weight: Weight of the dice term. Reference SAM reports a focal:dice
            ratio of 20:1; the shipped defaults are re-derived on this repo's
            code rather than pasted -- see the module's test
            ``test_measured_term_scales`` and decision D-036.
        from_logits: Whether ``y_pred`` carries logits (the default; SAM's
            ``low_res_logits`` does) or probabilities.
        config: ``LossConfig`` forwarded to ``SegmentationLosses``. Defaults to
            ``LossConfig(num_classes=2)`` because the focal adapter is 2-channel.
        name: Keras loss name.
        reduction: Keras reduction. The underlying methods already reduce to a
            scalar, so this stays ``"sum_over_batch_size"`` over a scalar.
        **kwargs: Forwarded to ``keras.losses.Loss``.

    Call args:
        y_true: Binary GT mask stack ``(B, M, h, w)``.
        y_pred: Predicted mask logits (or probabilities) ``(B, M, h, w)``.

    Returns:
        Scalar loss.

    Raises:
        ValueError: if either tensor is not a rank-4 mask stack with static
            spatial extents.
    """

    def __init__(
        self,
        focal_weight: float = 20.0,
        dice_weight: float = 1.0,
        from_logits: bool = True,
        config: Optional[LossConfig] = None,
        name: str = "sam_mask_loss",
        reduction: str = "sum_over_batch_size",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, reduction=reduction, **kwargs)
        self.focal_weight = float(focal_weight)
        self.dice_weight = float(dice_weight)
        self.from_logits = bool(from_logits)
        self.config = config if config is not None else LossConfig(num_classes=2)
        self._focal = SegmentationWrapperLoss("focal", self.config)
        self._dice = SegmentationWrapperLoss("dice", self.config)

    def call(self, y_true: Any, y_pred: Any) -> Any:
        """Compute ``focal_weight * focal + dice_weight * dice``."""
        pred_shape = _static_mask_stack_shape(y_pred, "SAMMaskLoss(y_pred)")
        true_shape = _static_mask_stack_shape(y_true, "SAMMaskLoss(y_true)")

        probabilities = ops.sigmoid(y_pred) if self.from_logits else y_pred
        truth = ops.cast(y_true, probabilities.dtype)
        truth = _match_mask_axis(truth, true_shape, pred_shape)

        # DECISION plan-2026-08-03T191222-1d751f81/D-036: focal gets a TWO-channel
        # one-hot and dice gets a ONE-channel channels-last stack. Do NOT
        # "simplify" either adapter to the other's layout, and above all do NOT
        # pass the raw (B, M, h, w) stack to either: 1-channel focal is
        # bit-identically blind to every negative pixel (measured 0.041383
        # before and after destroying all of them) and raw-layout dice reduces
        # (num_masks, height) instead of (height, width) and returns a plausible
        # number without raising. `_require_channels_last` refuses both.
        dice_pred = to_dice_layout(probabilities)
        dice_true = to_dice_layout(truth)
        _require_channels_last(dice_pred, DICE_CHANNELS, "SAMMaskLoss dice path")

        focal_pred = to_focal_layout(probabilities)
        focal_true = to_focal_layout(truth)
        _require_channels_last(focal_pred, FOCAL_CHANNELS, "SAMMaskLoss focal path")

        focal_term = self._focal(focal_true, focal_pred)
        dice_term = self._dice(dice_true, dice_pred)
        return self.focal_weight * focal_term + self.dice_weight * dice_term

    def get_config(self) -> Dict[str, Any]:
        """Full serialization config."""
        config = super().get_config()
        config.update(
            {
                "focal_weight": self.focal_weight,
                "dice_weight": self.dice_weight,
                "from_logits": self.from_logits,
                "config": keras.saving.serialize_keras_object(self.config),
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SAMMaskLoss":
        """Rebuild from :meth:`get_config` output."""
        config = dict(config)
        config["config"] = keras.saving.deserialize_keras_object(config["config"])
        return cls(**config)


@keras.saving.register_keras_serializable()
class SAMIoULoss(keras.losses.Loss):
    """
    MSE between the predicted IoU and the IoU the prediction actually achieved.

    Call args:
        y_true: Structurally unused -- see the note below. Any tensor Keras
            routes to this output key is accepted.
        y_pred: ``(B, M, 2)`` where ``[..., 0]`` is the model's predicted IoU
            and ``[..., 1]`` is the achieved IoU, already ``stop_gradient``-ed
            by the model.

    Returns:
        Scalar MSE.

    Raises:
        ValueError: if ``y_pred``'s trailing axis is not exactly 2 -- the one
            mis-wiring that would otherwise train the IoU head against garbage.

    Note:
        ``y_true`` carries no information here, and that is a property of the
        quantity rather than an oversight: the achieved IoU is a function of the
        *prediction* and the GT together, so it exists only inside the model.
        A ``tf.data`` pipeline cannot produce it, and stock ``compile(loss=...)``
        hands each loss only its own output key. The model therefore packs the
        prediction and its stop-gradient target into one tensor, and this loss
        reads both from ``y_pred``. The alternative -- a custom ``train_step``
        -- is forbidden by standing instruction and is unnecessary.
    """

    def __init__(
        self,
        name: str = "sam_iou_loss",
        reduction: str = "sum_over_batch_size",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, reduction=reduction, **kwargs)

    def call(self, y_true: Any, y_pred: Any) -> Any:
        """Compute ``mean((predicted_iou - achieved_iou) ** 2)``."""
        shape = tuple(y_pred.shape)
        if len(shape) != 3 or shape[-1] != 2:
            raise ValueError(
                "SAMIoULoss expects y_pred of shape (batch, num_masks, 2) -- "
                "[..., 0] the predicted IoU and [..., 1] the achieved IoU "
                "computed by the model; got shape "
                f"{shape}. Route this loss to SAMTrainingModel's "
                "'iou_supervision' output key, not to 'iou_predictions'."
            )
        predicted = y_pred[..., 0]
        achieved = ops.stop_gradient(y_pred[..., 1])
        return ops.mean(ops.square(predicted - achieved))

    def get_config(self) -> Dict[str, Any]:
        """Full serialization config."""
        return super().get_config()
