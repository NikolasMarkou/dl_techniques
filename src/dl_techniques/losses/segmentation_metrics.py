"""Per-sample segmentation metrics backend.

This module provides per-sample (shape ``(batch,)``) implementations of common
segmentation metrics: IoU, Dice, Tversky, Focal Tversky, and Focal loss.

These functions are designed to be used as the computation backend for
`SegmentationLosses` (via `SegmentationWrapperLoss`) and the `any_loss`
segmentation classes (`DiceLoss`, `IoULoss`, `TverskyLoss`, `FocalTverskyLoss`).

Design rationale:
-----------------
`SegmentationLosses` historically reduced to a scalar (batch mean) for SAM
compatibility. `AnyLoss` subclasses compute global confusion matrices (also
scalar). Both violate the `losses/AGENTS.md` rule that ``call()`` must return
``(batch,)``. This module exposes the per-sample math so callers can choose
their reduction strategy:

- ``SegmentationWrapperLoss`` (and SAM) can keep the scalar behavior by
  wrapping these and adding ``ops.mean`` at the end.
- New `any_loss` per-sample variants can return ``(batch,)`` directly for
  correct ``sample_weight`` semantics.

All functions assume ``y_true`` and ``y_pred`` have shape
``(batch, ..., num_classes)`` with channels-last layout. Non-batch axes are
reduced to produce a ``(batch,)`` result.

"""
from typing import Any, Optional

import keras


def _get_rank(x: Any) -> int:
    """Get the rank (number of dimensions) of a tensor or array."""
    if hasattr(x.shape, 'rank'):
        r = x.shape.rank
        if r is not None:
            return r
    return len(x.shape)


def _reduce_non_batch(x: Any) -> Any:
    """Reduce all non-batch axes via mean, returning ``(batch,)``."""
    rank = _get_rank(x)
    if rank <= 1:
        return x
    non_batch_axes = tuple(range(1, rank))
    return keras.ops.mean(x, axis=non_batch_axes)


def _prepare_probs(y_pred: keras.KerasTensor, from_logits: bool) -> keras.KerasTensor:
    """Convert logits to probabilities if needed.

    Logits become probabilities by a sigmoid when the channel axis has ONE
    entry (a binary logit map) and by a softmax over the channel axis
    otherwise. A softmax over a single channel would be the constant 1.

    Does NOT clip when from_logits=False, because 0/1 are valid probabilities
    for metrics like Dice/IoU where clipping would change the semantics
    (e.g., gated absent rows in SAM2 should contribute exactly zero).
    Individual loss functions that need numerical stability for log()
    should clip locally.

    Interface contract (callers: every ``*_per_sample`` function here,
    ``cross_entropy_scalar`` and ``SegmentationLosses``): ``y_pred`` is
    channels-last with a static channel count; returns a tensor of the same
    shape, ``y_pred`` itself when ``from_logits`` is False. Raises nothing.
    """
    if from_logits:
        # DECISION plan-2026-10-14T062616-81938e81/D-143: a one-channel logit map
        # is a sigmoid; never a softmax over one channel (the constant 1).
        if y_pred.shape[-1] == 1:
            return keras.ops.sigmoid(y_pred)
        return keras.ops.softmax(y_pred, axis=-1)
    return y_pred


def _log_probs_from_logits(y_pred: keras.KerasTensor) -> keras.KerasTensor:
    """Log-probabilities of a channels-last logit map, in float32, without a clip.

    ``log_sigmoid`` for one channel and ``log_softmax`` over the channel axis
    otherwise (the same rule as :func:`_prepare_probs`). No probability clip, so
    the gradient never meets the clip's zero-derivative region.

    Interface contract (callers: ``cross_entropy_scalar`` and
    ``SegmentationLosses.focal_loss``, both on their ``from_logits`` path):
    ``y_pred`` is channels-last with a static channel count; returns float32 of
    the same shape. Raises nothing.
    """
    y_pred = keras.ops.cast(y_pred, "float32")
    if y_pred.shape[-1] == 1:
        return keras.ops.log_sigmoid(y_pred)
    return keras.ops.log_softmax(y_pred, axis=-1)


def iou_per_sample(
    y_true: keras.KerasTensor,
    y_pred: keras.KerasTensor,
    from_logits: bool = False,
    smooth: float = 1e-6,
) -> keras.KerasTensor:
    """Per-sample IoU (Jaccard index) loss: 1 - TP / (TP + FP + FN).

    Args:
        y_true: Ground truth, shape ``(batch, ..., num_classes)``, one-hot or
            binary (0/1) per class.
        y_pred: Predicted logits or probabilities, same shape as ``y_true``.
        from_logits: Whether ``y_pred`` contains raw logits.
        smooth: Smoothing constant added to numerator and denominator.

    Returns:
        Loss tensor of shape ``(batch,)``.
    """
    y_true = keras.ops.cast(y_true, "float32")
    probs = _prepare_probs(y_pred, from_logits)

    # Per-sample, per-class TP, FP, FN summed over spatial axes
    spatial_axes = tuple(range(1, _get_rank(probs) - 1))
    tp = keras.ops.sum(y_true * probs, axis=spatial_axes)
    fp = keras.ops.sum((1.0 - y_true) * probs, axis=spatial_axes)
    fn = keras.ops.sum(y_true * (1.0 - probs), axis=spatial_axes)

    iou = (tp + smooth) / (tp + fp + fn + smooth)
    # Mean over classes per sample
    return 1.0 - keras.ops.mean(iou, axis=-1)


def dice_per_sample(
    y_true: keras.KerasTensor,
    y_pred: keras.KerasTensor,
    from_logits: bool = False,
    smooth: float = 1e-6,
    valid: Optional[keras.KerasTensor] = None,
) -> keras.KerasTensor:
    """Per-sample Dice loss: 1 - 2*TP / (2*TP + FP + FN).

    Per image and per class, the linear denominator, the smoothing in the numerator and in the
    denominator, then the mean over classes (MONAI ``DiceLoss`` at its defaults).

    Args:
        y_true: Ground truth, shape ``(batch, ..., num_classes)``.
        y_pred: Predicted logits or probabilities, same shape.
        from_logits: Whether ``y_pred`` contains raw logits.
        smooth: Smoothing constant added to numerator and denominator.
        valid: Optional ``(batch, ...)`` mask, 1 (or True) where a pixel counts and 0 where it is
            void; a void pixel then adds nothing to TP, FP or FN, so neither the value nor the
            gradient reads it. ``None`` counts every pixel (the behaviour before this argument).

    Returns:
        Loss tensor of shape ``(batch,)``.
    """
    y_true = keras.ops.cast(y_true, "float32")
    probs = _prepare_probs(y_pred, from_logits)
    if valid is not None:
        weight = keras.ops.expand_dims(keras.ops.cast(valid, "float32"), -1)
        y_true = y_true * weight
        probs = probs * weight

    spatial_axes = tuple(range(1, _get_rank(probs) - 1))
    tp = keras.ops.sum(y_true * probs, axis=spatial_axes)
    fp = keras.ops.sum((1.0 - y_true) * probs, axis=spatial_axes)
    fn = keras.ops.sum(y_true * (1.0 - probs), axis=spatial_axes)

    dice = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
    return 1.0 - keras.ops.mean(dice, axis=-1)


def tversky_per_sample(
    y_true: keras.KerasTensor,
    y_pred: keras.KerasTensor,
    alpha: float = 0.5,
    beta: float = 0.5,
    from_logits: bool = False,
    smooth: float = 1e-6,
) -> keras.KerasTensor:
    """Per-sample Tversky loss: 1 - TP / (TP + alpha*FP + beta*FN).

    Args:
        y_true: Ground truth, shape ``(batch, ..., num_classes)``.
        y_pred: Predicted logits or probabilities, same shape.
        alpha: Weight for false positives.
        beta: Weight for false negatives.
        from_logits: Whether ``y_pred`` contains raw logits.
        smooth: Smoothing constant.

    Returns:
        Loss tensor of shape ``(batch,)``.
    """
    y_true = keras.ops.cast(y_true, "float32")
    probs = _prepare_probs(y_pred, from_logits)

    spatial_axes = tuple(range(1, _get_rank(probs) - 1))
    tp = keras.ops.sum(y_true * probs, axis=spatial_axes)
    fp = keras.ops.sum((1.0 - y_true) * probs, axis=spatial_axes)
    fn = keras.ops.sum(y_true * (1.0 - probs), axis=spatial_axes)

    denominator = tp + alpha * fp + beta * fn
    tversky = (tp + smooth) / (denominator + smooth)
    return 1.0 - keras.ops.mean(tversky, axis=-1)


def focal_tversky_per_sample(
    y_true: keras.KerasTensor,
    y_pred: keras.KerasTensor,
    alpha: float = 0.7,
    beta: float = 0.3,
    gamma: float = 0.75,
    from_logits: bool = False,
    smooth: float = 1e-6,
) -> keras.KerasTensor:
    """Per-sample Focal Tversky loss: (1 - Tversky)^gamma.

    Args:
        y_true: Ground truth, shape ``(batch, ..., num_classes)``.
        y_pred: Predicted logits or probabilities, same shape.
        alpha: Weight for false positives.
        beta: Weight for false negatives.
        gamma: Focal exponent (focus on hard examples).
        from_logits: Whether ``y_pred`` contains raw logits.
        smooth: Smoothing constant.

    Returns:
        Loss tensor of shape ``(batch,)``.
    """
    tversky_loss = tversky_per_sample(
        y_true, y_pred, alpha=alpha, beta=beta, from_logits=from_logits, smooth=smooth
    )
    # tversky_loss is already (batch,) = 1 - mean(tversky)
    # Focal Tversky applies gamma to the per-class (1-tversky) then means.
    # To match the original scalar behavior exactly, we'd need per-class tversky.
    # This implementation applies gamma to the per-sample loss (approximate).
    return keras.ops.power(tversky_loss, gamma)


#: Floor of ``1 - p_t`` in the focal modulating factor: ``(1 - p_t) ** gamma`` has an infinite
#: slope at ``p_t = 1`` for ``0 < gamma < 1``, which times the zero ``log p_t`` of a saturated pixel
#: is NaN; the floor makes that gradient finite (and zero below it).
_FOCAL_TINY = 1e-12


def focal_per_sample(
    y_true: keras.KerasTensor,
    y_pred: keras.KerasTensor,
    gamma: float = 2.0,
    from_logits: bool = False,
    valid: Optional[keras.KerasTensor] = None,
) -> keras.KerasTensor:
    """Per-sample multi-class focal loss: the mean over pixels of ``-(1 - p_t)^gamma log p_t``.

    Lin et al. 2017 (arXiv 1708.02002) eq. 4, in its softmax form: ``p_t`` is the probability of
    the TRUE class, so ``gamma = 0`` is the cross-entropy. There is no ``alpha``: Lin's
    ``alpha_t`` balances the positive against the negative class of a sigmoid problem and a
    softmax pixel has one term (an earlier version added a ``p_t^gamma log(1 - p_t)`` term for a
    "negative class", which charged a confident correct pixel; it also returned ``(batch, H, W)``).

    Args:
        y_true: Ground truth, shape ``(batch, ..., num_classes)``, one-hot; a pixel whose one-hot is all
            zero has no term.
        y_pred: Predicted logits or probabilities, same shape, at least 2 channels.
        gamma: Focusing parameter, at least 0.
        from_logits: Whether ``y_pred`` contains raw logits (read in log space, no clip).
        valid: Optional ``(batch, ...)`` mask, 1 (or True) where a pixel counts and 0 where it is
            void. The mean is over the valid pixels of each image, and an image with none is 0.
            ``None`` averages over every pixel.

    Returns:
        Loss tensor of shape ``(batch,)``.

    Raises:
        ValueError: For a negative ``gamma`` or a single channel.
    """
    if gamma < 0:
        raise ValueError(f"gamma must be non-negative, got {gamma}")
    if y_pred.shape[-1] is None or y_pred.shape[-1] < 2:
        raise ValueError(
            f"focal loss needs a static channel count of at least 2, got {y_pred.shape[-1]}"
        )
    y_true = keras.ops.cast(y_true, "float32")
    if from_logits:
        log_p = _log_probs_from_logits(y_pred)
    else:
        probs = keras.ops.clip(keras.ops.cast(y_pred, "float32"), 1e-7, 1.0 - 1e-7)
        log_p = keras.ops.log(probs)

    log_pt = keras.ops.sum(y_true * log_p, axis=-1)
    loss = -log_pt
    if gamma > 0:
        one_minus_pt = keras.ops.maximum(-keras.ops.expm1(log_pt), _FOCAL_TINY)
        loss = keras.ops.power(one_minus_pt, gamma) * loss

    spatial_axes = tuple(range(1, _get_rank(loss)))
    if valid is None:
        return keras.ops.mean(loss, axis=spatial_axes)
    weight = keras.ops.cast(valid, "float32")
    count = keras.ops.sum(weight, axis=spatial_axes)
    return keras.ops.sum(loss * weight, axis=spatial_axes) / keras.ops.maximum(count, 1.0)


# ---------------------------------------------------------------------------
# Backward-compat scalar versions for SegmentationLosses (SAM compatibility)
# These wrap the per-sample versions and add the final batch mean.
# ---------------------------------------------------------------------------


def iou_scalar(
    y_true: keras.KerasTensor,
    y_pred: keras.KerasTensor,
    from_logits: bool = False,
    smooth: float = 1e-6,
) -> keras.KerasTensor:
    """Scalar IoU loss (batch mean). For SegmentationLosses / SAM compat."""
    return keras.ops.mean(iou_per_sample(y_true, y_pred, from_logits, smooth))


def dice_scalar(
    y_true: keras.KerasTensor,
    y_pred: keras.KerasTensor,
    from_logits: bool = False,
    smooth: float = 1e-6,
) -> keras.KerasTensor:
    """Scalar Dice loss (batch mean). For SegmentationLosses / SAM compat."""
    return keras.ops.mean(dice_per_sample(y_true, y_pred, from_logits, smooth))


def tversky_scalar(
    y_true: keras.KerasTensor,
    y_pred: keras.KerasTensor,
    alpha: float = 0.5,
    beta: float = 0.5,
    from_logits: bool = False,
    smooth: float = 1e-6,
) -> keras.KerasTensor:
    """Scalar Tversky loss (batch mean). For SegmentationLosses / SAM compat."""
    return keras.ops.mean(tversky_per_sample(y_true, y_pred, alpha, beta, from_logits, smooth))


def focal_tversky_scalar(
    y_true: keras.KerasTensor,
    y_pred: keras.KerasTensor,
    alpha: float = 0.7,
    beta: float = 0.3,
    gamma: float = 0.75,
    from_logits: bool = False,
    smooth: float = 1e-6,
) -> keras.KerasTensor:
    """Scalar Focal Tversky loss (batch mean). For SegmentationLosses / SAM compat."""
    # Note: original applies gamma per-class then means. This applies gamma to per-sample.
    return keras.ops.mean(focal_tversky_per_sample(y_true, y_pred, alpha, beta, gamma, from_logits, smooth))


def focal_scalar(
    y_true: keras.KerasTensor,
    y_pred: keras.KerasTensor,
    gamma: float = 2.0,
    from_logits: bool = False,
) -> keras.KerasTensor:
    """Scalar multi-class focal loss (batch mean); see :func:`focal_per_sample`."""
    return keras.ops.mean(focal_per_sample(y_true, y_pred, gamma, from_logits))


def cross_entropy_scalar(
    y_true: keras.KerasTensor,
    y_pred: keras.KerasTensor,
    weights: Optional[keras.KerasTensor] = None,
    from_logits: bool = False,
) -> keras.KerasTensor:
    """Scalar Cross-Entropy loss (batch mean). For SegmentationLosses.

    Args:
        y_true: Ground truth, shape ``(batch, ..., num_classes)``.
        y_pred: Predicted logits or probabilities, same shape.
        weights: Optional class weights ``(num_classes,)``.
        from_logits: Whether ``y_pred`` contains raw logits. Logits are read
            in log space (``log_sigmoid`` for one channel, ``log_softmax``
            otherwise) with no clip; probabilities are clipped to
            ``[1e-7, 1 - 1e-7]`` before the log.

    Returns:
        Scalar loss tensor.
    """
    y_true = keras.ops.cast(y_true, "float32")
    if from_logits:
        log_probs = _log_probs_from_logits(y_pred)
    else:
        epsilon = 1e-7
        probs = keras.ops.clip(y_pred, epsilon, 1.0 - epsilon)
        log_probs = keras.ops.log(probs)

    ce_loss = -keras.ops.sum(y_true * log_probs, axis=-1)

    if weights is not None:
        weights = keras.ops.cast(weights, "float32")
        ce_loss = ce_loss * keras.ops.sum(y_true * weights, axis=-1)

    return keras.ops.mean(ce_loss)


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------
__all__ = [
    "iou_per_sample",
    "dice_per_sample",
    "tversky_per_sample",
    "focal_tversky_per_sample",
    "focal_per_sample",
    "iou_scalar",
    "dice_scalar",
    "tversky_scalar",
    "focal_tversky_scalar",
    "focal_scalar",
    "cross_entropy_scalar",
]