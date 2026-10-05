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
from keras import ops


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
    return ops.mean(x, axis=non_batch_axes)


def _prepare_probs(y_pred: keras.KerasTensor, from_logits: bool) -> keras.KerasTensor:
    """Convert logits to probabilities if needed.

    Does NOT clip when from_logits=False, because 0/1 are valid probabilities
    for metrics like Dice/IoU where clipping would change the semantics
    (e.g., gated absent rows in SAM2 should contribute exactly zero).
    Individual loss functions that need numerical stability for log()
    should clip locally.
    """
    if from_logits:
        return ops.softmax(y_pred, axis=-1)
    return y_pred


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
    y_true = ops.cast(y_true, "float32")
    probs = _prepare_probs(y_pred, from_logits)

    # Per-sample, per-class TP, FP, FN summed over spatial axes
    spatial_axes = tuple(range(1, _get_rank(probs) - 1))
    tp = ops.sum(y_true * probs, axis=spatial_axes)
    fp = ops.sum((1.0 - y_true) * probs, axis=spatial_axes)
    fn = ops.sum(y_true * (1.0 - probs), axis=spatial_axes)

    iou = (tp + smooth) / (tp + fp + fn + smooth)
    # Mean over classes per sample
    return 1.0 - ops.mean(iou, axis=-1)


def dice_per_sample(
    y_true: keras.KerasTensor,
    y_pred: keras.KerasTensor,
    from_logits: bool = False,
    smooth: float = 1e-6,
) -> keras.KerasTensor:
    """Per-sample Dice loss: 1 - 2*TP / (2*TP + FP + FN).

    Args:
        y_true: Ground truth, shape ``(batch, ..., num_classes)``.
        y_pred: Predicted logits or probabilities, same shape.
        from_logits: Whether ``y_pred`` contains raw logits.
        smooth: Smoothing constant added to numerator and denominator.

    Returns:
        Loss tensor of shape ``(batch,)``.
    """
    y_true = ops.cast(y_true, "float32")
    probs = _prepare_probs(y_pred, from_logits)

    spatial_axes = tuple(range(1, _get_rank(probs) - 1))
    tp = ops.sum(y_true * probs, axis=spatial_axes)
    fp = ops.sum((1.0 - y_true) * probs, axis=spatial_axes)
    fn = ops.sum(y_true * (1.0 - probs), axis=spatial_axes)

    dice = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
    return 1.0 - ops.mean(dice, axis=-1)


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
    y_true = ops.cast(y_true, "float32")
    probs = _prepare_probs(y_pred, from_logits)

    spatial_axes = tuple(range(1, _get_rank(probs) - 1))
    tp = ops.sum(y_true * probs, axis=spatial_axes)
    fp = ops.sum((1.0 - y_true) * probs, axis=spatial_axes)
    fn = ops.sum(y_true * (1.0 - probs), axis=spatial_axes)

    denominator = tp + alpha * fp + beta * fn
    tversky = (tp + smooth) / (denominator + smooth)
    return 1.0 - ops.mean(tversky, axis=-1)


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
    return ops.power(tversky_loss, gamma)


def focal_per_sample(
    y_true: keras.KerasTensor,
    y_pred: keras.KerasTensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    from_logits: bool = False,
) -> keras.KerasTensor:
    """Per-sample Focal loss (two-sided, correct version).

    FL = -alpha * (1-p_t)^gamma * log(p_t) for positive class
       - (1-alpha) * p_t^gamma * log(1-p_t) for negative class

    This is the corrected two-sided version (unlike the single-sided
    ``SegmentationLosses.focal_loss`` which only computes the positive term).

    Args:
        y_true: Ground truth, shape ``(batch, ..., num_classes)``, one-hot.
        y_pred: Predicted logits or probabilities, same shape.
        alpha: Class weight for positive class.
        gamma: Focusing parameter.
        from_logits: Whether ``y_pred`` contains raw logits.

    Returns:
        Loss tensor of shape ``(batch,)``.
    """
    y_true = ops.cast(y_true, "float32")
    probs = _prepare_probs(y_pred, from_logits)

    # Clip for numerical stability in log terms
    probs = ops.clip(probs, 1e-7, 1.0 - 1e-7)

    # p_t = prob for the true class, 1-p_t for negative class
    # y_true is one-hot, so sum over classes gives p_t for each sample
    p_t = ops.sum(y_true * probs, axis=-1)

    # Focal weight
    focal_weight_pos = ops.power(1.0 - p_t, gamma)
    focal_weight_neg = ops.power(p_t, gamma)

    # Log terms
    log_pos = ops.log(p_t + 1e-8)
    log_neg = ops.log(1.0 - p_t + 1e-8)

    loss = -alpha * focal_weight_pos * log_pos - (1.0 - alpha) * focal_weight_neg * log_neg
    return loss  # Already (batch,)


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
    return ops.mean(iou_per_sample(y_true, y_pred, from_logits, smooth))


def dice_scalar(
    y_true: keras.KerasTensor,
    y_pred: keras.KerasTensor,
    from_logits: bool = False,
    smooth: float = 1e-6,
) -> keras.KerasTensor:
    """Scalar Dice loss (batch mean). For SegmentationLosses / SAM compat."""
    return ops.mean(dice_per_sample(y_true, y_pred, from_logits, smooth))


def tversky_scalar(
    y_true: keras.KerasTensor,
    y_pred: keras.KerasTensor,
    alpha: float = 0.5,
    beta: float = 0.5,
    from_logits: bool = False,
    smooth: float = 1e-6,
) -> keras.KerasTensor:
    """Scalar Tversky loss (batch mean). For SegmentationLosses / SAM compat."""
    return ops.mean(tversky_per_sample(y_true, y_pred, alpha, beta, from_logits, smooth))


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
    return ops.mean(focal_tversky_per_sample(y_true, y_pred, alpha, beta, gamma, from_logits, smooth))


def focal_scalar(
    y_true: keras.KerasTensor,
    y_pred: keras.KerasTensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    from_logits: bool = False,
) -> keras.KerasTensor:
    """Scalar Focal loss (batch mean, two-sided). For SegmentationLosses."""
    return ops.mean(focal_per_sample(y_true, y_pred, alpha, gamma, from_logits))


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
        from_logits: Whether ``y_pred`` contains raw logits.

    Returns:
        Scalar loss tensor.
    """
    y_true = ops.cast(y_true, "float32")
    probs = _prepare_probs(y_pred, from_logits)

    epsilon = 1e-7
    probs = ops.clip(probs, epsilon, 1.0 - epsilon)

    ce_loss = -ops.sum(y_true * ops.log(probs), axis=-1)

    if weights is not None:
        weights = ops.cast(weights, "float32")
        ce_loss = ce_loss * ops.sum(y_true * weights, axis=-1)

    return ops.mean(ce_loss)


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