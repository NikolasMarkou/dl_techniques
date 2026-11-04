"""Segmentation losses on logits and sparse targets, with a void label: Dice, CE + Dice and focal.

Three Keras losses for a segmenter that emits per-pixel LOGITS ``(B, H, W, C)`` and is trained on
integer class maps ``(B, H, W)`` in which one value (``ignore_index``, e.g. 255) means "no label".
They hold no mathematics of their own: each turns the sparse map into a one-hot tensor and a
validity mask and calls the per-sample dense backend of :mod:`~dl_techniques.losses.segmentation_metrics`
(:func:`~dl_techniques.losses.segmentation_metrics.dice_per_sample`,
:func:`~dl_techniques.losses.segmentation_metrics.focal_per_sample`), so there is one Dice and one
focal in the package. The definitions, their sources (Milletari et al. 2016, Lin et al. 2017, the
MONAI, nnU-Net and segmentation_models_pytorch implementations) and the choices are in the
decision log of the plan that added this module (D-014 of plan-2026-11-03T221936-98b866fb).

Interface contract
------------------
``SparseDiceLoss(ignore_index=None, smooth=1e-5, name=..., reduction="sum_over_batch_size")``
    ``loss_b = 1 - mean_c (2 sum_n m g_c p_c + s) / (sum_n m g_c + sum_n m p_c + s)`` per image ``b``,
    ``p = softmax(logits)``, ``g`` the one-hot target, ``m`` the valid-pixel mask, ``s = smooth``.
    All classes count, background included.

``SparseFocalLoss(ignore_index=None, gamma=2.0, ...)``
    ``loss_b = sum_n m (1 - p_t)^gamma (-log p_t) / max(sum_n m, 1)`` per image (Lin et al., eq. 4,
    softmax form; ``gamma = 0`` is the cross-entropy). No ``alpha``.

``SparseCeDiceLoss(ignore_index=None, smooth=1e-5, ...)``
    ``SparseFocalLoss(gamma=0)`` plus ``SparseDiceLoss``, weights 1 and 1 (nnU-Net).

Common to the three:
    :param ignore_index: the void value of the targets, or ``None`` for no void. A void pixel
        changes neither the value nor the gradient of the loss: its logits are multiplied by 0
        before any sum. (The gradient at a void pixel is exactly 0 for finite logits; an infinite
        or NaN logit there gives NaN, as in stock Keras.)
    :param name: / ``reduction``: as :class:`keras.losses.Loss`.
    Call: ``y_true`` integer class ids ``(B, H, W)`` (or ``(B, H, W, 1)``; Keras hands them over as
    floats, which is exact for ids below 2**24), ``y_pred`` float logits ``(B, H, W, C)`` with a
    static ``C`` of at least 2, of any float dtype (the computation is float32).
    Returns: ``(B,)`` float32, ONE VALUE PER IMAGE, so ``sample_weight`` selects images and Keras'
    ``sum_over_batch_size`` averages the images with equal weight. An image with no valid pixel is
    exactly 0 with a zero gradient (that needs ``smooth > 0``) and still counts in the divisor.
    The stock ``SparseCategoricalCrossentropy(ignore_class=)`` differs here: Keras divides its
    per-pixel loss by the batch's valid-pixel count, a mean over pixels of the whole batch.
    Failure mode: ``ValueError`` at construction for ``smooth <= 0`` or ``gamma < 0``, and at call
    time for a non-static or single-channel class axis; a label that is neither a class nor
    ``ignore_index`` gives an all-zero one-hot (it is neither counted nor rejected: validate the
    masks before ``fit``, as ``train.api`` does).
    Serialization: registered through ``register_dl_technique``; ``get_config`` round-trips, and a
    ``.keras`` file that compiled with one reloads with no ``train`` import.
"""

from typing import Any, Dict, Optional, Tuple

import keras

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.losses.segmentation_metrics import dice_per_sample, focal_per_sample
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

#: Dice smoothing, in numerator and denominator (MONAI ``smooth_nr = smooth_dr`` and nnU-Net ``smooth``).
DEFAULT_DICE_SMOOTH = 1e-5

#: Focal focusing parameter (Lin et al.: "we found gamma = 2 to work best").
DEFAULT_FOCAL_GAMMA = 2.0


def _targets(
    y_true: Any, y_pred: Any, ignore_index: Optional[int]
) -> Tuple[Any, Any, Any]:
    """Return ``(logits, one_hot, valid)``: float32 logits, the one-hot of the ids, and the mask.

    A void pixel has an all-zero one-hot row and ``valid`` 0; with ``ignore_index=None`` every
    pixel is valid.
    """
    num_classes = y_pred.shape[-1]
    if num_classes is None or num_classes < 2:
        raise ValueError(
            f"a segmentation loss needs a static class axis of at least 2 channels, got {num_classes}"
        )
    logits = keras.ops.cast(y_pred, "float32")
    if len(y_true.shape) == len(y_pred.shape) and y_true.shape[-1] == 1:
        y_true = keras.ops.squeeze(y_true, axis=-1)
    ids = keras.ops.cast(y_true, "int32")
    if ignore_index is None:
        valid = keras.ops.ones_like(ids, dtype="float32")
        safe = ids
    else:
        is_valid = keras.ops.not_equal(ids, int(ignore_index))
        valid = keras.ops.cast(is_valid, "float32")
        safe = keras.ops.where(is_valid, ids, 0)
    one_hot = keras.ops.one_hot(safe, num_classes, dtype="float32") * keras.ops.expand_dims(valid, -1)
    return logits, one_hot, valid


def _dice(logits: Any, one_hot: Any, valid: Any, smooth: float) -> Any:
    """Return the ``(B,)`` Dice loss of the softmax of ``logits``; the shared body of two losses."""
    probs = keras.ops.softmax(logits, axis=-1)
    return dice_per_sample(one_hot, probs, from_logits=False, smooth=smooth, valid=valid)


def _check_ignore(ignore_index: Optional[int]) -> None:
    if ignore_index is not None and (isinstance(ignore_index, bool) or not isinstance(ignore_index, int)):
        raise ValueError(f"ignore_index must be an int or None, got {ignore_index!r}")


def _check_smooth(smooth: float) -> None:
    # smooth > 0 is what keeps an image with no valid pixel at (0 + s) / (0 + s) = 1, loss 0.
    if not smooth > 0:
        raise ValueError(f"smooth must be positive, got {smooth!r}")


@register_dl_technique("dl_techniques.losses.sparse_segmentation_loss")
class SparseDiceLoss(keras.losses.Loss):
    """Soft Dice loss on logits and sparse targets with a void label; see the module contract."""

    def __init__(
        self,
        ignore_index: Optional[int] = None,
        smooth: float = DEFAULT_DICE_SMOOTH,
        name: str = "sparse_dice",
        reduction: str = "sum_over_batch_size",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, reduction=reduction, **kwargs)
        _check_ignore(ignore_index)
        _check_smooth(smooth)
        self.ignore_index = ignore_index
        self.smooth = float(smooth)

    def call(self, y_true: Any, y_pred: Any) -> Any:
        logits, one_hot, valid = _targets(y_true, y_pred, self.ignore_index)
        return _dice(logits, one_hot, valid, self.smooth)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"ignore_index": self.ignore_index, "smooth": self.smooth})
        return config


@register_dl_technique("dl_techniques.losses.sparse_segmentation_loss")
class SparseFocalLoss(keras.losses.Loss):
    """Multi-class focal loss on logits and sparse targets with a void label; see the module contract."""

    def __init__(
        self,
        ignore_index: Optional[int] = None,
        gamma: float = DEFAULT_FOCAL_GAMMA,
        name: str = "sparse_focal",
        reduction: str = "sum_over_batch_size",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, reduction=reduction, **kwargs)
        _check_ignore(ignore_index)
        if not gamma >= 0:
            raise ValueError(f"gamma must be non-negative, got {gamma!r}")
        self.ignore_index = ignore_index
        self.gamma = float(gamma)

    def call(self, y_true: Any, y_pred: Any) -> Any:
        logits, one_hot, valid = _targets(y_true, y_pred, self.ignore_index)
        return focal_per_sample(one_hot, logits, gamma=self.gamma, from_logits=True, valid=valid)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"ignore_index": self.ignore_index, "gamma": self.gamma})
        return config


@register_dl_technique("dl_techniques.losses.sparse_segmentation_loss")
class SparseCeDiceLoss(keras.losses.Loss):
    """Per-image cross-entropy plus soft Dice (weights 1 and 1); see the module contract."""

    def __init__(
        self,
        ignore_index: Optional[int] = None,
        smooth: float = DEFAULT_DICE_SMOOTH,
        name: str = "sparse_ce_dice",
        reduction: str = "sum_over_batch_size",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, reduction=reduction, **kwargs)
        _check_ignore(ignore_index)
        _check_smooth(smooth)
        self.ignore_index = ignore_index
        self.smooth = float(smooth)

    def call(self, y_true: Any, y_pred: Any) -> Any:
        logits, one_hot, valid = _targets(y_true, y_pred, self.ignore_index)
        # DECISION plan-2026-11-03T221936-98b866fb/D-014
        # The cross-entropy is the focal function at gamma 0 (Lin: "when gamma = 0, FL is equivalent
        # to CE"), so there is one per-image masked mean in the package. Do NOT add a second CE here
        # and do NOT swap in stock SparseCategoricalCrossentropy: its Keras mask divides by the
        # valid-pixel count of the WHOLE BATCH, a batch-level term that breaks the per-row rule
        # (loss(w=[1,..,1,0]) * B/(B-1) == loss(batch[:-1])).
        cross_entropy = focal_per_sample(one_hot, logits, gamma=0.0, from_logits=True, valid=valid)
        return cross_entropy + _dice(logits, one_hot, valid, self.smooth)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"ignore_index": self.ignore_index, "smooth": self.smooth})
        return config
