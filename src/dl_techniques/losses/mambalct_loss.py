"""MambaLCT tracking loss: L1 box regression plus GIoU.

This module holds :class:`MambaLCTBoxLoss`, the box-regression objective
for the MambaLCT tracker (Li et al., 2024, Eq. 10 without the classification
term)::

    L_box = l1_weight * L1 + giou_weight * (1 - GIoU)

Boxes are ``(cx, cy, w, h)`` in normalized search-crop coordinates. The
classification term stays at compile time (stock binary cross-entropy over
the per-frame score) and the ``l1``/``giou`` balance lives here as
constructor arguments, following the ``siamese_tracking_loss.py``
precedent of keeping balance weights out of the packed label.

``call`` returns one value per sample ``(batch,)``: the per-frame losses
are averaged over the clip length ``T`` inside each sample, so each row
selects itself and ``sample_weight`` stays correct under any ``reduction=``.

References:
    - Li et al., 2024. MambaLCT: Boosting Tracking via Long-term Context
      State Space Model. (https://arxiv.org/abs/2412.13615)
"""

import keras
from typing import Any, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.utils.bounding_box import bbox_iou

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.losses.mambalct_loss")
class MambaLCTBoxLoss(keras.losses.Loss):
    """L1 plus GIoU loss over per-frame boxes.

    Args:
        l1_weight: Weight of the L1 term. Defaults to 5.0 (paper Eq. 10).
        giou_weight: Weight of the ``(1 - GIoU)`` term. Defaults to 2.0.
        reduction: Keras reduction. Defaults to ``"sum_over_batch_size"``.
        name: Loss name.

    Input shape:
        ``y_true`` / ``y_pred`` are ``(batch, frames, 4)`` in normalized
        ``(cx, cy, w, h)`` coordinates.

    Output shape:
        ``(batch,)`` — one value per sample.
    """

    def __init__(
        self,
        l1_weight: float = 5.0,
        giou_weight: float = 2.0,
        reduction: str = "sum_over_batch_size",
        name: str = "mambalct_box_loss",
        **kwargs: Any,
    ) -> None:
        super().__init__(reduction=reduction, name=name, **kwargs)
        if l1_weight < 0:
            raise ValueError(f"l1_weight must be non-negative, got {l1_weight}")
        if giou_weight < 0:
            raise ValueError(
                f"giou_weight must be non-negative, got {giou_weight}"
            )
        self.l1_weight = l1_weight
        self.giou_weight = giou_weight
        logger.info(
            f"MambaLCTBoxLoss configured: l1_weight={l1_weight}, "
            f"giou_weight={giou_weight}"
        )

    def call(
        self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Compute the weighted box loss per sample.

        Args:
            y_true: Ground-truth boxes ``(batch, frames, 4)``.
            y_pred: Predicted boxes ``(batch, frames, 4)``.

        Returns:
            Per-sample loss ``(batch,)``.
        """
        l1 = keras.ops.mean(
            keras.ops.abs(y_true - y_pred), axis=(-1, -2)
        )
        giou = bbox_iou(y_true, y_pred, xywh=True, GIoU=True)
        giou_term = keras.ops.mean(1.0 - giou, axis=-1)
        return self.l1_weight * l1 + self.giou_weight * giou_term

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        Returns:
            Dictionary containing all constructor arguments.
        """
        config = super().get_config()
        config.update(
            {
                "l1_weight": self.l1_weight,
                "giou_weight": self.giou_weight,
            }
        )
        return config
