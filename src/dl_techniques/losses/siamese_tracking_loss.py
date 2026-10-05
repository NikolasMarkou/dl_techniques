"""Siamese tracking losses: SiamFC logistic loss and DaSiamRPN cls/reg losses.

This module holds the three training objectives for the Siamese trackers in
``dl_techniques.models.vision``:

1. :class:`SiamFCLogisticLoss` -- the radius-based logistic loss over the
   single-channel response map (Bertinetto et al., ECCV 2016). Labels come
   from :func:`create_siamfc_label`, which transcribes the reference
   ``create_BCELogit_loss_label`` convention (positive disc, ignored ring,
   negative surround).
2. :class:`DaSiamRPNClsLoss` -- per-anchor 2-way softmax cross-entropy from
   logits (Zhu et al., ECCV 2018, via Li et al., CVPR 2018).
3. :class:`DaSiamRPNRegLoss` -- per-anchor smooth-L1 (Huber) box-delta
   regression over positive anchors only.

Masking follows the packed-label convention (:func:`pack_matches` in
``lightglue_loss.py`` is the precedent): ignore/valid flags travel inside
``y_true`` next to the values, never in ``sample_weight``, so each ``call``
reduces to per-sample ``(B,)`` itself and stays correct under any
``reduction=``. The cls-vs-reg balance weight is applied at COMPILE time by
the trainer via ``loss_weights`` and is NOT baked into either class (same
rule as ``SuperPointDetectorLoss`` and its ``lambda_d``).

All ops are ``keras.ops``-only and graph-safe. Label builders are pure NumPy:
they run in the data pipeline, not in the graph.

References:
    - Bertinetto et al., 2016. Fully-Convolutional Siamese Networks for
      Object Tracking. (https://arxiv.org/abs/1606.09549)
    - Zhu et al., 2018. Distractor-aware Siamese Networks for Visual Object
      Tracking. ECCV 2018. (https://arxiv.org/abs/1808.06048)
    - Li et al., 2018. High Performance Visual Tracking with Siamese Region
      Proposal Network. CVPR 2018.
"""

import keras
import numpy as np
from typing import Any, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.losses.huber_loss import HuberLoss

# ---------------------------------------------------------------------
# label builders (pure NumPy, data-pipeline side)
# ---------------------------------------------------------------------


def create_siamfc_label(
    score_size: int,
    pos_radius_px: float = 25.0,
    neg_radius_px: float = 50.0,
    total_stride: int = 8,
) -> np.ndarray:
    """Radius-based SiamFC training label for a centered pair.

    Transcribes the reference ``create_BCELogit_loss_label``: distances are
    measured in search-image pixels from the (possibly between-pixel) map
    center, then compared against the radii divided by the total stride.
    Channel 0 holds the positive/negative value (1 inside the positive disc,
    0 elsewhere); channel 1 holds the valid flag (1 for positive or
    negative pixels, 0 for the ignored ring between the radii).

    Args:
        score_size: Square score-map extent (17 for 127/255).
        pos_radius_px: Positive radius in search-image pixels (ref: 25).
        neg_radius_px: Negative radius in search-image pixels (ref: 50).
        total_stride: Network total stride (ref: 8).

    Returns:
        Array of shape ``(score_size, score_size, 2)``, float32.

    Raises:
        ValueError: If sizes/radii are not positive or ``neg`` does not
            exceed ``pos``.
    """
    if score_size <= 0:
        raise ValueError(f"score_size must be positive, got {score_size}")
    if pos_radius_px < 0 or neg_radius_px <= pos_radius_px:
        raise ValueError(
            f"need 0 <= pos ({pos_radius_px}) < neg ({neg_radius_px})"
        )
    if total_stride <= 0:
        raise ValueError(f"total_stride must be positive, got {total_stride}")
    pos_thr = pos_radius_px / total_stride
    neg_thr = neg_radius_px / total_stride
    center = (score_size - 1) / 2.0
    line = np.arange(score_size, dtype=np.float64) - center
    # dist_sq[i, j] over (row, col); built explicitly to keep axis order obvious.
    rr = np.repeat(line**2, score_size).reshape(score_size, score_size)
    dist_sq = rr + rr.T
    label = np.zeros((score_size, score_size, 2), dtype=np.float32)
    label[:, :, 0] = (dist_sq <= pos_thr**2).astype(np.float32)
    label[:, :, 1] = (
        (dist_sq <= pos_thr**2) | (dist_sq > neg_thr**2)
    ).astype(np.float32)
    return label


def _group_anchor_dim(
    y_pred: keras.KerasTensor, y_true: keras.KerasTensor, inner: int
) -> keras.KerasTensor:
    """Reshape a flat model output ``(B, S, S, A * inner)`` to grouped form.

    The anchor count is read off the packed ``y_true`` (``(B, S, S, A, *)``);
    both extents must be statically known, which the ``tf.data`` pipeline
    guarantees via its output signature.

    Args:
        y_pred: Flat logits/deltas ``(B, S, S, A * inner)``.
        y_true: Packed targets ``(B, S, S, A, *)``.
        inner: Trailing extent per anchor (2 for cls, 4 for reg).

    Returns:
        Grouped tensor ``(B, S, S, A, inner)``.

    Raises:
        ValueError: If ``S`` or ``A`` is not statically known.
    """
    grid = y_pred.shape[1]
    anchors = y_true.shape[3]
    if grid is None or anchors is None:
        raise ValueError(
            f"score extent and anchor count must be statically known, got "
            f"y_pred.shape={tuple(y_pred.shape)}, "
            f"y_true.shape={tuple(y_true.shape)}"
        )
    return keras.ops.reshape(y_pred, (-1, grid, grid, anchors, inner))


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.losses.siamese_tracking_loss")
class SiamFCLogisticLoss(keras.losses.Loss):
    """Logistic loss over the SiamFC response map with an ignored ring.

    **Intent**: Train the SiamFC similarity map so centered pairs peak at the
    map center. Each valid pixel contributes ``softplus(-y * v)`` with
    ``y = +1`` (positive) or ``-1`` (negative); ignored-ring pixels contribute
    nothing.

    **Formulation**:
    For response logits ``v`` and packed label ``(value, mask)``,

        L = sum(mask * softplus(-(2*value - 1) * v)) / max(sum(mask), 1)

    per sample. A fully-ignored sample (the reference's empty-frame label)
    yields exactly 0, never NaN.

    Args:
        name: Loss instance name. Defaults to ``"siamfc_logistic_loss"``.
        **kwargs: Forwarded to :class:`keras.losses.Loss` (e.g. ``reduction``).

    Input shapes:
        - ``y_true``: packed label ``(B, S, S, 2)`` from
          :func:`create_siamfc_label` (channel 0 value, channel 1 valid flag).
        - ``y_pred``: response LOGITS ``(B, S, S, 1)``.

    Output:
        Per-sample loss ``(B,)``.
    """

    def __init__(
        self,
        name: str = "siamfc_logistic_loss",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        logger.info("SiamFCLogisticLoss initialized (masked logistic loss from logits)")

    def call(
        self,
        y_true: keras.KerasTensor,
        y_pred: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """Compute the masked logistic response loss.

        Args:
            y_true: Packed label ``(B, S, S, 2)``.
            y_pred: Response logits ``(B, S, S, 1)``.

        Returns:
            Per-sample loss ``(B,)``.
        """
        value = y_true[..., 0]
        mask = y_true[..., 1]
        logits = y_pred[..., 0]
        signed = 2.0 * value - 1.0
        per_pixel = keras.ops.softplus(-signed * logits)
        valid_count = keras.ops.maximum(
            keras.ops.sum(mask, axis=(1, 2)), 1.0
        )
        return keras.ops.sum(per_pixel * mask, axis=(1, 2)) / valid_count

    def get_config(self) -> Dict[str, Any]:
        """Return the base config (no extra hyperparameters)."""
        return super().get_config()


@register_dl_technique("dl_techniques.losses.siamese_tracking_loss")
class DaSiamRPNClsLoss(keras.losses.Loss):
    """Per-anchor 2-way classification loss from logits with packed ignore mask.

    **Intent**: Train the RPN object/background head. Anchors matched
    positive/negative by the data pipeline contribute sparse softmax
    cross-entropy; ignored anchors (the IoU band between thresholds)
    contribute nothing.

    Args:
        name: Loss instance name. Defaults to ``"dasiamrpn_cls_loss"``.
        **kwargs: Forwarded to :class:`keras.losses.Loss` (e.g. ``reduction``).

    Input shapes:
        - ``y_true``: packed ``(B, S, S, A, 2)`` (channel 0: integer label
          0/1 as float, channel 1: valid weight 0/1).
        - ``y_pred``: raw LOGITS ``(B, S, S, A * 2)`` in the model output
          layout, reshaped to ``(B, S, S, A, 2)`` inside. ``S`` and ``A``
          must be statically known (the ``tf.data`` pipeline always provides
          them); otherwise a ``ValueError`` is raised.

    Output:
        Per-sample loss ``(B,)``.
    """

    def __init__(
        self,
        name: str = "dasiamrpn_cls_loss",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        logger.info("DaSiamRPNClsLoss initialized (packed sparse softmax CE from logits)")

    def call(
        self,
        y_true: keras.KerasTensor,
        y_pred: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """Compute the masked per-anchor classification loss.

        Args:
            y_true: Packed labels ``(B, S, S, A, 2)``.
            y_pred: Raw logits ``(B, S, S, A, 2)``.

        Returns:
            Per-sample loss ``(B,)``.
        """
        labels = keras.ops.cast(y_true[..., 0], "int32")
        weight = y_true[..., 1]
        # Ignored anchors carry label -1, which is out of range for the
        # sparse lookup and yields NaN (not an error) in graph mode -- and
        # 0 * NaN is still NaN. Clamp to the valid range first; the weight
        # (0 on ignored anchors) removes them from the mean regardless.
        safe_labels = keras.ops.maximum(labels, 0)
        logits = _group_anchor_dim(y_pred, y_true, 2)
        per_anchor = keras.losses.sparse_categorical_crossentropy(
            safe_labels, logits, from_logits=True
        )
        valid_count = keras.ops.maximum(
            keras.ops.sum(weight, axis=(1, 2, 3)), 1.0
        )
        return keras.ops.sum(per_anchor * weight, axis=(1, 2, 3)) / valid_count

    def get_config(self) -> Dict[str, Any]:
        """Return the base config (no extra hyperparameters)."""
        return super().get_config()


@register_dl_technique("dl_techniques.losses.siamese_tracking_loss")
class DaSiamRPNRegLoss(keras.losses.Loss):
    """Per-anchor smooth-L1 box-delta regression over positive anchors only.

    **Intent**: Train the RPN box head. Only anchors matched positive carry a
    regression target; every other anchor's weight is 0 and contributes
    nothing, including its (meaningless) delta values.

    Args:
        huber_delta: Smooth-L1 knee. Defaults to 1.0.
        name: Loss instance name. Defaults to ``"dasiamrpn_reg_loss"``.
        **kwargs: Forwarded to :class:`keras.losses.Loss` (e.g. ``reduction``).

    Input shapes:
        - ``y_true``: packed ``(B, S, S, A, 5)`` (channels 0-3: target
          ``(dx, dy, dw, dh)`` deltas, channel 4: positive weight 0/1).
        - ``y_pred``: predicted deltas ``(B, S, S, A * 4)`` in the model
          output layout, reshaped to ``(B, S, S, A, 4)`` inside. ``S`` and
          ``A`` must be statically known; otherwise a ``ValueError`` is
          raised.

    Output:
        Per-sample loss ``(B,)``.
    """

    def __init__(
        self,
        huber_delta: float = 1.0,
        name: str = "dasiamrpn_reg_loss",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        if huber_delta <= 0.0:
            raise ValueError(f"huber_delta must be positive, got {huber_delta}")
        self.huber_delta = huber_delta
        # The per-coordinate smooth-L1 math lives in HuberLoss (single source
        # for the piecewise formula); this class only adds the packed
        # positive-mask reduction the RPN layout needs.
        self._huber = HuberLoss(delta=huber_delta)
        logger.info(
            f"DaSiamRPNRegLoss initialized (smooth-L1, delta={huber_delta})"
        )

    def call(
        self,
        y_true: keras.KerasTensor,
        y_pred: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """Compute the masked smooth-L1 regression loss.

        Args:
            y_true: Packed targets ``(B, S, S, A, 5)``.
            y_pred: Predicted deltas ``(B, S, S, A, 4)``.

        Returns:
            Per-sample loss ``(B,)``.
        """
        target = y_true[..., :4]
        weight = y_true[..., 4]
        pred = _group_anchor_dim(y_pred, y_true, 4)
        # Static S/A per the documented contract (_group_anchor_dim enforces
        # it); only the batch stays dynamic.
        grid, anchors = target.shape[1], target.shape[3]
        per_anchor = self._huber.call(
            keras.ops.reshape(target, (-1, 4)), keras.ops.reshape(pred, (-1, 4))
        )
        per_anchor = keras.ops.reshape(per_anchor, (-1, grid, grid, anchors))
        valid_count = keras.ops.maximum(
            keras.ops.sum(weight, axis=(1, 2, 3)), 1.0
        )
        return keras.ops.sum(per_anchor * weight, axis=(1, 2, 3)) / valid_count

    def get_config(self) -> Dict[str, Any]:
        """Return the config with ``huber_delta``."""
        config = super().get_config()
        config.update({"huber_delta": self.huber_delta})
        return config
