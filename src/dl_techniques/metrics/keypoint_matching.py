"""Match precision, recall and F1 for learned keypoint matchers.

Scores the hard matches that :func:`filter_matches` extracts from the final-layer log
assignment of a LightGlue-style matcher against the homography ground truth of
:mod:`dl_techniques.utils.keypoint_matching`. The labels use the packing of
:func:`dl_techniques.losses.lightglue_loss.pack_matches` (``(B, M + N)`` int, ``j >= 0``
match, ``-1`` dustbin, ``-2`` ignored or padded), so the same ``y_true`` feeds
:class:`LightGlueLoss` and this metric.
"""

import keras
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.layers.matching.match_assignment import filter_matches
from dl_techniques.losses.lightglue_loss import unpack_matches

# ---------------------------------------------------------------------

_MODES = ("precision", "recall", "f1")


@register_dl_technique("dl_techniques.metrics.keypoint_matching")
class KeypointMatchMetric(keras.metrics.Metric):
    """Streaming precision, recall or F1 of mutual-nearest keypoint matches.

    **Counting rule**. A predicted match ``(i, j)`` comes from
    ``filter_matches(final_layer_log_scores, threshold)``. It is a true positive when
    ``labels0[i] == j``. It is not a prediction at all when ``labels0[i] == -2`` or
    ``labels1[j] == -2`` (ignored or padded), so it enters neither the numerator nor the
    denominator. Ground-truth positives are the entries with ``labels0 >= 0``.

    * precision = TP / predicted, recall = TP / ground truth, F1 = 2 TP / (predicted +
      ground truth). An empty denominator gives 0, never NaN.

    **Interface contract**. ``update_state(y_true, y_pred, sample_weight=None, *,
    mask0=None, mask1=None)``.

    * ``y_true``: ``(B, M + N)`` packed labels (int or float, rounded and cast to int32).
    * ``y_pred``: ``log_assignments`` ``(B, L, M+1, N+1)`` (the last layer is scored) or
      one layer ``(B, M+1, N+1)``. Same tensor :class:`LightGlueLoss` receives, so
      ``compile(metrics={"log_assignments": KeypointMatchMetric(...)})`` works with stock
      ``fit``. M and N must be static.
    * ``sample_weight``: optional ``(B,)`` per-image weight on the counts.
    * ``mask0`` ``(B, M)``, ``mask1`` ``(B, N)``: optional real-keypoint masks passed on to
      ``filter_matches``. ``compile`` cannot supply them; a padded batch scored without
      them lets padded slots (log score 0 in the model output) compete for partners, so
      pass them when calling directly from a pipeline.

    :param mode: ``'precision'``, ``'recall'`` or ``'f1'``.
    :param threshold: Match threshold on ``exp(score)`` (LightGlue default 0.1).
    :param name: Metric name; defaults to ``mode``.
    :param dtype: Metric dtype (state is float32 whatever the policy).
    :raises ValueError: Unknown ``mode`` or ``threshold`` outside ``[0, 1]``.
    """

    def __init__(
        self,
        mode: str = "precision",
        threshold: float = 0.1,
        name: Optional[str] = None,
        dtype: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        super().__init__(name=name or f"match_{mode}", dtype=dtype, **kwargs)
        self.mode = mode
        self.threshold = float(threshold)
        self.true_positives = self.add_variable(
            shape=(), initializer="zeros", dtype="float32", name="true_positives")
        self.predicted = self.add_variable(
            shape=(), initializer="zeros", dtype="float32", name="predicted")
        self.ground_truth = self.add_variable(
            shape=(), initializer="zeros", dtype="float32", name="ground_truth")

    def update_state(
        self,
        y_true: Any,
        y_pred: Any,
        sample_weight: Optional[Any] = None,
        *,
        mask0: Optional[Any] = None,
        mask1: Optional[Any] = None,
    ) -> None:
        """Accumulate the three counts for one batch (see the class docstring)."""
        if len(y_pred.shape) == 4:
            y_pred = y_pred[:, -1]
        if len(y_pred.shape) != 3 or y_pred.shape[1] is None or y_pred.shape[2] is None:
            raise ValueError(
                "y_pred must be (B, [L,] M+1, N+1) with static M, N, got "
                f"{tuple(y_pred.shape)}")
        m = y_pred.shape[1] - 1
        labels0, labels1 = unpack_matches(y_true, m)
        pred0, _, _, _ = filter_matches(
            keras.ops.cast(y_pred, "float32"), self.threshold, mask0, mask1)
        n = y_pred.shape[2] - 1
        # labels1 at the predicted partner; clip keeps the -1 (no match) gather legal
        partner = keras.ops.clip(pred0, 0, n - 1)
        partner_label = keras.ops.take_along_axis(labels1, partner, axis=1)
        is_pred = keras.ops.logical_and(pred0 >= 0, labels0 != -2)
        is_pred = keras.ops.logical_and(is_pred, partner_label != -2)
        is_tp = keras.ops.logical_and(is_pred, labels0 == pred0)
        is_gt = labels0 >= 0

        def count(flags: Any) -> Any:
            per_image = keras.ops.sum(keras.ops.cast(flags, "float32"), axis=1)
            if sample_weight is not None:
                weight = keras.ops.cast(keras.ops.reshape(sample_weight, (-1,)), "float32")
                per_image = per_image * weight
            return keras.ops.sum(per_image)

        self.true_positives.assign_add(count(is_tp))
        self.predicted.assign_add(count(is_pred))
        self.ground_truth.assign_add(count(is_gt))

    def result(self) -> Any:
        """Current precision, recall or F1; 0 when the denominator is 0."""
        tp, pred, gt = self.true_positives, self.predicted, self.ground_truth
        if self.mode == "precision":
            num, den = tp, pred
        elif self.mode == "recall":
            num, den = tp, gt
        else:
            num, den = 2.0 * tp, pred + gt
        return keras.ops.where(den > 0, num / keras.ops.maximum(den, 1.0), 0.0)

    def reset_state(self) -> None:
        """Zero the three counters."""
        for var in (self.true_positives, self.predicted, self.ground_truth):
            var.assign(keras.ops.zeros_like(var))

    def get_config(self) -> Dict[str, Any]:
        """Return the full constructor configuration."""
        config = super().get_config()
        config.update({"mode": self.mode, "threshold": self.threshold})
        return config
