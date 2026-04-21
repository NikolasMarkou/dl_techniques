"""Brier Score metrics for Keras 3.

A proper scoring rule for probabilistic classification: the mean squared error
between predicted probabilities and one-hot / binary targets.  Complements
cross-entropy loss (used for training) with a metric that explicitly rewards
calibration — being confident *and* correct.

See :doc:`../../../research/brier_score.md` for the full treatment (proper
scoring rule proof, Murphy decomposition, sample-size caveats, and when the
Brier Score complements accuracy / AUC / F1).

Two public metrics:

- :class:`BrierScore` — binary / multi-label (sigmoid or probability inputs).
  Range ``[0, 1]``.  "Always-predict-0.5" baseline = 0.25.
- :class:`CategoricalBrierScore` — multi-class (softmax or probability inputs),
  accepts sparse integer labels or one-hot labels.  Range ``[0, 2]``.
  Uses a memory-efficient sparse path for segmentation-style workloads.

Both metrics:

- Inherit from :class:`keras.metrics.Metric`.
- Are ``@keras.saving.register_keras_serializable()``.
- Implement the full contract: ``update_state``, ``result``, ``reset_state``,
  ``get_config``.
- Accept ``sample_weight``.
- Use ``keras.backend.epsilon()`` for numeric stability of the ``result``
  denominator.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import keras

# ---------------------------------------------------------------------------
# BrierScore — binary / multi-label
# ---------------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class BrierScore(keras.metrics.Metric):
    """Binary or multi-label Brier Score.

    For binary classification with a single output per sample, computes

    .. math::

        BS = \\frac{1}{N} \\sum_{i=1}^{N} (p_i - o_i)^2

    For multi-label classification with ``K`` independent binary outputs per
    sample, computes the mean over all ``N × K`` elements (range ``[0, 1]``).

    :param from_logits: If ``True`` (default), inputs are raw logits and a
        ``sigmoid`` is applied internally.  If ``False``, inputs are assumed
        to already be probabilities in ``[0, 1]``.
    :param name: Metric name, default ``"brier_score"``.
    :param dtype: Metric state dtype; defaults to ``float32``.

    .. code-block:: python

        # Multi-label COCO classification (80 classes, sigmoid / BCE).
        model.compile(
            optimizer=...,
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=[
                BrierScore(from_logits=True, name="brier"),
                # Lower is better.  "Always-negative" baseline on COCO
                # (avg 4.5 / 80 positives) ≈ 0.056.
            ],
        )
    """

    def __init__(
        self,
        from_logits: bool = True,
        name: str = "brier_score",
        dtype: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.from_logits = bool(from_logits)
        self.sum_squared_error = self.add_weight(
            name="sum_squared_error", initializer="zeros"
        )
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = keras.ops.cast(y_true, self.dtype or "float32")
        y_pred = keras.ops.cast(y_pred, self.dtype or "float32")
        if self.from_logits:
            y_pred = keras.ops.sigmoid(y_pred)
        sq = keras.ops.square(y_pred - y_true)

        if sample_weight is not None:
            sample_weight = keras.ops.cast(
                sample_weight, self.dtype or "float32"
            )
            # Broadcast per-sample weights across the class/feature dims.
            # sample_weight may be (B,), (B, K), or any broadcastable shape.
            sq_shape = keras.ops.shape(sq)
            sw_shape = keras.ops.shape(sample_weight)
            # Ensure broadcasting: add trailing dims to sample_weight if needed.
            while len(sw_shape) < len(sq_shape):
                sample_weight = keras.ops.expand_dims(sample_weight, axis=-1)
                sw_shape = keras.ops.shape(sample_weight)
            weighted_sq = sq * sample_weight
            # Effective element count = sum(sample_weight broadcast to sq shape)
            broadcast_w = keras.ops.broadcast_to(sample_weight, sq_shape)
            total = keras.ops.sum(broadcast_w)
            self.sum_squared_error.assign_add(keras.ops.sum(weighted_sq))
            self.count.assign_add(total)
        else:
            self.sum_squared_error.assign_add(keras.ops.sum(sq))
            self.count.assign_add(
                keras.ops.cast(keras.ops.size(y_true), self.dtype or "float32")
            )

    def result(self):
        return self.sum_squared_error / (self.count + keras.backend.epsilon())

    def reset_state(self):
        self.sum_squared_error.assign(0.0)
        self.count.assign(0.0)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"from_logits": self.from_logits})
        return config


# ---------------------------------------------------------------------------
# CategoricalBrierScore — multi-class (supports sparse integer labels)
# ---------------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class CategoricalBrierScore(keras.metrics.Metric):
    """Multi-class Brier Score with optional sparse-label fast path.

    Per-sample Brier:

    .. math::

        BS_i = \\sum_{k=1}^{K} (p_{ik} - o_{ik})^2, \\quad
        BS = \\frac{1}{N} \\sum_i BS_i

    Range: ``[0, 2]`` (per the Brier-score convention for multi-class).

    Supports both dense one-hot targets and sparse integer targets.  For
    sparse targets with one-hot expansion being prohibitive (e.g. semantic
    segmentation at H×W×K), uses the algebraic identity

    .. math::

        \\sum_k (p_k - o_k)^2 = \\|p\\|^2 - 2 p_c + 1,
        \\qquad c = \\text{index of the true class},

    so the full one-hot tensor is never materialized.

    :param from_logits: If ``True`` (default), ``y_pred`` is softmax-applied
        internally.  If ``False``, ``y_pred`` must already be a probability
        simplex on the last axis.
    :param sparse_labels: If ``True``, ``y_true`` is treated as integer class
        indices (shape broadcasts to ``y_pred`` sans last axis).  If ``False``
        (default), ``y_true`` must be one-hot with the same shape as
        ``y_pred``.
    :param axis: Class axis on ``y_pred``.  Default ``-1``.
    :param name: Metric name.
    :param dtype: Metric state dtype.

    .. code-block:: python

        # Semantic segmentation, sparse integer masks:
        CategoricalBrierScore(from_logits=True, sparse_labels=True, name="brier")
    """

    def __init__(
        self,
        from_logits: bool = True,
        sparse_labels: bool = False,
        axis: int = -1,
        name: str = "categorical_brier_score",
        dtype: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.from_logits = bool(from_logits)
        self.sparse_labels = bool(sparse_labels)
        self.axis = int(axis)
        self.sum_brier = self.add_weight(name="sum_brier", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        fdtype = self.dtype or "float32"
        y_pred = keras.ops.cast(y_pred, fdtype)
        if self.from_logits:
            y_pred = keras.ops.softmax(y_pred, axis=self.axis)

        if self.sparse_labels:
            # Sparse fast path: BS_i = ||p_i||^2 - 2 * p_{i,c_i} + 1
            y_true_int = keras.ops.cast(y_true, "int32")
            sum_sq_p = keras.ops.sum(
                keras.ops.square(y_pred), axis=self.axis
            )
            p_correct = keras.ops.take_along_axis(
                y_pred,
                keras.ops.expand_dims(y_true_int, axis=self.axis),
                axis=self.axis,
            )
            p_correct = keras.ops.squeeze(p_correct, axis=self.axis)
            per_sample = sum_sq_p - 2.0 * p_correct + 1.0
        else:
            y_true_f = keras.ops.cast(y_true, fdtype)
            per_sample = keras.ops.sum(
                keras.ops.square(y_pred - y_true_f), axis=self.axis
            )

        if sample_weight is not None:
            sample_weight = keras.ops.cast(sample_weight, fdtype)
            ps_shape = keras.ops.shape(per_sample)
            sw_shape = keras.ops.shape(sample_weight)
            # Broadcast per-sample weights if shapes don't match.
            while len(sw_shape) < len(ps_shape):
                sample_weight = keras.ops.expand_dims(sample_weight, axis=-1)
                sw_shape = keras.ops.shape(sample_weight)
            broadcast_w = keras.ops.broadcast_to(sample_weight, ps_shape)
            self.sum_brier.assign_add(keras.ops.sum(per_sample * broadcast_w))
            self.count.assign_add(keras.ops.sum(broadcast_w))
        else:
            self.sum_brier.assign_add(keras.ops.sum(per_sample))
            self.count.assign_add(
                keras.ops.cast(keras.ops.size(per_sample), fdtype)
            )

    def result(self):
        return self.sum_brier / (self.count + keras.backend.epsilon())

    def reset_state(self):
        self.sum_brier.assign(0.0)
        self.count.assign(0.0)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "from_logits": self.from_logits,
                "sparse_labels": self.sparse_labels,
                "axis": self.axis,
            }
        )
        return config
