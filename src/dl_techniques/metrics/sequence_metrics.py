"""
Sequence Metrics
================

Metrics for evaluating sequence-level predictions, commonly used in
memory-augmented neural networks, sequence-to-sequence models, and
binary sequence tasks.

Example::

    from dl_techniques.metrics.sequence_metrics import (
        SequenceAccuracy,
        BitErrorRate,
    )

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[SequenceAccuracy(), BitErrorRate()],
    )
"""

import keras
from keras import ops
from typing import Any, Dict, Optional


@keras.saving.register_keras_serializable(package="dl_techniques.metrics")
class SequenceAccuracy(keras.metrics.Metric):
    """Sequence-level accuracy metric.

    Measures the fraction of complete sequences that are exactly correct.
    A sequence is considered correct only if **all** elements match after
    binarization with the given threshold.

    Args:
        threshold: Threshold for converting continuous predictions to
            binary values.
        name: Metric name for logging.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        name: str = "sequence_accuracy",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.correct = self.add_weight(name="correct", initializer="zeros")
        self.total = self.add_weight(name="total", initializer="zeros")

    def update_state(
        self,
        y_true: keras.KerasTensor,
        y_pred: keras.KerasTensor,
        sample_weight: Optional[keras.KerasTensor] = None,
    ) -> None:
        """Update metric state with new predictions.

        Args:
            y_true: Ground truth sequences.
            y_pred: Predicted sequences (continuous values).
            sample_weight: Optional per-sample weights.
        """
        y_pred_binary = ops.cast(y_pred > self.threshold, y_true.dtype)

        # Check if entire sequence matches: reduce all dims except batch
        matches = ops.all(
            ops.equal(y_true, y_pred_binary),
            axis=tuple(range(1, len(ops.shape(y_true)))),
        )

        if sample_weight is not None:
            matches_float = ops.cast(matches, "float32")
            sample_weight = ops.cast(sample_weight, "float32")
            self.correct.assign_add(ops.sum(matches_float * sample_weight))
            self.total.assign_add(ops.sum(sample_weight))
        else:
            self.correct.assign_add(
                ops.cast(ops.sum(matches), self.correct.dtype)
            )
            self.total.assign_add(
                ops.cast(ops.shape(y_true)[0], self.total.dtype)
            )

    def result(self) -> keras.KerasTensor:
        """Compute final sequence accuracy."""
        return self.correct / (self.total + keras.backend.epsilon())

    def reset_state(self) -> None:
        self.correct.assign(0.0)
        self.total.assign(0.0)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"threshold": self.threshold})
        return config


@keras.saving.register_keras_serializable(package="dl_techniques.metrics")
class BitErrorRate(keras.metrics.Metric):
    """Bit Error Rate (BER) metric for binary vector outputs.

    Measures the fraction of individual bits that differ between prediction
    and ground truth after thresholding. Lower is better.

    Args:
        threshold: Threshold for binarizing continuous predictions.
        name: Metric name for logging.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        name: str = "bit_error_rate",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.errors = self.add_weight(name="errors", initializer="zeros")
        self.total = self.add_weight(name="total", initializer="zeros")

    def update_state(
        self,
        y_true: keras.KerasTensor,
        y_pred: keras.KerasTensor,
        sample_weight: Optional[keras.KerasTensor] = None,
    ) -> None:
        """Update metric state.

        Args:
            y_true: Ground truth binary values.
            y_pred: Predicted values (continuous).
            sample_weight: Optional per-sample weights (unused).
        """
        y_pred_binary = ops.cast(y_pred > self.threshold, y_true.dtype)
        bit_errors = ops.cast(ops.not_equal(y_true, y_pred_binary), "float32")

        self.errors.assign_add(ops.sum(bit_errors))
        self.total.assign_add(ops.cast(ops.size(y_true), self.total.dtype))

    def result(self) -> keras.KerasTensor:
        """Compute bit error rate."""
        return self.errors / (self.total + keras.backend.epsilon())

    def reset_state(self) -> None:
        self.errors.assign(0.0)
        self.total.assign(0.0)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"threshold": self.threshold})
        return config
