import keras
from keras import ops
import tensorflow as tf
from typing import Optional

# ---------------------------------------------------------------------

class CapsuleAccuracy(keras.metrics.Metric):
    """Custom accuracy metric for capsule networks based on capsule lengths."""

    def __init__(self, name: str = "capsule_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight: Optional[tf.Tensor] = None):
        """Update accuracy state based on capsule lengths.

        Args:
            y_true: One-hot encoded true labels.
            y_pred: Dictionary containing 'length' key with capsule lengths.
            sample_weight: Optional sample weights.
        """
        if isinstance(y_pred, dict) and "length" in y_pred:
            lengths = y_pred["length"]
        else:
            lengths = y_pred

        y_true_classes = ops.argmax(y_true, axis=1)
        y_pred_classes = ops.argmax(lengths, axis=1)

        matches = ops.cast(ops.equal(y_true_classes, y_pred_classes), self.dtype)

        if sample_weight is not None:
            sample_weight = ops.cast(sample_weight, self.dtype)
            matches = ops.multiply(matches, sample_weight)

        self.total.assign_add(ops.sum(matches))
        self.count.assign_add(ops.cast(ops.size(matches), self.dtype))

    def result(self):
        return ops.divide_no_nan(self.total, self.count)

    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)

# ---------------------------------------------------------------------
