import keras
from keras import ops
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class CapsuleAccuracy(keras.metrics.Metric):
    """Custom accuracy metric for capsule networks based on capsule lengths.

    Args:
        name: Name of the metric.
        **kwargs: Additional keyword arguments passed to parent Metric class.
    """

    def __init__(self, name: str = "capsule_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(
        self,
        y_true: keras.KerasTensor,
        y_pred: keras.KerasTensor,
        sample_weight: Optional[keras.KerasTensor] = None,
    ):
        """Update accuracy state based on capsule lengths.

        Args:
            y_true: One-hot encoded true labels.
            y_pred: Dictionary containing 'length' key with capsule lengths,
                or a tensor of capsule lengths directly.
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

    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary for serialization."""
        return super().get_config()


# ---------------------------------------------------------------------
