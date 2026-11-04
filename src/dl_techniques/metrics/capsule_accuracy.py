"""
Accuracy metric for Capsule Networks.

In a standard neural network classifier, the model outputs a scalar score
(logit) for each class, which is typically converted to a probability
distribution via softmax. Standard accuracy metrics (like
:class:keras.metrics.CategoricalAccuracy) simply take the argmax over these
scalars.

In a Capsule Network (Sabour et al., 2017), the output for each class is not a
scalar, but an instantiation vector (a "capsule"). The probability that a
specific entity exists in the input is represented by the L2 norm (length) of
its corresponding capsule vector. Therefore, the predicted class is the one
whose capsule has the greatest length.

This metric differs from standard Keras accuracy in two ways:

1.  Mathematical definition: It explicitly computes the argmax over capsule
    vector lengths rather than logits or probabilities.
2.  Interface contract: Capsule layers frequently output a dictionary containing
    both the raw routing vectors and their computed lengths (to feed both
    reconstruction and margin losses). This metric seamlessly accepts that
    dictionary, extracting the "length" key to perform the accuracy check, which
    would cause a standard Keras metric to crash with a type error.

References: - Sabour, S., Frosst, N., & Hinton, G. E. (2017). Dynamic Routing
Between Capsules. NeurIPS 2017. (https://arxiv.org/abs/1710.09829)
"""

import keras
from keras import ops
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.metrics.capsule_accuracy")
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
