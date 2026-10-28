"""
Probabilities to logits.

`ProbabilityLogits` maps a tensor of probabilities (or any scores in the open unit
interval) to their log-odds, ``log(p) - log(1 - p)``. It exists for classifiers whose
output is a per-class probability that is not a softmax, such as the capsule lengths of
a CapsNet: a layer that applies its own softmax to such an output (for example
:class:`~dl_techniques.layers.statistics.conformal.prediction_set.ConformalPredictionSet`)
sees a nearly uniform distribution when the inputs are all in ``[0, 1]``, because the
softmax of values that differ by at most 1 cannot be sharp. The log-odds spread the
same values over the real line and keep their order.
"""

import keras
from typing import Any, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.keras_registration import register_dl_technique


# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.statistics.probability_logits")
class ProbabilityLogits(keras.layers.Layer):
    """
    Element-wise log-odds of a tensor of probabilities.

    The output is ``log(p) - log1p(-p)`` with ``p`` first clipped to
    ``[epsilon, 1 - epsilon]``, so ``p`` equal to 0 or 1 gives the finite values
    ``-/+ log((1 - epsilon) / epsilon)`` instead of an infinity. The map is strictly
    increasing on the clipped range, so it never changes the order of the classes
    (inputs beyond the clip tie at the ends). The arithmetic runs in float32 whatever
    the policy, because ``1 - 1e-6`` rounds to exactly 1 in float16; the result is cast
    back to the input dtype. The layer has no weights.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────┐
        │ p  (any shape, values [0,1]) │
        └──────────────┬───────────────┘
                       ▼
        ┌──────────────────────────────┐
        │ cast to float32              │
        │ clip to [eps, 1 - eps]       │
        └──────────────┬───────────────┘
                       ▼
        ┌──────────────────────────────┐
        │ log(p) - log1p(-p)           │
        │ cast back to the input dtype │
        └──────────────┬───────────────┘
                       ▼
        ┌──────────────────────────────┐
        │ logits  (same shape as p)    │
        └──────────────────────────────┘

    :param epsilon: Distance from 0 and 1 at which the input is clipped. Must lie in
        ``(0, 0.5)``. Defaults to ``1e-6``.
    :type epsilon: float
    :param kwargs: Additional keyword arguments for the Layer base class.
    :type kwargs: Any

    :raises ValueError: If ``epsilon`` is not a number in ``(0, 0.5)``.

    Input shape:
        Any shape; values are expected in ``[0, 1]``.

    Output shape:
        The input shape.

    Example:
        >>> layer = ProbabilityLogits()
        >>> logits = layer(keras.ops.convert_to_tensor([[0.1, 0.5, 0.9]]))
    """

    def __init__(self, epsilon: float = 1e-6, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) \
                or not 0.0 < epsilon < 0.5:
            raise ValueError(
                f"epsilon must be a number in (0, 0.5), got {epsilon!r}"
            )
        self.epsilon = float(epsilon)

    def call(self, inputs: keras.KerasTensor, training: Any = None) -> keras.KerasTensor:
        """Return the log-odds of ``inputs``.

        :param inputs: Probabilities, any shape.
        :param training: Unused; present for the Keras call contract.
        :return: Log-odds with the shape and dtype of ``inputs``.
        """
        p = keras.ops.clip(
            keras.ops.cast(inputs, "float32"), self.epsilon, 1.0 - self.epsilon
        )
        logits = keras.ops.log(p) - keras.ops.log1p(-p)
        return keras.ops.cast(logits, keras.backend.standardize_dtype(inputs.dtype))

    def compute_output_shape(self, input_shape: Any) -> Any:
        """The output has the input's shape."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor arguments."""
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config
