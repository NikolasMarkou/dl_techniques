"""
``ConformalPredictionSet`` turns classifier logits into a conformal prediction set
with a fixed, non-trainable threshold.

Split conformal prediction attaches a coverage guarantee to any classifier: the set of
labels whose Adaptive-Prediction-Sets score is at most a threshold ``q_hat`` contains
the true label with probability at least ``1 - alpha`` when ``q_hat`` is the
finite-sample ``1 - alpha`` quantile of the true labels' scores on held-out data.
``q_hat`` is fit host-side (:func:`dl_techniques.utils.conformal_classification.calibrate_aps`,
or :meth:`~ConformalLayer.calibrate` on the APS scores) and assigned into a frozen
weight, so the forward pass is a fixed function of the logits: the probabilities are
sorted in descending order, and a label is in the set iff the cumulative probability up
to and including it is at most ``q_hat``. That is the
deterministic (``u = 1``) form of APS. It is valid, a little conservative, and, unlike
the randomized form, needs no random draw at inference.

Appending the layer to a trained classifier gives a single Functional model that exports
a calibrated set predictor as one ``.keras`` file
(:func:`attach_conformal_sets`). The coverage is marginal, and it needs calibration and
deployment data to be exchangeable; see the utility module for the caveats.

References:
    - Romano, Sesia and Candes, 2020. Classification with Valid and Adaptive
      Coverage. NeurIPS. (https://arxiv.org/abs/2006.02544)
    - Angelopoulos and Bates, 2021. A Gentle Introduction to Conformal
      Prediction and Distribution-Free Uncertainty Quantification.
      (https://arxiv.org/abs/2107.07511)

"""

from typing import Any, Optional, Tuple

import keras

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.keras_registration import register_dl_technique
from .base import ConformalLayer

# ---------------------------------------------------------------------

#: Any ``q_hat`` at or above this keeps every label (the inclusive mass is at most
#: 1 plus float rounding), so an infinite calibrated ``q_hat`` is stored as this value.
_Q_HAT_KEEPS_EVERY_LABEL = 2.0


@register_dl_technique("dl_techniques.layers.statistics.conformal.prediction_set")
class ConformalPredictionSet(ConformalLayer):
    """Non-trainable, fixed-threshold conformal prediction-set layer.

    Holds the single frozen scalar ``q_hat`` (the APS threshold). Its ``call(logits)``:

    1. takes the softmax of the logits (in float32);
    2. sorts the probabilities in descending order (ties by class index);
    3. keeps a label iff the cumulative probability through it is at most ``q_hat``;
    4. returns ``(probs, in_set)``, with ``in_set`` a float 0/1 mask of shape ``(B, C)``
       in the original class order.

    It matches :func:`dl_techniques.utils.conformal_classification.predict_sets` with
    ``randomized=False``, to float32 rounding at the threshold. Fit it with
    :meth:`~ConformalLayer.calibrate` on the scores of
    :func:`dl_techniques.utils.conformal_classification.aps_scores` (``randomized=False``),
    or assign a value from
    :func:`~dl_techniques.utils.conformal_classification.calibrate_aps` with
    :meth:`~ConformalLayer.set_q_hat`. It is the classification sibling of
    :class:`~dl_techniques.layers.statistics.conformal.interval.ConformalInterval` and
    :class:`~dl_techniques.layers.statistics.conformal.quantile_head.ConformalQuantileHead`.

    :param q_hat_init: Initial threshold. Defaults to ``2.0``, which is uncalibrated:
        every label is in every set. Values above ``2.0`` (and ``+inf``) are stored as
        ``2.0``, which already keeps every label.
    :type q_hat_init: float
    :param kwargs: Additional keyword arguments for the Layer base class.
        ``trainable`` is forced to ``False``.

    Input shape:
        ``(batch, num_classes)`` logits.

    Output shape:
        A tuple ``(probs, in_set)``, both ``(batch, num_classes)``.

    :raises ValueError: If ``q_hat_init`` is negative or NaN.

    Example:
        >>> layer = ConformalPredictionSet()
        >>> layer(keras.ops.zeros((2, 5)))  # builds the layer
        >>> layer.set_q_hat(0.9)
        >>> probs, in_set = layer(keras.ops.zeros((2, 5)))
    """

    FROZEN = True

    def __init__(self, q_hat_init: float = _Q_HAT_KEEPS_EVERY_LABEL, **kwargs: Any) -> None:
        super().__init__(q_hat_init=q_hat_init, **kwargs)

    def _normalize_q_hat(self, value: float, argument: str = "q_hat") -> float:
        return min(super()._normalize_q_hat(value, argument), _Q_HAT_KEEPS_EVERY_LABEL)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the frozen threshold weight.

        :param input_shape: Shape of the incoming logits, ``(batch, num_classes)``.
        :raises ValueError: If the logits are not rank 2 or have fewer than 2 classes.
        """
        if len(input_shape) != 2:
            raise ValueError(
                f"ConformalPredictionSet takes (batch, num_classes) logits, "
                f"got shape {tuple(input_shape)}"
            )
        if input_shape[-1] is not None and input_shape[-1] < 2:
            raise ValueError(
                f"ConformalPredictionSet needs at least 2 classes, got "
                f"{input_shape[-1]}"
            )
        self.add_q_hat_weight()
        super().build(input_shape)

    def call(
            self,
            logits: keras.KerasTensor,
            training: Optional[bool] = None,
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Return the class probabilities and the 0/1 prediction-set mask.

        :param logits: Classifier logits ``(batch, num_classes)``.
        :param training: Unused; present for the Keras call contract.
        :return: ``(probs, in_set)``, both ``(batch, num_classes)`` float32.
        """
        probs = keras.ops.softmax(keras.ops.cast(logits, "float32"), axis=-1)
        order = keras.ops.argsort(-probs, axis=-1)
        sorted_probs = keras.ops.take_along_axis(probs, order, axis=-1)
        # u = 1: the score of a label is the mass through it, inclusive. The comparison
        # runs in float32 whatever the policy (q_hat is never autocast).
        q_hat = keras.ops.cast(self.q_hat, "float32")
        in_sorted = keras.ops.cast(
            keras.ops.cumsum(sorted_probs, axis=-1) <= q_hat, "float32"
        )
        inverse = keras.ops.argsort(order, axis=-1)
        return probs, keras.ops.take_along_axis(in_sorted, inverse, axis=-1)

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...],
    ) -> Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]:
        """Both outputs have the logits' shape."""
        return input_shape, input_shape


# ---------------------------------------------------------------------


def attach_conformal_sets(
        model: keras.Model,
        tau: float,
        input_shape: Tuple[int, ...],
) -> keras.Model:
    """Build a model that maps an input to ``(logits, probs, in_set)``.

    Works for a Functional or a subclassed ``model``: it is called on a symbolic input.

    :param model: A trained single-output classifier that emits logits. A
        deep-supervised model must be reduced to its primary output first
        (``dl_techniques.utils.deep_supervision.create_inference_model_from_training_model``).
    :param tau: Calibrated threshold, applied with :meth:`ConformalLayer.set_q_hat`.
    :param input_shape: Per-sample input shape, without the batch axis.
    :return: A Functional model sharing ``model``'s weights.
    :raises ValueError: If ``model`` does not return a single logits tensor.
    """
    inputs = keras.Input(tuple(input_shape))
    logits = model(inputs)
    if isinstance(logits, (list, tuple, dict)):
        raise ValueError(
            "attach_conformal_sets needs a single-output model, got "
            f"{type(logits).__name__}"
        )
    layer = ConformalPredictionSet(name="conformal_prediction_set")
    probs, in_set = layer(logits)
    layer.set_q_hat(tau)
    return keras.Model(inputs, [logits, probs, in_set], name=f"{model.name}_conformal")
