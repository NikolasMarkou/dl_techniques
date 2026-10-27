"""
``ConformalPredictionSetLayer`` turns classifier logits into a conformal prediction set
with a fixed, non-trainable threshold.

Split conformal prediction attaches a coverage guarantee to any classifier: the set of
labels whose Adaptive-Prediction-Sets score is at most a threshold ``tau`` contains the
true label with probability at least ``1 - alpha`` when ``tau`` is the finite-sample
``1 - alpha`` quantile of the true labels' scores on held-out data. ``tau`` is fit
host-side (:func:`dl_techniques.utils.conformal_classification.calibrate_aps`) and
assigned into a frozen weight, so the forward pass is a fixed function of the logits:
the probabilities are sorted in descending order, and a label is in the set iff the
cumulative probability up to and including it is at most ``tau``. That is the
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

from typing import Any, Dict, Optional, Tuple

import keras

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

#: Any ``tau`` at or above this keeps every label (the inclusive mass is at most
#: 1 plus float rounding), so an infinite calibrated ``tau`` is stored as this value.
_TAU_KEEPS_EVERY_LABEL = 2.0


@register_dl_technique("dl_techniques.layers.statistics.conformal_prediction_set")
class ConformalPredictionSetLayer(keras.layers.Layer):
    """Non-trainable, fixed-threshold conformal prediction-set layer.

    Holds a single non-trainable scalar weight ``tau``. Its ``call(logits)``:

    1. takes the softmax of the logits (in float32);
    2. sorts the probabilities in descending order (ties by class index);
    3. keeps a label iff the cumulative probability through it is at most ``tau``;
    4. returns ``(probs, in_set)``, with ``in_set`` a float 0/1 mask of shape ``(B, C)``
       in the original class order.

    It matches :func:`dl_techniques.utils.conformal_classification.predict_sets` with
    ``randomized=False``, to float32 rounding at the threshold.

    :param tau_init: Initial threshold. Defaults to ``2.0``, which is uncalibrated:
        every label is in every set. The real value is fit host-side and either
        passed here or applied later via :meth:`calibrate`.
    :type tau_init: float
    :param kwargs: Additional keyword arguments for the Layer base class.
        ``trainable`` is forced to ``False``.

    Input shape:
        ``(batch, num_classes)`` logits.

    Output shape:
        A tuple ``(probs, in_set)``, both ``(batch, num_classes)``.

    Example:
        >>> layer = ConformalPredictionSetLayer()
        >>> layer.calibrate(0.9)
        >>> probs, in_set = layer(keras.ops.zeros((2, 5)))
    """

    def __init__(self, tau_init: float = _TAU_KEEPS_EVERY_LABEL, **kwargs: Any) -> None:
        # The whole layer is non-trainable: no gradient ever reaches tau. Drop an
        # inbound `trainable` (Keras round-trips it through the config) so it does not
        # collide with the explicit trainable=False below.
        kwargs.pop("trainable", None)
        super().__init__(trainable=False, **kwargs)
        self.tau_init = self._clamp(tau_init)

        # Created in build().
        self.tau = None

    @staticmethod
    def _clamp(tau: float) -> float:
        tau = float(tau)
        if tau != tau:
            raise ValueError(f"tau must not be NaN, got {tau}")
        return min(tau, _TAU_KEEPS_EVERY_LABEL)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the non-trainable scalar threshold weight.

        :param input_shape: Shape of the incoming logits, ``(batch, num_classes)``.
        :raises ValueError: If the logits are not rank 2.
        """
        if len(input_shape) != 2:
            raise ValueError(
                f"ConformalPredictionSetLayer takes (batch, num_classes) logits, "
                f"got shape {tuple(input_shape)}"
            )
        self.tau = self.add_weight(
            name="conformal_tau",
            shape=(),
            initializer=keras.initializers.Constant(self.tau_init),
            trainable=False,
            dtype="float32",
        )
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
        # u = 1: the score of a label is the mass through it, inclusive.
        in_sorted = keras.ops.cast(
            keras.ops.cumsum(sorted_probs, axis=-1) <= self.tau, "float32"
        )
        inverse = keras.ops.argsort(order, axis=-1)
        return probs, keras.ops.take_along_axis(in_sorted, inverse, axis=-1)

    def calibrate(self, tau: float) -> None:
        """Assign the fitted threshold into the frozen weight.

        The host-side fit is
        :func:`dl_techniques.utils.conformal_classification.calibrate_aps` with
        ``randomized=False``. An infinite ``tau`` (a calibration set too small for the
        target ``alpha``) is stored as a value that keeps every label.

        :param tau: The fitted threshold.
        :raises ValueError: If ``tau`` is NaN.
        """
        value = self._clamp(tau)
        self.tau.assign(value)
        logger.info(f"ConformalPredictionSetLayer '{self.name}' calibrated tau={value:.6g}")

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...],
    ) -> Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]:
        """Both outputs have the logits' shape."""
        return input_shape, input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return the serialization config.

        Stores the current calibrated ``tau``, not just the construction-time
        ``tau_init``, so a config-only reload path (``from_config`` /
        ``clone_model``) recovers the fitted threshold even without the ``.keras``
        weights archive.
        """
        config = super().get_config()
        config.update({
            "tau_init": float(keras.ops.convert_to_numpy(self.tau))
            if self.tau is not None else self.tau_init,
        })
        return config


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
    :param tau: Calibrated threshold, applied with :meth:`ConformalPredictionSetLayer.calibrate`.
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
    layer = ConformalPredictionSetLayer(name="conformal_prediction_set")
    probs, in_set = layer(logits)
    layer.calibrate(tau)
    return keras.Model(inputs, [logits, probs, in_set], name=f"{model.name}_conformal")
