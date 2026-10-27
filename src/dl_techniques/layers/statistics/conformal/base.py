"""
``ConformalLayer`` is the base of the split-conformal layers.

Split conformal prediction attaches a coverage guarantee to any predictor: a held-out
calibration set yields scores (how wrong the predictor was), and the finite-sample
quantile ``q_hat`` of those scores is the size of the correction that makes the
predictor's output cover the truth with probability at least ``1 - alpha``. What differs
between the layers is only how ``q_hat`` is applied:

* :class:`~dl_techniques.layers.statistics.conformal.interval.ConformalInterval` widens a
  point estimate by ``q_hat``;
* :class:`~dl_techniques.layers.statistics.conformal.quantile_head.ConformalQuantileHead`
  widens learned quantiles by ``q_hat``;
* :class:`~dl_techniques.layers.statistics.conformal.prediction_set.ConformalPredictionSet`
  keeps the classes whose cumulative probability is at most ``q_hat``.

This base owns what they share: one frozen scalar weight ``q_hat``, its validation, the
host-side fit :meth:`ConformalLayer.calibrate` (the exact order statistic, see
:func:`dl_techniques.utils.conformal_denoiser_intervals.conformal_quantile`), the
assignment :meth:`ConformalLayer.set_q_hat` and the rule that ``get_config`` stores the
CURRENT calibrated value so a config-only reload keeps it.

References:
    - Vovk et al., 2005. Algorithmic Learning in a Random World. Springer.
    - Angelopoulos and Bates, 2021. A Gentle Introduction to Conformal
      Prediction and Distribution-Free Uncertainty Quantification.
      (https://arxiv.org/abs/2107.07511)

"""

import keras
import numpy as np
from typing import Any, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.conformal_denoiser_intervals import conformal_quantile
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.statistics.conformal.base")
class ConformalLayer(keras.layers.Layer):
    """Base of the layers that apply a calibrated conformal quantile ``q_hat``.

    Subclasses create ``q_hat`` in their own ``build`` with :meth:`add_q_hat_weight`
    and use it in ``call``. The weight is declared with ``autocast=False``, so it keeps
    its variable dtype inside ``call``; a subclass casts it to whatever dtype it
    compares or adds in.

    :param q_hat_init: Initial ``q_hat``. Non-negative; ``+inf`` is allowed. The fitted
        value is assigned later by :meth:`set_q_hat` or :meth:`calibrate`.
    :type q_hat_init: float
    :param kwargs: Additional keyword arguments for the Layer base class. When
        :attr:`FROZEN` is true ``trainable`` is forced to ``False``.

    :raises ValueError: If ``q_hat_init`` is negative or NaN.
    """

    #: True for a layer whose only weight is ``q_hat``: the whole layer is then frozen.
    FROZEN: bool = False

    def __init__(self, q_hat_init: float, **kwargs: Any) -> None:
        if self.FROZEN:
            # Keras round-trips `trainable` through the config; drop the inbound value
            # so it does not collide with the forced one.
            kwargs.pop("trainable", None)
            kwargs["trainable"] = False
        super().__init__(**kwargs)
        self.q_hat_init = self._normalize_q_hat(q_hat_init, "q_hat_init")

        # Created in build().
        self.q_hat = None

    def _normalize_q_hat(self, value: float, argument: str = "q_hat") -> float:
        """Validate ``value`` and return it as the float to store.

        Subclasses override this to bound the value; call ``super()`` first.

        :param value: Candidate ``q_hat``.
        :param argument: Name used in the error message.
        :raises ValueError: If ``value`` is negative or NaN.
        """
        if not value >= 0.0:  # also rejects NaN
            raise ValueError(f"{argument} must be non-negative, got {value}")
        return float(value)

    def add_q_hat_weight(self) -> None:
        """Create the frozen scalar ``q_hat``. Call it from ``build`` before ``super().build``."""
        self.q_hat = self.add_weight(
            name="conformal_q_hat",
            shape=(),
            initializer=keras.initializers.Constant(self.q_hat_init),
            trainable=False,
            dtype=self.variable_dtype,
            autocast=False,
        )

    def _require_built(self, action: str) -> None:
        if self.q_hat is None:
            raise ValueError(
                f"{type(self).__name__} '{self.name}' is not built: call the layer "
                f"(or build it) before {action}"
            )

    def set_q_hat(self, value: float) -> None:
        """Assign a fitted ``q_hat`` into the frozen weight.

        :param value: The fitted quantile. ``+inf`` is allowed (an unbounded interval, or
            a set that keeps every label).
        :raises ValueError: If the layer is not built, or ``value`` is negative or NaN.
        """
        self._require_built("set_q_hat()")
        value = self._normalize_q_hat(value)
        self.q_hat.assign(value)
        logger.info(f"{type(self).__name__} '{self.name}' q_hat={value:.6g}")

    def calibrate(self, scores: np.ndarray, alpha: float = 0.1) -> None:
        """Fit ``q_hat`` from held-out nonconformity scores and assign it.

        ``q_hat`` is the ``ceil((n + 1)(1 - alpha))``-th smallest score, the order
        statistic that gives coverage ``>= 1 - alpha`` for exchangeable data; when ``n``
        is too small for ``alpha`` it is ``+inf``. The scores come from the layer's
        host-side rule and a calibration set the model did not train on. Runs in NumPy
        and writes a Keras weight, so it is not part of a traced graph.

        :param scores: Nonconformity scores, shape ``(n,)``.
        :param alpha: Miscoverage level in ``(0, 1)``. Defaults to 0.1 (90% coverage).
        :raises ValueError: If the layer is not built, ``scores`` is empty, or ``alpha``
            is outside ``(0, 1)``.
        """
        self._require_built("calibrate()")
        self.set_q_hat(conformal_quantile(np.asarray(scores), alpha))

    def get_config(self) -> Dict[str, Any]:
        """Return the serialization config.

        ``q_hat_init`` holds the CURRENT calibrated ``q_hat``, not the construction-time
        value, so a config-only reload (``from_config`` / ``clone_model``) keeps the
        calibration even without the ``.keras`` weights archive.
        """
        config = super().get_config()
        config["q_hat_init"] = (
            float(keras.ops.convert_to_numpy(self.q_hat))
            if self.q_hat is not None else self.q_hat_init
        )
        return config
