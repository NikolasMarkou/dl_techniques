"""
``ConformalIntervalLayer`` expands a point estimate into a split-conformal
prediction interval using a fixed, non-trainable radius.

Split conformal prediction attaches a distribution-free coverage guarantee to
any point predictor by measuring residuals on held-out data and taking a
quantile of them as an interval radius. That radius ``q`` is fit host-side and
assigned into a frozen weight, so the layer's forward pass is a pure affine
expansion: ``mu_c = clip(mu, domain_min, domain_max)``, then
``lower = mu_c - q`` and ``upper = mu_c + q``. No gradient reaches ``q``; a
separate ``calibrate()`` method assigns the fitted value outside `call()`.
Appending this layer to a frozen denoiser yields a single Functional model
that exports a calibrated interval predictor as one ``.keras`` file. A single
radius covers exactly one deployment noise regime; per-sigma (Mondrian)
calibration within one graph is out of scope.

References:
    - Vovk et al., 2005. Algorithmic Learning in a Random World. Springer.
    - Lei et al., 2018. Distribution-Free Predictive Inference for Regression.
      JASA 113(523). (https://arxiv.org/abs/1604.04173)
    - Angelopoulos and Bates, 2021. A Gentle Introduction to Conformal
      Prediction and Distribution-Free Uncertainty Quantification.
      (https://arxiv.org/abs/2107.07511)
    - Mohan et al., 2020. Robust and Interpretable Blind Image Denoising via
      Bias-Free Convolutional Neural Networks.
      (https://arxiv.org/abs/1906.05478)

"""

from typing import Any, Dict, Optional, Tuple, Union

import keras

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.conformal_interval")
class ConformalIntervalLayer(keras.layers.Layer):
    """Non-trainable, fixed-weights conformal prediction-interval layer.

    Holds a single non-trainable scalar weight ``q`` (the split-conformal
    radius, expressed in the ``[0, 1]`` denoiser output domain). Its
    ``call(mu)``:

    1. clips the incoming point estimate ``mu`` to ``[domain_min, domain_max]``;
    2. returns ``(mu_c, mu_c - q, mu_c + q)`` when ``return_mu=True``, else
       ``(mu_c - q, mu_c + q)``.

    The layer mirrors
    :class:`~dl_techniques.layers.time_series.forecasting_layers.ConformalQuantileHead`
    (non-trainable ``add_weight`` calibration score + a ``calibrate()`` method
    that ``.assign()``\\ s the fitted value outside ``call()``), differing only
    in that it wraps a pre-computed point estimate rather than learned quantiles
    and that it also carries the calibrated ``q`` through ``get_config`` (see
    D-004 below).

    :param q_init: Initial scalar radius used as the weight's ``Constant``
        initializer. Defaults to ``0.0`` (uncalibrated). The real value is
        fit host-side and either passed here or applied later via
        :meth:`calibrate`.
    :type q_init: float
    :param domain_min: Lower clip bound for ``mu``. Defaults to ``0.0``.
    :type domain_min: float
    :param domain_max: Upper clip bound for ``mu``. Defaults to ``1.0``.
    :type domain_max: float
    :param return_mu: If ``True`` (default), ``call`` returns the 3-tuple
        ``(mu_c, lower, upper)``; otherwise the 2-tuple ``(lower, upper)``.
    :type return_mu: bool
    :param kwargs: Additional keyword arguments for the Layer base class.
        ``trainable`` is forced to ``False``.

    Example:
        >>> import keras
        >>> inp = keras.Input(shape=(8, 8, 3))
        >>> out = ConformalIntervalLayer()(inp)
        >>> model = keras.Model(inp, out)
        >>> model.layers[-1].calibrate(0.0488)  # radius from calibrate_per_sigma

    Note:
        Per-sigma (Mondrian) deployment in one graph is a future extension; this
        layer intentionally carries a single scalar radius (one noise regime per
        exported graph).
    """

    # DECISION plan_2026-07-12_e56909cd/D-001: denoiser domain is [0, 1], no compat
    # branch or migration shim for the legacy [-0.5, +0.5] bounds -- rebuild the graph instead. See decisions.md.
    def __init__(
            self,
            q_init: float = 0.0,
            domain_min: float = 0.0,
            domain_max: float = 1.0,
            return_mu: bool = True,
            **kwargs: Any,
    ) -> None:
        # The whole layer is non-trainable: no gradients ever flow to q.
        # Drop any inbound `trainable` (Keras round-trips it into the config via
        # super().get_config(), and from_config re-passes it in **kwargs) so it
        # does not collide with the explicit trainable=False below.
        kwargs.pop("trainable", None)
        super().__init__(trainable=False, **kwargs)
        self.q_init = float(q_init)
        self.domain_min = float(domain_min)
        self.domain_max = float(domain_max)
        self.return_mu = bool(return_mu)

        # Created in build().
        self.q = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the non-trainable scalar radius weight.

        :param input_shape: Shape of the incoming point estimate ``mu``.
        """
        self.q = self.add_weight(
            name="conformal_q",
            shape=(),
            initializer=keras.initializers.Constant(self.q_init),
            trainable=False,
            dtype=self.compute_dtype,
        )
        super().build(input_shape)

    def call(
            self,
            mu: keras.KerasTensor,
            training: Optional[bool] = None,
    ) -> Union[
        Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor],
        Tuple[keras.KerasTensor, keras.KerasTensor],
    ]:
        """Clip ``mu`` to the calibrated domain and emit the conformal interval.

        :param mu: Point estimate (already unwrapped; deep-supervision index-0
            unwrap stays the caller's job). Shape ``(..., domain)``.
        :param training: Unused; present for the Keras call contract.
        :return: ``(mu_c, mu_c - q, mu_c + q)`` when ``return_mu`` is ``True``,
            otherwise ``(mu_c - q, mu_c + q)``.
        """
        mu_c = keras.ops.clip(mu, self.domain_min, self.domain_max)
        lower = mu_c - self.q
        upper = mu_c + self.q
        return (mu_c, lower, upper) if self.return_mu else (lower, upper)

    def calibrate(self, q_value: float) -> None:
        """Assign the fitted conformal radius into the frozen weight.

        The host-side numpy calibration is done separately by
        :func:`dl_techniques.utils.conformal_denoiser_intervals.calibrate_per_sigma`
        (which returns a ``{sigma: q}`` dict; pick ``[sigma]`` for the target
        deployment noise level) or
        :func:`~dl_techniques.utils.conformal_denoiser_intervals.conformal_quantile`.
        This method only ``.assign()``\\ s that fitted scalar into the
        non-trainable graph weight, outside ``call()``.

        :param q_value: The fitted scalar conformal radius.
        """
        self.q.assign(float(q_value))
        logger.info(f"ConformalIntervalLayer '{self.name}' calibrated q={float(q_value):.6g}")

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...],
    ) -> Tuple[Tuple[Optional[int], ...], ...]:
        """Return the tuple-of-shapes matching the ``return_mu`` arity."""
        if self.return_mu:
            return (input_shape, input_shape, input_shape)
        return (input_shape, input_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return the serialization config.

        Stores the current calibrated ``q``, not just the construction-time
        ``q_init``, so a config-only reload path (``from_config`` /
        ``clone_model``) recovers the fitted radius even without the
        ``.keras`` weights archive. See decisions.md D-004.
        """
        config = super().get_config()
        config.update({
            "q_init": float(keras.ops.convert_to_numpy(self.q)) if self.q is not None else self.q_init,
            "domain_min": self.domain_min,
            "domain_max": self.domain_max,
            "return_mu": self.return_mu,
        })
        return config

# ---------------------------------------------------------------------
