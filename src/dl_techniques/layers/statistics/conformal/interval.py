"""
``ConformalInterval`` expands a point estimate into a split-conformal prediction
interval using a fixed, non-trainable radius.

Split conformal prediction attaches a distribution-free coverage guarantee to any point
predictor by measuring residuals on held-out data and taking a quantile of them as an
interval radius. That radius ``q_hat`` is fit host-side (:meth:`ConformalLayer.calibrate`,
from scores ``|truth - mu|``) and assigned into a frozen weight, so the layer's forward
pass is a pure affine expansion: ``mu_c = clip(mu, domain_min, domain_max)``, then
``lower = mu_c - q_hat`` and ``upper = mu_c + q_hat``. No gradient reaches ``q_hat``.
Appending this layer to a frozen denoiser yields a single Functional model that exports a
calibrated interval predictor as one ``.keras`` file. A single radius covers exactly one
deployment noise regime; per-sigma (Mondrian) calibration within one graph is out of
scope.

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

import keras
from typing import Any, Dict, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.keras_registration import register_dl_technique
from .base import ConformalLayer

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.statistics.conformal.interval")
class ConformalInterval(ConformalLayer):
    """Non-trainable, fixed-radius conformal prediction-interval layer.

    Holds the single frozen scalar ``q_hat`` (the split-conformal radius, expressed in
    the ``[0, 1]`` denoiser output domain). Its ``call(mu)``:

    1. clips the incoming point estimate ``mu`` to ``[domain_min, domain_max]``;
    2. returns ``(mu_c, mu_c - q_hat, mu_c + q_hat)`` when ``return_mu=True``, else
       ``(mu_c - q_hat, mu_c + q_hat)``.

    It is the point-estimate sibling of
    :class:`~dl_techniques.layers.statistics.conformal.quantile_head.ConformalQuantileHead`,
    which widens learned quantiles, and of
    :class:`~dl_techniques.layers.statistics.conformal.prediction_set.ConformalPredictionSet`,
    which turns classifier logits into a set.

    :param q_hat_init: Initial radius. Defaults to ``0.0`` (uncalibrated). The real value
        is fit host-side and either passed here or applied later via
        :meth:`~ConformalLayer.set_q_hat` or :meth:`~ConformalLayer.calibrate`.
    :type q_hat_init: float
    :param domain_min: Lower clip bound for ``mu``. Defaults to ``0.0``.
    :type domain_min: float
    :param domain_max: Upper clip bound for ``mu``. Must exceed ``domain_min``. Defaults
        to ``1.0``.
    :type domain_max: float
    :param return_mu: If ``True`` (default), ``call`` returns the 3-tuple
        ``(mu_c, lower, upper)``; otherwise the 2-tuple ``(lower, upper)``.
    :type return_mu: bool
    :param kwargs: Additional keyword arguments for the Layer base class.
        ``trainable`` is forced to ``False``.

    Input shape:
        Any shape ``(..., domain)``: the point estimate.

    Output shape:
        A tuple of tensors with the input's shape (three, or two without ``return_mu``).

    :raises ValueError: If ``q_hat_init`` is negative or NaN, or ``domain_min`` is not
        below ``domain_max``.

    Example:
        >>> import keras
        >>> inp = keras.Input(shape=(8, 8, 3))
        >>> out = ConformalInterval()(inp)
        >>> model = keras.Model(inp, out)
        >>> model.layers[-1].set_q_hat(0.0488)  # radius from calibrate_per_sigma

    Note:
        Per-sigma (Mondrian) deployment in one graph is a future extension; this
        layer intentionally carries a single scalar radius (one noise regime per
        exported graph).
    """

    FROZEN = True

    # DECISION plan_2026-07-12_e56909cd/D-001: denoiser domain is [0, 1], no compat
    # branch or migration shim for the legacy [-0.5, +0.5] bounds -- rebuild the graph instead. See decisions.md.
    def __init__(
            self,
            q_hat_init: float = 0.0,
            domain_min: float = 0.0,
            domain_max: float = 1.0,
            return_mu: bool = True,
            **kwargs: Any,
    ) -> None:
        super().__init__(q_hat_init=q_hat_init, **kwargs)
        if not domain_min < domain_max:
            raise ValueError(
                f"domain_min must be below domain_max, got domain_min={domain_min}, "
                f"domain_max={domain_max}"
            )
        self.domain_min = float(domain_min)
        self.domain_max = float(domain_max)
        self.return_mu = bool(return_mu)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the frozen radius weight.

        :param input_shape: Shape of the incoming point estimate ``mu``.
        """
        self.add_q_hat_weight()
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
        :return: ``(mu_c, mu_c - q_hat, mu_c + q_hat)`` when ``return_mu`` is ``True``,
            otherwise ``(mu_c - q_hat, mu_c + q_hat)``.
        """
        mu_c = keras.ops.clip(mu, self.domain_min, self.domain_max)
        q_hat = keras.ops.cast(self.q_hat, mu_c.dtype)
        lower = mu_c - q_hat
        upper = mu_c + q_hat
        return (mu_c, lower, upper) if self.return_mu else (lower, upper)

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...],
    ) -> Tuple[Tuple[Optional[int], ...], ...]:
        """Return the tuple-of-shapes matching the ``return_mu`` arity."""
        if self.return_mu:
            return (input_shape, input_shape, input_shape)
        return (input_shape, input_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return the serialization config (``q_hat_init`` is the calibrated value)."""
        config = super().get_config()
        config.update({
            "domain_min": self.domain_min,
            "domain_max": self.domain_max,
            "return_mu": self.return_mu,
        })
        return config
