"""Fixed-weights, non-trainable conformal prediction-interval layer.

This module provides :class:`ConformalIntervalLayer`, a serializable Keras 3
layer that bakes a post-hoc split-conformal radius ``q`` into a deployable
graph. It is meant to be appended after a frozen (bias-free / Miyasawa)
denoiser so that a single ``denoiser -> ConformalIntervalLayer`` Functional
model exports a calibrated interval predictor as one ``.keras`` artifact.

Design summary
--------------
- **Non-trainable.** The whole layer is constructed ``trainable=False`` and its
  single weight ``conformal_q`` (a scalar radius) is created with
  ``trainable=False``. No gradients ever flow to it.
- **Post-hoc calibration.** The radius ``q`` is NOT learned; it is fit
  host-side on held-out data by
  :func:`dl_techniques.utils.conformal_denoiser_intervals.calibrate_per_sigma`
  (or :func:`~dl_techniques.utils.conformal_denoiser_intervals.conformal_quantile`)
  and assigned into the frozen weight via :meth:`ConformalIntervalLayer.calibrate`
  OUTSIDE ``call()``, exactly mirroring
  :class:`dl_techniques.layers.time_series.forecasting_layers.ConformalQuantileHead`.
- **Domain clip ``[0, 1]``.** ``call()`` clips the incoming point estimate
  ``mu`` to ``[domain_min, domain_max]`` (default ``[0, 1]``), matching
  ``DOMAIN_MIN`` / ``DOMAIN_MAX`` and the ``_predict_mu`` clip in
  ``conformal_denoiser_intervals.py``. A mismatched domain silently breaks the
  coverage guarantee the numpy calibration was fit under.
- **Serialized domain / legacy artifacts.** ``domain_min`` / ``domain_max`` are
  carried in :meth:`ConformalIntervalLayer.get_config`, so a model SAVED before
  the ``[-0.5, +0.5] -> [0, 1]`` denoiser-domain migration keeps the OLD bounds
  baked into its ``.keras`` config and will reload with them. That is intended:
  such a graph wraps a legacy-domain denoiser and is invalid end-to-end anyway
  (a bias-free net cannot be domain-shifted post hoc). There is deliberately NO
  compat branch and NO migration shim — rebuild the graph around a ``[0, 1]``
  denoiser instead.
- **Scalar ``q`` (one deployment noise level).** One frozen scalar radius = one
  deployment noise regime. Baking a full per-sigma table into a single graph is
  a documented FUTURE EXTENSION (would require a second ``sigma`` call-time
  input plus an index-lookup against a ``(num_sigmas,)`` grid) and is out of
  scope here.

Backend-agnostic: uses ``keras.ops`` only, no raw TensorFlow ops.
"""

from typing import Any, Dict, Optional, Tuple, Union

import keras

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
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

    Args:
        q_init: Initial scalar radius used as the weight's ``Constant``
            initializer. Defaults to ``0.0`` (uncalibrated). The real value is
            fit host-side and either passed here or applied later via
            :meth:`calibrate`.
        domain_min: Lower clip bound for ``mu``. Defaults to ``0.0``.
        domain_max: Upper clip bound for ``mu``. Defaults to ``1.0``.
        return_mu: If ``True`` (default), ``call`` returns the 3-tuple
            ``(mu_c, lower, upper)``; otherwise the 2-tuple ``(lower, upper)``.
        **kwargs: Forwarded to :class:`keras.layers.Layer`. ``trainable`` is
            forced to ``False``.

    Example:
        >>> import keras
        >>> inp = keras.Input(shape=(8, 8, 3))
        >>> out = ConformalIntervalLayer()(inp)
        >>> model = keras.Model(inp, out)
        >>> model.layers[-1].calibrate(0.0488)  # radius from calibrate_per_sigma

    Note:
        Per-sigma (Mondrian) deployment in ONE graph is a future extension; this
        layer intentionally carries a single scalar radius (one noise regime per
        exported graph).
    """

    # DECISION plan_2026-07-12_e56909cd/D-001: the denoiser domain is [0, 1] and
    # these defaults are its only statement here. Do NOT add a compat branch, a
    # `pixel_domain` kwarg, or a from_config migration shim for models saved with
    # the legacy [-0.5, +0.5] bounds: a bias-free (degree-1 homogeneous) denoiser
    # cannot be domain-shifted post hoc, so such a graph is invalid end-to-end and
    # a shim would only make it silently wrong instead of loudly wrong. Rebuild the
    # graph around a [0, 1] denoiser. See decisions.md D-001 (INV-4, no-compat).
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

        Args:
            input_shape: Shape of the incoming point estimate ``mu``.
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

        Args:
            mu: Point estimate (already unwrapped; deep-supervision index-0
                unwrap stays the caller's job). Shape ``(..., domain)``.
            training: Unused; present for the Keras call contract.

        Returns:
            ``(mu_c, mu_c - q, mu_c + q)`` when ``return_mu`` is ``True``,
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
        non-trainable graph weight, OUTSIDE ``call()``.

        Args:
            q_value: The fitted scalar conformal radius.
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

        Storing the CURRENT calibrated ``q`` (not just the construction-time
        ``q_init``) is a deliberate belt-and-suspenders deviation from the
        ``ConformalQuantileHead`` precedent, whose ``get_config`` omits the
        calibrated value and relies solely on the ``.keras`` weights archive.
        See decisions.md D-004: carrying ``q`` in the config too lets any
        config-only reload path (e.g. ``from_config`` / ``clone_model``) recover
        the fitted radius even when it bypasses the weights archive.
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
