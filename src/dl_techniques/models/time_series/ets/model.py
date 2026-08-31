"""
Pure additive exponential smoothing (ETS) in single-source-of-error state-space form.

Why this model exists
---------------------
Every other forecaster under ``models/time_series/`` is **direct**: it emits the
whole ``[B, H, F]`` horizon block from per-step heads in a single pass, with no
recursion and nothing compounding across the horizon. That is a perfectly good
way to forecast, but it means the repository had **no model with a smoothing
parameter** -- and therefore no way to exercise, let alone verify, the result
that motivates the multistep losses in ``losses/multistep_loss.py``:

    Minimising h-steps-ahead errors SHRINKS a model's smoothing parameters
    toward zero, making it less reactive to noise and more stable over longer
    horizons (Svetunkov, Kourentzes & Killick, 2023).

That result is proven for **pure additive** ETS and ARIMA, and it arises through
**recursive** error accumulation. This model is the smallest object on which it
can be reproduced rather than merely cited, and
``tests/test_models/test_ets/`` does reproduce it.

The state space
---------------
Hyndman et al.'s additive ETS, written with a single error term::

    yhat_t = l_{t-1} + b_{t-1} + s_{t-m}          (one step ahead, from t-1)
    e_t    = y_t - yhat_t
    l_t    = l_{t-1} + b_{t-1} + alpha * e_t      (level)
    b_t    = b_{t-1} + beta * e_t                 (trend)
    s_t    = s_{t-m} + gamma * e_t                (seasonal)

Because the model is *pure additive*, the h-step-ahead forecast from origin t is
closed form -- no rollout loop, no sampling::

    yhat_{t+h|t} = l_t + h * b_t + s_{t + h - m*ceil(h/m)}

Only ``ANN`` (local level), ``AAN`` (local trend) and ``AAA`` (additive
seasonal) are implemented. Multiplicative and mixed variants are deliberately
absent: the shrinkage result does not extend to them, and shipping them here
would invite exactly the claim the paper declines to make.

Forecast origins
----------------
ADAM computes the multistep errors from *every in-sample origin*. Here the
**sliding-window dataset supplies the origins**: one training sample is one
context window plus its ``H``-step future, so a minibatch IS a sample of
forecast origins and ``MultistepLoss`` averages over it unchanged. This costs a
re-filter per overlapping window, which is nothing for a model with at most
three trainable scalars, and it buys a single code path -- ``call`` and
``_forecast`` are the same function, so there is no train/serve skew and no
mode flag.

Trainable surface
-----------------
``alpha``, ``beta`` and ``gamma`` only -- one scalar each, held in ``[0, 1]``
through a sigmoid. **The initial states are derived from the input window**
(level from the first seasonal period, trend from its first differences,
seasonal from the deviations of the first period, centred to sum to zero) rather
than fitted. That keeps the model batch-friendly and scale-adaptive, and it
keeps the trainable surface *exactly* the smoothing parameters -- which is what
the shrinkage claim is about. A model with fitted initial states would let the
optimiser trade shrinkage against initialisation and confound the measurement.

References
----------
-   Svetunkov, I., Kourentzes, N., & Killick, R. (2023). "Multi-step Estimators
    and Shrinkage Effect in Time Series Models". *Computational Statistics*.
    DOI: 10.1007/s00180-023-01377-x
-   Svetunkov, I. (2023). *Forecasting and Analytics with the Augmented Dynamic
    Adaptive Model (ADAM)*. https://openforecast.org/adam/
-   Hyndman, R.J., Koehler, A.B., Ord, J.K., & Snyder, R.D. (2008).
    *Forecasting with Exponential Smoothing: The State Space Approach*. Springer.
"""

import math
from typing import Any, Dict, Optional, Tuple

import keras
import numpy as np

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.models.time_series.forecast import Forecast, ForecastMixin

# ---------------------------------------------------------------------

#: The pure-additive ETS variants this model implements.
ETS_VARIANTS = ("ANN", "AAN", "AAA")

# ---------------------------------------------------------------------


def _inverse_sigmoid(value: float) -> float:
    """Return the logit of ``value``, for initialising a sigmoid-bounded weight.

    :param value: A smoothing parameter in the open interval ``(0, 1)``.
    :type value: float
    :return: ``log(value / (1 - value))``.
    :rtype: float
    """
    return math.log(value / (1.0 - value))


@register_dl_technique("dl_techniques.models.ets.model")
class ETSModel(keras.Model, ForecastMixin):
    """Pure additive ETS with trainable smoothing parameters.

    See the module docstring for the state-space equations, the closed-form
    h-step forecast, and why the initial states are derived rather than fitted.

    :param variant: ``"ANN"`` (level only), ``"AAN"`` (level + trend) or
        ``"AAA"`` (level + trend + additive seasonal).
    :type variant: str
    :param horizon: Number of steps to forecast, ``H``.
    :type horizon: int
    :param seasonal_period: Season length ``m``. Required (``> 1``) for
        ``"AAA"``; ignored otherwise.
    :type seasonal_period: Optional[int]
    :param alpha_init: Initial level smoothing parameter, in ``(0, 1)``.
    :type alpha_init: float
    :param beta_init: Initial trend smoothing parameter, in ``(0, 1)``. Unused
        by ``"ANN"``.
    :type beta_init: float
    :param gamma_init: Initial seasonal smoothing parameter, in ``(0, 1)``.
        Used only by ``"AAA"``.
    :type gamma_init: float
    :raises ValueError: If ``variant`` is not a pure-additive variant, if
        ``horizon`` is not positive, if ``"AAA"`` is requested without a
        ``seasonal_period > 1``, or if any smoothing initialiser is outside
        ``(0, 1)``.

    Example::

        model = ETSModel(variant="AAA", horizon=12, seasonal_period=12)
        model.compile(optimizer="adam", loss=MultistepLoss("gtmse", h=12))
        model.fit(context_windows, futures)      # [B, T] -> [B, H, 1]
        float(model.alpha)                        # the fitted smoothing parameter
    """

    def __init__(
        self,
        variant: str = "ANN",
        horizon: int = 1,
        seasonal_period: Optional[int] = None,
        alpha_init: float = 0.3,
        beta_init: float = 0.1,
        gamma_init: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if variant not in ETS_VARIANTS:
            raise ValueError(
                f"Unknown ETS variant {variant!r}. Expected one of "
                f"{sorted(ETS_VARIANTS)}. Multiplicative and mixed variants are "
                f"deliberately not implemented: the shrinkage result this model "
                f"exists to reproduce is proven for PURE ADDITIVE models only."
            )
        if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
            raise ValueError(f"horizon must be a positive integer, got {horizon!r}.")

        self.use_trend = variant in ("AAN", "AAA")
        self.use_seasonal = variant == "AAA"

        if self.use_seasonal:
            if not isinstance(seasonal_period, int) or seasonal_period < 2:
                raise ValueError(
                    f"variant='AAA' needs seasonal_period > 1, got "
                    f"{seasonal_period!r}."
                )
        for name, value in (
            ("alpha_init", alpha_init),
            ("beta_init", beta_init),
            ("gamma_init", gamma_init),
        ):
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be in (0, 1), got {value}.")

        self.variant = variant
        self.horizon = horizon
        self.seasonal_period = int(seasonal_period) if self.use_seasonal else 1
        self.alpha_init = float(alpha_init)
        self.beta_init = float(beta_init)
        self.gamma_init = float(gamma_init)

        self.alpha_raw = None
        self.beta_raw = None
        self.gamma_raw = None

    # -----------------------------------------------------------------
    # build
    # -----------------------------------------------------------------

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the one-to-three smoothing scalars.

        :param input_shape: ``(batch, time)`` or ``(batch, time, 1)``.
        :type input_shape: tuple
        :raises ValueError: If the input is not univariate, or if the context is
            shorter than one seasonal period.
        """
        if self.built:
            return

        if len(input_shape) == 3 and input_shape[-1] not in (None, 1):
            raise ValueError(
                f"ETSModel is univariate; expected a trailing feature axis of 1, "
                f"got input_shape={input_shape}."
            )
        if len(input_shape) not in (2, 3):
            raise ValueError(
                f"Expected input of shape (batch, time) or (batch, time, 1), "
                f"got {input_shape}."
            )

        time_steps = input_shape[1]
        if time_steps is not None and time_steps < self.seasonal_period + 1:
            raise ValueError(
                f"Context length {time_steps} is too short for "
                f"seasonal_period={self.seasonal_period}: the initial seasonal "
                f"state needs a full period plus one step to update."
            )

        self.alpha_raw = self.add_weight(
            name="alpha_raw",
            shape=(),
            initializer=keras.initializers.Constant(_inverse_sigmoid(self.alpha_init)),
            trainable=True,
        )
        if self.use_trend:
            self.beta_raw = self.add_weight(
                name="beta_raw",
                shape=(),
                initializer=keras.initializers.Constant(_inverse_sigmoid(self.beta_init)),
                trainable=True,
            )
        if self.use_seasonal:
            self.gamma_raw = self.add_weight(
                name="gamma_raw",
                shape=(),
                initializer=keras.initializers.Constant(_inverse_sigmoid(self.gamma_init)),
                trainable=True,
            )

        super().build(input_shape)

    # -----------------------------------------------------------------
    # smoothing parameters
    # -----------------------------------------------------------------

    def _smoothing(self) -> Tuple[Any, Any, Any]:
        """Return ``(alpha, beta, gamma)`` as tensors in ``[0, 1]``.

        Unused components are returned as the exact scalar ``0.0``, which makes
        their state update a no-op rather than a small nuisance term.

        :return: The three smoothing parameters.
        :rtype: tuple
        """
        alpha = keras.ops.sigmoid(self.alpha_raw)
        beta = keras.ops.sigmoid(self.beta_raw) if self.use_trend else 0.0
        gamma = keras.ops.sigmoid(self.gamma_raw) if self.use_seasonal else 0.0
        return alpha, beta, gamma

    @property
    def alpha(self) -> float:
        """The fitted level smoothing parameter.

        :return: ``sigmoid(alpha_raw)``.
        :rtype: float
        """
        return float(keras.ops.convert_to_numpy(keras.ops.sigmoid(self.alpha_raw)))

    @property
    def beta(self) -> float:
        """The fitted trend smoothing parameter, or ``0.0`` for ``"ANN"``.

        :return: ``sigmoid(beta_raw)``.
        :rtype: float
        """
        if not self.use_trend:
            return 0.0
        return float(keras.ops.convert_to_numpy(keras.ops.sigmoid(self.beta_raw)))

    @property
    def gamma(self) -> float:
        """The fitted seasonal smoothing parameter, or ``0.0`` when unused.

        :return: ``sigmoid(gamma_raw)``.
        :rtype: float
        """
        if not self.use_seasonal:
            return 0.0
        return float(keras.ops.convert_to_numpy(keras.ops.sigmoid(self.gamma_raw)))

    # -----------------------------------------------------------------
    # the recursion
    # -----------------------------------------------------------------

    @staticmethod
    def _as_series(inputs: Any) -> Any:
        """Squeeze an optional trailing feature axis, giving ``[B, T]``.

        :param inputs: ``[B, T]`` or ``[B, T, 1]``.
        :return: ``[B, T]``.
        """
        if len(inputs.shape) == 3:
            return keras.ops.squeeze(inputs, axis=-1)
        return inputs

    def _initial_state(self, series: Any) -> Any:
        """Derive the initial state from the window itself.

        The state is packed FLAT as ``[level, trend, s_{1-m} ... s_0]`` of width
        ``2 + m``. It is packed rather than carried as a tuple because
        ``keras.ops.scan`` on the TensorFlow backend requires the per-step
        output to have the same structure and shape as the carry, and a flat
        tensor makes that constraint trivial to satisfy.

        The seasonal component is centred to sum to zero, which is the standard
        identification constraint for additive seasonality: without it the level
        and the seasonal indices are only determined up to a shared constant.

        :param series: The context window, ``[B, T]``.
        :return: The packed initial state, ``[B, 2 + m]``.
        """
        m = self.seasonal_period
        warmup = m if self.use_seasonal else 1

        level = keras.ops.mean(series[:, :warmup], axis=1)

        if self.use_trend:
            time_steps = series.shape[1]
            if time_steps is None:
                raise ValueError(
                    "ETSModel needs a STATIC context length to derive the "
                    "initial trend state; got a dynamic time axis. Build the "
                    "model against a concrete input shape."
                )
            span = min(time_steps - 1, 2 * max(1, m))
            trend = keras.ops.mean(
                series[:, 1 : span + 1] - series[:, :span], axis=1
            )
        else:
            trend = keras.ops.zeros_like(level)

        if self.use_seasonal:
            seasonal = series[:, :m] - level[:, None]
            seasonal = seasonal - keras.ops.mean(seasonal, axis=1, keepdims=True)
        else:
            seasonal = keras.ops.zeros_like(level)[:, None]

        return keras.ops.concatenate(
            [level[:, None], trend[:, None], seasonal], axis=1
        )

    def _filter(self, series: Any) -> Tuple[Any, Any]:
        """Run the state recursion over the whole context window.

        :param series: ``[B, T]``.
        :return: ``(final_state [B, 2 + m], state_history [T, B, 2 + m])``.
        """
        alpha, beta, gamma = self._smoothing()

        def step(state: Any, observation: Any) -> Tuple[Any, Any]:
            level = state[:, 0]
            trend = state[:, 1]
            seasonal = state[:, 2:]

            prediction = level + trend + seasonal[:, 0]
            error = observation - prediction

            new_level = level + trend + alpha * error
            new_trend = trend + beta * error
            new_season = seasonal[:, 0] + gamma * error

            rolled = keras.ops.concatenate(
                [seasonal[:, 1:], new_season[:, None]], axis=1
            )
            new_state = keras.ops.concatenate(
                [new_level[:, None], new_trend[:, None], rolled], axis=1
            )
            # The TF backend requires the per-step output to mirror the carry.
            return new_state, new_state

        # scan consumes a leading time axis, so the batch axis moves inward.
        return keras.ops.scan(
            step, self._initial_state(series), keras.ops.transpose(series, (1, 0))
        )

    def _horizon_forecast(self, state: Any) -> Any:
        """Apply the closed-form h-step formula to a terminal state.

        :param state: Packed state ``[B, 2 + m]``.
        :return: Forecast of shape ``[B, H, 1]``.
        """
        level = state[:, 0:1]
        trend = state[:, 1:2]
        seasonal = state[:, 2:]

        steps = keras.ops.arange(1, self.horizon + 1, dtype=level.dtype)
        forecast = level + trend * steps[None, :]

        if self.use_seasonal:
            indices = np.arange(self.horizon) % self.seasonal_period
            forecast = forecast + keras.ops.take(
                seasonal, keras.ops.convert_to_tensor(indices), axis=1
            )

        return forecast[:, :, None]

    # -----------------------------------------------------------------
    # forward pass
    # -----------------------------------------------------------------

    def call(self, inputs: Any, training: Optional[bool] = None) -> Any:
        """Filter the context window, then forecast ``horizon`` steps from its end.

        There is one code path: training and inference call exactly this
        function, so no train/serve skew is possible.

        :param inputs: Context window, ``[B, T]`` or ``[B, T, 1]``.
        :param training: Unused; the model has no training-only behaviour.
        :return: Forecast of shape ``[B, H, 1]``.
        """
        del training
        series = self._as_series(inputs)
        final_state, _ = self._filter(series)
        return self._horizon_forecast(final_state)

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], int, int]:
        """Return the output shape ``(batch, horizon, 1)``.

        :param input_shape: ``(batch, time[, 1])``.
        :return: ``(batch, horizon, 1)``.
        """
        return (input_shape[0], self.horizon, 1)

    def fitted_values(self, inputs: Any) -> Tuple[Any, Any]:
        """Return the in-sample one-step-ahead fitted values and residuals.

        The state history is a free by-product of the filtering scan, so this
        costs one extra pass and no extra recursion.

        :param inputs: Context window, ``[B, T]`` or ``[B, T, 1]``.
        :return: ``(fitted [B, T], residuals [B, T])``, both aligned with the
            input's time axis. ``fitted[:, t]`` is the forecast of ``y_t`` made
            from information up to ``t-1``.
        """
        if not self.built:
            # This is a public entry point that does NOT go through
            # ``Model.__call__``, so nothing has built the weights yet.
            self.build(tuple(inputs.shape))

        series = self._as_series(inputs)
        _, history = self._filter(series)

        # history[t] is the state AFTER absorbing y_t, so the one-step-ahead
        # prediction of y_t comes from history[t-1], with the initial state
        # standing in at t = 0.
        history = keras.ops.transpose(history, (1, 0, 2))  # [B, T, 2 + m]
        initial = self._initial_state(series)[:, None, :]
        previous = keras.ops.concatenate([initial, history[:, :-1, :]], axis=1)

        fitted = previous[:, :, 0] + previous[:, :, 1] + previous[:, :, 2]
        return fitted, series - fitted

    # -----------------------------------------------------------------
    # Forecast contract
    # -----------------------------------------------------------------

    def _forecast(self, x: Any, **kwargs: Any) -> Forecast:
        """Produce a point :class:`Forecast`.

        This is a POINT model: it must not fabricate intervals, so
        ``quantiles`` is ``None`` and callers should test
        ``forecast.has_quantiles()``.

        :param x: Context window, ``[B, T]`` or ``[B, T, 1]``.
        :return: A :class:`Forecast` with ``point`` of shape ``[B, H, 1]``.
        :rtype: Forecast
        """
        del kwargs
        point = keras.ops.convert_to_numpy(self(x, training=False))
        return Forecast(point=np.asarray(point))

    # -----------------------------------------------------------------
    # serialization
    # -----------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor configuration.

        :return: The base model config plus this model's six knobs.
        :rtype: dict
        """
        config = super().get_config()
        config.update(
            {
                "variant": self.variant,
                "horizon": self.horizon,
                "seasonal_period": self.seasonal_period if self.use_seasonal else None,
                "alpha_init": self.alpha_init,
                "beta_init": self.beta_init,
                "gamma_init": self.gamma_init,
            }
        )
        return config


# ---------------------------------------------------------------------


def create_ets(
    variant: str = "ANN",
    horizon: int = 1,
    seasonal_period: Optional[int] = None,
    **kwargs: Any,
) -> ETSModel:
    """Build an :class:`ETSModel`.

    :param variant: ``"ANN"``, ``"AAN"`` or ``"AAA"``.
    :type variant: str
    :param horizon: Number of steps to forecast.
    :type horizon: int
    :param seasonal_period: Season length; required for ``"AAA"``.
    :type seasonal_period: Optional[int]
    :param kwargs: Forwarded to :class:`ETSModel` (``alpha_init``,
        ``beta_init``, ``gamma_init``, ``name``, ...).
    :return: The configured model.
    :rtype: ETSModel
    """
    logger.info(
        f"Creating ETSModel variant={variant} horizon={horizon} "
        f"seasonal_period={seasonal_period}"
    )
    return ETSModel(
        variant=variant,
        horizon=horizon,
        seasonal_period=seasonal_period,
        **kwargs,
    )

# ---------------------------------------------------------------------
