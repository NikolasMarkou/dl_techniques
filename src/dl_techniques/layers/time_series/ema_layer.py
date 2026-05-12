"""
Adaptive EMA Slope Filter Layer.

This module implements an adaptive EMA-based slope filter for time series
trading signal generation, inspired by the strategy of filtering trades
based on EMA slope conditions.

The key insight is that trading when the EMA slope is within certain bounds
(e.g., between -15 and +15) can produce better risk-adjusted returns than
trading only on positive or negative slopes.

References:
    - LeBeau, C. (1992). *Computer Analysis of the Futures Markets* — EMA slope
      based filtering as a regime-detection primitive.
    - Bollinger, J. (2001). *Bollinger on Bollinger Bands* — slope as
      trend/volatility regime indicator.
    - Koenker, R. & Bassett, G. (1978). "Regression Quantiles." *Econometrica*
      46(1): 33-50 — quantile-loss formulation used by the downstream slope
      quantile head.
    - EMA slope = EMA(current) - EMA(lookback_period bars ago)
    - Trade signals based on slope thresholds (above, below, between)
"""

import keras
from keras import ops
from typing import Any, Dict, Optional, Tuple, Union

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ExponentialMovingAverage(keras.layers.Layer):
    """
    Computes Exponential Moving Average over time series data.

    The EMA is computed recursively as:
        EMA_t = alpha * x_t + (1 - alpha) * EMA_{t-1}

    where alpha = 2 / (period + 1).

    **Architecture Overview:**

    .. code-block:: text

        Input: x = [x_0, x_1, x_2, ..., x_T]
                        │
                        ▼
               ┌────────────────────┐
               │  Initialize EMA_0  │
               │     EMA_0 = x_0    │
               └─────────┬──────────┘
                         │
                         ▼
               ┌────────────────────────────────────┐
               │  Recursive EMA Computation         │
               │  for t = 1 to T:                   │
               │    EMA_t = α·x_t + (1-α)·EMA_{t-1} │
               │                                    │
               │  where α = 2/(period + 1)          │
               └─────────┬──────────────────────────┘
                         │
                         ▼
               ┌────────────────────┐
               │  (Optional Adjust) │
               │  Bias correction   │
               │  for early steps   │
               └─────────┬──────────┘
                         │
                         ▼
        Output: EMA = [EMA_0, EMA_1, ..., EMA_T]
                (same shape as input)

    :param period: The EMA period (window size for smoothing factor calculation).
    :type period: int
    :param adjust: If True, uses adjusted weights for the beginning of the series.
    :type adjust: bool
    :param kwargs: Additional layer arguments.
    """

    def __init__(
        self,
        period: int = 25,
        adjust: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if period < 1:
            raise ValueError(f"period must be >= 1, got {period}")

        self.period = period
        self.adjust = adjust
        self.alpha = 2.0 / (period + 1.0)

    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """
        Compute EMA over the time dimension.

        Uses `keras.ops.scan` to materialise the recurrence in a single
        Scan op (XLA-friendly, ONNX-clean). The recurrence is preserved
        bit-identically to the previous Python-loop implementation;
        carry is `(ema_prev, t)` with `t` an int32 step counter so the
        adjust=True weight `1 - (1-α)^(t+1)` matches the original.

        :param inputs: Input tensor of shape (batch, time_steps, features)
            or (batch, time_steps).
        :type inputs: keras.KerasTensor
        :return: EMA values with same shape as input.
        :rtype: keras.KerasTensor
        """
        ndim = len(inputs.shape)

        if ndim == 2:
            # (batch, time_steps) -> (batch, time_steps, 1)
            x = ops.expand_dims(inputs, axis=-1)
        else:
            x = inputs

        # Edge case: T == 1 → output equals input verbatim (no recurrence).
        # Use static shape when known so the graph stays clean.
        static_T = x.shape[1]
        if static_T == 1:
            return inputs

        alpha = ops.cast(self.alpha, dtype=x.dtype)
        one_minus_alpha = ops.cast(1.0 - self.alpha, dtype=x.dtype)

        # DECISION plan_2026-05-12_5f0e087c/D-001:
        # Preserve the CUSTOM adjust=True semantics verbatim — divide
        # the recurrence output (NOT the cumulative weighted sum) by
        # `1 - (1-α)^(t+1)`. This is NOT pandas-canonical EMA-adjust,
        # but it is the documented contract of every existing
        # adaptive_ema test. ops.scan keeps the exact same ordering
        # and intermediate values as the previous Python-for loop;
        # do not switch to ops.associative_scan (different ordering →
        # ~1e-6 float drift → would flake serialization round-trip).
        #
        # Implementation note: TF backend of ops.scan requires the
        # per-iteration output `y` to match the carry in shape/dtype,
        # and forbids tuple carries here. So:
        #   - carry = ema_prev (B, F) only
        #   - per-step weight `1 - (1-α)^(t+1)` is precomputed outside
        #     the scan (depends on t alone) and fed in via xs.
        # This keeps the scan body pure-elementwise on the carry.
        adjust = self.adjust

        # Precompute adjust weights for t = 1..T-1 (length T-1) matching the
        # original loop's `t + 1` exponent. Use static T when available so
        # the graph stays clean; fall back to dynamic shape otherwise.
        T_dynamic = ops.shape(x)[1]
        # t ranges 1..T-1 → exponent ranges 2..T. We build floats for the
        # exponent inside the scan body via `ops.arange`.
        t_arange = ops.arange(1, T_dynamic, dtype=x.dtype)  # (T-1,)
        exponents = t_arange + 1.0  # (T-1,), each is `t + 1`
        weights_1d = 1.0 - ops.power(one_minus_alpha, exponents)
        weights_1d = ops.maximum(weights_1d, ops.cast(1e-10, x.dtype))  # (T-1,)

        # xs leading axis is time-1; pack (x_t, w_t) by concatenating w_t as
        # an extra feature → simpler: feed two scans-ready tensors via a list.
        x_rest_time_major = ops.transpose(x[:, 1:, :], axes=(1, 0, 2))  # (T-1, B, F)
        # Broadcast weight to (T-1, 1, 1) so step body can mul/div via it.
        w_time_major = ops.reshape(weights_1d, (-1, 1, 1))  # (T-1, 1, 1)

        def step(ema_prev, xw):
            # xw is a list [x_t (B,F), w_t (1,1)] — keras packs the lists from
            # the leading axis of each xs element.
            x_t, w_t = xw
            ema_current = alpha * x_t + one_minus_alpha * ema_prev
            ema_adjusted = ema_current / w_t  # (B, F) / (1, 1)
            ema_out = ops.where(
                ops.cast(adjust, "bool"), ema_adjusted, ema_current
            )
            return ema_out, ema_out

        ema_0 = x[:, 0, :]  # (B, F)

        _, ema_rest = ops.scan(step, ema_0, [x_rest_time_major, w_time_major])
        # ema_rest: (T-1, B, F) → (B, T-1, F)
        ema_rest = ops.transpose(ema_rest, axes=(1, 0, 2))
        ema = ops.concatenate([ops.expand_dims(ema_0, axis=1), ema_rest], axis=1)

        if ndim == 2:
            ema = ops.squeeze(ema, axis=-1)

        return ema

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute output shape.

        :param input_shape: Input tensor shape.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape (same as input).
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return layer configuration.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "period": self.period,
            "adjust": self.adjust,
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class EMASlopeFilter(keras.layers.Layer):
    """
    Computes EMA slope and generates trading signals based on slope thresholds.

    The slope is computed as:
        slope_t = EMA_t - EMA_{t - lookback_period}

    Trading signals are generated based on slope conditions:
        - 'above': slope > upper_threshold
        - 'below': slope < lower_threshold
        - 'between': lower_threshold <= slope <= upper_threshold

    The 'between' condition has shown the best risk-adjusted returns in
    certain market conditions, producing favorable net-profit/drawdown ratios.

    **Architecture Overview:**

    .. code-block:: text

        Input: price = [p_0, p_1, ..., p_T]
        Shape: (batch, time_steps) or (batch, time_steps, features)
                            │
                            ▼
               ┌────────────────────────────┐
               │  ExponentialMovingAverage  │
               │      period = ema_period   │
               └─────────────┬──────────────┘
                             │
                             ▼
                EMA = [EMA_0, EMA_1, ..., EMA_T]
                             │
                ┌────────────┴────────────┐
                │                         │
                ▼                         ▼
        ┌───────────────┐       ┌─────────────────────┐
        │  EMA current  │       │  EMA lagged         │
        │  [EMA_t]      │       │  [0, ..., EMA_{t-L}]│
        └───────┬───────┘       │  L = lookback_period│
                │               └──────────┬──────────┘
                │                          │
                └──────────┬───────────────┘
                           │
                           ▼
               ┌───────────────────────┐
               │    Slope Calculation  │
               │  slope_t = EMA_t -    │
               │           EMA_{t-L}   │
               └───────────┬───────────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
    ┌───────────┐    ┌───────────┐    ┌─────────────┐
    │  Above?   │    │  Below?   │    │  Between?   │
    │ slope > U │    │ slope < L │    │ L ≤ slope ≤ │
    │           │    │           │    │      U      │
    └─────┬─────┘    └─────┬─────┘    └──────┬──────┘
          │                │                 │
          ▼                ▼                 ▼
    signal_above     signal_below     signal_between
       (0/1)            (0/1)             (0/1)
                           │
                           ▼
        Output Dictionary (mode="all"):
        {
            "ema": (batch, time, ...),
            "slope": (batch, time, ...),
            "signal_above": (batch, time, ...),
            "signal_below": (batch, time, ...),
            "signal_between": (batch, time, ...)
        }

    :param ema_period: Period for EMA calculation.
    :type ema_period: int
    :param lookback_period: Number of bars to look back for slope calculation.
    :type lookback_period: int
    :param upper_threshold: Upper threshold for slope filtering.
    :type upper_threshold: float
    :param lower_threshold: Lower threshold for slope filtering.
    :type lower_threshold: float
    :param output_mode: Output mode - 'all' (EMA, slope, signals),
        'signals_only', 'slope_only', or 'ema_only'.
    :type output_mode: str
    :param adjust_ema: If True, uses adjusted EMA calculation.
    :type adjust_ema: bool
    :param kwargs: Additional layer arguments.
    """

    def __init__(
        self,
        ema_period: int = 25,
        lookback_period: int = 25,
        upper_threshold: float = 15.0,
        lower_threshold: float = -15.0,
        output_mode: str = "all",
        adjust_ema: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if ema_period < 1:
            raise ValueError(f"ema_period must be >= 1, got {ema_period}")
        if lookback_period < 1:
            raise ValueError(f"lookback_period must be >= 1, got {lookback_period}")
        if lower_threshold > upper_threshold:
            raise ValueError(
                f"lower_threshold ({lower_threshold}) must be <= "
                f"upper_threshold ({upper_threshold})"
            )
        valid_modes = {"all", "signals_only", "slope_only", "ema_only"}
        if output_mode not in valid_modes:
            raise ValueError(f"output_mode must be one of {valid_modes}")

        self.ema_period = ema_period
        self.lookback_period = lookback_period
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.output_mode = output_mode
        self.adjust_ema = adjust_ema

        # Sub-layer created in __init__ per Golden Rule
        self.ema_layer = ExponentialMovingAverage(
            period=ema_period,
            adjust=adjust_ema,
            name="ema",
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer.

        :param input_shape: Input tensor shape.
        :type input_shape: Tuple[Optional[int], ...]
        """
        self.ema_layer.build(input_shape)
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]:
        """
        Compute EMA, slope, and trading signals.

        :param inputs: Input tensor of shape (batch, time_steps, features)
            or (batch, time_steps).
        :type inputs: keras.KerasTensor
        :param training: Training mode flag (unused, for API compatibility).
        :type training: Optional[bool]
        :return: Depending on output_mode:
            - 'all': Dict with 'ema', 'slope', 'signal_above', 'signal_below',
              'signal_between' tensors
            - 'signals_only': Dict with signal tensors only
            - 'slope_only': Slope tensor
            - 'ema_only': EMA tensor
        :rtype: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]
        """
        # Compute EMA
        ema = self.ema_layer(inputs)

        if self.output_mode == "ema_only":
            return ema

        # Compute slope: EMA_t - EMA_{t - lookback_period}
        # Pad with zeros for early timesteps where lookback is not available
        ndim = len(inputs.shape)
        if ndim == 2:
            # (batch, time_steps)
            ema_lagged = ops.concatenate([
                ops.zeros((ops.shape(ema)[0], self.lookback_period), dtype=ema.dtype),
                ema[:, :-self.lookback_period],
            ], axis=1)
        else:
            # (batch, time_steps, features)
            pad_shape = (
                ops.shape(ema)[0],
                self.lookback_period,
                ema.shape[-1] if ema.shape[-1] is not None else ops.shape(ema)[-1],
            )
            ema_lagged = ops.concatenate([
                ops.zeros(pad_shape, dtype=ema.dtype),
                ema[:, :-self.lookback_period, :],
            ], axis=1)

        slope = ema - ema_lagged

        if self.output_mode == "slope_only":
            return slope

        # Generate trading signals
        upper = ops.cast(self.upper_threshold, dtype=slope.dtype)
        lower = ops.cast(self.lower_threshold, dtype=slope.dtype)

        # Signal: 1.0 when condition is met, 0.0 otherwise
        signal_above = ops.cast(slope > upper, dtype=slope.dtype)
        signal_below = ops.cast(slope < lower, dtype=slope.dtype)
        signal_between = ops.cast(
            ops.logical_and(slope >= lower, slope <= upper),
            dtype=slope.dtype,
        )

        if self.output_mode == "signals_only":
            return {
                "signal_above": signal_above,
                "signal_below": signal_below,
                "signal_between": signal_between,
            }

        # output_mode == "all"
        return {
            "ema": ema,
            "slope": slope,
            "signal_above": signal_above,
            "signal_below": signal_below,
            "signal_between": signal_between,
        }

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Union[Tuple[Optional[int], ...], Dict[str, Tuple[Optional[int], ...]]]:
        """
        Compute output shape(s).

        :param input_shape: Input tensor shape.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape(s) depending on output_mode.
        :rtype: Union[Tuple[Optional[int], ...], Dict[str, Tuple[Optional[int], ...]]]
        """
        if self.output_mode in {"ema_only", "slope_only"}:
            return input_shape

        if self.output_mode == "signals_only":
            return {
                "signal_above": input_shape,
                "signal_below": input_shape,
                "signal_between": input_shape,
            }

        # output_mode == "all"
        return {
            "ema": input_shape,
            "slope": input_shape,
            "signal_above": input_shape,
            "signal_below": input_shape,
            "signal_between": input_shape,
        }

    def get_config(self) -> Dict[str, Any]:
        """
        Return layer configuration.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "ema_period": self.ema_period,
            "lookback_period": self.lookback_period,
            "upper_threshold": self.upper_threshold,
            "lower_threshold": self.lower_threshold,
            "output_mode": self.output_mode,
            "adjust_ema": self.adjust_ema,
        })
        return config

# ---------------------------------------------------------------------
