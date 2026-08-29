"""
Exponential moving average and the EMA slope filter.

Two layers live here. ``ExponentialMovingAverage`` smooths a series over its
time axis. ``EMASlopeFilter`` wraps that layer, turns the smoothed series into
a slope, and thresholds the slope into three 0/1 trading signals.

The slope is a plain difference over ``lookback_period`` bars:
``slope_t = EMA_t - EMA_{t-L}``. The signal that motivated the layer is the
"between" one: trading only while the slope sits inside a band (say -15 to
+15) has produced better risk-adjusted returns than trading on the sign of
the slope alone.

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
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.time_series.ema_layer")
class ExponentialMovingAverage(keras.layers.Layer):
    """
    Smooths a series over its time axis with an exponential moving average.

    The recurrence is ``EMA_t = α·x_t + (1-α)·EMA_{t-1}`` with
    ``α = 2 / (period + 1)``. The first step is seeded with the first sample,
    ``EMA_0 = x_0``.

    ``adjust=True`` divides every step after the first by
    ``w_t = 1 - (1-α)^(t+1)``, which pulls the early steps up toward the data.
    The divided value is what carries into the next step, so the correction
    compounds through the recurrence. Read the diagram before changing this:
    it is not pandas' ``ewm(adjust=True)``. See the DECISION anchor in
    ``call``.

    **Architecture Overview:**

    .. code-block:: text

        Input: x [B, T] or [B, T, F]
                       │
                       ▼
        ┌──────────────────────────────────┐
        │ rank 2 → expand to [B, T, 1]     │
        └────────────────┬─────────────────┘
                         │
                         ├─► T == 1: return input unchanged
                         ▼
        ┌──────────────────────────────────┐
        │ EMA_0 = x[:, 0, :]      [B, F]   │
        │ seed only, never divided by w    │
        └────────────────┬─────────────────┘
                         │ carry [B, F]
                         ▼
        ┌──────────────────────────────────┐
        │ ops.scan over t = 1 .. T-1       │
        │   c     = α·x_t + (1-α)·EMA_{t-1}│
        │   w_t   = 1 - (1-α)^(t+1)        │
        │   EMA_t = c / w_t   (adjust)     │
        │   EMA_t = c         (else)       │
        │ EMA_t is the carry, so a divided │
        │ value feeds the next step        │
        └────────────────┬─────────────────┘
                         │ [T-1, B, F]
                         ▼
        ┌──────────────────────────────────┐
        │ transpose to [B, T-1, F], concat │
        │ EMA_0 in front, squeeze rank 2   │
        └────────────────┬─────────────────┘
                         ▼
        Output: EMA, same shape as the input

    Both branches of the ``adjust`` fork are computed every step and selected
    with ``ops.where``, so the flag costs one extra divide either way.

    Input shape:
        ``(batch, time_steps)`` or ``(batch, time_steps, features)``.

    Output shape:
        Same as the input.

    Example:
        >>> import keras
        >>> layer = ExponentialMovingAverage(period=25, adjust=False)
        >>> x = keras.random.normal((4, 128, 3))
        >>> ema = layer(x)
        >>> ema.shape
        (4, 128, 3)

    :param period: EMA period. Sets the smoothing factor ``α = 2/(period+1)``.
        Defaults to 25.
    :type period: int
    :param adjust: If True, divide each step after the first by
        ``1 - (1-α)^(t+1)``. Defaults to True.
    :type adjust: bool
    :param kwargs: Additional arguments for the Layer base class.
    :raises ValueError: If ``period`` is less than 1.
    :ivar alpha: The smoothing factor, ``2 / (period + 1)``.
    :vartype alpha: float
    """

    def __init__(
        self,
        period: int = 25,
        adjust: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the ExponentialMovingAverage layer.

        :param period: EMA period. Sets ``α = 2/(period+1)``. Defaults to 25.
        :type period: int
        :param adjust: If True, apply the ``1 - (1-α)^(t+1)`` division.
            Defaults to True.
        :type adjust: bool
        :param kwargs: Additional arguments for the Layer base class.
        :raises ValueError: If ``period`` is less than 1.
        """
        super().__init__(**kwargs)
        if period < 1:
            raise ValueError(f"period must be >= 1, got {period}")

        self.period = period
        self.adjust = adjust
        self.alpha = 2.0 / (period + 1.0)

    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """
        Compute the EMA over the time dimension.

        ``keras.ops.scan`` runs the recurrence as a single Scan op, which is
        XLA-friendly and ONNX-clean. It gives the same values as the Python
        for-loop this replaced.

        The scan carry is ``ema_prev`` alone, a ``(B, F)`` tensor. The TF
        backend wants the per-step output ``y`` to match the carry in shape
        and dtype, and will not take a tuple carry here. So the step counter
        cannot ride along in the carry. The adjust weight depends only on
        ``t``, so it is precomputed outside the scan and fed in through
        ``xs``. That keeps the scan body elementwise on the carry.

        A one-step series is returned as-is. There is no recurrence to run.

        :param inputs: Input tensor of shape ``(batch, time_steps, features)``
            or ``(batch, time_steps)``.
        :type inputs: keras.KerasTensor
        :return: EMA values, same shape as the input.
        :rtype: keras.KerasTensor
        """
        ndim = len(inputs.shape)

        if ndim == 2:
            # Give a rank-2 input a feature axis: (B, T) -> (B, T, 1).
            x = ops.expand_dims(inputs, axis=-1)
        else:
            x = inputs

        # A one-step series has no recurrence, so return the input verbatim.
        # Read the static shape when it is known; that keeps the graph clean.
        static_T = x.shape[1]
        if static_T == 1:
            return inputs

        alpha = ops.cast(self.alpha, dtype=x.dtype)
        one_minus_alpha = ops.cast(1.0 - self.alpha, dtype=x.dtype)

        # DECISION plan_2026-05-12_5f0e087c/D-001: adjust=True divides each
        # step by `1 - (1-α)^(t+1)`, then feeds the divided value back as the
        # carry. Not pandas: off ewm(span=3, adjust=True) by 5.88 on an 8-step
        # probe, while adjust=False matches to 8.6e-07. Do not "fix" it to
        # pandas and do not use ops.associative_scan. See decisions.md D-001.
        adjust = self.adjust

        # Precompute the adjust weights for t = 1..T-1, one per scan step.
        T_dynamic = ops.shape(x)[1]
        t_arange = ops.arange(1, T_dynamic, dtype=x.dtype)
        # The exponent is `t + 1`, matching the original Python loop.
        exponents = t_arange + 1.0
        weights_1d = 1.0 - ops.power(one_minus_alpha, exponents)
        weights_1d = ops.maximum(weights_1d, ops.cast(1e-10, x.dtype))

        # ops.scan walks the LEADING axis of each xs entry, so the samples go
        # in time-major: (B, T-1, F) -> (T-1, B, F).
        x_rest_time_major = ops.transpose(x[:, 1:, :], axes=(1, 0, 2))
        # Weights become (T-1, 1, 1) so each step's (1, 1) slice broadcasts
        # against the (B, F) carry.
        w_time_major = ops.reshape(weights_1d, (-1, 1, 1))

        def step(ema_prev, xw):
            """
            Advance the EMA by one time step.

            :param ema_prev: Previous EMA value, shape (B, F).
            :param xw: List of the two per-step slices keras takes from the
                leading axis of each xs entry: x_t (B, F) and w_t (1, 1).
            :return: Tuple of (next carry, per-step output), both the new EMA.
            """
            x_t, w_t = xw
            ema_current = alpha * x_t + one_minus_alpha * ema_prev
            ema_adjusted = ema_current / w_t
            ema_out = ops.where(
                ops.cast(adjust, "bool"), ema_adjusted, ema_current
            )
            return ema_out, ema_out

        # Seed the carry with the first sample. It is never divided by w.
        ema_0 = x[:, 0, :]

        _, ema_rest = ops.scan(step, ema_0, [x_rest_time_major, w_time_major])
        # Back to batch-major: (T-1, B, F) -> (B, T-1, F).
        ema_rest = ops.transpose(ema_rest, axes=(1, 0, 2))
        ema = ops.concatenate([ops.expand_dims(ema_0, axis=1), ema_rest], axis=1)

        if ndim == 2:
            ema = ops.squeeze(ema, axis=-1)

        return ema

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape. The EMA never changes shape.

        :param input_shape: Input tensor shape.
        :type input_shape: Tuple[Optional[int], ...]
        :return: The input shape, unchanged.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return the layer configuration for serialization.

        :return: Configuration dictionary carrying ``period`` and ``adjust``.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "period": self.period,
            "adjust": self.adjust,
        })
        return config

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.time_series.ema_layer")
class EMASlopeFilter(keras.layers.Layer):
    """
    Turns a price series into an EMA slope and three 0/1 trading signals.

    The slope is a difference over ``Lb = lookback_period`` bars:
    ``slope_t = EMA_t - EMA_{t-Lb}``. Three signals threshold it against
    ``hi = upper_threshold`` and ``lo = lower_threshold``:

    - ``signal_above``: ``slope > hi``
    - ``signal_below``: ``slope < lo``
    - ``signal_between``: ``lo <= slope <= hi``

    The "between" signal is the one that motivated the layer. Trading only
    inside the band has produced better net-profit/drawdown ratios than
    trading on the sign of the slope.

    ``output_mode`` decides what ``call`` returns, and two of the four modes
    return early, before the work below them runs.

    **Architecture Overview:**

    .. code-block:: text

        Input: price [B, T] or [B, T, F]
                         │
                         ▼
        ┌──────────────────────────────────┐
        │ ExponentialMovingAverage         │
        │   period = ema_period            │
        │   adjust = adjust_ema            │
        └────────────────┬─────────────────┘
                         │ ema, input shape
                         ├─► 'ema_only': return ema
                         ▼
        ┌──────────────────────────────────┐
        │ ema_lagged = Lb zeros, then      │
        │              ema[:, :-Lb]        │
        │ slope = ema - ema_lagged         │
        └────────────────┬─────────────────┘
                         │ slope, input shape
                         ├─► 'slope_only': return slope
                         ▼
        ┌──────────────────────────────────┐
        │ signal_above   = slope > hi      │
        │ signal_below   = slope < lo      │
        │ signal_between = lo<=slope<=hi   │
        │ each cast to slope.dtype, 0 or 1 │
        └────────────────┬─────────────────┘
                         │
              ┌──────────┴───────────┐
              ▼                      ▼
        'signals_only'             'all'
        {signal_above,      {ema, slope,
         signal_below,       signal_above,
         signal_between}     signal_below,
                             signal_between}

    The first ``Lb`` steps are lagged against padded zeros, so their "slope"
    is just ``EMA_t``. Drop or mask them before reading a signal.

    Input shape:
        ``(batch, time_steps)`` or ``(batch, time_steps, features)``.

    Output shape:
        For ``'ema_only'`` and ``'slope_only'``, one tensor of the input
        shape. For ``'signals_only'``, a dict of 3 tensors of the input
        shape. For ``'all'``, a dict of 5 tensors of the input shape.

    Example:
        >>> import keras
        >>> layer = EMASlopeFilter(ema_period=25, lookback_period=25)
        >>> out = layer(keras.random.normal((4, 128)))
        >>> sorted(out.keys())
        ['ema', 'signal_above', 'signal_below', 'signal_between', 'slope']

    :param ema_period: Period passed to the inner EMA layer. Defaults to 25.
    :type ema_period: int
    :param lookback_period: Bars to look back when differencing the EMA.
        Defaults to 25.
    :type lookback_period: int
    :param upper_threshold: Upper slope threshold. Defaults to 15.0.
    :type upper_threshold: float
    :param lower_threshold: Lower slope threshold. Defaults to -15.0.
    :type lower_threshold: float
    :param output_mode: One of ``'all'``, ``'signals_only'``,
        ``'slope_only'``, ``'ema_only'``. Defaults to ``'all'``.
    :type output_mode: str
    :param adjust_ema: Passed through as the inner EMA layer's ``adjust``.
        Defaults to True.
    :type adjust_ema: bool
    :param kwargs: Additional arguments for the Layer base class.
    :raises ValueError: If ``ema_period`` or ``lookback_period`` is less than
        1, if ``lower_threshold`` exceeds ``upper_threshold``, or if
        ``output_mode`` is not one of the four names above.
    :ivar ema_layer: The inner ``ExponentialMovingAverage``, built in
        ``__init__`` and named ``"ema"``.
    :vartype ema_layer: ExponentialMovingAverage
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
        """
        Initialize the EMASlopeFilter layer.

        :param ema_period: Period for the inner EMA layer. Defaults to 25.
        :type ema_period: int
        :param lookback_period: Bars to look back for the slope. Defaults
            to 25.
        :type lookback_period: int
        :param upper_threshold: Upper slope threshold. Defaults to 15.0.
        :type upper_threshold: float
        :param lower_threshold: Lower slope threshold. Defaults to -15.0.
        :type lower_threshold: float
        :param output_mode: One of ``'all'``, ``'signals_only'``,
            ``'slope_only'``, ``'ema_only'``. Defaults to ``'all'``.
        :type output_mode: str
        :param adjust_ema: The inner EMA layer's ``adjust``. Defaults to True.
        :type adjust_ema: bool
        :param kwargs: Additional arguments for the Layer base class.
        :raises ValueError: On a bad period, an inverted threshold pair, or an
            unknown ``output_mode``.
        """
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

        # Sub-layer created in __init__, per the Golden Rule.
        self.ema_layer = ExponentialMovingAverage(
            period=ema_period,
            adjust=adjust_ema,
            name="ema",
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the inner EMA layer, then the layer itself.

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
        Compute the EMA, the slope, and the trading signals.

        The mode decides how far this runs. ``'ema_only'`` returns before the
        slope is built; ``'slope_only'`` returns before the signals are.

        :param inputs: Input tensor of shape ``(batch, time_steps, features)``
            or ``(batch, time_steps)``.
        :type inputs: keras.KerasTensor
        :param training: Training mode flag. Unused, kept for API
            compatibility.
        :type training: Optional[bool]
        :return: One tensor for ``'ema_only'`` (the EMA) and ``'slope_only'``
            (the slope), both of the input shape. A dict of the 3 signal
            tensors for ``'signals_only'``. A dict of ``'ema'``, ``'slope'``
            and the 3 signals for ``'all'``.
        :rtype: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]
        """
        ema = self.ema_layer(inputs)

        if self.output_mode == "ema_only":
            return ema

        # slope_t = EMA_t - EMA_{t-lookback_period}. The first lookback_period
        # steps have no history, so the lag is padded with zeros there and
        # their slope is just EMA_t.
        ndim = len(inputs.shape)
        if ndim == 2:
            # Rank-2 input: (batch, time_steps).
            ema_lagged = ops.concatenate([
                ops.zeros((ops.shape(ema)[0], self.lookback_period), dtype=ema.dtype),
                ema[:, :-self.lookback_period],
            ], axis=1)
        else:
            # Rank-3 input: (batch, time_steps, features).
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

        upper = ops.cast(self.upper_threshold, dtype=slope.dtype)
        lower = ops.cast(self.lower_threshold, dtype=slope.dtype)

        # Each signal is 1.0 where its condition holds and 0.0 elsewhere.
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
        Compute the output shape or shapes. No mode changes the shape.

        :param input_shape: Input tensor shape.
        :type input_shape: Tuple[Optional[int], ...]
        :return: The input shape for ``'ema_only'`` and ``'slope_only'``. A
            dict of 3 copies of it for ``'signals_only'``, 5 for ``'all'``.
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
        Return the layer configuration for serialization.

        :return: Configuration dictionary carrying all six constructor
            arguments.
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
