"""
Adaptive EMA Slope Filter Layer.

This module implements an adaptive EMA-based slope filter for time series
trading signal generation, inspired by the strategy of filtering trades
based on EMA slope conditions.

The key insight is that trading when the EMA slope is within certain bounds
(e.g., between -15 and +15) can produce better risk-adjusted returns than
trading only on positive or negative slopes.

References:
    - EMA slope = EMA(current) - EMA(lookback_period bars ago)
    - Trade signals based on slope thresholds (above, below, between)
"""

import keras
from keras import ops
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.time_series.ema_layer import ExponentialMovingAverage
from dl_techniques.layers.time_series.quantile_head_variable_io import QuantileSequenceHead

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class AdaptiveEMASlopeFilterModel(keras.Model):
    """
    Model wrapper for adaptive EMA slope filtering with learnable thresholds.

    This model can optionally learn the optimal slope thresholds during training
    instead of using fixed values. The thresholds are constrained to maintain
    lower_threshold <= upper_threshold.

    Optionally integrates a QuantileSequenceHead for probabilistic slope
    forecasting, providing uncertainty quantification on the slope values.

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
           EMA current              EMA lagged
                │                         │
                └──────────┬──────────────┘
                           │
                           ▼
               ┌───────────────────────┐
               │    Slope Calculation  │
               │  slope_t = EMA_t -    │
               │           EMA_{t-L}   │
               └───────────┬───────────┘
                           │
            ┌──────────────┴──────────────┐
            │                             │
            ▼                             ▼
        (if quantile_head_config)    (threshold path)
            │                             │
            ▼                             ▼
    ┌───────────────────┐    ┌──────────────────────────────┐
    │ QuantileSequence  │    │    Learnable Thresholds      │
    │      Head         │    │  upper_threshold (trainable) │
    │                   │    │  lower_threshold (trainable) │
    │ Outputs quantiles │    │                              │
    │ of slope values   │    │  Constraint: lower ≤ upper   │
    │ [q_0.1, q_0.5,    │    │  via midpoint + softplus     │
    │  q_0.9, ...]      │    └──────────────┬───────────────┘
    └─────────┬─────────┘                   │
              │              ┌──────────────┼────────────────┐
              │              │              │                │
              │              ▼              ▼                ▼
              │        ┌───────────┐  ┌───────────┐  ┌─────────────┐
              │        │  Training │  │  Training │  │  Training   │
              │        │  (soft)   │  │  (soft)   │  │  (soft)     │
              │        │ σ(slope-U)│  │ σ(L-slope)│  │ 1-above-    │
              │        │    /T     │  │    /T     │  │   below     │
              │        ├───────────┤  ├───────────┤  ├─────────────┤
              │        │ Inference │  │ Inference │  │ Inference   │
              │        │  (hard)   │  │  (hard)   │  │  (hard)     │
              │        │ slope > U │  │ slope < L │  │ L≤slope≤U   │
              │        └─────┬─────┘  └─────┬─────┘  └──────┬──────┘
              │              │              │               │
              ▼              ▼              ▼               ▼
        slope_quantiles signal_above signal_below  signal_between
         (optional)
                           │
                           ▼
        Output Dictionary:
        {
            "ema": (batch, time, ...),
            "slope": (batch, time, ...),
            "signal_above": (batch, time, ...),
            "signal_below": (batch, time, ...),
            "signal_between": (batch, time, ...),
            "upper_threshold": scalar,
            "lower_threshold": scalar,
            "slope_quantiles": (batch, time, num_quantiles)  # if enabled
        }

    :param ema_period: Period for EMA calculation.
    :type ema_period: int
    :param lookback_period: Number of bars to look back for slope calculation.
    :type lookback_period: int
    :param initial_upper_threshold: Initial upper threshold value.
    :type initial_upper_threshold: float
    :param initial_lower_threshold: Initial lower threshold value.
    :type initial_lower_threshold: float
    :param learnable_thresholds: If True, thresholds are trainable parameters.
    :type learnable_thresholds: bool
    :param adjust_ema: If True, uses adjusted EMA calculation.
    :type adjust_ema: bool
    :param quantile_head_config: Optional configuration dict for QuantileSequenceHead.
        If provided, enables probabilistic slope forecasting. Expected keys:
        - num_quantiles (int): Number of quantiles to predict (required).
        - dropout_rate (float): Dropout rate, default 0.1.
        - enforce_monotonicity (bool): Enforce quantile ordering, default True.
        - use_bias (bool): Use bias in projection, default True.
    :type quantile_head_config: Optional[Dict[str, Any]]
    :param kwargs: Additional model arguments.
    """

    def __init__(
        self,
        ema_period: int = 25,
        lookback_period: int = 25,
        initial_upper_threshold: float = 15.0,
        initial_lower_threshold: float = -15.0,
        learnable_thresholds: bool = False,
        adjust_ema: bool = True,
        quantile_head_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if ema_period < 1:
            raise ValueError(f"ema_period must be >= 1, got {ema_period}")
        if lookback_period < 1:
            raise ValueError(f"lookback_period must be >= 1, got {lookback_period}")
        if initial_lower_threshold > initial_upper_threshold:
            raise ValueError(
                f"initial_lower_threshold ({initial_lower_threshold}) must be <= "
                f"initial_upper_threshold ({initial_upper_threshold})"
            )

        self.ema_period = ema_period
        self.lookback_period = lookback_period
        self.initial_upper_threshold = initial_upper_threshold
        self.initial_lower_threshold = initial_lower_threshold
        self.learnable_thresholds = learnable_thresholds
        self.adjust_ema = adjust_ema
        self.quantile_head_config = quantile_head_config

        # EMA layer
        self.ema_layer = ExponentialMovingAverage(
            period=ema_period,
            adjust=adjust_ema,
            name="ema",
        )

        # Optional quantile head for probabilistic slope forecasting
        if quantile_head_config is not None:
            if "num_quantiles" not in quantile_head_config:
                raise ValueError(
                    "quantile_head_config must contain 'num_quantiles' key"
                )

            self.quantile_head = QuantileSequenceHead(
                num_quantiles=quantile_head_config["num_quantiles"],
                dropout_rate=quantile_head_config.get("dropout_rate", 0.1),
                enforce_monotonicity=quantile_head_config.get(
                    "enforce_monotonicity", True
                ),
                use_bias=quantile_head_config.get("use_bias", True),
                name="slope_quantile_head",
            )
        else:
            self.quantile_head = None

        self.upper_threshold_var = None
        self.lower_threshold_var = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the model and create threshold variables.

        :param input_shape: Input tensor shape.
        :type input_shape: Tuple[Optional[int], ...]
        """
        self.ema_layer.build(input_shape)

        # Create threshold variables
        self.upper_threshold_var = self.add_weight(
            name="upper_threshold",
            shape=(),
            initializer=keras.initializers.Constant(self.initial_upper_threshold),
            trainable=self.learnable_thresholds,
            dtype=self.dtype,
        )
        self.lower_threshold_var = self.add_weight(
            name="lower_threshold",
            shape=(),
            initializer=keras.initializers.Constant(self.initial_lower_threshold),
            trainable=self.learnable_thresholds,
            dtype=self.dtype,
        )

        # Build quantile head if configured
        if self.quantile_head is not None:
            # Quantile head expects (batch, time, features)
            # Slope output has same time dimension, feature dim = 1 or original
            ndim = len(input_shape)
            if ndim == 2:
                quantile_input_shape = (input_shape[0], input_shape[1], 1)
            else:
                quantile_input_shape = input_shape
            self.quantile_head.build(quantile_input_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> Dict[str, keras.KerasTensor]:
        """
        Compute EMA, slope, and trading signals with potentially learnable thresholds.

        :param inputs: Input tensor of shape (batch, time_steps, features)
            or (batch, time_steps).
        :type inputs: keras.KerasTensor
        :param training: Training mode flag.
        :type training: Optional[bool]
        :return: Dict with 'ema', 'slope', 'signal_above', 'signal_below',
            'signal_between', 'upper_threshold', 'lower_threshold' tensors.
            If quantile_head_config is set, also includes 'slope_quantiles'.
        :rtype: Dict[str, keras.KerasTensor]
        """
        # Compute EMA
        ema = self.ema_layer(inputs)

        # Compute slope
        ndim = len(inputs.shape)
        if ndim == 2:
            ema_lagged = ops.concatenate([
                ops.zeros((ops.shape(ema)[0], self.lookback_period), dtype=ema.dtype),
                ema[:, :-self.lookback_period],
            ], axis=1)
        else:
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

        # Ensure lower <= upper via softplus parameterization if learnable
        if self.learnable_thresholds:
            # Use midpoint + half_range parameterization
            midpoint = (self.upper_threshold_var + self.lower_threshold_var) / 2.0
            half_range = ops.softplus(
                ops.abs(self.upper_threshold_var - self.lower_threshold_var) / 2.0
            )
            upper = midpoint + half_range
            lower = midpoint - half_range
        else:
            upper = self.upper_threshold_var
            lower = self.lower_threshold_var

        upper = ops.cast(upper, dtype=slope.dtype)
        lower = ops.cast(lower, dtype=slope.dtype)

        # Generate signals (soft during training if learnable, hard otherwise)
        if training and self.learnable_thresholds:
            # Soft signals using sigmoid for gradient flow
            temperature = 1.0
            signal_above = ops.sigmoid((slope - upper) / temperature)
            signal_below = ops.sigmoid((lower - slope) / temperature)
            signal_between = 1.0 - signal_above - signal_below
            signal_between = ops.maximum(signal_between, 0.0)
        else:
            # Hard signals
            signal_above = ops.cast(slope > upper, dtype=slope.dtype)
            signal_below = ops.cast(slope < lower, dtype=slope.dtype)
            signal_between = ops.cast(
                ops.logical_and(slope >= lower, slope <= upper),
                dtype=slope.dtype,
            )

        outputs = {
            "ema": ema,
            "slope": slope,
            "signal_above": signal_above,
            "signal_below": signal_below,
            "signal_between": signal_between,
            "upper_threshold": upper,
            "lower_threshold": lower,
        }

        # Compute slope quantiles if quantile head is configured
        if self.quantile_head is not None:
            # Ensure slope is 3D for quantile head
            if ndim == 2:
                slope_3d = ops.expand_dims(slope, axis=-1)
            else:
                slope_3d = slope
            slope_quantiles = self.quantile_head(slope_3d, training=training)
            outputs["slope_quantiles"] = slope_quantiles

        return outputs

    def get_config(self) -> Dict[str, Any]:
        """
        Return model configuration.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "ema_period": self.ema_period,
            "lookback_period": self.lookback_period,
            "initial_upper_threshold": self.initial_upper_threshold,
            "initial_lower_threshold": self.initial_lower_threshold,
            "learnable_thresholds": self.learnable_thresholds,
            "adjust_ema": self.adjust_ema,
            "quantile_head_config": self.quantile_head_config,
        })
        return config


# ---------------------------------------------------------------------
