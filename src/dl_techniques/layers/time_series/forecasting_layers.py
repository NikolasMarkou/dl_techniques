"""
Forecasting layers based on Valeriy Manokhin's forecasting framework.

Three layers and one builder. Each layer puts a forecasting principle into the
architecture instead of leaving it to the loss function.

1. ``NaiveResidual`` adds the network output on top of a random-walk baseline.
2. ``ForecastabilityGate`` mixes a deep forecast with a naive one, using a
   learned weight.
3. ``ConformalQuantileHead`` emits three quantiles and carries a conformal
   offset for calibrated intervals.

``create_manokhin_compliant_model`` wires all three into one model with two
outputs.

Section numbers below refer to the forecasting science guide: forecastability
assessment (Section 2), conformalized quantile regression (Section 5),
validity-first hierarchy (Section 6), naive benchmark principle (Section 8).
"""

import keras
import numpy as np
from keras import ops, layers, initializers, regularizers
from typing import Optional, Union, Tuple, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.activation_serialization import (
    serialize_activation,
    deserialize_activation,
)
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.time_series.forecasting_layers")
class NaiveResidual(layers.Layer):
    """
    Add a learned residual on top of a random-walk baseline.

    The layer repeats the last observed value across the horizon and adds the
    network's output to it. The network therefore only has to learn what beats
    the naive forecast, not the forecast itself. If the network weights decay
    to zero, the layer still emits a clean random walk.

    Formula::

        output = network_output + naive_forecast
        naive_forecast = repeat(inputs[:, -1, :], forecast_length)

    Reference: Section 8 - The Naive Benchmark Principle

    **Architecture Overview:**

    .. code-block:: text

        Both inputs are tensors passed to call(); this layer owns
        no weights.

                inputs                  network_output
            (batch, Tb, feat)        (batch, Tf, feat)
                    │                       │
                    ▼                       │
        ┌───────────────────────┐           │
        │  take last step       │           │
        │  inputs[:, -1, :]     │           │
        │  (batch, feat)        │           │
        └───────────┬───────────┘           │
                    │                       │
                    ▼                       │
        ┌───────────────────────┐           │
        │  repeat Tf times      │           │
        │  along axis 1         │           │
        │  (batch, Tf, feat)    │           │
        └───────────┬───────────┘           │
                    │                       │
                    ▼                       ▼
            naive_forecast      +    network_output
                    │                       │
                    └───────────┬───────────┘
                                │
                                ▼
                    output (batch, Tf, feat)

        Tb = backcast_len, Tf = forecast_length.

    :param forecast_length: Number of time steps to forecast.
    :type forecast_length: int
    :param name: Layer name.
    :type name: str or None
    :param kwargs: Additional keyword arguments passed to the base Layer.
    """

    def __init__(
            self,
            forecast_length: int,
            name: Optional[str] = None,
            **kwargs
    ):
        """
        Initialize the NaiveResidual layer.

        :param forecast_length: Number of time steps to forecast.
        :type forecast_length: int
        :param name: Layer name.
        :type name: str or None
        :param kwargs: Additional keyword arguments for the base Layer.
        """
        super().__init__(name=name, **kwargs)
        self.forecast_length = forecast_length

    def call(
            self,
            inputs: keras.KerasTensor,
            network_output: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Combine network prediction with naive baseline.

        :param inputs: Historical data with shape ``(batch, backcast_len, features)``.
        :type inputs: keras.KerasTensor
        :param network_output: Learned residual with shape
            ``(batch, forecast_len, features)``.
        :type network_output: keras.KerasTensor
        :return: Combined forecast ``network_output + naive_forecast`` with shape
            ``(batch, forecast_len, features)``.
        :rtype: keras.KerasTensor
        """
        # Extract the last observed value (Random Walk baseline)
        # Shape: (batch, features)
        last_observed = inputs[:, -1, :]

        # Project into the future (Naive Forecast)
        # Repeat the last value for the full horizon
        # Shape: (batch, forecast_length, features)
        naive_forecast = ops.repeat(
            ops.expand_dims(last_observed, axis=1),
            self.forecast_length,
            axis=1
        )

        # Add the network's learned "Value Added" to the naive baseline
        return network_output + naive_forecast

    def compute_output_shape(self, input_shape):
        """
        Compute the output shape.

        Two input forms are accepted. A list of two shapes is read as
        ``[inputs, network_output]`` and the second one is used. A single shape
        is used as-is. Only the batch and feature axes are read from it; the
        time axis is always ``forecast_length``.

        :param input_shape: One shape tuple, or a list of the two call shapes.
        :type input_shape: tuple or list
        :return: ``(batch, forecast_length, features)``.
        :rtype: tuple
        """
        if isinstance(input_shape, (list, tuple)) and len(input_shape) == 2 and isinstance(input_shape[0], (list, tuple)):
            net_shape = input_shape[1]
        else:
            net_shape = input_shape
        return (net_shape[0], self.forecast_length, net_shape[-1])

    def get_config(self) -> Dict[str, Any]:
        """
        Return the layer configuration for serialization.

        :return: Dictionary containing the layer configuration.
        :rtype: dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "forecast_length": self.forecast_length,
        })
        return config

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.time_series.forecasting_layers")
class ForecastabilityGate(layers.Layer):
    """
    Blend a deep forecast with a naive one, using a learned weight.

    A small sub-network reads the raw history and emits one scalar
    ``alpha in [0, 1]`` per sample. At ``alpha`` near 0 the input looks like
    noise and the naive baseline wins. At ``alpha`` near 1 the input looks
    patterned and the deep forecast wins. This keeps a high-capacity model from
    fitting noise on high-entropy series.

    Formula::

        alpha = complexity_analyzer(flatten(inputs))
        output = alpha * deep_forecast + (1 - alpha) * naive_forecast

    Reference: Section 2 - Forecastability Assessment

    **Architecture Overview:**

    .. code-block:: text

        call() takes THREE tensors. Only the alpha path owns
        weights. deep_forecast and naive_forecast pass straight
        into the blend.

                 inputs           deep_forecast  naive_forecast
              (B, Tb, feat)       (B, Tf, feat)  (B, Tf, feat)
                    │                    │                  │
                    ▼                    │                  │
        ┌───────────────────────┐        │                  │
        │  flatten              │        │                  │
        │  (B, Tb*feat)         │        │                  │
        └───────────┬───────────┘        │                  │
                    ▼                    │                  │
        ┌───────────────────────┐        │                  │
        │  Dense(hidden_units)  │        │                  │
        │  + activation         │        │                  │
        └───────────┬───────────┘        │                  │
                    ▼                    │                  │
        ┌───────────────────────┐        │                  │
        │  Dense(1, sigmoid)    │        │                  │
        │  alpha (B, 1)         │        │                  │
        └───────────┬───────────┘        │                  │
                    ▼                    │                  │
        ┌───────────────────────┐        │                  │
        │  expand_dims axis=-1  │        │                  │
        │  alpha (B, 1, 1)      │        │                  │
        └───────────┬───────────┘        │                  │
                    ▼                    ▼                  ▼
        ┌────────────────────────────────────────────────────────┐
        │  alpha * deep_forecast + (1 - alpha) * naive_forecast  │
        └────────────────────────────┬───────────────────────────┘
                                     ▼
                          output (B, Tf, feat)

        B = batch, Tb = backcast_len, Tf = forecast_length.

    :param hidden_units: Number of hidden units in complexity analyzer.
        Defaults to 16.
    :type hidden_units: int
    :param activation: Activation function for hidden layer.
        Defaults to ``'relu'``.
    :type activation: str
    :param kernel_initializer: Initializer for kernel weights.
        Defaults to ``'glorot_uniform'``.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param kernel_regularizer: Regularizer for kernel weights.
    :type kernel_regularizer: keras.regularizers.Regularizer or None
    :param forecast_length: Length ``Tf`` of the forecast this gate emits. The
        layer cannot infer it: ``deep_forecast`` and ``naive_forecast`` arrive
        as extra ``call`` arguments, so ``build`` only ever sees the backcast
        shape. Supplying it is what lets ``compute_output_shape`` declare the
        tensor ``call`` really returns. Defaults to ``None``, which keeps the
        older, unreliable behaviour and warns.
    :type forecast_length: int or None
    :param name: Layer name.
    :type name: str or None
    :param kwargs: Additional keyword arguments passed to the base Layer.
    """

    def __init__(
            self,
            hidden_units: int = 16,
            activation: str = 'relu',
            kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            forecast_length: Optional[int] = None,
            name: Optional[str] = None,
            **kwargs
    ):
        """
        Initialize the ForecastabilityGate.

        :param hidden_units: Number of hidden units in complexity analyzer.
        :type hidden_units: int
        :param activation: Activation function for hidden layer.
        :type activation: str
        :param kernel_initializer: Initializer for kernel weights.
        :type kernel_initializer: str or keras.initializers.Initializer
        :param kernel_regularizer: Regularizer for kernel weights.
        :type kernel_regularizer: keras.regularizers.Regularizer or None
        :param forecast_length: Length of the emitted forecast, used solely by
            ``compute_output_shape``. ``None`` keeps the older behaviour.
        :type forecast_length: int or None
        :param name: Layer name.
        :type name: str or None
        :param kwargs: Additional keyword arguments for the base Layer.
        """
        super().__init__(name=name, **kwargs)
        self.hidden_units = hidden_units
        self.activation = deserialize_activation(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.forecast_length = forecast_length

        # Complexity analyzer will be built in build()
        self.complexity_analyzer = None

    def build(self, input_shape: Union[Tuple, list]):
        """
        Build the complexity analyzer sub-network.

        The analyzer is a two-layer ``keras.Sequential``: ``hidden_units`` wide
        with the configured activation, then one sigmoid unit. It is built on
        the flattened history shape ``(batch, backcast_len * features)``. If a
        list of shapes arrives, only the first one is used.

        :param input_shape: Shape of the ``inputs`` tensor, or a list of the
            call shapes, in which case the first is used.
        :type input_shape: tuple or list
        """
        # Handle multiple input shapes (inputs, deep_forecast, naive_forecast)
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        # Tiny sub-network to estimate forecastability
        # Input: flattened time series
        # Output: scalar alpha in [0, 1]
        self.complexity_analyzer = keras.Sequential([
            layers.Dense(
                self.hidden_units,
                activation=self.activation,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name='complexity_hidden'
            ),
            layers.Dense(
                1,
                activation='sigmoid',
                kernel_initializer=self.kernel_initializer,
                name='complexity_gate'
            )
        ], name='complexity_analyzer')

        # Build the analyzer with flattened input shape
        flat_shape = (input_shape[0], input_shape[1] * input_shape[2])
        self.complexity_analyzer.build(flat_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            deep_forecast: keras.KerasTensor,
            naive_forecast: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Compute gated combination of deep and naive forecasts.

        :param inputs: Raw historical data with shape
            ``(batch, backcast_len, features)``.
        :type inputs: keras.KerasTensor
        :param deep_forecast: Output from deep network with shape
            ``(batch, forecast_len, features)``.
        :type deep_forecast: keras.KerasTensor
        :param naive_forecast: Output from naive baseline with shape
            ``(batch, forecast_len, features)``.
        :type naive_forecast: keras.KerasTensor
        :param training: Whether in training mode.
        :type training: bool or None
        :return: Weighted combination with shape
            ``(batch, forecast_len, features)``.
        :rtype: keras.KerasTensor
        """
        # Flatten input for complexity analysis
        # Shape: (batch, backcast_len * features)
        flat_input = ops.reshape(inputs, (ops.shape(inputs)[0], -1))

        # Compute forecastability score alpha
        # alpha -> 1.0 if signal is strong (trust deep model)
        # alpha -> 0.0 if noisy (trust naive baseline)
        # Shape: (batch, 1)
        alpha = self.complexity_analyzer(flat_input, training=training)

        # Broadcast alpha to match forecast shape
        # Shape: (batch, 1, 1)
        alpha = ops.expand_dims(alpha, axis=-1)

        # Weighted average
        # If alpha = 0 (pure noise) -> return naive forecast
        # If alpha = 1 (clear pattern) -> return deep forecast
        return (alpha * deep_forecast) + ((1.0 - alpha) * naive_forecast)

    def compute_output_shape(self, input_shape):
        """
        Compute the output shape, which is the FORECAST shape.

        Keras hands this method a single shape under the functional API,
        because ``deep_forecast`` and ``naive_forecast`` are extra ``call``
        arguments rather than part of ``inputs`` — so the only shape available
        is the backcast one, ``(batch, backcast_len, features)``. With
        ``forecast_length`` supplied to the constructor the time axis is
        replaced by it, and both the single-shape and the list-of-three forms
        return ``(batch, forecast_length, features)``, which is exactly what
        ``call`` produces. On a model built with ``input_shape=(24, 3)`` and
        ``forecast_length=8`` the graph now declares ``(None, 8, 3)`` and
        ``predict`` returns ``(5, 8, 3)`` for a batch of 5.

        With ``forecast_length=None`` — the pre-existing constructor form —
        the older behaviour is kept and one warning is logged: the single-shape
        branch then returns the backcast shape, so ``model.output_shape`` is
        not the shape the model really emits and the runtime shape must be
        read instead. Passing ``forecast_length`` removes the caveat.

        The FEATURE axis is carried through unchanged from the shape this
        method is handed, so a caller whose ``deep_forecast`` carries a
        different feature count than the backcast will see a declared shape
        that follows the input rather than the forecast branch.

        :param input_shape: One shape tuple, or a list of the three call shapes.
        :type input_shape: tuple or list
        :return: ``(batch, forecast_length, features)`` when ``forecast_length``
            is set; otherwise the deep forecast shape for the list form, or
            ``input_shape`` unchanged for the single-shape form.
        :rtype: tuple
        """
        is_call_shape_list = (
            isinstance(input_shape, (list, tuple))
            and len(input_shape) == 3
            and isinstance(input_shape[0], (list, tuple))
        )
        reference = input_shape[1] if is_call_shape_list else input_shape

        if self.forecast_length is not None:
            reference = tuple(reference)
            return (reference[0], self.forecast_length) + reference[2:]

        if not is_call_shape_list:
            # DECISION plan-2026-08-30T020716-ebbaf641/D-004: warn and return
            # the backcast shape. Do NOT promote this to a raise, however wrong
            # the returned shape is: create_manokhin_compliant_model builds a
            # functional graph whose construction calls this method, so raising
            # breaks model construction, not merely shape introspection.
            logger.warning(
                "ForecastabilityGate.compute_output_shape was given only the "
                "backcast shape %s and has no forecast_length, so it is "
                "returning that shape unchanged. The layer actually emits "
                "(batch, forecast_length, features). Pass forecast_length=<Tf> "
                "to the constructor to declare the real output shape.",
                tuple(reference),
            )
        return reference

    def get_config(self) -> Dict[str, Any]:
        """
        Return the layer configuration for serialization.

        :return: Dictionary containing the layer configuration.
        :rtype: dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "hidden_units": self.hidden_units,
            "activation": serialize_activation(self.activation),
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "forecast_length": self.forecast_length,
        })
        return config

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.time_series.forecasting_layers")
class ConformalQuantileHead(layers.Layer):
    """
    Quantile output layer for Conformalized Quantile Regression (CQR).

    The layer projects encoded features to three quantiles per time step and
    feature: a lower quantile (typically ``alpha/2``, e.g. 0.05 for a 90%
    interval), the median (0.50), and an upper quantile (typically
    ``1 - alpha/2``, e.g. 0.95).

    It also carries a non-trainable scalar ``q_hat``, the conformal offset.
    ``call()`` never applies it. Only ``predict_intervals()`` does, and only
    after ``calibrate()`` has set it from a held-out set. Until then ``q_hat``
    is 0 and the intervals are the raw learned quantiles.

    Reference: Section 5 - Conformalized Quantile Regression (CQR),
    Section 6 - Validity-First Hierarchy

    **Architecture Overview:**

    .. code-block:: text

        This is the call() path. No conformal offset here.

        input: encoded features (B, input_dim)
                           │
                           ▼
        ┌─────────────────────────────────────┐
        │  Dense(Tf * output_dim * 3)         │
        └──────────────────┬──────────────────┘
                           ▼
        ┌─────────────────────────────────────┐
        │  reshape to (B, Tf, output_dim, 3)  │
        └──────────────────┬──────────────────┘
                           ▼
             output (B, Tf, output_dim, 3)

        B = batch, Tf = forecast_length. On the last axis,
        0 = lower, 1 = median, 2 = upper.

    **Calibration path:**

    .. code-block:: text

        calibrate() writes the non-trainable weight q_hat:

            q_hat = quantile(scores, clip(p, 0, 1))
            p     = (1 - alpha) * (1 + 1/n)

        predict_intervals() then forks the three channels:

                      preds (B, Tf, out, 3)
                                │
                ┌───────────────┴───────────────┐
                ▼               ▼               ▼
           preds[..., 0]   preds[..., 1]   preds[..., 2]
                │               │               │
                ▼               │               ▼
        ┌───────────────┐       │       ┌───────────────┐
        │ lower - q_hat │       │       │ upper + q_hat │
        └───────┬───────┘       │       └───────┬───────┘
                ▼               ▼               ▼
              lower          median           upper

        The returned tuple is (median, lower, upper), in that
        order. Each element has shape (B, Tf, output_dim).

    :param forecast_length: Number of time steps to forecast.
    :type forecast_length: int
    :param output_dim: Number of features to predict.
    :type output_dim: int
    :param kernel_initializer: Initializer for projection layer weights.
        Defaults to ``'glorot_uniform'``.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param kernel_regularizer: Regularizer for projection layer weights.
    :type kernel_regularizer: keras.regularizers.Regularizer or None
    :param name: Layer name.
    :type name: str or None
    :param kwargs: Additional keyword arguments passed to the base Layer.
    """

    def __init__(
            self,
            forecast_length: int,
            output_dim: int,
            kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            name: Optional[str] = None,
            **kwargs
    ):
        """
        Initialize the ConformalQuantileHead.

        :param forecast_length: Number of time steps to forecast.
        :type forecast_length: int
        :param output_dim: Number of features to predict.
        :type output_dim: int
        :param kernel_initializer: Initializer for projection layer weights.
        :type kernel_initializer: str or keras.initializers.Initializer
        :param kernel_regularizer: Regularizer for projection layer weights.
        :type kernel_regularizer: keras.regularizers.Regularizer or None
        :param name: Layer name.
        :type name: str or None
        :param kwargs: Additional keyword arguments for the base Layer.
        """
        super().__init__(name=name, **kwargs)
        self.forecast_length = forecast_length
        self.output_dim = output_dim
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

        # Projection layer will be built in build()
        self.projection = None

        # Non-trainable conformal calibration score Q
        # Initialized to zero, set via calibrate() method
        self.q_hat = None

    def build(self, input_shape: Tuple):
        """
        Build the projection layer and the calibration weight.

        The projection is one ``Dense`` of width
        ``forecast_length * output_dim * 3``. The calibration weight ``q_hat``
        has shape ``(1,)``, starts at zero and is not trainable; ``calibrate()``
        writes it.

        :param input_shape: Shape of input tensor ``(batch_size, input_dim)``.
        :type input_shape: tuple
        """
        # Output 3 quantiles per (time_step, feature)
        total_outputs = self.forecast_length * self.output_dim * 3

        self.projection = layers.Dense(
            total_outputs,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name='quantile_projection'
        )
        self.projection.build(input_shape)

        # Initialize conformal calibration score
        self.q_hat = self.add_weight(
            name="conformal_q",
            shape=(1,),
            initializer=initializers.Zeros(),
            trainable=False,
            dtype=self.dtype
        )

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Project inputs to quantile predictions.

        :param inputs: Encoded features with shape ``(batch, input_dim)``.
        :type inputs: keras.KerasTensor
        :param training: Whether in training mode.
        :type training: bool or None
        :return: Quantile predictions with shape
            ``(batch, forecast_len, output_dim, 3)`` where the last dimension
            contains ``[lower_quantile, median, upper_quantile]``.
        :rtype: keras.KerasTensor
        """
        # Project to all quantile outputs
        # Shape: (batch, forecast_len * output_dim * 3)
        x = self.projection(inputs, training=training)

        # Reshape to separate time, features, and quantiles
        # Shape: (batch, forecast_len, output_dim, 3)
        # 3 channels: 0=lower(alpha/2), 1=median(0.5), 2=upper(1-alpha/2)
        x = ops.reshape(
            x,
            (-1, self.forecast_length, self.output_dim, 3)
        )

        return x

    def calibrate(
            self,
            calibration_scores: np.ndarray,
            alpha: float = 0.1
    ) -> None:
        """
        Set the conformal offset ``q_hat`` from calibration data.

        Call this after training, on a held-out calibration set. Each score is
        a nonconformity score, ``max(q_low - y_true, y_true - q_high)``.

        The quantile taken is ``(1 - alpha) * (1 + 1/n)``, clipped to
        ``[0, 1]``. The ``1 + 1/n`` term is the finite-sample correction from
        Section 7 of the forecasting science guide. This runs in NumPy and
        writes a Keras weight, so it is not part of a traced graph.

        :param calibration_scores: Nonconformity scores from the calibration
            set, with shape ``(n_calibration_samples,)``.
        :type calibration_scores: numpy.ndarray
        :param alpha: Significance level. Defaults to 0.1 for 90% coverage.
        :type alpha: float
        :return: Nothing; ``q_hat`` is updated in place.
        :rtype: None
        """
        n = len(calibration_scores)
        # Compute adjusted quantile for finite-sample coverage
        # Formula from Section 7: (1-alpha)(1 + 1/n)
        adjusted_quantile = (1.0 - alpha) * (1.0 + 1.0 / n)

        # Clip to [0, 1] to handle edge cases
        adjusted_quantile = np.clip(adjusted_quantile, 0.0, 1.0)

        # Compute the calibration threshold
        q_value = np.quantile(calibration_scores, adjusted_quantile)

        # Update the non-trainable weight
        self.q_hat.assign([q_value])

    def predict_intervals(
            self,
            inputs: Union[keras.KerasTensor, np.ndarray]
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]:
        """
        Compute calibrated prediction intervals at inference time.

        Runs ``call()`` with ``training=False``, then widens the interval by the
        conformal offset::

            calibrated_lower = model_lower - q_hat
            calibrated_upper = model_upper + q_hat

        The median is returned unchanged. Note the return order: the median
        comes first, not the lower bound. If ``calibrate()`` has not run,
        ``q_hat`` is 0 and the bounds are the raw learned quantiles.

        :param inputs: Encoded features with shape ``(batch, input_dim)``.
        :type inputs: keras.KerasTensor or numpy.ndarray
        :return: Tuple of (median, lower, upper) predictions, each with shape
            ``(batch, forecast_len, output_dim)``.
        :rtype: tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]
        """
        # Get uncalibrated quantile predictions
        # Shape: (batch, forecast_len, output_dim, 3)
        preds = self.call(inputs, training=False)

        # Extract quantiles
        # Each has shape (batch, forecast_len, output_dim)
        lower_pred = preds[..., 0]
        median_pred = preds[..., 1]
        upper_pred = preds[..., 2]

        # Apply conformal adjustment
        q = self.q_hat
        lower_calibrated = lower_pred - q
        upper_calibrated = upper_pred + q

        return median_pred, lower_calibrated, upper_calibrated

    def compute_output_shape(self, input_shape):
        """
        Compute the output shape.

        Only the batch axis is read from ``input_shape``; the rest comes from
        the constructor arguments.

        :param input_shape: Shape of the input tensor ``(batch, input_dim)``.
        :type input_shape: tuple
        :return: ``(batch, forecast_length, output_dim, 3)``.
        :rtype: tuple
        """
        return (input_shape[0], self.forecast_length, self.output_dim, 3)

    def get_config(self) -> Dict[str, Any]:
        """
        Return the layer configuration for serialization.

        :return: Dictionary containing the layer configuration.
        :rtype: dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "forecast_length": self.forecast_length,
            "output_dim": self.output_dim,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
        })
        return config

# ---------------------------------------------------------------------

def create_manokhin_compliant_model(
        input_shape: Tuple[int, int],
        forecast_length: int,
        hidden_units: int = 128,
        gate_hidden_units: int = 16,
        gate_activation: str = 'relu'
) -> keras.Model:
    """
    Build a forecasting model that wires all three layers of this module.

    The model has one input and two outputs. Output 1 is a point forecast: a
    small MLP produces a deep forecast, ``NaiveResidual`` produces a pure naive
    baseline, and ``ForecastabilityGate`` blends them. Output 2 is a quantile
    forecast from ``ConformalQuantileHead``, for conformal intervals.

    The hidden representation ``x`` from ``deep_hidden`` is shared: both the
    deep forecast branch and the quantile head read it.

    ``NaiveResidual`` is fed ``zeros_like(deep_forecast)`` as its network
    output, so it returns the naive baseline alone. Only the shape and dtype of
    the deep forecast reach it, not its values.

    **Architecture Overview:**

    .. code-block:: text

           input (B, Tb, feat)
                    │
                    ▼
        ┌───────────────────────┐
        │  Flatten              │
        │  (B, Tb*feat)         │
        └───────────┬───────────┘
                    ▼
        ┌───────────────────────┐
        │  Dense(hidden_units)  │
        │  relu  -> x           │
        │  (B, hidden_units)    │
        └───────────┬───────────┘
                    ├──────────────────────────────────┐
                    ▼                                  │
        ┌───────────────────────┐                      │
        │  Dense(Tf*feat)       │                      │
        │  reshape              │                      │
        │  deep (B, Tf, feat)   │                      │
        └───────────┬───────────┘                      │
                    ├────────────────────┐             │
                    ▼                    │             │
        ┌───────────────────────┐        │             │
        │  NaiveResidual        │        │             │
        │  input + zeros_like   │        │             │
        │  naive (B, Tf, feat)  │        │             │
        └───────────┬───────────┘        │             │
                    ▼                    ▼             │
        ┌──────────────────────────────────────────┐   │
        │  ForecastabilityGate                     │   │
        │  also reads input                        │   │
        │  point (B, Tf, feat)                     │   │
        └───────────┬──────────────────────────────┘   │
                    ▼                                  ▼
          final_point_forecast             ┌───────────────────────┐
              (B, Tf, feat)                │  ConformalQuantile    │
            = model output 1               │  Head                 │
                                           │  reads x (shared with │
                                           │  the deep branch)     │
                                           │  (B, Tf, feat, 3)     │
                                           └───────────┬───────────┘
                                                       ▼
                                                   quantiles
                                               (B, Tf, feat, 3)
                                               = model output 2

        B = batch, Tb = input_shape[0], feat = input_shape[1],
        Tf = forecast_length.

    :param input_shape: Tuple of ``(backcast_length, features)`` for input data.
        Both entries must be concrete integers; ``features`` is read directly to
        size the deep output and the quantile head.
    :type input_shape: tuple[int, int]
    :param forecast_length: Number of time steps to forecast.
    :type forecast_length: int
    :param hidden_units: Width of the shared ``deep_hidden`` Dense layer.
        Defaults to 128.
    :type hidden_units: int
    :param gate_hidden_units: Width of the gate's complexity analyzer.
        Defaults to 16.
    :type gate_hidden_units: int
    :param gate_activation: Hidden activation for the gate's complexity
        analyzer. Defaults to ``'relu'``.
    :type gate_activation: str
    :return: A ``keras.Model`` named ``manokhin_compliant_forecaster``. It takes
        ``(batch, backcast_length, features)`` and returns two tensors: a point
        forecast ``(batch, forecast_length, features)`` and quantiles
        ``(batch, forecast_length, features, 3)``.
    :rtype: keras.Model

    Note:
        The gate is constructed with ``forecast_length``, so the graph declares
        output 1 as ``(None, Tf, feat)`` — the shape it really emits. Measured
        with ``input_shape=(24, 3)`` and ``forecast_length=8``:
        ``model.outputs[0]`` reports ``(None, 8, 3)`` and ``predict`` on 5
        samples returns ``(5, 8, 3)``.

    Note:
        The returned model does not currently survive a ``.keras`` save/load
        round-trip. ``model.save(path)`` succeeds, but ``keras.saving.
        load_model(path)`` raises ``TypeError: <class 'keras.src.models.
        functional.Functional'> could not be deserialized properly ... Exception
        encountered: 'NoneType' object is not subscriptable``, because the saved
        Functional config comes back empty (``'config': {}``,
        ``'build_config': {'input_shape': None}``). This is not caused by the
        gate's ``forecast_length`` argument: the same graph rebuilt with the
        gate constructed WITHOUT ``forecast_length`` fails identically, so the
        defect predates it. Persist weights instead — ``model.save_weights`` /
        ``model.load_weights`` onto a freshly built model — until it is fixed.
    """
    # Input
    inputs = keras.Input(shape=input_shape, name='input')

    # ===== 1. Deep Model (Efficiency) =====
    # Learn complex patterns if they exist
    x = layers.Flatten(name='flatten')(inputs)
    x = layers.Dense(
        hidden_units,
        activation='relu',
        name='deep_hidden'
    )(x)
    deep_forecast = layers.Dense(
        forecast_length * input_shape[1],
        name='deep_output'
    )(x)
    deep_forecast = ops.reshape(
        deep_forecast,
        (-1, forecast_length, input_shape[1])
    )

    # ===== 2. Naive Baseline (Benchmarks) =====
    # Compute pure naive forecast for the gate
    naive_layer = NaiveResidual(
        forecast_length,
        name='naive_residual'
    )
    # Pass zero network output to get pure naive baseline
    pure_naive = naive_layer(
        inputs,
        ops.zeros_like(deep_forecast)
    )

    # ===== 3. Forecastability Assessment (Gate) =====
    # Decide if deep model adds value or just predicts noise
    # If noise, suppress deep_forecast and favor naive baseline
    gate = ForecastabilityGate(
        hidden_units=gate_hidden_units,
        activation=gate_activation,
        forecast_length=forecast_length,
        name='forecastability_gate'
    )
    final_point_forecast = gate(inputs, deep_forecast, pure_naive)

    # ===== 4. Uncertainty Quantification (Validity) =====
    # Project to quantiles for conformal prediction
    quantile_head = ConformalQuantileHead(
        forecast_length=forecast_length,
        output_dim=input_shape[1],
        name='quantile_head'
    )
    quantiles = quantile_head(x)

    # Create model with both outputs
    model = keras.Model(
        inputs=inputs,
        outputs=[final_point_forecast, quantiles],
        name='manokhin_compliant_forecaster'
    )

    return model

# ---------------------------------------------------------------------
