"""
Forecasting Layers based on Valeriy Manokhin's Scientific Framework.

This module implements three novel layers that embed scientific forecasting
principles directly into neural architectures:

1. **NaiveResidual**: Structural enforcement of the Naive Benchmark Principle
2. **ForecastabilityGate**: Learnable complexity assessment and switching
3. **ConformalQuantileHead**: Built-in support for Conformalized Quantile Regression

These layers implement principles from the forecasting science guide:

- Forecastability Assessment (Section 2)
- Naive Benchmark Principle (Section 8)
- Conformalized Quantile Regression (Section 5)
- Validity-First Hierarchy (Section 6)
"""

import keras
import numpy as np
from keras import ops, layers, initializers, regularizers
from typing import Optional, Union, Tuple, Dict, Any

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class NaiveResidual(layers.Layer):
    """
    Structural implementation of the Naive Benchmark Principle.

    This layer forces the network to learn only the Forecast Value Added (FVA)
    on top of a naive baseline (Random Walk). If network weights decay to zero,
    the model gracefully degrades to a perfect Random Walk forecast rather than
    outputting degraded predictions.

    Mathematical formulation:
        ``output = network_output + naive_forecast``

    where:
        ``naive_forecast = repeat(last_observed_value, forecast_length)``

    This ensures that the network optimizes for FVA rather than raw MSE,
    gradient descent explicitly learns "what beats naive", and the model has
    a guaranteed baseline of competence.

    Reference: Section 8 - The Naive Benchmark Principle

    **Architecture Overview:**

    .. code-block:: text

        Inputs: historical (batch, backcast_len, features)
                network_output (batch, forecast_len, features)
                    │                       │
                    ▼                       │
        ┌───────────────────────┐           │
        │  Extract last value   │           │
        │  inputs[:, -1, :]     │           │
        └───────────┬───────────┘           │
                    │                       │
                    ▼                       │
        ┌───────────────────────┐           │
        │  Repeat forecast_len  │           │
        │  times along axis=1   │           │
        └───────────┬───────────┘           │
                    │                       │
                    ▼                       ▼
            naive_forecast      +    network_output
                    │                       │
                    └───────────┬───────────┘
                                │
                                ▼
                    Output (batch, forecast_len, features)

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

@keras.saving.register_keras_serializable()
class ForecastabilityGate(layers.Layer):
    """
    Learnable gate for weighing deep predictions versus naive forecasts.

    This layer implements a differentiable switch based on input complexity.
    It computes a forecastability score ``alpha in [0, 1]`` where:

    - ``alpha ~ 0``: Input is noisy/unforecastable, favor naive baseline
    - ``alpha ~ 1``: Input has clear patterns, trust deep network

    This prevents "overfitting to noise" common in complex models like
    Transformers when applied to high-entropy data.

    Mathematical formulation:
        ``output = alpha * deep_forecast + (1 - alpha) * naive_forecast``
        ``alpha = ComplexityAnalyzer(inputs)``

    Reference: Section 2 - Forecastability Assessment

    **Architecture Overview:**

    .. code-block:: text

        Inputs: historical (batch, backcast_len, features)
                deep_forecast (batch, forecast_len, features)
                naive_forecast (batch, forecast_len, features)
                    │
                    ▼
        ┌───────────────────────────┐
        │  Flatten                  │
        │  (batch, backcast*feat)   │
        └───────────┬───────────────┘
                    │
                    ▼
        ┌───────────────────────────┐
        │  Dense(hidden_units, act) │
        └───────────┬───────────────┘
                    │
                    ▼
        ┌───────────────────────────┐
        │  Dense(1, sigmoid)        │
        │  alpha in [0, 1]          │
        └───────────┬───────────────┘
                    │
                    ▼
        ┌───────────────────────────────────────────┐
        │  alpha * deep_forecast                    │
        │  + (1 - alpha) * naive_forecast           │
        └─────────────────┬─────────────────────────┘
                          │
                          ▼
                Output (batch, forecast_len, features)

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
        :param name: Layer name.
        :type name: str or None
        :param kwargs: Additional keyword arguments for the base Layer.
        """
        super().__init__(name=name, **kwargs)
        self.hidden_units = hidden_units
        self.activation = activation
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = kernel_regularizer

        # Complexity analyzer will be built in build()
        self.complexity_analyzer = None

    def build(self, input_shape: Union[Tuple, list]):
        """
        Build the complexity analyzer sub-network.

        :param input_shape: Shape of inputs tensor, or list of shapes if
            multiple inputs.
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

    def get_config(self) -> Dict[str, Any]:
        """
        Return the layer configuration for serialization.

        :return: Dictionary containing the layer configuration.
        :rtype: dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "hidden_units": self.hidden_units,
            "activation": self.activation,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ConformalQuantileHead(layers.Layer):
    """
    Output layer designed for Conformalized Quantile Regression (CQR).

    This layer outputs three quantile predictions per feature: lower quantile
    (typically ``alpha/2``, e.g. 0.05 for 90% interval), median (0.50), and
    upper quantile (typically ``1-alpha/2``, e.g. 0.95 for 90% interval).

    The layer includes built-in structure for quantile outputs, a non-trainable
    calibration score Q for conformal prediction, and methods for calibration
    and prediction with valid intervals.

    Mathematical formulation:
        Training: Learn ``q_low``, ``q_median``, ``q_high``
        Calibration: ``Q = Quantile(calibration_scores, 1-alpha)``
        Inference:
            ``final_lower = q_low - Q``
            ``final_upper = q_high + Q``

    Reference: Section 5 - Conformalized Quantile Regression (CQR),
    Section 6 - Validity-First Hierarchy

    **Architecture Overview:**

    .. code-block:: text

        Input: encoded features (batch, input_dim)
                    │
                    ▼
        ┌───────────────────────────────────────┐
        │  Dense(forecast_len * output_dim * 3) │
        └───────────────────┬───────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │  Reshape to                           │
        │  (batch, forecast_len, output_dim, 3) │
        └───────────────────┬───────────────────┘
                            │
                            ▼
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
          q_low [0]    q_median [1]   q_high [2]
              │             │             │
              │      (at inference)       │
              ▼                           ▼
        ┌──────────┐               ┌──────────┐
        │ q_low - Q│               │q_high + Q│
        └──────────┘               └──────────┘
              │             │             │
              ▼             ▼             ▼
        calibrated     median      calibrated
          lower                      upper

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
        self.kernel_regularizer = kernel_regularizer

        # Projection layer will be built in build()
        self.projection = None

        # Non-trainable conformal calibration score Q
        # Initialized to zero, set via calibrate() method
        self.q_hat = None

    def build(self, input_shape: Tuple):
        """
        Build the projection layer and calibration score.

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
        Update the conformal calibration score Q based on calibration data.

        This method should be called after training using a held-out calibration
        set. The calibration scores are the nonconformity scores computed as:
            ``score = max(q_low - y_true, y_true - q_high)``

        The adjusted quantile is ``(1-alpha)(1 + 1/n)`` per Section 7 of the
        forecasting science guide.

        :param calibration_scores: Array of nonconformity scores from
            calibration set with shape ``(n_calibration_samples,)``.
        :type calibration_scores: numpy.ndarray
        :param alpha: Significance level. Defaults to 0.1 for 90% coverage.
        :type alpha: float
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
        Compute calibrated prediction intervals during inference.

        Applies the conformal adjustment to the learned quantiles:
            ``calibrated_lower = model_lower - Q``
            ``calibrated_upper = model_upper + Q``

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
        lower_pred = preds[..., 0]  # Shape: (batch, forecast_len, output_dim)
        median_pred = preds[..., 1]
        upper_pred = preds[..., 2]

        # Apply conformal adjustment
        q = self.q_hat
        lower_calibrated = lower_pred - q
        upper_calibrated = upper_pred + q

        return median_pred, lower_calibrated, upper_calibrated

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
    Create a model that structurally enforces Manokhin's forecasting principles.

    This factory function creates a complete forecasting model with a deep
    network for learning patterns (Efficiency), naive baseline integration
    (Benchmarks), forecastability assessment gate (Noise rejection), and
    uncertainty quantification via quantiles (Validity).

    The model has two outputs: a point forecast (gated combination of deep +
    naive predictions) and a quantile forecast for conformalized prediction
    intervals.

    :param input_shape: Tuple of ``(backcast_length, features)`` for input data.
    :type input_shape: tuple[int, int]
    :param forecast_length: Number of time steps to forecast.
    :type forecast_length: int
    :param hidden_units: Number of hidden units in deep network.
        Defaults to 128.
    :type hidden_units: int
    :param gate_hidden_units: Number of hidden units in forecastability gate.
        Defaults to 16.
    :type gate_hidden_units: int
    :param gate_activation: Activation function for gate. Defaults to ``'relu'``.
    :type gate_activation: str
    :return: Keras Model with inputs of shape ``(batch, backcast_len, features)``
        and outputs ``[point_forecast, quantiles]``.
    :rtype: keras.Model
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
