"""
Forecasting Layers based on Valeriy Manokhin's Scientific Framework.

This module implements three novel layers that embed scientific forecasting principles
directly into neural architectures:

1. NaiveResidual: Structural enforcement of the Naive Benchmark Principle
2. ForecastabilityGate: Learnable complexity assessment and switching
3. ConformalQuantileHead: Built-in support for Conformalized Quantile Regression

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
        output = network_output + naive_forecast

    where:
        naive_forecast = repeat(last_observed_value, forecast_length)

    This ensures that:
    1. The network optimizes for FVA, not raw MSE
    2. Gradient descent explicitly learns "what beats naive"
    3. Model has a guaranteed baseline of competence

    Reference: Section 8 - The Naive Benchmark Principle

    Args:
        forecast_length: Integer, number of time steps to forecast.
        name: String, layer name.
        **kwargs: Additional keyword arguments passed to the base Layer.

    Input shape:
        (batch_size, backcast_length, features)

    Output shape:
        (batch_size, forecast_length, features)

    Example:
        >>> # Create layer for 12-step forecast
        >>> naive_residual = NaiveResidual(forecast_length=12)
        >>>
        >>> # Use in a model
        >>> inputs = keras.Input(shape=(24, 1))
        >>> x = layers.Dense(64, activation='relu')(layers.Flatten()(inputs))
        >>> network_output = layers.Dense(12)(x)
        >>> network_output = ops.reshape(network_output, (-1, 12, 1))
        >>> final_forecast = naive_residual(inputs, network_output)
    """

    def __init__(
            self,
            forecast_length: int,
            name: Optional[str] = None,
            **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.forecast_length = forecast_length

    def call(
            self,
            inputs: keras.KerasTensor,
            network_output: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Combine network prediction with naive baseline.

        Args:
            inputs: Historical data with shape (batch, backcast_len, features).
            network_output: Learned residual with shape (batch, forecast_len, features).

        Returns:
            Combined forecast: network_output + naive_forecast with shape
            (batch, forecast_len, features).
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
        """Serialization configuration."""
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
    It computes a forecastability score α ∈ [0, 1] where:
    - α ≈ 0: Input is noisy/unforecastable → favor naive baseline
    - α ≈ 1: Input has clear patterns → trust deep network

    This prevents "overfitting to noise" common in complex models like Transformers
    when applied to high-entropy data.

    Mathematical formulation:
        output = α * deep_forecast + (1 - α) * naive_forecast
        α = ComplexityAnalyzer(inputs)

    Reference: Section 2 - Forecastability Assessment

    Args:
        hidden_units: Integer, number of hidden units in complexity analyzer.
            Default: 16.
        activation: String or callable, activation function for hidden layer.
            Default: 'relu'.
        kernel_initializer: Initializer for kernel weights. Default: 'glorot_uniform'.
        kernel_regularizer: Regularizer for kernel weights. Default: None.
        name: String, layer name.
        **kwargs: Additional keyword arguments passed to the base Layer.

    Input shapes:
        inputs: (batch_size, backcast_length, features)
        deep_forecast: (batch_size, forecast_length, features)
        naive_forecast: (batch_size, forecast_length, features)

    Output shape:
        (batch_size, forecast_length, features)

    Example:
        >>> gate = ForecastabilityGate(hidden_units=32, activation='gelu')
        >>>
        >>> # In a model
        >>> inputs = keras.Input(shape=(24, 1))
        >>> deep_forecast = ... # from deep network
        >>> naive_forecast = ... # from NaiveResidual
        >>> final_forecast = gate(inputs, deep_forecast, naive_forecast)
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

        Args:
            input_shape: Shape of inputs tensor, or list of shapes if multiple inputs.
        """
        # Handle multiple input shapes (inputs, deep_forecast, naive_forecast)
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        # Tiny sub-network to estimate forecastability
        # Input: flattened time series
        # Output: scalar α ∈ [0, 1]
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

        Args:
            inputs: Raw historical data with shape (batch, backcast_len, features).
            deep_forecast: Output from deep network with shape (batch, forecast_len, features).
            naive_forecast: Output from naive baseline with shape (batch, forecast_len, features).
            training: Boolean, whether in training mode.

        Returns:
            Weighted combination with shape (batch, forecast_len, features).
        """
        # Flatten input for complexity analysis
        # Shape: (batch, backcast_len * features)
        flat_input = ops.reshape(inputs, (ops.shape(inputs)[0], -1))

        # Compute forecastability score α
        # α → 1.0 if signal is strong (trust deep model)
        # α → 0.0 if noisy (trust naive baseline)
        # Shape: (batch, 1)
        alpha = self.complexity_analyzer(flat_input, training=training)

        # Broadcast alpha to match forecast shape
        # Shape: (batch, 1, 1)
        alpha = ops.expand_dims(alpha, axis=-1)

        # Weighted average
        # If α = 0 (pure noise) → return naive forecast
        # If α = 1 (clear pattern) → return deep forecast
        return (alpha * deep_forecast) + ((1.0 - alpha) * naive_forecast)

    def get_config(self) -> Dict[str, Any]:
        """Serialization configuration."""
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

    This layer outputs three quantile predictions per feature:
    - Lower quantile (typically α/2, e.g., 0.05 for 90% interval)
    - Median (0.50)
    - Upper quantile (typically 1-α/2, e.g., 0.95 for 90% interval)

    The layer includes:
    1. Built-in structure for quantile outputs
    2. Non-trainable calibration score Q for conformal prediction
    3. Methods for calibration and prediction with valid intervals

    During training, the model learns quantile predictions.
    After training, use calibrate() with a calibration set to compute Q.
    During inference, use predict_intervals() to get calibrated predictions.

    Mathematical formulation:
        Training: Learn q_low, q_median, q_high
        Calibration: Q = Quantile(calibration_scores, 1-α)
        Inference:
            final_lower = q_low - Q
            final_upper = q_high + Q

    Reference: Section 5 - Conformalized Quantile Regression (CQR)
              Section 6 - Validity-First Hierarchy

    Args:
        forecast_length: Integer, number of time steps to forecast.
        output_dim: Integer, number of features to predict.
        kernel_initializer: Initializer for projection layer weights.
            Default: 'glorot_uniform'.
        kernel_regularizer: Regularizer for projection layer weights.
            Default: None.
        name: String, layer name.
        **kwargs: Additional keyword arguments passed to the base Layer.

    Input shape:
        (batch_size, input_dim) - typically output of an encoder/flattened features

    Output shape:
        (batch_size, forecast_length, output_dim, 3)
        where the last dimension contains [lower_quantile, median, upper_quantile]

    Example:
        >>> # Create head for 12-step, 1-feature forecast
        >>> head = ConformalQuantileHead(forecast_length=12, output_dim=1)
        >>>
        >>> # Training
        >>> inputs = keras.Input(shape=(128,))  # encoded features
        >>> quantiles = head(inputs)  # shape: (batch, 12, 1, 3)
        >>>
        >>> # After training, calibrate on validation set
        >>> # calibration_scores = compute_nonconformity_scores(val_data)
        >>> # head.calibrate(calibration_scores, alpha=0.1)
        >>>
        >>> # Inference with calibrated intervals
        >>> # median, lower, upper = head.predict_intervals(test_inputs)
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

        Args:
            input_shape: Shape of input tensor (batch_size, input_dim).
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

        Args:
            inputs: Encoded features with shape (batch, input_dim).
            training: Boolean, whether in training mode.

        Returns:
            Quantile predictions with shape (batch, forecast_len, output_dim, 3).
            Last dimension: [lower_quantile, median, upper_quantile].
        """
        # Project to all quantile outputs
        # Shape: (batch, forecast_len * output_dim * 3)
        x = self.projection(inputs, training=training)

        # Reshape to separate time, features, and quantiles
        # Shape: (batch, forecast_len, output_dim, 3)
        # 3 channels: 0=lower(α/2), 1=median(0.5), 2=upper(1-α/2)
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

        This method should be called after training using a held-out calibration set.
        The calibration scores are the nonconformity scores computed as:
            score = max(q_low - y_true, y_true - q_high)

        Args:
            calibration_scores: Array of nonconformity scores from calibration set.
                Shape: (n_calibration_samples,).
            alpha: Significance level. Default: 0.1 for 90% coverage.
                The quantile used is (1-α)(1 + 1/n).

        Example:
            >>> # After training, compute scores on calibration set
            >>> scores = []
            >>> for x_cal, y_cal in calibration_data:
            ...     quantiles = model.predict(x_cal)  # shape: (batch, T, D, 3)
            ...     q_low = quantiles[..., 0]
            ...     q_high = quantiles[..., 2]
            ...     score = np.maximum(q_low - y_cal, y_cal - q_high)
            ...     scores.append(score)
            >>> scores = np.concatenate(scores)
            >>> head.calibrate(scores, alpha=0.1)
        """
        n = len(calibration_scores)
        # Compute adjusted quantile for finite-sample coverage
        # Formula from Section 7: (1-α)(1 + 1/n)
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

        This applies the conformal adjustment to the learned quantiles:
            calibrated_lower = model_lower - Q
            calibrated_upper = model_upper + Q

        Args:
            inputs: Encoded features with shape (batch, input_dim).

        Returns:
            Tuple of (median, lower, upper) predictions, each with shape
            (batch, forecast_len, output_dim).

        Example:
            >>> # After calibration
            >>> median, lower, upper = head.predict_intervals(test_inputs)
            >>> # Now lower and upper provide valid 90% prediction intervals
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
        """Serialization configuration."""
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

    This factory function creates a complete forecasting model with:
    1. Deep network for learning patterns (Efficiency)
    2. Naive baseline integration (Benchmarks)
    3. Forecastability assessment gate (Noise rejection)
    4. Uncertainty quantification via quantiles (Validity)

    The model has two outputs:
    - Point forecast: Gated combination of deep + naive predictions
    - Quantile forecast: For conformalized prediction intervals

    Args:
        input_shape: Tuple of (backcast_length, features) for input data.
        forecast_length: Integer, number of time steps to forecast.
        hidden_units: Integer, number of hidden units in deep network.
            Default: 128.
        gate_hidden_units: Integer, number of hidden units in forecastability gate.
            Default: 16.
        gate_activation: String, activation function for gate.
            Default: 'relu'.

    Returns:
        keras.Model with inputs of shape (batch, backcast_len, features)
        and outputs [point_forecast, quantiles] where:
        - point_forecast: (batch, forecast_len, features)
        - quantiles: (batch, forecast_len, features, 3)

    Example:
        >>> # Create model for 24-step history, 12-step forecast
        >>> model = create_manokhin_compliant_model(
        ...     input_shape=(24, 1),
        ...     forecast_length=12,
        ...     hidden_units=128
        ... )
        >>>
        >>> # Compile with appropriate losses
        >>> model.compile(
        ...     optimizer='adam',
        ...     loss=['mse', quantile_loss],  # Custom quantile loss needed
        ...     loss_weights=[1.0, 0.5]
        ... )
        >>>
        >>> # Train
        >>> model.fit(x_train, [y_train, y_train_quantiles], ...)
        >>>
        >>> # Calibrate quantile head
        >>> # ... calibration logic here ...
        >>>
        >>> # Inference
        >>> point_pred, quantiles = model.predict(x_test)
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
