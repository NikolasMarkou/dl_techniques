"""
Residual Autocorrelation Function (ACF) analysis and regularization layer.

This module implements a specialized layer for monitoring and regularizing the
autocorrelation of residuals in time series forecasting models. The layer helps
ensure residuals approximate white noise, which is a fundamental assumption for
optimal forecasting performance and model reliability.

Key Features and Applications:

1. **White Noise Validation**: Monitors residual autocorrelation to validate the
   key assumption that model residuals should be uncorrelated (white noise).

2. **Diagnostic Tool**: Provides detailed ACF statistics for model diagnostics
   without affecting the forward pass, making it easy to insert into existing models.

3. **Regularization Mechanism**: Optional regularization loss that penalizes
   significant autocorrelations, encouraging models to produce better residuals.

4. **Time Series Specific**: Designed specifically for sequential prediction tasks
   where temporal dependencies in residuals indicate model inadequacy.

The layer computes residuals as (predictions - targets), calculates their
autocorrelation function up to a specified maximum lag, and can optionally
add regularization loss to minimize autocorrelations at targeted lags.

Mathematical Foundation:
The autocorrelation function at lag k is computed as:
    ACF(k) = Cov(X_t, X_{t-k}) / Var(X_t)

Where Cov represents covariance and Var represents variance. For white noise,
ACF(k) should be approximately 0 for all k > 0.
"""

import keras
import numpy as np
from keras import ops
from typing import Optional, Union, Tuple, Any, Dict, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class ResidualACFLayer(keras.layers.Layer):
    """
    Residual Autocorrelation Function analysis and regularization layer for time series models.

    This layer monitors and optionally regularizes the autocorrelation structure of
    residuals in time series forecasting models. It helps ensure that model residuals
    are closer to white noise, which is a key assumption for good forecasting performance
    and reliable uncertainty estimates.

    The layer operates as a pass-through diagnostic tool that computes residuals
    (predictions - targets), calculates their autocorrelation function (ACF), and
    can add regularization terms to minimize unwanted temporal dependencies in residuals.

    Key diagnostic capabilities:
    - **ACF computation**: Efficient calculation using Keras operations for GPU acceleration
    - **Lag-specific targeting**: Focus regularization on specific temporal patterns
    - **Threshold-based penalties**: Additional penalties for ACF values exceeding thresholds
    - **Real-time monitoring**: ACF statistics available during training for diagnostics
    - **Pass-through design**: Predictions flow unchanged, enabling easy model insertion

    Regularization strategies:
    - **L2 penalty**: Quadratic penalty on ACF values to encourage white noise residuals
    - **Threshold penalty**: Additional penalty for ACF values exceeding significance thresholds
    - **Selective targeting**: Focus on specific lags relevant to domain patterns (daily, weekly, etc.)
    - **Absolute vs signed**: Option to penalize both positive and negative autocorrelations

    Mathematical formulation:
        For residuals r_t = predictions_t - targets_t:

        - ``ACF(k) = Cov(r_t, r_{t-k}) / Var(r_t)`` (autocorrelation at lag k)
        - ``L2_loss = Σ_k (ACF(k))²`` (basic regularization)
        - ``Threshold_loss = Σ_k max(0, |ACF(k)| - threshold)²`` (threshold penalty)
        - ``Total_loss = λ * (L2_loss + Threshold_loss)`` (combined regularization)

    Args:
        max_lag: int, maximum lag to compute ACF for. Should be chosen based on
            expected temporal patterns in your domain. For daily data, consider
            7 (weekly), 30 (monthly), etc. Must be >= 1. Defaults to 40.
        regularization_weight: Optional[float], weight for ACF regularization loss.
            If None, layer operates in monitoring-only mode. Should be tuned based
            on main loss magnitude. Typical values: 0.01-1.0. Defaults to None.
        target_lags: Optional[List[int]], specific lags to target for regularization.
            If None, targets all lags from 1 to max_lag. Use to focus on domain-specific
            patterns (e.g., [1, 7, 30] for daily data with weekly/monthly patterns).
            All values must be between 1 and max_lag. Defaults to None.
        acf_threshold: float, threshold above which ACF values are considered
            statistically significant and penalized more heavily. Typical values
            based on white noise bounds: ~0.1 for large samples. Must be >= 0.
            Defaults to 0.1.
        use_absolute_acf: bool, whether to use absolute ACF values for regularization.
            When True, penalizes both positive and negative autocorrelations equally.
            When False, preserves sign information. Defaults to True.
        epsilon: float, small constant added for numerical stability in variance
            calculations. Should be much smaller than expected residual variance.
            Defaults to 1e-7.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        List of two tensors:
        - predictions: (..., sequence_length, features) - Model predictions
        - targets: (..., sequence_length, features) - Ground truth targets

        Both tensors must have identical shapes.

    Output shape:
        Same as predictions input: (..., sequence_length, features)
        The layer is pass-through - predictions are returned unchanged.

    Attributes:
        acf_values: keras.KerasTensor or None, most recent ACF computation results.
            Shape: (..., max_lag + 1, features) where index 0 is lag 0 (always 1.0).
            Available after forward pass for monitoring and diagnostics.

    Example:
        ```python
        # Basic ACF monitoring without regularization
        inputs = keras.Input(shape=(100, 1))
        lstm = keras.layers.LSTM(64, return_sequences=True)(inputs)
        predictions = keras.layers.Dense(1)(lstm)
        targets = keras.Input(shape=(100, 1))

        # Add monitoring layer
        monitored_preds = ResidualACFLayer(
            max_lag=20,
            regularization_weight=None  # Monitoring only
        )([predictions, targets])

        model = keras.Model([inputs, targets], monitored_preds)

        # Financial time series with specific pattern targeting
        financial_acf = ResidualACFLayer(
            max_lag=30,
            regularization_weight=0.1,
            target_lags=[1, 5, 22],  # Daily, weekly, monthly patterns
            acf_threshold=0.05,      # Stricter threshold for financial data
            use_absolute_acf=True
        )

        # Weather forecasting with seasonal patterns
        weather_acf = ResidualACFLayer(
            max_lag=365,
            target_lags=[1, 7, 30, 90, 365],  # Multi-scale patterns
            regularization_weight=0.05,
            acf_threshold=0.15
        )

        # Complete training setup with callback monitoring
        from dl_techniques.layers.statistics.residual_acf import ACFMonitorCallback

        # Build model with ACF layer
        acf_layer = ResidualACFLayer(
            max_lag=24,
            regularization_weight=0.2,
            target_lags=[1, 2, 3, 6, 12, 24]
        )
        monitored_predictions = acf_layer([predictions, targets])

        model = keras.Model([inputs, targets], monitored_predictions)
        model.compile(optimizer='adam', loss='mse')

        # Train with ACF monitoring
        acf_callback = ACFMonitorCallback('residual_acf', log_frequency=50)
        model.fit(
            [train_x, train_y], train_y,
            validation_data=([val_x, val_y], val_y),
            epochs=100,
            callbacks=[acf_callback]
        )

        # Access ACF statistics after training
        acf_summary = acf_layer.get_acf_summary()
        print(f"Mean |ACF|: {acf_summary['mean_abs_acf']:.4f}")
        print(f"Significant lags: {acf_summary['significant_lags']}")
        ```

    Raises:
        ValueError: If max_lag < 1.
        ValueError: If regularization_weight < 0.
        ValueError: If acf_threshold < 0.
        ValueError: If target_lags contains values outside [1, max_lag].
        ValueError: If input shapes don't match or aren't lists of 2 tensors.

    Note:
        This layer is particularly valuable in:
        - **Economic forecasting**: Where autocorrelated residuals violate efficiency assumptions
        - **Supply chain**: Where residual patterns indicate systematic forecasting bias
        - **Energy demand**: Where autocorrelations suggest missing seasonal components
        - **Medical monitoring**: Where residual patterns may indicate model inadequacy

        The regularization weight should be tuned carefully - too high values can
        over-constrain the model, while too low values provide insufficient regularization.
        Start with small values (0.01-0.1) and increase based on validation performance.

        For computational efficiency with very long sequences, consider using
        target_lags to focus on the most important temporal patterns rather than
        all lags up to max_lag.
    """

    def __init__(
        self,
        max_lag: int = 40,
        regularization_weight: Optional[float] = None,
        target_lags: Optional[List[int]] = None,
        acf_threshold: float = 0.1,
        use_absolute_acf: bool = True,
        epsilon: float = 1e-7,
        **kwargs: Any
    ) -> None:
        """Initialize the ResidualACFLayer."""
        super().__init__(**kwargs)

        # Validate input parameters
        if max_lag < 1:
            raise ValueError(f"max_lag must be >= 1, got {max_lag}")
        if regularization_weight is not None and regularization_weight < 0:
            raise ValueError(f"regularization_weight must be >= 0, got {regularization_weight}")
        if acf_threshold < 0:
            raise ValueError(f"acf_threshold must be >= 0, got {acf_threshold}")

        # Store configuration parameters
        self.max_lag = max_lag
        self.regularization_weight = regularization_weight
        self.target_lags = target_lags if target_lags is not None else list(range(1, max_lag + 1))
        self.acf_threshold = acf_threshold
        self.use_absolute_acf = use_absolute_acf
        self.epsilon = epsilon

        # Validate target lags
        for lag in self.target_lags:
            if lag < 1 or lag > max_lag:
                raise ValueError(f"target_lags must be between 1 and {max_lag}, got {lag}")

        # Initialize ACF storage for monitoring
        self.acf_values = None

        logger.debug(f"Initialized ResidualACFLayer with max_lag={max_lag}, "
                     f"regularization_weight={regularization_weight}")

    def build(self, input_shape: Union[List[Tuple], Tuple[Tuple]]) -> None:
        """
        Build the layer and validate input shapes.

        Args:
            input_shape: Tuple of shapes for (predictions, targets).

        Raises:
            ValueError: If input_shape is not a list/tuple of 2 shapes or shapes don't match.
        """
        if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 2:
            raise ValueError("ResidualACFLayer expects a list of 2 inputs: [predictions, targets]")

        pred_shape, target_shape = input_shape

        # Validate that prediction and target shapes match
        if pred_shape != target_shape:
            raise ValueError(f"Predictions shape {pred_shape} must match targets shape {target_shape}")

        logger.debug(f"Built ResidualACFLayer with input shapes: {input_shape}")

        # Always call parent build at the end
        super().build(input_shape)

    def compute_acf(self, residuals: keras.KerasTensor) -> keras.KerasTensor:
        """
        Compute autocorrelation function of residuals using efficient Keras operations.

        This method implements the ACF computation entirely with Keras ops for GPU
        acceleration and backend compatibility. It handles variable sequence lengths
        and multiple features efficiently.

        Args:
            residuals: Input residuals tensor of shape (..., sequence_length, features).

        Returns:
            ACF values tensor of shape (..., max_lag + 1, features) where the
            first dimension (index 0) corresponds to lag 0 (always 1.0).
        """
        # Get tensor dimensions using Keras ops
        shape = ops.shape(residuals)
        seq_length = shape[-2]

        # Center the residuals by removing mean
        mean = ops.mean(residuals, axis=-2, keepdims=True)
        centered = residuals - mean

        # Compute variance (lag 0 autocovariance) with numerical stability
        variance = ops.mean(ops.square(centered), axis=-2, keepdims=True) + self.epsilon

        # Get batch and feature dimensions for creating output tensor
        batch_shape = ops.shape(residuals)[:-2]
        n_features = ops.shape(residuals)[-1]

        # Initialize list to collect ACF values for each lag
        acf_list = []

        # ACF at lag 0 is always 1.0 by definition
        ones_shape = ops.concatenate([batch_shape, ops.array([1]), ops.array([n_features])], axis=0)
        acf_lag0 = ops.ones(ones_shape, dtype=residuals.dtype)
        acf_list.append(acf_lag0)

        # Compute ACF for each lag from 1 to max_lag
        for lag in range(1, self.max_lag + 1):
            # Check if we have sufficient sequence length for this lag
            if lag < seq_length:
                # Calculate the length of overlapping segments
                overlap_length = seq_length - lag

                # Extract overlapping segments using tensor slicing
                segment1 = centered[..., :overlap_length, :]  # Earlier segment
                segment2 = centered[..., lag:, :]             # Later segment

                # Compute element-wise product and average for autocovariance
                cross_product = segment1 * segment2
                autocovariance = ops.mean(cross_product, axis=-2, keepdims=True)

                # Normalize by variance to get correlation coefficient
                acf_lag = autocovariance / variance
            else:
                # Insufficient data for this lag, ACF is 0
                zeros_shape = ops.concatenate([batch_shape, ops.array([1]), ops.array([n_features])], axis=0)
                acf_lag = ops.zeros(zeros_shape, dtype=residuals.dtype)

            acf_list.append(acf_lag)

        # Concatenate all ACF values along the lag dimension
        acf = ops.concatenate(acf_list, axis=-2)

        return acf

    def call(
        self,
        inputs: Union[List[keras.KerasTensor], Tuple[keras.KerasTensor]],
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass: compute ACF statistics and optional regularization loss.

        Args:
            inputs: List of [predictions, targets] tensors with matching shapes.
            training: Boolean indicating training mode. Regularization is only
                applied during training.

        Returns:
            The predictions tensor unchanged (pass-through behavior).

        Raises:
            ValueError: If inputs is not a list/tuple of exactly 2 tensors.
        """
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError("ResidualACFLayer expects a list of 2 inputs: [predictions, targets]")

        predictions, targets = inputs

        # Compute residuals (prediction errors)
        residuals = predictions - targets

        # Compute autocorrelation function
        acf = self.compute_acf(residuals)

        # Store ACF values for monitoring and diagnostics
        self.acf_values = acf

        # Apply regularization loss if specified and in training mode
        if self.regularization_weight is not None and training:
            # Extract ACF values at target lags (excluding lag 0 which is always 1)
            target_acf_list = []
            for lag in self.target_lags:
                # Use tensor slicing to extract specific lag values
                lag_acf = acf[..., lag:lag + 1, :]
                target_acf_list.append(lag_acf)

            # Concatenate ACF values for all target lags
            target_acf = ops.concatenate(target_acf_list, axis=-2)

            # Apply absolute value transformation if specified
            if self.use_absolute_acf:
                target_acf = ops.abs(target_acf)

            # Compute regularization loss components
            # L2 penalty: quadratic penalty on ACF values
            l2_loss = ops.mean(ops.square(target_acf))

            # Threshold penalty: additional penalty for significant ACF values
            excess = ops.maximum(ops.abs(target_acf) - self.acf_threshold, 0.0)
            threshold_penalty = ops.mean(ops.square(excess))

            # Combined regularization loss
            reg_loss = self.regularization_weight * (l2_loss + threshold_penalty)

            # Add regularization loss to the layer's losses
            self.add_loss(reg_loss)

            logger.debug(f"ResidualACFLayer: Added ACF regularization loss: {reg_loss}")

        # Return predictions unchanged (pass-through design)
        return predictions

    def get_acf_summary(self) -> Optional[Dict[str, float]]:
        """
        Get comprehensive summary statistics of the most recent ACF computation.

        This method provides detailed diagnostics about the autocorrelation structure
        of residuals, useful for model evaluation and hyperparameter tuning.

        Returns:
            Dictionary with ACF statistics or None if no ACF has been computed yet.
            Contains keys:
            - 'mean_abs_acf': Mean absolute ACF value across all lags > 0
            - 'max_abs_acf': Maximum absolute ACF value across all lags > 0
            - 'significant_lags': Number of lags with |ACF| > threshold
            - 'acf_lag_N': ACF value at specific lags (first 5 target lags)

        Example:
            ```python
            # After forward pass
            summary = layer.get_acf_summary()
            if summary:
                print(f"Mean |ACF|: {summary['mean_abs_acf']:.4f}")
                print(f"Significant lags: {summary['significant_lags']}")
                print(f"ACF at lag 1: {summary.get('acf_lag_1', 'N/A'):.4f}")
            ```
        """
        if self.acf_values is None:
            return None

        # Convert to NumPy for statistical computations
        acf_np = ops.convert_to_numpy(self.acf_values)

        # Extract ACF values excluding lag 0 (which is always 1.0)
        acf_lags = acf_np[..., 1:, :]  # Shape: (..., max_lag, features)

        # Compute summary statistics
        summary = {
            "mean_abs_acf": float(np.mean(np.abs(acf_lags))),
            "max_abs_acf": float(np.max(np.abs(acf_lags))),
            "significant_lags": int(np.sum(np.abs(acf_lags) > self.acf_threshold)),
        }

        # Add specific lag values for the first few target lags
        for i, lag in enumerate(self.target_lags[:5]):
            if 1 <= lag <= self.max_lag:
                # Average ACF value across batch and features for this lag
                lag_value = float(np.mean(acf_np[..., lag, :]))
                summary[f"acf_lag_{lag}"] = lag_value

        logger.debug(f"ACF Summary: {summary}")
        return summary

    def compute_output_shape(
        self,
        input_shape: Union[List[Tuple], Tuple[Tuple]]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Tuple of shapes for (predictions, targets).

        Returns:
            Output shape tuple (same as predictions shape since layer is pass-through).
        """
        # Return the shape of predictions (first input) - pass-through behavior
        return input_shape[0]

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns:
            Dictionary containing all layer configuration parameters needed
            for reconstruction during model loading.
        """
        config = super().get_config()
        config.update({
            "max_lag": self.max_lag,
            "regularization_weight": self.regularization_weight,
            "target_lags": self.target_lags,
            "acf_threshold": self.acf_threshold,
            "use_absolute_acf": self.use_absolute_acf,
            "epsilon": self.epsilon,
        })
        return config


# ---------------------------------------------------------------------

class ACFMonitorCallback(keras.callbacks.Callback):
    """
    Callback to monitor and log ACF statistics during training.

    This callback works with ResidualACFLayer to provide real-time monitoring
    of residual autocorrelation patterns throughout the training process,
    helping diagnose model performance and convergence issues.

    Args:
        layer_name: str, name of the ResidualACFLayer to monitor. Must match
            the name assigned to the layer in the model.
        log_frequency: int, frequency (in batches) at which to log ACF statistics.
            Higher values reduce logging overhead but provide less granular monitoring.
            Defaults to 100.

    Example:
        ```python
        # Setup model with named ACF layer
        acf_layer = ResidualACFLayer(max_lag=20, name='residual_acf')

        # Create monitoring callback
        monitor = ACFMonitorCallback(
            layer_name='residual_acf',
            log_frequency=50  # Log every 50 batches
        )

        # Train with monitoring
        model.fit(
            [train_x, train_y], train_y,
            callbacks=[monitor],
            epochs=100
        )
        ```
    """

    def __init__(
        self,
        layer_name: str,
        log_frequency: int = 100
    ) -> None:
        """Initialize the ACF monitor callback."""
        super().__init__()
        self.layer_name = layer_name
        self.log_frequency = log_frequency
        self.batch_count = 0

    def on_train_batch_end(self, batch: int, logs: Optional[Dict] = None) -> None:
        """
        Log ACF statistics at specified intervals during training.

        Args:
            batch: Current batch number.
            logs: Dictionary of training metrics (unused but required by API).
        """
        self.batch_count += 1

        if self.batch_count % self.log_frequency == 0:
            try:
                # Retrieve the ACF layer from the model
                acf_layer = self.model.get_layer(self.layer_name)

                if isinstance(acf_layer, ResidualACFLayer):
                    summary = acf_layer.get_acf_summary()

                    if summary is not None:
                        logger.info(f"Batch {self.batch_count} - ACF Statistics:")
                        logger.info(f"  Mean |ACF|: {summary['mean_abs_acf']:.4f}")
                        logger.info(f"  Max |ACF|: {summary['max_abs_acf']:.4f}")
                        logger.info(f"  Significant lags: {summary['significant_lags']}")

                        # Log individual lag values
                        for key, value in summary.items():
                            if key.startswith("acf_lag_"):
                                logger.info(f"  {key}: {value:.4f}")

            except Exception as e:
                logger.warning(f"Could not monitor ACF layer '{self.layer_name}': {e}")


# ---------------------------------------------------------------------
