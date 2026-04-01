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
    Residual Autocorrelation Function analysis and regularization layer.

    Pass-through diagnostic layer that computes residuals ``r_t = pred_t - target_t``,
    calculates their ACF up to ``max_lag``, and optionally adds regularization loss
    ``lambda * (sum_k ACF(k)^2 + sum_k max(0, |ACF(k)| - threshold)^2)`` to
    encourage white-noise residuals. The ACF at lag ``k`` is computed as
    ``Cov(r_t, r_{t-k}) / Var(r_t)``. Predictions pass through unchanged, making
    the layer easy to insert into existing models for monitoring or regularization.

    **Architecture Overview:**

    .. code-block:: text

        ┌────────────────────────────────────┐
        │ [Predictions, Targets]             │
        └──────────┬─────────────────────────┘
                   ▼
        ┌────────────────────────────────────┐
        │ Residuals = Predictions - Targets  │
        └──────────┬─────────────────────────┘
                   ▼
        ┌────────────────────────────────────┐
        │ Compute ACF(0..max_lag)            │
        │ Center, autocovariance, normalize  │
        └──────────┬─────────────────────────┘
                   │
           ┌───────┴───────┐
           ▼               ▼
        ┌──────────┐ ┌──────────────────┐
        │ Store    │ │ Regularization   │
        │ ACF for  │ │ L2 + threshold   │
        │ monitor  │ │ penalty (train)  │
        └──────────┘ └──────────────────┘
                   │
                   ▼
        ┌────────────────────────────────────┐
        │ Output = Predictions (pass-through)│
        └────────────────────────────────────┘

    :param max_lag: Maximum lag to compute ACF for. Must be >= 1. Defaults to 40.
    :type max_lag: int
    :param regularization_weight: Weight for ACF regularization loss. If ``None``,
        monitoring-only mode. Defaults to ``None``.
    :type regularization_weight: float | None
    :param target_lags: Specific lags to target for regularization. If ``None``,
        targets all lags from 1 to ``max_lag``. Defaults to ``None``.
    :type target_lags: list[int] | None
    :param acf_threshold: Threshold above which ACF values receive extra penalty.
        Defaults to 0.1.
    :type acf_threshold: float
    :param use_absolute_acf: Whether to use absolute ACF values for regularization.
        Defaults to ``True``.
    :type use_absolute_acf: bool
    :param epsilon: Small constant for numerical stability. Defaults to 1e-7.
    :type epsilon: float
    :param kwargs: Additional keyword arguments for the Layer base class.
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
        """Build the layer and validate input shapes.

        :param input_shape: Tuple of shapes for ``(predictions, targets)``.
        :type input_shape: list[tuple] | tuple[tuple]
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
        """Compute autocorrelation function of residuals.

        :param residuals: Residuals of shape ``(..., sequence_length, features)``.
        :type residuals: keras.KerasTensor
        :return: ACF values of shape ``(..., max_lag + 1, features)``.
        :rtype: keras.KerasTensor
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
        """Forward pass computing ACF statistics and optional regularization.

        :param inputs: List of ``[predictions, targets]`` tensors.
        :type inputs: list[keras.KerasTensor] | tuple[keras.KerasTensor]
        :param training: Boolean for training mode.
        :type training: bool | None
        :return: Predictions tensor unchanged (pass-through).
        :rtype: keras.KerasTensor
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
        """Get summary statistics of the most recent ACF computation.

        :return: Dictionary with ACF statistics, or ``None`` if not yet computed.
        :rtype: dict[str, float] | None
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
        """Compute the output shape of the layer.

        :param input_shape: Tuple of shapes for ``(predictions, targets)``.
        :type input_shape: list[tuple] | tuple[tuple]
        :return: Output shape (same as predictions).
        :rtype: tuple[int | None, ...]
        """
        # Return the shape of predictions (first input) - pass-through behavior
        return input_shape[0]

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        :return: Configuration dictionary.
        :rtype: dict[str, Any]
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

    Works with ``ResidualACFLayer`` to provide real-time monitoring of residual
    autocorrelation patterns, helping diagnose model convergence issues.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────┐
        │  on_train_batch_end      │
        └────────────┬─────────────┘
                     ▼
        ┌──────────────────────────┐
        │  Retrieve ACF layer      │
        │  by name from model      │
        └────────────┬─────────────┘
                     ▼
        ┌──────────────────────────┐
        │  get_acf_summary()       │
        │  ─► Log statistics       │
        └──────────────────────────┘

    :param layer_name: Name of the ResidualACFLayer to monitor.
    :type layer_name: str
    :param log_frequency: Frequency (in batches) for logging. Defaults to 100.
    :type log_frequency: int
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
        """Log ACF statistics at specified intervals during training.

        :param batch: Current batch number.
        :type batch: int
        :param logs: Training metrics dictionary.
        :type logs: dict | None
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
