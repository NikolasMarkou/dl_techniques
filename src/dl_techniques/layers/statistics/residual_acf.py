"""Residual Autocorrelation Function (ACF) analysis and regularization layer.

This module implements a Keras layer for monitoring and regularizing the autocorrelation
of residuals in time series forecasting models, helping ensure residuals approximate
white noise for better forecasting performance.
"""

import keras
import numpy as np
from keras import ops
from typing import Optional, Union, Tuple, Any, Dict, List

from dl_techniques.utils.logger import logger


@keras.saving.register_keras_serializable(package="dl_techniques")
class ResidualACFLayer(keras.layers.Layer):
    """Residual Autocorrelation Function (ACF) analysis and regularization layer.

    This layer monitors and optionally regularizes the autocorrelation of residuals
    in time series forecasting models. It helps ensure that model residuals are 
    closer to white noise, which is a key assumption for good forecasting models.

    The layer computes residuals (predictions - targets), calculates their ACF,
    and can add a regularization term to minimize autocorrelation at specified lags.

    Args:
        max_lag: int, maximum lag to compute ACF for. Default is 40.
        regularization_weight: float or None, weight for ACF regularization loss.
            If None, no regularization is applied (monitoring only). Default is None.
        target_lags: list of int or None, specific lags to target for regularization.
            If None, all lags from 1 to max_lag are targeted. Default is None.
        acf_threshold: float, threshold above which ACF values are considered
            significant and penalized more heavily. Default is 0.1.
        use_absolute_acf: bool, whether to use absolute ACF values for regularization.
            This penalizes both positive and negative autocorrelations. Default is True.
        epsilon: float, small constant for numerical stability. Default is 1e-7.
        name: str, name of the layer. Default is None.
        **kwargs: Additional keyword arguments for the base Layer class.

    Input shape:
        Tuple of two tensors:
        - predictions: (..., sequence_length, features) 
        - targets: (..., sequence_length, features)

    Output shape:
        Same as predictions input: (..., sequence_length, features)

    Raises:
        ValueError: If max_lag < 1, regularization_weight < 0, acf_threshold < 0,
            or target_lags contains invalid values.

    Example:
        >>> # Create a simple forecasting model with ACF monitoring
        >>> inputs = keras.Input(shape=(100, 1))
        >>> lstm = keras.layers.LSTM(32, return_sequences=True)(inputs)
        >>> predictions = keras.layers.Dense(1)(lstm)
        >>> 
        >>> # Add ACF layer for monitoring only
        >>> targets = keras.Input(shape=(100, 1))
        >>> monitored_predictions = ResidualACFLayer(max_lag=20)([predictions, targets])
        >>> 
        >>> # Or with regularization to enforce white noise residuals
        >>> regularized_predictions = ResidualACFLayer(
        ...     max_lag=20, 
        ...     regularization_weight=0.1,
        ...     target_lags=[1, 2, 3, 7, 14]  # Focus on daily and weekly patterns
        ... )([predictions, targets])

    Note:
        - The layer passes predictions through unchanged, making it easy to insert
          into existing models for diagnostic purposes.
        - ACF values are stored in self.acf_values for monitoring during training.
        - When used with regularization, it adds to the model's losses.
    """

    def __init__(
            self,
            max_lag: int = 40,
            regularization_weight: Optional[float] = None,
            target_lags: Optional[List[int]] = None,
            acf_threshold: float = 0.1,
            use_absolute_acf: bool = True,
            epsilon: float = 1e-7,
            name: Optional[str] = None,
            **kwargs: Any
    ) -> None:
        """Initialize the ResidualACFLayer.

        Args:
            max_lag: Maximum lag for ACF computation.
            regularization_weight: Weight for regularization loss.
            target_lags: Specific lags to target for regularization.
            acf_threshold: Threshold for significant ACF values.
            use_absolute_acf: Whether to use absolute ACF values.
            epsilon: Small constant for numerical stability.
            name: Layer name.
            **kwargs: Additional layer arguments.
        """
        super().__init__(name=name, **kwargs)

        # Validate inputs
        if max_lag < 1:
            raise ValueError(f"max_lag must be >= 1, got {max_lag}")
        if regularization_weight is not None and regularization_weight < 0:
            raise ValueError(f"regularization_weight must be >= 0, got {regularization_weight}")
        if acf_threshold < 0:
            raise ValueError(f"acf_threshold must be >= 0, got {acf_threshold}")

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

        # For storing ACF values during forward pass
        self.acf_values = None
        self._build_input_shape = None

        logger.debug(f"Initialized ResidualACFLayer with max_lag={max_lag}, "
                     f"regularization_weight={regularization_weight}")

    def build(self, input_shape: Union[List[Tuple], Tuple[Tuple]]) -> None:
        """Build the layer.

        Args:
            input_shape: Tuple of shapes for (predictions, targets).

        Raises:
            ValueError: If input_shape is not a list/tuple of 2 shapes or shapes don't match.
        """
        if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 2:
            raise ValueError("ResidualACFLayer expects a list of 2 inputs: [predictions, targets]")

        pred_shape, target_shape = input_shape

        # Validate shapes match
        if pred_shape != target_shape:
            raise ValueError(f"Predictions shape {pred_shape} must match targets shape {target_shape}")

        self._build_input_shape = input_shape
        super().build(input_shape)

        logger.debug(f"Built ResidualACFLayer with input shapes: {input_shape}")

    def compute_acf(self, residuals: keras.KerasTensor) -> keras.KerasTensor:
        """Compute autocorrelation function of residuals using Keras operations.

        Args:
            residuals: Tensor of shape (..., sequence_length, features).

        Returns:
            ACF values of shape (..., max_lag + 1, features).
        """
        # Get dimensions - using Keras ops for shape
        shape = ops.shape(residuals)
        seq_length = shape[-2]

        # Center the residuals
        mean = ops.mean(residuals, axis=-2, keepdims=True)
        centered = residuals - mean

        # Compute variance (lag 0 autocorrelation) with numerical stability
        variance = ops.mean(ops.square(centered), axis=-2, keepdims=True) + self.epsilon

        # Get batch and feature dimensions for creating ACF array
        batch_shape = ops.shape(residuals)[:-2]
        n_features = ops.shape(residuals)[-1]

        # Initialize list to collect ACF values
        acf_list = []

        # ACF at lag 0 is always 1
        ones_shape = ops.concatenate([batch_shape, ops.array([1]), ops.array([n_features])], axis=0)
        acf_lag0 = ops.ones(ones_shape, dtype=residuals.dtype)
        acf_list.append(acf_lag0)

        # Compute ACF for each lag
        for lag in range(1, self.max_lag + 1):
            # Check if we have enough sequence length for this lag
            if lag < seq_length:
                # Calculate overlap length
                overlap_length = seq_length - lag

                # Extract overlapping segments using slicing
                segment1 = centered[..., :overlap_length, :]
                segment2 = centered[..., lag:, :]

                # Compute cross-correlation
                cross_product = segment1 * segment2
                autocovariance = ops.mean(cross_product, axis=-2, keepdims=True)

                # Normalize by variance to get correlation
                acf_lag = autocovariance / variance
            else:
                # No overlap possible, ACF is 0
                zeros_shape = ops.concatenate([batch_shape, ops.array([1]), ops.array([n_features])], axis=0)
                acf_lag = ops.zeros(zeros_shape, dtype=residuals.dtype)

            acf_list.append(acf_lag)

        # Stack all ACF values along the lag dimension
        acf = ops.concatenate(acf_list, axis=-2)

        return acf

    def call(
            self,
            inputs: Union[List[keras.KerasTensor], Tuple[keras.KerasTensor]],
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass of the layer.

        Args:
            inputs: List of [predictions, targets] tensors.
            training: Boolean indicating training mode.

        Returns:
            The predictions tensor unchanged.

        Raises:
            ValueError: If inputs is not a list/tuple of 2 tensors.
        """
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError("ResidualACFLayer expects a list of 2 inputs: [predictions, targets]")

        predictions, targets = inputs

        # Compute residuals
        residuals = predictions - targets

        # Compute ACF
        acf = self.compute_acf(residuals)

        # Store ACF values for monitoring (convert to concrete tensor if needed)
        self.acf_values = acf

        # Add regularization loss if specified and in training mode
        if self.regularization_weight is not None and training:
            # Extract ACF values at target lags (skip lag 0)
            target_acf_list = []
            for lag in self.target_lags:
                # Use slicing to extract specific lag
                lag_acf = acf[..., lag:lag + 1, :]
                target_acf_list.append(lag_acf)

            # Concatenate target lags
            target_acf = ops.concatenate(target_acf_list, axis=-2)

            # Apply absolute value if specified
            if self.use_absolute_acf:
                target_acf = ops.abs(target_acf)

            # Compute regularization loss components
            # L2 penalty on ACF values
            l2_loss = ops.mean(ops.square(target_acf))

            # Additional penalty for values above threshold
            excess = ops.maximum(ops.abs(target_acf) - self.acf_threshold, 0.0)
            threshold_penalty = ops.mean(ops.square(excess))

            # Total regularization loss
            reg_loss = self.regularization_weight * (l2_loss + threshold_penalty)

            # Add to layer losses
            self.add_loss(reg_loss)

            logger.debug(f"ResidualACFLayer: Added ACF regularization loss")

        # Return predictions unchanged (pass-through)
        return predictions

    def compute_output_shape(self, input_shape: Union[List[Tuple], Tuple[Tuple]]) -> Tuple:
        """Compute the output shape of the layer.

        Args:
            input_shape: Tuple of shapes for (predictions, targets).

        Returns:
            Output shape (same as predictions shape).
        """
        # Return the shape of predictions (first input)
        return input_shape[0]

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
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

    def get_build_config(self) -> Dict[str, Any]:
        """Get build configuration for layer reconstruction.

        Returns:
            Dictionary containing build configuration.
        """
        return {"input_shape": self._build_input_shape}

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build layer from configuration.

        Args:
            config: Dictionary containing build configuration.
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

    def get_acf_summary(self) -> Optional[Dict[str, float]]:
        """Get summary statistics of the most recent ACF computation.

        Returns:
            Dictionary with ACF statistics or None if no ACF computed yet.
            Contains keys: mean_abs_acf, max_abs_acf, significant_lags,
            and acf_lag_N for each of the first 5 target lags.
        """
        if self.acf_values is None:
            return None

        # Convert to numpy for statistics computation
        acf_np = ops.convert_to_numpy(self.acf_values)

        # Compute statistics (excluding lag 0 which is always 1)
        acf_lags = acf_np[..., 1:, :]  # Skip lag 0

        summary = {
            "mean_abs_acf": float(np.mean(np.abs(acf_lags))),
            "max_abs_acf": float(np.max(np.abs(acf_lags))),
            "significant_lags": int(np.sum(np.abs(acf_lags) > self.acf_threshold)),
        }

        # Add specific lag values if they're in target_lags (show first 5)
        for i, lag in enumerate(self.target_lags[:5]):
            if lag <= self.max_lag:
                # Average across batch and features
                lag_value = float(np.mean(acf_np[..., lag, :]))
                summary[f"acf_lag_{lag}"] = lag_value

        logger.debug(f"ACF Summary: {summary}")
        return summary


# Optional: ACF visualization callback for monitoring during training
class ACFMonitorCallback(keras.callbacks.Callback):
    """Callback to monitor and log ACF statistics during training.

    This callback works with ResidualACFLayer to track autocorrelation
    patterns throughout training.

    Args:
        layer_name: str, name of the ResidualACFLayer to monitor.
        log_frequency: int, frequency (in batches) to log ACF statistics.
            Default is 100.
    """

    def __init__(
            self,
            layer_name: str,
            log_frequency: int = 100
    ) -> None:
        """Initialize the ACF monitor callback.

        Args:
            layer_name: Name of the ResidualACFLayer to monitor.
            log_frequency: How often to log statistics.
        """
        super().__init__()
        self.layer_name = layer_name
        self.log_frequency = log_frequency
        self.batch_count = 0

    def on_train_batch_end(self, batch: int, logs: Optional[Dict] = None) -> None:
        """Log ACF statistics at specified intervals.

        Args:
            batch: Current batch number.
            logs: Dictionary of metrics.
        """
        self.batch_count += 1

        if self.batch_count % self.log_frequency == 0:
            try:
                # Get the ACF layer
                acf_layer = self.model.get_layer(self.layer_name)

                if isinstance(acf_layer, ResidualACFLayer):
                    summary = acf_layer.get_acf_summary()

                    if summary is not None:
                        logger.info(f"Batch {self.batch_count} - ACF Statistics:")
                        logger.info(f"  Mean |ACF|: {summary['mean_abs_acf']:.4f}")
                        logger.info(f"  Max |ACF|: {summary['max_abs_acf']:.4f}")
                        logger.info(f"  Significant lags: {summary['significant_lags']}")

                        # Log specific lag values
                        for key, value in summary.items():
                            if key.startswith("acf_lag_"):
                                logger.info(f"  {key}: {value:.4f}")

            except Exception as e:
                logger.warning(f"Could not monitor ACF layer: {e}")