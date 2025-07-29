"""
Reversible Instance Normalization (RevIN) Layer for Time-Series Forecasting.

This module implements RevIN as described in:
"Reversible Instance Normalization for Accurate Time-Series Forecasting
against Distribution Shift" (Kim et al., ICLR 2022).
"""

import keras
from keras import ops
from typing import Union

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class RevIN(keras.layers.Layer):
    """Reversible Instance Normalization for Time-Series Forecasting.

    RevIN is a normalization method that addresses distribution shift problems
    in time-series forecasting by removing and restoring statistical information
    of time-series instances. It computes statistics per instance and provides
    both normalization and denormalization capabilities.

    The layer applies the following transformations:
    1. Normalization: (x - mean) / std_dev, optionally followed by affine transform
    2. Denormalization: Reverses the normalization to restore original scale

    Args:
        num_features: Number of features or channels in the input time series.
        eps: Small value added for numerical stability. Defaults to 1e-5.
        affine: If True, RevIN has learnable affine parameters (weight and bias).
            Defaults to True.
        affine_weight_initializer: Initializer for affine weight parameter.
            Defaults to "ones".
        affine_bias_initializer: Initializer for affine bias parameter.
            Defaults to "zeros".
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        3D tensor with shape: `(batch_size, sequence_length, num_features)`

    Output shape:
        3D tensor with shape: `(batch_size, sequence_length, num_features)`

    Example:
        ```python
        import keras
        import numpy as np
        from dl_techniques.layers.time_series.revin import RevIN

        # Create RevIN layer for 10 features
        revin = RevIN(num_features=10)

        # Sample input data: (batch=32, seq_len=100, features=10)
        x_input = np.random.randn(32, 100, 10)

        # Normalize input time series (stores statistics internally)
        x_norm = revin(x_input)

        # Pass through forecasting model
        model = keras.Sequential([
            keras.layers.LSTM(64, return_sequences=True),
            keras.layers.Dense(10)
        ])
        predictions = model(x_norm)

        # Denormalize predictions to restore original scale
        predictions_denorm = revin.denormalize(predictions)
        ```

    Note:
        The layer stores normalization statistics (mean and std) computed from
        the input during the forward pass. These statistics are then used for
        denormalization. This makes RevIN particularly suitable for time series
        forecasting where you want to preserve the scale information from input
        to output.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True,
        affine_weight_initializer: Union[str, keras.initializers.Initializer] = "ones",
        affine_bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.affine_weight_initializer = keras.initializers.get(affine_weight_initializer)
        self.affine_bias_initializer = keras.initializers.get(affine_bias_initializer)

        # Will be initialized in build()
        self.affine_weight = None
        self.affine_bias = None

        # Statistics storage (will be set during forward pass)
        self._mean = None
        self._stdev = None
        self._build_input_shape = None

    def build(self, input_shape) -> None:
        """Build the layer weights.

        Args:
            input_shape: Shape tuple of the input tensor.
        """
        self._build_input_shape = input_shape

        if self.affine:
            self.affine_weight = self.add_weight(
                name="affine_weight",
                shape=(self.num_features,),
                initializer=self.affine_weight_initializer,
                trainable=True,
            )

            self.affine_bias = self.add_weight(
                name="affine_bias",
                shape=(self.num_features,),
                initializer=self.affine_bias_initializer,
                trainable=True,
            )

        super().build(input_shape)

    def _get_statistics(self, x):
        """Compute mean and standard deviation statistics for the input.

        This method computes instance-wise statistics by reducing over the
        sequence length dimension while preserving batch and feature dimensions.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, num_features).
        """
        # Reduce over all dimensions except batch (0) and features (last)
        # For input shape (batch, seq_len, features), reduce over seq_len (axis=1)
        axes_to_reduce = tuple(range(1, len(x.shape) - 1))

        self._mean = ops.mean(x, axis=axes_to_reduce, keepdims=True)
        variance = ops.var(x, axis=axes_to_reduce, keepdims=True)
        self._stdev = ops.sqrt(variance + self.eps)

    def _normalize(self, x):
        """Apply normalization with optional affine transformation.

        Args:
            x: Input tensor to normalize.

        Returns:
            Normalized tensor.
        """
        # Subtract mean and divide by standard deviation
        x_norm = (x - self._mean) / self._stdev

        # Apply learnable affine transformation if enabled
        if self.affine:
            x_norm = x_norm * self.affine_weight + self.affine_bias

        return x_norm

    def _denormalize(self, x):
        """Reverse the normalization process.

        Args:
            x: Input tensor to denormalize.

        Returns:
            Denormalized tensor with original scale and offset restored.

        Raises:
            ValueError: If statistics have not been computed yet.
        """
        if self._mean is None or self._stdev is None:
            raise ValueError(
                "Cannot denormalize: statistics not computed. "
                "Call the layer with input data first to compute statistics."
            )

        # Reverse affine transformation if enabled
        if self.affine:
            # Note: Adding eps^2 as in original implementation for numerical stability
            x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)

        # Reverse normalization: multiply by std and add mean
        x = x * self._stdev + self._mean

        return x

    def call(self, inputs, training=None):
        """Apply RevIN normalization.

        This method computes statistics from the input and applies normalization.
        The computed statistics are stored internally for later denormalization.

        Args:
            inputs: Input tensor of shape (batch_size, seq_len, num_features).
            training: Boolean indicating whether in training mode.

        Returns:
            Normalized tensor with same shape as input.
        """
        # Compute and store statistics for current input
        self._get_statistics(inputs)

        # Apply normalization
        return self._normalize(inputs)

    def denormalize(self, inputs):
        """Apply RevIN denormalization.

        This method reverses the normalization applied by the forward pass,
        using the statistics computed during the last call to the layer.

        Args:
            inputs: Input tensor to denormalize, typically model predictions.

        Returns:
            Denormalized tensor with original scale and mean restored.

        Raises:
            ValueError: If no statistics have been computed yet.
        """
        return self._denormalize(inputs)

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input tensor.

        Returns:
            Output shape (same as input shape).
        """
        return input_shape

    def get_config(self):
        """Return the layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "num_features": self.num_features,
            "eps": self.eps,
            "affine": self.affine,
            "affine_weight_initializer": keras.initializers.serialize(self.affine_weight_initializer),
            "affine_bias_initializer": keras.initializers.serialize(self.affine_bias_initializer),
        })
        return config

    def get_build_config(self):
        """Get build configuration for serialization.

        Returns:
            Dictionary containing the build configuration.
        """
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config):
        """Build from configuration after loading.

        Args:
            config: Dictionary containing the build configuration.
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

# ---------------------------------------------------------------------
