import keras
from keras import ops
from typing import Optional, Any, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class StandardScaler(keras.layers.Layer):
    """Standard scaling layer for time series normalization.

    Applies z-score normalization: (x - mean) / std to the input along the last dimension.
    Handles NaN values by replacing them with nan_replacement before computation.

    This layer computes statistics on-the-fly for each batch. The last computed
    statistics are always available for inverse transformation. Optionally,
    when `store_stats=True`, creates persistent weights to store running statistics.

    Args:
        epsilon: Small value to prevent division by zero. Must be positive.
        nan_replacement: Value to replace NaN values with during computation.
        store_stats: Whether to store scaling statistics in persistent weights.
            When True, creates non-trainable weights to store the last computed
            mean and standard deviation for model persistence.
        axis: Axis along which to compute normalization statistics. Defaults to -1.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        N-D tensor with shape: `(..., features)`

    Output shape:
        Same shape as input.

    Example:
        >>> scaler = StandardScaler()
        >>> x = tf.random.normal([32, 100, 10])  # (batch, time, features)
        >>> normalized = scaler(x)
        >>> print(normalized.shape)
        (32, 100, 10)

        >>> # Inverse transform using last computed statistics
        >>> reconstructed = scaler.inverse_transform(normalized)

        >>> # With persistent statistics storage
        >>> scaler = StandardScaler(store_stats=True)
        >>> normalized = scaler(x)
        >>> # Statistics are now stored in model weights for persistence
    """

    def __init__(
            self,
            epsilon: float = 1e-5,
            nan_replacement: float = 0.0,
            store_stats: bool = False,
            axis: int = -1,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        self.epsilon = epsilon
        self.nan_replacement = nan_replacement
        self.store_stats = store_stats
        self.axis = axis

        # These will be initialized in build() if store_stats=True
        self.stored_mean = None
        self.stored_std = None
        self._built_input_shape = None

        # These will be set during call() for inverse transform
        self._last_mean = None
        self._last_std = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer's state.

        Args:
            input_shape: Shape tuple of the input tensor.
        """
        self._built_input_shape = input_shape

        if self.store_stats:
            # Compute the shape for statistics storage
            stats_shape = list(input_shape)
            if self.axis == -1 or self.axis == len(input_shape) - 1:
                stats_shape[-1] = 1
            else:
                # Handle other axis values
                stats_shape[self.axis] = 1

            # Create non-trainable weights to store statistics
            self.stored_mean = self.add_weight(
                name="stored_mean",
                shape=tuple(stats_shape[1:]),  # Remove batch dimension
                initializer="zeros",
                trainable=False,
            )

            self.stored_std = self.add_weight(
                name="stored_std",
                shape=tuple(stats_shape[1:]),  # Remove batch dimension
                initializer="ones",
                trainable=False,
            )

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply standard scaling to inputs.

        Args:
            inputs: Input tensor to normalize.
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.

        Returns:
            Normalized tensor with same shape as input.
        """
        # Replace NaN values
        x = ops.where(ops.isnan(inputs), self.nan_replacement, inputs)

        # Compute mean and std along the specified axis
        mean = ops.mean(x, axis=self.axis, keepdims=True)

        # Use the same computation as before for better numerical consistency
        variance = ops.mean(ops.square(x - mean), axis=self.axis, keepdims=True)
        std = ops.sqrt(variance + self.epsilon)

        # Handle case where std is zero or very small
        std = ops.maximum(std, self.epsilon)

        # Apply normalization
        normalized = (x - mean) / std

        # Store current statistics as instance variables for inverse transform
        # This ensures we can invert using the exact same statistics
        self._last_mean = mean
        self._last_std = std

        # Store statistics in weights if requested (for persistence)
        if self.store_stats and self.built:
            # Update stored statistics (average across batch dimension)
            batch_mean = ops.mean(mean, axis=0)
            batch_std = ops.mean(std, axis=0)

            self.stored_mean.assign(batch_mean)
            self.stored_std.assign(batch_std)

        return normalized

    def inverse_transform(
            self,
            scaled_inputs: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Inverse transform the scaled data back to original scale.

        Args:
            scaled_inputs: Scaled tensor to transform back.

        Returns:
            Tensor transformed back to original scale.

        Raises:
            RuntimeError: If no statistics are available for inverse transform.
        """
        if (not hasattr(self, '_last_mean') or not hasattr(self, '_last_std') or
                self._last_mean is None or self._last_std is None):
            raise RuntimeError(
                "Layer must be called at least once before inverse_transform. "
                "No scaling statistics available."
            )

        return scaled_inputs * self._last_std + self._last_mean

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input.

        Returns:
            Output shape (same as input shape).
        """
        return input_shape

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "epsilon": self.epsilon,
            "nan_replacement": self.nan_replacement,
            "store_stats": self.store_stats,
            "axis": self.axis,
        })
        return config

    def get_build_config(self) -> dict[str, Any]:
        """Get build configuration for serialization.

        Returns:
            Dictionary containing the build configuration.
        """
        return {
            "input_shape": self._built_input_shape,
        }

    def build_from_config(self, config: dict[str, Any]) -> None:
        """Build the layer from a config created with get_build_config.

        Args:
            config: Dictionary containing the build configuration.
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

    def reset_stats(self) -> None:
        """Reset stored statistics to initial values.

        Only works if store_stats=True and layer is built.
        Also clears the last computed statistics used for inverse transform.
        """
        # Clear instance variables used for inverse transform
        if hasattr(self, '_last_mean'):
            delattr(self, '_last_mean')
        if hasattr(self, '_last_std'):
            delattr(self, '_last_std')

        if not self.store_stats:
            logger.warning("reset_stats called but store_stats=False")
            return

        if not self.built or self.stored_mean is None or self.stored_std is None:
            logger.warning("Cannot reset stats: layer not built or no stored statistics")
            return

        self.stored_mean.assign(ops.zeros_like(self.stored_mean))
        self.stored_std.assign(ops.ones_like(self.stored_std))

    def get_stats(self) -> Optional[Tuple[keras.KerasTensor, keras.KerasTensor]]:
        """Get the currently stored statistics.

        Returns:
            Tuple of (mean, std) tensors if available, None otherwise.
        """
        if (not self.store_stats or not self.built or
                self.stored_mean is None or self.stored_std is None):
            return None

        return self.stored_mean, self.stored_std

# ---------------------------------------------------------------------
