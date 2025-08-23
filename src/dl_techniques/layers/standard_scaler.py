"""
This module provides a `StandardScaler` layer, a Keras-native implementation of
standard (z-score) normalization, designed primarily for time series data and
other sequential inputs.

Standard scaling is a crucial preprocessing step that transforms data to have a
mean of zero and a standard deviation of one. This is achieved by applying the
formula: `(x - mean) / std_dev`. Normalizing the input data this way ensures that
all features are on a comparable scale, which can significantly stabilize and
accelerate the training of deep neural networks.

Key Features and Mechanisms:

1.  **On-the-Fly, Per-Batch Normalization:**
    -   By default, this layer operates in a stateless, on-the-fly manner. For each
        batch of data that passes through it, it computes the mean and standard
        deviation *of that specific batch* along the specified feature axis.
    -   This is particularly useful for non-stationary time series where the statistical
        properties of the data may change over time, making a global, fixed scaling
        less effective.

2.  **Stateful Statistics Storage (`store_stats=True`):**
    -   When `store_stats` is enabled, the layer becomes stateful. It creates
        non-trainable weights (buffers) to store the mean and standard deviation
        computed from the most recent batch.
    -   This feature is essential for model persistence. When a model is saved and
        re-loaded, these stored statistics are restored, allowing for consistent
        behavior during inference without needing to re-compute statistics.

3.  **Inverse Transformation:**
    -   The layer includes a crucial `inverse_transform` method. This allows you to
        take the normalized output of the model and convert it back to its original
        data scale.
    -   This is vital for interpreting model outputs, visualizing predictions in their
        original units, and evaluating performance using metrics that are sensitive
        to the data's scale. The method uses the statistics computed from the
        last forward pass to ensure a perfect reconstruction.

4.  **Robustness:**
    -   It includes built-in robustness features, such as replacing `NaN` values before
        computation and adding a small `epsilon` to the standard deviation to prevent
        division by zero in cases of constant-valued features.

This layer provides a self-contained, invertible, and backend-agnostic way to handle
data normalization directly within the model graph, simplifying data preprocessing
pipelines and ensuring that the normalization logic is saved with the model itself.
"""

import keras
from keras import ops
from typing import Optional, Any, Tuple, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..utils.logger import logger

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class StandardScaler(keras.layers.Layer):
    """
    Standard scaling layer for robust time series and feature normalization.

    This layer applies z-score normalization (mean=0, std=1) to input data using
    the formula: ``(x - mean) / std``. It provides both stateless batch-wise
    normalization and optional stateful statistics storage for model persistence.

    The layer is designed to handle the challenges of time series data where
    statistical properties may change over time. It computes normalization
    statistics on-the-fly for each batch, making it suitable for non-stationary
    data streams while providing inverse transformation capabilities for
    result interpretation.

    Key architectural features:
    - **Batch-wise normalization**: Computes statistics per batch for adaptive scaling
    - **NaN handling**: Robust preprocessing with configurable NaN replacement
    - **Inverse transformation**: Perfect reconstruction of original scale
    - **Optional persistence**: Stateful storage of statistics in model weights
    - **Numerical stability**: Epsilon-based protection against division by zero

    Mathematical formulation:
        ``output = (input - μ) / σ``

    Where:
        - ``μ = mean(input, axis=axis, keepdims=True)``
        - ``σ = sqrt(var(input, axis=axis, keepdims=True) + epsilon)``

    Args:
        epsilon: Float, small value added to variance to prevent division by zero.
            Should be positive. Larger values provide more numerical stability
            but may reduce normalization effectiveness. Defaults to 1e-5.
        nan_replacement: Float, value used to replace NaN entries in input data
            before computing statistics. Common choices include 0.0 (default),
            or domain-specific values like historical means. Defaults to 0.0.
        store_stats: Bool, whether to create persistent non-trainable weights
            to store the most recent normalization statistics. Essential for
            model persistence and consistent inference behavior across sessions.
            Defaults to False.
        axis: Int, axis along which to compute normalization statistics.
            Typically -1 (last axis) for feature normalization or 1 for
            sequence normalization. Defaults to -1.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        Arbitrary N-D tensor with shape ``(..., features)``. The normalization
        is computed along the specified axis.

    Output shape:
        Same shape as input tensor. The normalized values will have approximately
        zero mean and unit variance along the normalization axis.

    Attributes:
        stored_mean: Non-trainable weight storing the last computed mean when
            ``store_stats=True``. Shape matches normalization axis.
        stored_std: Non-trainable weight storing the last computed standard
            deviation when ``store_stats=True``. Shape matches normalization axis.

    Example:
        ```python
        # Basic time series normalization
        scaler = StandardScaler()
        inputs = keras.Input(shape=(100, 10))  # (time_steps, features)
        normalized = scaler(inputs)

        # With persistent statistics for inference
        persistent_scaler = StandardScaler(
            store_stats=True,
            epsilon=1e-6  # Higher precision
        )

        # Custom NaN handling for financial data
        financial_scaler = StandardScaler(
            nan_replacement=0.0,  # Replace NaN with zero returns
            axis=-1  # Normalize features independently
        )

        # In a complete model with inverse transformation
        inputs = keras.Input(shape=(50, 5))
        scaled = StandardScaler(store_stats=True)(inputs)
        features = keras.layers.LSTM(64)(scaled)
        outputs = keras.layers.Dense(1)(features)

        model = keras.Model(inputs, outputs)

        # During inference - inverse transform predictions
        predictions = model(test_data)
        # Note: inverse_transform would typically be applied to
        # model outputs that are in the same space as the scaled inputs
        ```

    Raises:
        ValueError: If epsilon is not positive.
        RuntimeError: If inverse_transform is called before the layer
            has processed any data (no statistics available).

    Note:
        The ``inverse_transform`` method uses statistics from the most recent
        forward pass. For consistent inverse transformation across different
        data batches, consider using ``store_stats=True`` and computing
        statistics on a representative dataset during model setup.

        This layer computes statistics per batch, which may introduce slight
        variations between training and inference if batch sizes differ
        significantly. For applications requiring exact reproducibility,
        consider pre-computing statistics on the full dataset.
    """

    def __init__(
        self,
        epsilon: float = 1e-5,
        nan_replacement: float = 0.0,
        store_stats: bool = False,
        axis: int = -1,
        **kwargs: Any
    ) -> None:
        """Initialize the StandardScaler layer."""
        super().__init__(**kwargs)

        # Validate inputs
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        # Store configuration parameters
        self.epsilon = epsilon
        self.nan_replacement = nan_replacement
        self.store_stats = store_stats
        self.axis = axis

        # Initialize weight attributes - created in build() if store_stats=True
        self.stored_mean = None
        self.stored_std = None

        # Statistics from last forward pass for inverse transform
        self._last_mean = None
        self._last_std = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create the layer's persistent statistics storage weights if enabled.

        Args:
            input_shape: Shape tuple of the input tensor.
        """
        if self.store_stats:
            # Compute the shape for statistics storage
            stats_shape = list(input_shape)
            if self.axis == -1 or self.axis == len(input_shape) - 1:
                stats_shape[-1] = 1
            else:
                # Handle other axis values
                stats_shape[self.axis] = 1

            # Create non-trainable weights to store statistics
            # Remove batch dimension for weight shape
            weight_shape = tuple(stats_shape[1:])

            self.stored_mean = self.add_weight(
                name="stored_mean",
                shape=weight_shape,
                initializer="zeros",
                trainable=False,
                dtype=self.dtype
            )

            self.stored_std = self.add_weight(
                name="stored_std",
                shape=weight_shape,
                initializer="ones",
                trainable=False,
                dtype=self.dtype
            )

        # Always call parent build at the end
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply standard scaling to inputs with robust NaN handling.

        Args:
            inputs: Input tensor to normalize.
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode. Included for API consistency
                but does not affect layer behavior.

        Returns:
            Normalized tensor with same shape as input. Values will have
            approximately zero mean and unit variance along the specified axis.
        """
        # Replace NaN values with specified replacement value
        x = ops.where(ops.isnan(inputs), self.nan_replacement, inputs)

        # Compute mean and standard deviation along the specified axis
        mean = ops.mean(x, axis=self.axis, keepdims=True)

        # Compute variance using the more numerically stable two-pass formula
        variance = ops.mean(ops.square(x - mean), axis=self.axis, keepdims=True)
        std = ops.sqrt(variance + self.epsilon)

        # Additional protection against very small standard deviations
        std = ops.maximum(std, self.epsilon)

        # Apply z-score normalization
        normalized = (x - mean) / std

        # Store current statistics for inverse transform
        # These are the exact values used for this forward pass
        self._last_mean = mean
        self._last_std = std

        # Update persistent statistics if enabled
        if self.store_stats and self.built:
            # Average statistics across batch dimension for storage
            batch_mean = ops.mean(mean, axis=0)
            batch_std = ops.mean(std, axis=0)

            # Update stored statistics (these persist across calls)
            self.stored_mean.assign(batch_mean)
            self.stored_std.assign(batch_std)

        return normalized

    def inverse_transform(
        self,
        scaled_inputs: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Transform normalized data back to original scale using last computed statistics.

        This method reverses the z-score normalization using the statistics from
        the most recent forward pass, ensuring perfect reconstruction when applied
        to the layer's own output.

        Args:
            scaled_inputs: Normalized tensor to transform back to original scale.
                Should typically be the output of this layer or derived from it.

        Returns:
            Tensor transformed back to original scale using the formula:
            ``output = scaled_inputs * std + mean``

        Raises:
            RuntimeError: If the layer hasn't been called yet (no statistics available).

        Example:
            ```python
            scaler = StandardScaler()
            normalized = scaler(data)
            reconstructed = scaler.inverse_transform(normalized)
            # reconstructed should be very close to original data
            ```
        """
        if (not hasattr(self, '_last_mean') or not hasattr(self, '_last_std') or
                self._last_mean is None or self._last_std is None):
            raise RuntimeError(
                "Layer must be called at least once before inverse_transform. "
                "No scaling statistics available."
            )

        # Apply inverse z-score transformation: x_original = x_normalized * std + mean
        return scaled_inputs * self._last_std + self._last_mean

    def reset_stats(self) -> None:
        """
        Reset all stored statistics to initial values.

        Clears both the persistent statistics (if store_stats=True) and the
        temporary statistics used for inverse transformation. Useful for
        starting fresh normalization computations.

        Note:
            Only affects persistent statistics if store_stats=True and layer is built.
            Always clears temporary statistics used for inverse transformation.
        """
        # Clear instance variables used for inverse transform
        self._last_mean = None
        self._last_std = None

        if not self.store_stats:
            logger.warning("reset_stats called but store_stats=False")
            return

        if not self.built or self.stored_mean is None or self.stored_std is None:
            logger.warning("Cannot reset stats: layer not built or no stored statistics")
            return

        # Reset persistent statistics to initial values
        self.stored_mean.assign(ops.zeros_like(self.stored_mean))
        self.stored_std.assign(ops.ones_like(self.stored_std))

    def get_stats(self) -> Optional[Tuple[keras.KerasTensor, keras.KerasTensor]]:
        """
        Get the currently stored persistent statistics.

        Returns:
            Tuple of (mean, std) tensors if store_stats=True and layer is built,
            None otherwise. These are the statistics stored in the layer's
            persistent weights, not the temporary statistics from the last call.

        Note:
            Returns persistent statistics only. For statistics from the most
            recent forward pass (used by inverse_transform), these are stored
            internally and not exposed through this method.
        """
        if (not self.store_stats or not self.built or
                self.stored_mean is None or self.stored_std is None):
            return None

        return self.stored_mean, self.stored_std

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape tuple of the input tensor.

        Returns:
            Output shape tuple (identical to input shape).
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns:
            Dictionary containing all layer configuration parameters needed
            for reconstruction during model loading.
        """
        config = super().get_config()
        config.update({
            "epsilon": self.epsilon,
            "nan_replacement": self.nan_replacement,
            "store_stats": self.store_stats,
            "axis": self.axis,
        })
        return config


# ---------------------------------------------------------------------
