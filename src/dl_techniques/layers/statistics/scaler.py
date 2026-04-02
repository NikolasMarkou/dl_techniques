"""
Unified Scaler Layer - A Comprehensive Normalization Solution.

This module provides a unified normalization layer that combines the capabilities
of Reversible Instance Normalization (RevIN) and Standard Scaling (z-score
normalization), offering a flexible and powerful tool for various deep learning
applications, particularly in time series forecasting and feature normalization.

**Core Design Philosophy:**

The UnifiedScaler layer addresses the need for a single, flexible normalization
component that can handle both instance-wise normalization (as in RevIN for time
series forecasting) and standard feature-wise normalization (as in StandardScaler
for general preprocessing). By unifying these approaches, it eliminates the need
to maintain separate normalization layers and provides a consistent API for all
normalization needs.

**Key Capabilities:**

1. **Flexible Axis Normalization:**
   - Support for normalization along any axis (time steps, features, or custom)
   - Per-instance normalization (axis=1) for time series with distribution shift
   - Per-feature normalization (axis=-1) for standard preprocessing
   - Multi-axis normalization for advanced use cases

2. **Optional Affine Transformation:**
   - Learnable scale (γ) and shift (β) parameters
   - Allows the model to learn optimal data representation post-normalization
   - Can be enabled/disabled independently

3. **Robust NaN Handling:**
   - Configurable NaN replacement strategy
   - Ensures numerical stability in real-world data scenarios

4. **Persistent Statistics Storage:**
   - Optional storage of normalization statistics as non-trainable weights
   - Essential for model persistence and consistent inference behavior
   - Enables reproducible transformations across sessions

5. **Perfect Inverse Transformation:**
   - Dual methods: `inverse_transform()` and `denormalize()` (equivalent)
   - Uses stored statistics to reconstruct original data scale
   - Critical for interpreting model outputs and evaluating predictions

6. **Utility Methods:**
   - `reset_stats()`: Clear stored statistics
   - `get_stats()`: Retrieve current normalization parameters
   - Comprehensive state management

**Mathematical Foundation:**

For input tensor `x` with shape `(batch, ..., features)`:

1. **Statistics Computation:**
   - `μ = mean(x, axis=axis, keepdims=True)`
   - `σ = sqrt(var(x, axis=axis, keepdims=True) + epsilon)`

2. **Normalization:**
   - `x_norm = (x - μ) / σ`

3. **Optional Affine Transform:**
   - `output = γ ⊙ x_norm + β` (if affine=True)

4. **Inverse Transformation:**
   - If affine: `x = (output - β) / γ`
   - `x_original = x * σ + μ`

Where:
- `⊙` denotes element-wise multiplication
- `γ`, `β` are learnable parameters (shape matches normalized dimensions)
- `μ`, `σ` are computed per specified axis

**Use Cases:**

- **Time Series Forecasting:** Instance normalization (axis=1) to handle
  distribution shifts across different time series instances
- **Feature Preprocessing:** Standard normalization (axis=-1) for consistent
  feature scaling in multi-variate data
- **Online Learning:** Adaptive normalization with persistent statistics for
  streaming data scenarios
- **Model Interpretability:** Inverse transformation to evaluate predictions
  in original data scale

**References:**
    - Kim et al., "Reversible Instance Normalization for Accurate Time-Series
      Forecasting against Distribution Shift", ICLR 2022.
      https://arxiv.org/abs/2107.03445
"""

import keras
from keras import ops
from typing import Optional, Union, Tuple, Any, Dict


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class UnifiedScaler(keras.layers.Layer):
    """
    Unified normalization layer combining RevIN and StandardScaler capabilities.

    Performs z-score normalization ``x_norm = (x - mu) / sigma`` along configurable
    axes with optional learnable affine transform ``y = gamma * x_norm + beta``.
    Statistics (mean ``mu`` and standard deviation ``sigma = sqrt(var + eps)``) are
    computed per forward pass and stored for exact inverse transformation. NaN values
    are replaced with ``nan_replacement`` before computing statistics. Persistent
    non-trainable weights can store batch-averaged statistics for model serialization.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────┐
        │  Input (batch, ..., features)│
        └──────────────┬───────────────┘
                       ▼
        ┌──────────────────────────────┐
        │  NaN ─► nan_replacement      │
        └──────────────┬───────────────┘
                       ▼
        ┌──────────────────────────────┐
        │  mu = mean(x, axis)          │
        │  sigma = sqrt(var + eps)     │
        └──────────────┬───────────────┘
                       ▼
        ┌──────────────────────────────┐
        │  x_norm = (x - mu) / sigma   │
        └──────────────┬───────────────┘
                       ▼
        ┌──────────────────────────────┐
        │  y = gamma * x_norm + beta   │
        │  (if affine=True)            │
        └──────────────┬───────────────┘
                       ▼
        ┌──────────────────────────────┐
        │  Output (same shape)         │
        └──────────────────────────────┘

        Inverse: x' = (y - beta) / gamma * sigma + mu

    :param num_features: Number of features/channels. Defaults to ``None`` (inferred).
    :type num_features: int | None
    :param axis: Axis/axes for normalization statistics. Defaults to -1.
    :type axis: int | tuple[int, ...]
    :param eps: Small value for numerical stability. Defaults to 1e-5.
    :type eps: float
    :param affine: Whether to apply learnable affine transform. Defaults to ``False``.
    :type affine: bool
    :param affine_weight_initializer: Initializer for scale gamma. Defaults to ``"ones"``.
    :type affine_weight_initializer: str | keras.initializers.Initializer
    :param affine_bias_initializer: Initializer for shift beta. Defaults to ``"zeros"``.
    :type affine_bias_initializer: str | keras.initializers.Initializer
    :param nan_replacement: Value to replace NaN entries. Defaults to 0.0.
    :type nan_replacement: float
    :param store_stats: Whether to store statistics as persistent weights.
        Defaults to ``False``.
    :type store_stats: bool
    :param kwargs: Additional keyword arguments for Layer base class.
    """

    def __init__(
            self,
            num_features: Optional[int] = None,
            axis: Union[int, Tuple[int, ...]] = -1,
            eps: float = 1e-5,
            affine: bool = False,
            affine_weight_initializer: Union[str, keras.initializers.Initializer] = "ones",
            affine_bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
            nan_replacement: float = 0.0,
            store_stats: bool = False,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if num_features is not None and num_features <= 0:
            raise ValueError(f"num_features must be positive, got {num_features}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")

        # Store configuration
        self.num_features = num_features
        self.axis = axis if isinstance(axis, (tuple, list)) else (axis,)
        self.eps = eps
        self.affine = affine
        self.affine_weight_initializer = keras.initializers.get(affine_weight_initializer)
        self.affine_bias_initializer = keras.initializers.get(affine_bias_initializer)
        self.nan_replacement = nan_replacement
        self.store_stats = store_stats

        # Weight attributes (created in build)
        self.affine_weight = None
        self.affine_bias = None
        self.stored_mean = None
        self.stored_std = None

        # Statistics from last forward pass (for inverse transform)
        self._last_mean = None
        self._last_std = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the layer's weights and validate input shape.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple[int | None, ...]
        """
        # Validate input shape
        if len(input_shape) < 2:
            raise ValueError(
                f"UnifiedScaler expects at least 2D input, got shape {input_shape}"
            )

        # Infer num_features if not provided
        if self.num_features is None:
            self._inferred_num_features = input_shape[-1]
            if self._inferred_num_features is None:
                raise ValueError(
                    "Last dimension of input must be defined when num_features=None"
                )
        else:
            self._inferred_num_features = self.num_features

        # Validate axis configuration
        rank = len(input_shape)
        normalized_axes = tuple(
            ax if ax >= 0 else rank + ax for ax in self.axis
        )

        if any(ax >= rank or ax < 0 for ax in normalized_axes):
            raise ValueError(
                f"Invalid axis {self.axis} for input with rank {rank}"
            )

        # Calculate shape for statistics (keepdims after reduction)
        self._stats_shape = list(input_shape)
        for ax in sorted(normalized_axes, reverse=True):
            self._stats_shape[ax] = 1
        self._stats_shape = tuple(self._stats_shape)

        # Calculate shape for affine parameters (reduced dimensions)
        # Affine parameters should match the feature dimensions, not the normalized axes
        self._affine_shape = [
            dim for i, dim in enumerate(input_shape)
            if i not in normalized_axes
        ]
        # Handle case where all dimensions are normalized (unlikely but possible)
        if not self._affine_shape:
            self._affine_shape = [1]
        else:
            # Remove batch dimension
            self._affine_shape = self._affine_shape[1:]
        self._affine_shape = tuple(self._affine_shape)

        # Create affine parameters if enabled
        if self.affine:
            # For most common case (3D input, axis=1), shape is (num_features,)
            # For general case, shape matches non-normalized dimensions
            affine_param_shape = (self._inferred_num_features,) if input_shape[
                                                                       -1] == self._inferred_num_features else self._affine_shape

            self.affine_weight = self.add_weight(
                name="affine_weight",
                shape=affine_param_shape,
                initializer=self.affine_weight_initializer,
                trainable=True,
            )

            self.affine_bias = self.add_weight(
                name="affine_bias",
                shape=affine_param_shape,
                initializer=self.affine_bias_initializer,
                trainable=True,
            )

        # Create persistent statistics storage if enabled
        if self.store_stats:
            # Remove batch dimension for weight shape
            weight_shape = tuple(self._stats_shape[1:])

            self.stored_mean = self.add_weight(
                name="stored_mean",
                shape=weight_shape,
                initializer="zeros",
                trainable=False,
            )

            self.stored_std = self.add_weight(
                name="stored_std",
                shape=weight_shape,
                initializer="ones",
                trainable=False,
            )

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply normalization to inputs.

        :param inputs: Input tensor to normalize.
        :type inputs: keras.KerasTensor
        :param training: Boolean for training mode.
        :type training: bool | None
        :return: Normalized tensor with same shape as input.
        :rtype: keras.KerasTensor
        """
        # Replace NaN values with specified replacement value
        x = ops.where(ops.isnan(inputs), self.nan_replacement, inputs)

        # Compute mean and standard deviation along specified axes
        mean = ops.mean(x, axis=self.axis, keepdims=True)

        # Compute variance using the stable two-pass formula
        variance = ops.mean(ops.square(x - mean), axis=self.axis, keepdims=True)
        std = ops.sqrt(variance + self.eps)

        # Additional protection against very small standard deviations
        std = ops.maximum(std, self.eps)

        # Apply z-score normalization
        x_norm = (x - mean) / std

        # Store current statistics for inverse transform
        self._last_mean = mean
        self._last_std = std

        # Update persistent statistics if enabled
        if self.store_stats and self.built:
            # Average statistics across batch dimension for storage
            batch_mean = ops.mean(mean, axis=0)
            batch_std = ops.mean(std, axis=0)

            self.stored_mean.assign(batch_mean)
            self.stored_std.assign(batch_std)

        # Apply learnable affine transformation if enabled
        if self.affine:
            x_norm = x_norm * self.affine_weight + self.affine_bias

        return x_norm

    def inverse_transform(self, scaled_inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Transform normalized data back to original scale.

        :param scaled_inputs: Normalized tensor to denormalize.
        :type scaled_inputs: keras.KerasTensor
        :return: Tensor in original scale.
        :rtype: keras.KerasTensor
        """
        if self._last_mean is None or self._last_std is None:
            raise RuntimeError(
                "Cannot perform inverse transformation: statistics not computed. "
                "Call the layer with input data first to compute statistics."
            )

        x = scaled_inputs

        # Reverse affine transformation if enabled
        if self.affine:
            x = (x - self.affine_bias) / self.affine_weight

        # Reverse normalization: multiply by std and add mean
        x = x * self._last_std + self._last_mean

        return x

    def denormalize(self, scaled_inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Apply denormalization (alias for inverse_transform).

        :param scaled_inputs: Normalized tensor to denormalize.
        :type scaled_inputs: keras.KerasTensor
        :return: Denormalized tensor.
        :rtype: keras.KerasTensor
        """
        return self.inverse_transform(scaled_inputs)

    def reset_stats(self) -> None:
        """Reset all stored statistics to initial values."""
        # Clear instance variables used for inverse transform
        self._last_mean = None
        self._last_std = None

        # Reset persistent statistics if they exist
        if self.store_stats and self.built:
            if self.stored_mean is not None and self.stored_std is not None:
                self.stored_mean.assign(ops.zeros_like(self.stored_mean))
                self.stored_std.assign(ops.ones_like(self.stored_std))

    def get_stats(self) -> Optional[Tuple[keras.KerasTensor, keras.KerasTensor]]:
        """Get the currently stored persistent statistics.

        :return: Tuple of ``(mean, std)`` tensors, or ``None`` if unavailable.
        :rtype: tuple[keras.KerasTensor, keras.KerasTensor] | None
        """
        if (not self.store_stats or not self.built or
                self.stored_mean is None or self.stored_std is None):
            return None

        return self.stored_mean, self.stored_std

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple[int | None, ...]
        :return: Output shape tuple (identical to input).
        :rtype: tuple[int | None, ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return the layer configuration for serialization.

        :return: Configuration dictionary.
        :rtype: dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "num_features": self.num_features,
            "axis": self.axis[0] if len(self.axis) == 1 else self.axis,
            "eps": self.eps,
            "affine": self.affine,
            "affine_weight_initializer": keras.initializers.serialize(
                self.affine_weight_initializer
            ),
            "affine_bias_initializer": keras.initializers.serialize(
                self.affine_bias_initializer
            ),
            "nan_replacement": self.nan_replacement,
            "store_stats": self.store_stats,
        })
        return config

# ---------------------------------------------------------------------