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

    This layer provides flexible, axis-configurable normalization with optional
    affine transformation, persistent statistics storage, and perfect inverse
    transformation. It unifies the functionality of Reversible Instance
    Normalization (RevIN) and Standard Scaling (z-score normalization) into
    a single, comprehensive component.

    **Intent**: Provide a one-stop normalization solution that handles both
    instance-wise normalization for time series (RevIN-style) and standard
    feature normalization (StandardScaler-style), with full support for
    inverse transformation, persistence, and robustness features.

    **Architecture & Process:**
    ```
    Input(shape=[batch, ..., features])
           ↓
    NaN Handling: Replace NaN with nan_replacement
           ↓
    Compute Statistics: μ = mean(x), σ = std(x) along axis
           ↓
    Normalize: x_norm = (x - μ) / σ
           ↓
    Affine Transform: output = γ * x_norm + β (if affine=True)
           ↓
    [Store statistics if store_stats=True]
           ↓
    Output(shape=[batch, ..., features])

    Later...
           ↓
    Inverse Transform: x = (output - β) / γ * σ + μ
           ↓
    Original Scale Restored
    ```

    **Mathematical Operations:**
    1. **NaN Handling**: x = where(isnan(x), nan_replacement, x)
    2. **Statistics**: μ = mean(x, axis), σ = sqrt(var(x, axis) + ε)
    3. **Normalization**: x̂ = (x - μ) / σ
    4. **Affine** (optional): y = γ ⊙ x̂ + β
    5. **Inverse**: x' = (y - β) / γ * σ + μ

    Where:
    - μ, σ are computed along specified axis with keepdims=True
    - γ (scale) and β (shift) are learnable parameters if affine=True
    - ε is epsilon for numerical stability
    - ⊙ denotes element-wise multiplication

    Args:
        num_features: Integer, number of features/channels in the input. Used
            primarily for affine parameters shape when affine=True. Must be positive.
            For 3D inputs (batch, seq_len, features), this should match the last
            dimension. Defaults to None (inferred from input_shape).
        axis: Integer or tuple of integers, axis/axes along which to compute
            normalization statistics. Use 1 for per-instance time series
            normalization (RevIN-style), -1 for per-feature normalization
            (StandardScaler-style). Can be tuple for multi-axis normalization.
            Defaults to -1.
        eps: Float, small value added to standard deviation for numerical
            stability. Prevents division by zero when variance is very small.
            Must be positive. Defaults to 1e-5.
        affine: Boolean, whether to apply learnable affine transformation after
            normalization. When True, adds learnable scale (γ) and shift (β)
            parameters. Useful for allowing model to learn optimal representation.
            Defaults to False.
        affine_weight_initializer: Initializer for affine weight parameter γ.
            Only used when affine=True. Defaults to "ones".
        affine_bias_initializer: Initializer for affine bias parameter β.
            Only used when affine=True. Defaults to "zeros".
        nan_replacement: Float, value used to replace NaN entries in input data
            before computing statistics. Ensures robustness to missing or invalid
            data. Defaults to 0.0.
        store_stats: Boolean, whether to create persistent non-trainable weights
            to store the most recent normalization statistics. Essential for
            model persistence and consistent inference. When True, statistics
            are saved with the model and restored on loading. Defaults to False.
        **kwargs: Additional keyword arguments for Layer base class (name,
            trainable, dtype, etc.).

    Input shape:
        N-D tensor with arbitrary shape. Common shapes:
        - 2D: `(batch_size, features)` for tabular data
        - 3D: `(batch_size, sequence_length, features)` for time series
        Higher dimensional inputs are supported.

    Output shape:
        Same shape as input. Normalized values will have approximately zero
        mean and unit variance along the specified axis/axes.

    Attributes:
        affine_weight: Scale parameter γ if affine=True, else None. Shape matches
            the dimensions being normalized (excluding normalized axes).
        affine_bias: Bias parameter β if affine=True, else None. Same shape as
            affine_weight.
        stored_mean: Non-trainable weight storing mean if store_stats=True,
            else None. Shape matches statistics reduction.
        stored_std: Non-trainable weight storing standard deviation if
            store_stats=True, else None. Same shape as stored_mean.

    Methods:
        inverse_transform: Transform normalized data back to original scale.
        denormalize: Alias for inverse_transform (for RevIN compatibility).
        reset_stats: Reset stored persistent statistics to initial values.
        get_stats: Retrieve current persistent statistics (mean, std).

    Example:
        ```python
        import keras

        # RevIN-style: per-instance time series normalization with affine
        revin_style = UnifiedScaler(
            num_features=10,
            axis=1,  # Normalize across time dimension
            affine=True,
            store_stats=True
        )

        # StandardScaler-style: per-feature normalization
        scaler_style = UnifiedScaler(
            axis=-1,  # Normalize across feature dimension
            store_stats=True,
            nan_replacement=0.0
        )

        # Complete time series forecasting example
        inputs = keras.Input(shape=(100, 10))  # (seq_len, features)

        # Normalize input
        normalized = UnifiedScaler(num_features=10, axis=1, affine=True)(inputs)

        # Model processes normalized data
        lstm_out = keras.layers.LSTM(64, return_sequences=True)(normalized)
        predictions = keras.layers.Dense(10)(lstm_out)

        model = keras.Model(inputs, predictions)

        # During inference - inverse transform predictions
        scaler = model.layers[1]  # The UnifiedScaler layer
        original_scale_preds = scaler.inverse_transform(model_output)

        # Multi-axis normalization (advanced)
        multi_axis = UnifiedScaler(axis=(1, 2), eps=1e-6)

        # Custom initialization
        custom_scaler = UnifiedScaler(
            num_features=20,
            affine=True,
            affine_weight_initializer='glorot_uniform',
            affine_bias_initializer='normal',
            eps=1e-6
        )
        ```

    Raises:
        ValueError: If num_features is specified and not positive.
        ValueError: If eps is not positive.
        ValueError: If axis configuration is invalid during build.
        RuntimeError: If inverse_transform/denormalize is called before
            statistics are computed.

    Note:
        - The layer stores normalization statistics from the most recent forward
          pass for use in inverse transformation.
        - When store_stats=True, statistics are averaged across the batch dimension
          before storage, providing a representative value for the dataset.
        - For time series forecasting with RevIN behavior, use axis=1 with
          affine=True on 3D inputs (batch, seq_len, features).
        - For standard feature scaling behavior, use axis=-1 with store_stats=True.
        - The inverse_transform and denormalize methods are equivalent; both are
          provided for API compatibility with RevIN and StandardScaler conventions.
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
        """
        Create the layer's weights and validate input shape.

        Args:
            input_shape: Shape tuple of the input tensor.

        Raises:
            ValueError: If input shape is incompatible with configuration.
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
        """
        Apply normalization to inputs.

        This method computes statistics from the input, applies normalization,
        and optionally applies affine transformation. Statistics are stored
        internally for later inverse transformation.

        Args:
            inputs: Input tensor to normalize. Shape must match expected input
                shape from build().
            training: Boolean indicating whether in training mode. Included for
                API consistency but does not affect layer behavior.

        Returns:
            Normalized tensor with same shape as input. Values will have
            approximately zero mean and unit variance along the specified axis.
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
        """
        Transform normalized data back to original scale.

        This method reverses the normalization (and affine transformation if
        enabled) using statistics from the most recent forward pass, ensuring
        perfect reconstruction when applied to the layer's own output.

        Args:
            scaled_inputs: Normalized tensor to transform back to original scale.
                Should typically be the output of this layer or derived from it.

        Returns:
            Tensor transformed back to original scale.

        Raises:
            RuntimeError: If the layer hasn't been called yet (no statistics
                available for inverse transformation).

        Example:
            ```python
            scaler = UnifiedScaler(num_features=10, axis=1, affine=True)

            # Forward pass stores statistics
            normalized = scaler(inputs)

            # Model processes normalized data
            predictions = model(normalized)

            # Transform predictions back to original scale
            original_scale = scaler.inverse_transform(predictions)
            ```
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
        """
        Apply denormalization (alias for inverse_transform).

        This method is provided for compatibility with RevIN API conventions.
        It performs the exact same operation as inverse_transform().

        Args:
            scaled_inputs: Normalized tensor to denormalize.

        Returns:
            Denormalized tensor with original scale restored.

        Raises:
            RuntimeError: If statistics have not been computed yet.

        Example:
            ```python
            # RevIN-style usage
            revin = UnifiedScaler(num_features=10, axis=1, affine=True)
            x_norm = revin(x_input)
            predictions = model(x_norm)
            predictions_denorm = revin.denormalize(predictions)
            ```
        """
        return self.inverse_transform(scaled_inputs)

    def reset_stats(self) -> None:
        """
        Reset all stored statistics to initial values.

        Clears both the persistent statistics (if store_stats=True) and the
        temporary statistics used for inverse transformation. Useful for
        starting fresh normalization computations or when switching datasets.

        Note:
            Only affects persistent statistics if store_stats=True and layer
            is built. Always clears temporary statistics (_last_mean, _last_std).

        Example:
            ```python
            scaler = UnifiedScaler(store_stats=True)
            scaler(data_batch_1)  # Computes and stores stats

            # Switch to new dataset
            scaler.reset_stats()
            scaler(data_batch_2)  # Fresh statistics computed
            ```
        """
        # Clear instance variables used for inverse transform
        self._last_mean = None
        self._last_std = None

        # Reset persistent statistics if they exist
        if self.store_stats and self.built:
            if self.stored_mean is not None and self.stored_std is not None:
                self.stored_mean.assign(ops.zeros_like(self.stored_mean))
                self.stored_std.assign(ops.ones_like(self.stored_std))

    def get_stats(self) -> Optional[Tuple[keras.KerasTensor, keras.KerasTensor]]:
        """
        Get the currently stored persistent statistics.

        Returns:
            Tuple of (mean, std) tensors if store_stats=True and layer is built,
            None otherwise. These are the statistics stored in the layer's
            persistent weights, representing averaged statistics across batches.

        Note:
            Returns persistent statistics only. For statistics from the most
            recent forward pass (used by inverse_transform), these are stored
            internally as _last_mean and _last_std and not exposed through
            this method.

        Example:
            ```python
            scaler = UnifiedScaler(store_stats=True)
            scaler(training_data)  # Compute and store stats

            mean, std = scaler.get_stats()
            print(f"Stored mean: {mean}, std: {std}")
            ```
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
            Output shape tuple (identical to input shape for normalization).
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return the layer configuration for serialization.

        Returns:
            Dictionary containing all layer configuration parameters needed
            for reconstruction during model loading.
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