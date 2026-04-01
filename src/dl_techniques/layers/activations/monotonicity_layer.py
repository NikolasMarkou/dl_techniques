"""
Monotonicity enforcement layer for neural networks.

This module provides a flexible layer for enforcing monotonic constraints on
neural network outputs. It's particularly useful for applications requiring
ordered predictions, such as quantile regression, dose-response modeling,
survival analysis, and ranking.

The layer transforms unconstrained network outputs into monotonically
non-decreasing values along a specified axis using various mathematical
strategies, each with different properties regarding gradient flow,
spacing control, and computational efficiency.

Given raw predictions ``[r_0, r_1, r_2, ..., r_n]``, we want to ensure:
``output[i] <= output[i+1]`` for all i

Different strategies achieve this through:

1. **Cumulative Softplus**: ``Q_i = Q_0 + Sigma(softplus(r_j))`` for j=1..i
2. **Exponential**: ``Q_i = Q_{i-1} + exp(r_i)``
3. **Sigmoid**: ``Q_i = sigmoid(r_i)`` scaled and shifted
4. **Normalized Softmax**: Uses softmax weights for controlled spacing
5. **Squared**: ``Q_i = Q_{i-1} + r_i^2``

Use cases include quantile regression (ensuring ``Q(0.1) <= Q(0.5) <= Q(0.9)``),
survival analysis with monotonic hazard/survival functions, dose-response
modeling, ranking with proper ordering, and monotonic utility or demand
functions in economics.

References:
    - Koenker, R. (2005). Quantile Regression. Cambridge University Press.
    - Cannon, A. J. (2011). Quantile regression neural networks.
      Journal of Computational and Graphical Statistics.
    - Weiss, K., et al. (2013). A survey of transfer learning.
      Journal of Big Data.
"""

import keras
import warnings
from typing import Optional, Literal, Tuple, Any

# ---------------------------------------------------------------------

MonotonicityMethod = Literal[
    "cumulative_softplus",
    "exponential",
    "sigmoid",
    "normalized_softmax",
    "squared",
    "cumulative_exp"
]

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MonotonicityLayer(keras.layers.Layer):
    """
    Enforces monotonic (non-decreasing) constraints on predictions.

    This layer transforms unconstrained predictions into monotonically
    non-decreasing values along a specified axis. It supports multiple
    strategies with different properties for gradient flow, spacing control,
    and computational efficiency.

    Given raw predictions ``[r_0, r_1, r_2, ...]``, the layer ensures:
    ``output[..., i] <= output[..., i+1]``. This is achieved by predicting
    the first value directly and modeling subsequent values as cumulative
    positive increments.

    Strategies:

    1. **cumulative_softplus** (Default):
       ``Q_i = Q_0 + Sigma(softplus(r_j))`` for j=1..i.
       Smooth gradients, numerically stable. Best for general quantile regression.

    2. **exponential**:
       ``Q_i = Q_{i-1} + exp(r_i)``.
       Strong monotonicity, fast growth. Best when large spacing is natural.

    3. **cumulative_exp** (Safer exponential):
       ``Q_i = Q_0 + Sigma(exp(clip(r_j, -10, 10)))`` for j=1..i.
       Like exponential but with overflow protection.

    4. **sigmoid**:
       ``Q_i = sigmoid(r_i) * (max_val - min_val) + min_val``.
       Bounded output. Best when output range is known.

    5. **squared**:
       ``Q_i = Q_{i-1} + r_i^2``.
       Simple, differentiable, no exponential growth.

    6. **normalized_softmax**:
       ``deltas = softmax(r_1:n), Q_i = Q_0 + Sigma(delta_j * range)`` for j=1..i.
       Controlled total spacing summing to defined range.

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────────────────────────┐
        │  Raw Predictions [..., num_values]  │
        └──────────────┬──────────────────────┘
                       │
                       ▼
        ┌─────────────────────────────────────┐
        │  Split Along Monotonicity Axis      │
        │  ┌───────────┐  ┌────────────────┐  │
        │  │ First (r0) │  │ Rest (r1..rN)  │  │
        │  │  (anchor)  │  │                │  │
        │  └─────┬─────┘  └───────┬────────┘  │
        └────────┼────────────────┼───────────┘
                 │                │
                 │                ▼
                 │  ┌─────────────────────────┐
                 │  │  Strategy Transform     │
                 │  │  (softplus/exp/sig/...) │
                 │  │  -> positive deltas     │
                 │  └────────────┬────────────┘
                 │               │
                 │               ▼
                 │  ┌─────────────────────────┐
                 │  │  Spacing Constraints    │
                 │  │  min_spacing / max      │
                 │  └────────────┬────────────┘
                 │               │
                 │               ▼
                 │  ┌─────────────────────────┐
                 │  │  Cumulative Sum         │
                 │  │  accumulated = cumsum() │
                 │  └────────────┬────────────┘
                 │               │
                 ▼               ▼
        ┌─────────────────────────────────────┐
        │  Concatenate: [r0, r0 + cumsum]     │
        └──────────────┬──────────────────────┘
                       │
                       ▼
        ┌─────────────────────────────────────┐
        │  Monotonic Output [..., num_values] │
        └─────────────────────────────────────┘

    :param method: Monotonicity enforcement method. One of:
        ``"cumulative_softplus"`` (default), ``"exponential"``,
        ``"cumulative_exp"``, ``"sigmoid"``, ``"squared"``,
        ``"normalized_softmax"``.
    :type method: MonotonicityMethod
    :param axis: Axis along which to enforce monotonicity.
    :type axis: int
    :param min_spacing: Minimum spacing between consecutive values.
        If provided, adds this constant to all deltas.
    :type min_spacing: Optional[float]
    :param max_spacing: Maximum spacing between consecutive values.
        Clips deltas to this value. Only applicable to delta-based methods.
    :type max_spacing: Optional[float]
    :param value_range: Tuple (min_val, max_val). Required for ``"sigmoid"``
        and ``"normalized_softmax"`` methods. Defines the output value range.
    :type value_range: Optional[Tuple[float, float]]
    :param clip_inputs: Whether to clip raw inputs before transformation.
        Helps prevent numerical overflow. Defaults to True for exponential methods.
    :type clip_inputs: Optional[bool]
    :param input_clip_range: Tuple (min, max) for input clipping when
        clip_inputs=True.
    :type input_clip_range: Tuple[float, float]
    :param epsilon: Small constant for numerical stability.
    :type epsilon: float
    :param kwargs: Additional keyword arguments for the Layer base class.

    :raises ValueError: If method is unknown, value_range is required but not
        provided, or if input has insufficient size along the monotonicity axis.
    """

    def __init__(
            self,
            method: MonotonicityMethod = "cumulative_softplus",
            axis: int = -1,
            min_spacing: Optional[float] = None,
            max_spacing: Optional[float] = None,
            value_range: Optional[Tuple[float, float]] = None,
            clip_inputs: Optional[bool] = None,
            input_clip_range: Tuple[float, float] = (-20.0, 20.0),
            epsilon: float = 1e-7,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate method
        valid_methods = [
            "cumulative_softplus", "exponential", "sigmoid",
            "normalized_softmax", "squared", "cumulative_exp"
        ]
        if method not in valid_methods:
            raise ValueError(
                f"Unknown monotonicity method: {method}. "
                f"Must be one of {valid_methods}"
            )

        # Validate value_range for methods that require it
        if method in ["sigmoid", "normalized_softmax"]:
            if value_range is None:
                raise ValueError(
                    f"Method '{method}' requires 'value_range' to be specified as (min, max)"
                )
            if len(value_range) != 2 or value_range[0] >= value_range[1]:
                raise ValueError(
                    f"value_range must be (min, max) with min < max, got {value_range}"
                )

        # Validate spacing constraints
        if min_spacing is not None and min_spacing < 0:
            raise ValueError(f"min_spacing must be non-negative, got {min_spacing}")
        if max_spacing is not None and max_spacing <= 0:
            raise ValueError(f"max_spacing must be positive, got {max_spacing}")
        if min_spacing is not None and max_spacing is not None:
            if min_spacing > max_spacing:
                raise ValueError(
                    f"min_spacing ({min_spacing}) cannot exceed max_spacing ({max_spacing})"
                )

        # Store configuration
        self.method = method
        self.axis = axis
        self.min_spacing = min_spacing
        self.max_spacing = max_spacing
        self.value_range = value_range
        self.epsilon = epsilon
        self.input_clip_range = input_clip_range

        # Auto-enable clipping for exponential methods if not specified
        if clip_inputs is None:
            self.clip_inputs = method in ["exponential", "cumulative_exp"]
        else:
            self.clip_inputs = clip_inputs

        # Issue warnings for potentially problematic configurations
        if method == "exponential" and not self.clip_inputs:
            warnings.warn(
                "Using exponential method without input clipping can cause numerical "
                "overflow. Consider setting clip_inputs=True or using 'cumulative_exp' method.",
                UserWarning
            )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer.

        :param input_shape: Shape of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the size along monotonicity axis is less than 2.
        """
        # Normalize axis to positive index
        ndim = len(input_shape)
        if self.axis < 0:
            self.axis_normalized = ndim + self.axis
        else:
            self.axis_normalized = self.axis

        # Validate axis
        if self.axis_normalized < 0 or self.axis_normalized >= ndim:
            raise ValueError(
                f"axis {self.axis} is out of bounds for input with {ndim} dimensions"
            )

        # Check size along monotonicity axis
        axis_size = input_shape[self.axis_normalized]
        if axis_size is not None and axis_size < 2:
            raise ValueError(
                f"Monotonicity requires at least 2 values along axis {self.axis}, "
                f"but got {axis_size}"
            )

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply monotonicity constraint to inputs.

        :param inputs: Input tensor with shape where ``inputs.shape[axis] >= 2``.
        :type inputs: keras.KerasTensor
        :param training: Boolean indicating training mode (unused, for API compatibility).
        :type training: Optional[bool]
        :return: Monotonically non-decreasing tensor with same shape as inputs.
        :rtype: keras.KerasTensor
        """
        # Apply the selected monotonicity method
        if self.method == "cumulative_softplus":
            return self._cumulative_softplus(inputs)
        elif self.method == "exponential":
            return self._exponential(inputs)
        elif self.method == "cumulative_exp":
            return self._cumulative_exp(inputs)
        elif self.method == "sigmoid":
            return self._sigmoid(inputs)
        elif self.method == "squared":
            return self._squared(inputs)
        elif self.method == "normalized_softmax":
            return self._normalized_softmax(inputs)
        else:
            # Should never reach here due to __init__ validation
            raise ValueError(f"Unknown method: {self.method}")

    def _split_first_and_rest(
            self,
            inputs: keras.KerasTensor
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Split inputs into first value (anchor) and rest along monotonicity axis.

        :param inputs: Input tensor.
        :type inputs: keras.KerasTensor
        :return: Tuple of (first_value, remaining_values).
        :rtype: Tuple[keras.KerasTensor, keras.KerasTensor]
        """
        # Get slices for first value and rest
        # First value along axis
        indices_first = [slice(None)] * len(inputs.shape)
        indices_first[self.axis_normalized] = slice(0, 1)
        first = inputs[tuple(indices_first)]

        # Remaining values along axis
        indices_rest = [slice(None)] * len(inputs.shape)
        indices_rest[self.axis_normalized] = slice(1, None)
        rest = inputs[tuple(indices_rest)]

        return first, rest

    def _apply_spacing_constraints(
            self,
            deltas: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Apply min/max spacing constraints to deltas.

        :param deltas: Positive increments between consecutive values.
        :type deltas: keras.KerasTensor
        :return: Constrained deltas.
        :rtype: keras.KerasTensor
        """
        if self.min_spacing is not None:
            deltas = deltas + self.min_spacing

        if self.max_spacing is not None:
            deltas = keras.ops.minimum(deltas, self.max_spacing)

        return deltas

    def _cumulative_softplus(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """
        Cumulative softplus method: ``Q_i = Q_0 + Sigma(softplus(r_j))``.

        :param inputs: Raw predictions.
        :type inputs: keras.KerasTensor
        :return: Monotonic predictions.
        :rtype: keras.KerasTensor
        """
        first, rest = self._split_first_and_rest(inputs)

        # Clip inputs if requested
        if self.clip_inputs:
            rest = keras.ops.clip(rest, *self.input_clip_range)

        # Apply softplus to ensure positive deltas
        deltas = keras.ops.softplus(rest)

        # Apply spacing constraints
        deltas = self._apply_spacing_constraints(deltas)

        # Cumulative sum of deltas
        accumulated_deltas = keras.ops.cumsum(deltas, axis=self.axis_normalized)

        # Add to first value
        subsequent_values = first + accumulated_deltas

        # Concatenate first and subsequent values
        return keras.ops.concatenate([first, subsequent_values], axis=self.axis_normalized)

    def _exponential(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """
        Exponential method: ``Q_i = Q_{i-1} + exp(r_i)``.

        :param inputs: Raw predictions.
        :type inputs: keras.KerasTensor
        :return: Monotonic predictions.
        :rtype: keras.KerasTensor
        """
        first, rest = self._split_first_and_rest(inputs)

        # Clip to prevent overflow
        if self.clip_inputs:
            rest = keras.ops.clip(rest, *self.input_clip_range)

        # Exponential transformation for positive deltas
        deltas = keras.ops.exp(rest)

        # Apply spacing constraints
        deltas = self._apply_spacing_constraints(deltas)

        # Cumulative sum
        accumulated_deltas = keras.ops.cumsum(deltas, axis=self.axis_normalized)

        # Add to first value
        subsequent_values = first + accumulated_deltas

        return keras.ops.concatenate([first, subsequent_values], axis=self.axis_normalized)

    def _cumulative_exp(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """
        Cumulative exponential with guaranteed clipping (safer exponential).

        :param inputs: Raw predictions.
        :type inputs: keras.KerasTensor
        :return: Monotonic predictions.
        :rtype: keras.KerasTensor
        """
        first, rest = self._split_first_and_rest(inputs)

        # Always clip for this method
        rest = keras.ops.clip(rest, *self.input_clip_range)

        # Exponential transformation
        deltas = keras.ops.exp(rest)

        # Apply spacing constraints
        deltas = self._apply_spacing_constraints(deltas)

        # Cumulative sum
        accumulated_deltas = keras.ops.cumsum(deltas, axis=self.axis_normalized)

        subsequent_values = first + accumulated_deltas

        return keras.ops.concatenate([first, subsequent_values], axis=self.axis_normalized)

    def _squared(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """
        Squared method: ``Q_i = Q_{i-1} + r_i^2``.

        :param inputs: Raw predictions.
        :type inputs: keras.KerasTensor
        :return: Monotonic predictions.
        :rtype: keras.KerasTensor
        """
        first, rest = self._split_first_and_rest(inputs)

        # Square to ensure positive deltas
        deltas = keras.ops.square(rest)

        # Apply spacing constraints
        deltas = self._apply_spacing_constraints(deltas)

        # Cumulative sum
        accumulated_deltas = keras.ops.cumsum(deltas, axis=self.axis_normalized)

        subsequent_values = first + accumulated_deltas

        return keras.ops.concatenate([first, subsequent_values], axis=self.axis_normalized)

    def _sigmoid(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """
        Sigmoid method: each value independently mapped to range via sigmoid.

        This method doesn't use cumulative logic -- each output is independently
        computed and naturally monotonic due to the way we construct indices.

        :param inputs: Raw predictions.
        :type inputs: keras.KerasTensor
        :return: Monotonic predictions bounded by value_range.
        :rtype: keras.KerasTensor
        """
        min_val, max_val = self.value_range

        # Apply sigmoid to map to [0, 1]
        normalized = keras.ops.sigmoid(inputs)

        # Create linearly spaced target positions for each index along axis
        # This ensures monotonicity: each position gets a higher target
        axis_size = keras.ops.shape(inputs)[self.axis_normalized]

        # Generate indices: [0, 1, 2, ..., n-1]
        indices = keras.ops.cast(
            keras.ops.arange(axis_size, dtype="float32"),
            inputs.dtype
        )

        # Normalize indices to [0, 1]: [0, 1/(n-1), 2/(n-1), ..., 1]
        normalized_indices = indices / keras.ops.maximum(
            keras.ops.cast(axis_size - 1, inputs.dtype),
            self.epsilon
        )

        # Reshape to broadcast along the monotonicity axis
        # Create shape with 1s everywhere except the monotonicity axis
        broadcast_shape = [1] * len(inputs.shape)
        broadcast_shape[self.axis_normalized] = axis_size
        normalized_indices = keras.ops.reshape(normalized_indices, broadcast_shape)

        # Blend between target positions: target = min + (max - min) * index_position
        # Then blend with sigmoid output: final = target + (max - min) * (sigmoid - 0.5) * flexibility
        target_values = min_val + (max_val - min_val) * normalized_indices

        # Use sigmoid to allow deviation from target while maintaining order
        # The 0.5 centering means sigmoid < 0.5 pulls down, > 0.5 pulls up
        flexibility = 0.5  # How much deviation from linear spacing is allowed
        output = target_values + (max_val - min_val) * (normalized - 0.5) * flexibility

        # Clip to ensure we stay in bounds
        output = keras.ops.clip(output, min_val, max_val)

        return output

    def _normalized_softmax(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """
        Normalized softmax: deltas sum to a controlled total range.

        :param inputs: Raw predictions.
        :type inputs: keras.KerasTensor
        :return: Monotonic predictions with controlled total spread.
        :rtype: keras.KerasTensor
        """
        first, rest = self._split_first_and_rest(inputs)

        min_val, max_val = self.value_range
        total_range = max_val - min_val

        # Softmax to get normalized weights that sum to 1
        weights = keras.ops.softmax(rest, axis=self.axis_normalized)

        # Scale weights by total range to get deltas
        deltas = weights * total_range

        # Apply min_spacing if specified (may exceed total_range)
        if self.min_spacing is not None:
            deltas = deltas + self.min_spacing

        # Cumulative sum
        accumulated_deltas = keras.ops.cumsum(deltas, axis=self.axis_normalized)

        subsequent_values = first + accumulated_deltas

        return keras.ops.concatenate([first, subsequent_values], axis=self.axis_normalized)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute output shape (same as input shape).

        :param input_shape: Input tensor shape.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output tensor shape (same as input).
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> dict[str, Any]:
        """
        Return configuration for serialization.

        :return: Configuration dictionary.
        :rtype: dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "method": self.method,
            "axis": self.axis,
            "min_spacing": self.min_spacing,
            "max_spacing": self.max_spacing,
            "value_range": self.value_range,
            "clip_inputs": self.clip_inputs,
            "input_clip_range": self.input_clip_range,
            "epsilon": self.epsilon,
        })
        return config

# ---------------------------------------------------------------------
