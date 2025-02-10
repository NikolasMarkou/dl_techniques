"""Root Mean Square Normalization with Bounded Shell Support.

This implementation leverages geometric properties of high-dimensional spaces for more
effective normalization. Key geometric insights:

1. Concentration of Measure:
   In high dimensions (d → ∞), the volume of a d-sphere concentrates exponentially
   near its surface, following:
   P(|x| ∈ [r(1-ε), r]) ≈ 1 - O(exp(-dε²/2))

   This means:
   - Most vectors naturally concentrate at similar lengths
   - Shell-based normalization aligns with the geometry
   - Variance across dimensions is more important than absolute scale

2. Bounded Shell Dynamics [1-α, 1]:
   The normalization maps vectors into a spherical shell with:
   - Outer radius: 1 (maximum length)
   - Inner radius: 1-α (minimum length)
   - Thickness: α (adaptivity parameter)

   This creates a "comfort zone" that:
   - Preserves the benefits of normalization
   - Allows length variation within controlled bounds
   - Provides smoother optimization landscapes
   - Maintains representation capacity

Mathematical Properties:
- Preserves directional information
- Guarantees bounded outputs
- Smooth and differentiable
- Equivariant to input scaling

References:
[1] Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)
    https://arxiv.org/abs/1910.07467
[2] Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges
    (Bronstein et al., 2021)
[3] High-Dimensional Probability: An Introduction with Applications in Data Science
    (Vershynin, 2018)
"""

from typing import Optional, Tuple, Union, Dict, Any
import tensorflow as tf
from keras import Layer
from keras import initializers
from keras import regularizers
from keras import constraints
from keras import backend


class RMSNorm(Layer):
    """Root Mean Square Normalization with bounded shell support.

    This layer normalizes inputs to lie within a spherical shell of radius [1-α, 1],
    using RMS statistics and accounting for high-dimensional geometric effects.

    The normalization maintains running statistics and automatically adapts to the
    input dimensionality, making it particularly effective for deep networks.

    Args:
        axis: Integer or list of integers, the axes to normalize across.
            Typically -1 for the feature axis. Defaults to -1.
        alpha: Float in (0, 1), controls shell thickness. Smaller values enforce
            stricter normalization. For high dimensions (d > 100), values around
            1/sqrt(d) often work well. Defaults to 0.1.
        epsilon: Small float for numerical stability. Defaults to 1e-6.
        center: If True, add learnable offset (beta). Defaults to True.
        scale: If True, add learnable scale (gamma). Defaults to True.
        momentum: Float in [0, 1), momentum for moving statistics. Defaults to 0.99.
        moving_mean_initializer: Initializer for moving RMS. Defaults to "zeros".
        gamma_initializer: Initializer for gamma. Defaults to "ones".
        beta_initializer: Initializer for beta. Defaults to "zeros".
        gamma_regularizer: Optional regularizer for gamma.
        beta_regularizer: Optional regularizer for beta.
        gamma_constraint: Optional constraint for gamma.
        beta_constraint: Optional constraint for beta.
        **kwargs: Base layer keyword arguments.
    """

    def __init__(
            self,
            axis: Union[int, list[int]] = -1,
            alpha: float = 0.1,
            epsilon: float = 1e-6,
            center: bool = True,
            scale: bool = True,
            momentum: float = 0.99,
            moving_mean_initializer: Union[str, initializers.Initializer] = "zeros",
            gamma_initializer: Union[str, initializers.Initializer] = "ones",
            beta_initializer: Union[str, initializers.Initializer] = "zeros",
            gamma_regularizer: Optional[regularizers.Regularizer] = None,
            beta_regularizer: Optional[regularizers.Regularizer] = None,
            gamma_constraint: Optional[constraints.Constraint] = None,
            beta_constraint: Optional[constraints.Constraint] = None,
            **kwargs: Any,
    ):
        super().__init__(**kwargs)

        # Validate parameters
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if not 0 <= momentum < 1:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")

        self.axis = axis if isinstance(axis, (list, tuple)) else [axis]
        self.alpha = alpha
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.momentum = momentum

        # Initialize parameter creators
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_constraint = constraints.get(gamma_constraint)
        self.beta_constraint = constraints.get(beta_constraint)

        self.supports_masking = True

    def _validate_axis(self, ndims: int) -> list[int]:
        """Validates and converts axis parameter to list of positive indices.

        Args:
            ndims: Number of dimensions in input tensor.

        Returns:
            List of positive axis indices.

        Raises:
            ValueError: If any axis is invalid.
        """
        return [
            axis if axis >= 0 else axis + ndims
            for axis in self.axis
        ]

    def build(self, input_shape: tf.TensorShape) -> None:
        """Creates the layer weights and computed attributes.

        Args:
            input_shape: Shape of input tensor.

        Raises:
            ValueError: If input shape is invalid.
        """
        ndims = len(input_shape)
        reduction_axes = self._validate_axis(ndims)

        # Compute parameter shape - broadcast across normalized axes
        param_shape = [1] * ndims
        for i in range(ndims):
            if i not in reduction_axes:
                param_shape[i] = input_shape[i]

        # Count normalized dimensions for alpha scaling
        normalized_dims = sum(input_shape[axis] for axis in reduction_axes)
        self.dim_adjusted_alpha = self.alpha / tf.sqrt(tf.cast(normalized_dims, tf.float32))

        # Create trainable parameters if needed
        if self.scale:
            self.gamma = self.add_weight(
                name="gamma",
                shape=param_shape,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                trainable=True,
            )

        if self.center:
            self.beta = self.add_weight(
                name="beta",
                shape=param_shape,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                trainable=True,
            )

        # Create moving statistics
        # We track both RMS and length statistics for better adaptation
        self.moving_rms = self.add_weight(
            name="moving_rms",
            shape=param_shape,
            initializer=self.moving_mean_initializer,
            trainable=False,
        )

        self.moving_length = self.add_weight(
            name="moving_length",
            shape=param_shape,
            initializer=self.moving_mean_initializer,
            trainable=False,
        )

        self.built = True

    def _compute_statistics(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Computes RMS and length statistics efficiently.

        Args:
            inputs: Input tensor.

        Returns:
            Tuple of (RMS statistics, length statistics).
        """
        # Compute squares once for efficiency
        squares = tf.square(inputs)

        # Compute RMS (root mean square)
        mean_squares = tf.reduce_mean(squares, axis=self.axis, keepdims=True)
        rms = tf.sqrt(mean_squares + self.epsilon)

        # Compute lengths
        sum_squares = tf.reduce_sum(squares, axis=self.axis, keepdims=True)
        lengths = tf.sqrt(sum_squares + self.epsilon)

        return rms, lengths

    def _normalize_to_shell(
            self,
            inputs: tf.Tensor,
            rms: tf.Tensor,
            lengths: tf.Tensor
    ) -> tf.Tensor:
        """Normalizes inputs to lie within the [1-α, 1] shell.

        Args:
            inputs: Input tensor.
            rms: RMS statistics.
            lengths: Length statistics.

        Returns:
            Normalized tensor.
        """
        # Initial RMS normalization
        outputs = inputs / rms

        # Compute current lengths after RMS norm
        normed_lengths = lengths / rms

        # Scale factor to map into [1-α, 1] shell
        # Using smooth min/max for better gradients
        min_length = 1.0 - self.dim_adjusted_alpha
        max_length = 1.0

        # Compute scale factor using softplus for smoothness
        scale_factor = tf.minimum(
            tf.maximum(
                normed_lengths,
                min_length + self.epsilon
            ),
            max_length
        ) / tf.maximum(normed_lengths, self.epsilon)

        return outputs * scale_factor

    def call(
            self,
            inputs: tf.Tensor,
            training: Optional[bool] = None
    ) -> tf.Tensor:
        """Forward pass for RMS normalization.

        Args:
            inputs: Input tensor.
            training: Boolean indicating whether in training mode.

        Returns:
            Normalized tensor.
        """
        # Compute current statistics
        rms, lengths = self._compute_statistics(inputs)

        if training:
            # Update moving statistics with momentum
            self.moving_rms.assign(
                self.momentum * self.moving_rms +
                (1.0 - self.momentum) * rms
            )
            self.moving_length.assign(
                self.momentum * self.moving_length +
                (1.0 - self.momentum) * lengths
            )
        else:
            # Use moving statistics in inference mode
            rms = self.moving_rms
            lengths = self.moving_length

        # Normalize to bounded shell
        outputs = self._normalize_to_shell(inputs, rms, lengths)

        # Apply trainable parameters if needed
        if self.scale:
            outputs = outputs * self.gamma
        if self.center:
            outputs = outputs + self.beta

        return outputs

    def get_config(self) -> Dict[str, Any]:
        """Returns the config of the layer.

        Returns:
            Config dictionary.
        """
        config = {
            "axis": self.axis,
            "alpha": self.alpha,
            "epsilon": self.epsilon,
            "center": self.center,
            "scale": self.scale,
            "momentum": self.momentum,
            "moving_mean_initializer": initializers.serialize(
                self.moving_mean_initializer
            ),
            "gamma_initializer": initializers.serialize(self.gamma_initializer),
            "beta_initializer": initializers.serialize(self.beta_initializer),
            "gamma_regularizer": regularizers.serialize(self.gamma_regularizer),
            "beta_regularizer": regularizers.serialize(self.beta_regularizer),
            "gamma_constraint": constraints.serialize(self.gamma_constraint),
            "beta_constraint": constraints.serialize(self.beta_constraint),
        }
        base_config = super().get_config()
        return {**base_config, **config}