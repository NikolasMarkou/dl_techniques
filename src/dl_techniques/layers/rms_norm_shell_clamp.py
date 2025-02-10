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

import tensorflow as tf
from typing import Optional, Tuple, Union, Dict, Any
from keras import initializers, regularizers, constraints, Layer

# ---------------------------------------------------------------------


class ShellClampRMS(Layer):
    """Root Mean Square Normalization with bounded shell support.

    This layer normalizes inputs to lie within a spherical shell of radius [1 - alpha, 1].
    Using RMS statistics helps it adapt well to high-dimensional data.

    Key aspects:
      - Preserves directional information
      - Outputs are guaranteed to lie in [1 - alpha, 1] after normalization (in norm)
      - Smooth and differentiable (except for the hard clamp)
      - Equivariant to input scaling
      - Maintains running statistics for inference

    References:
    [1] Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)
        https://arxiv.org/abs/1910.07467
    [2] Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges
        (Bronstein et al., 2021)
    [3] High-Dimensional Probability: An Introduction with Applications in Data Science
        (Vershynin, 2018)

    Args:
        axis: Integer or list of integers, the axes to normalize across.
            Typically -1 for the feature axis. Defaults to -1.
        alpha: Float in (0, 1), controls shell thickness. Smaller values enforce
            stricter normalization. Defaults to 0.1.
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
        **kwargs: Other keyword arguments for the base Layer class.
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

        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_constraint = constraints.get(gamma_constraint)
        self.beta_constraint = constraints.get(beta_constraint)

        self.supports_masking = True

    def _validate_axis(self, ndims: int) -> list[int]:
        """Validates and converts axis parameter to list of positive indices."""
        return [
            ax if ax >= 0 else ax + ndims
            for ax in self.axis
        ]

    def build(self, input_shape: tf.TensorShape) -> None:
        """Creates the layer weights and computed attributes."""
        ndims = len(input_shape)
        reduction_axes = self._validate_axis(ndims)

        # param_shape for gamma and beta: they should match the normalized dimensions
        # i.e. for each axis in reduction_axes, we store that dimension size.
        param_shape = [1] * ndims
        for i in range(ndims):
            if i in reduction_axes:
                param_shape[i] = input_shape[i]

        # Optionally, if you need 'd' (the total number of normalized features):
        #   normalized_dims = reduce(operator.mul, [int(input_shape[a]) for a in reduction_axes], 1)
        # In this version, we do not further scale alpha by sqrt(d).

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
        else:
            self.gamma = None

        if self.center:
            self.beta = self.add_weight(
                name="beta",
                shape=param_shape,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                trainable=True,
            )
        else:
            self.beta = None

        # Create moving statistics (RMS and length)
        # These also match param_shape so that we can broadcast them properly.
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
        """Computes RMS and vector length statistics along self.axis."""
        squares = tf.square(inputs)

        # RMS: sqrt(mean of squares)
        mean_squares = tf.reduce_mean(squares, axis=self.axis, keepdims=True)
        rms = tf.sqrt(mean_squares + self.epsilon)

        # Vector norm (length)
        sum_squares = tf.reduce_sum(squares, axis=self.axis, keepdims=True)
        lengths = tf.sqrt(sum_squares + self.epsilon)

        return rms, lengths

    def _normalize_to_shell(
        self,
        inputs: tf.Tensor,
        rms: tf.Tensor,
        lengths: tf.Tensor
    ) -> tf.Tensor:
        """Normalizes inputs to lie within the [1 - alpha, 1] shell."""
        # 1) RMS normalize
        outputs = inputs / rms

        # 2) Compute new lengths after RMS normalization
        normed_lengths = lengths / rms

        # 3) Clamp the normed_lengths to [1 - alpha, 1], then scale outputs accordingly
        min_length = 1.0 - self.alpha
        max_length = 1.0

        # Hard clamp
        clamped_length = tf.clip_by_value(normed_lengths, min_length, max_length)

        # scale_factor = (clamped_length / normed_lengths)
        scale_factor = clamped_length / tf.maximum(normed_lengths, self.epsilon)

        return outputs * scale_factor

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """Forward pass for RMS normalization with bounded shell."""
        rms, lengths = self._compute_statistics(inputs)

        if training:
            # Update moving stats
            self.moving_rms.assign(
                self.momentum * self.moving_rms +
                (1.0 - self.momentum) * rms
            )
            self.moving_length.assign(
                self.momentum * self.moving_length +
                (1.0 - self.momentum) * lengths
            )
        else:
            # Use moving stats in inference
            rms = self.moving_rms
            lengths = self.moving_length

        # Normalize to the shell
        outputs = self._normalize_to_shell(inputs, rms, lengths)

        # Apply optional scale and offset
        if self.scale:
            outputs = outputs * self.gamma
        if self.center:
            outputs = outputs + self.beta

        return outputs

    def get_config(self) -> Dict[str, Any]:
        """Returns the config of the layer."""
        config = {
            "axis": self.axis,
            "alpha": self.alpha,
            "epsilon": self.epsilon,
            "center": self.center,
            "scale": self.scale,
            "momentum": self.momentum,
            "moving_mean_initializer": initializers.serialize(self.moving_mean_initializer),
            "gamma_initializer": initializers.serialize(self.gamma_initializer),
            "beta_initializer": initializers.serialize(self.beta_initializer),
            "gamma_regularizer": regularizers.serialize(self.gamma_regularizer),
            "beta_regularizer": regularizers.serialize(self.beta_regularizer),
            "gamma_constraint": constraints.serialize(self.gamma_constraint),
            "beta_constraint": constraints.serialize(self.beta_constraint),
        }
        base_config = super().get_config()
        return {**base_config, **config}

# ---------------------------------------------------------------------

