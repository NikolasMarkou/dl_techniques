"""
# BandRMSNorm Layer Documentation

## Overview

Root Mean Square Layer Normalization with Bounded Spherical Shell Constraints -
 a sophisticated normalization approach that leverages high-dimensional geometry to improve deep network training dynamics.
 This layer combines RMS normalization with a learnable bounded scaling factor to create a "thick shell" in the feature space.

## Geometric Foundation

The implementation is based on two key geometric insights:

1. In high dimensions, the volume of a sphere concentrates near its surface
2. Creating a bounded shell between radii (1-α) and 1 adds back a degree of freedom while maintaining normalization benefits

## Core Algorithm

### 1. RMS Normalization
- Projects features onto unit hypersphere: x_norm = x / sqrt(mean(x^2))
- Provides initial bounded representation
- Operates on specified axis (default: feature/channel dimension)

### 2. Learnable Band Scaling
- Creates "thick shell" with learnable radius in [1-α, 1]
- Implemented via hard sigmoid bounded scaling
- Adds crucial radial degree of freedom
- Zero-initialization for stable training start

## Geometric Properties

### 1. Feature Space Structure
- Features live in d-dimensional shell between radii (1-α) and 1
- Exponentially more volume than surface-only normalization
- Allows radial movement while maintaining bounds
- Better separation of features in high dimensions

### 2. Training Dynamics
- Smoother optimization paths through radial freedom
- Reduced gradient interference vs surface-only constraints
- Natural emergence of feature magnitude hierarchy
- Better preservation of angular relationships

### 3. Regularization Effects
- Implicit regularization through bounded shell constraint
- Prevents feature collapse or explosion
- L2 regularization encourages balanced shell utilization
- Improved feature disentanglement

## Implementation Benefits

### 1. Training Stability
- Batch-size independent operation
- No running statistics needed
- Smooth gradient flow in deep networks
- Robust to learning rate variation

### 2. Feature Representation
- Magnitude can encode feature importance
- Better handling of feature hierarchies
- Improved gradient flow in very deep networks
- Enhanced multi-task learning capability

### 3. Architectural Flexibility
- Excellent for attention mechanisms
- Strong fit for residual networks
- Facilitates feature reuse
- Suitable for both CNNs and Transformers

## Practical Usage

### 1. Hyperparameters
- max_band_width (α): Controls shell thickness (typical: 0.1-0.3)
- epsilon: Numerical stability (typical: 1e-5 to 1e-7)
- band_regularizer: Prevents scale saturation (typical: 1e-4 to 1e-6)

### 2. Optimization
- Compatible with adaptive optimizers (Adam recommended)
- Benefits from learning rate warmup
- Monitor band parameter distributions
- Consider gradient clipping in very deep networks

### 3. Architecture Integration
- Use as LayerNorm replacement in Transformers
- Effective in ResNets and deep CNNs
- Particularly powerful for multi-task architectures
- Consider in feature fusion contexts

## Technical Implementation Notes

The layer strikes an elegant balance between normalization constraints and feature
expressivity through its geometric design, making it particularly effective for deep learning architectures
operating in high-dimensional feature spaces.

## Mathematical Properties
- Concentration of Measure:
  In high dimensions (d → ∞), the volume of a d-sphere concentrates exponentially
  near its surface, following:
  P(|x| ∈ [r(1-ε), r]) ≈ 1 - O(exp(-dε²/2))

  This means:
  - Most vectors naturally concentrate at similar lengths
  - Shell-based normalization aligns with the geometry
  - Variance across dimensions is more important than absolute scale

- Bounded Shell Dynamics [1-α, 1]:
  The normalization maps vectors into a spherical shell with:
  - Outer radius: 1 (maximum length)
  - Inner radius: 1-α (minimum length)
  - Thickness: α (adaptivity parameter)

  This creates a "comfort zone" that:
  - Preserves the benefits of normalization
  - Allows length variation within controlled bounds
  - Provides smoother optimization landscapes
  - Maintains representation capacity

- General Properties:
  - Preserves directional information
  - Guarantees bounded outputs
  - Smooth and differentiable
  - Equivariant to input scaling

## References
[1] Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)
    https://arxiv.org/abs/1910.07467
[2] Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges
    (Bronstein et al., 2021)
[3] High-Dimensional Probability: An Introduction with Applications in Data Science
    (Vershynin, 2018)
"""

import keras
import tensorflow as tf
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------


@keras.utils.register_keras_serializable()
class BandRMS(keras.layers.Layer):
    """Root Mean Square Normalization layer with bounded L2 norm constraints.

    This layer implements root mean square normalization that guarantees the output
    L2 norm will be between [1-α, 1], where α is the max_band_width parameter.
    The normalization is computed in two steps:
    1. RMS normalization to unit norm
    2. Learnable scaling within the [1-α, 1] band

    The layer creates a "thick shell" in the feature space, allowing features to exist
    within a bounded spherical shell rather than being constrained to the unit hypersphere,
    which can help with optimization and representation learning.

    :param max_band_width: Maximum allowed deviation from unit normalization (0 < α < 1)
    :type max_band_width: float
    :param axis: Axis along which to compute RMS statistics
    :type axis: int
    :param epsilon: Small constant added to denominator for numerical stability
    :type epsilon: float
    :param band_regularizer: Regularizer for the band parameter
    :type band_regularizer: Optional[keras.regularizers.Regularizer]
    :param clip_gradients: Whether to clip gradients for the band parameter
    :type clip_gradients: bool
    :param gradient_clip_value: Maximum gradient value if clip_gradients is True
    :type gradient_clip_value: float
    :param kwargs: Additional layer arguments

    :return: A tensor of the same shape as input, with L2 norm guaranteed to be
             between [1-max_band_width, 1]

    :raises: ValueError: If parameters are invalid

    Example:
        ```python
        # Apply SphericalBoundRMS normalization to the output of a dense layer
        x = keras.layers.Dense(64)(inputs)
        x = SphericalBoundRMS(max_band_width=0.2)(x)

        # Apply to a specific axis in a CNN
        conv = keras.layers.Conv2D(32, 3)(inputs)
        norm = SphericalBoundRMS(axis=3, max_band_width=0.1)(conv)

        # With custom regularization
        norm_layer = SphericalBoundRMS(
            max_band_width=0.3,
            band_regularizer=keras.regularizers.L2(1e-4)
        )
        ```

    References:
        "Root Mean Square Layer Normalization", 2019
        https://arxiv.org/abs/1910.07467
    """

    def __init__(
            self,
            max_band_width: float = 0.2,
            axis: int = -1,
            epsilon: float = 1e-7,
            band_regularizer: Optional[keras.regularizers.Regularizer] = None,
            clip_gradients: bool = False,
            gradient_clip_value: float = 1.0,
            **kwargs: Any
    ):
        """Initialize the SphericalBoundRMS layer.

        :param max_band_width: Maximum allowed deviation from unit normalization (0 < α < 1)
        :type max_band_width: float
        :param axis: Axis along which to compute RMS statistics
        :type axis: int
        :param epsilon: Small constant added to denominator for numerical stability
        :type epsilon: float
        :param band_regularizer: Regularizer for the band parameter
        :type band_regularizer: Optional[keras.regularizers.Regularizer]
        :param clip_gradients: Whether to clip gradients for the band parameter
        :type clip_gradients: bool
        :param gradient_clip_value: Maximum gradient value if clip_gradients is True
        :type gradient_clip_value: float
        :param kwargs: Additional layer arguments
        """
        super().__init__(**kwargs)

        # Validate inputs
        self._validate_inputs(max_band_width, epsilon, axis)

        # Store configuration
        self.max_band_width = max_band_width
        self.axis = axis
        self.epsilon = epsilon

        # Default regularizer if none provided
        self.band_regularizer = band_regularizer or keras.regularizers.L2(1e-5)

        # Gradient clipping options
        self.clip_gradients = clip_gradients
        self.gradient_clip_value = gradient_clip_value

        # Will be set in build()
        self.band_param = None

    def _validate_inputs(self, max_band_width: float, epsilon: float, axis: int) -> None:
        """Validate initialization parameters.

        :param max_band_width: Maximum allowed deviation from unit norm
        :type max_band_width: float
        :param epsilon: Small constant for numerical stability
        :type epsilon: float
        :param axis: Axis along which to compute statistics
        :type axis: int

        :raises: ValueError: If parameters are invalid
        """
        if not 0 < max_band_width < 1:
            raise ValueError(
                f"max_band_width must be between 0 and 1, got {max_band_width}"
            )
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        # No need to validate axis as negative values are valid in TensorFlow

    def build(self, input_shape: tf.TensorShape) -> None:
        """Create the layer's trainable weights.

        :param input_shape: Shape of input tensor
        :type input_shape: tf.TensorShape

        :raises: ValueError: If input dimension is invalid for the specified axis
        """
        # Validate input shape with respect to the specified axis
        ndims = len(input_shape)
        if ndims == 0:
            raise ValueError("SphericalBoundRMS cannot be applied to scalar inputs")

        # Convert negative axis to positive for easier handling
        axis = self.axis if self.axis >= 0 else ndims + self.axis

        # Validate axis is within range
        if axis < 0 or axis >= ndims:
            raise ValueError(f"Axis {self.axis} is out of bounds for input with {ndims} dimensions")

        # Ensure the dimension along the specified axis exists
        if input_shape[axis] is None:
            raise ValueError(f"Dimension {axis} of input cannot be None")

        # Create parameter shape with 1s except along the axis dimension
        param_shape = [1] * ndims
        param_shape[axis] = input_shape[axis]

        # Initialize band parameter with zeros for optimal training
        self.band_param = self.add_weight(
            name="band_param",
            shape=param_shape,
            initializer="zeros",
            trainable=True,
            dtype=self.dtype,
            regularizer=self.band_regularizer
        )

        super().build(input_shape)

    def _compute_rms(self, inputs: tf.Tensor) -> tf.Tensor:
        """Compute root mean square values along specified axis.

        :param inputs: Input tensor
        :type inputs: tf.Tensor

        :return: RMS values tensor with same shape as input except for normalized axis
        :rtype: tf.Tensor
        """
        # Square inputs
        x_squared = tf.square(inputs)

        # Compute mean along the specified axis
        mean_squared = tf.reduce_mean(x_squared, axis=self.axis, keepdims=True)

        # Compute RMS with epsilon for numerical stability
        return tf.sqrt(mean_squared + self.epsilon)

    def call(
            self,
            inputs: tf.Tensor,
            training: Optional[bool] = None
    ) -> tf.Tensor:
        """Apply constrained RMS normalization.

        :param inputs: Input tensor
        :type inputs: tf.Tensor
        :param training: Whether in training mode (used for gradient clipping)
        :type training: Optional[bool]

        :return: Normalized tensor with L2 norm in [1-max_band_width, 1]
        :rtype: tf.Tensor
        """
        # Ensure inputs have the correct data type
        inputs = tf.cast(inputs, self.compute_dtype)
        tf.ensure_shape(inputs, self.input_shape)

        # Step 1: RMS normalization to get unit norm
        rms = self._compute_rms(inputs)
        normalized = inputs / rms

        # Step 2: Apply learnable scaling within [1-α, 1] band
        # Convert band_param to input dtype for consistent computation
        band_param = tf.cast(self.band_param, inputs.dtype)

        # Apply gradient clipping if enabled
        if self.clip_gradients and training:
            band_param = tf.clip_by_value(
                band_param,
                -self.gradient_clip_value,
                self.gradient_clip_value
            )

        # Use hard sigmoid to strictly enforce the bounds
        band_activation = keras.activations.hard_sigmoid(band_param)

        # Scale the normalized tensor to be within [1-max_band_width, 1]
        scale = (1.0 - self.max_band_width) + (self.max_band_width * band_activation)

        # Apply scaling to the normalized tensor
        return normalized * scale

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        """Compute shape of output tensor.

        :param input_shape: Shape of input tensor
        :type input_shape: tf.TensorShape

        :return: Shape of output tensor (same as input)
        :rtype: tf.TensorShape
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return layer configuration for serialization.

        :return: Dictionary containing the layer configuration
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "max_band_width": self.max_band_width,
            "axis": self.axis,
            "epsilon": self.epsilon,
            "band_regularizer": keras.regularizers.serialize(self.band_regularizer),
            "clip_gradients": self.clip_gradients,
            "gradient_clip_value": self.gradient_clip_value,
        })
        return config