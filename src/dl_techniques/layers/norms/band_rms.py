"""
BandRMS Layer: Root Mean Square Normalization with Bounded Spherical Shell Constraints

This layer implements an advanced normalization approach that leverages high-dimensional
geometry to improve deep network training dynamics. It combines RMS normalization with
a learnable bounded scaling factor to create a "thick shell" in the feature space.

The implementation is based on two key geometric insights:
1. In high dimensions, the volume of a sphere concentrates near its surface
2. Creating a bounded shell between radii (1-α) and 1 adds back a degree of freedom
   while maintaining normalization benefits

Core Algorithm:
1. RMS Normalization: Projects features onto unit hypersphere (x_norm = x / sqrt(mean(x^2)))
2. Learnable Band Scaling: Creates "thick shell" with learnable radius in [1-α, 1]

References:
[1] Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)
    https://arxiv.org/abs/1910.07467
"""

import keras
from keras import ops
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
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

    Args:
        max_band_width: Maximum allowed deviation from unit normalization (0 < α < 1).
            Controls the thickness of the spherical shell.
        axis: Axis along which to compute RMS statistics. Default is -1 (last dimension).
        epsilon: Small constant added to denominator for numerical stability.
        band_regularizer: Regularizer for the band parameter. Default is L2(1e-5).
        clip_gradients: Whether to clip gradients for the band parameter.
        gradient_clip_value: Maximum gradient value if clip_gradients is True.
        **kwargs: Additional layer arguments.

    Input shape:
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    Output shape:
        Same shape as input.

    Example:
    ```python
    # Apply BandRMS normalization to the output of a dense layer
    x = keras.layers.Dense(64)(inputs)
    x = BandRMS(max_band_width=0.2)(x)

    # Apply to a specific axis in a CNN
    conv = keras.layers.Conv2D(32, 3)(inputs)
    norm = BandRMS(axis=3, max_band_width=0.1)(conv)

    # With custom regularization
    norm_layer = BandRMS(
        max_band_width=0.3,
        band_regularizer=keras.regularizers.L2(1e-4)
    )
    ```
    """

    def __init__(
            self,
            max_band_width: float = 0.1,
            axis: int = -1,
            epsilon: float = 1e-7,
            band_regularizer: Optional[keras.regularizers.Regularizer] = None,
            clip_gradients: bool = False,
            gradient_clip_value: float = 1.0,
            **kwargs: Any
    ):
        """Initialize the BandRMS layer.

        Args:
            max_band_width: Maximum allowed deviation from unit normalization (0 < α < 1).
                Controls the thickness of the spherical shell.
            axis: Axis along which to compute RMS statistics. Default is -1 (last dimension).
            epsilon: Small constant added to denominator for numerical stability.
            band_regularizer: Regularizer for the band parameter. Default is L2(1e-5).
            clip_gradients: Whether to clip gradients for the band parameter.
            gradient_clip_value: Maximum gradient value if clip_gradients is True.
            **kwargs: Additional layer arguments.

        Raises:
            ValueError: If max_band_width is not between 0 and 1 or if epsilon is not positive.
        """
        super().__init__(**kwargs)

        # Validate inputs
        self._validate_inputs(max_band_width, epsilon)

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

    def _validate_inputs(self, max_band_width: float, epsilon: float) -> None:
        """Validate initialization parameters.

        Args:
            max_band_width: Maximum allowed deviation from unit norm.
            epsilon: Small constant for numerical stability.

        Raises:
            ValueError: If parameters are invalid.
        """
        if not 0 < max_band_width < 1:
            raise ValueError(
                f"max_band_width must be between 0 and 1, got {max_band_width}"
            )
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

    def build(self, input_shape):
        """Create the layer's trainable weights.

        Args:
            input_shape: Shape of input tensor.

        Raises:
            ValueError: If input dimension is invalid for the specified axis.
        """
        # Convert input_shape to list for easier manipulation
        input_shape = list(input_shape)

        # Validate input shape
        ndims = len(input_shape)
        if ndims == 0:
            raise ValueError("BandRMS cannot be applied to scalar inputs")

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
            regularizer=self.band_regularizer
        )

        super().build(input_shape)

    def _compute_rms(self, inputs):
        """Compute root mean square values along specified axis.

        Args:
            inputs: Input tensor.

        Returns:
            RMS values tensor with same shape as input except for normalized axis.
        """
        # Square inputs
        x_squared = ops.square(inputs)

        # Compute mean along the specified axis
        mean_squared = ops.mean(x_squared, axis=self.axis, keepdims=True)

        # Compute RMS with epsilon for numerical stability
        return ops.sqrt(mean_squared + self.epsilon)

    def call(self, inputs, training=None):
        """Apply constrained RMS normalization.

        Args:
            inputs: Input tensor.
            training: Boolean indicating whether in training mode (used for gradient clipping).

        Returns:
            Normalized tensor with L2 norm in [1-max_band_width, 1].
        """
        # Step 1: RMS normalization to get unit norm
        rms = self._compute_rms(inputs)
        normalized = inputs / rms

        # Step 2: Apply learnable scaling within [1-α, 1] band
        # Apply gradient clipping if enabled and in training mode
        band_param = self.band_param
        if self.clip_gradients and training:
            band_param = ops.clip(
                band_param,
                -self.gradient_clip_value,
                self.gradient_clip_value
            )

        # Use hard sigmoid to strictly enforce the bounds
        # Hard sigmoid: max(0, min(1, (x + 0.5) / 1))
        band_activation = ops.clip((band_param + 0.5), 0, 1)

        # Scale the normalized tensor to be within [1-max_band_width, 1]
        scale = (1.0 - self.max_band_width) + (self.max_band_width * band_activation)

        # Apply scaling to the normalized tensor
        return normalized * scale

    def compute_output_shape(self, input_shape):
        """Compute shape of output tensor.

        Args:
            input_shape: Shape of input tensor.

        Returns:
            Shape of output tensor (same as input).
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
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

# ---------------------------------------------------------------------
