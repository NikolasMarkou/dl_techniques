"""
BandLogitNorm Layer Implementation

This module implements a custom Keras layer that applies constrained RMS normalization
with adaptive scaling. The layer normalizes input tensors to unit L2 norm and then
applies a learned scaling factor bounded within a specified band around 1.0.

Mathematical Operation:
1. Normalize input to unit L2 norm: x_norm = x / ||x||_2
2. Apply LayerNormalization to the L2 norms: norm_scaled = LayerNorm(||x||_2)
3. Apply tanh activation to bound the normalized norms: bounded = tanh(norm_scaled)
4. Scale to [1-max_band_width, 1]: scale = max_band_width * (bounded + 1)/2 + (1 - max_band_width)
5. Apply scaling: output = x_norm * scale

This ensures the output has L2 norm in the range [1-max_band_width, 1].

Author: [Your Name]
Date: [Date]
Version: 1.0
"""

import keras
from keras import ops
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class BandLogitNorm(keras.layers.Layer):
    """
    Band-constrained logit normalization layer.

    This layer applies RMS normalization to input tensors and constrains the resulting
    L2 norm to lie within a specified band around 1.0. The scaling factor is learned
    through a LayerNormalization applied to the original L2 norms.

    :param max_band_width: Maximum allowed deviation from unit norm (0 < max_band_width < 1)
    :type max_band_width: float
    :param axis: Axis along which to compute the L2 norm, defaults to -1
    :type axis: int
    :param epsilon: Small constant for numerical stability, defaults to 1e-7
    :type epsilon: float
    :param kwargs: Additional keyword arguments passed to the base Layer
    :type kwargs: Any

    :raises ValueError: If max_band_width is not in (0, 1) or epsilon is not positive

    Example:
        >>> layer = BandLogitNorm(max_band_width=0.1)
        >>> output = layer(input_tensor)  # L2 norm will be in [0.9, 1.0]
    """

    def __init__(
            self,
            max_band_width: float = 0.01,
            axis: int = -1,
            epsilon: float = 1e-7,
            **kwargs: Any
    ):
        """
        Initialize the BandLogitNorm layer.

        :param max_band_width: Maximum allowed deviation from unit norm
        :type max_band_width: float
        :param axis: Axis along which to compute the L2 norm
        :type axis: int
        :param epsilon: Small constant for numerical stability
        :type epsilon: float
        :param kwargs: Additional keyword arguments
        :type kwargs: Any
        """
        super().__init__(**kwargs)

        # Validate inputs before storing
        self._validate_inputs(max_band_width, epsilon)

        # Store configuration parameters
        self.axis = axis
        self.epsilon = epsilon
        self.max_band_width = max_band_width

        # Initialize normalization layer (will be properly configured in build())
        self.norm = None

    def _validate_inputs(self, max_band_width: float, epsilon: float) -> None:
        """
        Validate initialization parameters.

        :param max_band_width: Maximum allowed deviation from unit norm
        :type max_band_width: float
        :param epsilon: Small constant for numerical stability
        :type epsilon: float
        :raises ValueError: If parameters are invalid
        """
        if not 0 < max_band_width < 1:
            raise ValueError(
                f"max_band_width must be between 0 and 1, got {max_band_width}"
            )
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

    def build(self, input_shape) -> None:
        """
        Build the layer by initializing the LayerNormalization sublayer.

        :param input_shape: Shape of the input tensor
        :type input_shape: tuple or tf.TensorShape
        """
        super().build(input_shape)

        # Initialize LayerNormalization with the same axis configuration
        # This normalizes the L2 norms to have zero mean and unit variance
        self.norm = keras.layers.LayerNormalization(
            axis=-1,  # Normalize across the last dimension of the norm tensor
            epsilon=self.epsilon,
            name=f"{self.name}_layer_norm"
        )

        # Build the normalization layer
        # The norm tensor will have shape [..., 1] due to keepdims=True
        norm_shape = list(input_shape)
        norm_shape[self.axis] = 1
        self.norm.build(norm_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply constrained RMS normalization.

        :param inputs: Input tensor to normalize
        :type inputs: keras.KerasTensor
        :param training: Whether the layer is in training mode
        :type training: Optional[bool]
        :returns: Normalized tensor with L2 norm in [1-max_band_width, 1]
        :rtype: keras.KerasTensor
        """
        x = inputs

        # Step 1: Compute L2 norm along the specified axis
        # Shape: [..., specified_axis, ...] -> [..., 1, ...] (with keepdims=True)
        x_squared = ops.square(x)
        x_sum_squared = ops.maximum(ops.sum(x_squared, axis=self.axis, keepdims=True), self.epsilon)
        x_length = ops.sqrt(x_sum_squared)  # Add epsilon for numerical stability

        # Step 2: Normalize to unit L2 norm
        # This gives us a tensor with ||x||_2 = 1
        x_normalized = x / x_length

        # Step 3: Apply LayerNormalization to the L2 norms
        # This centers the norms around 0 with unit standard deviation
        x_length_normalized = self.norm(x_length, training=training)

        # Step 4: Apply tanh activation to bound the normalized norms to [-1, +1]
        # This ensures the scaling factor will be well-behaved
        x_length_normalized = keras.activations.tanh(4 * x_length_normalized)

        # Step 5: Transform from [-1, +1] to [1-max_band_width, 1]
        # First scale to [0, 1]: (tanh_output + 1) / 2
        # Then scale to [1-max_band_width, 1]: max_band_width * [0,1] + (1 - max_band_width)
        scale = (x_length_normalized + 1.0) / 2.0
        scale = (1.0 - self.max_band_width) + self.max_band_width * scale

        # Step 6: Apply the learned scaling to the unit-normalized tensor
        # Final output has L2 norm in [1-max_band_width, 1]
        return x_normalized * scale

    def compute_output_shape(self, input_shape) -> tuple:
        """
        Compute the shape of the output tensor.

        :param input_shape: Shape of the input tensor
        :type input_shape: tuple or tf.TensorShape
        :returns: Shape of the output tensor (same as input)
        :rtype: tuple
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return the layer configuration for serialization.

        :returns: Dictionary containing the layer configuration
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "axis": self.axis,
            "epsilon": self.epsilon,
            "max_band_width": self.max_band_width
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'BandLogitNorm':
        """
        Create a layer instance from its configuration.

        :param config: Layer configuration dictionary
        :type config: Dict[str, Any]
        :returns: New layer instance
        :rtype: BandLogitNorm
        """
        return cls(**config)

# ---------------------------------------------------------------------