"""
BandLogitNorm Layer Implementation

This module implements a custom Keras layer that applies constrained L2 normalization
with adaptive scaling. The layer normalizes input tensors to unit L2 norm and then
applies a learned scaling factor bounded within a specified band around 1.0.

Mathematical Operation:
1. Normalize input to unit L2 norm: x_norm = x / ||x||_2
2. Apply LayerNormalization to the L2 norms: norm_scaled = LayerNorm(||x||_2)
3. Apply tanh activation to bound the normalized norms: bounded = tanh(4 * norm_scaled)
4. Scale to [1-max_band_width, 1]: scale = (1-α) + α * (bounded + 1)/2
5. Apply scaling: output = x_norm * scale

This ensures the output has L2 norm in the range [1-max_band_width, 1].
"""

import keras
from keras import ops
from typing import Any, Dict, Optional, Tuple, Union

from dl_techniques.utils.logger import logger


@keras.saving.register_keras_serializable()
class BandLogitNorm(keras.layers.Layer):
    """
    Band-constrained logit normalization layer.

    This layer applies L2 normalization to input tensors and constrains the resulting
    L2 norm to lie within a specified band around 1.0. The scaling factor is learned
    through a LayerNormalization applied to the original L2 norms, followed by a
    tanh activation for bounded scaling.

    The normalization process consists of several steps:
    1. Compute L2 norm along specified axis
    2. Normalize to unit L2 norm (places vectors on unit hypersphere)
    3. Apply LayerNormalization to the computed norms for adaptive scaling
    4. Use tanh activation to bound the scaling factors
    5. Map the bounded values to [1-max_band_width, 1] range
    6. Apply the learned scaling to the normalized vectors

    This approach allows the model to learn appropriate magnitudes for normalized
    vectors while keeping them within a controlled range for stability.

    Mathematical formulation:
        x_norm = x / ||x||_2
        norm_scaled = LayerNorm(||x||_2)
        bounded = tanh(4 * norm_scaled)
        scale = (1-α) + α * (bounded + 1) / 2
        output = x_norm * scale

    Where α is max_band_width and the final L2 norm is in [1-α, 1].

    Args:
        max_band_width: Float between 0 and 1, maximum allowed deviation from unit
            norm. Controls how much the L2 norm can deviate below 1.0. For example,
            max_band_width=0.1 allows norms in [0.9, 1.0]. Smaller values provide
            tighter constraints. Defaults to 0.01.
        axis: Integer, axis along which to compute the L2 norm. Typically -1 for
            the feature dimension in dense layers, or channel dimension in CNNs.
            Defaults to -1.
        epsilon: Float, small constant for numerical stability when computing
            norms and avoiding division by zero. Should be positive and small.
            Defaults to 1e-7.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        Arbitrary N-D tensor. The L2 norm is computed along the specified axis.
        Common shapes:
        - 2D: (batch_size, features) for dense layers
        - 3D: (batch_size, sequence_length, features) for sequence models
        - 4D: (batch_size, height, width, channels) for CNNs

    Output shape:
        Same shape as input.

    Attributes:
        norm: LayerNormalization layer applied to the computed L2 norms.
            This layer learns to adaptively scale the norms before applying
            the band constraint.

    Example:
        ```python
        # Basic usage with tight band constraint
        inputs = keras.Input(shape=(256,))
        x = keras.layers.Dense(128)(inputs)
        x = BandLogitNorm(max_band_width=0.05)(x)  # L2 norm in [0.95, 1.0]
        outputs = keras.layers.Dense(10)(x)
        model = keras.Model(inputs, outputs)

        # Usage in transformer with looser constraint
        inputs = keras.Input(shape=(128, 512))  # (seq_len, embed_dim)
        x = BandLogitNorm(axis=-1, max_band_width=0.2)(inputs)  # Norm in [0.8, 1.0]

        # Usage in CNN along channel dimension
        inputs = keras.Input(shape=(224, 224, 64))
        x = BandLogitNorm(axis=-1, max_band_width=0.1)(inputs)  # Per-pixel channel norm

        # Custom epsilon for numerical stability
        layer = BandLogitNorm(
            max_band_width=0.15,
            axis=-1,
            epsilon=1e-6  # Higher precision
        )
        ```

    Raises:
        ValueError: If max_band_width is not between 0 and 1.
        ValueError: If epsilon is not positive.

    Note:
        This implementation follows the modern Keras 3 pattern where all sub-layers
        are created in __init__ and Keras handles the building automatically. The
        LayerNormalization sub-layer learns to adaptively scale the L2 norms before
        applying the band constraint, providing more flexibility than fixed scaling.
    """

    def __init__(
        self,
        max_band_width: float = 0.01,
        axis: int = -1,
        epsilon: float = 1e-7,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if not (0.0 < max_band_width < 1.0):
            raise ValueError(
                f"max_band_width must be between 0 and 1, got {max_band_width}"
            )
        if epsilon <= 0.0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        # Store ALL configuration arguments as instance attributes
        self.max_band_width = max_band_width
        self.axis = axis
        self.epsilon = epsilon

        # CREATE sub-layers in __init__ following modern Keras 3 pattern
        # LayerNormalization will learn to adaptively scale the L2 norms
        self.norm = keras.layers.LayerNormalization(
            axis=-1,  # Normalize across the last dimension of the norm tensor
            epsilon=self.epsilon,
            name=f"{self.name}_layer_norm" if self.name else "layer_norm"
        )

        # No weights to create directly, so no custom build() method needed

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply constrained L2 normalization with adaptive scaling.

        Args:
            inputs: Input tensor to normalize.
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode. Passed to the internal
                LayerNormalization layer.

        Returns:
            Normalized tensor with L2 norm in [1-max_band_width, 1] range.
            The exact norm values are learned through the LayerNormalization
            applied to the original norms.
        """
        # Step 1: Compute L2 norm along the specified axis
        # Shape: [..., specified_axis, ...] -> [..., 1, ...] (with keepdims=True)
        x_squared = ops.square(inputs)
        x_sum_squared = ops.sum(x_squared, axis=self.axis, keepdims=True)

        # Add epsilon and clamp for numerical stability
        x_sum_squared = ops.maximum(x_sum_squared, self.epsilon)
        x_length = ops.sqrt(x_sum_squared)

        # Step 2: Normalize to unit L2 norm
        # This gives us a tensor with ||x||_2 = 1 (places vectors on unit hypersphere)
        x_normalized = inputs / x_length

        # Step 3: Apply LayerNormalization to the L2 norms
        # This learns to adaptively scale the norms, centering them around 0
        # with unit standard deviation, allowing the model to learn appropriate
        # magnitude adjustments
        x_length_normalized = self.norm(x_length, training=training)

        # Step 4: Apply tanh activation to bound the normalized norms to [-1, +1]
        # The factor of 4 provides good gradient flow while ensuring bounds
        # tanh(4x) has steeper gradients near 0 and saturates at ±1
        bounded_norms = keras.activations.tanh(4.0 * x_length_normalized)

        # Step 5: Transform from [-1, +1] to [1-max_band_width, 1]
        # First scale to [0, 1]: (tanh_output + 1) / 2
        # Then scale to target range: (1 - max_band_width) + max_band_width * [0,1]
        scale_01 = (bounded_norms + 1.0) / 2.0
        scale = (1.0 - self.max_band_width) + self.max_band_width * scale_01

        # Step 6: Apply the learned scaling to the unit-normalized tensor
        # Final output has L2 norm in [1-max_band_width, 1]
        return x_normalized * scale

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the shape of the output tensor.

        Args:
            input_shape: Shape tuple of the input tensor.

        Returns:
            Output shape tuple. Same as input shape for normalization layers.
        """
        return tuple(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Return the layer's configuration for serialization.

        This method must return ALL arguments needed to recreate the layer
        via __init__.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "max_band_width": self.max_band_width,
            "axis": self.axis,
            "epsilon": self.epsilon,
        })
        return config

    # Note: from_config() method is REMOVED as it's not needed in modern Keras 3.
    # Keras handles layer reconstruction automatically using the configuration
    # returned by get_config().