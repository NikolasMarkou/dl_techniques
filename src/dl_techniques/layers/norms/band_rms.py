"""
BandRMS Layer: RMS Normalization within a Learnable Spherical Shell.

This layer implements an advanced normalization technique that extends Root Mean
Square Normalization (RMSNorm) by constraining feature vectors to lie within a
learnable "thick spherical shell" in the activation space. This provides the
stability of normalization while granting the model additional representational
flexibility.

The layer operates in a two-step process:

1.  **RMS Normalization:**
    First, it applies standard RMS Normalization. For an input vector `x` of
    dimension `D`, this step computes `x_norm = x / sqrt(mean(x²) + ε)`.

    **Why mean instead of sum?** RMSNorm uses mean(x²) rather than sum(x²) to achieve
    dimension-independent normalization. This means:
    - The RMS value becomes 1 regardless of vector dimension
    - A 64-dim and 512-dim layer have similar normalization behavior
    - The L2 norm becomes approximately `sqrt(D)`, scaling predictably with dimension
    - Gradients don't explode with layer width due to the 1/D factor in derivatives
    - Better optimization stability across different model architectures

    This is fundamentally different from L2 normalization, which would place vectors
    on the unit hypersphere but create dimension-dependent scaling issues.

2.  **Learnable Band Scaling:**
    Second, instead of using a simple learnable gain like in LayerNorm or RMSNorm,
    this layer multiplies the normalized vector by a single scalar `s` that is constrained
    to a specific band: `s ∈ [1 - max_band_width, 1]`. This scalar `s` is
    learnable, controlled by a single trainable parameter that is mapped to the target
    range using a sigmoid function.

This design forces the final output vector's RMS value to be learned within the
`[1-α, 1]` band, effectively placing it in a high-dimensional shell. The dimension-
independent nature of RMS normalization ensures this band constraint works consistently
across layers of different widths, which can improve optimization dynamics and model
performance.

References:
[1] Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)
    https://arxiv.org/abs/1910.07467
"""

import keras
from keras import ops
from typing import Any, Dict, Optional, Union, Tuple

from dl_techniques.utils.logger import logger


@keras.saving.register_keras_serializable()
class BandRMS(keras.layers.Layer):
    """
    Root Mean Square Normalization layer with bounded RMS constraints.

    This layer implements root mean square normalization that guarantees the output
    RMS value will be between [1-α, 1], where α is the max_band_width parameter.
    Unlike L2 normalization which constrains the actual vector length, this approach
    constrains the Root Mean Square value, making it dimension-independent.

    The normalization is computed in two steps:
    1. RMS normalization: sets RMS=1, L2_norm≈sqrt(D) (dimension-independent scaling)
    2. Learnable scaling within the [1-α, 1] band using a single global parameter

    The layer creates a "thick shell" in the RMS space rather than geometric space,
    allowing features to exist within a bounded range while maintaining dimension-
    independent behavior across different layer widths.

    Mathematical formulation:
        First step: x_norm = x / sqrt(mean(x²) + ε)
        Second step: output = x_norm * scale, where scale ∈ [1-α, 1]

    Where the scale is learned via: scale = (1-α) + α * sigmoid(5 * band_param)

    Args:
        max_band_width: Float between 0 and 1, maximum allowed deviation from unit
            normalization. Controls the thickness of the spherical shell. Higher values
            allow more flexibility but may reduce normalization stability. Defaults to 0.1.
        axis: Integer or tuple of integers, axis or axes along which to compute RMS
            statistics. The default (-1) computes RMS over the last dimension. For
            convolutional layers, you might want axis=(1, 2) for spatial dimensions.
            Defaults to -1.
        epsilon: Float, small constant added to denominator for numerical stability.
            Should be positive and small. Defaults to 1e-7.
        band_initializer: String or keras.initializers.Initializer instance,
            initializer for the single band parameter. The parameter starts at the
            initializer value and is mapped through sigmoid to the band range.
            Defaults to "zeros" (which maps to the middle of the band).
        band_regularizer: Optional keras.regularizers.Regularizer instance for the
            band parameter. Helps prevent the parameter from becoming too extreme.
            Defaults to L2(1e-5) for stability.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        Arbitrary N-D tensor. The normalization is applied along the specified axis/axes.
        Common shapes:
        - 2D: (batch_size, features) for dense layers
        - 3D: (batch_size, sequence_length, features) for transformers
        - 4D: (batch_size, height, width, channels) for CNNs

    Output shape:
        Same shape as input.

    Attributes:
        band_param: Trainable scalar parameter that controls the scaling factor.
            Mapped to [1-max_band_width, 1] range via sigmoid activation.

    Example:
        ```python
        # Basic usage with dense layer
        inputs = keras.Input(shape=(784,))
        x = keras.layers.Dense(64)(inputs)
        x = BandRMS(max_band_width=0.2)(x)
        outputs = keras.layers.Dense(10)(x)
        model = keras.Model(inputs, outputs)

        # Usage in transformer with sequence dimension
        inputs = keras.Input(shape=(128, 512))  # (seq_len, embed_dim)
        x = BandRMS(axis=-1, max_band_width=0.1)(inputs)  # Normalize over embed_dim

        # Usage in CNN with spatial normalization
        inputs = keras.Input(shape=(224, 224, 3))
        x = keras.layers.Conv2D(32, 3)(inputs)
        x = BandRMS(axis=(1, 2), max_band_width=0.15)(x)  # Normalize over spatial dims

        # Custom configuration
        norm_layer = BandRMS(
            max_band_width=0.3,
            epsilon=1e-6,
            band_initializer='random_normal',
            band_regularizer=keras.regularizers.L2(1e-4)
        )
        ```

    References:
        - Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)
          https://arxiv.org/abs/1910.07467

    Raises:
        ValueError: If max_band_width is not between 0 and 1.
        ValueError: If epsilon is not positive.

    Note:
        This implementation follows the modern Keras 3 pattern where weights
        are created in build() and all sub-layers are created in __init__.
        The layer is designed to work efficiently with mixed precision training
        by casting to float32 for computations and back to the original dtype.
    """

    def __init__(
        self,
        max_band_width: float = 0.1,
        axis: Union[int, Tuple[int, ...]] = -1,
        epsilon: float = 1e-7,
        band_initializer: Union[str, keras.initializers.Initializer] = "zeros",
        band_regularizer: Optional[keras.regularizers.Regularizer] = None,
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
        self.band_initializer = keras.initializers.get(band_initializer)

        # Set default regularizer if none provided
        if band_regularizer is None:
            band_regularizer = keras.regularizers.L2(1e-5)
        self.band_regularizer = keras.regularizers.get(band_regularizer)

        # Initialize weight attribute to None - will be created in build()
        self.band_param = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create the layer's trainable weights.

        This method is called automatically by Keras when the layer is first used.
        It creates the single trainable band parameter using add_weight().

        Args:
            input_shape: Shape tuple of the input tensor, including batch dimension.

        Raises:
            ValueError: If input_shape is invalid.
        """
        if len(input_shape) < 1:
            raise ValueError(f"Expected at least 1D input, got {len(input_shape)}D: {input_shape}")

        # CREATE the layer's single trainable weight using add_weight()
        self.band_param = self.add_weight(
            name="band_param",
            shape=(),  # Scalar parameter
            initializer=self.band_initializer,
            regularizer=self.band_regularizer,
            trainable=True,
        )

        # Let Keras know the build is complete
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply constrained RMS normalization.

        Args:
            inputs: Input tensor of arbitrary shape.
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode. Used for numerical stability
                in mixed precision training.

        Returns:
            Normalized tensor with RMS value in [1-max_band_width, 1] and
            L2 norm approximately in [(1-max_band_width)×sqrt(D), sqrt(D)],
            where D is the dimension size along the normalized axes.
        """
        # Cast to float32 for numerical stability in mixed precision training
        inputs_fp32 = ops.cast(inputs, "float32")

        # Step 1: RMS normalization to achieve dimension-independent scaling
        # Compute RMS: sqrt(mean(x²))
        # Using mean(x²) instead of sum(x²) ensures normalization is independent
        # of vector dimension - critical for consistent behavior across layer widths
        mean_square = ops.mean(
            ops.square(inputs_fp32),
            axis=self.axis,
            keepdims=True
        )

        # Compute RMS with numerical stability
        # Clamp to epsilon to protect against division by zero and sqrt(negative)
        rms = ops.maximum(
            ops.sqrt(mean_square + self.epsilon),
            self.epsilon
        )

        # Normalize by RMS: this sets RMS=1 and L2_norm≈sqrt(D)
        # This scaling is dimension-independent, unlike L2 normalization
        normalized = inputs_fp32 / rms

        # Step 2: Apply learnable scaling within [1-α, 1] band
        # Use sigmoid to map the band_param to [0, 1] with good gradient properties
        # The 5x multiplier ensures sigmoid(-5) ≈ 0 and sigmoid(+5) ≈ 1
        band_activation = ops.sigmoid(5.0 * self.band_param)

        # Map activation to the target band [1-max_band_width, 1]
        # When band_activation = 0: scale = 1 - max_band_width (minimum)
        # When band_activation = 1: scale = 1 (maximum)
        scale = (1.0 - self.max_band_width) + (self.max_band_width * band_activation)

        # Apply scaling to the normalized tensor
        # The scalar scale is automatically broadcast to all tensor elements
        output = normalized * scale

        # Cast back to original dtype to maintain precision compatibility
        return ops.cast(output, inputs.dtype)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

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
        via __init__. Uses keras serializers for complex objects.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "max_band_width": self.max_band_width,
            "axis": self.axis,
            "epsilon": self.epsilon,
            "band_initializer": keras.initializers.serialize(self.band_initializer),
            "band_regularizer": keras.regularizers.serialize(self.band_regularizer),
        })
        return config

    # Note: get_build_config() and build_from_config() methods are REMOVED
    # as they are deprecated in modern Keras 3. Keras handles the build
    # lifecycle automatically when following the modern pattern.