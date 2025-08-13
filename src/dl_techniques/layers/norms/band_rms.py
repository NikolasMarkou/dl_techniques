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
from typing import Any, Dict, Optional, Union

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class BandRMS(keras.layers.Layer):
    """Root Mean Square Normalization layer with bounded RMS constraints.

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

    Args:
        max_band_width: Maximum allowed deviation from unit normalization (0 < α < 1).
            Controls the thickness of the spherical shell.
        axis: int or tuple of ints, default=-1
            Axis or axes along which to compute RMS statistics. The default (-1)
            computes RMS over the last dimension.
        epsilon: Small constant added to denominator for numerical stability.
        band_initializer: str or initializer, default="zeros"
            Initializer for the single band parameter.
        band_regularizer: Regularizer for the band parameter. Default is L2(1e-5).
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
            axis: Union[int, tuple] = -1,
            epsilon: float = 1e-7,
            band_initializer: Union[str, keras.initializers.Initializer] = "zeros",
            band_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any
    ):
        """Initialize the BandRMS layer.

        Args:
            max_band_width: Maximum allowed deviation from unit normalization (0 < α < 1).
                Controls the thickness of the spherical shell.
            axis: int or tuple of ints, default=-1
                Axis or axes along which to compute RMS statistics.
            epsilon: Small constant added to denominator for numerical stability.
            band_initializer: Initializer for the single band parameter.
            band_regularizer: Regularizer for the band parameter. Default is L2(1e-5).
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
        self.band_initializer = keras.initializers.get(band_initializer)

        # Default regularizer if none provided
        self.band_regularizer = band_regularizer or keras.regularizers.L2(1e-5)

        # Will be set in build()
        self.band_param = None
        self._build_input_shape = None

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
        """
        self._build_input_shape = input_shape

        # Create a single scalar band parameter
        self.band_param = self.add_weight(
            name="band_param",
            shape=(),  # Scalar shape
            initializer=self.band_initializer,
            trainable=True,
            regularizer=self.band_regularizer
        )

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply constrained RMS normalization.

        Args:
            inputs: Input tensor.
            training: Boolean indicating whether in training mode.

        Returns:
            Normalized tensor with RMS value in [1-max_band_width, 1] and
            L2 norm approximately in [(1-max_band_width)×sqrt(D), sqrt(D)].
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

        # clamp it for stability, protection for division by zero and sqrt(negative)
        rms = ops.maximum(
            ops.sqrt(mean_square + self.epsilon),
            self.epsilon
        )

        # Normalize by RMS: this sets RMS=1 and L2_norm≈sqrt(D)
        # This scaling is dimension-independent, unlike L2 normalization
        normalized = inputs_fp32 / rms

        # Step 2: Apply learnable scaling within [1-α, 1] band
        # Use sigmoid to map the band_param to [0, 1]
        # with 5x multiplier, sigmoid(-5) ~ 0, sigmoid(+5) ~ 1
        band_activation = ops.sigmoid(5.0 * self.band_param)

        # Scale the activation to be within [1-max_band_width, 1]
        # When band_activation = 0: scale = 1 - max_band_width
        # When band_activation = 1: scale = 1
        scale = (1.0 - self.max_band_width) + (self.max_band_width * band_activation)

        # Apply scaling to the normalized tensor
        # The single scalar scale is automatically broadcast to all elements
        output = normalized * scale

        # Cast back to original dtype
        return ops.cast(output, inputs.dtype)

    def compute_output_shape(self, input_shape) -> tuple:
        """Compute the shape of output tensor.

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
            "band_initializer": keras.initializers.serialize(self.band_initializer),
            "band_regularizer": keras.regularizers.serialize(self.band_regularizer),
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get build configuration for serialization."""
        return {"input_shape": self._build_input_shape}

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build layer from configuration."""
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

# ---------------------------------------------------------------------