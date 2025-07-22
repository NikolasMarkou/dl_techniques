"""
Adaptive BandRMS Layer: RMS Normalization with Log-Transformed RMS-Statistics-Based Scaling.

This layer implements an advanced normalization technique that extends Root Mean
Square Normalization (RMSNorm) by using logarithmically transformed RMS statistics
to dynamically compute scaling factors. This creates adaptive "thick spherical shell"
constraints based on the magnitude characteristics of the input data.

The layer operates in a four-step process:

1.  **RMS Normalization:**
    First, it applies standard RMS Normalization. For an input vector `x` of
    dimension `D`, this step computes `x_norm = x / sqrt(mean(x²) + ε)` and
    retains the RMS statistics for further processing.

2.  **Logarithmic Transformation:**
    The computed RMS statistics undergo a logarithmic transformation to stabilize
    variance and handle the long-tailed nature of magnitude distributions. This
    makes the statistics more numerically stable and symmetric for processing.

3.  **Dense Projection:**
    The log-transformed RMS statistics are passed through a dense layer to compute
    adaptive scaling parameters that depend on the input's magnitude characteristics.

4.  **Adaptive Band Scaling:**
    The scaling parameters are constrained to [1 - max_band_width, 1] using sigmoid
    activation and applied to the RMS-normalized features. This creates input-dependent
    spherical shell constraints where the "thickness" adapts based on RMS statistics.

This approach leverages the mathematical properties of logarithmic transformation:
- **Variance Stabilization**: Compresses dynamic range of RMS values
- **Symmetry**: Transforms log-normal-like RMS distributions to more Gaussian-like
- **Batch Independence**: No dependence on batch statistics, unlike BatchNorm
- **Numerical Stability**: Better handling of extreme RMS values

The key insight is using log-transformed RMS statistics (capturing input "energy"
in a stable numerical space) as conditioning information to determine appropriate
normalization strength for each specific input pattern.

References:
[1] Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)
    https://arxiv.org/abs/1910.07467
"""

import keras
from keras import ops
from typing import Any, Dict, Optional, Union

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class AdaptiveBandRMS(keras.layers.Layer):
    """Adaptive Root Mean Square Normalization layer with log-transformed RMS scaling.

    This layer implements root mean square normalization with scaling factors computed
    from logarithmically transformed RMS statistics. The log transformation stabilizes
    the variance of RMS values and creates a more symmetric distribution for the
    dense layer to process.

    The key innovation is using log-transformed magnitude characteristics (RMS statistics)
    to determine appropriate normalization strength, creating input-adaptive "thick shells"
    in the RMS space with improved numerical stability.

    The normalization pipeline:
    1. RMS normalization: sets RMS=1, L2_norm≈sqrt(D) + extract RMS statistics
    2. Log transformation: log(RMS) for variance stabilization and symmetry
    3. Dense projection: compute scaling parameters from log-transformed RMS
    4. Adaptive scaling: apply RMS-conditioned scaling within [1-α, 1] band

    This allows the layer to learn different normalization behaviors based on input
    magnitude patterns with better handling of extreme values and improved numerical
    stability compared to batch-dependent normalization approaches.

    Args:
        max_band_width: Maximum allowed deviation from unit normalization (0 < α < 1).
            Controls the maximum thickness of the adaptive spherical shell.
        axis: int or tuple of ints, default=-1
            Axis or axes along which to compute RMS statistics. The default (-1)
            computes RMS over the last dimension.
        epsilon: Small constant added to denominator for numerical stability.
        band_initializer: str or initializer, default="zeros"
            Initializer for the dense layer computing band parameters.
        band_regularizer: Regularizer for the dense layer weights.
        **kwargs: Additional layer arguments.

    Input shape:
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    Output shape:
        Same shape as input.

    Example:
    ```python
    # Apply Adaptive BandRMS normalization to the output of a dense layer
    x = keras.layers.Dense(64)(inputs)
    x = AdaptiveBandRMS(max_band_width=0.2)(x)

    # Apply to a specific axis in a CNN
    conv = keras.layers.Conv2D(32, 3)(inputs)
    norm = AdaptiveBandRMS(axis=3, max_band_width=0.1)(conv)

    # With custom regularization and dense units
    norm_layer = AdaptiveBandRMS(
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
        """Initialize the Adaptive BandRMS layer.

        Args:
            max_band_width: Maximum allowed deviation from unit normalization (0 < α < 1).
                Controls the maximum thickness of the adaptive spherical shell.
            axis: int or tuple of ints, default=-1
                Axis or axes along which to compute RMS statistics.
            epsilon: Small constant added to denominator for numerical stability.
            band_initializer: Initializer for the dense layer computing band parameters.
            band_regularizer: Regularizer for the dense layer weights.
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
        self.band_regularizer = band_regularizer

        # Sublayers - will be initialized in build()
        self.dense_layer = None
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
        """Create the layer's sublayers and weights.

        Args:
            input_shape: Shape of input tensor.
        """
        self._build_input_shape = input_shape

        # Get the feature dimension
        feature_dim = input_shape[-1]

        # Create dense layer to process log-transformed RMS statistics
        self.dense_layer = keras.layers.Dense(
            units=feature_dim,  # Output per-feature scaling factors
            kernel_initializer=self.band_initializer,
            kernel_regularizer=self.band_regularizer,
            use_bias=True,
            name="band_dense"
        )

        # Build dense layer with appropriate shape
        # Log-transformed RMS statistics will have shape [batch, 1] after reshaping
        log_rms_shape = [None, 1]
        self.dense_layer.build(log_rms_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply adaptive constrained RMS normalization with log-transformed statistics.

        Args:
            inputs: Input tensor.
            training: Boolean indicating whether in training mode.

        Returns:
            Normalized tensor with adaptive RMS constraints based on log-transformed
            input magnitude characteristics.
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

        # Clamp for stability, protection for division by zero and sqrt(negative)
        rms = ops.maximum(
            ops.sqrt(mean_square + self.epsilon),
            self.epsilon
        )

        # Normalize by RMS: this sets RMS=1 and L2_norm≈sqrt(D)
        # This scaling is dimension-independent, unlike L2 normalization
        normalized = inputs_fp32 / rms

        # Step 2: Logarithmic transformation of RMS statistics
        # This stabilizes variance and handles the long-tailed nature of RMS distributions
        # log(RMS) transforms log-normal-like distributions to more Gaussian-like
        # and compresses the dynamic range for better numerical stability
        rms_reshaped = ops.reshape(rms, [-1, 1])  # Shape: [batch, 1]

        # Apply log transformation with additional epsilon for numerical safety
        # Since RMS >= epsilon > 0, log(RMS) is always well-defined
        log_rms = ops.log(rms_reshaped)

        # Step 3: Dense projection to compute adaptive band parameters
        # The dense layer processes the log-transformed RMS statistics
        band_logits = self.dense_layer(log_rms, training=training)

        # Step 4: Apply adaptive scaling within [1-α, 1] band
        # Use sigmoid to map the band logits to [0, 1]
        # with 5x multiplier for steeper gradients
        band_activation = ops.sigmoid(5.0 * band_logits)

        # Scale the activation to be within [1-max_band_width, 1]
        # When band_activation = 0: scale = 1 - max_band_width
        # When band_activation = 1: scale = 1
        scale = (1.0 - self.max_band_width) + (self.max_band_width * band_activation)

        # Apply adaptive scaling to the RMS-normalized tensor
        # The scale has shape [batch, feature_dim] for per-feature scaling
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