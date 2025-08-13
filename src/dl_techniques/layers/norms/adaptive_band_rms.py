"""
Adaptive BandRMS Layer: RMS Normalization with Log-Transformed RMS-Statistics-Based Scaling (FIXED).

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

    **FIX**: Correctly reshapes and broadcasts scaling factors to match original
    tensor dimensions, supporting both 2D (dense) and 4D (convolutional) inputs.

This approach leverages the mathematical properties of logarithmic transformation:
- **Variance Stabilization**: Compresses dynamic range of RMS values
- **Symmetry**: Transforms log-normal-like RMS distributions to more Gaussian-like
- **Batch Independence**: No dependence on batch statistics, unlike BatchNorm
- **Numerical Stability**: Better handling of extreme RMS values
- **Spatial Awareness**: Correctly handles spatial dimensions in convolutional layers

The key insight is using log-transformed RMS statistics (capturing input "energy"
in a stable numerical space) as conditioning information to determine appropriate
normalization strength for each specific input pattern, while maintaining proper
tensor structure for both dense and convolutional architectures.

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
    """Adaptive Root Mean Square Normalization layer with log-transformed RMS scaling (FIXED).

    This layer implements root mean square normalization with scaling factors computed
    from logarithmically transformed RMS statistics. The log transformation stabilizes
    the variance of RMS values and creates a more symmetric distribution for the
    dense layer to process.

    The key innovation is using log-transformed magnitude characteristics (RMS statistics)
    to determine appropriate normalization strength, creating input-adaptive "thick shells"
    in the RMS space with improved numerical stability and correct tensor handling.

    The normalization pipeline:
    1. RMS normalization: sets RMS=1, L2_norm≈sqrt(D) + extract RMS statistics
    2. Log transformation: log(RMS) for variance stabilization and symmetry
    3. Dense projection: compute scaling parameters from log-transformed RMS (FIXED shapes)
    4. Adaptive scaling: apply RMS-conditioned scaling within [1-α, 1] band (FIXED broadcasting)

    This allows the layer to learn different normalization behaviors based on input
    magnitude patterns with better handling of extreme values, improved numerical
    stability, and correct support for both dense and convolutional architectures.

    Args:
        max_band_width: Maximum allowed deviation from unit normalization (0 < α < 1).
            Controls the maximum thickness of the adaptive spherical shell.
        axis: int or tuple of ints, default=-1
            Axis or axes along which to compute RMS statistics. The default (-1)
            computes RMS over the last dimension. For 4D tensors:
            - axis=-1 or 3: Channel-wise normalization
            - axis=(1,2): Spatial normalization
            - axis=(1,2,3): Global normalization
        epsilon: Small constant added to denominator for numerical stability.
        band_initializer: str or initializer, default="zeros"
            Initializer for the dense layer computing band parameters.
        band_regularizer: Regularizer for the dense layer weights.
        **kwargs: Additional layer arguments.

    Input shape:
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
        Supports both 2D (batch, features) and 4D (batch, height, width, channels).

    Output shape:
        Same shape as input.

    Example:
    ```python
    # Apply Adaptive BandRMS normalization to the output of a dense layer
    x = keras.layers.Dense(64)(inputs)
    x = AdaptiveBandRMS(max_band_width=0.2)(x)

    # Apply to a specific axis in a CNN - channel-wise normalization
    conv = keras.layers.Conv2D(32, 3)(inputs)
    norm = AdaptiveBandRMS(axis=-1, max_band_width=0.1)(conv)

    # Spatial normalization in CNN
    norm = AdaptiveBandRMS(axis=(1, 2), max_band_width=0.15)(conv)

    # Global normalization in CNN
    norm = AdaptiveBandRMS(axis=(1, 2, 3), max_band_width=0.2)(conv)

    # With custom regularization
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

        # FIX: Track tensor dimensionality for proper shape handling
        self._is_conv_layer = False
        self._normalization_strategy = None

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

    def _analyze_normalization_strategy(self, input_shape) -> tuple:
        """Analyze input shape and axis to determine normalization strategy.

        FIX: This method determines the correct dense layer dimensions and
        broadcasting strategy based on input tensor shape and normalization axis.

        Args:
            input_shape: Shape of input tensor.

        Returns:
            Tuple of (dense_input_dim, dense_output_dim, strategy_name)

        Raises:
            ValueError: If axis configuration is invalid for the input shape.
        """
        self._is_conv_layer = len(input_shape) == 4

        if not self._is_conv_layer:
            # Standard 2D case (batch, features)
            return 1, input_shape[-1], "feature_wise"

        # 4D convolutional case - validate and determine strategy
        if isinstance(self.axis, int):
            if self.axis == -1 or self.axis == 3:
                # Channel-wise normalization: one scaling factor per channel
                self._normalization_strategy = "channel_wise"
                return 1, input_shape[-1], "channel_wise"
            else:
                raise ValueError(
                    f"For 4D tensors, single axis must be -1 or 3 (channels), got {self.axis}"
                )

        elif isinstance(self.axis, (list, tuple)):
            axis_set = set(self.axis)
            if axis_set == {1, 2}:
                # Spatial normalization: scaling per channel based on spatial RMS
                self._normalization_strategy = "spatial"
                return 1, input_shape[-1], "spatial"
            elif axis_set == {1, 2, 3}:
                # Global normalization: single scaling factor for entire feature map
                self._normalization_strategy = "global"
                return 1, 1, "global"
            else:
                raise ValueError(
                    f"For 4D tensors, axis tuple must be (1,2) or (1,2,3), got {self.axis}"
                )
        else:
            raise ValueError(f"axis must be int or tuple, got {type(self.axis)}")

    def build(self, input_shape):
        """Create the layer's sublayers and weights.

        Args:
            input_shape: Shape of input tensor.
        """
        self._build_input_shape = input_shape

        # FIX: Analyze normalization strategy to determine correct dimensions
        dense_input_dim, dense_output_dim, strategy = self._analyze_normalization_strategy(input_shape)

        # Create dense layer to process log-transformed RMS statistics
        # FIX: Use correct dimensions determined by normalization strategy
        self.dense_layer = keras.layers.Dense(
            units=dense_output_dim,  # Correct output dimension for each strategy
            kernel_initializer=self.band_initializer,
            kernel_regularizer=self.band_regularizer,
            use_bias=True,
            name="band_dense"
        )

        # Build dense layer with appropriate input shape
        # FIX: Dense layer input is always [batch, 1] for log-transformed RMS statistics
        log_rms_shape = [None, dense_input_dim]
        self.dense_layer.build(log_rms_shape)

        super().build(input_shape)

    def _compute_rms_statistics(self, inputs_fp32: keras.KerasTensor) -> keras.KerasTensor:
        """Compute and aggregate RMS statistics for log transformation.

        FIX: This method properly aggregates RMS statistics while preserving
        batch dimension and handling different normalization strategies.

        Args:
            inputs_fp32: Input tensor in float32.

        Returns:
            RMS statistics tensor with shape [batch, 1] ready for log transformation.
        """
        # Step 1: Compute RMS with keepdims=True (preserves tensor structure)
        mean_square = ops.mean(
            ops.square(inputs_fp32),
            axis=self.axis,
            keepdims=True
        )

        rms = ops.maximum(
            ops.sqrt(mean_square + self.epsilon),
            self.epsilon
        )

        # Step 2: Aggregate RMS statistics to [batch, 1] for dense layer
        # FIX: Proper aggregation strategy based on normalization type
        if not self._is_conv_layer:
            # 2D case: RMS shape is already [batch, 1]
            rms_stats = ops.reshape(rms, [-1, 1])

        else:
            # 4D case: Need to aggregate appropriately
            if self._normalization_strategy == "channel_wise":
                # RMS shape: [batch, height, width, 1]
                # Aggregate across spatial dimensions for global channel statistic
                rms_stats = ops.mean(rms, axis=(1, 2), keepdims=False)  # [batch, 1]

            elif self._normalization_strategy == "spatial":
                # RMS shape: [batch, 1, 1, channels]
                # Aggregate across channels for global spatial statistic
                rms_stats = ops.mean(rms, axis=(1, 2, 3), keepdims=False)  # [batch]
                rms_stats = ops.reshape(rms_stats, [-1, 1])  # [batch, 1]

            elif self._normalization_strategy == "global":
                # RMS shape: [batch, 1, 1, 1]
                # Already a single statistic per batch
                rms_stats = ops.reshape(rms, [-1, 1])  # [batch, 1]

        return rms_stats, rms

    def _apply_adaptive_scaling(
            self,
            normalized: keras.KerasTensor,
            band_logits: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Apply adaptive scaling with proper broadcasting.

        FIX: This method ensures scaling factors are correctly reshaped and
        broadcast to match the original tensor dimensions.

        Args:
            normalized: RMS-normalized tensor.
            band_logits: Output from dense layer.

        Returns:
            Adaptively scaled tensor with same shape as input.
        """
        # Convert logits to scaling factors in [1-α, 1] range
        band_activation = ops.sigmoid(5.0 * band_logits)
        scale = (1.0 - self.max_band_width) + (self.max_band_width * band_activation)

        # FIX: Reshape scaling factors for proper broadcasting
        if not self._is_conv_layer:
            # 2D case: scale shape [batch, features] broadcasts naturally
            pass

        else:
            # 4D case: Need to reshape for proper broadcasting
            if self._normalization_strategy == "channel_wise":
                # scale shape: [batch, channels] -> [batch, 1, 1, channels]
                scale = ops.reshape(scale, [-1, 1, 1, scale.shape[-1]])

            elif self._normalization_strategy == "spatial":
                # scale shape: [batch, channels] -> [batch, 1, 1, channels]
                scale = ops.reshape(scale, [-1, 1, 1, scale.shape[-1]])

            elif self._normalization_strategy == "global":
                # scale shape: [batch, 1] -> [batch, 1, 1, 1]
                scale = ops.reshape(scale, [-1, 1, 1, 1])

        return normalized * scale

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
        # FIX: Properly aggregate RMS statistics while preserving batch structure
        rms_stats, _ = self._compute_rms_statistics(inputs_fp32)

        # Apply log transformation with additional epsilon for numerical safety
        # Since RMS >= epsilon > 0, log(RMS) is always well-defined
        # This stabilizes variance and handles the long-tailed nature of RMS distributions
        log_rms = ops.log(rms_stats)

        # Step 3: Dense projection to compute adaptive band parameters
        # The dense layer processes the log-transformed RMS statistics
        # FIX: Dense layer now receives correctly shaped input [batch, 1]
        band_logits = self.dense_layer(log_rms, training=training)

        # Step 4: Apply adaptive scaling within [1-α, 1] band
        # FIX: Proper reshaping and broadcasting for different tensor types
        output = self._apply_adaptive_scaling(normalized, band_logits)

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