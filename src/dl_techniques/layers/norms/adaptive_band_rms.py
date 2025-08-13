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
- **Spatial Awareness**: Handles spatial dimensions in convolutional layers

The key insight is using log-transformed RMS statistics (capturing input "energy"
in a stable numerical space) as conditioning information to determine appropriate
normalization strength for each specific input pattern.

References:
[1] Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)
    https://arxiv.org/abs/1910.07467
"""

import keras
from keras import ops
from typing import Any, Dict, Optional, Union, Tuple

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
    stability for both dense and convolutional architectures.

    Args:
        max_band_width: Float between 0 and 1, maximum allowed deviation from unit
            normalization. Controls the maximum thickness of the adaptive spherical shell.
            Defaults to 0.1.
        axis: int or tuple of ints, axis or axes along which to compute RMS statistics.
            The default (-1) computes RMS over the last dimension. For 4D tensors:
            - axis=-1 or 3: Channel-wise normalization
            - axis=(1,2): Spatial normalization
            - axis=(1,2,3): Global normalization
            Defaults to -1.
        epsilon: Float, small constant added to denominator for numerical stability.
            Defaults to 1e-7.
        band_initializer: String or initializer, initializer for the dense layer
            computing band parameters. Defaults to "zeros".
        band_regularizer: Optional regularizer for the dense layer weights.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        Arbitrary. Supports both 2D (batch, features) and 4D (batch, height, width, channels)
        tensors. Use the keyword argument `input_shape` (tuple of integers, does not
        include the samples axis) when using this layer as the first layer in a model.

    Output shape:
        Same shape as input.

    Raises:
        ValueError: If max_band_width is not between 0 and 1, or if epsilon is not positive.

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
        outputs = norm_layer(inputs)
        ```

    Note:
        This implementation follows the modern Keras 3 pattern where all sub-layers
        are created in __init__ and Keras handles the building automatically. This
        ensures proper serialization and eliminates common build errors.
    """

    def __init__(
            self,
            max_band_width: float = 0.1,
            axis: Union[int, tuple] = -1,
            epsilon: float = 1e-7,
            band_initializer: Union[str, keras.initializers.Initializer] = "zeros",
            band_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        """Initialize the Adaptive BandRMS layer.

        Args:
            max_band_width: Maximum allowed deviation from unit normalization.
            axis: Axis or axes along which to compute RMS statistics.
            epsilon: Small constant for numerical stability.
            band_initializer: Initializer for the dense layer computing band parameters.
            band_regularizer: Regularizer for the dense layer weights.
            **kwargs: Additional layer arguments.

        Raises:
            ValueError: If max_band_width is not between 0 and 1 or if epsilon is not positive.
        """
        super().__init__(**kwargs)

        # Validate inputs
        if not 0 < max_band_width < 1:
            raise ValueError(
                f"max_band_width must be between 0 and 1, got {max_band_width}"
            )
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        # Store configuration parameters
        self.max_band_width = max_band_width
        self.axis = axis
        self.epsilon = epsilon
        self.band_initializer = keras.initializers.get(band_initializer)
        self.band_regularizer = band_regularizer

        # CREATE sub-layer in __init__ following modern pattern
        # The dense layer processes log-transformed RMS statistics (always scalar input)
        # and outputs scaling parameters that will be broadcast appropriately
        self.band_dense = keras.layers.Dense(
            units=1,  # Always output single scaling parameter
            kernel_initializer=self.band_initializer,
            kernel_regularizer=self.band_regularizer,
            use_bias=True,
            name="band_dense"
        )

        # Initialize weight references to None - created in build() if needed
        # For this layer, all weights are handled by the Dense sub-layer

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the layer's weights.

        This method is called automatically by Keras when the layer is first used.
        Since this layer uses a Dense sub-layer for all weights, no additional
        weights need to be created here.

        Args:
            input_shape: Shape tuple of the input tensor, including batch dimension.
        """
        # Validate input shape
        if len(input_shape) < 2:
            raise ValueError(f"Expected at least 2D input, got {len(input_shape)}D: {input_shape}")

        # Note: The Dense sub-layer will be built automatically by Keras when first called
        # No additional weights need to be created here

        super().build(input_shape)

    def _compute_rms_statistics(self, inputs: keras.KerasTensor) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Compute RMS normalization and extract statistics for adaptive scaling.

        Args:
            inputs: Input tensor in float32.

        Returns:
            Tuple of (rms_stats, normalized) where:
            - rms_stats: RMS statistics with shape [batch, 1] for dense layer
            - normalized: RMS-normalized input tensor
        """
        # Step 1: Compute RMS with keepdims=True (preserves tensor structure)
        mean_square = ops.mean(
            ops.square(inputs),
            axis=self.axis,
            keepdims=True
        )

        rms = ops.maximum(
            ops.sqrt(mean_square + self.epsilon),
            self.epsilon
        )

        # Step 2: Normalize by RMS
        normalized = inputs / rms

        # Step 3: Extract RMS statistics for adaptive scaling
        # Aggregate RMS to [batch, 1] for the dense layer
        if isinstance(self.axis, (list, tuple)):
            # Multiple axes - take mean across all non-batch dimensions
            rms_stats = ops.mean(rms, axis=list(range(1, len(rms.shape))), keepdims=False)
        else:
            # Single axis - take mean across all dimensions except batch
            axes_to_reduce = [i for i in range(1, len(rms.shape))]
            if axes_to_reduce:
                rms_stats = ops.mean(rms, axis=axes_to_reduce, keepdims=False)
            else:
                rms_stats = ops.reshape(rms, [-1])

        # Ensure shape is [batch, 1] for dense layer
        rms_stats = ops.reshape(rms_stats, [-1, 1])

        return rms_stats, normalized

    def _apply_adaptive_scaling(
            self,
            normalized: keras.KerasTensor,
            band_logits: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Apply adaptive scaling with proper broadcasting.

        Args:
            normalized: RMS-normalized tensor.
            band_logits: Output from dense layer with shape [batch, 1].

        Returns:
            Adaptively scaled tensor with same shape as input.
        """
        # Convert logits to scaling factors in [1-α, 1] range
        # Use strong sigmoid scaling for better gradient flow
        band_activation = ops.sigmoid(5.0 * band_logits)
        scale = (1.0 - self.max_band_width) + (self.max_band_width * band_activation)

        # Reshape scaling factors for proper broadcasting
        input_ndim = len(normalized.shape)

        if input_ndim == 2:
            # 2D case: [batch, features] - scale broadcasts naturally from [batch, 1]
            scale = ops.broadcast_to(scale, normalized.shape)
        elif input_ndim == 4:
            # 4D case: need to reshape for proper broadcasting
            if isinstance(self.axis, int) and (self.axis == -1 or self.axis == 3):
                # Channel-wise: scale shape [batch, 1] -> [batch, 1, 1, 1]
                scale = ops.reshape(scale, [-1, 1, 1, 1])
            else:
                # Other cases: reshape for broadcasting
                scale = ops.reshape(scale, [-1, 1, 1, 1])
        else:
            # General case: reshape for broadcasting across all non-batch dimensions
            new_shape = [-1] + [1] * (input_ndim - 1)
            scale = ops.reshape(scale, new_shape)

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

        # Step 1: Compute RMS normalization and extract statistics
        rms_stats, normalized = self._compute_rms_statistics(inputs_fp32)

        # Step 2: Apply logarithmic transformation for variance stabilization
        # Since RMS >= epsilon > 0, log(RMS) is always well-defined
        # This stabilizes variance and handles long-tailed RMS distributions
        log_rms = ops.log(ops.maximum(rms_stats, self.epsilon))

        # Step 3: Dense projection to compute adaptive band parameters
        # The dense layer processes log-transformed RMS statistics
        band_logits = self.band_dense(log_rms, training=training)

        # Step 4: Apply adaptive scaling within [1-α, 1] band
        output = self._apply_adaptive_scaling(normalized, band_logits)

        # Cast back to original dtype
        return ops.cast(output, inputs.dtype)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
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