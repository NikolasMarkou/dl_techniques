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
    variance and handle the long-tailed nature of magnitude distributions.

3.  **Dense Projection:**
    The log-transformed RMS statistics are passed through a dense layer to compute
    adaptive scaling parameters that depend on the input's magnitude characteristics.

4.  **Adaptive Band Scaling:**
    The scaling parameters are constrained to [1 - max_band_width, 1] using sigmoid
    activation and applied to the RMS-normalized features.

References:
[1] Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)
    https://arxiv.org/abs/1910.07467
"""

import keras
from keras import ops
from typing import Any, Dict, Optional, Union, Tuple, List

from dl_techniques.utils.logger import logger


@keras.saving.register_keras_serializable()
class AdaptiveBandRMS(keras.layers.Layer):
    """
    Adaptive Root Mean Square Normalization with log-transformed RMS scaling.

    This layer implements advanced RMS normalization where scaling factors are computed
    from logarithmically transformed RMS statistics. The log transformation stabilizes
    variance and creates more symmetric distributions for adaptive scaling computation.

    **Intent**: Provide input-adaptive normalization that creates "thick spherical shells"
    in RMS space, where shell thickness adapts based on input magnitude characteristics
    with improved numerical stability through log-transformed statistics.

    **Architecture**:
    ```
    Input(shape=[..., features])
           ↓
    RMS Statistics: rms = sqrt(mean(x²) + ε)
           ↓
    RMS Normalization: x_norm = x / rms
           ↓
    Log Transform: log_rms = log(aggregate(rms))
           ↓
    Dense Projection: band_params = Dense(log_rms)
           ↓
    Sigmoid Scaling: scale = (1-α) + α*σ(band_params)
           ↓
    Adaptive Scaling: output = x_norm * scale
           ↓
    Output(shape=[..., features])
    ```

    **Mathematical Operations**:
    1. **RMS computation**: rms = √(E[x²] + ε)
    2. **Normalization**: x̂ = x / rms
    3. **Log transform**: log_rms = log(rms_aggregate)
    4. **Adaptive scaling**: y = x̂ * scale(log_rms)

    Where scale(·) ∈ [1-α, 1] is computed via dense projection and sigmoid activation.

    This robust version supports arbitrary tensor shapes (2D, 3D, 4D, etc.) with
    flexible axis configurations, making it suitable for dense layers, sequence models,
    convolutions, and any tensor processing scenarios.

    Args:
        max_band_width: Float between 0 and 1, maximum allowed deviation from unit
            normalization. Controls the thickness of the adaptive spherical shell.
            Smaller values create tighter constraints. Defaults to 0.1.
        axis: Integer or tuple of integers specifying axes along which to compute
            RMS statistics. Similar to keras.layers.LayerNormalization:
            - For 2D (batch, features): axis=-1 (feature-wise)
            - For 3D (batch, seq, features): axis=-1 or axis=1
            - For 4D (batch, H, W, channels): axis=-1 or axis=(1,2)
            Defaults to -1.
        epsilon: Float, small positive constant added to denominator for numerical
            stability. Should be small but not too small to avoid underflow.
            Defaults to 1e-7.
        band_initializer: Initializer for the dense layer computing scaling parameters.
            String name or Initializer instance. Defaults to 'zeros' for stable
            initialization near unit scaling.
        band_regularizer: Optional regularizer for the dense layer weights.
            Can help prevent overfitting of adaptive scaling. Defaults to None.
        **kwargs: Additional arguments for Layer base class (name, trainable, etc.).

    Input shape:
        Arbitrary N-D tensor with shape: `(batch_size, ...)`.
        The normalization axes must have defined dimensions (not None).

    Output shape:
        Same shape as input: `(batch_size, ...)`.
        Values are RMS-normalized with adaptive scaling applied.

    Attributes:
        dense_layer: Dense layer for computing scaling parameters from log-RMS stats.
        Created during build() since output size depends on input shape and axes.

    Example:
        ```python
        # Dense layer usage (2D tensors)
        inputs = keras.Input(shape=(128,))
        x = keras.layers.Dense(64)(inputs)
        x = AdaptiveBandRMS(max_band_width=0.2)(x)

        # Sequence model usage (3D tensors)
        inputs = keras.Input(shape=(100, 256))  # seq_len=100, features=256
        x = AdaptiveBandRMS(axis=-1)(inputs)    # Feature-wise normalization
        x = AdaptiveBandRMS(axis=1)(inputs)     # Sequence-wise normalization

        # CNN usage (4D tensors)
        inputs = keras.Input(shape=(32, 32, 64))  # H=32, W=32, channels=64
        x = AdaptiveBandRMS(axis=-1)(inputs)      # Channel-wise normalization
        x = AdaptiveBandRMS(axis=(1, 2))(inputs) # Spatial normalization

        # Custom configuration with regularization
        norm_layer = AdaptiveBandRMS(
            max_band_width=0.3,
            axis=(1, 2),
            band_regularizer=keras.regularizers.L2(1e-4)
        )
        outputs = norm_layer(inputs)
        ```

    Raises:
        ValueError: If max_band_width is not between 0 and 1.
        ValueError: If epsilon is not positive.
        ValueError: If axis configuration includes batch dimension (axis 0).
        ValueError: If axis is out of bounds for input tensor rank.

    Note:
        This layer uses dynamic sub-layer creation in build() rather than __init__()
        because the dense layer's output size depends on the input shape and
        normalization axes. This is necessary for proper parameter shape computation.
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

        # Validate inputs early
        self._validate_inputs(max_band_width, epsilon)

        # Store ALL configuration - required for get_config()
        self.max_band_width = max_band_width
        self.axis = axis
        self.epsilon = epsilon
        self.band_initializer = keras.initializers.get(band_initializer)
        self.band_regularizer = keras.regularizers.get(band_regularizer)

        # Shape computation results (set in build)
        self._param_shape = None
        self._scaling_axes = None

        # Sub-layer created in build() due to dynamic architecture
        # Note: Cannot follow standard pattern due to input-shape dependency
        self.dense_layer = None

        logger.debug(
            f"Initialized AdaptiveBandRMS: max_band_width={max_band_width}, "
            f"axis={axis}, epsilon={epsilon}"
        )

    def _validate_inputs(self, max_band_width: float, epsilon: float) -> None:
        """
        Validate initialization parameters.

        Args:
            max_band_width: Maximum deviation from unit normalization.
            epsilon: Numerical stability constant.

        Raises:
            ValueError: If parameters are invalid.
        """
        if not 0 < max_band_width < 1:
            raise ValueError(
                f"max_band_width must be between 0 and 1, got {max_band_width}"
            )
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

    def _compute_param_shape_and_axes(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Tuple[int, ...], List[int]]:
        """
        Compute parameter shape and scaling axes based on input shape and normalization axes.

        Determines what dimensions should have independent scaling parameters,
        similar to how LayerNormalization computes parameter shapes.

        Args:
            input_shape: Shape of input tensor including batch dimension.

        Returns:
            Tuple of (param_shape, scaling_axes) where:
            - param_shape: Shape for scaling parameters (without batch dim)
            - scaling_axes: List of axes that will be scaled independently

        Raises:
            ValueError: If axis configuration is invalid.
        """
        input_rank = len(input_shape)

        # Normalize axis to positive integers
        if isinstance(self.axis, int):
            axes = [self.axis]
        else:
            axes = list(self.axis)

        # Convert negative indices and validate
        normalized_axes = []
        for ax in axes:
            if ax < 0:
                ax = input_rank + ax
            if ax == 0:
                raise ValueError(
                    "axis 0 (batch dimension) cannot be normalized"
                )
            if ax < 0 or ax >= input_rank:
                raise ValueError(
                    f"axis {ax} is out of bounds for input with {input_rank} dimensions"
                )
            normalized_axes.append(ax)

        # Remove duplicates and sort
        normalized_axes = sorted(set(normalized_axes))

        # Check for global normalization (all non-batch axes)
        is_global = (
            input_rank > 2 and
            len(normalized_axes) == input_rank - 1
        )

        if is_global:
            # Global normalization: single parameter broadcasts everywhere
            param_shape = [1] * (input_rank - 1)
            scaling_axes = []
            return tuple(param_shape), scaling_axes

        # Compute parameter shape for non-global cases
        param_shape = []
        scaling_axes = []

        for i in range(1, input_rank):  # Skip batch dimension
            if i in normalized_axes:
                param_shape.append(input_shape[i])
                scaling_axes.append(i)
            else:
                param_shape.append(1)

        return tuple(param_shape), scaling_axes

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and create sub-layers with proper parameter sizing.

        Note: This layer uses dynamic sub-layer creation because the dense layer's
        output size depends on input shape and normalization axes. This deviates
        from the standard Keras pattern where sub-layers are created in __init__(),
        but is necessary for proper parameter shape computation.
        """
        # Compute parameter configuration
        self._param_shape, self._scaling_axes = self._compute_param_shape_and_axes(
            input_shape
        )

        # Calculate number of scaling parameters needed
        num_params = 1
        for dim in self._param_shape:
            if dim is not None:
                num_params *= dim

        # Create dense layer for log-RMS to scaling-parameter projection
        self.dense_layer = keras.layers.Dense(
            units=num_params,
            kernel_initializer=self.band_initializer,
            kernel_regularizer=self.band_regularizer,
            use_bias=True,
            name="band_dense"
        )

        # Build the dense layer explicitly
        # Input is always [batch, 1] for aggregated log-RMS statistics
        log_rms_input_shape = (None, 1)
        self.dense_layer.build(log_rms_input_shape)

        logger.debug(
            f"Built AdaptiveBandRMS: param_shape={self._param_shape}, "
            f"num_params={num_params}, dense_input_shape={log_rms_input_shape}"
        )

        # Always call parent build at the end
        super().build(input_shape)

    def _aggregate_rms_statistics(
        self,
        rms_tensor: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Aggregate RMS statistics to single scalar per batch element.

        For the dense layer processing, we need one representative RMS value
        per batch element. This is computed by averaging all RMS values
        across non-batch dimensions.

        Args:
            rms_tensor: RMS tensor with keepdims=True, same rank as input.

        Returns:
            Aggregated RMS statistics with shape [batch, 1].
        """
        # Aggregate over all non-batch dimensions
        aggregation_axes = list(range(1, len(rms_tensor.shape)))

        if aggregation_axes:
            rms_stats = ops.mean(rms_tensor, axis=aggregation_axes, keepdims=True)
        else:
            # Handle edge case of 1D input (batch,)
            rms_stats = rms_tensor

        # Ensure shape is [batch, 1] for dense layer
        return ops.reshape(rms_stats, [-1, 1])

    def _reshape_scaling_factors(
        self,
        scaling_factors: keras.KerasTensor,
        input_shape: Tuple[Optional[int], ...]
    ) -> keras.KerasTensor:
        """
        Reshape scaling factors from dense layer for proper broadcasting.

        Args:
            scaling_factors: Output from dense layer, shape [batch, num_params].
            input_shape: Original input tensor shape.

        Returns:
            Reshaped scaling factors ready for element-wise multiplication.
        """
        batch_size = ops.shape(scaling_factors)[0]
        target_shape = [batch_size] + list(self._param_shape)
        return ops.reshape(scaling_factors, target_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply adaptive RMS normalization with log-transformed statistics.

        Args:
            inputs: Input tensor of arbitrary shape.
            training: Boolean indicating training mode.

        Returns:
            Normalized tensor with adaptive RMS-based scaling.
        """
        # Cast to float32 for numerical stability in mixed precision
        inputs_fp32 = ops.cast(inputs, "float32")

        # Step 1: Compute RMS for normalization
        mean_square = ops.mean(
            ops.square(inputs_fp32),
            axis=self.axis,
            keepdims=True
        )

        rms = ops.maximum(
            ops.sqrt(mean_square + self.epsilon),
            self.epsilon
        )

        # Apply standard RMS normalization
        normalized = inputs_fp32 / rms

        # Step 2: Aggregate RMS statistics for dense layer input
        rms_stats = self._aggregate_rms_statistics(rms)

        # Step 3: Logarithmic transformation for variance stabilization
        log_rms = ops.log(rms_stats)

        # Step 4: Dense projection to compute adaptive scaling parameters
        band_logits = self.dense_layer(log_rms, training=training)

        # Step 5: Convert to scaling factors in [1-α, 1] range
        band_activation = ops.sigmoid(5.0 * band_logits)
        scale_factors = (1.0 - self.max_band_width) + (
            self.max_band_width * band_activation
        )

        # Step 6: Reshape for broadcasting and apply adaptive scaling
        scale_factors = self._reshape_scaling_factors(scale_factors, inputs.shape)
        output = normalized * ops.cast(scale_factors, "float32")

        # Cast back to original dtype
        return ops.cast(output, inputs.dtype)

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape (same as input)."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration for serialization.

        Returns all constructor parameters needed for layer reconstruction.
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