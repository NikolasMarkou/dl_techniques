"""
Zero-Centered Root Mean Square Normalization Layer for Deep Neural Networks

This module implements Zero-Centered RMS (Root Mean Square) Normalization, an advanced
normalization technique that combines the computational efficiency of RMSNorm with the
stabilizing zero-mean property of LayerNorm. This variant addresses the "mean shift"
problem in standard RMSNorm while maintaining computational advantages.

Mathematical Formulation:
    Given an input tensor x with shape (..., d), Zero-Centered RMS normalization computes:

    μ = mean(x) over specified axes
    x_centered = x - μ
    RMS(x_centered) = sqrt(mean(x_centered²) + ε)
    output = (x_centered / RMS(x_centered)) * γ

    Where:
    - μ is the mean computed over specified axes (centering step)
    - mean(x_centered²) is computed over the same specified axes
    - ε is a small epsilon for numerical stability
    - γ is an optional learnable scaling parameter

Key Differences from Standard Normalization:
    - LayerNorm: (x - μ) / σ * γ + β (centers, scales, and shifts)
    - RMSNorm: x / RMS(x) * γ (only scales, no centering)
    - Zero-Centered RMSNorm: (x - μ) / RMS(x - μ) * γ (centers and scales, no shift)

This makes Zero-Centered RMSNorm conceptually similar to LayerNorm without the bias term,
but framed as an enhancement to RMSNorm that prevents mean drift.

Performance Benefits:
    - Prevents abnormal growth of layer normalization weights
    - Maintains training stability through zero-mean outputs
    - Combines efficiency with stabilization
    - Better gradient flow compared to standard RMSNorm
    - Particularly effective in large language models like Qwen3-Next

References:
    - Used in Qwen3-Next model for solving abnormal growth issues in layer normalization weights
    - Builds upon concepts from both LayerNorm and RMSNorm literature
"""

# Core Keras imports - always use full paths
import keras
from keras import ops, initializers
from typing import Optional, Union, Tuple, Dict, Any

# DL-Techniques framework imports
from dl_techniques.utils.logger import logger


@keras.saving.register_keras_serializable()
class ZeroCenteredRMSNorm(keras.layers.Layer):
    """
    Zero-Centered Root Mean Square Normalization layer for enhanced training stability.

    This layer implements zero-centered root mean square normalization by first centering
    the inputs around zero, then normalizing by their RMS value. This approach combines
    the computational efficiency of RMSNorm with the stabilizing zero-mean property of
    LayerNorm, preventing mean drift and abnormal weight growth.

    **Intent**: Provide enhanced normalization that prevents abnormal growth of layer
    normalization weights while maintaining computational efficiency, particularly
    beneficial for transformer architectures and large language models.

    **Architecture**:
    ```
    Input(shape=[..., features])
           ↓
    Compute: μ = mean(x)
           ↓
    Center: x_centered = x - μ
           ↓
    Compute: rms = sqrt(mean(x_centered²) + ε)
           ↓
    Normalize: x_norm = x_centered / rms
           ↓
    Scale: output = x_norm * γ (if use_scale=True)
           ↓
    Output(shape=[..., features])
    ```

    **Mathematical Operations**:
    1. **Centering**: μ = E[x], x_centered = x - μ
    2. **RMS Computation**: rms = √(E[x_centered²] + ε)
    3. **Normalization**: x̂ = x_centered / rms
    4. **Scaling**: y = γ ⊙ x̂ (if use_scale=True)

    Where:
    - μ is computed per feature across normalization axes
    - γ (scale) is a learnable parameter if use_scale=True
    - ε is a small constant for numerical stability
    - ⊙ denotes element-wise multiplication

    Args:
        axis: Axis or axes along which to compute mean and RMS statistics.
            The default (-1) computes statistics over the last dimension. For multi-axis
            normalization, pass a tuple (e.g., (-2, -1) for normalizing over last two
            dimensions). Defaults to -1.
        epsilon: Small constant added to denominator for numerical stability.
            Should be positive and typically in range [1e-8, 1e-5]. Defaults to 1e-6.
        use_scale: Boolean, whether to use a learnable scaling parameter after
            normalization. When True, adds a trainable parameter that can help the model
            learn appropriate scaling. Defaults to True.
        scale_initializer: Initializer for the scale parameter when ``use_scale=True``.
            Common choices include "ones" (default), "zeros", or custom initializers.
            Defaults to "ones".
        **kwargs: Additional keyword arguments passed to the parent Layer class.

    Input shape:
        N-D tensor with shape: ``(batch_size, ..., features)``

        The layer can handle any dimensionality, with normalization applied along
        the specified axis/axes.

    Output shape:
        Same shape as input: ``(batch_size, ..., features)``

    Attributes:
        scale: Learnable scale parameter if use_scale=True, else None. Created in build().

    Raises:
        ValueError: If epsilon is not positive.
        ValueError: If attempting to normalize along dynamic axes during build.
        TypeError: If axis is not int or tuple of ints.

    Example:
        Basic usage with default parameters:

        .. code-block:: python

            import keras
            from dl_techniques.layers.norms.zero_centered_rms_norm import ZeroCenteredRMSNorm

            # Simple case - normalize last dimension
            inputs = keras.Input(shape=(512,))
            normalized = ZeroCenteredRMSNorm()(inputs)
            model = keras.Model(inputs, normalized)

        Custom configuration for transformer layers:

        .. code-block:: python

            # Pre-normalization transformer block with enhanced stability
            def stable_transformer_block(inputs, hidden_size=768):
                # Zero-centered RMSNorm before attention
                norm_inputs = ZeroCenteredRMSNorm(
                    axis=-1,
                    epsilon=1e-5,
                    use_scale=True,
                    scale_initializer='ones'
                )(inputs)

                # Multi-head attention
                attention_out = keras.layers.MultiHeadAttention(
                    num_heads=12,
                    key_dim=64
                )(norm_inputs, norm_inputs)

                # Residual connection
                x = inputs + attention_out

                # Zero-centered RMSNorm before FFN
                norm_x = ZeroCenteredRMSNorm()(x)

                # Feed-forward network
                ffn_out = keras.layers.Dense(3072, activation='relu')(norm_x)
                ffn_out = keras.layers.Dense(hidden_size)(ffn_out)

                # Residual connection
                return x + ffn_out

        Large language model integration (Qwen3-Next style):

        .. code-block:: python

            class QwenBlock(keras.layers.Layer):
                def __init__(self, hidden_size=4096, **kwargs):
                    super().__init__(**kwargs)
                    # Use Zero-Centered RMSNorm for stability
                    self.attention_norm = ZeroCenteredRMSNorm(epsilon=1e-5)
                    self.ffn_norm = ZeroCenteredRMSNorm(epsilon=1e-5)
                    self.attention = keras.layers.MultiHeadAttention(...)
                    self.feed_forward = keras.layers.Dense(...)

                def call(self, inputs):
                    # Pre-normalization with zero-centering
                    norm_inputs = self.attention_norm(inputs)
                    attn_out = self.attention(norm_inputs, norm_inputs)
                    x = inputs + attn_out

                    norm_x = self.ffn_norm(x)
                    ffn_out = self.feed_forward(norm_x)
                    return x + ffn_out

        Multi-axis normalization:

        .. code-block:: python

            # Normalize over spatial dimensions for images
            inputs = keras.Input(shape=(32, 32, 256))  # (H, W, C)

            # Normalize over height and width, keep channels separate
            spatial_norm = ZeroCenteredRMSNorm(axis=(1, 2))(inputs)

            # Normalize over all feature dimensions
            full_norm = ZeroCenteredRMSNorm(axis=(-3, -2, -1))(inputs)

        Mixed precision training optimization:

        .. code-block:: python

            # Zero-Centered RMSNorm with mixed precision
            keras.mixed_precision.set_global_policy('mixed_float16')

            inputs = keras.Input(shape=(1024,), dtype='float16')
            normalized = ZeroCenteredRMSNorm(
                epsilon=1e-5,  # Slightly larger epsilon for fp16 stability
                use_scale=True
            )(inputs)

            model = keras.Model(inputs, normalized)

    Note:
        - Prevents abnormal growth of layer normalization weights compared to standard RMSNorm
        - Maintains zero-mean property across layers for enhanced training stability
        - Particularly effective in transformer architectures and large language models
        - The implementation automatically handles mixed precision training with appropriate casting
        - Scale parameter shape is automatically inferred from normalization axes
        - This implementation follows modern Keras 3 patterns for robust serialization
        - Conceptually similar to LayerNorm without bias, but framed as enhanced RMSNorm
    """

    def __init__(
            self,
            axis: Union[int, Tuple[int, ...]] = -1,
            epsilon: float = 1e-6,
            use_scale: bool = True,
            scale_initializer: Union[str, initializers.Initializer] = "ones",
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs early
        self._validate_inputs(axis, epsilon)

        # Store ALL configuration parameters - required for get_config()
        self.axis = axis
        self.epsilon = epsilon
        self.use_scale = use_scale
        self.scale_initializer = initializers.get(scale_initializer)

        # Initialize weight attributes - created in build()
        self.scale = None

        logger.debug(f"Initialized ZeroCenteredRMSNorm with axis={axis}, epsilon={epsilon}, use_scale={use_scale}")

    def _validate_inputs(self, axis: Union[int, Tuple[int, ...]], epsilon: float) -> None:
        """
        Validate initialization parameters.

        Args:
            axis: Normalization axis/axes to validate.
            epsilon: Epsilon value to validate.

        Raises:
            ValueError: If epsilon is not positive.
            TypeError: If axis is not int or tuple of ints.
        """
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        # Validate axis type
        if isinstance(axis, (list, tuple)):
            if not all(isinstance(ax, int) for ax in axis):
                raise TypeError(f"All elements in axis must be integers, got {axis}")
        elif not isinstance(axis, int):
            raise TypeError(f"axis must be int or tuple of ints, got {type(axis)}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create the layer's own weights.

        This is called automatically when the layer first processes input.
        Following modern Keras 3 Pattern 1: Simple Layer (No Sub-layers).

        Args:
            input_shape: Shape tuple indicating input tensor shape.
                First dimension (batch size) may be None.

        Raises:
            ValueError: If attempting to create scale parameter with dynamic shape
                along normalization axes.
        """
        if self.use_scale:
            # Determine the shape for the scale parameter
            # Scale parameter shape matches the input shape along normalization axes
            if isinstance(self.axis, int):
                param_axes = [self.axis]
            else:
                param_axes = list(self.axis)

            # Convert negative axes to positive for shape computation
            param_axes = [ax % len(input_shape) if ax < 0 else ax for ax in param_axes]

            param_shape = tuple(input_shape[i] for i in param_axes)

            # Check for dynamic dimensions along normalization axes
            if any(dim is None for dim in param_shape):
                raise ValueError(
                    f"Cannot create 'scale' parameter for ZeroCenteredRMSNorm. The "
                    f"normalization axis {self.axis} corresponds to dynamic "
                    f"dimensions in input_shape {input_shape}. "
                    f"Scale parameter shape would be {param_shape}."
                )

            # Create layer's own weights using add_weight()
            self.scale = self.add_weight(
                name="scale",
                shape=param_shape,
                initializer=self.scale_initializer,
                trainable=True,
            )

            logger.debug(f"Created scale parameter with shape {param_shape}")

        # Always call parent build at the end
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply Zero-Centered RMS normalization to inputs.

        Args:
            inputs: Input tensor of any shape. Normalization is applied along
                the axes specified during initialization.
            training: Boolean indicating whether the layer should behave in training mode.
                Not used in Zero-Centered RMSNorm but kept for consistency with other
                normalization layers.

        Returns:
            Zero-centered RMS normalized tensor with the same shape as inputs.

        Note:
            The computation is performed in float32 for numerical stability in mixed
            precision training, then cast back to the original input dtype.
        """
        # Store original dtype for casting back
        original_dtype = inputs.dtype

        # Cast to float32 for numerical stability in mixed precision training
        inputs_fp32 = ops.cast(inputs, "float32")

        # Step 1: Compute mean and center the input
        mean = ops.mean(
            inputs_fp32,
            axis=self.axis,
            keepdims=True
        )

        centered_inputs = inputs_fp32 - mean

        # Step 2: Compute RMS of centered inputs: sqrt(mean(x_centered²) + ε)
        mean_square = ops.mean(
            ops.square(centered_inputs),
            axis=self.axis,
            keepdims=True
        )

        # Add epsilon for numerical stability and compute RMS
        rms = ops.sqrt(mean_square + self.epsilon)

        # Step 3: Normalize by RMS
        normalized = centered_inputs / rms

        # Apply learnable scale if enabled
        if self.use_scale:
            normalized = normalized * self.scale

        # Cast back to original dtype
        return ops.cast(normalized, original_dtype)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape tuple of the input tensor.

        Returns:
            Output shape tuple (same as input shape for normalization layers).
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration for serialization.

        Following modern Keras 3 patterns, this method returns ALL constructor
        arguments needed to recreate this layer instance.

        Returns:
            Dictionary containing all constructor arguments.
        """
        config = super().get_config()
        config.update({
            "axis": self.axis,
            "epsilon": self.epsilon,
            "use_scale": self.use_scale,
            "scale_initializer": initializers.serialize(self.scale_initializer),
        })
        return config

# ---------------------------------------------------------------------