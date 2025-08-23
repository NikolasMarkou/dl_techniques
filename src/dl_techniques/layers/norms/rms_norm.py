"""
Root Mean Square Normalization Layer for Deep Neural Networks

This module implements RMS (Root Mean Square) Normalization, a normalization technique
that can help stabilize training and improve gradient flow in deep neural networks.
RMS normalization is particularly effective in transformer architectures and has been
shown to provide computational benefits over LayerNorm in certain scenarios.

Mathematical Formulation:
    Given an input tensor x with shape (..., d), RMS normalization computes:

    RMS(x) = sqrt(mean(x²) + ε)
    output = (x / RMS(x)) * γ

    Where:
    - mean(x²) is computed over specified axes (typically the feature dimension)
    - ε is a small epsilon for numerical stability
    - γ is an optional learnable scaling parameter

Key Differences from LayerNorm:
    - LayerNorm: (x - μ) / σ * γ + β (centers and scales)
    - RMSNorm: x / RMS(x) * γ (only scales, no centering)

This makes RMSNorm computationally more efficient as it doesn't require computing
the mean for centering, only the RMS for scaling.

Performance Benefits:
    - Reduced computational overhead (no mean subtraction)
    - Better gradient flow in some architectures
    - Maintains similar normalization benefits to LayerNorm
    - More stable in mixed precision training
    - Approximately 10-15% faster than LayerNorm in practice

References:
    - Zhang, B., & Sennrich, R. (2019). "Root Mean Square Layer Normalization."
      Advances in Neural Information Processing Systems, 32.
      https://arxiv.org/abs/1910.07467
"""

import keras
from keras import ops
from typing import Optional, Any, Dict, Union, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class RMSNorm(keras.layers.Layer):
    """
    Root Mean Square Normalization layer for stabilized training in deep networks.

    This layer implements root mean square normalization by normalizing inputs by their
    RMS (Root Mean Square) value. RMS normalization can help stabilize training and
    improve gradient flow in deep networks, particularly in transformer architectures.

    The normalization is computed as:

    .. math::
        \\text{rms} = \\sqrt{\\text{mean}(\\text{input}^2) + \\varepsilon}

    .. math::
        \\text{output} = \\frac{\\text{input}}{\\text{rms}} \\times \\text{scale}

    Where scale is a learnable parameter when ``use_scale=True``.

    Args:
        axis: Axis or axes along which to compute RMS statistics.
            The default (-1) computes RMS over the last dimension. For multi-axis normalization,
            pass a tuple (e.g., (-2, -1) for normalizing over last two dimensions).
            Defaults to -1.
        epsilon: Small constant added to denominator for numerical stability.
            Should be positive and typically in range [1e-8, 1e-5]. Defaults to 1e-6.
        use_scale: Whether to use a learnable scaling parameter after
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

    Raises:
        ValueError: If epsilon is not positive.
        ValueError: If attempting to normalize along dynamic axes during build.

    Example:
        Basic usage with default parameters:

        .. code-block:: python

            import keras
            from dl_techniques.layers.norms.rms_norm import RMSNorm

            # Simple case - normalize last dimension
            inputs = keras.Input(shape=(512,))
            normalized = RMSNorm()(inputs)
            model = keras.Model(inputs, normalized)

        Custom configuration for transformer layers:

        .. code-block:: python

            # Pre-normalization transformer block
            def transformer_block(inputs, hidden_size=768):
                # RMSNorm before attention
                norm_inputs = RMSNorm(
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

                # RMSNorm before FFN
                norm_x = RMSNorm()(x)

                # Feed-forward network
                ffn_out = keras.layers.Dense(3072, activation='relu')(norm_x)
                ffn_out = keras.layers.Dense(hidden_size)(ffn_out)

                # Residual connection
                return x + ffn_out

        Multi-axis normalization:

        .. code-block:: python

            # Normalize over spatial dimensions for images
            inputs = keras.Input(shape=(32, 32, 256))  # (H, W, C)

            # Normalize over height and width, keep channels separate
            spatial_norm = RMSNorm(axis=(1, 2))(inputs)

            # Normalize over all feature dimensions
            full_norm = RMSNorm(axis=(-3, -2, -1))(inputs)

        Integration in large language models:

        .. code-block:: python

            # LLaMA-style architecture with RMSNorm
            class LLaMABlock(keras.layers.Layer):
                def __init__(self, hidden_size=4096, **kwargs):
                    super().__init__(**kwargs)
                    self.attention_norm = RMSNorm(epsilon=1e-5)
                    self.ffn_norm = RMSNorm(epsilon=1e-5)
                    self.attention = keras.layers.MultiHeadAttention(...)
                    self.feed_forward = keras.layers.Dense(...)

                def call(self, inputs):
                    # Pre-normalization pattern
                    norm_inputs = self.attention_norm(inputs)
                    attn_out = self.attention(norm_inputs, norm_inputs)
                    x = inputs + attn_out

                    norm_x = self.ffn_norm(x)
                    ffn_out = self.feed_forward(norm_x)
                    return x + ffn_out

        Mixed precision training optimization:

        .. code-block:: python

            # RMSNorm is particularly stable in mixed precision
            keras.mixed_precision.set_global_policy('mixed_float16')

            inputs = keras.Input(shape=(1024,), dtype='float16')
            normalized = RMSNorm(
                epsilon=1e-5,  # Slightly larger epsilon for fp16 stability
                use_scale=True
            )(inputs)

            model = keras.Model(inputs, normalized)

    Note:
        - RMSNorm is approximately 10-15% faster than LayerNorm due to avoiding mean computation
        - Particularly effective in transformer architectures and large language models
        - The implementation automatically handles mixed precision training with appropriate casting
        - Scale parameter shape is automatically inferred from normalization axes
        - This implementation follows modern Keras 3 patterns for robust serialization
    """

    def __init__(
        self,
        axis: Union[int, Tuple[int, ...]] = -1,
        epsilon: float = 1e-6,
        use_scale: bool = True,
        scale_initializer: Union[str, keras.initializers.Initializer] = "ones",
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs early
        self._validate_inputs(axis, epsilon)

        # Store ALL configuration parameters - required for get_config()
        self.axis = axis
        self.epsilon = epsilon
        self.use_scale = use_scale
        self.scale_initializer = keras.initializers.get(scale_initializer)

        # Initialize weight attributes - created in build()
        self.scale = None

        logger.debug(f"Initialized RMSNorm with axis={axis}, epsilon={epsilon}, use_scale={use_scale}")

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
                    f"Cannot create 'scale' parameter for RMSNorm. The "
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
        Apply RMS normalization to inputs.

        Args:
            inputs: Input tensor of any shape. Normalization is applied along
                the axes specified during initialization.
            training: Boolean indicating whether the layer should behave in training mode.
                Not used in RMSNorm but kept for consistency with other normalization layers.

        Returns:
            RMS normalized tensor with the same shape as inputs.

        Note:
            The computation is performed in float32 for numerical stability in mixed
            precision training, then cast back to the original input dtype.
        """
        # Store original dtype for casting back
        original_dtype = inputs.dtype

        # Cast to float32 for numerical stability in mixed precision training
        inputs_fp32 = ops.cast(inputs, "float32")

        # Compute RMS: sqrt(mean(x²) + ε)
        mean_square = ops.mean(
            ops.square(inputs_fp32),
            axis=self.axis,
            keepdims=True
        )

        # Add epsilon for numerical stability and compute RMS
        rms = ops.sqrt(mean_square + self.epsilon)

        # Normalize by RMS
        normalized = inputs_fp32 / rms

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
            "scale_initializer": keras.initializers.serialize(self.scale_initializer),
        })
        return config

# ---------------------------------------------------------------------
