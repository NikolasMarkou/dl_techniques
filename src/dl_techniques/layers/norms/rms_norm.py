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

References:
    - Zhang, B., & Sennrich, R. (2019). "Root Mean Square Layer Normalization." 
      Advances in Neural Information Processing Systems, 32.
      https://arxiv.org/abs/1910.07467

    - Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021). 
      "RoFormer: Enhanced Transformer with Rotary Position Embedding."
      https://arxiv.org/abs/2104.09864 (Uses RMSNorm in practice)

    - Touvron, H., et al. (2023). "LLaMA: Open and Efficient Foundation Language Models."
      https://arxiv.org/abs/2302.13971 (Uses RMSNorm in LLaMA architecture)

Usage Examples:
    Basic usage:
    >>> inputs = keras.Input(shape=(512,))
    >>> normalized = RMSNorm()(inputs)

    With custom parameters:
    >>> rms_norm = RMSNorm(
    ...     axis=-1,           # Normalize over last dimension
    ...     epsilon=1e-6,      # Numerical stability
    ...     use_scale=True     # Use learnable scaling
    ... )
    >>> outputs = rms_norm(inputs)

    In a transformer block:
    >>> def transformer_block(x):
    ...     # Pre-normalization pattern common in modern transformers
    ...     norm_x = RMSNorm()(x)
    ...     attn_out = MultiHeadAttention(...)(norm_x, norm_x)
    ...     x = x + attn_out  # Residual connection
    ...     
    ...     norm_x = RMSNorm()(x)
    ...     ffn_out = FeedForward(...)(norm_x)
    ...     return x + ffn_out  # Residual connection
"""

import keras
from keras import ops
from typing import Optional, Any, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class RMSNorm(keras.layers.Layer):
    """
    Root Mean Square Normalization layer.

    This layer implements root mean square normalization by normalizing inputs by their
    RMS (Root Mean Square) value. RMS normalization can help stabilize training and 
    improve gradient flow in deep networks.

    The normalization is computed as:
        rms = sqrt(mean(input²) + epsilon)
        output = input / rms * scale

    Where scale is a learnable parameter when use_scale=True.

    Args:
        axis: int or tuple of ints, default=-1
            Axis or axes along which to compute RMS statistics. The default (-1)
            computes RMS over the last dimension.
        epsilon: float, default=1e-6
            Small constant added to denominator for numerical stability.
        use_scale: bool, default=True
            Whether to use a learnable scaling parameter after normalization.
        scale_initializer: str or initializer, default="ones"
            Initializer for the scale parameter when use_scale=True.
        **kwargs: Additional keyword arguments for the Layer parent class.

    Input shape:
        N-D tensor with shape: (batch_size, ..., features)

    Output shape:
        Same shape as input.
    """

    def __init__(
            self,
            axis: int = -1,
            epsilon: float = 1e-6,
            use_scale: bool = True,
            scale_initializer: str = "ones",
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self._validate_inputs(epsilon)

        self.axis = axis
        self.epsilon = epsilon
        self.use_scale = use_scale
        self.scale_initializer = keras.initializers.get(scale_initializer)

        # Will be set in build()
        self.scale = None
        self._build_input_shape = None

    def _validate_inputs(self, epsilon: float) -> None:
        """Validate initialization parameters."""
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

    def build(self, input_shape) -> None:
        """Build the layer weights."""
        self._build_input_shape = input_shape

        if self.use_scale:
            # Determine the shape for the scale parameter
            if isinstance(self.axis, int):
                axes = [self.axis]
            else:
                axes = list(self.axis)

            # Convert negative axis to positive
            axes = [ax if ax >= 0 else len(input_shape) + ax for ax in axes]

            # Create scale parameter with shape matching the input except for normalized axes
            param_shape = []
            for i, dim_size in enumerate(input_shape):
                if i in axes:
                    param_shape.append(1)
                else:
                    param_shape.append(dim_size)

            # Remove batch dimension if present (None)
            if param_shape[0] is None:
                param_shape = param_shape[1:]

            self.scale = self.add_weight(
                name="scale",
                shape=param_shape,
                initializer=self.scale_initializer,
                trainable=True,
            )

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply RMS normalization.

        Args:
            inputs: Input tensor
            training: Whether in training mode (unused but kept for consistency)

        Returns:
            RMS normalized tensor
        """
        # Cast to float32 for numerical stability in mixed precision training
        inputs_fp32 = ops.cast(inputs, "float32")

        # Compute RMS: sqrt(mean(x²))
        mean_square = ops.mean(
            ops.square(inputs_fp32),
            axis=self.axis,
            keepdims=True
        )
        rms = ops.sqrt(mean_square + self.epsilon)

        # Normalize
        normalized = inputs_fp32 / rms

        # Apply learnable scale if enabled
        if self.use_scale:
            normalized = normalized * self.scale

        # Cast back to original dtype
        return ops.cast(normalized, inputs.dtype)

    def compute_output_shape(self, input_shape) -> tuple:
        """Compute output shape (same as input shape)."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            "axis": self.axis,
            "epsilon": self.epsilon,
            "use_scale": self.use_scale,
            "scale_initializer": keras.initializers.serialize(self.scale_initializer),
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
