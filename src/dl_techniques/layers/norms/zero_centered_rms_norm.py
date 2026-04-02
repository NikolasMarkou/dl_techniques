"""
Zero-Centered Root Mean Square Normalization Layer for Deep Neural Networks

This module implements Zero-Centered RMS (Root Mean Square) Normalization, an advanced
normalization technique that combines the computational efficiency of RMSNorm with the
stabilizing zero-mean property of LayerNorm. This variant addresses the "mean shift"
problem in standard RMSNorm while maintaining computational advantages.

Mathematical Formulation:
    Given an input tensor x with shape (..., d), Zero-Centered RMS normalization computes:

    mu = mean(x) over specified axes
    x_centered = x - mu
    RMS(x_centered) = sqrt(mean(x_centered^2) + epsilon)
    output = (x_centered / RMS(x_centered)) * gamma

    Where:
    - mu is the mean computed over specified axes (centering step)
    - mean(x_centered^2) is computed over the same specified axes
    - epsilon is a small epsilon for numerical stability
    - gamma is an optional learnable scaling parameter

Key Differences from Standard Normalization:
    - LayerNorm: (x - mu) / sigma * gamma + beta (centers, scales, and shifts)
    - RMSNorm: x / RMS(x) * gamma (only scales, no centering)
    - Zero-Centered RMSNorm: (x - mu) / RMS(x - mu) * gamma (centers and scales, no shift)

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


import keras
from keras import ops, initializers
from typing import Optional, Union, Tuple, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ZeroCenteredRMSNorm(keras.layers.Layer):
    """
    Zero-Centered Root Mean Square Normalization layer for enhanced training stability.

    This layer implements zero-centered root mean square normalization by first centering
    the inputs around zero, then normalizing by their RMS value. This approach combines
    the computational efficiency of RMSNorm with the stabilizing zero-mean property of
    LayerNorm, preventing mean drift and abnormal weight growth.

    The normalization is computed as:

    1. Centering: mu = E[x], x_centered = x - mu
    2. RMS Computation: rms = sqrt(E[x_centered^2] + epsilon)
    3. Normalization: x_hat = x_centered / rms
    4. Scaling: y = gamma * x_hat (if use_scale=True)

    Where mu is computed per feature across normalization axes, gamma (scale) is a
    learnable parameter if use_scale=True, and epsilon is a small constant for
    numerical stability.

    This layer is particularly beneficial for transformer architectures and large
    language models, preventing abnormal growth of layer normalization weights while
    maintaining computational efficiency. Conceptually similar to LayerNorm without
    bias, but framed as enhanced RMSNorm.

    **Architecture Overview:**

    .. code-block:: text

        Input: x (batch, ..., features)
                │
                ▼
        ┌───────────────────────────────┐
        │  μ = mean(x) along axis       │
        └────────────┬──────────────────┘
                     │
                     ▼
        ┌───────────────────────────────┐
        │  x_centered = x - μ           │
        └────────────┬──────────────────┘
                     │
                     ▼
        ┌───────────────────────────────┐
        │  RMS = √(mean(x_centered²)+ε) │
        └────────────┬──────────────────┘
                     │
                     ▼
        ┌───────────────────────────────┐
        │  normalized = x_centered / RMS│
        └────────────┬──────────────────┘
                     │
                     ▼
        ┌───────────────────────────────┐
        │  output = normalized × γ      │
        │  (if use_scale=True)          │
        └────────────┬──────────────────┘
                     │
                     ▼
        Output: (batch, ..., features)

    :param axis: Axis or axes along which to compute mean and RMS statistics.
        The default (-1) computes statistics over the last dimension. For multi-axis
        normalization, pass a tuple (e.g., (-2, -1) for normalizing over last two
        dimensions).
    :type axis: Union[int, Tuple[int, ...]]
    :param epsilon: Small constant added to denominator for numerical stability.
        Should be positive and typically in range [1e-8, 1e-5].
    :type epsilon: float
    :param use_scale: Whether to use a learnable scaling parameter after
        normalization. When True, adds a trainable parameter that can help the model
        learn appropriate scaling.
    :type use_scale: bool
    :param scale_initializer: Initializer for the scale parameter when
        ``use_scale=True``. Common choices include "ones" (default), "zeros",
        or custom initializers.
    :type scale_initializer: Union[str, initializers.Initializer]
    :param kwargs: Additional keyword arguments passed to the parent Layer class.

    :raises ValueError: If epsilon is not positive.
    :raises ValueError: If attempting to normalize along dynamic axes during build.
    :raises TypeError: If axis is not int or tuple of ints.
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

        :param axis: Normalization axis/axes to validate.
        :type axis: Union[int, Tuple[int, ...]]
        :param epsilon: Epsilon value to validate.
        :type epsilon: float
        :raises ValueError: If epsilon is not positive.
        :raises TypeError: If axis is not int or tuple of ints.
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

        :param input_shape: Shape tuple indicating input tensor shape.
            First dimension (batch size) may be None.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If attempting to create scale parameter with dynamic
            shape along normalization axes.
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

        :param inputs: Input tensor of any shape. Normalization is applied along
            the axes specified during initialization.
        :type inputs: keras.KerasTensor
        :param training: Boolean indicating whether the layer should behave in
            training mode. Not used in Zero-Centered RMSNorm but kept for
            consistency with other normalization layers.
        :type training: Optional[bool]
        :return: Zero-centered RMS normalized tensor with the same shape as inputs.
        :rtype: keras.KerasTensor
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

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple (same as input shape for normalization layers).
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration for serialization.

        Following modern Keras 3 patterns, this method returns ALL constructor
        arguments needed to recreate this layer instance.

        :return: Dictionary containing all constructor arguments.
        :rtype: Dict[str, Any]
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
