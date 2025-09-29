"""Selectively mask gradients during backpropagation without altering the forward pass.

This layer provides a mechanism to decouple the forward computation from the
gradient computation, a powerful technique for implementing advanced machine
learning models. While the input signal passes through this layer unchanged
during the forward pass (i.e., it functions as an identity operation), its
true utility lies in its behavior during backpropagation. It uses a binary
mask to selectively block, or "stop," gradients from flowing backward through
specific elements of the signal tensor.

Architecture:
    The layer operates on two input tensors of identical shape: a `signal`
    tensor and a binary `mask` tensor. The core of its design is a dual-path
    computation that ensures the forward pass is an identity function while
    the backward pass is selectively masked:

    `output = stop_gradient(signal) * mask + signal * (1 - mask)`

    During the forward pass, this equation simplifies to `signal`, as the two
    terms are mutually exclusive and their sum reconstructs the original
    tensor. However, during backpropagation, the `stop_gradient` operator
    ensures that no gradients flow through the first term. Consequently, the
    gradient of the output with respect to the signal is simply `(1 - mask)`,
    effectively applying the inverse of the mask to the upstream gradients.

Foundational Mathematics:
    The mathematical foundation of this layer is the concept of defining a
    custom gradient for an operation. The `stop_gradient` function is a crucial
    component, acting as an identity function in the forward direction but
    having a derivative of zero in the backward direction.

    Let `x` be the signal and `m` be the mask. The output `y` is:
    `y = stop_gradient(x) * m + x * (1 - m)`

    The forward computation is `y = x * m + x * (1 - m) = x * (m + 1 - m) = x`.

    For the backward pass, consider the gradient of a loss `L` with respect
    to `x`. Using the chain rule, `dL/dx = dL/dy * dy/dx`.
    The derivative of the output `y` with respect to the input `x` is:
    `dy/dx = (d/dx stop_gradient(x)) * m + (d/dx x) * (1 - m)`
          `= 0 * m + 1 * (1 - m)`
          `= 1 - m`
    Therefore, the incoming gradient `dL/dy` is multiplied element-wise by
    `(1 - m)`, blocking gradients where `m=1` and allowing them to pass where
    `m=0`.

References:
    This technique is a fundamental building block related to the Straight-Through
    Estimator (STE), used for training neural networks with discrete stochastic
    neurons. While this layer is more general, it shares the core principle
    of creating a "fake" or custom gradient to enable end-to-end training
    through otherwise non-differentiable or selectively controlled paths.

    1.  Bengio, Y., Léonard, N., & Courville, A. (2013). Estimating or
        Propagating Gradients Through Stochastic Neurons for Conditional
        Computation.
    2.  Van Den Oord, A., & Vinyals, O. (2017). Neural Discrete Representation
        Learning. (Uses `stop_gradient` to pass gradients through a discrete
        bottleneck in VQ-VAE).
"""

import keras
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable(package="custom_layers")
class SelectiveGradientMask(keras.layers.Layer):
    """
    Layer that selectively stops gradients based on a binary mask.

    This layer allows fine-grained control over gradient flow by selectively stopping
    gradients at specified positions based on a binary mask. During the forward pass,
    the signal remains unchanged, while during backpropagation, gradients are blocked
    where mask equals 1.

    The layer implements a dual-path mechanism:
    - Stopped gradient path: For masked regions (mask = 1)
    - Normal gradient path: For unmasked regions (mask = 0)

    Computation: `stop_gradient(signal) * mask + signal * (1 - mask)`

    Args:
        name: Optional name for the layer.
        dtype: Optional datatype for layer computations.
        **kwargs: Additional keyword arguments passed to parent class.

    Call arguments:
        inputs: List of two tensors [signal, mask] with identical shapes.
            - signal: Input tensor to process (any shape)
            - mask: Binary mask tensor (same shape as signal) with values 0 or 1,
                   where 1 indicates gradient should be stopped
        training: Boolean indicating training phase.

    Input shape:
        List of two tensors with identical shapes:
        - signal: (batch_size, ...)
        - mask: (batch_size, ...)

    Output shape:
        Same as signal input shape: (batch_size, ...)

    Examples:
        >>> # Basic usage
        >>> signal = keras.Input(shape=(28, 28, 1))
        >>> mask = keras.Input(shape=(28, 28, 1))
        >>> masked_output = SelectiveGradientMask()([signal, mask])
        >>> model = keras.Model(inputs=[signal, mask], outputs=masked_output)

        >>> # Dynamic mask generation
        >>> features = keras.Input(shape=(64,))
        >>> dense = keras.layers.Dense(64, activation='sigmoid')(features)
        >>> mask = keras.ops.cast(keras.ops.greater(dense, 0.5), "float32")
        >>> output = SelectiveGradientMask()([dense, mask])
    """

    def __init__(
        self,
        name: Optional[str] = None,
        dtype: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """Initialize the SelectiveGradientMask layer."""
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.supports_masking = True

    def build(
        self,
        input_shape: Union[List[Tuple[Optional[int], ...]], Tuple[Tuple[Optional[int], ...], ...]]
    ) -> None:
        """
        Build the layer by validating input shapes.

        Args:
            input_shape: List or tuple containing shapes of [signal, mask] tensors.

        Raises:
            ValueError: If not exactly 2 inputs or if shapes don't match.
        """
        # Validate we have exactly 2 inputs
        if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 2:
            raise ValueError(
                f"SelectiveGradientMask expects exactly 2 inputs [signal, mask], "
                f"got {len(input_shape) if isinstance(input_shape, (list, tuple)) else 'invalid'} inputs."
            )

        signal_shape, mask_shape = input_shape

        # Validate shapes match
        if signal_shape != mask_shape:
            raise ValueError(
                f"Signal shape {signal_shape} must match mask shape {mask_shape}. "
                f"Both tensors must have identical dimensions."
            )

        # Set input spec for validation
        self.input_spec = [
            keras.layers.InputSpec(shape=signal_shape),
            keras.layers.InputSpec(shape=mask_shape)
        ]

        super().build(input_shape)

    def call(
        self,
        inputs: Union[List[keras.KerasTensor], Tuple[keras.KerasTensor, keras.KerasTensor]],
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply selective gradient masking.

        During training, creates two parallel paths:
        - A gradient-stopped path where mask == 1
        - A normal gradient flow path where mask == 0

        During inference, the signal passes through unchanged.

        Args:
            inputs: List or tuple containing [signal, mask] tensors.
            training: Boolean indicating training phase.

        Returns:
            Output tensor with selective gradient masking applied.

        Raises:
            ValueError: If inputs is not a list/tuple of exactly 2 tensors.
        """
        # Validate inputs structure
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError(
                f"SelectiveGradientMask expects exactly 2 inputs [signal, mask], "
                f"got {type(inputs).__name__} with {len(inputs) if isinstance(inputs, (list, tuple)) else 'unknown'} elements."
            )

        signal, mask = inputs

        # During inference, return unmodified signal for efficiency
        if not training:
            return signal

        # Cast mask to signal dtype to ensure compatibility
        mask = keras.ops.cast(mask, signal.dtype)

        # Create dual-path selective masking
        # Path 1: Stopped gradient for masked regions (mask = 1)
        stopped_gradient_path = keras.ops.multiply(
            keras.ops.stop_gradient(signal),
            mask
        )

        # Path 2: Normal gradient for unmasked regions (mask = 0)
        normal_gradient_path = keras.ops.multiply(
            signal,
            keras.ops.subtract(1.0, mask)
        )

        # Combine both paths
        output = keras.ops.add(stopped_gradient_path, normal_gradient_path)

        return output

    def compute_output_shape(
        self,
        input_shape: Union[List[Tuple[Optional[int], ...]], Tuple[Tuple[Optional[int], ...], ...]]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape.

        Args:
            input_shape: List of input shapes [signal_shape, mask_shape].

        Returns:
            Output shape (same as signal shape).

        Raises:
            ValueError: If input_shape is invalid.
        """
        if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 2:
            raise ValueError(
                f"Expected list of 2 input shapes [signal_shape, mask_shape], "
                f"got {type(input_shape).__name__} with length {len(input_shape) if hasattr(input_shape, '__len__') else 'unknown'}"
            )

        signal_shape, mask_shape = input_shape

        if signal_shape != mask_shape:
            raise ValueError(
                f"Signal and mask shapes must be identical. "
                f"Got signal: {signal_shape}, mask: {mask_shape}"
            )

        return signal_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns:
            Dictionary containing layer configuration.
        """
        config = super().get_config()
        # This layer has no additional parameters to serialize
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SelectiveGradientMask":
        """
        Create layer from configuration dictionary.

        Args:
            config: Configuration dictionary from get_config().

        Returns:
            Reconstructed SelectiveGradientMask layer instance.
        """
        return cls(**config)

# ---------------------------------------------------------------------
