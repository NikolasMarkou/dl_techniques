"""
Selectively mask gradients during backpropagation without altering the forward pass.

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
    """Selectively stop gradients based on a binary mask.

    During the forward pass this layer acts as an identity on the signal
    tensor. During backpropagation the dual-path computation
    ``output = stop_gradient(signal) * mask + signal * (1 - mask)``
    blocks gradients where ``mask == 1`` and passes them where
    ``mask == 0``. The effective backward Jacobian is
    ``dy/dx = 1 - mask``, providing fine-grained element-wise gradient
    control. At inference time the signal is returned unchanged for
    efficiency.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────┐    ┌──────────────┐
        │  Signal      │    │  Mask        │
        │  [B, ...]    │    │  [B, ...]    │
        └──────┬───────┘    └─────┬────────┘
               │                  │
               │        ┌─────────┴──────────┐
               │        │                    │
               ▼        ▼                    ▼
        ┌────────────────────┐  ┌────────────────────┐
        │ stop_gradient(sig) │  │ signal * (1-mask)  │
        │ * mask             │  │ (gradient flows)   │
        └────────┬───────────┘  └────────┬───────────┘
                 │                       │
                 └───────────┬───────────┘
                             ▼
        ┌──────────────────────────────────┐
        │  Add → Output [B, ...]           │
        │  (forward = signal, backward     │
        │   grad *= (1 - mask))            │
        └──────────────────────────────────┘

    :param name: Optional layer name.
    :type name: Optional[str]
    :param dtype: Optional datatype for computations.
    :type dtype: Optional[str]
    :param kwargs: Additional keyword arguments for the Layer base class.
    :type kwargs: Any"""

    def __init__(
        self,
        name: Optional[str] = None,
        dtype: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """Initialise the SelectiveGradientMask layer."""
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.supports_masking = True

    def build(
        self,
        input_shape: Union[List[Tuple[Optional[int], ...]], Tuple[Tuple[Optional[int], ...], ...]]
    ) -> None:
        """Build the layer by validating input shapes.

        :param input_shape: List of two shape tuples ``[signal, mask]``.
        :type input_shape: Union[List, Tuple]"""
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
        """Apply selective gradient masking.

        :param inputs: List of ``[signal, mask]`` tensors.
        :type inputs: Union[List, Tuple]
        :param training: Whether in training mode.
        :type training: Optional[bool]
        :return: Output tensor (signal with masked gradients).
        :rtype: keras.KerasTensor"""
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
        """Compute the output shape (same as signal shape).

        :param input_shape: List of ``[signal_shape, mask_shape]``.
        :type input_shape: Union[List, Tuple]
        :return: Output shape tuple.
        :rtype: Tuple[Optional[int], ...]"""
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
        """Return layer configuration for serialization.

        :return: Dictionary containing layer configuration.
        :rtype: Dict[str, Any]"""
        config = super().get_config()
        # This layer has no additional parameters to serialize
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SelectiveGradientMask":
        """Create layer from a configuration dictionary.

        :param config: Configuration from ``get_config()``.
        :type config: Dict[str, Any]
        :return: Reconstructed layer instance.
        :rtype: SelectiveGradientMask"""
        return cls(**config)

# ---------------------------------------------------------------------
