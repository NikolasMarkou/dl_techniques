"""
``SelectiveGradientMask`` is an identity layer on the forward pass that
blocks gradients element-wise on the backward pass, based on a binary mask.

Given a signal ``x`` and mask ``m``, it computes
``y = stop_gradient(x) * m + x * (1 - m)``. The forward value is always
``x``, since the two terms are complementary and sum back to it. Because
``stop_gradient`` has zero derivative, the backward Jacobian is ``dy/dx = 1 - m``:
gradients are blocked where ``m == 1`` and pass through where ``m == 0``.
This is the same trick behind the Straight-Through Estimator and the
``stop_gradient`` bottleneck in VQ-VAE, applied element-wise rather than to
a whole tensor. At inference the signal is returned unchanged.

References:
    - Bengio et al., 2013. Estimating or Propagating Gradients Through
      Stochastic Neurons for Conditional Computation.
    - Van Den Oord and Vinyals, 2017. Neural Discrete Representation Learning.
"""

import keras
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.regularization.selective_gradient_mask")
class SelectiveGradientMask(keras.layers.Layer):
    """
    Selectively stop gradients based on a binary mask.

    During the forward pass this layer acts as an identity on the signal
    tensor. During backpropagation the dual-path computation
    ``output = stop_gradient(signal) * mask + signal * (1 - mask)``
    blocks gradients where ``mask == 1`` and passes them where
    ``mask == 0``. The effective backward Jacobian is
    ``dy/dx = 1 - mask``, providing fine-grained element-wise gradient
    control. At inference time the signal is returned unchanged for
    efficiency.

    Architecture:

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
        if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 2:
            raise ValueError(
                f"SelectiveGradientMask expects exactly 2 inputs [signal, mask], "
                f"got {len(input_shape) if isinstance(input_shape, (list, tuple)) else 'invalid'} inputs."
            )

        signal_shape, mask_shape = input_shape

        if signal_shape != mask_shape:
            raise ValueError(
                f"Signal shape {signal_shape} must match mask shape {mask_shape}. "
                f"Both tensors must have identical dimensions."
            )

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
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError(
                f"SelectiveGradientMask expects exactly 2 inputs [signal, mask], "
                f"got {type(inputs).__name__} with {len(inputs) if isinstance(inputs, (list, tuple)) else 'unknown'} elements."
            )

        signal, mask = inputs

        if not training:
            return signal

        mask = keras.ops.cast(mask, signal.dtype)

        stopped_gradient_path = keras.ops.multiply(
            keras.ops.stop_gradient(signal),
            mask
        )

        normal_gradient_path = keras.ops.multiply(
            signal,
            keras.ops.subtract(1.0, mask)
        )

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
