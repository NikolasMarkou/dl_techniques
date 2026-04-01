"""
HardSwish activation, a computationally efficient Swish variant.

This layer provides a hardware-friendly approximation of the Swish activation
function, designed to reduce computational cost in mobile and edge computing
environments. It preserves the core architectural benefit of Swish---its
self-gating mechanism---while replacing the expensive sigmoid function with a
piecewise linear counterpart, the HardSigmoid. This makes it a foundational
component of efficient architectures like MobileNetV3.

Architectural Design:
    The HardSwish function maintains the self-gating structure of the
    original Swish, where the input ``x`` is modulated by a gating signal
    derived from ``x`` itself. The key innovation is the replacement of the
    smooth ``sigmoid`` gate with the ``HardSigmoid`` approximation.
    -   Original Swish: ``output = x * sigmoid(x)``
    -   HardSwish: ``output = x * hard_sigmoid(x)``
    This design retains the desirable properties of Swish, such as
    non-monotonicity and improved gradient flow, but eliminates the need
    for costly exponential computations. The result is an activation that
    closely mimics Swish's behavior but is significantly faster to execute,
    especially in quantized or low-precision settings.

Mathematical Foundation:
    The standard Swish function is defined as:
        ``f(x) = x * sigma(x) = x / (1 + exp(-x))``

    HardSwish approximates this by substituting the sigmoid with its
    piecewise linear approximation, ``hard_sigmoid(x) = ReLU6(x + 3) / 6``:
        ``h(x) = x * hard_sigmoid(x) = x * [ReLU6(x + 3) / 6]``

    This results in a function defined by three segments:
    -   ``h(x) = 0``, for ``x <= -3``
    -   ``h(x) = x * (x + 3) / 6``, for ``-3 < x < 3``
    -   ``h(x) = x``, for ``x >= 3``

    The function is a composition of a zero region, a quadratic curve, and
    a linear identity region. The quadratic segment provides the crucial
    non-monotonic "dip" characteristic of Swish, which can enhance the
    expressive capacity of the network. By constructing this shape from
    simple arithmetic, HardSwish serves as an effective drop-in replacement
    for Swish in performance-critical applications.

References:
    - Howard, A., et al. (2019). "Searching for MobileNetV3."
    - Ramachandran, P., Zoph, B., & Le, Q. V. (2017). "Searching for
      Activation Functions." (Introduced the original Swish function).

"""

import keras
from typing import Optional, Tuple, Dict, Any

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class HardSwish(keras.layers.Layer):
    """Hard-swish activation function for efficient mobile-optimized networks.

    This layer implements a computationally efficient approximation of the Swish
    activation function: ``hard_swish(x) = x * ReLU6(x + 3) / 6``. It is
    specifically designed for mobile and edge computing applications, replacing
    the expensive sigmoid in standard Swish with a piecewise linear ReLU6-based
    approximation. The function is self-gating, non-monotonic, unbounded above,
    and uses only basic arithmetic operations.

    **Architecture Overview:**

    .. code-block:: text

        Input: x (batch, ..., features)
                │
                ├───────────────────────┐
                │                       │
                ▼                       ▼
        ┌──────────────┐   ┌────────────────────────┐
        │   Identity   │   │    Hard Sigmoid:        │
        │      x       │   │  ReLU6(x + 3) / 6      │
        └──────┬───────┘   └───────────┬────────────┘
               │                       │
               └───────┬───────────────┘
                       │
                       ▼
               ┌───────────────────────┐
               │ x * hard_sigmoid(x)   │
               └───────────┬───────────┘
                           │
                           ▼
        Output: (batch, ..., features)

    :param kwargs: Additional arguments for Layer base class (``name``, ``trainable``, etc.).

    References:
        - Searching for MobileNetV3: https://arxiv.org/abs/1905.02244
        - Swish: A Self-Gated Activation Function: https://arxiv.org/abs/1710.05941
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the HardSwish layer.

        :param kwargs: Additional keyword arguments for the Layer base class.
        """
        super().__init__(**kwargs)
        self.activation = keras.layers.ReLU(max_value=6.0)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply hard-swish activation: ``x * ReLU6(x + 3) / 6``.

        :param inputs: Input tensor of any shape.
        :type inputs: keras.KerasTensor
        :param training: Boolean indicating training or inference mode. Not used
            in this layer but included for API consistency.
        :type training: Optional[bool]
        :return: Tensor with same shape as inputs, transformed by hard-swish.
        :rtype: keras.KerasTensor
        """
        # Apply hard-swish: x * hard_sigmoid(x) = x * ReLU6(x + 3) / 6
        # This combines the input with its gating signal
        hard_sigmoid_part = self.activation(inputs + 3.0) / 6.0
        return inputs * hard_sigmoid_part

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape given input shape.

        :param input_shape: Shape tuple representing the input shape.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple, identical to input shape.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return the layer configuration for serialization.

        :return: Dictionary containing the layer configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        # No additional parameters to add for this parameter-free activation
        return config
