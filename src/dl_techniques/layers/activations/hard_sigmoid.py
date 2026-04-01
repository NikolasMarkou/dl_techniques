"""
A computationally efficient, piecewise linear sigmoid approximation.

This layer provides a hardware-friendly approximation of the standard
logistic sigmoid function. Its primary purpose is to replace the
computationally expensive exponential operation in the standard sigmoid
with simple arithmetic, making it ideal for deployment on resource-
constrained environments like mobile or edge devices, and for quantized
models. It is a core component in modern efficient architectures such as
MobileNetV3.

Architectural Design:
    The HardSigmoid is constructed as a piecewise linear function composed
    of three segments: a zero region, a linear ramp, and a saturation
    region at one. This structure mimics the S-shape of the true sigmoid
    while being implementable with basic, fast operations. The function is
    built by applying a linear transformation (``x + 3``), clamping the
    result to a fixed range ``[0, 6]`` (the ReLU6 operation), and finally
    scaling the output to ``[0, 1]``. This avoids any transcendental
    function calls, significantly reducing latency.

Mathematical Foundation:
    The standard logistic sigmoid is defined as:
        ``sigma(x) = 1 / (1 + exp(-x))``

    The HardSigmoid approximates this with the piecewise linear function:
        ``h(x) = ReLU6(x + 3) / 6 = max(0, min(6, x + 3)) / 6``

    This can be expressed in three parts:
        - ``h(x) = 0``              if x <= -3
        - ``h(x) = (x / 6) + 0.5`` if -3 < x < 3
        - ``h(x) = 1``              if x >= 3

    The linear segment ``(x/6) + 0.5`` serves as a first-order Taylor
    approximation of the sigmoid function around x=0, but shifted and
    scaled. The key insight is that this simple linear ramp is a
    "good enough" approximation for the gating and activation purposes
    served by the sigmoid in many neural network components.

References:
    - Howard, A., et al. (2019). "Searching for MobileNetV3."
    - Courbariaux, M., et al. (2015). "BinaryConnect: Training Deep
      Neural Networks with binary weights during propagations." (Introduced
      an early version of the hard sigmoid).

"""

import keras
from typing import Optional, Tuple, Dict, Any

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class HardSigmoid(keras.layers.Layer):
    """Hard-sigmoid activation function for efficient sigmoid approximation.

    This layer implements a piecewise linear approximation of the sigmoid
    function: ``hard_sigmoid(x) = max(0, min(6, x + 3)) / 6 = ReLU6(x + 3) / 6``.
    It is computationally more efficient than the standard sigmoid, making it
    particularly suitable for mobile and edge computing applications. It is
    commonly used in squeeze-and-excitation modules and other attention
    mechanisms where computational efficiency is critical.

    **Architecture Overview:**

    .. code-block:: text

        Input: x (batch, ..., features)
                │
                ▼
        ┌───────────────────┐
        │   Add Bias: x + 3 │
        └───────┬───────────┘
                │
                ▼
        ┌───────────────────────────┐
        │ ReLU6: max(0, min(6, x)) │
        └───────────┬───────────────┘
                    │
                    ▼
        ┌───────────────────┐
        │  Scale: x / 6    │
        └───────┬───────────┘
                │
                ▼
        Output: (batch, ..., features) in [0, 1]

    :param kwargs: Additional arguments for Layer base class (``name``, ``trainable``, etc.).

    References:
        - MobileNets: https://arxiv.org/abs/1704.04861
        - Squeeze-and-Excitation Networks: https://arxiv.org/abs/1709.01507
        - Searching for MobileNetV3: https://arxiv.org/abs/1905.02244
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the HardSigmoid layer.

        :param kwargs: Additional keyword arguments for the Layer base class.
        """
        super().__init__(**kwargs)
        self.activation = keras.layers.ReLU(max_value=6.0)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply hard-sigmoid activation: ``ReLU6(x + 3) / 6``.

        :param inputs: Input tensor of any shape.
        :type inputs: keras.KerasTensor
        :param training: Boolean indicating training or inference mode. Not used
            in this layer but included for API consistency.
        :type training: Optional[bool]
        :return: Tensor with same shape as inputs, values clamped to [0, 1].
        :rtype: keras.KerasTensor
        """
        # Apply hard-sigmoid: ReLU6(x + 3) / 6
        # This is equivalent to: max(0, min(6, x + 3)) / 6
        return self.activation(inputs + 3.0) / 6.0

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
        # No additional parameters to add for this simple activation
        return config

# ---------------------------------------------------------------------
