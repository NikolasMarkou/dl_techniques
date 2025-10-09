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
    built by applying a linear transformation (`x + 3`), clamping the
    result to a fixed range `[0, 6]` (the ReLU6 operation), and finally
    scaling the output to `[0, 1]`. This avoids any transcendental
    function calls, significantly reducing latency.

Mathematical Foundation:
    The standard logistic sigmoid is defined as:
        σ(x) = 1 / (1 + exp(-x))

    The HardSigmoid approximates this with the piecewise linear function:
        h(x) = ReLU6(x + 3) / 6 = max(0, min(6, x + 3)) / 6

    This can be expressed in three parts:
        h(x) = 0          if x <= -3
        h(x) = (x / 6) + 0.5  if -3 < x < 3
        h(x) = 1          if x >= 3

    The linear segment `(x/6) + 0.5` serves as a first-order Taylor
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
    """
    Hard-sigmoid activation function for efficient sigmoid approximation.

    This layer implements a piecewise linear approximation of the sigmoid function
    that is computationally more efficient than the standard sigmoid, making it
    particularly suitable for mobile and edge computing applications. It's commonly
    used in squeeze-and-excitation modules and other attention mechanisms where
    computational efficiency is critical.

    **Intent**: Provide a fast, hardware-friendly approximation of sigmoid activation
    that maintains similar gating properties while being much more efficient to compute
    than the standard sigmoid function, especially on mobile and embedded devices.

    **Mathematical Operation**:
        hard_sigmoid(x) = max(0, min(6, x + 3)) / 6
                        = ReLU6(x + 3) / 6

    This can be rewritten as:
        hard_sigmoid(x) = {
            0       if x ≤ -3
            (x+3)/6 if -3 < x < 3
            1       if x ≥ 3
        }

    **Architecture**:
    ```
    Input(shape=[..., features])
           ↓
    Add Bias: x + 3
           ↓
    ReLU6 Clamp: max(0, min(6, x))
           ↓
    Scale: result / 6
           ↓
    Output(shape=[..., features]) ∈ [0, 1]
    ```

    **Comparison with Standard Sigmoid**:
    - Standard sigmoid: σ(x) = 1 / (1 + e^(-x))
    - Hard sigmoid: h(x) = ReLU6(x + 3) / 6
    - Both functions map input to [0, 1] range
    - Hard sigmoid is piecewise linear vs sigmoid's smooth curve
    - Hard sigmoid requires only addition, comparison, and division
    - Standard sigmoid requires expensive exponential computation

    Args:
        **kwargs: Additional arguments for Layer base class (name, trainable, etc.).

    Input shape:
        N-D tensor with any shape: `(batch_size, ..., features)`.

    Output shape:
        N-D tensor with same shape as input: `(batch_size, ..., features)`.
        Values are clamped to range [0, 1].

    Example:
        ```python
        # Basic usage as activation layer
        inputs = keras.Input(shape=(128,))
        outputs = HardSigmoid()(inputs)

        # In a squeeze-and-excitation block
        x = keras.layers.GlobalAveragePooling2D()(feature_maps)
        x = keras.layers.Dense(channels // reduction_ratio)(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Dense(channels)(x)
        x = HardSigmoid(name='se_gate')(x)  # Efficient gating

        # Comparison of different activation ranges
        import numpy as np
        test_input = keras.ops.convert_to_tensor(np.linspace(-6, 6, 13))
        hard_sigmoid_layer = HardSigmoid()
        output = hard_sigmoid_layer(test_input)
        # Input:  [-6, -5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5,  6]
        # Output: [ 0,  0,  0,  0, 1/6, 2/6, 3/6, 4/6, 5/6,  1,  1,  1,  1]
        #         [ 0,  0,  0,  0,0.17,0.33, 0.5,0.67,0.83,  1,  1,  1,  1]
        ```

    References:
        - MobileNets: https://arxiv.org/abs/1704.04861
        - Squeeze-and-Excitation Networks: https://arxiv.org/abs/1709.01507
        - Searching for MobileNetV3: https://arxiv.org/abs/1905.02244

    Note:
        This activation is deterministic and has no trainable parameters. The
        piecewise linear nature makes it very efficient for quantized models
        and mobile deployment scenarios.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.activation = keras.layers.ReLU(max_value=6.0)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply hard-sigmoid activation to inputs.

        Args:
            inputs: Input tensor of any shape.
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode. Not used in this layer but
                included for API consistency.

        Returns:
            Tensor with same shape as inputs, values clamped to [0, 1].
        """
        # Apply hard-sigmoid: ReLU6(x + 3) / 6
        # This is equivalent to: max(0, min(6, x + 3)) / 6
        return self.activation(inputs + 3.0) / 6.0

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute output shape given input shape.

        Hard-sigmoid is an element-wise operation that preserves the input shape.

        Args:
            input_shape: Shape tuple representing the input shape.

        Returns:
            Output shape tuple, identical to input shape.
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration for serialization.

        Since this layer has no configurable parameters beyond the base Layer
        class, we only need to return the parent configuration.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        # No additional parameters to add for this simple activation
        return config

# ---------------------------------------------------------------------
