import keras
from typing import Optional, Tuple, Dict, Any

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class HardSwish(keras.layers.Layer):
    """
    Hard-swish activation function for efficient mobile-optimized neural networks.

    This layer implements a computationally efficient approximation of the Swish
    activation function (also known as SiLU) that is specifically designed for
    mobile and edge computing applications. Hard-swish replaces the expensive
    sigmoid computation in standard swish with a piecewise linear approximation
    using ReLU6, making it much more hardware-friendly while preserving most
    of the beneficial properties of the original swish function.

    **Intent**: Provide a mobile-optimized activation function that combines the
    beneficial properties of swish (smooth, self-gating, non-monotonic) with the
    computational efficiency required for real-time inference on resource-constrained
    devices like smartphones and embedded systems.

    **Mathematical Operation**:
        hard_swish(x) = x * hard_sigmoid(x)
                      = x * ReLU6(x + 3) / 6

    This can be expanded as:
        hard_swish(x) = {
            0                if x ≤ -3
            x * (x + 3) / 6  if -3 < x < 3
            x                if x ≥ 3
        }

    **Architecture**:
    ```
    Input(shape=[..., features])
           ↓
    Branch 1: Identity → x
           ↓
    Branch 2: Hard Sigmoid → x + 3 → ReLU6 → / 6
           ↓
    Element-wise Multiply: x * hard_sigmoid(x)
           ↓
    Output(shape=[..., features])
    ```

    **Comparison with Standard Swish**:
    - Standard swish: swish(x) = x * σ(x) = x / (1 + e^(-x))
    - Hard swish: h_swish(x) = x * ReLU6(x + 3) / 6
    - Both are self-gating functions that can produce negative values
    - Hard swish is piecewise linear vs swish's smooth curve
    - Hard swish eliminates expensive exponential computation
    - Both functions are non-monotonic and unbounded above

    **Key Properties**:
    - Self-gating: The function gates its own input
    - Non-monotonic: Can decrease then increase (unlike ReLU)
    - Unbounded above: Can produce arbitrarily large positive values
    - Bounded below: Cannot go below 0 when x ≤ -3
    - Smooth transitions: Despite being piecewise, has continuous derivatives
    - Mobile-friendly: Uses only basic arithmetic operations

    Args:
        **kwargs: Additional arguments for Layer base class (name, trainable, etc.).

    Input shape:
        N-D tensor with any shape: `(batch_size, ..., features)`.

    Output shape:
        N-D tensor with same shape as input: `(batch_size, ..., features)`.
        Values can be negative for small inputs but are non-negative for x ≥ -3.

    Example:
        ```python
        # Basic usage as activation layer
        inputs = keras.Input(shape=(256,))
        outputs = HardSwish()(inputs)

        # In a MobileNetV3-style block
        x = keras.layers.Conv2D(32, 3, padding='same')(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = HardSwish(name='mobilenet_activation')(x)  # Efficient activation

        # Comparison of activation behavior
        import numpy as np
        test_input = keras.ops.convert_to_tensor(np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4]))
        hard_swish_layer = HardSwish()
        output = hard_swish_layer(test_input)
        # Input:  [-4, -3, -2, -1,  0,  1,  2,  3,  4]
        # Output: [ 0,  0,-1/3,-1/3, 0, 2/3, 5/3,  3,  4]
        #         [ 0,  0,-0.33,-0.33, 0, 0.67, 1.67, 3, 4]

        # In a complete mobile model
        model = keras.Sequential([
            keras.layers.Conv2D(16, 3, strides=2, padding='same'),
            keras.layers.BatchNormalization(),
            HardSwish(),
            keras.layers.Conv2D(32, 3, strides=2, padding='same'),
            keras.layers.BatchNormalization(),
            HardSwish(),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(1000),
            HardSwish(name='final_activation')
        ])
        ```

    References:
        - Searching for MobileNetV3: https://arxiv.org/abs/1905.02244
        - Swish: A Self-Gated Activation Function: https://arxiv.org/abs/1710.05941
        - MobileNets: Efficient Convolutional Neural Networks: https://arxiv.org/abs/1704.04861
        - H-Swish activation in MobileNetV3: https://paperswithcode.com/method/h-swish

    Note:
        This activation is deterministic and has no trainable parameters. The
        piecewise linear nature of the hard-sigmoid component makes it highly
        efficient for quantized models and mobile deployment. It's particularly
        effective in the later layers of networks where the self-gating property
        helps with gradient flow and feature selection.
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
        Apply hard-swish activation to inputs.

        The hard-swish function applies a self-gating mechanism where each input
        element is multiplied by its corresponding hard-sigmoid value, creating
        a smooth, non-monotonic activation that can enhance gradient flow.

        Args:
            inputs: Input tensor of any shape.
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode. Not used in this layer but
                included for API consistency.

        Returns:
            Tensor with same shape as inputs, where each element has been
            transformed by the hard-swish function.
        """
        # Apply hard-swish: x * hard_sigmoid(x) = x * ReLU6(x + 3) / 6
        # This combines the input with its gating signal
        hard_sigmoid_part = self.activation(inputs + 3.0) / 6.0
        return inputs * hard_sigmoid_part

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute output shape given input shape.

        Hard-swish is an element-wise operation that preserves the input shape,
        applying the activation function independently to each element.

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
        class, we only need to return the parent configuration. This ensures
        proper serialization and deserialization of models containing this layer.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        # No additional parameters to add for this parameter-free activation
        return config