"""ReLU-k activation layer implementing f(x) = max(0, x)^k.

This layer applies a powered ReLU activation function, providing more
expressiveness than standard ReLU while maintaining computational
efficiency and preserving non-linearity.

The ReLU-k activation is defined as:
    `f(x) = max(0, x)^k`

Where `k` is a positive integer power parameter. When `k=1`, this layer is
equivalent to the standard `keras.layers.ReLU`. Higher values of `k` create
more aggressive non-linearities that can help with gradient flow in
certain network architectures.
"""

import keras
from keras import ops
from typing import Optional, Any, Dict, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class ReLUK(keras.layers.Layer):
    """ReLU-k activation layer implementing f(x) = max(0, x)^k.

    This layer applies a powered ReLU activation function which provides
    more expressiveness than standard ReLU while maintaining computational
    efficiency and preserving the non-linearity properties.

    The ReLU-k activation is defined as:
        f(x) = max(0, x)^k

    Where k is a positive integer power parameter. When k=1, this reduces
    to the standard ReLU activation. Higher values of k create more
    aggressive activations that can help with gradient flow in certain
    architectures.

    Mathematical formulation:
        output = max(0, input)^k

    Where k is the power parameter and the max operation ensures non-negativity
    before applying the power transformation.

    Args:
        k: Integer, power exponent for the ReLU function. Must be a positive integer.
            Default is 3. Higher values create more aggressive non-linearities.
            When k=1, equivalent to standard ReLU.
        **kwargs: Additional keyword arguments passed to the Layer base class,
            such as `name`, `dtype`, `trainable`, etc.

    Input shape:
        Arbitrary tensor with any shape. Use the keyword argument `input_shape`
        (tuple of integers, does not include the batch axis) when using this
        layer as the first layer in a model.

    Output shape:
        Same shape as the input tensor.

    Attributes:
        k: The power exponent parameter used in the activation function.

    Example:
        ```python
        # Basic usage
        layer = ReLUK(k=2)
        inputs = keras.Input(shape=(10,))
        outputs = layer(inputs)

        # In a model
        model = keras.Sequential([
            keras.layers.Dense(64, input_shape=(784,)),
            ReLUK(k=3),  # Apply ReLU^3 activation
            keras.layers.Dense(10)
        ])

        # Custom power
        custom_layer = ReLUK(k=5, name='relu_k_5')
        ```

    References:
        The ReLU-k activation is inspired by various works on alternative
        activation functions that provide more flexibility than standard ReLU
        while maintaining computational efficiency.

    Raises:
        ValueError: If k is not a positive integer.
        TypeError: If k is not an integer.

    Note:
        - For k=1, this is equivalent to standard ReLU activation
        - Higher k values may cause vanishing gradients for small positive inputs (x < 1)
        - Higher k values may cause exploding gradients for large positive inputs (x > 1)
        - Consider the range of your input values and network stability when choosing k
        - This layer has no trainable parameters and adds minimal computational overhead
    """

    def __init__(
            self,
            k: int = 3,
            **kwargs: Any
    ) -> None:
        """Initialize the ReLU-k activation layer.

        Args:
            k: Power exponent for the ReLU function. Must be a positive integer.
                Default is 3.
            **kwargs: Additional keyword arguments for the Layer parent class.

        Raises:
            ValueError: If k is not a positive integer.
            TypeError: If k is not an integer.
        """
        super().__init__(**kwargs)

        # Validate k parameter
        if not isinstance(k, int):
            raise TypeError(f"k must be an integer, got type {type(k).__name__}")
        if k <= 0:
            raise ValueError(f"k must be a positive integer, got {k}")

        # Store configuration
        self.k = k

        logger.info(f"Initialized ReLUK layer with k={k}")

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass of the ReLU-k activation.

        Applies the ReLU-k function: f(x) = max(0, x)^k

        Args:
            inputs: Input tensor of any shape.
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode. Not used in this activation
                layer but kept for API consistency.

        Returns:
            Output tensor with the same shape as inputs, after applying
            the ReLU-k activation function.
        """
        # Apply ReLU to get non-negative values: max(0, x)
        relu_output = ops.maximum(0.0, inputs)

        # Apply power transformation
        if self.k == 1:
            # Optimization: avoid unnecessary power operation for standard ReLU
            return relu_output
        else:
            # Apply power: max(0, x)^k
            return ops.power(relu_output, float(self.k))

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        For activation layers, the output shape is identical to the input shape
        since no dimensional transformation occurs.

        Args:
            input_shape: Shape tuple of the input tensor.

        Returns:
            Output shape tuple, identical to input_shape.
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Get the layer configuration for serialization.

        Returns all parameters passed to __init__ so the layer can be
        properly reconstructed during model loading.

        Returns:
            Dictionary containing the layer configuration, including
            the power parameter k and parent class configuration.
        """
        config = super().get_config()
        config.update({
            "k": self.k,
        })
        return config

    def __repr__(self) -> str:
        """Return string representation of the layer.

        Returns:
            String representation including the layer name and key parameters.
        """
        return f"ReLUK(k={self.k}, name='{self.name}')"


# ---------------------------------------------------------------------
