"""ReLU-k activation layer implementation for Keras 3.x.

This module implements the ReLU-k activation function, which applies
a power transformation to the standard ReLU activation.
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

    Args:
        k: Power exponent for the ReLU function. Must be a positive integer.
            Default is 3. Higher values create more aggressive non-linearities.
        **kwargs: Additional keyword arguments passed to the Layer parent class,
            such as `name`, `dtype`, `trainable`, etc.

    Input shape:
        Arbitrary. Use the keyword argument `input_shape` (tuple of integers,
        does not include the batch axis) when using this layer as the first
        layer in a model.

    Output shape:
        Same shape as the input.

    References:
        The ReLU-k activation is inspired by various works on alternative
        activation functions that provide more flexibility than standard ReLU.

    Note:
        - For k=1, this is equivalent to standard ReLU
        - Higher k values may cause vanishing gradients for small positive inputs
        - Consider the range of your input values when choosing k
    """

    def __init__(
            self,
            k: int = 3,
            **kwargs: Any
    ) -> None:
        """Initialize the ReLU-k activation layer.

        Args:
            k: Power exponent for the ReLU function. Must be a positive integer.
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

        self.k = k
        self._build_input_shape = None

        logger.info(f"Initialized ReLUK layer with k={k}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer weights and initialize internal state.

        This method is called automatically when the layer is first used.
        For activation layers, this typically just stores the input shape
        for serialization purposes.

        Args:
            input_shape: Shape tuple of the input tensor, including the batch
                dimension as None or an integer.
        """
        # Store input shape for serialization
        self._build_input_shape = input_shape

        logger.debug(f"Built ReLUK layer with input shape: {input_shape}")
        super().build(input_shape)

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
                training mode or inference mode. Not used in this layer but
                kept for API consistency.

        Returns:
            Output tensor with the same shape as inputs, after applying
            the ReLU-k activation function.
        """
        # Apply ReLU to get non-negative values
        relu_output = ops.maximum(0.0, inputs)

        # Apply power transformation
        if self.k == 1:
            # Optimization: avoid unnecessary power operation for standard ReLU
            return relu_output
        else:
            return ops.power(relu_output, float(self.k))

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        For activation layers, the output shape is identical to the input shape.

        Args:
            input_shape: Shape tuple of the input tensor.

        Returns:
            Output shape tuple, identical to input_shape.
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Get the layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration, including
            the power parameter k and parent class configuration.
        """
        config = super().get_config()
        config.update({
            "k": self.k,
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get the build configuration for serialization.

        This method returns the configuration needed to properly rebuild
        the layer after deserialization.

        Returns:
            Dictionary containing the build configuration, specifically
            the input shape used during the build process.
        """
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build the layer from a build configuration.

        This method is used during model loading to properly rebuild
        the layer's internal state.

        Args:
            config: Dictionary containing the build configuration,
                as returned by get_build_config().
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

    def __repr__(self) -> str:
        """Return string representation of the layer.

        Returns:
            String representation including the layer name and key parameters.
        """
        return f"ReLUK(k={self.k}, name='{self.name}')"

# ---------------------------------------------------------------------
