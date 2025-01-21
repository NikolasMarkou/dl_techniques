"""Implementation of Expanded Gating Range Activation Functions.

This module implements the activation functions proposed in the paper
"Expanded Gating Ranges Improve Activation Functions" by Allen Hao Huang.

The key components are:
- Base activation functions (GELU, SiLU)
- Expanded variants (xATLU, xGELU, xSiLU)

References:
    Huang, A. H. (2023). Expanded Gating Ranges Improve Activation Functions.
"""

import keras
import numpy as np
import tensorflow as tf
from typing import Optional, Union, Tuple


class BaseActivation(keras.layers.Layer):
    """Base class for all activation functions.

    Provides common functionality and interface for activation layers.
    """

    def __init__(
            self,
            trainable: bool = True,
            name: Optional[str] = None,
            dtype: Optional[Union[str, tf.dtypes.DType]] = None,
            dynamic: bool = False,
            **kwargs
    ) -> None:
        """Initialize the activation layer.

        Args:
            trainable: Whether the layer's variables are trainable.
            name: Name of the layer.
            dtype: Dtype of the layer's computations and weights.
            dynamic: Whether the layer is dynamic (can change its shape).
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(
            trainable=trainable,
            name=name,
            dtype=dtype,
            dynamic=dynamic,
            **kwargs
        )

    def get_config(self) -> dict:
        """Get layer configuration.

        Returns:
            Dictionary containing the layer configuration.
        """
        return super().get_config()


class GELU(BaseActivation):
    """Gaussian Error Linear Unit activation function.

    Implements GELU activation: x * Φ(x), where Φ is the standard normal CDF.
    """

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Apply GELU activation to inputs.

        Args:
            inputs: Input tensor.

        Returns:
            Tensor with GELU activation applied.
        """
        return 0.5 * inputs * (1 + tf.math.erf(inputs / tf.math.sqrt(2.0)))


class SiLU(BaseActivation):
    """Sigmoid Linear Unit activation function.

    Implements SiLU activation: x * sigmoid(x).
    """

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Apply SiLU activation to inputs.

        Args:
            inputs: Input tensor.

        Returns:
            Tensor with SiLU activation applied.
        """
        return inputs * tf.sigmoid(inputs)


class ExpandedActivation(BaseActivation):
    """Base class for expanded gating range activation functions.

    Implements the expanded gating mechanism with a trainable alpha parameter.
    """

    def build(self, input_shape: Union[tf.TensorShape, Tuple]) -> None:
        """Build the layer, creating the trainable alpha parameter.

        Args:
            input_shape: Shape of the input tensor.
        """
        self.alpha = self.add_weight(
            name='alpha',
            shape=(),
            initializer='zeros',
            trainable=True,
            dtype=self.dtype
        )
        super().build(input_shape)

    def get_config(self) -> dict:
        """Get layer configuration.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        return config


class xATLU(ExpandedActivation):
    """Expanded ArcTan Linear Unit activation function.

    Implements xATLU activation with trainable expanded gating range.
    """

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Apply xATLU activation to inputs.

        Args:
            inputs: Input tensor.

        Returns:
            Tensor with xATLU activation applied.
        """
        gate = (tf.math.atan(inputs) + np.pi / 2) / np.pi
        return inputs * (gate * (1 + 2 * self.alpha) - self.alpha)


class xGELU(ExpandedActivation):
    """Expanded Gaussian Error Linear Unit activation function.

    Implements xGELU activation with trainable expanded gating range.
    """

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Apply xGELU activation to inputs.

        Args:
            inputs: Input tensor.

        Returns:
            Tensor with xGELU activation applied.
        """
        gate = 0.5 * (1 + tf.math.erf(inputs / tf.math.sqrt(2.0)))
        return inputs * (gate * (1 + 2 * self.alpha) - self.alpha)


class xSiLU(ExpandedActivation):
    """Expanded Sigmoid Linear Unit activation function.

    Implements xSiLU activation with trainable expanded gating range.
    """

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Apply xSiLU activation to inputs.

        Args:
            inputs: Input tensor.

        Returns:
            Tensor with xSiLU activation applied.
        """
        gate = tf.sigmoid(inputs)
        return inputs * (gate * (1 + 2 * self.alpha) - self.alpha)


def get_activation(activation_name: str) -> BaseActivation:
    """Factory function to get activation layer by name.

    Args:
        activation_name: Name of the activation function.

    Returns:
        Instantiated activation layer.

    Raises:
        ValueError: If activation_name is not recognized.
    """
    activations = {
        'gelu': GELU,
        'silu': SiLU,
        'xatlu': xATLU,
        'xgelu': xGELU,
        'xsilu': xSiLU
    }

    activation_class = activations.get(activation_name.lower())
    if activation_class is None:
        raise ValueError(
            f'Unknown activation: {activation_name}. '
            f'Available activations: {list(activations.keys())}'
        )

    return activation_class()