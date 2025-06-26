"""
==========================================
Expanded Gating Range Activation Functions
==========================================

Overview
--------
This module provides a set of activation functions, including standard ones
(GELU, SiLU) and their expanded gating range variants (xATLU, xGELU, xSiLU).
The expanded versions introduce a trainable parameter α (alpha) that broadens
(or contracts) the effective gating range of each underlying function, potentially
improving performance and flexibility in deep neural networks.

Activation Functions
--------------------

1. GELU (Gaussian Error Linear Unit)
   - Purpose:
     GELU combines linear and Gaussian functions to provide a smooth, non-monotonic
     activation that often yields better performance than ReLU-based activations,
     particularly in Transformer-based architectures.
   - Equation:
       GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
   - Pros:
     * Generally achieves state-of-the-art results in many NLP tasks.
     * Smoothness can help with gradient-based optimization.
     * Widely adopted and easy to integrate.
   - Cons:
     * Slightly more computationally expensive than simpler functions (e.g., ReLU).
     * Not always guaranteed to outperform all alternatives in every domain.

2. SiLU (Sigmoid Linear Unit)
   - Purpose:
     SiLU (also known as Swish) blends linear and sigmoid behaviors, aiming for a
     more continuous and smooth activation compared to piecewise functions like ReLU.
   - Equation:
       SiLU(x) = x * sigmoid(x)
   - Pros:
     * Empirically shown to sometimes outperform ReLU in certain tasks.
     * Produces non-zero outputs for negative inputs, offering a smoother gradient.
   - Cons:
     * Slightly more expensive computationally than simple functions like ReLU.
     * Performance benefits can be task-dependent.

3. xATLU (Expanded ArcTan Linear Unit)
   - Purpose:
     xATLU uses the arctan function as the gating mechanism, then expands (or contracts)
     its range via the trainable parameter α. This can offer finer control over how
     inputs are mapped.
   - Equation:
       gate(x) = (arctan(x) + π/2) / π
       xATLU(x) = x * (gate(x) * (1 + 2α) - α)
   - Pros:
     * Arctan-based gating can yield smoother transitions for moderate input values.
     * The trainable α provides flexibility to adapt gating behavior for specific tasks.
   - Cons:
     * May be less intuitive compared to widely used activations like ReLU or SiLU.
     * Additional parameter α could lead to overfitting if not regularized properly.

4. xGELU (Expanded Gaussian Error Linear Unit)
   - Purpose:
     xGELU extends the GELU activation by introducing a trainable parameter α that
     adjusts how rapidly the gating changes. This allows for a customizable version
     of GELU that can adapt to various tasks and architectures.
   - Equation:
       gate(x) = 0.5 * (1 + erf(x / sqrt(2)))
       xGELU(x) = x * (gate(x) * (1 + 2α) - α)
   - Pros:
     * Maintains the smooth, non-monotonic properties of GELU while adding adaptability.
     * Can learn an optimal gating range specific to the dataset or model.
   - Cons:
     * More complex to tune due to the extra α parameter.
     * Increases memory and computation overhead slightly.

5. xSiLU (Expanded Sigmoid Linear Unit)
   - Purpose:
     xSiLU builds on the SiLU activation by expanding or contracting its sigmoid
     gating range via α, which can lead to a more flexible gradient flow.
   - Equation:
       gate(x) = sigmoid(x)
       xSiLU(x) = x * (gate(x) * (1 + 2α) - α)
   - Pros:
     * Combines the smoothness of SiLU with an adaptable gating range.
     * The extra parameter α can help capture task-specific activation characteristics.
   - Cons:
     * Requires careful tuning of α to avoid vanishing or exploding gate outputs.
     * Similar to xGELU, the added complexity may not always justify the gains.

General Pros and Cons of Expanded Activations
---------------------------------------------
- Pros:
  * Adaptable gating allows the network to tailor activation dynamics to the problem.
  * Potential for improved performance over fixed-range gating in certain tasks.

- Cons:
  * Additional trainable parameter α adds model complexity and potential risk of overfitting.
  * Requires more careful hyperparameter tuning, especially regarding initial values
    and regularization.

Usage
-----
All activation classes inherit from a common `BaseActivation` layer, ensuring they
can be used seamlessly in TensorFlow/Keras models. For instance:

    model.add(Dense(64))
    model.add(xGELU())

or via the factory method:

    activation_layer = get_activation("xgelu")
    model.add(activation_layer)

Reference
---------
Huang, A. H. (2023). Expanded Gating Ranges Improve Activation Functions.
"""


import keras
import numpy as np
import tensorflow as tf
from typing import Optional, Union, Tuple


# ---------------------------------------------------------------------


class BaseActivation(keras.layers.Layer):
    """
    Base class for all custom activation functions.

    This class provides a common interface and functionality for custom
    activation layers, including the standard Keras Layer configurations
    such as `trainable`, `dtype`, and `name`.
    """

    def __init__(
        self,
        trainable: bool = True,
        name: Optional[str] = None,
        dtype: Optional[Union[str, tf.dtypes.DType]] = None,
        dynamic: bool = False,
        **kwargs
    ) -> None:
        """
        Initialize the base activation layer.

        Args:
            trainable (bool): Whether the layer's variables are trainable.
            name (str, optional): Name of the layer.
            dtype (str or tf.dtypes.DType, optional): Data type for the
                layer's computations and variables.
            dynamic (bool): Whether the layer should be dynamic (able to
                handle varying shapes across calls).
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
        """
        Returns the configuration of the layer.

        This method is used to save and restore the layer's state during
        model serialization.

        Returns:
            dict: A dictionary containing the layer configuration.
        """
        config = super().get_config()
        return config


# ---------------------------------------------------------------------


class GELU(BaseActivation):
    """
    Gaussian Error Linear Unit (GELU) activation function.

    The GELU activation is defined as:
        GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2))).

    References:
        Hendrycks, D., & Gimpel, K. (2016). Gaussian Error Linear Units (GELUs).
    """

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Apply the GELU activation function to the inputs.

        Args:
            inputs (tf.Tensor): The input tensor.

        Returns:
            tf.Tensor: Output tensor after applying the GELU activation.
        """
        return 0.5 * inputs * (1 + tf.math.erf(inputs / tf.math.sqrt(2.0)))


# ---------------------------------------------------------------------


class SiLU(BaseActivation):
    """
    Sigmoid Linear Unit (SiLU) activation function.

    The SiLU activation is defined as:
        SiLU(x) = x * sigmoid(x).
    """

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Apply the SiLU activation function to the inputs.

        Args:
            inputs (tf.Tensor): The input tensor.

        Returns:
            tf.Tensor: Output tensor after applying the SiLU activation.
        """
        return inputs * tf.sigmoid(inputs)


# ---------------------------------------------------------------------


class ExpandedActivation(BaseActivation):
    """
    Base class for expanded gating range activation functions.

    This class introduces a trainable scalar parameter `alpha` that
    modifies the gating range of the activation function. Child classes
    use this parameter to expand or contract the range through which
    the gating function (e.g., sigmoid, erf, arctan) operates.
    """

    def build(self, input_shape: Union[tf.TensorShape, Tuple]) -> None:
        """
        Create the trainable parameter `alpha` for the expanded activation.

        Args:
            input_shape (Union[tf.TensorShape, Tuple]): Shape of the input.
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
        """
        Returns the configuration of the expanded activation layer.

        The config dictionary includes the parent layer's configuration
        plus any relevant custom parameters.

        Returns:
            dict: A dictionary containing the layer configuration.
        """
        config = super().get_config()
        # Could add custom parameters here if needed (e.g., initial value of alpha)
        return config


# ---------------------------------------------------------------------


class xATLU(ExpandedActivation):
    """
    Expanded ArcTan Linear Unit activation function.

    This variant uses an arctan-based gating function, further expanded
    by a trainable parameter `alpha`.

    The gating function is defined as:
        gate(x) = (arctan(x) + π/2) / π
    and the output is:
        x * (gate(x) * (1 + 2 * alpha) - alpha).
    """

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Apply the xATLU activation function to the inputs.

        Args:
            inputs (tf.Tensor): The input tensor.

        Returns:
            tf.Tensor: The output tensor after applying xATLU.
        """
        gate = (tf.math.atan(inputs) + np.pi / 2) / np.pi
        return inputs * (gate * (1 + 2 * self.alpha) - self.alpha)


# ---------------------------------------------------------------------


class xGELU(ExpandedActivation):
    """
    Expanded Gaussian Error Linear Unit (xGELU) activation function.

    This variant uses the GELU gating function, expanded by a trainable
    parameter `alpha`.

    The gating function is:
        gate(x) = 0.5 * (1 + erf(x / sqrt(2))),
    and the output is:
        x * (gate(x) * (1 + 2 * alpha) - alpha).
    """

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Apply the xGELU activation function to the inputs.

        Args:
            inputs (tf.Tensor): The input tensor.

        Returns:
            tf.Tensor: The output tensor after applying xGELU.
        """
        gate = 0.5 * (1 + tf.math.erf(inputs / tf.math.sqrt(2.0)))
        return inputs * (gate * (1 + 2 * self.alpha) - self.alpha)


# ---------------------------------------------------------------------


class xSiLU(ExpandedActivation):
    """
    Expanded Sigmoid Linear Unit (xSiLU) activation function.

    This variant uses the sigmoid gating function, expanded by a trainable
    parameter `alpha`.

    The gating function is:
        gate(x) = sigmoid(x),
    and the output is:
        x * (gate(x) * (1 + 2 * self.alpha) - self.alpha).
    """

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Apply the xSiLU activation function to the inputs.

        Args:
            inputs (tf.Tensor): The input tensor.

        Returns:
            tf.Tensor: The output tensor after applying xSiLU.
        """
        gate = tf.sigmoid(inputs)
        return inputs * (gate * (1 + 2 * self.alpha) - self.alpha)


# ---------------------------------------------------------------------


def get_activation(activation_name: str) -> BaseActivation:
    """
    Factory function to create an activation layer by name.

    Supported activation names:
        - 'gelu'
        - 'silu'
        - 'xatlu'
        - 'xgelu'
        - 'xsilu'

    Args:
        activation_name (str): Name of the desired activation function.

    Returns:
        BaseActivation: An instance of the specified activation class.

    Raises:
        ValueError: If `activation_name` is not recognized.
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
            f"Unknown activation: '{activation_name}'. "
            f"Available activations: {list(activations.keys())}"
        )

    return activation_class()

# ---------------------------------------------------------------------