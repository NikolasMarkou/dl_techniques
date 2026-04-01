"""
Expanded Gating Range Activation Functions.

This module provides a set of activation functions, including standard ones
(GELU, SiLU) and their expanded gating range variants (xATLU, xGELU, xSiLU).
The expanded versions introduce a trainable parameter alpha that broadens
(or contracts) the effective gating range of each underlying function, potentially
improving performance and flexibility in deep neural networks.

Activation Functions:

1. **GELU** (Gaussian Error Linear Unit):
   ``GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))``
   Combines linear and Gaussian functions for smooth, non-monotonic activation.

2. **SiLU** (Sigmoid Linear Unit / Swish):
   ``SiLU(x) = x * sigmoid(x)``
   Blends linear and sigmoid behaviors for smoother gradients than ReLU.

3. **xATLU** (Expanded ArcTan Linear Unit):
   ``gate(x) = (arctan(x) + pi/2) / pi``
   ``xATLU(x) = x * (gate(x) * (1 + 2*alpha) - alpha)``
   Uses arctan-based gating expanded by trainable alpha.

4. **xGELU** (Expanded Gaussian Error Linear Unit):
   ``gate(x) = 0.5 * (1 + erf(x / sqrt(2)))``
   ``xGELU(x) = x * (gate(x) * (1 + 2*alpha) - alpha)``
   Extends GELU gating with trainable alpha for adjustable range.

5. **xSiLU** (Expanded Sigmoid Linear Unit):
   ``gate(x) = sigmoid(x)``
   ``xSiLU(x) = x * (gate(x) * (1 + 2*alpha) - alpha)``
   Extends SiLU gating with trainable alpha.

6. **EluPlusOne**: ``ELU(x) + 1 + epsilon`` ensuring strictly positive outputs.

All activation classes inherit from a common ``BaseActivation`` layer, ensuring
they can be used seamlessly in Keras models. A ``get_activation`` factory
function provides name-based instantiation.

Reference:
    Huang, A. H. (2023). Expanded Gating Ranges Improve Activation Functions.
"""

import keras
import numpy as np
from typing import Optional, Union, Tuple, Dict, Any


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class BaseActivation(keras.layers.Layer):
    """
    Base class for all custom activation functions.

    This class provides a common interface and functionality for custom
    activation layers, including the standard Keras Layer configurations
    such as ``trainable``, ``dtype``, and ``name``. All activations in this
    module inherit from this base and apply element-wise transformations
    that preserve the input shape.

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────────────────────────┐
        │      Input x [..., features]        │
        └──────────────┬──────────────────────┘
                       │
                       ▼
        ┌─────────────────────────────────────┐
        │   Element-wise Activation f(x)      │
        │   (defined by subclass)             │
        └──────────────┬──────────────────────┘
                       │
                       ▼
        ┌─────────────────────────────────────┐
        │     Output f(x) [..., features]     │
        └─────────────────────────────────────┘

    :param trainable: Whether the layer's variables are trainable.
    :type trainable: bool
    :param name: Name of the layer.
    :type name: Optional[str]
    :param dtype: Data type for the layer's computations and variables.
    :type dtype: Optional[Union[str, keras.ops.dtype]]
    :param kwargs: Additional keyword arguments passed to the parent class.
    """

    def __init__(
        self,
        trainable: bool = True,
        name: Optional[str] = None,
        dtype: Optional[Union[str, keras.ops.dtype]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(
            trainable=trainable,
            name=name,
            dtype=dtype,
            **kwargs
        )

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute output shape -- element-wise activation preserves shape.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple, identical to input_shape.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return the configuration of the layer for serialization.

        :return: Dictionary containing the layer configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        return config


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class GELU(BaseActivation):
    """
    Gaussian Error Linear Unit (GELU) activation function.

    The GELU activation combines linear and Gaussian distributions to provide
    smooth, non-monotonic activation that often yields better performance than
    ReLU-based activations. It is defined as:
    ``GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))``

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────────────────────────┐
        │         Input x [...]               │
        └──────────────┬──────────────────────┘
                       │
                       ▼
        ┌─────────────────────────────────────┐
        │  GELU(x) = x * 0.5 * (1 + erf(x/  │
        │                          sqrt(2)))  │
        └──────────────┬──────────────────────┘
                       │
                       ▼
        ┌─────────────────────────────────────┐
        │         Output [...]                │
        └─────────────────────────────────────┘

    References:
        Hendrycks, D., & Gimpel, K. (2016). Gaussian Error Linear Units (GELUs).
    """

    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """
        Apply the GELU activation function to the inputs.

        :param inputs: The input tensor.
        :type inputs: keras.KerasTensor
        :return: Output tensor after applying the GELU activation.
        :rtype: keras.KerasTensor
        """
        return 0.5 * inputs * (1 + keras.ops.erf(inputs / keras.ops.sqrt(2.0)))


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class SiLU(BaseActivation):
    """
    Sigmoid Linear Unit (SiLU) activation function.

    Also known as Swish, SiLU blends linear and sigmoid behaviors for smoother
    gradients compared to piecewise functions like ReLU. It is defined as:
    ``SiLU(x) = x * sigmoid(x)``

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────────────────────────┐
        │         Input x [...]               │
        └──────────────┬──────────────────────┘
                       │
                       ▼
        ┌─────────────────────────────────────┐
        │  SiLU(x) = x * sigmoid(x)          │
        └──────────────┬──────────────────────┘
                       │
                       ▼
        ┌─────────────────────────────────────┐
        │         Output [...]                │
        └─────────────────────────────────────┘
    """

    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """
        Apply the SiLU activation function to the inputs.

        :param inputs: The input tensor.
        :type inputs: keras.KerasTensor
        :return: Output tensor after applying the SiLU activation.
        :rtype: keras.KerasTensor
        """
        return inputs * keras.ops.sigmoid(inputs)


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class ExpandedActivation(BaseActivation):
    """
    Base class for expanded gating range activation functions.

    This class introduces a trainable scalar parameter ``alpha`` that
    modifies the gating range of the activation function. Child classes
    use this parameter to expand or contract the range through which
    the gating function (e.g., sigmoid, erf, arctan) operates. The
    expanded activation formula is:
    ``f(x) = x * (gate(x) * (1 + 2*alpha) - alpha)``

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────────────────────────┐
        │         Input x [...]               │
        └──────────────┬──────────────────────┘
                       │
                       ├───────────────────┐
                       │                   │
                       ▼                   ▼
        ┌──────────────────────┐  ┌────────────────────────┐
        │   Identity: x        │  │ Expanded Gate:          │
        │                      │  │ g(x)*(1+2*alpha)-alpha  │
        └──────────┬───────────┘  └───────┬────────────────┘
                   │                      │
                   └──────────┬───────────┘
                              │ element-wise multiply
                              ▼
               ┌──────────────────────────────┐
               │     Output [...]              │
               └──────────────────────────────┘

    :param alpha_initializer: Initializer for the alpha parameter.
    :type alpha_initializer: Union[str, keras.initializers.Initializer]
    :param alpha_regularizer: Regularizer for the alpha parameter.
    :type alpha_regularizer: Optional[keras.regularizers.Regularizer]
    :param alpha_constraint: Constraint for the alpha parameter.
    :type alpha_constraint: Optional[keras.constraints.Constraint]
    :param kwargs: Additional keyword arguments passed to the parent class.
    """

    def __init__(
        self,
        alpha_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
        alpha_regularizer: Optional[keras.regularizers.Regularizer] = None,
        alpha_constraint: Optional[keras.constraints.Constraint] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Store configuration parameters
        self.alpha_initializer = keras.initializers.get(alpha_initializer)
        self.alpha_regularizer = keras.regularizers.get(alpha_regularizer)
        self.alpha_constraint = keras.constraints.get(alpha_constraint)

        # Initialize weight attribute - created in build()
        self.alpha = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create the trainable parameter ``alpha`` for the expanded activation.

        :param input_shape: Shape of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        """
        self.alpha = self.add_weight(
            name='alpha',
            shape=(),
            initializer=self.alpha_initializer,
            regularizer=self.alpha_regularizer,
            constraint=self.alpha_constraint,
            trainable=True,
            dtype=self.dtype
        )
        super().build(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Return the configuration of the expanded activation layer.

        :return: Dictionary containing the layer configuration including
            alpha parameter configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'alpha_initializer': keras.initializers.serialize(self.alpha_initializer),
            'alpha_regularizer': keras.regularizers.serialize(self.alpha_regularizer),
            'alpha_constraint': keras.constraints.serialize(self.alpha_constraint),
        })
        return config


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class xATLU(ExpandedActivation):
    """
    Expanded ArcTan Linear Unit activation function.

    This variant uses an arctan-based gating function, further expanded
    by a trainable parameter ``alpha``. The activation is computed as:
    ``gate(x) = (arctan(x) + pi/2) / pi``
    ``xATLU(x) = x * (gate(x) * (1 + 2 * alpha) - alpha)``

    The arctan-based gating provides smoother transitions for moderate input
    values compared to sigmoid-based gates.

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────────────────────────┐
        │         Input x [...]               │
        └──────────────┬──────────────────────┘
                       │
                       ├───────────────────┐
                       │                   │
                       ▼                   ▼
        ┌──────────────────────┐  ┌────────────────────────┐
        │   Identity: x        │  │ Gate: (arctan(x)+pi/2) │
        │                      │  │       / pi              │
        │                      │  │ Expand: g*(1+2a) - a   │
        └──────────┬───────────┘  └───────┬────────────────┘
                   │                      │
                   └──────────┬───────────┘
                              │ multiply
                              ▼
               ┌──────────────────────────────┐
               │     Output [...]              │
               └──────────────────────────────┘

    :param alpha_initializer: Initializer for the alpha parameter.
    :type alpha_initializer: Union[str, keras.initializers.Initializer]
    :param alpha_regularizer: Regularizer for the alpha parameter.
    :type alpha_regularizer: Optional[keras.regularizers.Regularizer]
    :param alpha_constraint: Constraint for the alpha parameter.
    :type alpha_constraint: Optional[keras.constraints.Constraint]
    :param kwargs: Additional keyword arguments passed to the parent class.
    """

    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """
        Apply the xATLU activation function to the inputs.

        :param inputs: The input tensor.
        :type inputs: keras.KerasTensor
        :return: The output tensor after applying xATLU activation.
        :rtype: keras.KerasTensor
        """
        gate = (keras.ops.arctan(inputs) + np.pi / 2) / np.pi
        return inputs * (gate * (1 + 2 * self.alpha) - self.alpha)


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class xGELU(ExpandedActivation):
    """
    Expanded Gaussian Error Linear Unit (xGELU) activation function.

    This variant uses the GELU gating function, expanded by a trainable
    parameter ``alpha``. The activation is computed as:
    ``gate(x) = 0.5 * (1 + erf(x / sqrt(2)))``
    ``xGELU(x) = x * (gate(x) * (1 + 2 * alpha) - alpha)``

    Maintains the smooth, non-monotonic properties of GELU while adding
    adaptability through the learnable gating range parameter.

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────────────────────────┐
        │         Input x [...]               │
        └──────────────┬──────────────────────┘
                       │
                       ├───────────────────┐
                       │                   │
                       ▼                   ▼
        ┌──────────────────────┐  ┌────────────────────────┐
        │   Identity: x        │  │ Gate: 0.5*(1+erf(x/    │
        │                      │  │            sqrt(2)))    │
        │                      │  │ Expand: g*(1+2a) - a   │
        └──────────┬───────────┘  └───────┬────────────────┘
                   │                      │
                   └──────────┬───────────┘
                              │ multiply
                              ▼
               ┌──────────────────────────────┐
               │     Output [...]              │
               └──────────────────────────────┘

    :param alpha_initializer: Initializer for the alpha parameter.
    :type alpha_initializer: Union[str, keras.initializers.Initializer]
    :param alpha_regularizer: Regularizer for the alpha parameter.
    :type alpha_regularizer: Optional[keras.regularizers.Regularizer]
    :param alpha_constraint: Constraint for the alpha parameter.
    :type alpha_constraint: Optional[keras.constraints.Constraint]
    :param kwargs: Additional keyword arguments passed to the parent class.
    """

    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """
        Apply the xGELU activation function to the inputs.

        :param inputs: The input tensor.
        :type inputs: keras.KerasTensor
        :return: The output tensor after applying xGELU activation.
        :rtype: keras.KerasTensor
        """
        gate = 0.5 * (1 + keras.ops.erf(inputs / keras.ops.sqrt(2.0)))
        return inputs * (gate * (1 + 2 * self.alpha) - self.alpha)


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class xSiLU(ExpandedActivation):
    """
    Expanded Sigmoid Linear Unit (xSiLU) activation function.

    This variant uses the sigmoid gating function, expanded by a trainable
    parameter ``alpha``. The activation is computed as:
    ``gate(x) = sigmoid(x)``
    ``xSiLU(x) = x * (gate(x) * (1 + 2 * alpha) - alpha)``

    Combines the smoothness of SiLU with an adaptable gating range for
    task-specific activation characteristics.

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────────────────────────┐
        │         Input x [...]               │
        └──────────────┬──────────────────────┘
                       │
                       ├───────────────────┐
                       │                   │
                       ▼                   ▼
        ┌──────────────────────┐  ┌────────────────────────┐
        │   Identity: x        │  │ Gate: sigmoid(x)       │
        │                      │  │ Expand: g*(1+2a) - a   │
        └──────────┬───────────┘  └───────┬────────────────┘
                   │                      │
                   └──────────┬───────────┘
                              │ multiply
                              ▼
               ┌──────────────────────────────┐
               │     Output [...]              │
               └──────────────────────────────┘

    :param alpha_initializer: Initializer for the alpha parameter.
    :type alpha_initializer: Union[str, keras.initializers.Initializer]
    :param alpha_regularizer: Regularizer for the alpha parameter.
    :type alpha_regularizer: Optional[keras.regularizers.Regularizer]
    :param alpha_constraint: Constraint for the alpha parameter.
    :type alpha_constraint: Optional[keras.constraints.Constraint]
    :param kwargs: Additional keyword arguments passed to the parent class.
    """

    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """
        Apply the xSiLU activation function to the inputs.

        :param inputs: The input tensor.
        :type inputs: keras.KerasTensor
        :return: The output tensor after applying xSiLU activation.
        :rtype: keras.KerasTensor
        """
        gate = keras.ops.sigmoid(inputs)
        return inputs * (gate * (1 + 2 * self.alpha) - self.alpha)


# ---------------------------------------------------------------------


def elu_plus_one_plus_epsilon(x: keras.KerasTensor) -> keras.KerasTensor:
    """
    Enhanced ELU activation to ensure positive values for rate parameters.

    This activation ensures that the output is always positive and greater than
    a small epsilon value, which is important for numerical stability as the
    rate parameter (lambda) of an Exponential distribution must be positive.
    Mathematical form: ``ELU(x) + 1 + epsilon``

    :param x: Input tensor.
    :type x: keras.KerasTensor
    :return: Tensor with ELU activation plus one plus a small epsilon.
    :rtype: keras.KerasTensor
    """
    return keras.activations.elu(x) + 1.0 + keras.backend.epsilon()


@keras.saving.register_keras_serializable()
class EluPlusOne(BaseActivation):
    """
    Enhanced ELU activation layer to ensure positive values.

    This activation ensures that the output is always positive and greater than
    a small epsilon value, which is important for numerical stability when used
    as rate parameters in probability distributions. Mathematical form:
    ``ELU(x) + 1 + epsilon``

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────────────────────────┐
        │         Input x [...]               │
        └──────────────┬──────────────────────┘
                       │
                       ▼
        ┌─────────────────────────────────────┐
        │  ELU(x) + 1 + epsilon              │
        │  Guarantees output > 0              │
        └──────────────┬──────────────────────┘
                       │
                       ▼
        ┌─────────────────────────────────────┐
        │   Positive Output [...]             │
        └─────────────────────────────────────┘
    """

    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """
        Apply the ELU+1+epsilon activation function to the inputs.

        :param inputs: The input tensor.
        :type inputs: keras.KerasTensor
        :return: Output tensor after applying ELU+1+epsilon activation.
        :rtype: keras.KerasTensor
        """
        return elu_plus_one_plus_epsilon(inputs)


# ---------------------------------------------------------------------


def get_activation(activation_name: str) -> BaseActivation:
    """
    Factory function to create an activation layer by name.

    Supported activation names: ``'gelu'``, ``'silu'``, ``'xatlu'``,
    ``'xgelu'``, ``'xsilu'``, ``'elu_plus_one'``.

    :param activation_name: Name of the desired activation function (case-insensitive).
    :type activation_name: str
    :return: An instance of the specified activation class.
    :rtype: BaseActivation
    :raises ValueError: If ``activation_name`` is not recognized.
    """
    activations = {
        'gelu': GELU,
        'silu': SiLU,
        'xatlu': xATLU,
        'xgelu': xGELU,
        'xsilu': xSiLU,
        'elu_plus_one': EluPlusOne,
    }

    activation_class = activations.get(activation_name.lower().strip())
    if activation_class is None:
        raise ValueError(
            f"Unknown activation: '{activation_name}'. "
            f"Available activations: {list(activations.keys())}"
        )

    return activation_class()
