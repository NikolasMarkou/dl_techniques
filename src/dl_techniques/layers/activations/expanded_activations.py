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
can be used seamlessly in Keras models. For instance:

    model.add(keras.layers.Dense(64))
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
from typing import Optional, Union, Tuple, Dict, Any


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="DLTechniques")
class BaseActivation(keras.layers.Layer):
    """
    Base class for all custom activation functions.

    This class provides a common interface and functionality for custom
    activation layers, including the standard Keras Layer configurations
    such as `trainable`, `dtype`, and `name`.

    Parameters
    ----------
    trainable : bool, optional
        Whether the layer's variables are trainable, by default True
    name : Optional[str], optional
        Name of the layer, by default None
    dtype : Optional[Union[str, keras.DType]], optional
        Data type for the layer's computations and variables, by default None
    **kwargs : Any
        Additional keyword arguments passed to the parent class

    Examples
    --------
    >>> activation = BaseActivation()
    >>> inputs = keras.Input(shape=(32,))
    >>> outputs = activation(inputs)
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

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the configuration of the layer.

        This method is used to save and restore the layer's state during
        model serialization.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the layer configuration
        """
        config = super().get_config()
        return config


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="DLTechniques")
class GELU(BaseActivation):
    """
    Gaussian Error Linear Unit (GELU) activation function.

    The GELU activation is defined as:
        GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))

    This activation function combines linear and Gaussian distributions to
    provide smooth, non-monotonic activation that often yields better
    performance than ReLU-based activations.

    Examples
    --------
    >>> gelu = GELU()
    >>> inputs = keras.Input(shape=(32,))
    >>> outputs = gelu(inputs)

    References
    ----------
    Hendrycks, D., & Gimpel, K. (2016). Gaussian Error Linear Units (GELUs).
    """

    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """
        Apply the GELU activation function to the inputs.

        Parameters
        ----------
        inputs : keras.KerasTensor
            The input tensor

        Returns
        -------
        keras.KerasTensor
            Output tensor after applying the GELU activation
        """
        return 0.5 * inputs * (1 + keras.ops.erf(inputs / keras.ops.sqrt(2.0)))


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="DLTechniques")
class SiLU(BaseActivation):
    """
    Sigmoid Linear Unit (SiLU) activation function.

    The SiLU activation is defined as:
        SiLU(x) = x * sigmoid(x)

    Also known as Swish, this activation blends linear and sigmoid behaviors
    for smoother gradients compared to piecewise functions like ReLU.

    Examples
    --------
    >>> silu = SiLU()
    >>> inputs = keras.Input(shape=(32,))
    >>> outputs = silu(inputs)
    """

    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """
        Apply the SiLU activation function to the inputs.

        Parameters
        ----------
        inputs : keras.KerasTensor
            The input tensor

        Returns
        -------
        keras.KerasTensor
            Output tensor after applying the SiLU activation
        """
        return inputs * keras.ops.sigmoid(inputs)


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="DLTechniques")
class ExpandedActivation(BaseActivation):
    """
    Base class for expanded gating range activation functions.

    This class introduces a trainable scalar parameter `alpha` that
    modifies the gating range of the activation function. Child classes
    use this parameter to expand or contract the range through which
    the gating function (e.g., sigmoid, erf, arctan) operates.

    Parameters
    ----------
    alpha_initializer : Union[str, keras.initializers.Initializer], optional
        Initializer for the alpha parameter, by default 'zeros'
    alpha_regularizer : Optional[keras.regularizers.Regularizer], optional
        Regularizer for the alpha parameter, by default None
    alpha_constraint : Optional[keras.constraints.Constraint], optional
        Constraint for the alpha parameter, by default None
    **kwargs : Any
        Additional keyword arguments passed to the parent class

    Attributes
    ----------
    alpha : keras.Variable
        Trainable parameter that modifies the gating range

    Examples
    --------
    >>> # This is a base class, use its subclasses instead
    >>> activation = xGELU()
    >>> inputs = keras.Input(shape=(32,))
    >>> outputs = activation(inputs)
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
        Create the trainable parameter `alpha` for the expanded activation.

        Parameters
        ----------
        input_shape : Union[keras.utils.SequenceLike, Tuple[Optional[int], ...]]
            Shape of the input tensor
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
        Returns the configuration of the expanded activation layer.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the layer configuration including
            alpha parameter configuration
        """
        config = super().get_config()
        config.update({
            'alpha_initializer': keras.initializers.serialize(self.alpha_initializer),
            'alpha_regularizer': keras.regularizers.serialize(self.alpha_regularizer),
            'alpha_constraint': keras.constraints.serialize(self.alpha_constraint),
        })
        return config


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="DLTechniques")
class xATLU(ExpandedActivation):
    """
    Expanded ArcTan Linear Unit activation function.

    This variant uses an arctan-based gating function, further expanded
    by a trainable parameter `alpha`.

    The activation is computed as:
        gate(x) = (arctan(x) + π/2) / π
        xATLU(x) = x * (gate(x) * (1 + 2 * alpha) - alpha)

    Parameters
    ----------
    alpha_initializer : Union[str, keras.initializers.Initializer], optional
        Initializer for the alpha parameter, by default 'zeros'
    alpha_regularizer : Optional[keras.regularizers.Regularizer], optional
        Regularizer for the alpha parameter, by default None
    alpha_constraint : Optional[keras.constraints.Constraint], optional
        Constraint for the alpha parameter, by default None
    **kwargs : Any
        Additional keyword arguments passed to the parent class

    Examples
    --------
    >>> xatlu = xATLU()
    >>> inputs = keras.Input(shape=(32,))
    >>> outputs = xatlu(inputs)

    >>> # With regularization
    >>> xatlu_reg = xATLU(alpha_regularizer=keras.regularizers.L2(1e-4))
    """

    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """
        Apply the xATLU activation function to the inputs.

        Parameters
        ----------
        inputs : keras.KerasTensor
            The input tensor

        Returns
        -------
        keras.KerasTensor
            The output tensor after applying xATLU activation
        """
        gate = (keras.ops.arctan(inputs) + np.pi / 2) / np.pi
        return inputs * (gate * (1 + 2 * self.alpha) - self.alpha)


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="DLTechniques")
class xGELU(ExpandedActivation):
    """
    Expanded Gaussian Error Linear Unit (xGELU) activation function.

    This variant uses the GELU gating function, expanded by a trainable
    parameter `alpha`.

    The activation is computed as:
        gate(x) = 0.5 * (1 + erf(x / sqrt(2)))
        xGELU(x) = x * (gate(x) * (1 + 2 * alpha) - alpha)

    Parameters
    ----------
    alpha_initializer : Union[str, keras.initializers.Initializer], optional
        Initializer for the alpha parameter, by default 'zeros'
    alpha_regularizer : Optional[keras.regularizers.Regularizer], optional
        Regularizer for the alpha parameter, by default None
    alpha_constraint : Optional[keras.constraints.Constraint], optional
        Constraint for the alpha parameter, by default None
    **kwargs : Any
        Additional keyword arguments passed to the parent class

    Examples
    --------
    >>> xgelu = xGELU()
    >>> inputs = keras.Input(shape=(32,))
    >>> outputs = xgelu(inputs)

    >>> # With custom initialization
    >>> xgelu_init = xGELU(alpha_initializer='uniform')
    """

    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """
        Apply the xGELU activation function to the inputs.

        Parameters
        ----------
        inputs : keras.KerasTensor
            The input tensor

        Returns
        -------
        keras.KerasTensor
            The output tensor after applying xGELU activation
        """
        gate = 0.5 * (1 + keras.ops.erf(inputs / keras.ops.sqrt(2.0)))
        return inputs * (gate * (1 + 2 * self.alpha) - self.alpha)


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="DLTechniques")
class xSiLU(ExpandedActivation):
    """
    Expanded Sigmoid Linear Unit (xSiLU) activation function.

    This variant uses the sigmoid gating function, expanded by a trainable
    parameter `alpha`.

    The activation is computed as:
        gate(x) = sigmoid(x)
        xSiLU(x) = x * (gate(x) * (1 + 2 * alpha) - alpha)

    Parameters
    ----------
    alpha_initializer : Union[str, keras.initializers.Initializer], optional
        Initializer for the alpha parameter, by default 'zeros'
    alpha_regularizer : Optional[keras.regularizers.Regularizer], optional
        Regularizer for the alpha parameter, by default None
    alpha_constraint : Optional[keras.constraints.Constraint], optional
        Constraint for the alpha parameter, by default None
    **kwargs : Any
        Additional keyword arguments passed to the parent class

    Examples
    --------
    >>> xsilu = xSiLU()
    >>> inputs = keras.Input(shape=(32,))
    >>> outputs = xsilu(inputs)

    >>> # With constraint
    >>> xsilu_const = xSiLU(alpha_constraint=keras.constraints.NonNeg())
    """

    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """
        Apply the xSiLU activation function to the inputs.

        Parameters
        ----------
        inputs : keras.KerasTensor
            The input tensor

        Returns
        -------
        keras.KerasTensor
            The output tensor after applying xSiLU activation
        """
        gate = keras.ops.sigmoid(inputs)
        return inputs * (gate * (1 + 2 * self.alpha) - self.alpha)


# ---------------------------------------------------------------------


def elu_plus_one_plus_epsilon(x: keras.KerasTensor) -> keras.KerasTensor:
    """
    Enhanced ELU activation to ensure positive values for rate parameters.

    This activation ensures that the output is always positive and greater than
    a small epsilon value, which is important for numerical stability as the
    rate parameter (λ) of an Exponential distribution must be positive.

    Mathematical form: ELU(x) + 1 + ε

    Parameters
    ----------
    x : keras.KerasTensor
        Input tensor

    Returns
    -------
    keras.KerasTensor
        Tensor with ELU activation plus one plus a small epsilon

    Examples
    --------
    >>> inputs = keras.ops.ones((4, 32))
    >>> outputs = elu_plus_one_plus_epsilon(inputs)
    """
    return keras.activations.elu(x) + 1.0 + keras.backend.epsilon()


@keras.saving.register_keras_serializable(package="DLTechniques")
class EluPlusOne(BaseActivation):
    """
    Enhanced ELU activation layer to ensure positive values.

    This activation ensures that the output is always positive and greater than
    a small epsilon value, which is important for numerical stability when used
    as rate parameters in probability distributions.

    Mathematical form: ELU(x) + 1 + ε

    Examples
    --------
    >>> elu_plus = EluPlusOne()
    >>> inputs = keras.Input(shape=(32,))
    >>> outputs = elu_plus(inputs)
    """

    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """
        Apply the ELU+1+ε activation function to the inputs.

        Parameters
        ----------
        inputs : keras.KerasTensor
            The input tensor

        Returns
        -------
        keras.KerasTensor
            Output tensor after applying ELU+1+ε activation
        """
        return elu_plus_one_plus_epsilon(inputs)


# ---------------------------------------------------------------------


def get_activation(activation_name: str) -> BaseActivation:
    """
    Factory function to create an activation layer by name.

    Supported activation names:
        - 'gelu': Gaussian Error Linear Unit
        - 'silu': Sigmoid Linear Unit (Swish)
        - 'xatlu': Expanded ArcTan Linear Unit
        - 'xgelu': Expanded Gaussian Error Linear Unit
        - 'xsilu': Expanded Sigmoid Linear Unit
        - 'elu_plus_one': Enhanced ELU ensuring positive outputs

    Parameters
    ----------
    activation_name : str
        Name of the desired activation function (case-insensitive)

    Returns
    -------
    BaseActivation
        An instance of the specified activation class

    Raises
    ------
    ValueError
        If `activation_name` is not recognized

    Examples
    --------
    >>> activation = get_activation("xgelu")
    >>> inputs = keras.Input(shape=(32,))
    >>> outputs = activation(inputs)

    >>> # Case insensitive
    >>> activation = get_activation("SILU")
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