"""
A powered ReLU activation function, ``f(x) = max(0, x)^k``.

This layer generalizes the standard Rectified Linear Unit (ReLU) by
introducing a power exponent ``k``, creating a higher-order non-linearity.
The primary architectural purpose is to allow a network to model more
complex polynomial relationships within a single activation, potentially
increasing the model's expressive power compared to the purely linear
positive region of a standard ReLU.

Architectural Overview:
    While standard ReLU (``k=1``) introduces a piecewise linear activation,
    ReLUK with ``k > 1`` introduces a piecewise polynomial activation. This
    changes the activation landscape significantly. For positive inputs,
    the function is no longer a simple linear pass-through but a convex
    polynomial curve. This allows a single neuron to learn a more complex,
    non-linear response to its inputs. The exponent ``k`` serves as a
    hyperparameter that controls the "aggressiveness" of this non-linearity;
    higher values of ``k`` create a function that is very flat near zero
    but grows extremely rapidly for larger inputs.

Mathematical Foundation:
    The function is defined as:
        ``f(x) = max(0, x)^k``

    The most critical difference from standard ReLU lies in its derivative,
    which directly impacts gradient flow during backpropagation. The
    derivative for positive inputs is:
        ``f'(x) = k * x^(k-1) for x > 0``

    Unlike standard ReLU, which has a constant gradient of 1 for all
    positive inputs, the gradient of ReLUK is dependent on the magnitude of
    the pre-activation ``x``. This has two major implications:
    1.  **Suppression of Small Signals**: For ``0 < x < 1``, the gradient
        ``k * x^(k-1)`` is less than ``k`` and can be smaller than 1 (for k>1),
        effectively dampening the gradient for small positive activations.
    2.  **Amplification of Large Signals**: For ``x > 1``, the gradient grows
        polynomially, strongly amplifying the gradient for large positive
        activations. This can accelerate learning for strong features but
        also carries a risk of exploding gradients.

    This input-dependent gradient scaling introduces a dynamic that can
    help the network focus on stronger signals, but requires careful
    initialization and potentially gradient clipping to ensure stability.

References:
    This function is a member of the broader class of polynomial-like
    activations explored as alternatives to linear rectifiers. Its
    conceptual roots lie in works that investigate the representation
    power of networks with higher-order activation functions.
    -   Ramachandran, P., Zoph, B., & Le, Q. V. (2017). "Searching for
        Activation Functions." (Exemplifies the search for non-linear
        activations beyond ReLU).
    -   Gouk, H., et al. (2021). "Regularisation of Neural Networks by
        Enforcing Lipschitz Continuity." (Discusses the role of activation
        functions in controlling network properties like Lipschitz
        constants).

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
    """ReLU-k activation layer implementing ``f(x) = max(0, x)^k``.

    This layer applies a powered ReLU activation function which provides
    more expressiveness than standard ReLU while maintaining computational
    efficiency. When ``k=1``, this reduces to the standard ReLU activation.
    Higher values of ``k`` create more aggressive non-linearities with
    input-dependent gradient scaling: small signals (``0 < x < 1``) are
    suppressed while large signals (``x > 1``) are amplified polynomially.

    **Architecture Overview:**

    .. code-block:: text

        Input: x (batch, ..., features)
                │
                ▼
        ┌───────────────────────┐
        │  ReLU: max(0, x)      │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  Power: result ^ k    │
        └───────────┬───────────┘
                    │
                    ▼
        Output: (batch, ..., features)

    :param k: Power exponent for the ReLU function. Must be a positive integer.
        Default is 3. When k=1, equivalent to standard ReLU.
    :type k: int
    :param kwargs: Additional keyword arguments passed to the Layer base class,
        such as ``name``, ``dtype``, ``trainable``, etc.

    :raises ValueError: If k is not a positive integer.
    :raises TypeError: If k is not an integer.

    References:
        The ReLU-k activation is inspired by various works on alternative
        activation functions that provide more flexibility than standard ReLU
        while maintaining computational efficiency.
    """

    def __init__(
            self,
            k: int = 3,
            **kwargs: Any
    ) -> None:
        """Initialize the ReLU-k activation layer.

        :param k: Power exponent for the ReLU function. Must be a positive integer.
            Default is 3.
        :type k: int
        :param kwargs: Additional keyword arguments for the Layer parent class.
        :raises ValueError: If k is not a positive integer.
        :raises TypeError: If k is not an integer.
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
        """Apply the ReLU-k activation: ``f(x) = max(0, x)^k``.

        :param inputs: Input tensor of any shape.
        :type inputs: keras.KerasTensor
        :param training: Boolean indicating training or inference mode. Not used
            in this activation layer but kept for API consistency.
        :type training: Optional[bool]
        :return: Output tensor with the same shape as inputs.
        :rtype: keras.KerasTensor
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

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple, identical to input_shape.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return the layer configuration for serialization.

        :return: Dictionary containing the layer configuration including
            the power parameter k and parent class configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "k": self.k,
        })
        return config

    def __repr__(self) -> str:
        """Return string representation of the layer.

        :return: String representation including the layer name and key parameters.
        :rtype: str
        """
        return f"ReLUK(k={self.k}, name='{self.name}')"


# ---------------------------------------------------------------------
