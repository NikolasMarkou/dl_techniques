"""
Swish activation function, used as a non-linear basis.

This layer implements the Swish activation function, ``f(x) = x * sigmoid(x)``,
which serves as a smooth, non-monotonic "basis function" to enhance the
expressive power of a neural network. Its primary architectural advantage
is its self-gating mechanism, where the function uses a transformation of
the input itself to modulate its own output. This property often leads to
improved performance and better gradient flow compared to activations like
ReLU.

The "self-gating" property is central to its design. The sigmoid of the
input acts as a soft, continuous gate. For strongly positive inputs (``x ->
inf``), the gate ``sigmoid(x)`` approaches 1, making the function behave like
the identity (``f(x) ~ x``). For strongly negative inputs (``x -> -inf``),
the gate approaches 0, suppressing the output (``f(x) ~ 0``). This provides a
smooth interpolation between a linear and a zeroing function, avoiding the
abrupt switch and "dying neuron" problem associated with ReLU's hard gate.

The function is formally defined as:
``f(x) = x * sigma(x) = x / (1 + exp(-x))``

where ``sigma`` is the standard logistic sigmoid function. The function
exhibits several key properties that contribute to its effectiveness:

-   **Smoothness**: The function is infinitely differentiable, which
    benefits gradient-based optimization by providing a more stable and
    consistent gradient landscape compared to the non-differentiable
    point of ReLU at x=0.
-   **Non-Monotonicity**: Unlike most common activations, Swish is not
    monotonic. It exhibits a slight dip for negative values before
    asymptotically approaching zero. This characteristic may increase
    the expressive capacity of the model by allowing it to capture more
    complex data patterns.
-   **Unbounded Above, Bounded Below**: The function is unbounded for
    positive inputs, preventing gradient saturation that can occur in
    saturating functions like sigmoid or tanh. It is bounded below,
    which can contribute to network regularization.

References:
    - Ramachandran, P., Zoph, B., & Le, Q. V. (2017). "Searching for
      Activation Functions."
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
class BasisFunction(keras.layers.Layer):
    """
    Basis function layer implementing the Swish activation: ``b(x) = x / (1 + e^(-x))``.

    This layer implements the basis function branch of PowerMLP, which enhances
    the expressiveness of neural networks by capturing complex non-linear
    relationships. The function is mathematically equivalent to the Swish
    activation function (``x * sigmoid(x)``) and provides smooth, differentiable
    transformations that help with gradient flow during training.

    The self-gating mechanism allows the network to dynamically adjust the
    activation strength based on input values, improving expressiveness and
    gradient flow compared to traditional activations like ReLU. The function
    is infinitely differentiable (C^inf), non-monotonic with a slight dip for
    negative values, unbounded above to prevent gradient saturation, and bounded
    below with output >= -0.278 (minimum at x ~ -1.278).

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────────────────────────┐
        │   Input x [..., features]           │
        └──────────────┬──────────────────────┘
                       │
                       ├───────────────────┐
                       │                   │
                       ▼                   ▼
        ┌──────────────────────┐  ┌────────────────┐
        │  Identity Branch: x  │  │ Gate: sigmoid(x)│
        └──────────┬───────────┘  └───────┬────────┘
                   │                      │
                   └──────────┬───────────┘
                              │ element-wise multiply
                              ▼
               ┌──────────────────────────────┐
               │  output = x * sigmoid(x)     │
               │         = x / (1 + e^(-x))   │
               └──────────────┬───────────────┘
                              │
                              ▼
               ┌──────────────────────────────┐
               │  Output [..., features]       │
               └──────────────────────────────┘

    :param kwargs: Additional keyword arguments passed to the Layer parent class,
        such as ``name``, ``dtype``, ``trainable``, etc.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the BasisFunction activation layer.

        :param kwargs: Additional keyword arguments for the Layer parent class,
            including ``name``, ``dtype``, ``trainable``, etc.
        """
        super().__init__(**kwargs)
        logger.info(f"Initialized BasisFunction layer: {self.name}")

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply the basis function activation: ``b(x) = x / (1 + e^(-x))``.

        This operation is fully differentiable and supports gradient computation
        for backpropagation. The function is applied element-wise to the input.

        :param inputs: Input tensor of any shape. Values can be any real number.
        :type inputs: keras.KerasTensor
        :param training: Boolean indicating whether the layer should behave in
            training mode or inference mode. Not used in this stateless
            activation layer, but kept for API consistency with other layers.
        :type training: Optional[bool]
        :return: Output tensor with the same shape as inputs, after applying
            the basis function transformation.
        :rtype: keras.KerasTensor
        """
        # Compute b(x) = x / (1 + e^(-x))
        # This is mathematically equivalent to x * sigmoid(x)
        # Using the division form can be more numerically stable
        return inputs / (1.0 + ops.exp(-inputs))

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple, identical to input_shape.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Get the layer configuration for serialization.

        :return: Dictionary containing the layer configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        return config

    def __repr__(self) -> str:
        """
        Return string representation of the layer.

        :return: String representation including class name and instance name.
        :rtype: str
        """
        return f"BasisFunction(name='{self.name}')"

# ---------------------------------------------------------------------
