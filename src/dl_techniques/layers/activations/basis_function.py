"""
Swish activation function, used as a non-linear basis.

This layer implements the Swish activation function, `f(x) = x * sigmoid(x)`,
which serves as a smooth, non-monotonic "basis function" to enhance the
expressive power of a neural network. Its primary architectural advantage
is its self-gating mechanism, where the function uses a transformation of
the input itself to modulate its own output. This property often leads to
improved performance and better gradient flow compared to activations like
ReLU.

The "self-gating" property is central to its design. The sigmoid of the
input acts as a soft, continuous gate. For strongly positive inputs (`x ->
inf`), the gate `sigmoid(x)` approaches 1, making the function behave like
the identity (`f(x) ≈ x`). For strongly negative inputs (`x -> -inf`),
the gate approaches 0, suppressing the output (`f(x) ≈ 0`). This provides a
smooth interpolation between a linear and a zeroing function, avoiding the
abrupt switch and "dying neuron" problem associated with ReLU's hard gate.

Mathematical Foundation:
    The function is formally defined as:
        f(x) = x * σ(x) = x / (1 + exp(-x))

    where `σ` is the standard logistic sigmoid function. The function
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
    Basis function layer implementing the Swish activation: b(x) = x / (1 + e^(-x)).

    This layer implements the basis function branch of PowerMLP, which enhances
    the expressiveness of neural networks by capturing complex non-linear
    relationships. The function is mathematically equivalent to the Swish
    activation function (x * sigmoid(x)) and provides smooth, differentiable
    transformations that help with gradient flow during training.

    **Intent**: Provide a smooth, self-gated activation function that combines
    the benefits of linear and non-linear transformations. The self-gating
    mechanism allows the network to dynamically adjust the activation strength
    based on input values, improving expressiveness and gradient flow compared
    to traditional activations like ReLU.

    **Architecture**:
    ```
    Input(shape=[..., features])
           ↓
    Compute: sigmoid(x) = 1 / (1 + e^(-x))
           ↓
    Self-gate: output = x * sigmoid(x) = x / (1 + e^(-x))
           ↓
    Output(shape=[..., features])
    ```

    **Mathematical Operations**:
    1. **Sigmoid computation**: σ(x) = 1 / (1 + e^(-x))
    2. **Self-gating**: b(x) = x * σ(x) = x / (1 + e^(-x))

    The function has several desirable properties:
    - **Smooth**: Infinitely differentiable everywhere (C^∞)
    - **Non-monotonic**: Unlike ReLU, exhibits slight dip for negative values
    - **Unbounded above**: Prevents gradient saturation for positive inputs
    - **Bounded below**: Output ≥ -0.278 (minimum at x ≈ -1.278)
    - **Self-gating**: Function modulates its own output based on input magnitude

    Args:
        **kwargs: Additional keyword arguments passed to the Layer parent class,
            such as `name`, `dtype`, `trainable`, etc.

    Input shape:
        Arbitrary tensor with shape: `(batch_size, ..., features)`.

    Output shape:
        Same shape as the input: `(batch_size, ..., features)`.

    Attributes:
        None. This is a stateless activation layer with no trainable parameters.

    Example:
        >>> import keras
        >>> import numpy as np
        >>>
        >>> # Create basis function layer
        >>> basis_layer = BasisFunction(name='swish_activation')
        >>>
        >>> # Apply to some test data
        >>> x = keras.ops.convert_to_tensor(np.random.randn(32, 10))
        >>> output = basis_layer(x)
        >>> print(f"Input shape: {x.shape}, Output shape: {output.shape}")
        >>>
        >>> # Use in a model (PowerMLP architecture)
        >>> inputs = keras.Input(shape=(784,))
        >>> x = keras.layers.Dense(64)(inputs)
        >>> x = BasisFunction()(x)
        >>> x = keras.layers.Dense(32)(x)
        >>> x = BasisFunction()(x)
        >>> outputs = keras.layers.Dense(10, activation='softmax')(x)
        >>> model = keras.Model(inputs=inputs, outputs=outputs)
        >>>
        >>> # Sequential model example
        >>> model = keras.Sequential([
        ...     keras.layers.Dense(64, input_shape=(784,)),
        ...     BasisFunction(),
        ...     keras.layers.Dense(32),
        ...     BasisFunction(),
        ...     keras.layers.Dense(10, activation='softmax')
        ... ])

    References:
        - Ramachandran et al. "Searching for Activation Functions" (2017)
        - The basis function is equivalent to Swish: f(x) = x * sigmoid(x)
        - Used in PowerMLP architectures for enhanced expressiveness
        - Also known as SiLU (Sigmoid Linear Unit) in some literature

    Note:
        - This is a stateless layer with no trainable parameters
        - Computationally efficient due to vectorized operations
        - Provides better gradient flow than ReLU-based activations
        - The self-gating property makes it particularly effective in deep networks
        - No custom build() method needed as there are no weights or sub-layers
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the BasisFunction activation layer.

        This layer has no trainable parameters and requires no configuration
        beyond the standard Layer arguments (name, dtype, etc.).

        Args:
            **kwargs: Additional keyword arguments for the Layer parent class,
                including 'name', 'dtype', 'trainable', etc.
        """
        super().__init__(**kwargs)
        logger.info(f"Initialized BasisFunction layer: {self.name}")

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass of the basis function activation.

        Applies the basis function: b(x) = x / (1 + e^(-x)) = x * sigmoid(x)

        This operation is fully differentiable and supports gradient computation
        for backpropagation. The function is applied element-wise to the input.

        Args:
            inputs: Input tensor of any shape. Values can be any real number.
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode. Not used in this stateless
                activation layer, but kept for API consistency with other layers.

        Returns:
            Output tensor with the same shape as inputs, after applying
            the basis function transformation. Each element y[i] = x[i] / (1 + e^(-x[i])).
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

        For activation layers, the output shape is identical to the input shape
        as the transformation is applied element-wise without changing dimensions.

        Args:
            input_shape: Shape tuple of the input tensor. Can include None for
                dynamic dimensions (typically batch size).

        Returns:
            Output shape tuple, identical to input_shape.
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Get the layer configuration for serialization.

        Returns a dictionary containing the configuration needed to recreate
        this layer. Since BasisFunction has no custom parameters beyond those
        in the parent Layer class, this simply returns the parent configuration.

        Returns:
            Dictionary containing the layer configuration. Includes standard
            Layer parameters like 'name', 'dtype', 'trainable', etc.
        """
        config = super().get_config()
        return config

    def __repr__(self) -> str:
        """
        Return string representation of the layer.

        Provides a concise, readable representation useful for debugging
        and logging.

        Returns:
            String representation including the layer class name and instance name.
        """
        return f"BasisFunction(name='{self.name}')"

# ---------------------------------------------------------------------