"""
Mish self-regularized, non-monotonic activation function.

This layer implements the Mish activation function, designed to improve the
performance and training stability of deep neural networks. It serves as a
smooth, continuously differentiable alternative to functions like ReLU and
Swish, combining their benefits while mitigating some of their drawbacks,
such as the dying ReLU problem and the less smooth gradient landscape.

Architectural Overview:
    Mish is architecturally a self-gated activation function, similar in
    principle to Swish (`x * sigmoid(x)`). The core idea is that the
    activation's output is a product of the input itself and a non-linear
    gating function derived from the input. In Mish, this gate is more
    complex than in Swish, aiming for a more favorable functional landscape.

    The function can be deconstructed into two main parts:
    1.  **Identity Path (`x`):** The input is passed through directly,
        preserving an unobstructed path for gradient flow, especially for
        large positive values.
    2.  **Gating Mechanism (`tanh(softplus(x))`):**
        -   `softplus(x)` acts as a smooth approximation of the ReLU
            function, mapping all inputs to positive values without the
            abrupt change at zero.
        -   `tanh(y)` takes this positive output and maps it to the range
            (0, 1), creating a smooth, saturating gate.

    The multiplication of these two components results in a function that
    is unbounded above (like ReLU) but smooth and non-monotonic (unlike
    ReLU), allowing it to capture more complex dependencies in the data.

Mathematical Foundation:
    The Mish activation function is formally defined as:
        f(x) = x * tanh(softplus(x))
    where `softplus(x) = log(1 + exp(x))`.

    This formulation gives rise to several key mathematical properties that
    contribute to its empirical success:
    -   **Unbounded Above**: As `x -> ∞`, `softplus(x) -> x` and
        `tanh(softplus(x)) -> 1`, so `f(x) ≈ x`. This prevents gradient
        saturation for positive inputs, similar to ReLU.
    -   **Bounded Below**: As `x -> -∞`, the function has a finite lower
        bound, which can contribute to regularization.
    -   **C-infinity Smoothness**: The function is infinitely
        differentiable, providing a very smooth gradient landscape that can
        stabilize and accelerate gradient-based optimization.
    -   **Non-Monotonicity**: Mish has a slight negative dip for small
        negative inputs before asymptotically approaching its lower bound.
        This property is believed to increase the expressiveness of the
        network.

References:
    - Misra, D. (2019). "Mish: A Self Regularized Non-Monotonic Neural
      Activation Function."

"""

import keras
import numpy as np
from keras import ops
from typing import Any, Dict, Optional, Tuple


# ---------------------------------------------------------------------
# Standalone activation functions
# ---------------------------------------------------------------------


def mish(inputs: keras.KerasTensor) -> keras.KerasTensor:
    """Compute the Mish activation function.

    Mish is defined as: f(x) = x * tanh(softplus(x))
    where softplus(x) = log(1 + exp(x))

    Args:
        inputs: Input tensor of any shape.

    Returns:
        Output tensor with same shape as inputs, after applying Mish activation.
    """
    # Calculate softplus: log(1 + exp(x))
    softplus = ops.softplus(inputs)
    # Calculate tanh of softplus
    tanh_softplus = ops.tanh(softplus)
    # Return x * tanh(softplus(x))
    return inputs * tanh_softplus


def saturated_mish(
        inputs: keras.KerasTensor,
        alpha: float = 3.0,
        beta: float = 0.5,
        mish_at_alpha: float = 1.0
) -> keras.KerasTensor:
    """Compute the Saturated Mish activation function.

    Args:
        inputs: Input tensor of any shape.
        alpha: Saturation threshold.
        beta: Controls transition steepness.
        mish_at_alpha: Pre-computed Mish value at alpha for efficiency.

    Returns:
        Output tensor with same shape as inputs, after applying Saturated Mish.
    """
    tmp_mish = mish(inputs)

    # Create a smooth sigmoid-based blending factor
    blend_factor = ops.sigmoid((inputs - alpha) / beta)

    # Combine both regions with smooth blending
    # For x <= alpha: mostly standard Mish
    # For x > alpha: gradually approach mish_at_alpha
    return tmp_mish * (1.0 - blend_factor) + mish_at_alpha * blend_factor


# ---------------------------------------------------------------------
# Keras layer implementations
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class Mish(keras.layers.Layer):
    """Mish activation function layer.

    Implementation of the Mish activation function from the paper "Mish: A Self
    Regularized Non-Monotonic Neural Activation Function" by Diganta Misra.

    The Mish function is a smooth, non-monotonic activation function that has
    shown improved performance over ReLU and other activation functions in many
    deep learning tasks. It combines the benefits of both ReLU-like and Swish-like
    activations.

    Mathematical formulation:
        f(x) = x * tanh(softplus(x))

    Where softplus(x) = log(1 + exp(x)) and tanh is the hyperbolic tangent function.

    Key properties:
        - Smooth and differentiable everywhere
        - Non-monotonic (has a small dip for negative values)  
        - Self-regularizing properties
        - Upper bounded by approximately x for large positive x
        - Lower bounded by approximately -0.31 for large negative x

    Args:
        **kwargs: Additional keyword arguments passed to the Layer base class,
            such as `name`, `dtype`, `trainable`, etc.

    Input shape:
        Arbitrary tensor of any shape.

    Output shape:
        Same shape as the input tensor.

    Example:
        ```python
        # Basic usage
        layer = Mish()
        inputs = keras.Input(shape=(64,))
        outputs = layer(inputs)

        # In a model
        model = keras.Sequential([
            keras.layers.Dense(128, input_shape=(784,)),
            Mish(),  # Apply Mish activation
            keras.layers.Dense(64),
            Mish(),  # Another Mish activation
            keras.layers.Dense(10, activation='softmax')
        ])

        # With custom name
        mish_layer = Mish(name='custom_mish')
        ```

    References:
        - Misra, Diganta. "Mish: A self regularized non-monotonic neural 
          activation function." arXiv preprint arXiv:1908.08681 (2019).
        - https://github.com/digantamisra98/Mish

    Note:
        - Mish has been shown to outperform ReLU, Leaky ReLU, and Swish in
          many computer vision_heads and NLP tasks
        - The function is computationally more expensive than ReLU but the
          performance gains often justify the additional cost
        - Works particularly well in deeper networks
        - Has self-regularizing properties that can help with generalization
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the Mish activation layer.

        Args:
            **kwargs: Additional keyword arguments for the Layer base class.
        """
        super().__init__(**kwargs)
        # This layer has no parameters to store

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass computation.

        Applies the Mish activation function: f(x) = x * tanh(softplus(x))

        Args:
            inputs: Input tensor of any shape.
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode. Not used in this activation
                layer but kept for API consistency.

        Returns:
            Tensor with the same shape as input after applying the Mish
            activation function.
        """
        return mish(inputs)

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
        """Get layer configuration for serialization.

        Returns all parameters passed to __init__ so the layer can be
        properly reconstructed during model loading.

        Returns:
            Dictionary containing the layer configuration.
        """
        return super().get_config()

    def __repr__(self) -> str:
        """Return string representation of the layer.

        Returns:
            String representation including the layer name.
        """
        return f"Mish(name='{self.name}')"


@keras.saving.register_keras_serializable()
class SaturatedMish(keras.layers.Layer):
    """SaturatedMish activation function with continuous transition at alpha.

    This is a variant of the Mish activation that includes a saturation mechanism
    to prevent extremely large activations. The function behaves like standard
    Mish for values below the threshold alpha, then smoothly transitions to a
    saturated region for larger values.

    The saturation helps with:
    - Preventing activation explosion in very deep networks
    - Improved numerical stability
    - Better gradient flow in certain architectures

    Mathematical formulation:
        For x <= α: f(x) ≈ x * tanh(softplus(x))  (standard Mish)
        For x > α:  f(x) smoothly blends toward a saturated value

    The transition is controlled by a sigmoid function for smoothness.

    Args:
        alpha: Float, the saturation threshold where the transition begins.
            Must be greater than 0. Values above this threshold will experience
            saturation. Defaults to 3.0.
        beta: Float, controls the steepness of the transition from standard Mish
            to saturated behavior. Smaller values create sharper transitions,
            larger values create smoother transitions. Must be greater than 0.
            Defaults to 0.5.
        **kwargs: Additional keyword arguments passed to the Layer base class,
            such as `name`, `dtype`, `trainable`, etc.

    Input shape:
        Arbitrary tensor of any shape.

    Output shape:
        Same shape as the input tensor.

    Attributes:
        alpha: The saturation threshold parameter.
        beta: The transition steepness parameter.
        mish_at_alpha: Pre-computed Mish value at alpha for efficiency.

    Example:
        ```python
        # Basic usage with default parameters
        layer = SaturatedMish()
        inputs = keras.Input(shape=(128,))
        outputs = layer(inputs)

        # Custom saturation threshold
        layer = SaturatedMish(alpha=5.0)  # Higher threshold

        # Custom transition steepness
        layer = SaturatedMish(alpha=3.0, beta=0.1)  # Sharper transition

        # In a deep network to prevent activation explosion
        model = keras.Sequential([
            keras.layers.Dense(512, input_shape=(784,)),
            SaturatedMish(alpha=4.0),  # Saturated Mish
            keras.layers.Dense(256),
            SaturatedMish(alpha=3.0),  # Different threshold
            keras.layers.Dense(10, activation='softmax')
        ])
        ```

    Raises:
        ValueError: If alpha or beta is not greater than 0.

    Note:
        - Useful in very deep networks where standard Mish might lead to
          activation explosion
        - The pre-computed mish_at_alpha value improves computational efficiency
        - The sigmoid-based transition ensures smoothness and differentiability
        - Choose alpha based on the typical range of your pre-activation values
    """

    def __init__(
            self,
            alpha: float = 3.0,
            beta: float = 0.5,
            **kwargs: Any
    ) -> None:
        """Initialize the SaturatedMish activation layer.

        Args:
            alpha: The saturation threshold. Must be greater than 0.
            beta: Controls the steepness of the transition. Must be greater than 0.
            **kwargs: Additional keyword arguments for the Layer base class.

        Raises:
            ValueError: If alpha or beta is not greater than 0.
        """
        super().__init__(**kwargs)

        # Validate parameters
        if alpha <= 0.0:
            raise ValueError(f"alpha must be greater than 0, got {alpha}")
        if beta <= 0.0:
            raise ValueError(f"beta must be greater than 0, got {beta}")

        # Store configuration parameters
        self.alpha = float(alpha)
        self.beta = float(beta)

        # Pre-compute mish value at alpha for efficiency
        # This is a constant that doesn't require gradients
        alpha_np = np.float32(self.alpha)
        softplus_alpha = np.log(1.0 + np.exp(alpha_np))
        tanh_softplus_alpha = np.tanh(softplus_alpha)
        self.mish_at_alpha = float(alpha_np * tanh_softplus_alpha)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass computation.

        Applies the SaturatedMish activation function with smooth transition.

        Args:
            inputs: Input tensor of any shape.
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode. Not used in this activation
                layer but kept for API consistency.

        Returns:
            Tensor with the same shape as input after applying the SaturatedMish
            activation function.
        """
        return saturated_mish(
            inputs,
            alpha=self.alpha,
            beta=self.beta,
            mish_at_alpha=self.mish_at_alpha
        )

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
        """Get layer configuration for serialization.

        Returns all parameters passed to __init__ so the layer can be
        properly reconstructed during model loading.

        Returns:
            Dictionary containing the layer configuration, including
            alpha and beta parameters along with parent class configuration.
        """
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'beta': self.beta
        })
        return config

    def __repr__(self) -> str:
        """Return string representation of the layer.

        Returns:
            String representation including the layer name and key parameters.
        """
        return f"SaturatedMish(alpha={self.alpha}, beta={self.beta}, name='{self.name}')"

# ---------------------------------------------------------------------