"""
Implement the Gompertz Linear Unit (GoLU) activation function.

This layer provides a self-gated activation function based on the Gompertz
function. Unlike traditional monolithic activations (e.g., ReLU) or other
gated units (e.g., Swish, GELU), GoLU is designed to leverage the unique
asymmetrical properties of the Gompertz curve to enhance neural network
training dynamics and generalization.

Architectural Design and Rationale
----------------------------------
The core architecture of GoLU is a self-gating mechanism, where the input
tensor `x` is element-wise multiplied by a gating signal derived from `x`
itself:

    GoLU(x) = x * Gompertz(x)

This design allows the activation to dynamically modulate the input signal.
For large negative inputs, the gate approaches zero, effectively pruning
the neuron. For large positive inputs, the gate approaches one, allowing the
signal to pass through linearly. The transition between these states is
governed by the Gompertz function, which provides a non-linear, data-driven
attenuation of the input.

Foundational Mathematics and Intuition
--------------------------------------
The standard gating function used is the Gompertz function, defined as:

    Gompertz(x) = exp(-exp(-x))

This equation is mathematically significant as it represents the Cumulative
Distribution Function (CDF) of the Standard Gumbel distribution. This
connection is the conceptual foundation of GoLU.

The primary insight is the inherent asymmetry of the Gumbel distribution,
which imparts a subtle right-skew to the Gompertz S-shaped curve. This
contrasts sharply with the gating functions underlying other activations like
Swish (Sigmoid, CDF of Logistic distribution) and GELU (Gaussian CDF), which
are symmetric around a central point.

The practical consequence of this asymmetry is a reduced slope near the
origin compared to its symmetric counterparts. This property is hypothesized
to induce several beneficial effects:
1.  **Reduced Activation Variance**: The gentler slope makes the activation
    less sensitive to small perturbations in the input, leading to more
    stable latent representations and a reduction in noise propagation.
2.  **Smoother Loss Landscapes**: The corresponding smaller gradients help
    smooth the optimization landscape, mitigating sharp variations that can
    trap optimizers. This encourages convergence to flatter minima, which are
    often associated with improved model generalization.
3.  **Implicit Regularization**: By promoting a broader distribution of
    learned weights, GoLU may act as an implicit regularizer, preventing
    over-reliance on a small subset of features.

This implementation also includes `alpha`, `beta`, and `gamma` parameters,
which generalize the standard Gompertz function, allowing for fine-grained
control over the gate's upper asymptote, displacement, and growth rate,
respectively.
"""

import keras
from keras import ops
from typing import Any, Dict

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class GoLU(keras.layers.Layer):
    """
    Gompertz Linear Unit (GoLU) activation function layer.

    GoLU is an element-wise, self-gated activation function that leverages the
    Gompertz function to gate the input tensor. This implementation provides a
    Keras-native, serializable layer compatible with all Keras 3 backends
    (TensorFlow, PyTorch, JAX).

    **Intent**: To provide a robust and production-ready Keras implementation of
    the GoLU activation, following modern API design and serialization patterns.

    **Architecture**:
    The layer applies a non-linear transformation to the input tensor without
    changing its shape.
    ```
    Input(shape=[...])
           ↓
    GoLU Activation: x * Gompertz(x)
           ↓
    Output(shape=[...])
    ```

    **Mathematical Operation**:
        `GoLU(x) = x * Gompertz(x)`
    where:
        `Gompertz(x) = alpha * exp(-beta * exp(-gamma * x))`

    The parameters `alpha`, `beta`, and `gamma` control the shape of the Gompertz
    gate, allowing for fine-tuned activation behavior. The default values of 1.0
    for all parameters correspond to the standard Gumbel distribution CDF as the gate.

    Args:
        alpha (float, optional): Controls the upper asymptote or scale of the gate.
            Defaults to 1.0.
        beta (float, optional): Controls the gate displacement along the input-axis.
            Defaults to 1.0.
        gamma (float, optional): Controls the growth rate of the gate.
            Defaults to 1.0.
        **kwargs: Additional arguments for the `Layer` base class (e.g., `name`).

    Input shape:
        Arbitrary. The layer is applied element-wise and does not depend on
        the input shape.

    Output shape:
        Same shape as the input.

    Example:
        ```python
        # Use GoLU as a standalone activation layer in a model
        model = keras.Sequential([
            keras.layers.Input(shape=(784,)),
            keras.layers.Dense(128),
            GoLU(),
            keras.layers.Dense(10, activation='softmax')
        ])
        model.summary()

        # Use GoLU with custom parameters for specialized behavior
        custom_golu_layer = GoLU(alpha=0.8, beta=1.2, gamma=0.9)

        # GoLU can also be used as a string identifier after registration
        # This is useful when defining models from configurations.
        # keras.saving.get_custom_objects()['GoLU'] = GoLU
        # layer = keras.layers.Dense(64, activation='GoLU')
        ```

    Note:
        The original research suggests keeping `alpha`, `beta`, and `gamma` as positive
        values to maintain the characteristic S-shape of the Gompertz gate.
        This layer does not contain trainable weights; its parameters are treated as
        fixed hyperparameters set during initialization.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        **kwargs: Any
    ) -> None:
        """Initializes the GoLU layer and stores its configuration."""
        super().__init__(**kwargs)

        # Store all configuration parameters. This is crucial for serialization.
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """
        Forward pass computation for the GoLU activation.

        Applies the function: x * alpha * exp(-beta * exp(-gamma * x)).
        This operation is backend-agnostic thanks to `keras.ops`.
        """
        # Gompertz(x) = alpha * exp(-beta * exp(-gamma * x))
        gompertz_gate = self.alpha * ops.exp(
            -self.beta * ops.exp(-self.gamma * inputs)
        )
        # GoLU(x) = x * Gompertz(x)
        return inputs * gompertz_gate

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        """Returns the output shape, which is identical to the input shape."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the configuration of the layer for serialization.

        This method ensures that all initialization parameters are saved,
        allowing the layer to be perfectly reconstructed from its config.
        """
        config = super().get_config()
        config.update({
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
        })
        return config

# ---------------------------------------------------------------------
