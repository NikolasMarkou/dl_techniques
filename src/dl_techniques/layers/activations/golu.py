"""
A Gompertz Linear Unit (GoLU) activation function.

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
    """Gompertz Linear Unit (GoLU) activation function layer.

    GoLU is an element-wise, self-gated activation function that leverages the
    Gompertz function to gate the input tensor. The function is defined as
    ``GoLU(x) = x * alpha * exp(-beta * exp(-gamma * x))``, where the default
    values of ``alpha=1.0``, ``beta=1.0``, ``gamma=1.0`` correspond to the
    standard Gumbel distribution CDF as the gate.

    **Architecture Overview:**

    .. code-block:: text

        Input: x (batch, ..., features)
                │
                ├──────────────────┐
                │                  │
                ▼                  ▼
        ┌──────────────┐   ┌──────────────────────────────┐
        │   Identity   │   │ Gompertz(x) =                │
        │      x       │   │  alpha * exp(-beta*exp(-γ*x)) │
        └──────┬───────┘   └──────────────┬───────────────┘
               │                          │
               └────────┬─────────────────┘
                        │
                        ▼
                ┌───────────────┐
                │   x * G(x)   │
                └───────┬───────┘
                        │
                        ▼
        Output: (batch, ..., features)

    :param alpha: Controls the upper asymptote or scale of the gate.
    :type alpha: float
    :param beta: Controls the gate displacement along the input-axis.
    :type beta: float
    :param gamma: Controls the growth rate of the gate.
    :type gamma: float
    :param kwargs: Additional arguments for the ``Layer`` base class (e.g., ``name``).
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        **kwargs: Any
    ) -> None:
        """Initialize the GoLU layer and store its configuration.

        :param alpha: Upper asymptote or scale of the Gompertz gate.
        :type alpha: float
        :param beta: Displacement of the gate along the input axis.
        :type beta: float
        :param gamma: Growth rate of the gate.
        :type gamma: float
        :param kwargs: Additional arguments for the ``Layer`` base class.
        """
        super().__init__(**kwargs)

        # Store all configuration parameters. This is crucial for serialization.
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Apply the GoLU activation: ``x * alpha * exp(-beta * exp(-gamma * x))``.

        :param inputs: Input tensor of any shape.
        :type inputs: keras.KerasTensor
        :return: Activated tensor with the same shape as inputs.
        :rtype: keras.KerasTensor
        """
        # Gompertz(x) = alpha * exp(-beta * exp(-gamma * x))
        gompertz_gate = self.alpha * ops.exp(
            -self.beta * ops.exp(-self.gamma * inputs)
        )
        # GoLU(x) = x * Gompertz(x)
        return inputs * gompertz_gate

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        """Return the output shape, which is identical to the input shape.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple
        :return: Output shape tuple, identical to input_shape.
        :rtype: tuple
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration of the layer for serialization.

        :return: Dictionary containing the layer configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
        })
        return config

# ---------------------------------------------------------------------
