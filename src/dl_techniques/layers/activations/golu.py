"""
The Gompertz Linear Unit (GoLU), a self-gated activation whose gate is
deliberately asymmetric.

Modern activations are gates, not thresholds. ReLU makes a hard binary decision at
zero; the self-gated family instead multiplies the input by a smooth function of
itself, ``x * g(x)``, so the network attenuates rather than truncates. The choice of
``g`` is where the design lives, and there is a pattern to the popular ones: Swish
uses the logistic CDF, GELU the Gaussian CDF. Both are the cumulative distribution
of a SYMMETRIC distribution, so both gates are symmetric about their midpoint.

GoLU asks what changes when that symmetry is dropped. Its gate is the Gompertz
function ``exp(-exp(-x))``, which is the CDF of the standard Gumbel distribution —
a right-skewed distribution, so the resulting S-curve is right-skewed too. The
consequence that matters is local: an asymmetric CDF has a gentler slope near the
origin than a symmetric one of comparable range, which is precisely the region most
pre-activations occupy.

Three effects are hypothesized to follow from that gentler slope. A less steep gate
makes the output less sensitive to small input perturbations, so activation variance
shrinks and less noise propagates forward. Smaller gate derivatives smooth the loss
surface, mitigating the sharp variations that trap optimizers and biasing
convergence toward flatter minima, which generalize better. And by not rewarding
sharp responses to individual features, the gate spreads learned weight mass more
broadly, acting as an implicit regularizer. These are the paper's claims about
training dynamics, not properties provable from the formula.

For large negative inputs the gate approaches zero and the unit is effectively
pruned; for large positive inputs it approaches ``alpha`` and the signal passes
through near-linearly. The three parameters generalize the standard function —
``alpha`` sets the upper asymptote, ``beta`` the displacement, ``gamma`` the growth
rate — and all default to 1.0, which recovers the plain Gumbel CDF. They are stored
constants, not trainable weights: this layer creates no variables.

Foundational mathematics::

    GoLU(x)     = x * Gompertz(x)
    Gompertz(x) = alpha * exp(-beta * exp(-gamma * x))

At ``alpha = beta = gamma = 1`` the gate is exactly the standard Gumbel CDF.

References:
    - Anonymous, 2025. GoLU: Gompertz Linear Units. (the activation this
      implements; asymmetry as the mechanism for reduced activation variance)
    - Hendrycks and Gimpel, 2016. Gaussian Error Linear Units (GELUs). (the
      Gaussian-CDF gate GoLU contrasts with)
      (https://arxiv.org/abs/1606.08415)
    - Ramachandran et al., 2017. Searching for Activation Functions. (Swish, the
      logistic-CDF gate) (https://arxiv.org/abs/1710.05941)
    - Gompertz, 1825. On the Nature of the Function Expressive of the Law of Human
      Mortality. Philosophical Transactions of the Royal Society 115.
    - Keskar et al., 2017. On Large-Batch Training for Deep Learning:
      Generalization Gap and Sharp Minima. (the flat-minima argument the smoother
      landscape claim rests on) (https://arxiv.org/abs/1609.04836)
"""

import keras
from typing import Any, Dict

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class GoLU(keras.layers.Layer):
    """
    Gompertz Linear Unit: element-wise self-gating with a right-skewed gate.

    Multiplies the input by a Gompertz function of itself,
    ``GoLU(x) = x * alpha * exp(-beta * exp(-gamma * x))``. At the defaults
    ``alpha = beta = gamma = 1.0`` the gate is the standard Gumbel distribution's
    CDF — asymmetric, unlike Swish's logistic CDF or GELU's Gaussian CDF, and
    therefore gentler in slope near the origin.

    Stateless: the three parameters are stored constants, so this layer creates no
    weights and needs no ``build``.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────────┐
        │  Input x [batch, …, features]        │
        └───────────────┬──────────────────────┘
                        ├──────────────────────────────┐
                        ▼                              │
        ┌──────────────────────────────────────┐       │
        │  Gompertz(x)                         │       │  identity
        │    = alpha · exp(−beta · exp(−γ·x))  │       │
        │                                      │       │
        │    x → −∞ :  gate → 0   (pruned)     │       │
        │    x → +∞ :  gate → alpha (linear)   │       │
        │    the CDF of a Gumbel — RIGHT-      │       │
        │    SKEWED, so the slope near 0 is    │       │
        │    gentler than a symmetric gate's   │       │
        └───────────────┬──────────────────────┘       │
                        ▼                              │
                       (×)◄─────────────────────────---┘
                        │
                        ▼
        ┌──────────────────────────────────────┐
        │  Output [batch, …, features]         │
        └──────────────────────────────────────┘

    **Gate family:**

    .. code-block:: text

        activation   gate g(x)          underlying distribution   symmetric?
        Swish        sigmoid(x)         logistic                  yes
        GELU         Phi(x)             Gaussian                  yes
        GoLU         exp(−exp(−x))      Gumbel                    NO — right-skew

        The asymmetry IS the design: it is what produces the shallower slope near
        the origin, and with it the reduced activation variance and smoother loss
        landscape the method is motivated by.

    :param alpha: Controls the upper asymptote or scale of the gate.
    :type alpha: float
    :param beta: Controls the gate displacement along the input-axis.
    :type beta: float
    :param gamma: Controls the growth rate of the gate.
    :type gamma: float
    :param kwargs: Additional arguments for the ``Layer`` base class (e.g., ``name``).

    Input shape:
        Arbitrary. The activation is element-wise, so any shape is accepted and no
        axis is privileged.

    Output shape:
        Same shape as the input.

    Example:
        >>> # Standard Gumbel-CDF gate
        >>> act = GoLU()
        >>> x = keras.random.normal((4, 128))
        >>> y = act(x)                                 # (4, 128)
        >>>
        >>> # As a layer in a stack
        >>> block = keras.Sequential([keras.layers.Dense(256), GoLU()])
        >>>
        >>> # Steeper gate, lower asymptote
        >>> act = GoLU(alpha=0.9, gamma=1.5)

    Note:
        ``alpha``, ``beta`` and ``gamma`` are fixed hyperparameters, not learned.
        A trainable variant would need them created as weights in ``build``, which
        this layer deliberately does not do.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        **kwargs: Any
    ) -> None:
        """
        Store the three gate parameters.

        No sub-layers and no weights are created: at the defaults this is exactly
        the Gumbel CDF gate, and every parameter is a constant folded into the
        expression in :meth:`call`.

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
        """
        Apply the GoLU activation: ``x * alpha * exp(-beta * exp(-gamma * x))``.

        The double exponential is evaluated as written; note that the inner
        ``exp(-gamma * x)`` grows without bound as ``x`` goes negative, so the
        outer exponential is what keeps the gate finite and drives it to zero
        there.

        :param inputs: Input tensor of any shape.
        :type inputs: keras.KerasTensor
        :return: Activated tensor with the same shape as inputs.
        :rtype: keras.KerasTensor
        """
        # Gompertz(x) = alpha * exp(-beta * exp(-gamma * x))
        gompertz_gate = self.alpha * keras.ops.exp(
            -self.beta * keras.ops.exp(-self.gamma * inputs)
        )
        # GoLU(x) = x * Gompertz(x)
        return inputs * gompertz_gate

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        """
        Return the output shape, which is identical to the input shape.

        The activation is element-wise, so nothing about the shape changes.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple
        :return: Output shape tuple, identical to input_shape.
        :rtype: tuple
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return the configuration of the layer for serialization.

        :return: Dictionary containing the layer configuration — every
            constructor argument, which for a stateless layer is its entire state.
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