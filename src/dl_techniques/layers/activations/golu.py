"""
The Gompertz Linear Unit (GoLU), a self-gated activation with an off-centre gate.

Modern activations gate rather than threshold. ReLU makes a hard yes/no
decision at zero. The self-gated family instead multiplies the input by a
smooth function of itself, ``x * g(x)``, so the network attenuates instead of
truncating. The design choice is ``g``. Swish uses the logistic CDF, GELU the
Gaussian CDF. Both are the CDF of a symmetric distribution, so both gates pass
through 0.5 at ``x = 0``.

GoLU drops that symmetry. Its gate is the Gompertz function ``exp(-exp(-x))``,
the CDF of the standard Gumbel distribution, which is right-skewed. The direct
consequence is measurable: the gate reads 0.3679 at ``x = 0``, not 0.5. At the
origin the unit passes about a third of its input rather than half.

Measured gate slopes, all three of which peak at ``x = 0``:

- Gompertz ``exp(-exp(-x))``: slope 0.3679 at the origin.
- Logistic ``sigmoid(x)`` (Swish): slope 0.2500.
- Gaussian ``Phi(x)`` (GELU): slope 0.3989.

So GoLU's gate is steeper than Swish's at the origin and shallower than
GELU's. Do not describe it as "the gentler gate" -- that ordering is wrong
against Swish.

The paper argues the asymmetry reduces activation variance, smooths the loss
surface and acts as an implicit regularizer. Those are claims about training
dynamics from the paper, not properties you can read off the formula.

For large negative inputs the gate goes to zero and the unit is effectively
pruned. In float32 at the defaults, ``GoLU(-50.0)`` is exactly ``-0.0``, but
not because anything overflowed: the inner ``exp(-gamma*x)`` reaches
5.1847055e+21 there, which is finite, and the outer ``exp`` of that number
underflows to 0.0. The inner term itself only overflows to ``inf`` below about
``x = -88.723``, and the output has already been exactly ``-0.0`` since about
``x = -4.47``. For large positive inputs the gate approaches ``alpha`` and the
signal passes through near-linearly.

The three parameters generalize the standard function. ``alpha`` sets the
upper asymptote, ``beta`` the displacement, ``gamma`` the growth rate. All
default to 1.0, which recovers the plain Gumbel CDF. They are stored
constants, not trainable weights: this layer creates no variables.

Foundational mathematics::

    GoLU(x)     = x * Gompertz(x)
    Gompertz(x) = alpha * exp(-beta * exp(-gamma * x))

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
    ``alpha = beta = gamma = 1.0`` the gate is the standard Gumbel CDF. That
    gate is asymmetric, so it reads 0.3679 at ``x = 0`` rather than the 0.5 a
    symmetric gate gives. Output shape equals input shape.

    The layer owns no weights. The three parameters are stored constants, so
    there is no ``build``.

    **Architecture Overview:**

    .. code-block:: text

                             x  [B, ..., F]
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
        ┌───────────────────────┐       ┌───────────────────────┐
        │ identity: x           │       │ gate = Gompertz(x)    │
        └───────────┬───────────┘       └───────────┬───────────┘
                    │                               │ in (0, alpha)
                    └───────────────┬───────────────┘
                                    │  x * gate
                                    ▼
                             y  [B, ..., F]

    ``Gompertz(x) = alpha * exp(-beta * exp(-gamma * x))``. The left branch is
    the tensor itself, not a weight or a sub-layer. The stated ``(0, alpha)``
    range holds for positive ``alpha``, ``beta`` and ``gamma``.

    **Gate family, measured:**

    .. code-block:: text

        activation  gate g(x)       g(0)    slope at 0   distribution
        Swish       sigmoid(x)      0.5000  0.2500       logistic
        GELU        Phi(x)          0.5000  0.3989       Gaussian
        GoLU        exp(-exp(-x))   0.3679  0.3679       Gumbel

        All three gates reach their steepest point at x = 0. GoLU's is
        steeper than Swish's there and shallower than GELU's, so it is not
        "the gentle gate". What sets it apart is g(0): it is not 0.5, because
        a Gumbel is not symmetric.

    :param alpha: Upper asymptote, or scale, of the gate.
    :type alpha: float
    :param beta: Displacement of the gate along the input axis.
    :type beta: float
    :param gamma: Growth rate of the gate.
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
        Making them trainable would mean creating them as weights in ``build``,
        which this layer does not do.
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

        No sub-layers and no weights are created. Each parameter is a plain
        constant folded into the expression in :meth:`call`.

        :param alpha: Upper asymptote or scale of the Gompertz gate.
        :type alpha: float
        :param beta: Displacement of the gate along the input axis.
        :type beta: float
        :param gamma: Growth rate of the gate.
        :type gamma: float
        :param kwargs: Additional arguments for the ``Layer`` base class.
        """
        super().__init__(**kwargs)

        # get_config() reads these back, so all three must be stored.
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """
        Apply ``x * alpha * exp(-beta * exp(-gamma * x))`` element-wise.

        The double exponential is evaluated as written. The inner
        ``exp(-gamma * x)`` grows as ``x`` goes negative, and the outer
        exponential of that large number underflows to 0.0 long before the
        inner one overflows: in float32 the inner term is still finite
        (5.1847055e+21) at ``x = -50`` and only reaches ``inf`` below about
        ``x = -88.723``. Either way the gate is 0.0, never ``NaN``. Measured:
        ``GoLU(-50.0)`` and ``GoLU(-100.0)`` both return ``-0.0``.

        :param inputs: Input tensor of any shape.
        :type inputs: keras.KerasTensor
        :return: Tensor of the same shape as ``inputs``.
        :rtype: keras.KerasTensor
        """
        gompertz_gate = self.alpha * keras.ops.exp(
            -self.beta * keras.ops.exp(-self.gamma * inputs)
        )
        return inputs * gompertz_gate

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        """
        Return the output shape, which equals the input shape.

        The activation is element-wise, so nothing about the shape changes.

        :param input_shape: Shape of the input tensor.
        :type input_shape: tuple
        :return: The same shape tuple, unchanged.
        :rtype: tuple
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return the config needed to rebuild the layer.

        :return: The base Layer config plus ``alpha``, ``beta`` and ``gamma``.
            With no weights, that is the layer's entire state.
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