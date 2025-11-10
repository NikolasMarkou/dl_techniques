"""
A generalized L2 regularization penalty supporting weight decay or growth.

This regularizer implements the standard L2 penalty, also known as weight decay
or ridge regression, but generalizes it by allowing the regularization factor
to be negative. This seemingly simple change fundamentally alters the
regularizer's objective, transforming it from a mechanism for model
simplification into one that encourages weight expansion.

The mathematical formulation for the penalty is:
`Penalty = λ * ||w||²₂`
where `λ` is the regularization factor (`l2`) and `w` is the tensor of weights.

The conceptual underpinning depends on the sign of `λ`:
1.  **Standard Regularization (λ > 0)**: This is the conventional use case.
    The penalty term is positive and proportional to the squared magnitude of
    the weights. When added to the main loss function, the optimizer is
    incentivized to minimize both the task-specific error and the weight
    magnitudes. This discourages complex models with large weights that might
    overfit the training data, effectively "decaying" the weights toward zero.

2.  **Anti-Regularization (λ < 0)**: When the factor is negative, the penalty
    term becomes a reward for larger weights. To minimize the total loss, the
    optimizer is now encouraged to *increase* the squared L2 norm of the
    weights, pushing them away from the origin. This dynamic is inherently
    destabilizing and is not used for standard model training. Instead, it
    serves as a research tool for exploring network dynamics, studying the
    stability of optimization algorithms, or implementing unconventional
    learning objectives where parameter growth is explicitly desired.

By supporting both positive and negative factors, this regularizer provides a
unified framework for studying the impact of norm-based penalties and their
inverse on the optimization landscape.
"""

import math
import keras
from keras import ops

# ---------------------------------------------------------------------

def validate_float_arg(value, name):
    """check penalty number availability, raise ValueError if failed."""
    if (
        not isinstance(value, (float, int))
        or (math.isinf(value) or math.isnan(value))
    ):
        raise ValueError(
            f"Invalid value for argument {name}: expected a float."
            f"Received: {name}={value}"
        )
    return float(value)

# ---------------------------------------------------------------------

@keras.utils.register_keras_serializable()
class L2_custom(keras.regularizers.Regularizer):
    """A regularizer that applies a L2 regularization penalty but also allows negative l2 (forces the weights to increase)

    The L2 regularization penalty is computed as:
    `loss = l2 * reduce_sum(square(x))`

    L2 may be passed to a layer as a string identifier:

    >>> dense = Dense(3, kernel_regularizer='L2_custom')

    In this case, the default value used is `l2=0.01`.

    Arguments:
        l2: float, L2 regularization factor.
    """

    def __init__(self, l2=0.01):
        l2 = 0.01 if l2 is None else l2
        validate_float_arg(l2, name="l2")
        self.l2 = l2

    def __call__(self, x):
        return self.l2 * ops.sum(ops.square(x))

    def get_config(self):
        return {"l2": float(self.l2)}

# ---------------------------------------------------------------------
