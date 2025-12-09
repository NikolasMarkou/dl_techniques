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


@keras.utils.register_keras_serializable()
class SigmoidL2(keras.regularizers.Regularizer):
    """A regularizer combining L2 penalty with sigmoid saturation for diminishing returns.

    This regularizer provides:

    - **Positive l2**: Encourages smaller weight values (standard penalty behavior).
    - **Negative l2**: Encourages larger weight values (inverted penalty).
    - **Diminishing returns**: Uses exponential saturation so the marginal penalty
      decreases as weight magnitude increases.
    - **Bounded contributions**: Per-element contributions are bounded by min_val and max_val.

    The regularization loss is computed as:

    .. math::

        \\text{loss} = l2 \\cdot \\sum \\left[ \\text{min\\_val} +
        (\\text{max\\_val} - \\text{min\\_val}) \\cdot (1 - e^{-\\text{scale} \\cdot x^2}) \\right]

    The term :math:`1 - e^{-\\text{scale} \\cdot x^2}` provides sigmoid-like saturation:

    - At :math:`x = 0`: contribution is min_val
    - As :math:`|x| \\to \\infty`: contribution approaches max_val asymptotically

    Example usage:

    >>> layer = keras.layers.Dense(64, kernel_regularizer=SigmoidL2(l2=0.01))
    >>> layer = keras.layers.Dense(64, kernel_regularizer=SigmoidL2(l2=-0.01))  # Encourages larger weights

    :param l2: Regularization factor. Positive penalizes large weights,
        negative encourages large weights. Defaults to 0.01.
    :type l2: float
    :param scale: Controls how quickly the saturation occurs. Higher values
        cause faster saturation. Defaults to 1.0.
    :type scale: float
    :param min_val: Minimum per-element contribution (before l2 scaling).
        Defaults to 0.0.
    :type min_val: float
    :param max_val: Maximum per-element contribution (before l2 scaling).
        Defaults to 1.0.
    :type max_val: float
    """

    def __init__(
        self,
        l2: float = 0.01,
        scale: float = 1.0,
        min_val: float = 0.0,
        max_val: float = 1.0,
    ) -> None:
        """Initialize the SigmoidL2 regularizer.

        :param l2: Regularization factor.
        :param scale: Saturation rate control.
        :param min_val: Minimum per-element contribution.
        :param max_val: Maximum per-element contribution.
        :raises ValueError: If scale is not positive or min_val > max_val.
        """
        l2 = 0.01 if l2 is None else l2
        scale = 1.0 if scale is None else scale
        min_val = 0.0 if min_val is None else min_val
        max_val = 1.0 if max_val is None else max_val

        validate_float_arg(l2, name="l2")
        validate_float_arg(scale, name="scale")
        validate_float_arg(min_val, name="min_val")
        validate_float_arg(max_val, name="max_val")

        if scale <= 0:
            raise ValueError(
                f"scale must be positive. Received: scale={scale}"
            )
        if min_val > max_val:
            raise ValueError(
                f"min_val must be <= max_val. "
                f"Received: min_val={min_val}, max_val={max_val}"
            )

        self.l2 = l2
        self.scale = scale
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Compute the regularization loss for input weights.

        :param x: Weight tensor to regularize.
        :type x: keras.KerasTensor
        :return: Scalar regularization loss.
        :rtype: keras.KerasTensor
        """
        squared = keras.ops.square(x)

        # Exponential saturation: 0 at x=0, approaches 1 as |x| -> inf
        saturation = 1.0 - keras.ops.exp(-self.scale * squared)

        # Scale to [min_val, max_val] range
        per_element = self.min_val + (self.max_val - self.min_val) * saturation

        return self.l2 * keras.ops.sum(per_element)

    def get_config(self) -> dict:
        """Return the config of the regularizer.

        :return: Configuration dictionary.
        :rtype: dict
        """
        return {
            "l2": float(self.l2),
            "scale": float(self.scale),
            "min_val": float(self.min_val),
            "max_val": float(self.max_val),
        }

# ---------------------------------------------------------------------
