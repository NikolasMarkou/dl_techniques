"""L2 penalty that also accepts a negative factor.

Provides :class:`L2_custom`, the standard L2 (weight decay / ridge) penalty
generalized so the factor may be negative, and :func:`validate_float_arg`, the
finite-float check it uses.

The penalty is ``lambda * ||w||^2``. The sign of ``lambda`` decides what it
does:

1. ``lambda > 0`` is ordinary regularization. The penalty grows with the
   squared weight magnitude, so the optimizer trades task error against weight
   size and decays the weights toward zero.
2. ``lambda < 0`` rewards larger weights. The optimizer is pushed to increase
   the squared L2 norm, driving weights away from the origin. This is
   destabilizing and is not for ordinary training. It is a research tool for
   probing network dynamics, optimizer stability, or objectives where parameter
   growth is wanted.
"""

import math
import keras
from keras import ops
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

def validate_float_arg(value, name):
    """Check that a penalty factor is a finite number and return it as a float.

    :param value: The value to check.
    :param name: Argument name, used in the error message.
    :type name: str
    :return: ``value`` coerced to ``float``.
    :rtype: float
    :raises ValueError: If ``value`` is not an ``int`` or ``float``, or is
        infinite or NaN.
    """
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

@register_dl_technique("dl_techniques.regularizers.l2_custom")
class L2_custom(keras.regularizers.Regularizer):
    """Apply an L2 penalty whose factor may be negative.

    The penalty is ``loss = l2 * reduce_sum(square(x))``. A positive ``l2``
    shrinks the weights; a negative one grows them.

    **Penalty shape:**

    .. code-block:: text

        loss
         ^                    l2 > 0
         |    \\             /
         |      \\         /          weights pulled toward 0
         |        \\_____/
         +------------0------------> w

        loss
         |        _______
         |      /        \\           l2 < 0
         |    /            \\         weights pushed away from 0
         +------------0------------> w

    Pass an instance, not a string: ``keras.regularizers.get`` resolves only
    the built-in names, so ``kernel_regularizer='L2_custom'`` raises
    ``ValueError``. The class is registered for serialization, so a model
    holding an instance saves and reloads normally.

    :param l2: L2 regularization factor. May be negative. ``None`` is treated
        as ``0.01``.
    :type l2: float or None

    :ivar l2: The validated factor.
    :vartype l2: float

    :raises ValueError: If ``l2`` is not a finite number.

    Example:
        >>> dense = keras.layers.Dense(3, kernel_regularizer=L2_custom(0.01))
        >>> grow = keras.layers.Dense(3, kernel_regularizer=L2_custom(-0.01))
    """

    def __init__(self, l2=0.01):
        """Validate and store the L2 factor.

        :param l2: L2 regularization factor; ``None`` means ``0.01``.
        :type l2: float or None
        :raises ValueError: If ``l2`` is not a finite number.
        """
        l2 = 0.01 if l2 is None else l2
        validate_float_arg(l2, name="l2")
        self.l2 = l2

    def __call__(self, x):
        """Compute the penalty for a weight tensor.

        :param x: The weight tensor.
        :type x: tensor
        :return: The scalar penalty ``l2 * sum(x**2)``.
        :rtype: tensor
        """
        return self.l2 * ops.sum(ops.square(x))

    def get_config(self):
        """Return the constructor arguments for serialization.

        :return: A dict holding ``l2``.
        :rtype: dict
        """
        return {"l2": float(self.l2)}


# ---------------------------------------------------------------------
