import math
import keras
from keras import ops
from typing import Dict, Any, Union

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
