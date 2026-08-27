"""
ReLU raised to a power: ``f(x) = max(0, x)^k``.

Standard ReLU passes positive values through unchanged. This one raises them
to an integer power ``k``, so the positive half is a polynomial curve instead
of a straight line. ``k=1`` gives back plain ReLU. The exponent is a
hyperparameter, not a learned weight.

The interesting part is the gradient. For ``x > 0`` it is ``k * x^(k-1)``,
which depends on how large ``x`` is. Plain ReLU has gradient 1 everywhere on
the positive side. Two consequences with ``k > 1``:

- Small activations are damped. At ``0 < x < 1`` the factor ``x^(k-1)``
  shrinks the gradient.
- Large activations are amplified. At ``x > 1`` the gradient grows
  polynomially with ``x``.

That amplification is the failure mode to watch: with ``k=3`` an activation
of 10 already carries a gradient factor of 300. Use gradient clipping or a
careful initialization if you raise ``k``.

References:
    -   Ramachandran, P., Zoph, B., & Le, Q. V. (2017). "Searching for
        Activation Functions."
    -   Gouk, H., et al. (2021). "Regularisation of Neural Networks by
        Enforcing Lipschitz Continuity." (On how the activation controls
        network properties such as the Lipschitz constant.)

"""

import keras
from typing import Optional, Any, Dict, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class ReLUK(keras.layers.Layer):
    """ReLU-k activation: ``f(x) = max(0, x)^k``.

    Applies a ReLU, then raises the result to the integer power ``k``. With
    ``k=1`` the power step is skipped and this is plain ReLU. With ``k > 1``
    the positive half becomes a polynomial curve, which damps small
    activations and amplifies large ones. Output shape equals input shape and
    every output is non-negative.

    The layer owns no weights. ``k`` is validated once in ``__init__`` and
    stored, so an invalid value fails at construction, not at the first call.

    **Architecture Overview:**

    .. code-block:: text

               x  [B, ..., F]
                      │
                      ▼
        ┌───────────────────────────┐
        │ ReLU: max(0.0, x)         │
        └─────────────┬─────────────┘
                      │
              ┌───────┴───────┐
              │ k == 1        │ k > 1
              ▼               ▼
        ┌───────────┐   ┌───────────┐
        │ relu_out  │   │ relu_out^k│
        └─────┬─────┘   └─────┬─────┘
              └───────┬───────┘
                      │
                      ▼
          y  [B, ..., F]  >= 0

    The fork is decided at construction time from ``self.k``, not per batch.

    :param k: Power exponent. Must be a positive integer. Default is 3.
    :type k: int
    :param kwargs: Arguments for the Layer base class (``name``, ``dtype``,
        ``trainable``, and so on).

    :raises TypeError: If ``k`` is not an ``int``.
    :raises ValueError: If ``k`` is an ``int`` but not positive.
    """

    def __init__(
            self,
            k: int = 3,
            **kwargs: Any
    ) -> None:
        """Validate ``k`` and store it.

        :param k: Power exponent. Must be a positive integer. Default is 3.
        :type k: int
        :param kwargs: Arguments for the Layer base class.
        :raises TypeError: If ``k`` is not an ``int``.
        :raises ValueError: If ``k`` is an ``int`` but not positive.
        """
        super().__init__(**kwargs)

        if not isinstance(k, int):
            raise TypeError(f"k must be an integer, got type {type(k).__name__}")
        if k <= 0:
            raise ValueError(f"k must be a positive integer, got {k}")

        self.k = k

        logger.info(f"Initialized ReLUK layer with k={k}")

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply ``max(0, x)^k`` element-wise.

        :param inputs: Input tensor of any shape.
        :type inputs: keras.KerasTensor
        :param training: Training or inference mode. Unused here; the layer
            behaves the same either way. Kept for API consistency.
        :type training: Optional[bool]
        :return: Tensor of the same shape as ``inputs``, all values >= 0.
        :rtype: keras.KerasTensor
        """
        relu_output = keras.ops.maximum(0.0, inputs)

        # k == 1 is plain ReLU, so skip the power op entirely.
        if self.k == 1:
            return relu_output
        else:
            # power() wants a float exponent, hence the cast.
            return keras.ops.power(relu_output, float(self.k))

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Return the output shape, which equals the input shape.

        :param input_shape: Shape of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: The same shape tuple, unchanged.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return the config needed to rebuild the layer.

        :return: The base Layer config plus ``k``.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "k": self.k,
        })
        return config

    def __repr__(self) -> str:
        """Return a short representation showing ``k`` and the layer name.

        :return: A string such as ``ReLUK(k=3, name='relu_k')``.
        :rtype: str
        """
        return f"ReLUK(k={self.k}, name='{self.name}')"


# ---------------------------------------------------------------------
