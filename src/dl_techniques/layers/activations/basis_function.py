"""
Swish, used as the smooth basis branch of PowerMLP.

The function is ``f(x) = x * sigmoid(x) = x / (1 + exp(-x))``. The input
gates itself: the sigmoid acts as a soft switch that opens towards 1 for
large positive inputs (so ``f(x) ~ x``) and closes towards 0 for large
negative ones (so ``f(x) ~ 0``). That is the same job ReLU's hard cutoff
does, done smoothly, so there is no kink at zero and no dead region.

Three properties matter in practice:

- **Smooth.** Differentiable everywhere, unlike ReLU at ``x = 0``.
- **Non-monotonic.** It dips below zero for negative inputs, bottoming out
  at ``f(-1.2785) = -0.2785``, then rises back towards 0.
- **Unbounded above, bounded below.** Positive inputs never saturate;
  outputs are never below ``-0.2785``.

``dl_techniques.layers.ffn.power_mlp_layer`` uses this layer as the
``BasisFunction -> Dense`` branch that runs alongside its ReLU-k branch.

References:
    - Ramachandran, P., Zoph, B., & Le, Q. V. (2017). "Searching for
      Activation Functions."
"""

import keras
from typing import Optional, Any, Dict, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.activations.basis_function")
class BasisFunction(keras.layers.Layer):
    """Swish activation, written as ``b(x) = x / (1 + exp(-x))``.

    This is the basis-function branch of PowerMLP. It is the same function as
    ``x * sigmoid(x)``, but ``call()`` computes the division form directly
    rather than calling a sigmoid op. Output shape equals input shape.

    The output is smooth everywhere, non-monotonic, unbounded above, and never
    below ``-0.2785`` (its minimum, at ``x = -1.2785``).

    The layer owns no weights.

    **Architecture Overview:**

    .. code-block:: text

                          x  [B, ..., F]
                                 │
                 ┌───────────────┴───────────────┐
                 │                               │
                 ▼                               ▼
        ┌─────────────────┐             ┌─────────────────┐
        │ numerator: x    │             │ 1.0 + exp(-x)   │
        └────────┬────────┘             └────────┬────────┘
                 │  [B, ..., F]                  │  >= 1.0
                 └───────────────┬───────────────┘
                                 │  divide
                                 ▼
                   ┌───────────────────────────┐
                   │ x / (1.0 + exp(-x))       │
                   └─────────────┬─────────────┘
                                 │
                                 ▼
                          y  [B, ..., F]

    The left branch is the tensor itself, not a weight or a sub-layer.

    Note:
        At very negative inputs ``exp(-x)`` overflows to ``inf`` in float32,
        and the division returns ``-0.0``. That is the correct limit, so the
        layer does not produce ``NaN`` there. Measured: ``b(-100.0) == -0.0``.

    :param kwargs: Arguments for the Layer base class (``name``, ``dtype``,
        ``trainable``, and so on).
    """

    def __init__(self, **kwargs: Any) -> None:
        """Create the layer. There is nothing to configure.

        :param kwargs: Arguments for the Layer base class (``name``,
            ``dtype``, ``trainable``, and so on).
        """
        super().__init__(**kwargs)
        logger.info(f"Initialized BasisFunction layer: {self.name}")

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply ``x / (1 + exp(-x))`` element-wise.

        :param inputs: Input tensor of any shape. Any real values are allowed.
        :type inputs: keras.KerasTensor
        :param training: Training or inference mode. Unused here; the layer
            behaves the same either way. Kept for API consistency.
        :type training: Optional[bool]
        :return: Tensor of the same shape as ``inputs``.
        :rtype: keras.KerasTensor
        """
        return inputs / (1.0 + keras.ops.exp(-inputs))

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

        :return: The base Layer config. This layer adds nothing to it.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        return config

    def __repr__(self) -> str:
        """Return a short representation showing the layer name.

        :return: A string such as ``BasisFunction(name='basis_function')``.
        :rtype: str
        """
        return f"BasisFunction(name='{self.name}')"

# ---------------------------------------------------------------------
