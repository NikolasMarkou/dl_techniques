"""
Hard swish: Swish with the sigmoid gate replaced by a hard sigmoid.

Swish is ``x * sigmoid(x)``: the input gates itself. Hard swish keeps that
structure but swaps the sigmoid for its piecewise-linear stand-in, so the
whole activation is ``x * ReLU6(x + 3) / 6``. No exponential is computed.
MobileNetV3 uses it in place of Swish for that reason.

Written out, the function has three segments:

- ``h(x) = 0`` for ``x <= -3``
- ``h(x) = x * (x + 3) / 6`` for ``-3 < x < 3``
- ``h(x) = x`` for ``x >= 3``

So it is a flat region, a parabola, then the identity. The parabola dips
below zero, bottoming out at ``h(-1.5) = -0.375``. That dip is the
non-monotonic part of Swish that a plain ReLU does not have.

References:
    - Howard, A., et al. (2019). "Searching for MobileNetV3."
    - Ramachandran, P., Zoph, B., & Le, Q. V. (2017). "Searching for
      Activation Functions." (The original Swish.)

"""

import keras
from typing import Optional, Tuple, Dict, Any
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.activations.hard_swish")
class HardSwish(keras.layers.Layer):
    """Hard-swish activation: ``x * ReLU6(x + 3) / 6``.

    A cheap approximation of Swish for mobile and edge use. The input gates
    itself through a hard sigmoid, so only adds, a clamp, a divide and a
    multiply are needed. The function is non-monotonic (it dips to ``-0.375``
    at ``x = -1.5``) and unbounded above. Output shape equals input shape.

    The layer owns no weights. It wraps one ``keras.layers.ReLU`` with
    ``max_value=6.0``; it does not instantiate the ``HardSigmoid`` layer.

    **Architecture Overview:**

    .. code-block:: text

                          x  [B, ..., F]
                                 │
                 ┌───────────────┴───────────────┐
                 │                               │
                 ▼                               ▼
        ┌─────────────────┐             ┌─────────────────┐
        │ identity: x     │             │ ReLU6(x + 3) / 6│
        └────────┬────────┘             └────────┬────────┘
                 │  [B, ..., F]                  │  in [0, 1]
                 └───────────────┬───────────────┘
                                 │
                                 ▼
                   ┌───────────────────────────┐
                   │ x * hard_sigmoid(x)       │
                   └─────────────┬─────────────┘
                                 │
                                 ▼
                          y  [B, ..., F]

    The left branch is the tensor itself, not a weight or a sub-layer.

    :param kwargs: Arguments for the Layer base class (``name``,
        ``trainable``, ``dtype``, and so on).

    References:
        - Searching for MobileNetV3: https://arxiv.org/abs/1905.02244
        - Swish: A Self-Gated Activation Function: https://arxiv.org/abs/1710.05941
    """

    def __init__(self, **kwargs: Any) -> None:
        """Create the layer and its ReLU6 sub-layer.

        :param kwargs: Arguments for the Layer base class.
        """
        super().__init__(**kwargs)
        # Name the sub-layer explicitly. Without name=, Keras draws from a
        # process-global counter, so the same layer gets re_lu / re_lu_1 /
        # re_lu_2 across constructions and a reloaded model's sub-layer names
        # do not match the saved ones. ReLU is stateless, so nothing depends
        # on the name today; this keeps a stable serialized path anyway.
        self.activation = keras.layers.ReLU(max_value=6.0, name="relu6")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the wrapped ReLU6 sub-layer.

        :param input_shape: Shape of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        """
        self.activation.build(input_shape)
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply ``x * ReLU6(x + 3) / 6`` element-wise.

        :param inputs: Input tensor of any shape.
        :type inputs: keras.KerasTensor
        :param training: Training or inference mode. Unused here; the layer
            behaves the same either way. Kept for API consistency.
        :type training: Optional[bool]
        :return: Tensor of the same shape as ``inputs``.
        :rtype: keras.KerasTensor
        """
        hard_sigmoid_part = self.activation(inputs + 3.0) / 6.0
        return inputs * hard_sigmoid_part

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

# ---------------------------------------------------------------------
