"""
Hard sigmoid: a piecewise-linear stand-in for the logistic sigmoid.

The logistic sigmoid ``1 / (1 + exp(-x))`` needs an exponential. This module
replaces it with an add, a clamp and a divide: ``ReLU6(x + 3) / 6``. There is
no transcendental call, so it is cheap on mobile and edge hardware and it
quantizes cleanly. MobileNetV3 uses it as the gate in its
squeeze-and-excitation blocks.

The result is three straight segments:

- ``h(x) = 0`` for ``x <= -3``
- ``h(x) = x / 6 + 0.5`` for ``-3 < x < 3``
- ``h(x) = 1`` for ``x >= 3``

The middle segment is a straight line through ``(0, 0.5)`` with slope
``1/6``. The true sigmoid has slope ``1/4`` at zero, so this is a shape
match, not a tangent line. It is close enough for gating, which is what a
sigmoid is usually doing in these blocks.

References:
    - Howard, A., et al. (2019). "Searching for MobileNetV3."
    - Courbariaux, M., et al. (2015). "BinaryConnect: Training Deep
      Neural Networks with binary weights during propagations." (An early
      version of the hard sigmoid.)

"""

import keras
from typing import Optional, Tuple, Dict, Any

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class HardSigmoid(keras.layers.Layer):
    """Hard-sigmoid activation: ``ReLU6(x + 3) / 6``.

    A piecewise-linear approximation of the logistic sigmoid that avoids the
    exponential. The output has the same shape as the input and lies in
    ``[0, 1]``. Use it wherever a sigmoid is only being used to squash a
    value into ``[0, 1]``, such as the gate in a squeeze-and-excitation block.

    The layer owns no weights. It wraps a ``keras.layers.ReLU`` with
    ``max_value=6.0`` and does the shift and the divide around it.

    **Architecture Overview:**

    .. code-block:: text

               x  [B, ..., F]
                      │
                      ▼
        ┌───────────────────────────┐
        │ x + 3.0                   │
        └─────────────┬─────────────┘
                      │  [B, ..., F]
                      ▼
        ┌───────────────────────────┐
        │ ReLU6: max(0, min(6, x))  │
        └─────────────┬─────────────┘
                      │  in [0, 6]
                      ▼
        ┌───────────────────────────┐
        │ divide by 6.0             │
        └─────────────┬─────────────┘
                      │
                      ▼
          y  [B, ..., F]  in [0, 1]

    The ReLU6 box is the wrapped sub-layer; the other two boxes are
    plain arithmetic in ``call()``.

    :param kwargs: Arguments for the Layer base class (``name``,
        ``trainable``, ``dtype``, and so on).

    References:
        - MobileNets: https://arxiv.org/abs/1704.04861
        - Squeeze-and-Excitation Networks: https://arxiv.org/abs/1709.01507
        - Searching for MobileNetV3: https://arxiv.org/abs/1905.02244
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
        """Apply ``ReLU6(x + 3) / 6`` element-wise.

        :param inputs: Input tensor of any shape.
        :type inputs: keras.KerasTensor
        :param training: Training or inference mode. Unused here; the layer
            behaves the same either way. Kept for API consistency.
        :type training: Optional[bool]
        :return: Tensor of the same shape as ``inputs``, with values in
            ``[0, 1]``.
        :rtype: keras.KerasTensor
        """
        return self.activation(inputs + 3.0) / 6.0

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
