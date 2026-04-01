"""
This module implements the Stochastic Gradient regularization technique.

Stochastic Gradient is a regularization method that randomly blocks the gradient
flow during the backward pass. Unlike Stochastic Depth, which drops the entire
residual block in the forward pass, this layer only affects the gradient computation.
The forward pass remains an identity function at all times.

Key features and behavior of this `StochasticGradient` implementation:

1.  **Gradient Dropping during Training (`training=True`):**
    - With a probability `drop_path_rate`, the gradient is stopped from flowing
      backward through this layer. This is achieved by applying `keras.ops.stop_gradient`
      to the input tensor, effectively treating the layer's output as a constant
      with respect to its input for that training step.
    - If the gradient is not dropped, the layer acts as an identity function in both
      the forward and backward passes, allowing the gradient to flow through unchanged.

2.  **Identity Forward Pass:**
    - The layer always returns the input tensor unmodified during the forward pass,
      regardless of whether it's in training or inference mode. This ensures that the
      network's activations are not directly altered.

3.  **No-Op during Inference (`training=False`):**
    - During inference, the layer is an identity function and has no effect on the
      gradient, as gradients are not computed.

By randomly preventing gradient updates for certain paths, Stochastic Gradient can
encourage the network to learn more robust and less co-dependent features. It can be
seen as a way to regularize the training process by introducing noise into the
gradient updates.

Reference:
-   `keras.ops.stop_gradient` for the underlying mechanism.
"""

import keras
from typing import Optional, Dict, Any, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..utils.logger import logger


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class StochasticGradient(keras.layers.Layer):
    """
    Stochastic Gradient dropping regularization for deep networks.

    This layer stochastically stops gradient flow during backpropagation with
    probability ``drop_path_rate``. The forward pass is always an identity
    function -- unlike Stochastic Depth, only the backward pass is affected.
    During inference the layer has no effect.

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────────────────────┐
        │  Input [any shape]              │
        └──────────────┬──────────────────┘
                       ▼
        ┌─────────────────────────────────┐
        │  Forward: identity (always)     │
        │  Backward (training):           │
        │    p < keep_prob → pass grads   │
        │    p ≥ keep_prob → stop_gradient│
        └──────────────┬──────────────────┘
                       ▼
        ┌─────────────────────────────────┐
        │  Output [same shape as input]   │
        └─────────────────────────────────┘

    :param drop_path_rate: Probability of stopping the gradient. Must be in
        ``[0, 1)``. Defaults to 0.5.
    :type drop_path_rate: float
    :param kwargs: Additional keyword arguments for the parent Layer class.
    :type kwargs: Any
    """

    def __init__(
            self,
            drop_path_rate: float = 0.5,
            **kwargs: Any
    ) -> None:
        """
        Initialize the StochasticGradient layer.

        :param drop_path_rate: Probability of dropping the gradient. Must be in ``[0, 1)``.
        :type drop_path_rate: float
        :param kwargs: Additional keyword arguments for the parent Layer class.
        :type kwargs: Any
        """
        super().__init__(**kwargs)

        # Validate drop_path_rate
        if not isinstance(drop_path_rate, (int, float)):
            raise TypeError("drop_path_rate must be a number")
        if not 0.0 <= drop_path_rate < 1.0:
            raise ValueError(
                f"drop_path_rate must be in [0, 1), got {drop_path_rate}"
            )

        self.drop_path_rate = float(drop_path_rate)

        logger.info(
            f"Created StochasticGradient layer '{self.name}' with "
            f"drop_path_rate={self.drop_path_rate}"
        )

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass of the layer.

        :param inputs: Input tensor.
        :type inputs: keras.KerasTensor
        :param training: Whether in training mode.
        :type training: bool or None
        :return: Output tensor (same as input; gradient may be stopped during training).
        :rtype: keras.KerasTensor
        """
        if training is False or self.drop_path_rate == 0.0:
            return inputs

        # Determine whether to drop the gradient for this call
        keep_prob =1.0 - self.drop_path_rate

        random_tensor = keras.random.uniform(shape=[])

        # Use a conditional to apply stop_gradient.
        # This ensures that the graph is traceable by frameworks like TF.
        return keras.ops.cond(
            random_tensor < keep_prob,
            lambda: inputs,
            lambda: keras.ops.stop_gradient(inputs)
        )

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple
        :return: Output shape tuple (same as input shape).
        :rtype: tuple
        """
        return tuple(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Return the config dictionary for layer serialization.

        :return: Dictionary containing the layer configuration.
        :rtype: dict
        """
        config = super().get_config()
        config.update({
            "drop_path_rate": self.drop_path_rate,
        })
        return config

# ---------------------------------------------------------------------
