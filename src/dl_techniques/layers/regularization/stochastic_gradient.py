"""
StochasticGradient, a layer that randomly blocks gradient flow, not activations.

The forward pass is always the identity, in both training and inference, so
activations, normalization statistics, and model output are unaffected. Only
backpropagation is stochastic: each call draws one scalar `u ~ U(0, 1)`, and
with keep probability `p = 1 - drop_path_rate` the gradient passes through
unchanged (`u < p`), otherwise it is severed with `stop_gradient` (`u >= p`).
Because the decision is one scalar per call, not per sample, an entire path
is either live or dead for the whole batch. This differs from Stochastic
Depth, which also changes the forward activations and needs an inference-time
correction; this layer needs none, since its forward map never changes.

References:
    - Huang et al., 2016. Deep Networks with Stochastic Depth.
      (https://arxiv.org/abs/1603.09382)
    - Srivastava et al., 2014. Dropout: A Simple Way to Prevent Neural Networks
      from Overfitting. JMLR 15(56).
    - Larsson et al., 2017. FractalNet: Ultra-Deep Neural Networks without
      Residuals. (https://arxiv.org/abs/1605.07648)
"""

import keras
from typing import Optional, Dict, Any, Tuple

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique


@register_dl_technique("dl_techniques.layers.regularization.stochastic_gradient")
class StochasticGradient(keras.layers.Layer):
    """
    Stochastic Gradient dropping regularization for deep networks.

    This layer stochastically stops gradient flow during backpropagation with
    probability ``drop_path_rate``. The forward pass is always an identity
    function -- unlike Stochastic Depth, only the backward pass is affected.
    During inference the layer has no effect.

    Architecture:

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

        keep_prob =1.0 - self.drop_path_rate
        random_tensor = keras.random.uniform(shape=[])

        # keras.ops.cond keeps the branch traceable under graph/compiled execution.
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
