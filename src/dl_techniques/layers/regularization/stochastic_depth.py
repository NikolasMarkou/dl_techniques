"""
StochasticDepth, per-sample dropping of a residual branch during training.

In a residual network, each block adds a correction to an identity path, so
a block can be skipped without breaking the forward signal. This layer wraps
a residual branch and, during training, zeroes it for a random subset of
batch samples: `mask ~ Bernoulli(p)`, `y = x * mask / p` with `p = 1 -
drop_path_rate`. The draw is independent per sample and constant across that
sample's remaining dimensions, so a network trains on a different effective
depth each step. The `1/p` scaling keeps `E[y] = x`, so no correction is
needed at inference: the layer is then a pure identity. The sibling
`StochasticGradient` layer draws one scalar shared across the whole batch
instead of one draw per sample.

References:
    - Huang et al., 2016. Deep Networks with Stochastic Depth.
      (https://arxiv.org/abs/1603.09382)
    - Veit et al., 2016. Residual Networks Behave Like Ensembles of Relatively
      Shallow Networks. (https://arxiv.org/abs/1605.06431)
    - Srivastava et al., 2014. Dropout: A Simple Way to Prevent Neural Networks
      from Overfitting. JMLR 15(56).
"""

import keras
from typing import Optional, Dict, Any, Tuple

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique


@register_dl_technique("dl_techniques.layers.regularization.stochastic_depth")
class StochasticDepth(keras.layers.Layer):
    """
    Stochastic Depth regularization for deep networks.

    This layer implements per-sample dropping of residual paths. During
    training, each sample's path is independently dropped (zeroed) with
    probability ``drop_path_rate``, and surviving samples are scaled by
    ``1 / (1 - drop_path_rate)`` to maintain expected activation magnitude.
    During inference the layer acts as an identity function.

    Architecture:

    .. code-block:: text

        ┌─────────────────────────────────┐
        │  Input [B, ...]                 │
        └──────────────┬──────────────────┘
                       ▼
        ┌─────────────────────────────────┐
        │  Training:                      │
        │    mask ~ Bernoulli(keep_prob)  │
        │      shape (B, 1, ..., 1)       │
        │    output = input * mask / p    │
        │  Inference:                     │
        │    output = input (identity)    │
        └──────────────┬──────────────────┘
                       ▼
        ┌─────────────────────────────────┐
        │  Output [B, ...]                │
        └─────────────────────────────────┘

    :param drop_path_rate: Probability of dropping the residual path, drawn
        independently per sample. Must be in ``[0, 1)``. Defaults to 0.5.
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
        Initialize the StochasticDepth layer.

        :param drop_path_rate: Probability of dropping the residual path, drawn
            independently per sample. Must be in ``[0, 1)``.
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
            f"Created StochasticDepth layer '{self.name}' with "
            f"drop_path_rate={self.drop_path_rate}"
        )

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass of the layer.

        :param inputs: Input tensor with shape ``(batch_size, ...)``.
        :type inputs: keras.KerasTensor
        :param training: Whether in training mode. When training, each of the
            ``batch_size`` samples is dropped or kept by its own independent draw.
        :type training: bool or None
        :return: Output tensor with same shape as input.
        :rtype: keras.KerasTensor
        """
        if training is False or self.drop_path_rate == 0.0:
            return inputs

        # One draw per sample, broadcast across that sample's other dims.
        input_shape = keras.ops.shape(inputs)
        batch_size = input_shape[0]
        remaining_dims = len(input_shape) - 1
        noise_shape = [batch_size] + [1] * remaining_dims

        random_tensor = keras.random.uniform(noise_shape)
        keep_prob = 1.0 - self.drop_path_rate
        binary_mask = keras.ops.cast(random_tensor < keep_prob, inputs.dtype)
        output = (inputs * binary_mask) / keep_prob

        return output

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