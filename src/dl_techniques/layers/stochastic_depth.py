"""
Stochastic Depth regularization via per-sample residual path dropping.

This layer embodies the principle of implicit network ensembling, a design
paradigm that treats a single very deep network as a distribution over shallower
networks rather than as one fixed architecture. The core idea is that in a
residual network each block contributes an additive correction to an identity
path, so an individual block can be removed without breaking the forward
signal. Randomly removing blocks during training therefore samples a different
effective depth on every step, and the network learns a representation that no
single block is indispensable to.

Architecturally, the layer guards a residual branch and is applied to that
branch's output before it is added back to the shortcut. During training, a
Bernoulli decision is drawn and the branch is either kept or zeroed:

`mask ~ Bernoulli(p)` with `p = 1 - drop_path_rate`
`y = x * mask / p`

The decision is drawn independently per sample. The noise shape is
`(batch_size, 1, ..., 1)`, one draw per batch element broadcast across all of
that element's remaining dimensions, which makes the drop an all-or-nothing
event for a given sample's entire path while leaving different samples in the
batch on different effective depths. This matches the semantics of DropPath as
implemented in timm. The sibling `StochasticGradient` layer takes the batch-wide
alternative, drawing `keras.random.uniform(shape=[])`, a single scalar shared
across the whole batch.

The division by `p` is what makes the layer consistent between the two regimes.
Since the mask is Bernoulli with mean `p`, the inverted-dropout scaling gives
`E[y] = x`, so the expected activation magnitude entering downstream layers,
along with the running statistics accumulated by any normalization that follows,
matches what those layers will see when nothing is dropped. Consequently
inference requires no correction at all: the layer becomes a pure identity, the
full depth of the network is used, and the model's output is deterministic.

The distinction from a gradient-only regularizer is that this layer alters the
forward computation. Activations, normalization statistics, and effective depth
all vary from step to step, and the backward pass follows from that change
rather than being manipulated directly: a dropped branch has zero output and
therefore contributes no gradient to its own parameters, while the shortcut
still carries the gradient onward. This is precisely the mechanism that
shortens the effective backpropagation path, mitigating vanishing gradients in
networks too deep to train reliably at full depth, reducing co-adaptation
between adjacent blocks, and acting as a strong regularizer at essentially no
inference cost.

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

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class StochasticDepth(keras.layers.Layer):
    """
    Stochastic Depth regularization for deep networks.

    This layer implements per-sample dropping of residual paths. During
    training, each sample's path is independently dropped (zeroed) with
    probability ``drop_path_rate``, and surviving samples are scaled by
    ``1 / (1 - drop_path_rate)`` to maintain expected activation magnitude.
    During inference the layer acts as an identity function.

    **Architecture Overview:**

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

        # Validate drop_path_rate
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
        # During inference, act as identity
        if training is False or self.drop_path_rate == 0.0:
            return inputs

        # Calculate noise shape dynamically: (batch_size, 1, 1, ..., 1)
        # One independent draw per sample, held constant across that sample's
        # spatial/feature dimensions
        input_shape = keras.ops.shape(inputs)
        batch_size = input_shape[0]
        remaining_dims = len(input_shape) - 1

        # Create noise shape for broadcasting
        noise_shape = [batch_size] + [1] * remaining_dims

        # Apply dropout with dynamic noise shape
        # We create a random mask and apply it manually for better control
        # Generate random tensor with the noise shape
        random_tensor = keras.random.uniform(noise_shape)
        keep_prob = 1.0 - self.drop_path_rate

        # Create binary mask
        binary_mask = keras.ops.cast(random_tensor < keep_prob, inputs.dtype)

        # Scale and apply mask
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

# ---------------------------------------------------------------------