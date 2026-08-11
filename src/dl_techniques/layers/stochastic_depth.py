"""
Stochastic Depth is a regularization method primarily used in very deep neural networks,
particularly those with residual connections (e.g., ResNets, Vision Transformers). Its
purpose is to improve training stability and generalization by randomly dropping
entire residual blocks (or paths) during training.

Key features and behavior of this `StochasticDepth` implementation:

1.  **Per-sample Dropping:** The drop decision is drawn independently for each
    *sample* in the batch. The noise shape is `(batch_size, 1, ..., 1)`, so every
    batch element gets its own Bernoulli draw, broadcast across that sample's
    remaining dimensions. These are the same semantics as DropPath in timm. The
    sibling `StochasticGradient` layer (`stochastic_gradient.py`) is the batch-wide
    one: it draws `keras.random.uniform(shape=[])`, a single scalar shared by the
    whole batch.

2.  **During Training (`training=True`):**
    - Each sample's residual path is zeroed with probability `drop_path_rate`,
      effectively "dropping" or bypassing, for that sample only, the residual
      connection that this layer guards.
    - Samples whose path is not dropped are scaled by `1 / (1 - drop_path_rate)`.
      This scaling is crucial for maintaining the expected magnitude of activations
      across dropped paths, ensuring that the expected output during training matches
      the output during inference.

3.  **During Inference (`training=False`):**
    - The layer acts as an identity function; the input tensor is passed through
      unchanged. No paths are dropped, and no scaling is applied, as the scaling factor
      from training ensures the expected output magnitude is preserved.

4.  **Dynamic Noise Shape:**
    The noise shape is calculated dynamically from the input rank as
    `(batch_size, 1, ..., 1)`, so the mask is constant across all spatial or feature
    dimensions of a given sample. This is what makes the "drop" an all-or-nothing
    decision for that sample's entire path, while leaving the decision independent
    between samples.

By randomly dropping residual paths, Stochastic Depth helps mitigate the vanishing
gradient problem in very deep networks, reduces co-adaptation between layers, and
encourages individual blocks to learn more robust features.

Reference:
-   "Deep Networks with Stochastic Depth" by Gao Huang et al. (https://arxiv.org/abs/1603.09382)
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