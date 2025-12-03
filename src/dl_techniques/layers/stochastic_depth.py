"""
Stochastic Depth is a regularization method primarily used in very deep neural networks,
particularly those with residual connections (e.g., ResNets, Vision Transformers). Its
purpose is to improve training stability and generalization by randomly dropping
entire residual blocks (or paths) during training.

Key features and behavior of this `StochasticDepth` implementation:

1.  **Batch-wise Dropping:** Unlike some implementations (e.g., DropPath in timm) that
    randomly drop paths independently for each *sample* in a batch, this layer
    implements "batch-wise" dropping. This means that if a residual path is
    dropped, it is dropped for *all* samples within the current training batch.
    This simplifies the implementation and aligns with the original paper's "per-batch"
    drop mode.

2.  **During Training (`training=True`):**
    - With a probability `drop_path_rate`, the layer outputs a tensor of zeros, effectively
      "dropping" or bypassing the residual connection that this layer guards.
    - If the path is not dropped, the input tensor is scaled by `1 / (1 - drop_path_rate)`.
      This scaling is crucial for maintaining the expected magnitude of activations
      across dropped paths, ensuring that the expected output during training matches
      the output during inference.

3.  **During Inference (`training=False`):**
    - The layer acts as an identity function; the input tensor is passed through
      unchanged. No paths are dropped, and no scaling is applied, as the scaling factor
      from training ensures the expected output magnitude is preserved.

4.  **Dynamic Noise Shape:**
    The noise shape is calculated dynamically based on input dimensions to ensure
    that the dropout mask is consistent across all spatial or feature dimensions
    of the input for a given sample, making the "drop" an all-or-nothing decision for
    the entire path.

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
    Implements Stochastic Depth for deep networks.

    This layer implements batch-wise dropping of residual paths as described in the
    Stochastic Depth paper. Unlike sample-wise dropping methods (like DropPath in timm),
    this implementation drops the same paths for all samples in a batch.

    The layer acts as an identity function during inference and randomly drops entire
    paths during training with the specified probability.

    Args:
        drop_path_rate: Float between 0 and 1, probability of dropping the residual path.
            Must be in the range [0, 1). Defaults to 0.5.
        **kwargs: Additional keyword arguments passed to the parent Layer class.

    Input shape:
        Arbitrary tensor with at least 2 dimensions (batch_size, ...).

    Output shape:
        Same shape as input.

    Raises:
        TypeError: If drop_path_rate is not a number.
        ValueError: If drop_path_rate is not in the range [0, 1).

    References:
        - Deep Networks with Stochastic Depth (https://arxiv.org/abs/1603.09382)
        - timm library implementation (https://github.com/rwightman/pytorch-image-models)

    Example:
        ```python
        # Basic usage in a residual block
        inputs = keras.Input(shape=(32, 32, 64))
        x = keras.layers.Conv2D(64, 3, padding='same')(inputs)
        x = StochasticDepth(drop_path_rate=0.2)(x)
        x = keras.layers.Add()([inputs, x])  # Residual connection

        # Higher drop rates for deeper layers
        stoch_depth = StochasticDepth(drop_path_rate=0.3)
        x = stoch_depth(x, training=True)  # Explicit training mode
        ```

    Note:
        This implementation uses dynamic noise shape calculation to handle inputs
        of varying dimensions without requiring build-time shape information.
    """

    def __init__(
        self,
        drop_path_rate: float = 0.5,
        **kwargs: Any
    ) -> None:
        """
        Initialize the StochasticDepth layer.

        Args:
            drop_path_rate: Probability of dropping the residual path.
                Must be in the range [0, 1).
            **kwargs: Additional keyword arguments passed to the parent Layer class.

        Raises:
            TypeError: If drop_path_rate is not a number.
            ValueError: If drop_path_rate is not in [0, 1).
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

        During training, randomly drops the entire path with probability drop_path_rate.
        During inference, acts as identity function.

        Args:
            inputs: Input tensor with shape (batch_size, ...).
            training: Boolean indicating whether in training mode. If None,
                uses the current training phase from the backend.

        Returns:
            Output tensor with same shape as input. During training, the tensor
            may be zeroed out with probability drop_path_rate. During inference,
            the tensor is returned unchanged.
        """
        # During inference, act as identity
        if training is False:
            return inputs

        # During training, apply stochastic depth
        if self.drop_path_rate == 0.0:
            return inputs

        # Calculate noise shape dynamically: (batch_size, 1, 1, ..., 1)
        # This ensures the same dropout decision for all spatial/feature dimensions
        input_shape = keras.ops.shape(inputs)
        batch_size = input_shape[0]
        remaining_dims = len(input_shape) - 1

        # Create noise shape for broadcasting
        noise_shape = [batch_size] + [1] * remaining_dims

        # Apply dropout with dynamic noise shape
        # We create a random mask and apply it manually for better control
        if training is not False:  # training=True or training=None
            # Generate random tensor with the noise shape
            random_tensor = keras.random.uniform(noise_shape)
            keep_prob = 1.0 - self.drop_path_rate

            # Create binary mask
            binary_mask = keras.ops.cast(random_tensor < keep_prob, inputs.dtype)

            # Scale and apply mask
            output = inputs * binary_mask / keep_prob

            return output
        else:
            return inputs

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape tuple of the input tensor.

        Returns:
            Tuple representing output shape (same as input shape).
        """
        return tuple(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Return the config dictionary for layer serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "drop_path_rate": self.drop_path_rate,
        })
        return config

# ---------------------------------------------------------------------