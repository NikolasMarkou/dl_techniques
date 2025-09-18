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
    Implements Stochastic Gradient dropping for deep networks.

    This layer stochastically stops the gradient flow during backpropagation based on
    the `drop_path_rate`. The forward pass is always an identity function.

    Args:
        drop_path_rate: Float between 0 and 1, the probability of stopping the
            gradient. Must be in the range [0, 1). Defaults to 0.5.
        **kwargs: Additional keyword arguments passed to the parent Layer class.

    Input shape:
        Arbitrary tensor.

    Output shape:
        Same shape as input.

    Raises:
        TypeError: If drop_path_rate is not a number.
        ValueError: If drop_path_rate is not in the range [0, 1).

    Example:
        ```python
        # Basic usage in a sequential model
        inputs = keras.Input(shape=(784,))
        x = keras.layers.Dense(128, activation='relu')(inputs)
        x = StochasticGradient(drop_path_rate=0.2)(x) # Stochastically drop gradient
        x = keras.layers.Dense(10, activation='softmax')(x)
        model = keras.Model(inputs, x)

        # In a residual connection
        res_inputs = keras.Input(shape=(32, 32, 64))
        res_x = keras.layers.Conv2D(64, 3, padding='same')(res_inputs)
        # Gradient from this path may be dropped
        res_x = StochasticGradient(drop_path_rate=0.3)(res_x)
        res_output = keras.layers.Add()([res_inputs, res_x])
        ```

    Note:
        This layer's effect is only present during training (`training=True`). During
        inference, it has no impact.
    """

    def __init__(
            self,
            drop_path_rate: float = 0.5,
            **kwargs: Any
    ) -> None:
        """
        Initialize the StochasticGradient layer.

        Args:
            drop_path_rate: Probability of dropping the gradient.
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

        Stops the gradient based on `drop_path_rate` during training.

        Args:
            inputs: Input tensor.
            training: Boolean indicating whether in training mode.

        Returns:
            The output tensor, which is the same as the input tensor. The gradient
            may be stopped during training.
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
