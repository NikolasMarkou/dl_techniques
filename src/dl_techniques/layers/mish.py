import keras
import tensorflow as tf
from typing import Any, Dict

# ---------------------------------------------------------------------

@keras.utils.register_keras_serializable()
class Mish(keras.layers.Layer):
    """Mish activation function.

    Implementation of the Mish activation function from the paper
    "Mish: A Self Regularized Non-Monotonic Neural Activation Function"
    (https://arxiv.org/abs/1908.08681).

    The function is computed as: f(x) = x * tanh(softplus(x))

    Attributes:
        None
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the Mish activation layer.

        Args:
            **kwargs: Additional layer keywords arguments.
        """
        super().__init__(**kwargs)

    def call(self, inputs: tf.Tensor, training: bool = False, **kwargs: Any) -> tf.Tensor:
        """Forward pass computation.

        Args:
            inputs: Input tensor.
            training: Whether in training mode. Defaults to False.
            **kwargs: Additional keywords arguments.

        Returns:
            tf.Tensor: Activated tensor.
        """
        return inputs * tf.math.tanh(tf.math.softplus(inputs))

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration.

        Returns:
            Dict[str, Any]: Layer configuration dictionary.
        """
        return super().get_config()


# ---------------------------------------------------------------------

@keras.utils.register_keras_serializable()
class ScaledMish(keras.layers.Layer):
    """Scaled Mish activation function.

    A variant of Mish that smoothly saturates at ±alpha. The function is computed as:
    f(x) = α * tanh(mish(x)/α), where mish(x) = x * tanh(softplus(x))

    Attributes:
        alpha (float): Scaling factor that determines the saturation bounds.
    """

    def __init__(self, alpha: float = 2.0, **kwargs: Any) -> None:
        """Initialize the Scaled Mish activation layer.

        Args:
            alpha: Scaling factor for activation bounds. Defaults to 2.0.
            **kwargs: Additional layer keywords arguments.

        Raises:
            ValueError: If alpha is less than or equal to 0.
        """
        super().__init__(**kwargs)
        if alpha <= 0:
            raise ValueError(f"Alpha must be positive. Got {alpha}")
        self._alpha = alpha

    def call(self, inputs: tf.Tensor, training: bool = False, **kwargs: Any) -> tf.Tensor:
        """Forward pass computation.

        Args:
            inputs: Input tensor.
            training: Whether in training mode. Defaults to False.
            **kwargs: Additional keywords arguments.

        Returns:
            tf.Tensor: Activated tensor.
        """
        mish_value = inputs * tf.nn.tanh(tf.nn.softplus(inputs))
        scaled_mish = self._alpha * tf.nn.tanh(mish_value / self._alpha)
        return scaled_mish

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration.

        Returns:
            Dict[str, Any]: Layer configuration dictionary.
        """
        config = super().get_config()
        config.update({
            'alpha': self._alpha
        })
        return config

# ---------------------------------------------------------------------
