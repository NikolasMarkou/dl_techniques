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
class SaturatedMish(keras.layers.Layer):
    """ SaturatedMish activation function with continuous transition at alpha.

    The function behaves as follows:
    - For x <= alpha: f(x) = x * tanh(softplus(x)) (standard Mish)
    - For x > alpha: f(x) smoothly blends between the Mish value at alpha and
      a slightly higher asymptotic value, creating a continuous transition

    Attributes:
        alpha (float): The saturation threshold. Defaults to 3.0.
        beta (float): Controls the steepness of the transition. A smaller beta
                     makes the transition sharper, while a larger beta makes it
                     smoother. Defaults to 0.5.
    """

    def __init__(self, alpha: float = 3.0, beta: float = 0.5, **kwargs):
        """Initialize the SaturatedMish activation layer."""
        super().__init__(**kwargs)
        if alpha <= 0.0:
            raise ValueError("alpha must be greater than 0.")
        if beta <= 0.0:
            raise ValueError("beta must be greater than 0.")
        self.alpha = tf.convert_to_tensor(alpha, dtype=tf.float32)
        self.beta = tf.convert_to_tensor(beta, dtype=tf.float32)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Forward pass computation.

        Args:
            inputs (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Activated tensor with smooth saturation beyond alpha.
        """
        # Compute softplus
        softplus = tf.nn.softplus(inputs)

        # Standard Mish activation
        mish = inputs * tf.tanh(softplus)

        # Compute the Mish activation value at alpha (our saturation reference point)
        # This ensures continuity at the transition point
        mish_at_alpha = self.alpha * tf.tanh(tf.nn.softplus(self.alpha))

        # Create a smooth sigmoid-based blending factor
        sigmoid_blend = tf.sigmoid((inputs - self.alpha) / self.beta)

        # Combine both regions with smooth blending
        # For x <= alpha: mostly standard Mish
        # For x > alpha: gradually approach mish_at_alpha + small margin
        output = mish * (1 - sigmoid_blend) + mish_at_alpha * sigmoid_blend

        return output

    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'alpha': float(self.alpha),
            'beta': float(self.beta)
        })
        return config

# ---------------------------------------------------------------------
