import keras
import tensorflow as tf


# ---------------------------------------------------------------------

@keras.utils.register_keras_serializable()
class Mish(keras.layers.Layer):
    """
    Mish: A Self Regularized Non-Monotonic Neural Activation Function
    https://arxiv.org/abs/1908.08681v1
    """

    def __init__(self, **kwargs):
        super().__init__(trainable=False, **kwargs)

    def call(self, inputs, training=False, **kwargs):
        return inputs * tf.math.tanh(tf.math.softplus(inputs))

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# ---------------------------------------------------------------------

@keras.utils.register_keras_serializable()
class ScaledMish(keras.layers.Layer):
    """
    Scaled Mish: A variant of the Mish activation function that smoothly saturates at a positive value alpha.
    """

    def __init__(self, alpha=1.0, **kwargs):
        super().__init__(trainable=False, **kwargs)
        self._alpha = alpha

    def call(self, inputs, training: bool = False, **kwargs):
        mish_value = inputs * tf.nn.tanh(tf.nn.softplus(inputs))
        scaled_mish = self._alpha * tf.nn.tanh(mish_value / self._alpha)
        return scaled_mish

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'alpha': self._alpha
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# ---------------------------------------------------------------------
