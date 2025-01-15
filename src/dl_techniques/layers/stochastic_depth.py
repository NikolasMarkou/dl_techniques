import keras
import numpy as np
import tensorflow as tf
from typing import Tuple, Union, List

# ---------------------------------------------------------------------


@keras.utils.register_keras_serializable()
class StochasticDepth(keras.layers.Layer):
    """
    Stochastic Depth module.

    It performs batch-wise dropping rather than sample-wise. In libraries like
    `timm`, it's similar to `DropPath` layers that drops residual paths
    sample-wise.

    References:
      - https://github.com/rwightman/pytorch-image-models
      - "Deep Networks with Stochastic Depth" (https://arxiv.org/abs/1603.09382)

    Args:
        drop_path_rate (float): Probability of dropping paths. Should be within [0, 1].

    Returns:
        Tensor either with the residual path dropped or kept.
    """

    def __init__(self, drop_path_rate: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        if not 0.0 <= drop_path_rate < 1.0:
            raise ValueError("drop_path_rate must be between 0.0 and 1.0")
        self.drop_path_rate = drop_path_rate
        self.dropout = None

    def build(self, input_shape: tf.TensorShape):
        dims = len(input_shape)
        noise_shape = (input_shape[0],) + (1,) * (dims - 1)
        self.dropout = tf.keras.layers.Dropout(
            rate=self.drop_path_rate,
            noise_shape=noise_shape
        )

    def call(self, inputs, training: bool = None):
        x = inputs
        return self.dropout(x, training=training)

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        return input_shape

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"drop_path_rate": self.drop_path_rate})
        return config

# ---------------------------------------------------------------------

