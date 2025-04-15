import keras
import numpy as np
import tensorflow as tf
from keras.api import Model
from keras.api import layers
from keras.api import losses
from keras.api import optimizers
from keras.api import initializers
from typing import Optional, Dict, Any, List, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

class PositionalEmbedding(keras.layers.Layer):
    """Learned positional embedding with enhanced stability."""

    def __init__(
            self,
            max_seq_len: int,
            dim: int,
            dropout: float = 0.0,
            scale: float = 0.02,
            **kwargs
    ) -> None:
        """
        Args:
            max_seq_len: Maximum sequence length
            dim: Embedding dimension
            dropout: Dropout rate
            scale: Initialization scale for embeddings
        """
        super().__init__(**kwargs)
        self.max_seq_len = max_seq_len
        self.dim = dim
        self.dropout_rate = dropout
        self.scale = scale
        self.pos_embedding = None
        self.dropout = None

    def build(self, input_shape):
        # Initialize embeddings with truncated normal
        self.pos_embedding = self.add_weight(
            "pos_embedding",
            shape=(1, self.max_seq_len, self.dim),
            initializer=initializers.TruncatedNormal(stddev=self.scale),
            trainable=True
        )

        self.dropout = layers.Dropout(self.dropout_rate)

        super().build(input_shape)

    def call(
            self,
            x: tf.Tensor,
            training: bool = False
    ) -> tf.Tensor:
        """Add positional embeddings to input.

        Args:
            x: Input tensor [batch_size, seq_len, dim]
            training: Whether in training mode

        Returns:
            Tensor with positional information added
        """
        seq_len = tf.shape(x)[1]
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds maximum length {self.max_seq_len}"
            )

        # Add positions
        positions = self.pos_embedding[:, :seq_len, :]
        x = x + positions

        return self.dropout(x, training=training)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "max_seq_len": self.max_seq_len,
            "dim": self.dim,
            "dropout": self.dropout_rate,
            "scale": self.scale
        })
        return config