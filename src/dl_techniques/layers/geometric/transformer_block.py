import keras
import numpy as np
from keras import ops
from typing import Optional, Dict, List, Any, Tuple

from .anchor_attention import AnchorAttention
from .shared_weights_cross_attention import SharedWeightsCrossAttention

from dl_techniques.utils.logger import logger


@keras.saving.register_keras_serializable()
class TransformerBlock(keras.layers.Layer):
    """Standard transformer block with customizable attention mechanism."""

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            mlp_ratio: float = 4.0,
            dropout: float = 0.0,
            attention_class: str = "standard",
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout_rate = dropout
        self.attention_class = attention_class

        # Will be built in build()
        self.norm1 = None
        self.attention = None
        self.norm2 = None
        self.mlp = None

    def build(self, input_shape):
        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-6, name="norm1")

        # Choose attention mechanism
        if self.attention_class == "anchor":
            self.attention = AnchorAttention(dim=self.dim, num_heads=self.num_heads, name="attention")
        elif self.attention_class == "shared_cross":
            self.attention = SharedWeightsCrossAttention(dim=self.dim, num_heads=self.num_heads, name="attention")
        else:
            # Standard self-attention (we'll implement a simple version)
            self.attention = AnchorAttention(dim=self.dim, num_heads=self.num_heads, name="attention")

        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-6, name="norm2")

        # MLP
        mlp_dim = int(self.dim * self.mlp_ratio)
        self.mlp = keras.Sequential([
            keras.layers.Dense(mlp_dim, activation="gelu", name="mlp_dense1"),
            keras.layers.Dropout(self.dropout_rate) if self.dropout_rate > 0 else keras.layers.Lambda(lambda x: x),
            keras.layers.Dense(self.dim, name="mlp_dense2")
        ], name="mlp")

        super().build(input_shape)

    def call(self, x, training=None, **attention_kwargs):
        # Attention with residual
        x = x + self.attention(self.norm1(x), training=training, **attention_kwargs)

        # MLP with residual
        x = x + self.mlp(self.norm2(x), training=training)

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "dropout": self.dropout_rate,
            "attention_class": self.attention_class,
        })
        return config