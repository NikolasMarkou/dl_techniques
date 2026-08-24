"""
ActionEmbedder — maps per-timestep action vectors to the model's embedding space.

Upstream PyTorch (`/tmp/lewm_source/module.py:Embedder`):

.. code-block:: python

    Conv1d(action_dim, smoothed_dim, kernel_size=1)
    -> permute -> Linear(smoothed_dim, mlp_scale * emb_dim) -> SiLU
    -> Linear(mlp_scale * emb_dim, emb_dim)

In Keras we use `Conv1D` (channels-last) directly — so the PyTorch
`permute(0,2,1)` pair is a no-op here. Input is `(B, T, action_dim)`, output
is `(B, T, emb_dim)`.
"""

import keras
from typing import Any, Dict, Optional, Tuple


@keras.saving.register_keras_serializable()
class ActionEmbedder(keras.layers.Layer):
    """Embed per-timestep action vectors to the model's embedding space.

    :param action_dim: dimension of the raw action vector (e.g. 2 for PushT).
    :param smoothed_dim: intermediate dim after the 1x1 Conv1D "patch embed".
    :param emb_dim: output embedding dimension (= model D).
    :param mlp_scale: hidden-layer width multiplier of the 2-layer MLP.
    :param kwargs: passthrough to `keras.layers.Layer`.
    """

    def __init__(
        self,
        action_dim: int,
        smoothed_dim: int,
        emb_dim: int,
        mlp_scale: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if action_dim <= 0 or smoothed_dim <= 0 or emb_dim <= 0:
            raise ValueError(
                f"action_dim/smoothed_dim/emb_dim must be positive. "
                f"Got: {action_dim}, {smoothed_dim}, {emb_dim}."
            )
        self.action_dim = action_dim
        self.smoothed_dim = smoothed_dim
        self.emb_dim = emb_dim
        self.mlp_scale = mlp_scale

        # Conv1D with kernel_size=1 in channels-last form = per-timestep Dense.
        self.patch_embed = keras.layers.Conv1D(
            filters=smoothed_dim, kernel_size=1, strides=1,
            padding="valid", name="patch_embed",
        )
        self.fc1 = keras.layers.Dense(mlp_scale * emb_dim, activation=None, name="fc1")
        self.act = keras.layers.Activation("silu", name="act")
        self.fc2 = keras.layers.Dense(emb_dim, activation=None, name="fc2")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        if len(input_shape) != 3:
            raise ValueError(
                f"ActionEmbedder expects input of rank 3 (B, T, action_dim). "
                f"Got shape={input_shape}"
            )
        self.patch_embed.build(input_shape)
        patch_out = tuple(list(input_shape[:-1]) + [self.smoothed_dim])
        self.fc1.build(patch_out)
        fc1_out = tuple(list(input_shape[:-1]) + [self.mlp_scale * self.emb_dim])
        self.act.build(fc1_out)
        self.fc2.build(fc1_out)
        super().build(input_shape)

    def call(self, x: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        # Upstream does .float() — we trust the caller's dtype.
        x = self.patch_embed(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        return tuple(list(input_shape[:-1]) + [self.emb_dim])

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "action_dim": self.action_dim,
            "smoothed_dim": self.smoothed_dim,
            "emb_dim": self.emb_dim,
            "mlp_scale": self.mlp_scale,
        })
        return config
