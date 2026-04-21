"""
MLPProjector — 2-layer MLP used by LeWM for the projector and pred_proj.

Upstream PyTorch (`/tmp/lewm_source/module.py:MLP`):

.. code-block:: python

    Linear(input_dim, hidden_dim) -> LayerNorm(hidden_dim) -> GELU
    -> Linear(hidden_dim, output_dim)

In upstream JEPA the `norm_fn` defaults to `nn.LayerNorm`. We expose the
same default here. (The description in plan.md said "BatchNorm1d" following
a reading of upstream `JEPA`, but the actual `MLP` class defaults to
`LayerNorm` and that's what upstream LeWM wires in. We follow the
upstream-code truth, not the plan note.)
"""

import keras
from typing import Any, Dict, Optional, Tuple


@keras.saving.register_keras_serializable()
class MLPProjector(keras.layers.Layer):
    """2-layer MLP with intermediate normalization + GELU.

    :param input_dim: expected last-dim of the input (kept for clarity;
        Keras infers from input_shape in build).
    :param hidden_dim: width of the hidden layer.
    :param output_dim: output last-dim. Defaults to input_dim.
    :param use_layer_norm: if True (default), apply LayerNormalization on
        the hidden activation. Follows upstream `norm_fn=nn.LayerNorm`.
    :param kwargs: passthrough to `keras.layers.Layer`.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: Optional[int] = None,
        use_layer_norm: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError(
                f"input_dim and hidden_dim must be positive. "
                f"Got input_dim={input_dim}, hidden_dim={hidden_dim}."
            )
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.use_layer_norm = use_layer_norm

        self.fc1 = keras.layers.Dense(hidden_dim, activation=None, name="fc1")
        self.norm = (
            keras.layers.LayerNormalization(epsilon=1e-6, name="norm")
            if use_layer_norm else None
        )
        self.act = keras.layers.Activation("gelu", name="act")
        self.fc2 = keras.layers.Dense(self.output_dim, activation=None, name="fc2")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        self.fc1.build(input_shape)
        hidden_shape = tuple(list(input_shape[:-1]) + [self.hidden_dim])
        if self.norm is not None:
            self.norm.build(hidden_shape)
        self.act.build(hidden_shape)
        self.fc2.build(hidden_shape)
        super().build(input_shape)

    def call(self, x: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        x = self.fc1(x)
        if self.norm is not None:
            x = self.norm(x, training=training)
        x = self.act(x)
        x = self.fc2(x)
        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        return tuple(list(input_shape[:-1]) + [self.output_dim])

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "use_layer_norm": self.use_layer_norm,
        })
        return config
