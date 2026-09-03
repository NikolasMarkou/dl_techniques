"""``MLPProjector``, a 2-layer MLP LeWM uses for both the projector and pred_proj heads.

It runs Dense, then an optional LayerNormalization, then GELU, then a
second Dense. Normalization defaults on and uses LayerNormalization rather
than BatchNormalization, so a batch of size 1 still works.
"""

import keras
from typing import Any, Dict, Optional, Tuple
from dl_techniques.utils.keras_registration import register_dl_technique


@register_dl_technique("dl_techniques.models.lewm.projector")
class MLPProjector(keras.layers.Layer):
    """2-layer MLP with intermediate normalization + GELU.

    Architecture:

    .. code-block:: text

        input  [..., input_dim]
           |
           v
        Dense -> LayerNorm (optional) -> GELU -> Dense
           |
           v
        output  [..., output_dim]

    :param input_dim: expected last-dim of the input (kept for clarity;
        Keras infers from input_shape in build).
    :param hidden_dim: width of the hidden layer.
    :param output_dim: output last-dim. Defaults to input_dim.
    :param use_layer_norm: apply LayerNormalization on the hidden activation.
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
        # DECISION plan_2026-04-21_8416bc0b/D-002: LayerNorm, not BatchNorm; a batch of 1 would break BatchNorm.
        # Matches upstream MLP's norm_fn default. See decisions.md.
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
