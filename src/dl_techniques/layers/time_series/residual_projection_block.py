"""Residual projection block shared by the Timestral patch embedding and quantile head.

``output(silu(hidden(x))) + residual(x)``, three Dense layers named after the
reference PyTorch state dict (``hidden_layer``, ``output_layer``,
``residual_layer``).
"""

import keras

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.time_series.residual_projection_block")
class ResidualProjectionBlock(keras.layers.Layer):
    """Residual block: output(silu(hidden(x))) + residual(x).

    Used by Timestral's patch embedding and its quantile head. The sublayer names
    ``hidden_layer``, ``output_layer`` and ``residual_layer`` are the reference
    state-dict names, so they must not change.
    """

    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        output_dims: int,
        use_bias: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims
        self.use_bias = use_bias

    def build(self, input_shape):
        self.hidden_layer = keras.layers.Dense(
            self.hidden_dims,
            use_bias=self.use_bias,
            name="hidden_layer",
            dtype=self.dtype_policy,
        )
        self.output_layer = keras.layers.Dense(
            self.output_dims,
            use_bias=self.use_bias,
            name="output_layer",
            dtype=self.dtype_policy,
        )
        self.residual_layer = keras.layers.Dense(
            self.output_dims,
            use_bias=self.use_bias,
            name="residual_layer",
            dtype=self.dtype_policy,
        )
        self.activation = keras.activations.silu
        # Sublayers are built here, in call order, so build() creates the full
        # weight set and a .keras round trip restores every weight.
        hidden_shape = tuple(input_shape[:-1]) + (self.hidden_dims,)
        self.hidden_layer.build(input_shape)
        self.output_layer.build(hidden_shape)
        self.residual_layer.build(input_shape)
        super().build(input_shape)

    def call(self, x):
        hidden = self.activation(self.hidden_layer(x))
        out = self.output_layer(hidden)
        return out + self.residual_layer(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_dims": self.input_dims,
            "hidden_dims": self.hidden_dims,
            "output_dims": self.output_dims,
            "use_bias": self.use_bias,
        })
        return config
