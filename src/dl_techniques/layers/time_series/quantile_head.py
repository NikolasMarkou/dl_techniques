"""
TiRex-inspired time series forecasting components for Keras.

This module implements time series forecasting layers inspired by the TiRex architecture,
adapted to work with Keras and our project's available components.
"""

import keras
from keras import ops
from typing import Union, Any

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class QuantileHead(keras.layers.Layer):
    """
    Quantile prediction head for probabilistic forecasting.

    This layer takes a feature vector and projects it to a series of quantile
    predictions for a specified forecast horizon.

    Args:
        num_quantiles: Integer, number of quantiles to predict.
        output_length: Integer, length of the forecast horizon.
        dropout_rate: Float, dropout rate for regularization.
        use_bias: Boolean, whether to use a bias in the projection layer.
        kernel_initializer: Initializer for the projection layer's kernel weights.
        bias_initializer: Initializer for the projection layer's bias vector.
        **kwargs: Additional keyword arguments for the Layer base class.
    """

    def __init__(
            self,
            num_quantiles: int,
            output_length: int,
            dropout_rate: float = 0.1,
            use_bias: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
            bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.num_quantiles = num_quantiles
        self.output_length = output_length
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)

        # Layers will be initialized in build()
        self.projection = None
        self.dropout = None
        self._build_input_shape = None

    def build(self, input_shape):
        """Build the quantile prediction head."""
        self._build_input_shape = input_shape

        # A single dense layer to project the input features to the quantile predictions.
        self.projection = keras.layers.Dense(
            self.num_quantiles * self.output_length,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            name="quantile_projection"
            # No activation function is used, which is standard for regression outputs.
        )

        if self.dropout_rate > 0:
            self.dropout = keras.layers.Dropout(self.dropout_rate)

        super().build(input_shape)

    def call(self, inputs, training=None):
        """Predict quantiles from the input feature vector."""
        x = inputs

        if self.dropout is not None:
            x = self.dropout(x, training=training)

        # Project the features to the flattened quantile predictions.
        quantile_preds = self.projection(x, training=training)

        # Reshape to [batch_size, num_quantiles, output_length]
        # Using -1 for the batch dimension makes the layer robust to dynamic batch sizes.
        quantiles = ops.reshape(
            quantile_preds,
            (-1, self.num_quantiles, self.output_length)
        )

        return quantiles

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the layer."""
        batch_size = input_shape[0]
        return (batch_size, self.num_quantiles, self.output_length)

    def get_config(self):
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            "num_quantiles": self.num_quantiles,
            "output_length": self.output_length,
            "dropout_rate": self.dropout_rate,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
        })
        return config

    def get_build_config(self):
        """Store the input shape for serialization."""
        return {"input_shape": self._build_input_shape}

# ---------------------------------------------------------------------
