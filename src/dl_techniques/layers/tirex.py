"""
TiRex-inspired time series forecasting components for Keras.

This module implements time series forecasting layers inspired by the TiRex architecture,
adapted to work with Keras and our project's available components.
"""

import keras
from keras import ops
from typing import Optional, Union, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .norms import RMSNorm
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class QuantileHead(keras.layers.Layer):
    """
    Quantile prediction head for probabilistic forecasting.

    Predicts multiple quantiles simultaneously for uncertainty quantification.

    Args:
        num_quantiles: Integer, number of quantiles to predict.
        output_length: Integer, length of the forecast horizon.
        hidden_dim: Integer, hidden dimension for the prediction head.
        dropout_rate: Float, dropout rate for regularization.
        use_bias: Boolean, whether to use bias in layers.
        kernel_initializer: Initializer for kernel weights.
        bias_initializer: Initializer for bias vector.
        **kwargs: Additional keyword arguments for the Layer base class.
    """

    def __init__(
            self,
            num_quantiles: int,
            output_length: int,
            hidden_dim: int = 256,
            dropout_rate: float = 0.1,
            use_bias: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
            bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.num_quantiles = num_quantiles
        self.output_length = output_length
        self.hidden_dim = hidden_dim
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

        # Project to quantile predictions
        self.projection = keras.layers.Dense(
            self.num_quantiles * self.output_length,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            name="quantile_projection"
        )

        if self.dropout_rate > 0:
            self.dropout = keras.layers.Dropout(self.dropout_rate)

        self.projection.build(input_shape)
        super().build(input_shape)

    def call(self, inputs, training=None):
        """Predict quantiles."""
        x = inputs

        if self.dropout is not None:
            x = self.dropout(x, training=training)

        # Project to quantile predictions
        quantile_preds = self.projection(x, training=training)

        # Reshape to [batch_size, num_quantiles, output_length]
        batch_size = ops.shape(inputs)[0]
        quantiles = ops.reshape(
            quantile_preds,
            (batch_size, self.num_quantiles, self.output_length)
        )

        return quantiles

    def compute_output_shape(self, input_shape):
        """Compute output shape."""
        batch_size = input_shape[0]
        return (batch_size, self.num_quantiles, self.output_length)

    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "num_quantiles": self.num_quantiles,
            "output_length": self.output_length,
            "hidden_dim": self.hidden_dim,
            "dropout_rate": self.dropout_rate,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
        })
        return config

    def get_build_config(self):
        """Get build configuration."""
        return {"input_shape": self._build_input_shape}

    def build_from_config(self, config):
        """Build from configuration."""
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MixedSequentialBlock(keras.layers.Layer):
    """
    Mixed sequential block combining LSTM and self-attention mechanisms.

    This block can operate as either an LSTM block, a Transformer block, or a hybrid
    depending on the configuration.

    Args:
        embed_dim: Integer, embedding dimension.
        num_heads: Integer, number of attention heads (for transformer mode).
        lstm_units: Integer, number of LSTM units (for LSTM mode).
        ff_dim: Integer, feed-forward dimension.
        block_type: String, type of block ('lstm', 'transformer', or 'mixed').
        dropout_rate: Float, dropout rate for regularization.
        use_layer_norm: Boolean, whether to use layer normalization.
        activation: String or callable, activation function for feed-forward layers.
        **kwargs: Additional keyword arguments for the Layer base class.
    """

    def __init__(
            self,
            embed_dim: int,
            num_heads: int = 8,
            lstm_units: Optional[int] = None,
            ff_dim: Optional[int] = None,
            block_type: str = 'mixed',
            dropout_rate: float = 0.1,
            use_layer_norm: bool = True,
            activation: Union[str, callable] = 'relu',
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.lstm_units = lstm_units if lstm_units is not None else embed_dim
        self.ff_dim = ff_dim if ff_dim is not None else embed_dim * 4
        self.block_type = block_type
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.activation = activation

        # Validate block type
        if block_type not in ['lstm', 'transformer', 'mixed']:
            raise ValueError(f"block_type must be one of ['lstm', 'transformer', 'mixed'], got: {block_type}")

        # Layers will be initialized in build()
        self.lstm_layer = None
        self.attention_layer = None
        self.norm1 = None
        self.norm2 = None
        self.norm3 = None
        self.ff_layer1 = None
        self.ff_layer2 = None
        self.dropout1 = None
        self.dropout2 = None
        self.dropout3 = None
        self.projection = None
        self._build_input_shape = None

    def build(self, input_shape):
        """Build the mixed sequential block."""
        self._build_input_shape = input_shape

        # Initialize layers based on block type
        if self.block_type in ['lstm', 'mixed']:
            self.lstm_layer = keras.layers.LSTM(
                self.lstm_units,
                return_sequences=True,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate,
                name="lstm"
            )

        if self.block_type in ['transformer', 'mixed']:
            self.attention_layer = keras.layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.embed_dim // self.num_heads,
                dropout=self.dropout_rate,
                name="attention"
            )

        # Normalization layers
        if self.use_layer_norm:
            self.norm1 = RMSNorm(name="norm1")
            self.norm2 = RMSNorm(name="norm2")
            if self.block_type == 'mixed':
                self.norm3 = RMSNorm(name="norm3")

        # Feed-forward layers
        self.ff_layer1 = keras.layers.Dense(
            self.ff_dim,
            activation=self.activation,
            name="ff1"
        )
        self.ff_layer2 = keras.layers.Dense(
            self.embed_dim,
            name="ff2"
        )

        # Dropout layers
        self.dropout1 = keras.layers.Dropout(self.dropout_rate)
        self.dropout2 = keras.layers.Dropout(self.dropout_rate)
        if self.block_type == 'mixed':
            self.dropout3 = keras.layers.Dropout(self.dropout_rate)

        # Projection layer for LSTM output to match embed_dim
        if self.block_type in ['lstm', 'mixed'] and self.lstm_units != self.embed_dim:
            self.projection = keras.layers.Dense(self.embed_dim, name="lstm_projection")

        # Build sublayers
        if self.lstm_layer is not None:
            self.lstm_layer.build(input_shape)
            lstm_output_shape = list(input_shape)
            lstm_output_shape[-1] = self.lstm_units
            if self.projection is not None:
                self.projection.build(tuple(lstm_output_shape))

        if self.attention_layer is not None:
            self.attention_layer.build(input_shape, input_shape)

        if self.norm1 is not None:
            self.norm1.build(input_shape)
        if self.norm2 is not None:
            self.norm2.build(input_shape)
        if self.norm3 is not None:
            self.norm3.build(input_shape)

        # FF layers
        self.ff_layer1.build(input_shape)
        ff1_output_shape = list(input_shape)
        ff1_output_shape[-1] = self.ff_dim
        self.ff_layer2.build(tuple(ff1_output_shape))

        super().build(input_shape)

    def call(self, inputs, training=None, mask=None):
        """Forward pass through the mixed sequential block."""
        x = inputs

        # LSTM processing
        if self.block_type in ['lstm', 'mixed']:
            if self.norm1 is not None:
                lstm_input = self.norm1(x, training=training)
            else:
                lstm_input = x

            lstm_output = self.lstm_layer(lstm_input, training=training, mask=mask)

            # Project LSTM output if needed
            if self.projection is not None:
                lstm_output = self.projection(lstm_output, training=training)

            lstm_output = self.dropout1(lstm_output, training=training)

            if self.block_type == 'lstm':
                x = x + lstm_output
            else:  # mixed
                x = x + lstm_output

        # Attention processing
        if self.block_type in ['transformer', 'mixed']:
            norm_layer = self.norm2 if self.block_type == 'transformer' else self.norm3
            dropout_layer = self.dropout2 if self.block_type == 'transformer' else self.dropout3

            if norm_layer is not None:
                attn_input = norm_layer(x, training=training)
            else:
                attn_input = x

            attn_output = self.attention_layer(
                attn_input, attn_input,
                training=training,
                attention_mask=mask
            )
            attn_output = dropout_layer(attn_output, training=training)
            x = x + attn_output

        # Feed-forward processing
        if self.norm2 is not None and self.block_type != 'mixed':
            ff_input = self.norm2(x, training=training)
        elif self.norm2 is not None:  # mixed case
            ff_input = self.norm2(x, training=training)
        else:
            ff_input = x

        ff_output = self.ff_layer1(ff_input, training=training)
        ff_output = self.ff_layer2(ff_output, training=training)

        final_dropout = self.dropout2 if self.block_type != 'mixed' else self.dropout2
        ff_output = final_dropout(ff_output, training=training)

        return x + ff_output

    def compute_output_shape(self, input_shape):
        """Output shape is same as input shape."""
        return input_shape

    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "lstm_units": self.lstm_units,
            "ff_dim": self.ff_dim,
            "block_type": self.block_type,
            "dropout_rate": self.dropout_rate,
            "use_layer_norm": self.use_layer_norm,
            "activation": self.activation,
        })
        return config

    def get_build_config(self):
        """Get build configuration."""
        return {"input_shape": self._build_input_shape}

    def build_from_config(self, config):
        """Build from configuration."""
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

# ---------------------------------------------------------------------
