"""
TiRex-inspired time series forecasting components for Keras.

This module implements time series forecasting layers inspired by the TiRex architecture,
adapted to work with Keras and our project's available components.
"""

import keras
from typing import Optional, Union, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..norms.rms_norm import RMSNorm


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MixedSequentialBlock(keras.layers.Layer):
    """
    Mixed sequential block combining LSTM and self-attention mechanisms.

    This block implements a pre-LayerNorm architecture. It can operate as an
    LSTM block, a standard Transformer block, or a sequential hybrid.

    Args:
        embed_dim: Integer, embedding dimension.
        num_heads: Integer, number of attention heads (for transformer/mixed mode).
        lstm_units: Integer, number of LSTM units (for lstm/mixed mode).
        ff_dim: Integer, the dimension of the feed-forward network.
        block_type: String, type of block ('lstm', 'transformer', or 'mixed').
        dropout_rate: Float, dropout rate for regularization.
        use_layer_norm: Boolean, whether to use layer normalization (RMSNorm).
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
        self.activation = keras.activations.get(activation)

        # Validate block type
        if block_type not in ['lstm', 'transformer', 'mixed']:
            raise ValueError(f"block_type must be one of ['lstm', 'transformer', 'mixed'], got: {block_type}")

        # Layers will be initialized in build()
        self.lstm_layer = None
        self.attention_layer = None
        self.norm1 = None
        self.norm2 = None
        self.norm3 = None  # Only for 'mixed' type
        self.ff_layer1 = None
        self.ff_layer2 = None
        self.dropout1 = None
        self.dropout2 = None
        self.dropout3 = None  # Only for 'mixed' type
        self.projection = None
        self._build_input_shape = None

    def build(self, input_shape):
        """Build the mixed sequential block's layers."""
        self._build_input_shape = input_shape

        # --- Layer Initialization ---
        # Initialize layers based on block type. Keras will build them on first call.
        if self.block_type in ['lstm', 'mixed']:
            self.lstm_layer = keras.layers.LSTM(
                self.lstm_units,
                return_sequences=True,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate,
                name="lstm"
            )
            # Projection layer if LSTM output dim doesn't match embedding dim
            if self.lstm_units != self.embed_dim:
                self.projection = keras.layers.Dense(self.embed_dim, name="lstm_projection")

        if self.block_type in ['transformer', 'mixed']:
            self.attention_layer = keras.layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.embed_dim // self.num_heads,
                dropout=self.dropout_rate,
                name="attention"
            )

        # Normalization layers (Pre-LN architecture)
        if self.use_layer_norm:
            self.norm1 = RMSNorm(name="norm1")
            self.norm2 = RMSNorm(name="norm2")
            if self.block_type == 'mixed':
                self.norm3 = RMSNorm(name="norm3")

        # Feed-forward layers (common to all block types)
        self.ff_layer1 = keras.layers.Dense(self.ff_dim, activation=self.activation, name="ff1")
        self.ff_layer2 = keras.layers.Dense(self.embed_dim, name="ff2")

        # Dropout layers
        self.dropout1 = keras.layers.Dropout(self.dropout_rate)
        self.dropout2 = keras.layers.Dropout(self.dropout_rate)
        if self.block_type == 'mixed':
            self.dropout3 = keras.layers.Dropout(self.dropout_rate)

        super().build(input_shape)

    def _transformer_block(self, inputs, training=None, mask=None):
        """Implements the standard Pre-LN Transformer block data flow."""
        x = inputs

        # --- First Sub-layer: Multi-head Self-Attention ---
        norm_input = self.norm1(x, training=training) if self.use_layer_norm else x
        attn_output = self.attention_layer(
            query=norm_input, value=norm_input, key=norm_input,
            training=training, attention_mask=mask
        )
        attn_output = self.dropout1(attn_output, training=training)
        x = x + attn_output  # Residual connection

        # --- Second Sub-layer: Feed-Forward Network ---
        norm_output = self.norm2(x, training=training) if self.use_layer_norm else x
        ff_output = self.ff_layer1(norm_output, training=training)
        ff_output = self.ff_layer2(ff_output, training=training)
        ff_output = self.dropout2(ff_output, training=training)
        return x + ff_output # Residual connection

    def _lstm_block(self, inputs, training=None, mask=None):
        """Implements an LSTM block followed by a Feed-Forward network."""
        x = inputs

        # --- First Sub-layer: LSTM ---
        norm_input = self.norm1(x, training=training) if self.use_layer_norm else x
        lstm_output = self.lstm_layer(norm_input, training=training, mask=mask)
        if self.projection:
            lstm_output = self.projection(lstm_output, training=training)
        lstm_output = self.dropout1(lstm_output, training=training)
        x = x + lstm_output  # Residual connection

        # --- Second Sub-layer: Feed-Forward Network ---
        norm_output = self.norm2(x, training=training) if self.use_layer_norm else x
        ff_output = self.ff_layer1(norm_output, training=training)
        ff_output = self.ff_layer2(ff_output, training=training)
        ff_output = self.dropout2(ff_output, training=training)
        return x + ff_output # Residual connection

    def _mixed_block(self, inputs, training=None, mask=None):
        """Implements a sequential LSTM -> Attention -> FFN flow."""
        x = inputs

        # --- Block 1: LSTM ---
        norm1_input = self.norm1(x, training=training) if self.use_layer_norm else x
        lstm_output = self.lstm_layer(norm1_input, training=training, mask=mask)
        if self.projection:
            lstm_output = self.projection(lstm_output, training=training)
        lstm_output = self.dropout1(lstm_output, training=training)
        x = x + lstm_output  # Residual 1

        # --- Block 2: Attention ---
        # Note: The original code used norm3 and dropout3 here, that is preserved.
        norm3_input = self.norm3(x, training=training) if self.use_layer_norm else x
        attn_output = self.attention_layer(
            query=norm3_input, value=norm3_input, key=norm3_input,
            training=training, attention_mask=mask
        )
        attn_output = self.dropout3(attn_output, training=training)
        x = x + attn_output # Residual 2

        # --- Block 3: Feed-Forward Network ---
        norm2_output = self.norm2(x, training=training) if self.use_layer_norm else x
        ff_output = self.ff_layer1(norm2_output, training=training)
        ff_output = self.ff_layer2(ff_output, training=training)
        ff_output = self.dropout2(ff_output, training=training)
        return x + ff_output # Residual 3

    def call(self, inputs, training=None, mask=None):
        """Forward pass dispatching to the correct block type."""
        if self.block_type == 'transformer':
            return self._transformer_block(inputs, training=training, mask=mask)
        elif self.block_type == 'lstm':
            return self._lstm_block(inputs, training=training, mask=mask)
        elif self.block_type == 'mixed':
            return self._mixed_block(inputs, training=training, mask=mask)
        else:
            raise RuntimeError(f"Invalid block_type encountered: {self.block_type}")

    def compute_output_shape(self, input_shape):
        """Output shape is the same as the input shape."""
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
            "activation": keras.activations.serialize(self.activation),
        })
        return config

    def get_build_config(self):
        """Store the input shape for serialization."""
        return {"input_shape": self._build_input_shape}

# ---------------------------------------------------------------------
