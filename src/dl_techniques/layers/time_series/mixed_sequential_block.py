"""
TiRex-inspired time series forecasting components for Keras.

This module implements time series forecasting layers inspired by the TiRex architecture,
adapted to work with Keras and our project's available components.
"""

import keras
from typing import Optional, Union, Tuple, Callable, Any, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..norms.rms_norm import RMSNorm

# ---------------------------------------------------------------------

BlockType = Literal['lstm', 'transformer', 'mixed']

@keras.saving.register_keras_serializable()
class MixedSequentialBlock(keras.layers.Layer):
    """
    Mixed sequential block combining LSTM and self-attention mechanisms for time series processing.

    This layer implements a flexible architecture that can operate in three modes:
    LSTM-only, Transformer-only, or a hybrid sequential combination. It uses Pre-LayerNorm
    architecture with residual connections, making it suitable for deep time series models
    where both recurrent processing and self-attention are beneficial.

    **Intent**: Provide a configurable building block for time series models that can
    leverage the temporal modeling strengths of LSTMs and the global context modeling
    of self-attention mechanisms, either independently or in combination.

    **Architecture Modes**:

    **1. LSTM Mode** (`block_type='lstm'`):
    ```
    Input(shape=[batch, seq_len, embed_dim])
           ↓
    RMSNorm → LSTM → Dropout → Residual(+Input)
           ↓
    RMSNorm → FFN → Dropout → Residual(+Previous)
           ↓
    Output(shape=[batch, seq_len, embed_dim])
    ```

    **2. Transformer Mode** (`block_type='transformer'`):
    ```
    Input(shape=[batch, seq_len, embed_dim])
           ↓
    RMSNorm → MultiHeadAttention → Dropout → Residual(+Input)
           ↓
    RMSNorm → FFN → Dropout → Residual(+Previous)
           ↓
    Output(shape=[batch, seq_len, embed_dim])
    ```

    **3. Mixed Mode** (`block_type='mixed'`):
    ```
    Input(shape=[batch, seq_len, embed_dim])
           ↓
    RMSNorm → LSTM → Dropout → Residual(+Input)
           ↓
    RMSNorm → MultiHeadAttention → Dropout → Residual(+Previous)
           ↓
    RMSNorm → FFN → Dropout → Residual(+Previous)
           ↓
    Output(shape=[batch, seq_len, embed_dim])
    ```

    **Key Features**:
    - Pre-LayerNorm architecture for training stability
    - Residual connections for gradient flow
    - Configurable LSTM and attention components
    - Dropout regularization at each stage
    - Optional dimension projection for LSTM outputs

    Args:
        embed_dim: Integer, embedding dimension and output dimension. Must be positive.
            This dimension is maintained throughout all transformations.
        num_heads: Integer, number of attention heads for transformer/mixed modes.
            Must be positive and divide evenly into embed_dim. Defaults to 8.
        lstm_units: Optional integer, number of LSTM units for lstm/mixed modes.
            If None, defaults to embed_dim. Must be positive if specified.
        ff_dim: Optional integer, dimension of feed-forward network hidden layer.
            If None, defaults to embed_dim * 4 (standard transformer ratio). Must be positive.
        block_type: BlockType, architecture mode. Must be one of:
            - 'lstm': LSTM → FFN
            - 'transformer': MultiHeadAttention → FFN
            - 'mixed': LSTM → MultiHeadAttention → FFN
            Defaults to 'mixed'.
        dropout_rate: Float between 0 and 1, dropout rate for all dropout layers.
            Applied after LSTM, attention, and FFN outputs. Defaults to 0.1.
        use_layer_norm: Boolean, whether to apply RMSNorm before each sub-layer.
            Following Pre-LN architecture. Defaults to True.
        activation: String or callable, activation function for feed-forward network.
            Applied in the first FFN layer. Defaults to 'relu'.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        3D tensor with shape: `(batch_size, sequence_length, embed_dim)`.
        Typical time series input where sequence_length is the time dimension.

    Output shape:
        3D tensor with shape: `(batch_size, sequence_length, embed_dim)`.
        Shape is preserved through all processing modes.

    Attributes:
        lstm_layer: LSTM layer (lstm/mixed modes only).
        attention_layer: MultiHeadAttention layer (transformer/mixed modes only).
        projection: Dense layer for LSTM output projection (if lstm_units != embed_dim).
        norm1, norm2, norm3: RMSNorm layers for Pre-LN architecture.
        ff_layer1, ff_layer2: Feed-forward network layers.
        dropout1, dropout2, dropout3: Dropout layers for regularization.

    Example:
        ```python
        # Mixed LSTM + Attention block for time series
        block = MixedSequentialBlock(
            embed_dim=256,
            num_heads=8,
            block_type='mixed'
        )

        # Time series input: 64 time steps, 256 features
        inputs = keras.Input(shape=(64, 256))
        outputs = block(inputs)  # Shape: (batch, 64, 256)

        # LSTM-only block for purely sequential processing
        lstm_block = MixedSequentialBlock(
            embed_dim=128,
            lstm_units=256,  # Different LSTM size
            block_type='lstm',
            dropout_rate=0.2
        )

        # Transformer-only block for global context
        transformer_block = MixedSequentialBlock(
            embed_dim=512,
            num_heads=16,
            ff_dim=2048,  # Larger FFN
            block_type='transformer'
        )

        # In a complete model
        inputs = keras.Input(shape=(sequence_length, feature_dim))
        x = keras.layers.Dense(embed_dim)(inputs)  # Project to embed_dim

        for _ in range(num_layers):
            x = MixedSequentialBlock(embed_dim=embed_dim)(x)

        outputs = keras.layers.Dense(output_dim)(x)
        model = keras.Model(inputs, outputs)
        ```

    Note:
        The mixed mode processes inputs sequentially: LSTM captures local temporal
        patterns, attention captures global dependencies, and FFN provides
        non-linear transformation. This combination is particularly effective
        for long time series with both local and global patterns.

    References:
        - Attention Is All You Need (Transformer): https://arxiv.org/abs/1706.03762
        - On Layer Normalization in the Transformer Architecture: https://arxiv.org/abs/2002.04745
        - TiRex: Time series forecasting with mixed architectures
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        lstm_units: Optional[int] = None,
        ff_dim: Optional[int] = None,
        block_type: BlockType = 'mixed',
        dropout_rate: float = 0.1,
        use_layer_norm: bool = True,
        activation: Union[str, Callable] = 'relu',
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        if lstm_units is not None and lstm_units <= 0:
            raise ValueError(f"lstm_units must be positive if specified, got {lstm_units}")
        if ff_dim is not None and ff_dim <= 0:
            raise ValueError(f"ff_dim must be positive if specified, got {ff_dim}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")
        if block_type not in ['lstm', 'transformer', 'mixed']:
            raise ValueError(f"block_type must be one of ['lstm', 'transformer', 'mixed'], got {block_type}")

        # Store configuration
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.lstm_units = lstm_units if lstm_units is not None else embed_dim
        self.ff_dim = ff_dim if ff_dim is not None else embed_dim * 4
        self.block_type = block_type
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.activation = keras.activations.get(activation)

        # CREATE all sub-layers in __init__ (following modern Keras 3 pattern)

        # LSTM components (for 'lstm' and 'mixed' modes)
        if self.block_type in ['lstm', 'mixed']:
            self.lstm_layer = keras.layers.LSTM(
                units=self.lstm_units,
                return_sequences=True,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate,
                name="lstm"
            )

            # Projection layer if LSTM output dim doesn't match embedding dim
            if self.lstm_units != self.embed_dim:
                self.projection = keras.layers.Dense(
                    units=self.embed_dim,
                    name="lstm_projection"
                )
            else:
                self.projection = None
        else:
            self.lstm_layer = None
            self.projection = None

        # Attention components (for 'transformer' and 'mixed' modes)
        if self.block_type in ['transformer', 'mixed']:
            self.attention_layer = keras.layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.embed_dim // self.num_heads,
                dropout=self.dropout_rate,
                name="attention"
            )
        else:
            self.attention_layer = None

        # Normalization layers (Pre-LN architecture)
        if self.use_layer_norm:
            self.norm1 = RMSNorm(name="norm1")
            self.norm2 = RMSNorm(name="norm2")
            # Mixed mode needs an extra norm layer
            if self.block_type == 'mixed':
                self.norm3 = RMSNorm(name="norm3")
            else:
                self.norm3 = None
        else:
            self.norm1 = None
            self.norm2 = None
            self.norm3 = None

        # Feed-forward network layers (common to all block types)
        self.ff_layer1 = keras.layers.Dense(
            units=self.ff_dim,
            activation=self.activation,
            name="ff1"
        )
        self.ff_layer2 = keras.layers.Dense(
            units=self.embed_dim,
            name="ff2"
        )

        # Dropout layers
        self.dropout1 = keras.layers.Dropout(rate=self.dropout_rate, name="dropout1")
        self.dropout2 = keras.layers.Dropout(rate=self.dropout_rate, name="dropout2")
        if self.block_type == 'mixed':
            self.dropout3 = keras.layers.Dropout(rate=self.dropout_rate, name="dropout3")
        else:
            self.dropout3 = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and all its sub-layers.

        CRITICAL: Explicitly build each sub-layer for robust serialization.
        """
        # Build sub-layers based on block type and configuration

        # Build LSTM components if present
        if self.lstm_layer is not None:
            self.lstm_layer.build(input_shape)

            # Build projection if it exists
            if self.projection is not None:
                # LSTM output shape: (batch, seq_len, lstm_units)
                lstm_output_shape = (*input_shape[:-1], self.lstm_units)
                self.projection.build(lstm_output_shape)

        # Build attention component if present
        if self.attention_layer is not None:
            self.attention_layer.build(input_shape)

        # Build normalization layers if present
        if self.norm1 is not None:
            self.norm1.build(input_shape)
        if self.norm2 is not None:
            self.norm2.build(input_shape)
        if self.norm3 is not None:
            self.norm3.build(input_shape)

        # Build feed-forward network
        self.ff_layer1.build(input_shape)

        # FF layer 2 input shape: (batch, seq_len, ff_dim)
        ff1_output_shape = (*input_shape[:-1], self.ff_dim)
        self.ff_layer2.build(ff1_output_shape)

        # Build dropout layers
        self.dropout1.build(input_shape)
        self.dropout2.build(input_shape)
        if self.dropout3 is not None:
            self.dropout3.build(input_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def _transformer_block(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
        mask: Optional[keras.KerasTensor] = None
    ) -> keras.KerasTensor:
        """Implements the standard Pre-LN Transformer block data flow."""
        x = inputs

        # First Sub-layer: Multi-head Self-Attention
        norm_input = self.norm1(x, training=training) if self.use_layer_norm else x
        attn_output = self.attention_layer(
            query=norm_input,
            value=norm_input,
            key=norm_input,
            training=training,
            attention_mask=mask
        )
        attn_output = self.dropout1(attn_output, training=training)
        x = x + attn_output  # Residual connection

        # Second Sub-layer: Feed-Forward Network
        norm_output = self.norm2(x, training=training) if self.use_layer_norm else x
        ff_output = self.ff_layer1(norm_output, training=training)
        ff_output = self.ff_layer2(ff_output, training=training)
        ff_output = self.dropout2(ff_output, training=training)
        return x + ff_output  # Residual connection

    def _lstm_block(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
        mask: Optional[keras.KerasTensor] = None
    ) -> keras.KerasTensor:
        """Implements an LSTM block followed by a Feed-Forward network."""
        x = inputs

        # First Sub-layer: LSTM
        norm_input = self.norm1(x, training=training) if self.use_layer_norm else x
        lstm_output = self.lstm_layer(norm_input, training=training, mask=mask)
        if self.projection is not None:
            lstm_output = self.projection(lstm_output, training=training)
        lstm_output = self.dropout1(lstm_output, training=training)
        x = x + lstm_output  # Residual connection

        # Second Sub-layer: Feed-Forward Network
        norm_output = self.norm2(x, training=training) if self.use_layer_norm else x
        ff_output = self.ff_layer1(norm_output, training=training)
        ff_output = self.ff_layer2(ff_output, training=training)
        ff_output = self.dropout2(ff_output, training=training)
        return x + ff_output  # Residual connection

    def _mixed_block(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
        mask: Optional[keras.KerasTensor] = None
    ) -> keras.KerasTensor:
        """Implements a sequential LSTM → Attention → FFN flow."""
        x = inputs

        # Block 1: LSTM
        norm1_input = self.norm1(x, training=training) if self.use_layer_norm else x
        lstm_output = self.lstm_layer(norm1_input, training=training, mask=mask)
        if self.projection is not None:
            lstm_output = self.projection(lstm_output, training=training)
        lstm_output = self.dropout1(lstm_output, training=training)
        x = x + lstm_output  # Residual 1

        # Block 2: Attention
        norm3_input = self.norm3(x, training=training) if self.use_layer_norm else x
        attn_output = self.attention_layer(
            query=norm3_input,
            value=norm3_input,
            key=norm3_input,
            training=training,
            attention_mask=mask
        )
        attn_output = self.dropout3(attn_output, training=training)
        x = x + attn_output  # Residual 2

        # Block 3: Feed-Forward Network
        norm2_output = self.norm2(x, training=training) if self.use_layer_norm else x
        ff_output = self.ff_layer1(norm2_output, training=training)
        ff_output = self.ff_layer2(ff_output, training=training)
        ff_output = self.dropout2(ff_output, training=training)
        return x + ff_output  # Residual 3

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
        mask: Optional[keras.KerasTensor] = None
    ) -> keras.KerasTensor:
        """Forward pass dispatching to the correct block type."""
        if self.block_type == 'transformer':
            return self._transformer_block(inputs, training=training, mask=mask)
        elif self.block_type == 'lstm':
            return self._lstm_block(inputs, training=training, mask=mask)
        elif self.block_type == 'mixed':
            return self._mixed_block(inputs, training=training, mask=mask)
        else:
            # This should never happen due to validation in __init__
            raise RuntimeError(f"Invalid block_type encountered: {self.block_type}")

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Output shape is the same as the input shape."""
        return input_shape

    def get_config(self) -> dict[str, Any]:
        """Return configuration for serialization."""
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

# ---------------------------------------------------------------------