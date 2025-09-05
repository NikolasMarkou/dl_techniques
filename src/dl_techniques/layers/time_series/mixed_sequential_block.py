"""
TiRex-inspired time series forecasting components for Keras.

This module implements time series forecasting layers inspired by the TiRex architecture,
adapted to work with Keras and our project's available components.
"""

import keras
from typing import Optional, Union, Tuple, Callable, Any, Literal, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..norms import create_normalization_layer
from ..attention import create_attention_layer
from ..ffn import create_ffn_layer

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
    Norm → LSTM → Dropout → Residual(+Input)
           ↓
    Norm → FFN → Dropout → Residual(+Previous)
           ↓
    Output(shape=[batch, seq_len, embed_dim])
    ```

    **2. Transformer Mode** (`block_type='transformer'`):
    ```
    Input(shape=[batch, seq_len, embed_dim])
           ↓
    Norm → MultiHeadAttention → Dropout → Residual(+Input)
           ↓
    Norm → FFN → Dropout → Residual(+Previous)
           ↓
    Output(shape=[batch, seq_len, embed_dim])
    ```

    **3. Mixed Mode** (`block_type='mixed'`):
    ```
    Input(shape=[batch, seq_len, embed_dim])
           ↓
    Norm → LSTM → Dropout → Residual(+Input)
           ↓
    Norm → MultiHeadAttention → Dropout → Residual(+Previous)
           ↓
    Norm → FFN → Dropout → Residual(+Previous)
           ↓
    Output(shape=[batch, seq_len, embed_dim])
    ```

    **Key Features**:
    - Pre-LayerNorm architecture for training stability
    - Residual connections for gradient flow
    - Configurable LSTM and attention components using factory patterns
    - Dropout regularization at each stage
    - Optional dimension projection for LSTM outputs
    - Unified factory-based component creation

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
        use_layer_norm: Boolean, whether to apply normalization before each sub-layer.
            Following Pre-LN architecture. Defaults to True.
        normalization_type: String, type of normalization to use from factory.
            Options include 'layer_norm', 'rms_norm', 'batch_norm', etc.
            Defaults to 'rms_norm'.
        attention_type: String, type of attention mechanism from factory.
            Options include 'multi_head', 'anchor', 'differential', etc.
            Defaults to 'multi_head'.
        ffn_type: String, type of feed-forward network from factory.
            Options include 'mlp', 'swiglu', 'glu', etc. Defaults to 'mlp'.
        activation: String or callable, activation function for feed-forward network.
            Applied in the FFN layer. Defaults to 'relu'.
        normalization_args: Optional dictionary of additional arguments for normalization layers.
            Passed to the normalization factory. Defaults to None.
        attention_args: Optional dictionary of additional arguments for attention layer.
            Passed to the attention factory. Defaults to None.
        ffn_args: Optional dictionary of additional arguments for FFN layer.
            Passed to the FFN factory. Defaults to None.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        3D tensor with shape: `(batch_size, sequence_length, embed_dim)`.
        Typical time series input where sequence_length is the time dimension.

    Output shape:
        3D tensor with shape: `(batch_size, sequence_length, embed_dim)`.
        Shape is preserved through all processing modes.

    Attributes:
        lstm_layer: LSTM layer (lstm/mixed modes only).
        attention_layer: Attention layer from factory (transformer/mixed modes only).
        projection: Dense layer for LSTM output projection (if lstm_units != embed_dim).
        norm1, norm2, norm3: Normalization layers from factory for Pre-LN architecture.
        ffn_layer: Feed-forward network layer from factory.
        dropout1, dropout2, dropout3: Dropout layers for regularization.

    Example:
        ```python
        # Mixed LSTM + Attention block for time series
        block = MixedSequentialBlock(
            embed_dim=256,
            num_heads=8,
            block_type='mixed',
            normalization_type='rms_norm',
            attention_type='multi_head',
            ffn_type='swiglu'
        )

        # Time series input: 64 time steps, 256 features
        inputs = keras.Input(shape=(64, 256))
        outputs = block(inputs)  # Shape: (batch, 64, 256)

        # LSTM-only block with custom arguments
        lstm_block = MixedSequentialBlock(
            embed_dim=128,
            lstm_units=256,
            block_type='lstm',
            dropout_rate=0.2,
            ffn_type='glu',
            ffn_args={'dropout_rate': 0.15}
        )

        # Transformer-only block with differential attention
        transformer_block = MixedSequentialBlock(
            embed_dim=512,
            num_heads=16,
            block_type='transformer',
            attention_type='differential',
            ffn_type='swiglu',
            attention_args={'lambda_init': 0.9}
        )
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
        normalization_type: str = 'rms_norm',
        attention_type: str = 'multi_head',
        ffn_type: str = 'mlp',
        activation: Union[str, Callable] = 'relu',
        normalization_args: Optional[Dict[str, Any]] = None,
        attention_args: Optional[Dict[str, Any]] = None,
        ffn_args: Optional[Dict[str, Any]] = None,
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

        # Store ALL configuration parameters
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.lstm_units = lstm_units if lstm_units is not None else embed_dim
        self.ff_dim = ff_dim if ff_dim is not None else embed_dim * 4
        self.block_type = block_type
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.normalization_type = normalization_type
        self.attention_type = attention_type
        self.ffn_type = ffn_type
        self.activation = activation
        self.normalization_args = normalization_args or {}
        self.attention_args = attention_args or {}
        self.ffn_args = ffn_args or {}

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
            # Prepare attention arguments with required parameters
            attention_kwargs = self.attention_args.copy()

            # Map attention type to required parameters
            if self.attention_type == 'multi_head':
                attention_kwargs.update({
                    'embed_dim': self.embed_dim,
                    'num_heads': self.num_heads,
                    'dropout_rate': self.dropout_rate
                })
            elif self.attention_type == 'differential':
                attention_kwargs.update({
                    'dim': self.embed_dim,
                    'num_heads': self.num_heads,
                    'head_dim': self.embed_dim // self.num_heads,
                    'dropout': self.dropout_rate
                })
            elif self.attention_type in ['anchor', 'perceiver']:
                attention_kwargs.update({
                    'dim': self.embed_dim,
                    'num_heads': self.num_heads,
                    'dropout': self.dropout_rate
                })
            elif self.attention_type == 'adaptive_multi_head':
                attention_kwargs.update({
                    'num_heads': self.num_heads,
                    'key_dim': self.embed_dim // self.num_heads,
                    'dropout': self.dropout_rate
                })
            else:
                # Default parameters for other attention types
                attention_kwargs.update({
                    'dim': self.embed_dim,
                    'num_heads': self.num_heads
                })

            self.attention_layer = create_attention_layer(
                attention_type=self.attention_type,
                name="attention",
                **attention_kwargs
            )
        else:
            self.attention_layer = None

        # Normalization layers (Pre-LN architecture)
        if self.use_layer_norm:
            self.norm1 = create_normalization_layer(
                normalization_type=self.normalization_type,
                name="norm1",
                **self.normalization_args
            )
            self.norm2 = create_normalization_layer(
                normalization_type=self.normalization_type,
                name="norm2",
                **self.normalization_args
            )
            # Mixed mode needs an extra norm layer
            if self.block_type == 'mixed':
                self.norm3 = create_normalization_layer(
                    normalization_type=self.normalization_type,
                    name="norm3",
                    **self.normalization_args
                )
            else:
                self.norm3 = None
        else:
            self.norm1 = None
            self.norm2 = None
            self.norm3 = None

        # Feed-forward network using factory
        ffn_kwargs = self.ffn_args.copy()

        # Map FFN type to required parameters
        if self.ffn_type in ['mlp', 'differential', 'glu', 'geglu', 'residual']:
            ffn_kwargs.update({
                'hidden_dim': self.ff_dim,
                'output_dim': self.embed_dim,
                'activation': self.activation,
                'dropout_rate': self.dropout_rate
            })
        elif self.ffn_type == 'swiglu':
            ffn_kwargs.update({
                'output_dim': self.embed_dim,
                'ffn_expansion_factor': self.ff_dim // self.embed_dim,
                'dropout_rate': self.dropout_rate
            })
        elif self.ffn_type == 'swin_mlp':
            ffn_kwargs.update({
                'hidden_dim': self.ff_dim,
                'output_dim': self.embed_dim,
                'activation': self.activation,
                'dropout_rate': self.dropout_rate
            })
        else:
            # Default parameters
            ffn_kwargs.update({
                'hidden_dim': self.ff_dim,
                'output_dim': self.embed_dim
            })

        self.ffn_layer = create_ffn_layer(
            ffn_type=self.ffn_type,
            name="ffn",
            **ffn_kwargs
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
        self.ffn_layer.build(input_shape)

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
        attn_output = self.attention_layer(norm_input, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        x = x + attn_output  # Residual connection

        # Second Sub-layer: Feed-Forward Network
        norm_output = self.norm2(x, training=training) if self.use_layer_norm else x
        ff_output = self.ffn_layer(norm_output, training=training)
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
        ff_output = self.ffn_layer(norm_output, training=training)
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
        attn_output = self.attention_layer(norm3_input, training=training)
        attn_output = self.dropout3(attn_output, training=training)
        x = x + attn_output  # Residual 2

        # Block 3: Feed-Forward Network
        norm2_output = self.norm2(x, training=training) if self.use_layer_norm else x
        ff_output = self.ffn_layer(norm2_output, training=training)
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

    def get_config(self) -> Dict[str, Any]:
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
            "normalization_type": self.normalization_type,
            "attention_type": self.attention_type,
            "ffn_type": self.ffn_type,
            "activation": self.activation,
            "normalization_args": self.normalization_args,
            "attention_args": self.attention_args,
            "ffn_args": self.ffn_args,
        })
        return config