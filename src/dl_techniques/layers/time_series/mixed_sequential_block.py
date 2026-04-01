"""
A hybrid sequential block combining recurrent and attention mechanisms.

This layer provides a flexible and powerful building block for deep time series
models by unifying the two dominant paradigms for sequence processing:
recurrence (LSTMs) and self-attention (Transformers). It is designed to
capture the complex, multi-faceted dependencies present in time series data.

The block's design is founded on the principle that LSTMs and self-attention
possess complementary inductive biases, each excelling at modeling different
types of temporal patterns:

-   **LSTMs (Recurrence)**: Excel at capturing local, sequential dependencies.
    Their stateful, step-by-step processing makes them inherently adept at
    modeling temporal ordering and evolving states over time.
-   **Self-Attention (Transformers)**: Excels at capturing global, long-range
    dependencies. By allowing every time step to directly interact with every
    other time step, it can identify content-based relationships regardless of
    their distance in the sequence.

The ``mixed`` architecture operationalizes this synergy by processing the input
sequentially: first with an LSTM, then with a self-attention layer. The
hypothesis is that the LSTM first enriches each time step with a summary of
its local, historical context. The self-attention layer then operates on these
context-aware representations, allowing it to model global interactions
between semantically rich, localized events rather than raw time steps.

For training stability, especially in deep architectures, the block adopts the
Pre-Layer Normalization (Pre-LN) structure. Normalization is applied *before*
the main transformation in each sub-layer, which has been shown to promote
smoother gradient flow and more stable training dynamics compared to the
original Post-LN design.

The LSTM core gating mechanism (input, forget, output gates) controls
information flow through the cell. The self-attention mechanism is governed by:
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

References:
    - Hochreiter & Schmidhuber (1997). Long Short-Term Memory.
      https://www.bioinf.jku.at/publications/older/2604.pdf
    - Vaswani et al. (2017). Attention Is All You Need.
      https://arxiv.org/abs/1706.03762
    - Xiong et al. (2020). On Layer Normalization in the Transformer Architecture.
      https://arxiv.org/abs/2002.04745
"""

import keras
from typing import Optional, Union, Tuple, Callable, Any, Literal, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..ffn import create_ffn_layer, FFNType
from ..attention import create_attention_layer, AttentionType
from ..norms import create_normalization_layer, NormalizationType
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

BlockType = Literal['lstm', 'transformer', 'mixed']


@keras.saving.register_keras_serializable()
class MixedSequentialBlock(keras.layers.Layer):
    """
    Hybrid sequential block combining LSTM and self-attention for time series.

    This layer implements a flexible architecture that can operate in three modes:
    LSTM-only, Transformer-only, or a hybrid sequential combination. It uses
    Pre-LayerNorm architecture with residual connections, making it suitable for
    deep time series models where both recurrent processing and self-attention
    are beneficial.

    The mixed mode processes inputs sequentially: LSTM captures local temporal
    patterns, attention captures global dependencies, and FFN provides
    non-linear transformation. This combination is particularly effective
    for long time series with both local and global patterns.

    **Architecture Overview:**

    .. code-block:: text

        Mixed mode (block_type='mixed'):

        Input: x (batch, seq_len, embed_dim)
                        |
                        v
               +--------+---------+
               | Norm1 -> LSTM    |
               | [-> Projection]  |
               | -> Dropout1      |
               +--------+---------+
                        |
                    x + output  (Residual 1)
                        |
                        v
               +--------+---------+
               | Norm3 -> MHA     |
               | -> Dropout3      |
               +--------+---------+
                        |
                    x + output  (Residual 2)
                        |
                        v
               +--------+---------+
               | Norm2 -> FFN     |
               | -> Dropout2      |
               +--------+---------+
                        |
                    x + output  (Residual 3)
                        |
                        v
               Output: (batch, seq_len, embed_dim)

        LSTM mode omits the MHA sub-layer.
        Transformer mode omits the LSTM sub-layer.

    :param embed_dim: Embedding dimension and output dimension. Must be positive.
    :type embed_dim: int
    :param num_heads: Number of attention heads for transformer/mixed modes.
        Must divide evenly into embed_dim.
    :type num_heads: int
    :param lstm_units: Number of LSTM units for lstm/mixed modes.
        Defaults to embed_dim if None.
    :type lstm_units: int or None
    :param ff_dim: Dimension of feed-forward network hidden layer.
        Defaults to embed_dim * 4 if None.
    :type ff_dim: int or None
    :param block_type: Architecture mode: 'lstm', 'transformer', or 'mixed'.
    :type block_type: str
    :param dropout_rate: Dropout rate for all dropout layers (0 to 1).
    :type dropout_rate: float
    :param use_layer_norm: Whether to apply normalization before each sub-layer
        (Pre-LN architecture).
    :type use_layer_norm: bool
    :param normalization_type: Type of normalization from factory
        (e.g., 'layer_norm', 'rms_norm', 'batch_norm').
    :type normalization_type: str
    :param attention_type: Type of attention mechanism from factory
        (e.g., 'multi_head', 'anchor', 'differential').
    :type attention_type: str
    :param ffn_type: Type of feed-forward network from factory
        (e.g., 'mlp', 'swiglu', 'glu').
    :type ffn_type: str
    :param activation: Activation function for the feed-forward network.
    :type activation: str or callable
    :param normalization_args: Additional arguments for normalization layers.
    :type normalization_args: dict or None
    :param attention_args: Additional arguments for attention layer.
    :type attention_args: dict or None
    :param ffn_args: Additional arguments for FFN layer.
    :type ffn_args: dict or None
    :param kwargs: Additional keyword arguments for the Layer base class.

    :raises ValueError: If embed_dim, num_heads, or dropout_rate are invalid,
        or if embed_dim is not divisible by num_heads, or if block_type is
        not one of 'lstm', 'transformer', 'mixed'.
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
        normalization_type: NormalizationType = 'rms_norm',
        attention_type: AttentionType = 'multi_head',
        ffn_type: FFNType = 'mlp',
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
                #dropout=self.dropout_rate,
                #recurrent_dropout=self.dropout_rate,
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
                    'dim': self.embed_dim,
                    'num_heads': self.num_heads,
                    'dropout_rate': self.dropout_rate
                })
            elif self.attention_type == 'window':
                window_defaults = {
                    'dim': self.embed_dim,
                    'num_heads': self.num_heads,
                    'dropout_rate': self.dropout_rate,
                    'window_size': 8,
                    'normalization': 'softmax'
                }
                # Allow attention_args to override defaults (e.g. window_size)
                window_defaults.update(attention_kwargs)
                attention_kwargs = window_defaults
            elif self.attention_type == 'differential':
                attention_kwargs.update({
                    'dim': self.embed_dim,
                    'num_heads': self.num_heads,
                    'head_dim': self.embed_dim // self.num_heads,
                    'dropout_rate': self.dropout_rate
                })
            elif self.attention_type in ['anchor', 'perceiver']:
                attention_kwargs.update({
                    'dim': self.embed_dim,
                    'num_heads': self.num_heads,
                    'dropout_rate': self.dropout_rate
                })
            elif self.attention_type == 'adaptive_multi_head':
                attention_kwargs.update({
                    'num_heads': self.num_heads,
                    'key_dim': self.embed_dim // self.num_heads,
                    'dropout_rate': self.dropout_rate
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

        :param input_shape: Shape of the input tensor.
        :type input_shape: tuple
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
        """
        Execute the standard Pre-LN Transformer block data flow.

        :param inputs: Input tensor of shape (batch, seq_len, embed_dim).
        :type inputs: keras.KerasTensor
        :param training: Whether in training mode.
        :type training: bool or None
        :param mask: Optional attention mask.
        :type mask: keras.KerasTensor or None
        :return: Transformed tensor with same shape as input.
        :rtype: keras.KerasTensor
        """
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
        """
        Execute an LSTM block followed by a feed-forward network.

        :param inputs: Input tensor of shape (batch, seq_len, embed_dim).
        :type inputs: keras.KerasTensor
        :param training: Whether in training mode.
        :type training: bool or None
        :param mask: Optional mask for LSTM.
        :type mask: keras.KerasTensor or None
        :return: Transformed tensor with same shape as input.
        :rtype: keras.KerasTensor
        """
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
        """
        Execute a sequential LSTM, Attention, FFN flow.

        :param inputs: Input tensor of shape (batch, seq_len, embed_dim).
        :type inputs: keras.KerasTensor
        :param training: Whether in training mode.
        :type training: bool or None
        :param mask: Optional mask for LSTM.
        :type mask: keras.KerasTensor or None
        :return: Transformed tensor with same shape as input.
        :rtype: keras.KerasTensor
        """
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
        """
        Forward pass dispatching to the correct block type.

        :param inputs: Input tensor of shape (batch, seq_len, embed_dim).
        :type inputs: keras.KerasTensor
        :param training: Whether in training mode.
        :type training: bool or None
        :param mask: Optional mask tensor.
        :type mask: keras.KerasTensor or None
        :return: Output tensor of shape (batch, seq_len, embed_dim).
        :rtype: keras.KerasTensor

        :raises RuntimeError: If an invalid block_type is encountered.
        """
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
        """
        Compute the output shape, which is identical to the input shape.

        :param input_shape: Shape of the input tensor.
        :type input_shape: tuple
        :return: Output shape, same as input.
        :rtype: tuple
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration dictionary for serialization.

        :return: Configuration dictionary.
        :rtype: dict
        """
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
