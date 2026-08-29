"""
A hybrid sequential block that combines an LSTM with self-attention.

This module holds one layer, :class:`MixedSequentialBlock`. It is a building
block for deep time series models and runs in three modes: LSTM only,
Transformer only, or both in sequence. The two mechanisms live in one class
because they capture different things.

-   An LSTM walks the sequence step by step. It is good at local order and at
    state that evolves over time.
-   Self-attention lets every step read every other step directly. It is good
    at long-range links that depend on content, not on distance.

The ``mixed`` mode runs the LSTM first and attention second. The idea is that
the LSTM gives each step a summary of its own recent history. Attention then
compares context-rich steps instead of raw ones.

All three modes use Pre-LN. Normalization runs *before* each sub-layer, not
after. Pre-LN trains more stably in deep stacks than the original Post-LN
order. Every sub-layer ends in a residual add, so the block returns its input
shape unchanged.

Self-attention is::

    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

The normalization, attention and FFN sub-layers are built by the shared
factories in ``layers/norms``, ``layers/attention`` and ``layers/ffn``. All
three factories raise on a keyword the chosen type does not accept, which is
what makes forwarding a caller's arguments awkward here. Two anchors below
record how this block handles that: D-011 pre-filters one allowlisted key
before calling the attention factory, and D-021 renames one key for the
``differential`` FFN.

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
from ..ffn.factory import assemble_ffn_config
from ..attention import create_attention_layer, AttentionType
from ..attention.factory import ATTENTION_REGISTRY, assemble_attention_config
from ..norms import create_normalization_layer, NormalizationType
from dl_techniques.utils.logger import logger
from dl_techniques.utils.activation_serialization import (
    serialize_activation,
    deserialize_activation,
)
from dl_techniques.utils.keras_registration import register_dl_technique

# DECISION plan-2026-08-17T183311-79c63e38/D-011: caller `attention_args` keys
# this block may drop when the chosen attention_type rejects them. Adding a key
# here hides a caller typo in that key from the factory, so keep the tuple short.
# The full rule lives on the drop loop in __init__ (search this name). See D-011.
_CONDITIONAL_ATTENTION_ARG_KEYS: Tuple[str, ...] = ('window_size',)

# ---------------------------------------------------------------------

# The three modes of MixedSequentialBlock. 'lstm' runs LSTM + FFN,
# 'transformer' runs attention + FFN, 'mixed' runs LSTM + attention + FFN.
BlockType = Literal['lstm', 'transformer', 'mixed']


@register_dl_technique("dl_techniques.layers.time_series.mixed_sequential_block")
class MixedSequentialBlock(keras.layers.Layer):
    """
    Hybrid sequential block combining LSTM and self-attention for time series.

    The block runs in three modes, chosen by ``block_type``: LSTM only,
    Transformer only, or both in sequence. Every mode is Pre-LN with a residual
    add per sub-layer, so the output shape always equals the input shape.

    In ``mixed`` mode the LSTM runs first and attention second. The LSTM picks
    up local temporal structure, attention picks up long-range structure, and
    the FFN adds the non-linear mixing. That ordering is the reason to use this
    layer instead of a plain Transformer block.

    Sub-layers are built by the shared factories, so ``normalization_type``,
    ``attention_type`` and ``ffn_type`` select from those registries. The three
    ``*_args`` dicts are passed through to them. All three factories raise on a
    key the chosen type does not accept, with one allowlisted exception; see the
    D-011 anchors in ``__init__``.

    **Architecture Overview:**

    .. code-block:: text

        Input x  [B, T, embed_dim]
                          │
                     block_type
            ┌─────────────┼─────────────┐
            ▼             ▼             ▼
         'lstm'     'transformer'    'mixed'
            │             │             │
            ▼             ▼             ▼
      ┌───────────┐ ┌───────────┐ ┌───────────┐
      │ LSTM sub  │ │ Attn sub  │ │ LSTM sub  │
      └─────┬─────┘ └─────┬─────┘ └─────┬─────┘
            │             │             ▼
            │             │       ┌───────────┐
            │             │       │ Attn sub  │
            │             │       └─────┬─────┘
            ▼             ▼             ▼
      ┌───────────┐ ┌───────────┐ ┌───────────┐
      │ FFN sub   │ │ FFN sub   │ │ FFN sub   │
      └─────┬─────┘ └─────┬─────┘ └─────┬─────┘
            └─────────────┴─────────────┘
                          ▼
        Output x  [B, T, embed_dim]

    Each box is one Pre-LN sub-layer: normalize, transform, drop out, add the
    residual. The per-sub-layer diagrams are on ``_lstm_block``,
    ``_transformer_block`` and ``_mixed_block``. Only ``mixed`` builds ``norm3``
    and ``dropout3``; the other two modes leave them ``None``.

    :param embed_dim: Embedding dimension, and also the output dimension. Must
        be positive.
    :type embed_dim: int
    :param num_heads: Number of attention heads, used by the 'transformer' and
        'mixed' modes. Must divide ``embed_dim`` exactly.
    :type num_heads: int
    :param lstm_units: Number of LSTM units, used by the 'lstm' and 'mixed'
        modes. Defaults to ``embed_dim``. A different value adds a Dense
        projection back to ``embed_dim`` so the residual add still lines up.
    :type lstm_units: int or None
    :param ff_dim: Hidden width of the feed-forward network. Defaults to
        ``embed_dim * 4``.
    :type ff_dim: int or None
    :param block_type: Which mode to run: 'lstm', 'transformer' or 'mixed'.
    :type block_type: str
    :param dropout_rate: Dropout rate for every dropout layer, in [0, 1].
    :type dropout_rate: float
    :param use_layer_norm: Normalize before each sub-layer (Pre-LN). When
        False, no normalization layer is built at all and each sub-layer reads
        its input directly.
    :type use_layer_norm: bool
    :param normalization_type: Key into the normalization factory, for example
        'layer_norm', 'rms_norm' or 'batch_norm'.
    :type normalization_type: str
    :param attention_type: Key into the attention factory, for example
        'multi_head', 'window', 'anchor' or 'differential'.
    :type attention_type: str
    :param ffn_type: Key into the FFN factory, for example 'mlp', 'swiglu' or
        'glu'.
    :type ffn_type: str
    :param activation: Activation for the feed-forward network. Not every FFN
        type takes one; see the branches in ``__init__``.
    :type activation: str or callable
    :param normalization_args: Extra keywords for the normalization layers.
        Passed to the factory unchanged.
    :type normalization_args: dict or None
    :param attention_args: Extra keywords for the attention layer. Merged on
        top of this block's own defaults, so a caller value wins.
    :type attention_args: dict or None
    :param ffn_args: Extra keywords for the FFN layer. Merged last and never
        filtered, so a caller value wins.
    :type ffn_args: dict or None
    :param kwargs: Additional keyword arguments for the Layer base class.

    :raises ValueError: If ``embed_dim``, ``num_heads``, ``lstm_units``,
        ``ff_dim`` or ``dropout_rate`` is out of range, if ``embed_dim`` is not
        divisible by ``num_heads``, or if ``block_type`` is not one of 'lstm',
        'transformer', 'mixed'.

    Input shape:
        3D tensor of shape ``(batch, seq_len, embed_dim)``.

    Output shape:
        3D tensor of shape ``(batch, seq_len, embed_dim)``. Same as the input.

    Example:
        .. code-block:: python

            block = MixedSequentialBlock(
                embed_dim=64,
                num_heads=8,
                block_type='mixed',
                ffn_type='swiglu',
            )
            y = block(keras.random.normal((2, 32, 64)))

    :ivar lstm_layer: The LSTM, or None outside 'lstm'/'mixed' mode.
    :vartype lstm_layer: keras.layers.LSTM or None
    :ivar projection: Dense map from ``lstm_units`` back to ``embed_dim``, or
        None when the two are equal.
    :vartype projection: keras.layers.Dense or None
    :ivar attention_layer: The attention layer, or None in 'lstm' mode.
    :vartype attention_layer: keras.layers.Layer or None
    :ivar ffn_layer: The feed-forward network. Built in every mode.
    :vartype ffn_layer: keras.layers.Layer
    :ivar norm1: Normalization before the first sub-layer, or None when
        ``use_layer_norm`` is False.
    :vartype norm1: keras.layers.Layer or None
    :ivar norm2: Normalization before the FFN sub-layer, or None.
    :vartype norm2: keras.layers.Layer or None
    :ivar norm3: Normalization before the attention sub-layer in 'mixed' mode.
        None in the other two modes.
    :vartype norm3: keras.layers.Layer or None
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
        self.activation = deserialize_activation(activation)
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
            # This block's own defaults, derived from its own hyperparameters.
            # `assemble_attention_config` filters these against the chosen
            # attention type, then merges the caller's `attention_args` on top
            # unfiltered, so a caller typo still reaches the factory's raise.
            if self.attention_type == 'multi_head':
                attention_defaults = {
                    'dim': self.embed_dim,
                    'num_heads': self.num_heads,
                    'dropout_rate': self.dropout_rate
                }
            elif self.attention_type == 'window':
                # DECISION plan-2026-08-17T183311-79c63e38/D-011: a
                # `'normalization': 'softmax'` default here was REMOVED, not
                # relocated. WindowAttention declares no such parameter (3
                # required + 17 optional, MEASURED), so it was discarded on
                # every build. Pass `probability_type` instead. See D-011.
                attention_defaults = {
                    'dim': self.embed_dim,
                    'num_heads': self.num_heads,
                    'dropout_rate': self.dropout_rate,
                    'window_size': 8
                }
            elif self.attention_type == 'differential':
                attention_defaults = {
                    'dim': self.embed_dim,
                    'num_heads': self.num_heads,
                    'head_dim': self.embed_dim // self.num_heads,
                    'dropout_rate': self.dropout_rate
                }
            elif self.attention_type in ['anchor', 'perceiver']:
                attention_defaults = {
                    'dim': self.embed_dim,
                    'num_heads': self.num_heads,
                    'dropout_rate': self.dropout_rate
                }
            elif self.attention_type == 'adaptive_multi_head':
                attention_defaults = {
                    'num_heads': self.num_heads,
                    'key_dim': self.embed_dim // self.num_heads,
                    'dropout_rate': self.dropout_rate
                }
            else:
                # Default parameters for other attention types
                attention_defaults = {
                    'dim': self.embed_dim,
                    'num_heads': self.num_heads
                }

            attention_kwargs = assemble_attention_config(
                self.attention_type, attention_defaults, self.attention_args
            )

            # DECISION plan-2026-08-17T183311-79c63e38/D-011: drop only the
            # allowlisted keys the chosen type rejects. None of `multi_head`'s
            # 11 registry params is `window_size`, so TiReX, which wires it at
            # every type, raises at its own default without this. Do NOT filter
            # the whole merged dict, nor push the test into callers. See D-011.
            _accepted = set(
                ATTENTION_REGISTRY[self.attention_type]['required_params']
            ) | set(ATTENTION_REGISTRY[self.attention_type]['optional_params'])
            for conditional_key in _CONDITIONAL_ATTENTION_ARG_KEYS:
                if conditional_key in attention_kwargs and (
                        conditional_key not in _accepted):
                    logger.debug(
                        f"MixedSequentialBlock: dropping conditional attention "
                        f"arg '{conditional_key}', which "
                        f"attention_type='{self.attention_type}' does not accept."
                    )
                    attention_kwargs.pop(conditional_key)

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

        # Feed-forward network, built through the factory.
        #
        # This block's own defaults go into `ffn_config`. The caller's
        # `self.ffn_args` is the third argument to `assemble_ffn_config`, so it
        # is merged last and is never filtered (D-017). Do not fold `ffn_args`
        # into `ffn_config`. That ordering would let this block's defaults
        # override what the caller asked for, and it would hide a caller typo
        # from `create_ffn_layer`, which raises on an unknown key.
        ffn_config: Dict[str, Any] = {
            'hidden_dim': self.ff_dim,
            'output_dim': self.embed_dim,
        }
        if self.ffn_type in ['mlp', 'glu', 'geglu', 'residual', 'swin_mlp']:
            ffn_config['activation'] = self.activation
            ffn_config['dropout_rate'] = self.dropout_rate
        elif self.ffn_type == 'differential':
            # DECISION plan-2026-07-30T140922-8af1028f/D-021: RENAME, do not
            # drop. DifferentialFFN takes `branch_activation`, so `activation`
            # was silently discarded by the pre-filter on every build here.
            # `gate_activation` stays unforwarded: the sigmoid gate defines the
            # layer. Do NOT merge this branch into the generic one. See D-021.
            ffn_config['branch_activation'] = self.activation
            ffn_config['dropout_rate'] = self.dropout_rate
        elif self.ffn_type == 'swiglu':
            # SwiGLU sizes itself from `ffn_expansion_factor`; passing
            # `hidden_dim` (which it accepts as OPTIONAL) would override that.
            del ffn_config['hidden_dim']
            ffn_config['ffn_expansion_factor'] = self.ff_dim // self.embed_dim
            ffn_config['dropout_rate'] = self.dropout_rate

        self.ffn_layer = create_ffn_layer(
            ffn_type=self.ffn_type,
            name="ffn",
            **assemble_ffn_config(self.ffn_type, ffn_config, self.ffn_args)
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
        Build the layer and every sub-layer that this mode created.

        Each sub-layer is built explicitly so that the weights exist before the
        first call and survive a save/load round trip. Sub-layers left as None
        by ``__init__`` are skipped. Only the LSTM projection gets a different
        shape, since it reads ``lstm_units`` channels rather than ``embed_dim``.

        :param input_shape: Shape of the input tensor,
            ``(batch, seq_len, embed_dim)``.
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
        Run the 'transformer' mode: attention, then FFN, both Pre-LN.

        **Block Internals:**

        .. code-block:: text

             x  [B, T, D]
             │
             ├──────────────┐
             │              ▼
             │      ┌───────────────┐
             │      │ Norm1 (opt)   │
             │      │ Attention     │
             │      │ Dropout1      │
             │      └───────┬───────┘
             │              │
            (+) ◄───────────┘
             │
             ├──────────────┐
             │              ▼
             │      ┌───────────────┐
             │      │ Norm2 (opt)   │
             │      │ FFN           │
             │      │ Dropout2      │
             │      └───────┬───────┘
             │              │
            (+) ◄───────────┘
             │
             ▼
            out  [B, T, D]

        ``(opt)`` marks the normalization layers, which are skipped when
        ``use_layer_norm`` is False. ``mask`` is accepted for signature parity
        with the other two blocks and is not used here.

        :param inputs: Input tensor of shape (batch, seq_len, embed_dim).
        :type inputs: keras.KerasTensor
        :param training: Whether in training mode.
        :type training: bool or None
        :param mask: Attention mask. Not forwarded to the attention layer.
        :type mask: keras.KerasTensor or None
        :return: Transformed tensor with the same shape as the input.
        :rtype: keras.KerasTensor
        """
        x = inputs

        # First Sub-layer: Multi-head Self-Attention
        norm_input = self.norm1(x, training=training) if self.use_layer_norm else x
        attn_output = self.attention_layer(norm_input, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        x = x + attn_output

        # Second Sub-layer: Feed-Forward Network
        norm_output = self.norm2(x, training=training) if self.use_layer_norm else x
        ff_output = self.ffn_layer(norm_output, training=training)
        ff_output = self.dropout2(ff_output, training=training)
        return x + ff_output

    def _lstm_block(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
        mask: Optional[keras.KerasTensor] = None
    ) -> keras.KerasTensor:
        """
        Run the 'lstm' mode: LSTM, then FFN, both Pre-LN.

        **Block Internals:**

        .. code-block:: text

             x  [B, T, D]
             │
             ├──────────────┐
             │              ▼
             │      ┌───────────────┐
             │      │ Norm1 (opt)   │
             │      │ LSTM          │
             │      │ Project (opt) │
             │      │ Dropout1      │
             │      └───────┬───────┘
             │              │
            (+) ◄───────────┘
             │
             ├──────────────┐
             │              ▼
             │      ┌───────────────┐
             │      │ Norm2 (opt)   │
             │      │ FFN           │
             │      │ Dropout2      │
             │      └───────┬───────┘
             │              │
            (+) ◄───────────┘
             │
             ▼
            out  [B, T, D]

        The norm layers are skipped when ``use_layer_norm`` is False. The
        projection exists only when ``lstm_units != embed_dim``; the LSTM
        output is ``[B, T, lstm_units]`` and the projection is what brings it
        back to ``D`` so the residual add works. ``mask`` reaches the LSTM.

        :param inputs: Input tensor of shape (batch, seq_len, embed_dim).
        :type inputs: keras.KerasTensor
        :param training: Whether in training mode.
        :type training: bool or None
        :param mask: Sequence mask, forwarded to the LSTM.
        :type mask: keras.KerasTensor or None
        :return: Transformed tensor with the same shape as the input.
        :rtype: keras.KerasTensor
        """
        x = inputs

        # First Sub-layer: LSTM
        norm_input = self.norm1(x, training=training) if self.use_layer_norm else x
        lstm_output = self.lstm_layer(norm_input, training=training, mask=mask)
        if self.projection is not None:
            lstm_output = self.projection(lstm_output, training=training)
        lstm_output = self.dropout1(lstm_output, training=training)
        x = x + lstm_output

        # Second Sub-layer: Feed-Forward Network
        norm_output = self.norm2(x, training=training) if self.use_layer_norm else x
        ff_output = self.ffn_layer(norm_output, training=training)
        ff_output = self.dropout2(ff_output, training=training)
        return x + ff_output

    def _mixed_block(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
        mask: Optional[keras.KerasTensor] = None
    ) -> keras.KerasTensor:
        """
        Run the 'mixed' mode: LSTM, then attention, then FFN, all Pre-LN.

        **Block Internals:**

        .. code-block:: text

             x  [B, T, D]
             │
             ├──────────────┐
             │              ▼
             │      ┌───────────────┐
             │      │ Norm1 (opt)   │
             │      │ LSTM          │
             │      │ Project (opt) │
             │      │ Dropout1      │
             │      └───────┬───────┘
             │              │
            (+) ◄───────────┘
             │
             ├──────────────┐
             │              ▼
             │      ┌───────────────┐
             │      │ Norm3 (opt)   │
             │      │ Attention     │
             │      │ Dropout3      │
             │      └───────┬───────┘
             │              │
            (+) ◄───────────┘
             │
             ├──────────────┐
             │              ▼
             │      ┌───────────────┐
             │      │ Norm2 (opt)   │
             │      │ FFN           │
             │      │ Dropout2      │
             │      └───────┬───────┘
             │              │
            (+) ◄───────────┘
             │
             ▼
            out  [B, T, D]

        Note the norm order: the attention sub-layer uses ``norm3`` and the FFN
        sub-layer uses ``norm2``. That is why ``norm3`` and ``dropout3`` exist
        only in this mode. Attention sees the LSTM output, not the raw input.
        ``mask`` reaches the LSTM only.

        :param inputs: Input tensor of shape (batch, seq_len, embed_dim).
        :type inputs: keras.KerasTensor
        :param training: Whether in training mode.
        :type training: bool or None
        :param mask: Sequence mask, forwarded to the LSTM.
        :type mask: keras.KerasTensor or None
        :return: Transformed tensor with the same shape as the input.
        :rtype: keras.KerasTensor
        """
        x = inputs

        # Block 1: LSTM
        norm1_input = self.norm1(x, training=training) if self.use_layer_norm else x
        lstm_output = self.lstm_layer(norm1_input, training=training, mask=mask)
        if self.projection is not None:
            lstm_output = self.projection(lstm_output, training=training)
        lstm_output = self.dropout1(lstm_output, training=training)
        x = x + lstm_output

        # Block 2: Attention
        norm3_input = self.norm3(x, training=training) if self.use_layer_norm else x
        attn_output = self.attention_layer(norm3_input, training=training)
        attn_output = self.dropout3(attn_output, training=training)
        x = x + attn_output

        # Block 3: Feed-Forward Network
        norm2_output = self.norm2(x, training=training) if self.use_layer_norm else x
        ff_output = self.ffn_layer(norm2_output, training=training)
        ff_output = self.dropout2(ff_output, training=training)
        return x + ff_output

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
        mask: Optional[keras.KerasTensor] = None
    ) -> keras.KerasTensor:
        """
        Forward pass. Dispatches to the block for the configured mode.

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
        Return the config needed to rebuild this layer.

        Every ``__init__`` parameter is returned. ``activation`` goes through
        ``serialize_activation`` so a callable survives the round trip.

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
            "activation": serialize_activation(self.activation),
            "normalization_args": self.normalization_args,
            "attention_args": self.attention_args,
            "ffn_args": self.ffn_args,
        })
        return config

# ---------------------------------------------------------------------
