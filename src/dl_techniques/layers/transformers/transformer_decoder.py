"""Canonical cross-attention transformer decoder block.

This module provides :class:`TransformerDecoderLayer`, the encoder-decoder
counterpart to :class:`~dl_techniques.layers.transformers.transformer.TransformerLayer`.
Where ``TransformerLayer`` implements a single self-attention sub-block (and is
therefore self-attention only), the decoder block composes **three** residual
sub-blocks:

1. Masked / causal self-attention over the decoder sequence.
2. Cross-attention from the decoder sequence (queries) to an external encoder
   memory (keys / values).
3. A position-wise feed-forward network.

All three sub-components are constructed through the shared component factories
(:func:`create_attention_layer`, :func:`create_ffn_from_config`,
:func:`create_normalization_layer`), so attention / FFN / normalization variants
are configurable without subclassing. The cross-attention sub-block is built on
:class:`~dl_techniques.layers.attention.multi_head_cross_attention.MultiHeadCrossAttention`
(factory key ``'multi_head_cross'``), which performs cross-attention when given
distinct ``query_input`` and ``kv_input`` tensors.

``Attention(Q, K, V) = softmax((Q K^T) / sqrt(d_k)) V``

The block exists to make cross-attention transformer blocks elsewhere in the
repo (e.g. DETR-style decoders, VLM / denoiser / memory cross-attention) replaceable
by a canonical, fully-serializable implementation.
"""

import keras
from keras import ops, initializers, regularizers
from typing import Optional, Union, Any, Dict, Tuple, Literal, Callable

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..ffn import create_ffn_from_config, FFNType
from ..attention import create_attention_layer, AttentionType
from ..norms import create_normalization_layer, NormalizationType

# ---------------------------------------------------------------------

NormalizationPositionType = Literal['post', 'pre']


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class TransformerDecoderLayer(keras.layers.Layer):
    """Encoder-decoder transformer block: masked self-attention + cross-attention + FFN.

    Implements a standard transformer *decoder* layer. Each of the three
    sub-blocks (self-attention, cross-attention, FFN) is wrapped in a residual
    connection and a normalization layer; the data flow is determined by
    ``normalization_position`` (``'pre'`` or ``'post'``), mirroring
    :class:`TransformerLayer` exactly so the two compose predictably in a stack.

    **Architecture Overview:**

    .. code-block:: text

        decoder_input (B, T, H)        encoder_output (B, S, H)
              │                               │
              ▼                               │
        [Norm] ─► Self-Attn(causal) ─► +Residual
              │                               │
              ▼                               ▼
        [Norm] ─► Cross-Attn(query=dec, kv=enc) ─► +Residual
              │
              ▼
        [Norm] ─► FFN ─► [Dropout] ─► +Residual
              │
              ▼
        output (B, T, H)

    :param hidden_size: Hidden dimension of the layer.
    :param num_heads: Number of attention heads (shared by self/cross attention).
    :param intermediate_size: FFN intermediate dimension.
    :param self_attention_type: Factory key for the self-attention sub-block.
        Default ``'multi_head'``. Non-default keys must be self-attention
        compatible and may require ``attention_args``.
    :param cross_attention_type: Factory key for the cross-attention sub-block.
        Default ``'multi_head_cross'`` (the canonical cross-attention primitive).
    :param attention_args: Extra args forwarded to the self-attention factory
        (override defaults).
    :param cross_attention_args: Extra args forwarded to the cross-attention factory.
    :param normalization_type: Normalization type. Default ``'layer_norm'``.
    :param normalization_position: ``'pre'`` or ``'post'``. Default ``'post'``.
    :param ffn_type: FFN architecture type. Default ``'mlp'``.
    :param ffn_args: Extra args forwarded to the FFN factory.
    :param dropout_rate: FFN output dropout rate. Default 0.1.
    :param attention_dropout_rate: Attention dropout rate. Default 0.1.
    :param use_causal_mask: If True and no ``self_attention_mask`` is supplied at
        call time, a causal (lower-triangular) keep-mask is generated so each
        decoder position attends only to itself and earlier positions.
    :param activation: FFN activation. Default ``'gelu'``.
    :param use_bias: Whether linear layers use bias. Default True.
    :param kernel_initializer: Kernel initializer.
    :param bias_initializer: Bias initializer.
    :param kernel_regularizer: Kernel regularizer.
    :param bias_regularizer: Bias regularizer.

    :raises ValueError: If dimension parameters are invalid.
    """

    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            intermediate_size: int,
            self_attention_type: AttentionType = 'multi_head',
            cross_attention_type: AttentionType = 'multi_head_cross',
            attention_args: Optional[Dict[str, Any]] = None,
            cross_attention_args: Optional[Dict[str, Any]] = None,
            normalization_type: NormalizationType = 'layer_norm',
            normalization_position: NormalizationPositionType = 'post',
            ffn_type: FFNType = 'mlp',
            ffn_args: Optional[Dict[str, Any]] = None,
            dropout_rate: float = 0.1,
            attention_dropout_rate: float = 0.1,
            use_causal_mask: bool = True,
            activation: Union[str, Callable] = 'gelu',
            use_bias: bool = True,
            kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            bias_regularizer: Optional[regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # --- Input Validation ---
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
            )
        if intermediate_size <= 0:
            raise ValueError(f"intermediate_size must be positive, got {intermediate_size}")
        if normalization_position not in ('pre', 'post'):
            raise ValueError(
                f"normalization_position must be 'pre' or 'post', got {normalization_position}"
            )

        # --- Configuration Storage ---
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.self_attention_type = self_attention_type
        self.cross_attention_type = cross_attention_type
        self.attention_args = attention_args or {}
        self.cross_attention_args = cross_attention_args or {}
        self.normalization_type = normalization_type
        self.normalization_position = normalization_position
        self.ffn_type = ffn_type
        self.ffn_args = ffn_args or {}
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.use_causal_mask = bool(use_causal_mask)
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        # --- Sub-layer creation (unbuilt) ---
        self.self_attention = create_attention_layer(
            self.self_attention_type, **self._self_attention_params('self_attention')
        )
        # DECISION plan_2026-06-12_0bb1729b/D-001: cross-attention is built on
        # MultiHeadCrossAttention ('multi_head_cross'); given distinct
        # query_input/kv_input it performs cross-attention. 'multi_head' (self)
        # cannot cross-attend, hence a distinct factory key here.
        self.cross_attention = create_attention_layer(
            self.cross_attention_type, **self._cross_attention_params('cross_attention')
        )
        self.ffn_layer = create_ffn_from_config(self._get_ffn_config('ffn'))

        self.self_attention_norm = create_normalization_layer(
            normalization_type=self.normalization_type, name='self_attention_norm'
        )
        self.cross_attention_norm = create_normalization_layer(
            normalization_type=self.normalization_type, name='cross_attention_norm'
        )
        self.ffn_norm = create_normalization_layer(
            normalization_type=self.normalization_type, name='ffn_norm'
        )
        self.dropout = keras.layers.Dropout(self.dropout_rate, name='ffn_dropout')

    # --- Sub-layer param builders ---

    def _self_attention_params(self, name: str) -> Dict[str, Any]:
        params: Dict[str, Any] = {'dim': self.hidden_size, 'num_heads': self.num_heads, 'name': name}
        if self.self_attention_type in ('multi_head', 'multi_head_cross'):
            params['dropout_rate'] = self.attention_dropout_rate
            params['use_bias'] = self.use_bias
        return {**params, **self.attention_args}

    def _cross_attention_params(self, name: str) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            'dim': self.hidden_size,
            'num_heads': self.num_heads,
            'dropout_rate': self.attention_dropout_rate,
            'use_bias': self.use_bias,
            'name': name,
        }
        return {**params, **self.cross_attention_args}

    def _get_ffn_config(self, name: str) -> Dict[str, Any]:
        config: Dict[str, Any] = {
            'type': self.ffn_type,
            'name': name,
            'dropout_rate': self.dropout_rate,
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer,
        }
        if self.ffn_type == 'swiglu':
            config.update({'output_dim': self.hidden_size, 'ffn_expansion_factor': 4, 'ffn_multiple_of': 256})
        elif self.ffn_type in ('mlp', 'glu', 'geglu', 'residual', 'swin_mlp', 'differential'):
            config.update({'hidden_dim': self.intermediate_size, 'output_dim': self.hidden_size, 'activation': self.activation})
        config.update(self.ffn_args)
        return config

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build all sub-layers. ``input_shape`` is the decoder input ``(B, T, H)``.

        The encoder memory shape is unknown at build time, so the cross-attention
        key/value sequence length is built as ``None`` (dynamic).
        """
        if self.built:
            return

        if len(input_shape) != 3:
            raise ValueError(f"Expected 3D decoder input shape, got {len(input_shape)}D: {input_shape}")
        if input_shape[-1] is not None and input_shape[-1] != self.hidden_size:
            raise ValueError(
                f"Input feature dimension ({input_shape[-1]}) must match hidden_size ({self.hidden_size})"
            )

        dec_shape = tuple(input_shape)
        enc_shape = (dec_shape[0], None, self.hidden_size)

        self.self_attention.build(dec_shape)
        # Cross-attention: query = decoder, kv = encoder memory (dynamic seq len).
        self.cross_attention.build([dec_shape, enc_shape])
        self.ffn_layer.build(dec_shape)
        self.self_attention_norm.build(dec_shape)
        self.cross_attention_norm.build(dec_shape)
        self.ffn_norm.build(dec_shape)
        self.dropout.build(dec_shape)

        super().build(input_shape)

    def _causal_keep_mask(self, seq_len: int, dtype: Any) -> keras.KerasTensor:
        """Lower-triangular keep-mask ``(1, T, T)``; ``mask[i, j] = 1 iff j <= i``.

        Built via an arange index comparison (``row >= col``) rather than
        ``ops.tril`` for backend portability and to match the repo's causal-mask
        idiom. The downstream attention applies ``scores + (1 - mask) * -1e9``.
        """
        row = ops.arange(seq_len)[:, None]
        col = ops.arange(seq_len)[None, :]
        mask = ops.cast(row >= col, dtype)
        return mask[None, :, :]

    def call(
            self,
            inputs: keras.KerasTensor,
            encoder_output: keras.KerasTensor,
            self_attention_mask: Optional[keras.KerasTensor] = None,
            cross_attention_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass.

        :param inputs: Decoder input ``(B, T, H)``.
        :param encoder_output: Encoder memory ``(B, S, H)`` (keys/values for cross-attn).
        :param self_attention_mask: Optional keep-mask ``(B, T, T)``. If None and
            ``use_causal_mask`` is True, a causal mask is generated.
        :param cross_attention_mask: Optional keep-mask ``(B, T, S)`` for cross-attn.
        :param training: Training mode flag.
        :return: Decoder output ``(B, T, H)``.
        """
        # Resolve the self-attention mask (causal default).
        self_mask = self_attention_mask
        if self_mask is None and self.use_causal_mask:
            self_mask = self._causal_keep_mask(ops.shape(inputs)[1], inputs.dtype)

        if self.normalization_position == 'pre':
            # 1. Self-attention
            residual = inputs
            x = self.self_attention_norm(inputs, training=training)
            x = self.self_attention(x, attention_mask=self_mask, training=training)
            x = x + residual

            # 2. Cross-attention
            residual = x
            y = self.cross_attention_norm(x, training=training)
            y = self.cross_attention(y, encoder_output, attention_mask=cross_attention_mask, training=training)
            x = y + residual

            # 3. FFN
            residual = x
            z = self.ffn_norm(x, training=training)
            z = self.ffn_layer(z, training=training)
            z = self.dropout(z, training=training)
            output = z + residual
        else:
            # 1. Self-attention
            residual = inputs
            x = self.self_attention(inputs, attention_mask=self_mask, training=training)
            x = self.self_attention_norm(x + residual, training=training)

            # 2. Cross-attention
            residual = x
            y = self.cross_attention(x, encoder_output, attention_mask=cross_attention_mask, training=training)
            x = self.cross_attention_norm(y + residual, training=training)

            # 3. FFN
            residual = x
            z = self.ffn_layer(x, training=training)
            z = self.dropout(z, training=training)
            output = self.ffn_norm(z + residual, training=training)

        return output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'hidden_size': self.hidden_size,
            'num_heads': self.num_heads,
            'intermediate_size': self.intermediate_size,
            'self_attention_type': self.self_attention_type,
            'cross_attention_type': self.cross_attention_type,
            'attention_args': self.attention_args,
            'cross_attention_args': self.cross_attention_args,
            'normalization_type': self.normalization_type,
            'normalization_position': self.normalization_position,
            'ffn_type': self.ffn_type,
            'ffn_args': self.ffn_args,
            'dropout_rate': self.dropout_rate,
            'attention_dropout_rate': self.attention_dropout_rate,
            'use_causal_mask': self.use_causal_mask,
            'activation': keras.activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        })
        return config
