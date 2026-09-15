"""Seven layers that make up the Byte Latent Transformer (BLT): ByteTokenizer,
EntropyModel, DynamicPatcher, PatchPooling, LocalEncoder, GlobalTransformer,
and LocalDecoder, using the shared `create_causal_attend_mask` helper from
`dl_techniques.utils.masking`.

BLT replaces a fixed subword vocabulary with entropy-driven patching over raw
UTF-8 bytes. A small causal EntropyModel scores each byte's next-byte
surprise; DynamicPatcher opens a new patch wherever that surprise crosses a
threshold, so predictable stretches merge into large patches and
hard-to-predict stretches get finer-grained compute. LocalEncoder attends over
bytes and pools each patch to one vector; GlobalTransformer attends across
patches; LocalDecoder combines local byte context with the preceding patch's
global representation to produce next-byte logits. Each stack is causal
because every call site hands its `TransformerLayer`s an explicit
`create_causal_attend_mask`; the attention layers mask only with what they are given.
`DynamicPatcher.compute_patch_ids` needs its `seq_len` passed explicitly under
a traced or XLA-compiled graph, since recovering it from the data makes the
output shape data-dependent. Patch slots beyond a sequence's boundary count
are empty rather than masked, and nothing here carries pretrained weights.

References:
    - Pagnoni et al., 2024. Byte Latent Transformer: Patches Scale Better
      Than Tokens. (https://arxiv.org/abs/2412.09871)
"""

import keras
from keras import ops
from typing import Optional, Dict, Any, List, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.masking import create_causal_attend_mask
from dl_techniques.utils.keras_registration import register_dl_technique

from ..transformers.transformer import TransformerLayer
from ..embedding.positional_embedding import PositionalEmbedding

@register_dl_technique("dl_techniques.layers.blt.blt_blocks")
class GlobalTransformer(keras.layers.Layer):
    """Apply causal self-attention across patch representations.

    Models long-range dependencies between patches, over a sequence that is
    much shorter than the underlying byte sequence.

    Architecture:

    .. code-block:: text

        patch representations [B, P, D_g]
              │
              ▼
        ┌──────────────────────────────┐
        │ patch positional embedding   │
        └──────────────────────────────┘
              │
              ▼
        ┌──────────────────────────────┐
        │ transformer layer x N        │◄── causal attend mask
        │ ffn width 4 * D_g            │
        └──────────────────────────────┘
              │
              ▼
        ┌──────────────────────────────┐
        │ layer norm                   │
        └──────────────────────────────┘
              │
              ▼
        contextualized patches [B, P, D_g]

    The mask is causal over the patch axis only. Empty patch slots carry no
    padding mask, so they take part in attention like any other patch.

    :param global_dim: Hidden dimension of the global transformer.
    :type global_dim: int
    :param num_global_layers: Number of transformer layers.
    :type num_global_layers: int
    :param num_heads_global: Number of attention heads.
    :type num_heads_global: int
    :param max_patches: Maximum number of patches per sequence, and the
        positional embedding's length.
    :type max_patches: int
    :param dropout_rate: Dropout rate for all layers.
    :type dropout_rate: float
    :param kwargs: Additional ``keras.layers.Layer`` arguments.
    """

    def __init__(
            self,
            global_dim: int = 768,
            num_global_layers: int = 12,
            num_heads_global: int = 12,
            max_patches: int = 512,
            dropout_rate: float = 0.1,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.global_dim = global_dim
        self.num_global_layers = num_global_layers
        self.num_heads_global = num_heads_global
        self.max_patches = max_patches
        self.dropout_rate = dropout_rate

        self.patch_positional_embedding = PositionalEmbedding(
            max_seq_len=self.max_patches,
            dim=self.global_dim,
            dropout_rate=self.dropout_rate,
            name='patch_positional_embedding'
        )

        self.transformer_layers = []
        for i in range(self.num_global_layers):
            layer = TransformerLayer(
                hidden_size=self.global_dim,
                num_heads=self.num_heads_global,
                intermediate_size=self.global_dim * 4,
                dropout_rate=self.dropout_rate,
                name=f'global_transformer_{i}'
            )
            self.transformer_layers.append(layer)

        self.layer_norm = keras.layers.LayerNormalization(name='global_transformer_norm')

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the global transformer layers.

        :param input_shape: Patch representation shape ``(batch, P, D_g)``.
        :type input_shape: Tuple[Optional[int], ...]
        """
        # Explicit builds, because a lazy first-call build leaves the weights
        # unloadable on a .keras reload.
        self.patch_positional_embedding.build(input_shape)
        pos_embedded_shape = self.patch_positional_embedding.compute_output_shape(input_shape)

        current_shape = pos_embedded_shape
        for layer in self.transformer_layers:
            layer.build(current_shape)
            current_shape = layer.compute_output_shape(current_shape)

        self.layer_norm.build(current_shape)

        super().build(input_shape)

    def call(
            self,
            patch_representations: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Run the global transformer forward.

        :param patch_representations: Patch representations, shape
            ``(batch_size, num_patches, global_dim)``.
        :type patch_representations: keras.KerasTensor
        :param training: Whether in training mode.
        :type training: Optional[bool]
        :return: Contextualized patch representations, same shape as input.
        :rtype: keras.KerasTensor
        """
        x = self.patch_positional_embedding(patch_representations, training=training)

        # Patch p's representation must not depend on the patches after it.
        attend_mask = create_causal_attend_mask(x)
        for layer in self.transformer_layers:
            x = layer(x, attention_mask=attend_mask, training=training)

        x = self.layer_norm(x)

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape.

        :param input_shape: Patch representation shape ``(batch, P, D_g)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: The input shape, since the layer preserves it.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return layer configuration.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'global_dim': self.global_dim,
            'num_global_layers': self.num_global_layers,
            'num_heads_global': self.num_heads_global,
            'max_patches': self.max_patches,
            'dropout_rate': self.dropout_rate
        })
        return config

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.blt.blt_blocks")
class LocalDecoder(keras.layers.Layer):
    """Generate next-byte logits from causal self-attention and patch context.

    Each decoder layer runs causal self-attention over bytes, then
    cross-attends to the preceding patch's global representation, adds that as
    a residual and normalizes. A prediction therefore combines local byte
    history with global context without reading the future.

    Architecture:

    .. code-block:: text

        byte tokens [B, S]        global context [B, P, D_g]
              │                             │
              ▼                             ▼
        ┌──────────────────────────┐  ┌──────────────────────────┐
        │ byte embedding           │  │ dense to D_l             │
        └──────────────────────────┘  │ (only if D_g != D_l)     │
              │                       └──────────────────────────┘
              ▼                             │
        ┌──────────────────────────┐        │
        │ positional embedding     │        │
        └──────────────────────────┘        │
              │ [B, S, D_l]                 │
              ▼                             ▼
        ┌──────────────────────────────────────────┐
        │ x N:                                     │
        │   transformer layer ◄── causal mask      │
        │   cross-attention   ◄── prev-patch keys  │
        │   residual add, then layer norm          │
        └──────────────────────────────────────────┘
              │
              ▼
        ┌──────────────────────────────┐
        │ layer norm                   │
        └──────────────────────────────┘
              │
              ▼
        ┌──────────────────────────────┐
        │ dense to vocab_size          │
        └──────────────────────────────┘
              │
              ▼
        logits [B, S, V]

    Cross-attention gather:

    .. code-block:: text

        patch_ids        0    0    1    1    2
        gather index     0    0    0    0    1    (clamped at 0)
        has_prev         0    0    1    1    1
        key vector       0    0   g_0  g_0  g_1

    :param vocab_size: Size of the byte vocabulary (typically 256 plus
        special tokens).
    :type vocab_size: int
    :param local_dim: Hidden dimension of the local decoder.
    :type local_dim: int
    :param global_dim: Hidden dimension of the global transformer's output. A
        value different from ``local_dim`` adds the context projection.
    :type global_dim: int
    :param num_local_layers: Number of transformer layers in the local decoder.
    :type num_local_layers: int
    :param num_heads_local: Number of attention heads, in both the
        self-attention and the cross-attention.
    :type num_heads_local: int
    :param max_sequence_length: Maximum sequence length in bytes.
    :type max_sequence_length: int
    :param dropout_rate: Dropout rate for all layers.
    :type dropout_rate: float
    :param kwargs: Additional ``keras.layers.Layer`` arguments.
    """

    def __init__(
            self,
            vocab_size: int = 260,
            local_dim: int = 512,
            global_dim: int = 768,
            num_local_layers: int = 6,
            num_heads_local: int = 8,
            max_sequence_length: int = 2048,
            dropout_rate: float = 0.1,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.local_dim = local_dim
        self.global_dim = global_dim
        self.num_local_layers = num_local_layers
        self.num_heads_local = num_heads_local
        self.max_sequence_length = max_sequence_length
        self.dropout_rate = dropout_rate

        self.byte_embedding = keras.layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.local_dim,
            name='decoder_byte_embedding'
        )

        self.positional_embedding = PositionalEmbedding(
            max_seq_len=self.max_sequence_length,
            dim=self.local_dim,
            dropout_rate=self.dropout_rate,
            name='decoder_positional_embedding'
        )

        # Matching widths need no projection, so the attribute stays None.
        self.context_projection = None
        if self.global_dim != self.local_dim:
            self.context_projection = keras.layers.Dense(
                self.local_dim,
                name='context_projection'
            )

        self.decoder_layers = []
        self.cross_attention_layers = []
        self.cross_attention_norms = []

        for i in range(self.num_local_layers):
            decoder_layer = TransformerLayer(
                hidden_size=self.local_dim,
                num_heads=self.num_heads_local,
                intermediate_size=self.local_dim * 4,
                dropout_rate=self.dropout_rate,
                name=f'decoder_transformer_{i}'
            )
            self.decoder_layers.append(decoder_layer)

            cross_attention = keras.layers.MultiHeadAttention(
                num_heads=self.num_heads_local,
                key_dim=max(self.local_dim // self.num_heads_local, 1),
                dropout=self.dropout_rate,
                name=f'cross_attention_{i}'
            )
            self.cross_attention_layers.append(cross_attention)

            cross_norm = keras.layers.LayerNormalization(name=f'cross_attention_norm_{i}')
            self.cross_attention_norms.append(cross_norm)

        self.layer_norm = keras.layers.LayerNormalization(name='decoder_norm')
        self.output_projection = keras.layers.Dense(
            self.vocab_size,
            name='output_projection'
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the local decoder layers.

        :param input_shape: Byte token shape ``(batch, seq_len)``, or a list
            whose first entry is that shape.
        :type input_shape: Tuple[Optional[int], ...]
        """
        byte_input_shape = input_shape[0] if isinstance(input_shape, list) else input_shape
        self.byte_embedding.build(byte_input_shape)

        embedded_shape = self.byte_embedding.compute_output_shape(byte_input_shape)

        self.positional_embedding.build(embedded_shape)
        pos_embedded_shape = self.positional_embedding.compute_output_shape(embedded_shape)

        if self.context_projection is not None:
            global_context_shape = (embedded_shape[0], None, self.global_dim)
            self.context_projection.build(global_context_shape)

        # The gathered keys are one per byte, so their length is the byte length.
        current_shape = pos_embedded_shape
        cross_attention_kv_shape = (current_shape[0], current_shape[1], self.local_dim)

        for i, (decoder_layer, cross_attention, cross_norm) in enumerate(
                zip(self.decoder_layers, self.cross_attention_layers, self.cross_attention_norms)
        ):
            decoder_layer.build(current_shape)
            decoder_output_shape = decoder_layer.compute_output_shape(current_shape)

            # Explicit build, because a lazy first-call build leaves the
            # cross-attention weights unloadable on a .keras reload.
            cross_attention.build(
                decoder_output_shape,
                cross_attention_kv_shape,
                cross_attention_kv_shape,
            )

            cross_norm.build(decoder_output_shape)

            current_shape = decoder_output_shape

        self.layer_norm.build(current_shape)
        self.output_projection.build(current_shape)

        super().build(input_shape)

    def call(
            self,
            byte_tokens: keras.KerasTensor,
            global_context: keras.KerasTensor,
            patch_ids: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Run the local decoder forward.

        :param byte_tokens: Byte tokens, shape ``(batch_size, seq_len)``.
        :type byte_tokens: keras.KerasTensor
        :param global_context: Global patch representations, shape
            ``(batch_size, num_patches, global_dim)``.
        :type global_context: keras.KerasTensor
        :param patch_ids: Patch ids, shape ``(batch_size, seq_len)``.
        :type patch_ids: keras.KerasTensor
        :param training: Whether in training mode.
        :type training: Optional[bool]
        :return: Logits, shape ``(batch_size, seq_len, vocab_size)``.
        :rtype: keras.KerasTensor
        """
        x = self.byte_embedding(byte_tokens)

        x = self.positional_embedding(x, training=training)

        if self.context_projection is not None:
            global_context = self.context_projection(global_context)

        attend_mask = create_causal_attend_mask(x)
        for i, (decoder_layer, cross_attention, cross_norm) in enumerate(
                zip(self.decoder_layers, self.cross_attention_layers, self.cross_attention_norms)
        ):
            x = decoder_layer(x, attention_mask=attend_mask, training=training)

            cross_attended = self._masked_cross_attention(
                x, global_context, patch_ids, cross_attention, training
            )

            # Post-norm around the cross-attention residual.
            x = x + cross_attended
            x = cross_norm(x)

        x = self.layer_norm(x)

        logits = self.output_projection(x)

        return logits

    def _masked_cross_attention(
            self,
            decoder_hidden: keras.KerasTensor,
            global_context: keras.KerasTensor,
            patch_ids: keras.KerasTensor,
            cross_attention: keras.layers.Layer,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Cross-attend to the preceding patch's global representation.

        Byte ``i`` reads the contextualized representation of patch
        ``patch_ids[i] - 1``, not of its own patch, and the attention is
        additionally masked causally over the gathered byte-length key
        sequence. Both restrictions are needed for the decoder to be causal:
        gathering a byte's own patch leaks the future, since that patch's
        representation is pooled over every byte in it including the target
        byte itself; and even with the previous-patch gather, a later key
        ``j`` may carry a patch index at or after ``patch_ids[i]``, so the
        causal mask over the key axis is not redundant.

        Bytes in patch 0 have no preceding patch. Their gather index is
        clamped to 0 and the gathered vector is then zeroed, so they receive
        no global context at all rather than reading their own patch. Zeroing
        the key rather than masking the query row also avoids a
        fully-masked softmax row.

        :param decoder_hidden: Decoder hidden states, shape ``(batch_size,
            seq_len, local_dim)``.
        :type decoder_hidden: keras.KerasTensor
        :param global_context: Global patch representations, shape
            ``(batch_size, num_patches, local_dim)``.
        :type global_context: keras.KerasTensor
        :param patch_ids: Patch ids, shape ``(batch_size, seq_len)``.
        :type patch_ids: keras.KerasTensor
        :param cross_attention: The ``MultiHeadAttention`` layer to apply.
        :type cross_attention: keras.layers.Layer
        :param training: Whether in training mode.
        :type training: Optional[bool]
        :return: Cross-attended output, shape ``(batch_size, seq_len,
            local_dim)``.
        :rtype: keras.KerasTensor
        """
        batch_size = ops.shape(decoder_hidden)[0]
        seq_len = ops.shape(decoder_hidden)[1]

        prev_patch_ids = ops.maximum(patch_ids - 1, 0)
        gather_idx = ops.expand_dims(prev_patch_ids, axis=-1)
        global_dim = ops.shape(global_context)[-1]
        gather_idx = ops.broadcast_to(gather_idx, (batch_size, seq_len, global_dim))
        position_context = ops.take_along_axis(global_context, gather_idx, axis=1)

        # Zeroing beats masking the row here: a fully masked softmax row is not.
        has_prev = ops.cast(
            ops.expand_dims(ops.greater(patch_ids, 0), axis=-1),
            position_context.dtype,
        )
        position_context = position_context * has_prev

        # keras MultiHeadAttention takes attend semantics at (B, T_q, T_k), and
        # the key axis here is the byte axis.
        cross_attend_mask = create_causal_attend_mask(decoder_hidden)

        attended = cross_attention(
            query=decoder_hidden,
            value=position_context,
            key=position_context,
            attention_mask=cross_attend_mask,
            training=training
        )

        return attended

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape.

        :param input_shape: Byte token shape ``(batch, seq_len)``, or a list
            whose first entry is that shape.
        :type input_shape: Tuple[Optional[int], ...]
        :return: ``(batch_size, seq_len, vocab_size)``.
        :rtype: Tuple[Optional[int], ...]
        """
        if isinstance(input_shape, list):
            # The byte-token shape carries the batch and sequence axes.
            batch_size = input_shape[0][0]
            seq_len = input_shape[0][1]
        else:
            batch_size = input_shape[0]
            seq_len = input_shape[1]
        return (batch_size, seq_len, self.vocab_size)

    def get_config(self) -> Dict[str, Any]:
        """Return layer configuration.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'local_dim': self.local_dim,
            'global_dim': self.global_dim,
            'num_local_layers': self.num_local_layers,
            'num_heads_local': self.num_heads_local,
            'max_sequence_length': self.max_sequence_length,
            'dropout_rate': self.dropout_rate
        })
        return config

# ---------------------------------------------------------------------