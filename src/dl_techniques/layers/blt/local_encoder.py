"""LocalEncoder: process bytes with causal attention, then pool them into
patch representations, using the shared `create_causal_attend_mask` helper
from `dl_techniques.utils.masking` and the sibling `PatchPooling` layer.
"""

import keras
from typing import Optional, Dict, Any, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.masking import create_causal_attend_mask
from dl_techniques.utils.keras_registration import register_dl_technique

from ..transformers.transformer import TransformerLayer
from ..embedding.positional_embedding import PositionalEmbedding
from .patch_pooling import PatchPooling

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.blt.local_encoder")
class LocalEncoder(keras.layers.Layer):
    """Process bytes with causal attention, then pool them into patches.

    Architecture:

    .. code-block:: text

        byte tokens [B, S]            patch_ids [B, S]
              │                             │
              ▼                             │
        ┌──────────────────────────────┐    │
        │ byte embedding               │    │
        └──────────────────────────────┘    │
              │ [B, S, D_l]                 │
              ▼                             │
        ┌──────────────────────────────┐    │
        │ positional embedding         │    │
        └──────────────────────────────┘    │
              │                             │
              ▼                             │
        ┌──────────────────────────────┐    │
        │ transformer layer x N        │◄── causal attend mask
        │ ffn width 4 * D_l            │    │
        └──────────────────────────────┘    │
              │                             │
              ▼                             │
        ┌──────────────────────────────┐    │
        │ layer norm                   │    │
        └──────────────────────────────┘    │
              │                             │
              ▼                             ▼
        ┌──────────────────────────────────────┐
        │ patch pooling                        │
        └──────────────────────────────────────┘
              │
              ▼
        patch representations [B, max_patches, D_g]

    The attention is causal over bytes but carries no padding mask, so padded
    positions are attended to as ordinary bytes.

    :param vocab_size: Size of the byte vocabulary (typically 256 plus
        special tokens).
    :type vocab_size: int
    :param local_dim: Hidden dimension of the local encoder.
    :type local_dim: int
    :param num_local_layers: Number of transformer layers in the local encoder.
    :type num_local_layers: int
    :param num_heads_local: Number of attention heads in the local transformer.
    :type num_heads_local: int
    :param max_sequence_length: Maximum sequence length in bytes.
    :type max_sequence_length: int
    :param max_patches: Maximum number of patches per sequence, forwarded to
        the pooling layer as its patch-slot count.
    :type max_patches: int
    :param dropout_rate: Dropout rate for all layers.
    :type dropout_rate: float
    :param patch_pooling_method: One of ``'max'``, ``'mean'``, ``'attention'``.
    :type patch_pooling_method: str
    :param global_dim: Output dimension, matching the global transformer's
        hidden dimension.
    :type global_dim: int
    :param cross_attention_queries: Number of queries for attention pooling.
    :type cross_attention_queries: int
    :param kwargs: Additional ``keras.layers.Layer`` arguments.
    """

    def __init__(
            self,
            vocab_size: int = 260,
            local_dim: int = 512,
            num_local_layers: int = 6,
            num_heads_local: int = 8,
            max_sequence_length: int = 2048,
            max_patches: int = 512,
            dropout_rate: float = 0.1,
            patch_pooling_method: str = 'attention',
            global_dim: int = 768,
            cross_attention_queries: int = 4,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.local_dim = local_dim
        self.num_local_layers = num_local_layers
        self.num_heads_local = num_heads_local
        self.max_sequence_length = max_sequence_length
        self.max_patches = max_patches
        self.dropout_rate = dropout_rate
        self.patch_pooling_method = patch_pooling_method
        self.global_dim = global_dim
        self.cross_attention_queries = cross_attention_queries

        self.byte_embedding = keras.layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.local_dim,
            name='byte_embedding'
        )

        self.positional_embedding = PositionalEmbedding(
            max_seq_len=self.max_sequence_length,
            dim=self.local_dim,
            dropout_rate=self.dropout_rate,
            name='positional_embedding'
        )

        self.transformer_layers = []
        for i in range(self.num_local_layers):
            layer = TransformerLayer(
                hidden_size=self.local_dim,
                num_heads=self.num_heads_local,
                intermediate_size=self.local_dim * 4,
                dropout_rate=self.dropout_rate,
                name=f'local_transformer_{i}'
            )
            self.transformer_layers.append(layer)

        self.patch_pooling = PatchPooling(
            pooling_method=self.patch_pooling_method,
            output_dim=self.global_dim,
            num_queries=self.cross_attention_queries,
            max_patches=self.max_patches,
            name='patch_pooling'
        )

        self.layer_norm = keras.layers.LayerNormalization(name='local_encoder_norm')

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the local encoder layers.

        :param input_shape: Byte token shape ``(batch, seq_len)``.
        :type input_shape: Tuple[Optional[int], ...]
        """
        # Explicit builds, because a lazy first-call build leaves the weights
        # unloadable on a .keras reload.
        self.byte_embedding.build(input_shape)

        embedded_shape = self.byte_embedding.compute_output_shape(input_shape)

        self.positional_embedding.build(embedded_shape)
        pos_embedded_shape = self.positional_embedding.compute_output_shape(embedded_shape)

        current_shape = pos_embedded_shape
        for layer in self.transformer_layers:
            layer.build(current_shape)
            current_shape = layer.compute_output_shape(current_shape)

        self.layer_norm.build(current_shape)
        norm_shape = current_shape

        self.patch_pooling.build(norm_shape)

        super().build(input_shape)

    def call(
            self,
            byte_tokens: keras.KerasTensor,
            patch_ids: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Run the local encoder forward.

        :param byte_tokens: Byte tokens, shape ``(batch_size, seq_len)``.
        :type byte_tokens: keras.KerasTensor
        :param patch_ids: Patch ids, shape ``(batch_size, seq_len)``.
        :type patch_ids: keras.KerasTensor
        :param training: Whether in training mode.
        :type training: Optional[bool]
        :return: Patch representations, shape ``(batch_size, max_patches,
            global_dim)``.
        :rtype: keras.KerasTensor
        """
        x = self.byte_embedding(byte_tokens)

        x = self.positional_embedding(x, training=training)

        # The pooled patch vectors feed a next-byte objective, so byte i must
        # not attend past itself.
        attend_mask = create_causal_attend_mask(x)
        for layer in self.transformer_layers:
            x = layer(x, attention_mask=attend_mask, training=training)

        x = self.layer_norm(x)

        patch_representations = self.patch_pooling(x, patch_ids, training=training)

        return patch_representations

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape.

        :param input_shape: Byte token shape ``(batch, seq_len)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: ``(batch_size, max_patches, global_dim)``.
        :rtype: Tuple[Optional[int], ...]
        """
        batch_size = input_shape[0]
        return (batch_size, self.max_patches, self.global_dim)

    def get_config(self) -> Dict[str, Any]:
        """Return layer configuration.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'local_dim': self.local_dim,
            'num_local_layers': self.num_local_layers,
            'num_heads_local': self.num_heads_local,
            'max_sequence_length': self.max_sequence_length,
            'max_patches': self.max_patches,
            'dropout_rate': self.dropout_rate,
            'patch_pooling_method': self.patch_pooling_method,
            'global_dim': self.global_dim,
            'cross_attention_queries': self.cross_attention_queries
        })
        return config

# ---------------------------------------------------------------------
