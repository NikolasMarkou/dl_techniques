"""GlobalTransformer, the Byte Latent Transformer's patch-level causal transformer,
using the shared `create_causal_attend_mask` helper from `dl_techniques.utils.masking`.

GlobalTransformer attends across patches, modeling long-range dependencies over a
sequence much shorter than the underlying byte sequence.

References:
    - Pagnoni et al., 2024. Byte Latent Transformer: Patches Scale Better
      Than Tokens. (https://arxiv.org/abs/2412.09871)
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

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.blt.global_transformer")
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
