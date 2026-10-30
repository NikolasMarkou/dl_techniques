"""
``PatchCausalTransformer``, a causal transformer stack that runs over
sequences the caller has already embedded, shaped
``(batch, num_patches, dim)``.

Hierarchical models first compress a long sequence (bytes, tokens,
video frames) into a much shorter sequence of patch vectors, then model
how those patches depend on each other. This layer is that second
stage: it adds a learned positional embedding, runs ``depth`` causal
``TransformerLayer`` blocks with an FFN of width ``4 * dim``, and applies
a final ``LayerNormalization``. Attention cost is quadratic in the patch
count, not the length of the sequence underneath, so the expensive
layers run once per patch instead of once per byte or token. The other
stacks in the package start one step earlier: ``TextDecoder`` embeds
token IDs and takes a padding mask, ``VisionEncoder`` embeds an image.

Use it as the global model in a byte- or token-patch architecture such
as the BLT latent transformer, or for autoregressive modelling over any
pooled representation, such as frame features or chunk embeddings.

Before use: input is ``(batch, P, dim)`` and output has the same shape.
Static ``P`` above ``max_patches`` raises ``ValueError`` in ``build``,
with a message that names ``max_seq_len``. The mask is causal only:
empty patch slots get no padding mask and take part in attention, as in
BLT, where ``max_patches`` is a fixed slot count. Mask or zero them
upstream if they must be ignored. The final norm uses ``epsilon=1e-3``
(see :data:`~.causal_stack.LAYER_NORM_EPSILON`). Two instances in one
model need distinct ``name_prefix`` values. The defaults
(``dim=768, depth=12, num_heads=12``) give a base-size stack.

Example::

    layer = PatchCausalTransformer(dim=64, depth=2, num_heads=4, max_patches=16)
    y = layer(np.zeros((2, 16, 64), dtype="float32"))
    y.shape  # (2, 16, 64)

References:
    - Pagnoni et al., 2024. Byte Latent Transformer: Patches Scale Better
      Than Tokens. The reference consumer of this stack.
      (https://arxiv.org/abs/2412.09871)
    - Yu et al., 2023. MEGABYTE: Predicting Million-byte Sequences with
      Multiscale Transformers. A global causal model over byte patches.
      (https://arxiv.org/abs/2305.07185)
    - Vaswani et al., 2017. Attention Is All You Need.
      (https://arxiv.org/abs/1706.03762)
"""

import keras
from typing import Optional, Dict, Any, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..embedding.positional_embedding import PositionalEmbedding
from dl_techniques.utils.keras_registration import register_dl_technique

from .causal_stack import (
    CausalTransformerStackMixin,
    LAYER_NORM_EPSILON,
    build_causal_stack_norm,
    build_causal_transformer_stack,
)

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.transformers.patch_causal_transformer")
class PatchCausalTransformer(CausalTransformerStackMixin, keras.layers.Layer):
    """Apply causal self-attention across pre-embedded sequence representations.

    Models long-range dependencies along an axis already pooled down to
    patch-sized vectors, so the quadratic cost is paid over a sequence much
    shorter than the one beneath it.

    Architecture:

    .. code-block:: text

        patch representations [B, P, D]
              │
              ▼
        ┌──────────────────────────────┐
        │ patch positional embedding   │
        └──────────────────────────────┘
              │
              ▼
        ┌──────────────────────────────┐
        │ transformer layer x N        │◄── causal attend mask
        │ ffn width 4 * D              │
        └──────────────────────────────┘
              │
              ▼
        ┌──────────────────────────────┐
        │ layer norm                   │
        └──────────────────────────────┘
              │
              ▼
        contextualized patches [B, P, D]

    The mask is causal over the patch axis only. Empty patch slots carry no
    padding mask, so they take part in attention like any other patch -- a
    deliberate choice inherited from BLT, where ``max_patches`` is a fixed slot
    count rather than a per-sample length. A caller needing real padding
    semantics must mask the empty slots itself before calling.

    :param dim: Hidden dimension of the stack, and of the positional table.
    :type dim: int
    :param depth: Number of transformer layers.
    :type depth: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param max_patches: Maximum number of patches per sequence, and the
        positional embedding's length. A statically longer input raises
        ``ValueError`` in :meth:`build` rather than being truncated; the
        message names ``max_seq_len``, since the check belongs to
        ``PositionalEmbedding`` and the two are the same number here.
    :type max_patches: int
    :param dropout_rate: Dropout rate for all layers.
    :type dropout_rate: float
    :param layer_norm_epsilon: Epsilon for the final norm. Defaults to
        :data:`~dl_techniques.layers.transformers.causal_stack.LAYER_NORM_EPSILON`
        (``1e-3``), stated explicitly rather than inherited from the Keras
        class default. Read that constant's note before changing it.
    :type layer_norm_epsilon: float
    :param name_prefix: Substring of every sub-layer name, so two instances in
        one model do not collide.
    :type name_prefix: str
    :param kwargs: Additional ``keras.layers.Layer`` arguments.

    :raises ValueError: If a static sequence length exceeds ``max_patches``,
        raised from :meth:`build` by ``PositionalEmbedding``.

    Example:

    .. code-block:: python

        import numpy as np
        from dl_techniques.layers.transformers import PatchCausalTransformer

        layer = PatchCausalTransformer(dim=64, depth=2, num_heads=4, max_patches=16)
        x = np.zeros((2, 16, 64), dtype="float32")
        layer(x).shape  # (2, 16, 64)
    """

    def __init__(
            self,
            dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            max_patches: int = 512,
            dropout_rate: float = 0.1,
            layer_norm_epsilon: float = LAYER_NORM_EPSILON,
            name_prefix: str = 'patch_causal_transformer',
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.max_patches = max_patches
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.name_prefix = name_prefix

        self.bind_causal_stack(
            positional_embedding=PositionalEmbedding(
                max_seq_len=self.max_patches,
                dim=self.dim,
                dropout_rate=self.dropout_rate,
                name=f'{self.name_prefix}_positional_embedding',
            ),
            layers=build_causal_transformer_stack(
                hidden_size=self.dim,
                num_heads=self.num_heads,
                depth=self.depth,
                dropout_rate=self.dropout_rate,
                name_prefix=self.name_prefix,
            ),
            final_norm=build_causal_stack_norm(
                f'{self.name_prefix}_norm',
                epsilon=self.layer_norm_epsilon,
            ),
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the stack.

        :param input_shape: Patch representation shape ``(batch, P, D)``.
        :type input_shape: Tuple[Optional[int], ...]
        """
        self.build_causal_stack(input_shape)
        super().build(input_shape)

    def call(
            self,
            patch_representations: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Run the stack forward.

        :param patch_representations: Patch representations, shape
            ``(batch_size, num_patches, dim)``.
        :type patch_representations: keras.KerasTensor
        :param training: Whether in training mode.
        :type training: Optional[bool]
        :return: Contextualized patch representations, same shape as input.
        :rtype: keras.KerasTensor
        """
        # Patch p's representation must not depend on the patches after it.
        return self.run_causal_stack(patch_representations, training=training)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape.

        :param input_shape: Patch representation shape ``(batch, P, D)``.
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
            'dim': self.dim,
            'depth': self.depth,
            'num_heads': self.num_heads,
            'max_patches': self.max_patches,
            'dropout_rate': self.dropout_rate,
            'layer_norm_epsilon': self.layer_norm_epsilon,
            'name_prefix': self.name_prefix,
        })
        return config

# ---------------------------------------------------------------------