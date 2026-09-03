"""Transformer blocks and encoder/decoder stacks for building models.

Two foundation layers cover most needs: :class:`TransformerLayer` for a
self-attention block and :class:`TransformerDecoderLayer` for the
encoder-decoder counterpart, both factory-driven so attention type, FFN
type, normalization type and normalization position are constructor
arguments rather than separate classes. Three complete stacks build on
them: :class:`VisionEncoder` (ViT-style), :class:`TextEncoder`
(BERT-style bidirectional) and :class:`TextDecoder` (GPT-style
autoregressive).

The remaining classes are specialized blocks that do not fit the
factory-driven shape:

- :class:`SwinTransformerBlock` and :class:`SwinConvBlock` — windowed and
  shifted-window attention, the second combined with a convolutional path.
- :class:`PerceiverTransformerLayer` — cross-attention for very large
  input sequences.
- :class:`EomtTransformer` — masked self-attention with object queries for
  instance segmentation; `use_masked_attention=True` requires a maskable
  `attention_type` (not `'fnet'`/`'anchor'`/`'lighthouse'`).
- :class:`AdaLNZeroConditionalBlock` — DiT-style adaptive layer-norm
  zero-initialized conditional block.
- :class:`AreaAttentionBlock` — the yolo12 attention stage over a 4D
  `[B, H, W, C]` feature map: `AreaAttention` plus a 1x1-conv MLP, each in
  a plain residual add, with no Pre/Post-Norm and no LayerScale.
- :class:`GatedLinearAttentionBlock` — a recurrent sequence mixer holding
  one `(head_dim, head_dim)` state per head, rewritten each timestep by a
  gated outer product; cost grows linearly with sequence length.
- :class:`EnergyTransformer` (arXiv:2302.07253) and its associative-memory
  module :class:`HopfieldNetwork` — replaces the `attn -> FFN` residual
  stream with `T` steps of gradient descent on one scalar energy.

A set of `create_*_encoder` functions build the complete stacks with
common presets, and type aliases (`AttentionType`, `FFNType`, etc.)
narrow their configuration arguments.
"""

# ---------------------------------------------------------------------
# Foundational Layer
# ---------------------------------------------------------------------

from .transformer import (
    TransformerLayer,
    AttentionType,
    FFNType,
    NormalizationType,
    NormalizationPositionType,
)
from .transformer_decoder import TransformerDecoderLayer

# ---------------------------------------------------------------------
# Vision Models
# ---------------------------------------------------------------------

from .vision_encoder import (
    VisionEncoder,
    PatchEmbedType,
    create_vision_encoder,
    create_vit_encoder,
    create_siglip_encoder,
)

# ---------------------------------------------------------------------
# Text Models
# ---------------------------------------------------------------------

from .text_encoder import (
    TextEncoder,
    EmbeddingType,
    PositionalType,
    create_text_encoder,
    create_bert_encoder,
    create_roberta_encoder,
    create_modern_encoder,
    create_efficient_encoder,
)
from .text_decoder import TextDecoder

# ---------------------------------------------------------------------
# Specialized and Hybrid Blocks
# ---------------------------------------------------------------------

from .swin_transformer_block import SwinTransformerBlock
from .swin_conv_block import SwinConvBlock
from .perceiver_transformer import PerceiverTransformerLayer
from .eomt_transformer import EomtTransformer
from .adaln_zero import AdaLNZeroConditionalBlock
from .free_transformer import BinaryMapper, FreeTransformerLayer
from .progressive_focused_transformer import PFTBlock
from .gated_linear_attention_block import GatedLinearAttentionBlock
from .area_attention_block import AreaAttentionBlock

# ---------------------------------------------------------------------
# Energy Transformer (arXiv:2302.07253)
# ---------------------------------------------------------------------

from .energy_transformer import EnergyTransformer, HopfieldNetwork

# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

__all__ = [
    # Foundational Layer
    "TransformerLayer",
    "TransformerDecoderLayer",

    # Vision Models
    "VisionEncoder",
    "create_vision_encoder",
    "create_vit_encoder",
    "create_siglip_encoder",

    # Text Models
    "TextEncoder",
    "TextDecoder",
    "create_text_encoder",
    "create_bert_encoder",
    "create_roberta_encoder",
    "create_modern_encoder",
    "create_efficient_encoder",

    # Specialized Blocks
    "SwinTransformerBlock",
    "SwinConvBlock",
    "PerceiverTransformerLayer",
    "EomtTransformer",
    "AdaLNZeroConditionalBlock",
    "BinaryMapper",
    "FreeTransformerLayer",
    "PFTBlock",
    "GatedLinearAttentionBlock",
    "AreaAttentionBlock",

    # Energy Transformer
    "EnergyTransformer",
    "HopfieldNetwork",

    # Type Aliases
    "AttentionType",
    "EmbeddingType",
    "FFNType",
    "NormalizationType",
    "NormalizationPositionType",
    "PatchEmbedType",
    "PositionalType",
]