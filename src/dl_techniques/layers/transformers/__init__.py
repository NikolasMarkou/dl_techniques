"""
Core Transformer Blocks for Building Advanced Models.

This module provides a collection of high-level, configurable building blocks
for creating state-of-the-art Transformer architectures. It is designed around
a core philosophy of modularity and configurability, allowing for the seamless
construction of a wide variety of models.

The primary components are:

Foundational Layer:
- TransformerLayer: The generic, factory-based building block for all
  Transformer models.

Complete Modality Stacks:
- VisionEncoder: A complete ViT-style encoder for image processing.
- TextEncoder: A complete BERT-style bidirectional encoder for natural
  language understanding.
- TextDecoder: A complete GPT-style autoregressive decoder for natural
  language generation.

Specialized and Hybrid Blocks:
- SwinTransformerBlock: The core block of the Swin Transformer, using
  windowed and shifted-window attention.
- SwinConvBlock: A hybrid block combining a Swin Transformer path with
  a convolutional path.
- PerceiverTransformerLayer: A block implementing Perceiver-style
  cross-attention for handling very large input sequences.
- EomtTransformer: A specialized layer for instance segmentation that
  uses masked self-attention with object queries.

Factory Functions and Type Aliases:
- A set of `create_*_encoder` functions for convenient model instantiation.
- Type aliases (`AttentionType`, `FFNType`, etc.) for clear configuration.
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

# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

__all__ = [
    # Foundational Layer
    "TransformerLayer",

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

    # Type Aliases
    "AttentionType",
    "EmbeddingType",
    "FFNType",
    "NormalizationType",
    "NormalizationPositionType",
    "PatchEmbedType",
    "PositionalType",
]