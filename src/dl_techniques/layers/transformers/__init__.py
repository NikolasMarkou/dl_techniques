"""
Core Transformer Blocks for Building Advanced Models.

This module provides a collection of high-level, configurable building blocks
for creating state-of-the-art Transformer architectures. It is designed around
a core philosophy of modularity and configurability, allowing for the seamless
construction of a wide variety of models.

The primary components are:

Foundational Layer:
- TransformerLayer: The generic, factory-based building block for all
  Transformer models (self-attention only).
- TransformerDecoderLayer: The encoder-decoder counterpart — masked/causal
  self-attention + cross-attention to encoder memory + FFN, factory-driven.

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
- AdaLNZeroConditionalBlock: DiT-style adaptive layer-norm zero-initialized
  conditional transformer block; norms/attention/FFN/AdaLN activation are
  factory-configurable.
- EnergyTransformer: the Energy Transformer block (arXiv:2302.07253) — replaces
  the `attn -> FFN` residual stream with T steps of gradient descent on a single
  scalar energy `E_ATT + E_HN`; paired with HopfieldNetwork, its associative-memory
  module (the paper's analog of the FFN, and NOT an MLP).
- HopfieldNetwork: the Energy Transformer associative memory (one tied `(K, D)`
  matrix, bias-free, strictly per token).

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