"""Shared skeleton for the embeddings study: encoder, block registry, heads."""

from .blocks import (
    BLOCK_REGISTRY,
    CliffordEncoderBlock,
    available_block_types,
    clifford_receptive_field,
    create_encoder_block,
)
from .encoder import EmbeddingEncoder

__all__ = [
    "BLOCK_REGISTRY",
    "CliffordEncoderBlock",
    "EmbeddingEncoder",
    "available_block_types",
    "clifford_receptive_field",
    "create_encoder_block",
]
