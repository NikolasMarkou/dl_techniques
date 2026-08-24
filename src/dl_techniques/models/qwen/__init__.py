"""Qwen3 — public API re-exports.

Three distinct models: `Qwen3` (dense), `Qwen3Next` (a different block layout
with its own variant table), and the embedding/reranker pair, which are
retrieval heads rather than generative models and so have no `create_*`.
"""
from .qwen3 import (
    Qwen3,
    create_qwen3,
    create_qwen3_classification,
    create_qwen3_generation,
)
from .qwen3_embeddings import Qwen3EmbeddingModel, Qwen3RerankerModel
from .qwen3_next import (
    Qwen3Next,
    create_qwen3_next,
    create_qwen3_next_classification,
    create_qwen3_next_generation,
)

__all__ = [
    "Qwen3",
    "Qwen3EmbeddingModel",
    "Qwen3Next",
    "Qwen3RerankerModel",
    "create_qwen3",
    "create_qwen3_classification",
    "create_qwen3_generation",
    "create_qwen3_next",
    "create_qwen3_next_classification",
    "create_qwen3_next_generation",
]
