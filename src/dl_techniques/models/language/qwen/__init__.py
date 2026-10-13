"""Qwen3 — public API re-exports.

Three distinct models: `Qwen3` (dense), `Qwen3Next` (a different block layout
with its own variant table), and the embedding/reranker pair, which are
retrieval heads rather than generative models and so have no `create_*`.
`create_qwen3` and `create_qwen3_next` return the class itself; a task head
comes from `include_top=True` with a `task_config`.
"""
from .qwen3 import Qwen3, create_qwen3
from .qwen3_embeddings import Qwen3EmbeddingModel, Qwen3RerankerModel
from .qwen3_next import Qwen3Next, create_qwen3_next

__all__ = [
    "Qwen3",
    "Qwen3EmbeddingModel",
    "Qwen3Next",
    "Qwen3RerankerModel",
    "create_qwen3",
    "create_qwen3_next",
]
