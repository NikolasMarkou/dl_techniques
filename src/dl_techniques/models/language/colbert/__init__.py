"""ColBERT late-interaction retrieval package.

Public surface is curated. This module is built up step by step; it currently
exports the scoring components, the tokenizer and the model with its factories.
See ``model.py`` for the late interaction rationale, the v1/v2 shared-network
fact and its citation, and ``components.py`` for the scoring mechanics.
"""

from .components import ColBERTProjection, MaxSimScorer
from .compression import ResidualCompressionCodec
from .tokenization import ColBERTTokenizer
from .model import (
    ColBERT,
    create_colbert,
    create_colbert_v1,
    create_colbert_v2,
)

__all__ = [
    "ColBERT",
    "ColBERTProjection",
    "ColBERTTokenizer",
    "MaxSimScorer",
    "ResidualCompressionCodec",
    "create_colbert",
    "create_colbert_v1",
    "create_colbert_v2",
]
