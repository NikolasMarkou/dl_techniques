"""ColBERT late-interaction retrieval package.

Public surface is curated. This module is built up step by step; it currently
exports the two scoring components. See ``components.py`` for the late
interaction mechanics and their references.
"""

from .components import ColBERTProjection, MaxSimScorer

__all__ = ["ColBERTProjection", "MaxSimScorer"]
