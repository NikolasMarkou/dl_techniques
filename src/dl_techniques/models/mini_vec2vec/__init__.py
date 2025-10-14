"""
Mini-Vec2Vec: Unsupervised Embedding Space Alignment.

This module provides the MiniVec2VecAligner model for aligning two embedding
spaces without parallel data using linear transformations.
"""

from dl_techniques.models.mini_vec2vec.model import (
    MiniVec2VecAligner,
    create_mini_vec2vec_aligner,
)

__all__ = [
    "MiniVec2VecAligner",
    "create_mini_vec2vec_aligner",
]