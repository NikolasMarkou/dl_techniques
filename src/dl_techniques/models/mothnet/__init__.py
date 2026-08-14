"""MothNet — public API re-exports.

Bio-inspired olfactory-learning network. `create_cyborg_features` is a feature
extractor, not a model factory, which is why there is no `create_mothnet`.
"""
from .model import MothNet, create_cyborg_features

__all__ = [
    "MothNet",
    "create_cyborg_features",
]
