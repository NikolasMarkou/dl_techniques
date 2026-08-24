"""CLIP — public API re-exports.

`model.py` is the standard CLIP; `clifford_clip.py` is a Clifford-algebra
variant sharing the contrastive objective but not the tower internals.
"""
from .clifford_clip import CliffordCLIP
from .model import CLIP, create_clip_model, create_clip_variant

__all__ = [
    "CLIP",
    "CliffordCLIP",
    "create_clip_model",
    "create_clip_variant",
]
