"""CLIP — public API re-exports.

`model.py` is the standard CLIP; `clifford_clip.py` is a Clifford-algebra
variant sharing the contrastive objective but not the tower internals.
"""
from .model import CliffordCLIP

__all__ = [
    "CliffordCLIP",
]
