"""CLIP — public API re-exports.

This package hosts the standard (non-Clifford) CLIP model. The Clifford-algebra
variant, CliffordCLIP, lives in its own sibling package,
`dl_techniques.models.vision_language.clifford_clip`.
"""
from .model import CLIP, create_clip_model, create_clip_variant

__all__ = [
    "CLIP",
    "create_clip_model",
    "create_clip_variant",
]
