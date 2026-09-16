"""CliffordCLIP — public API re-exports.

This package is CliffordCLIP's dedicated home: a Clifford-algebra variant of
CLIP sharing the contrastive objective with the standard model in the
sibling `dl_techniques.models.vision_language.clip` package, but not the
tower internals.
"""
from .model import CliffordCLIP

__all__ = [
    "CliffordCLIP",
]
