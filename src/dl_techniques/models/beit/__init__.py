"""BEiT vision model public API.

BEiT (*BERT Pre-Training of Image Transformers*, arXiv:2106.08254) is a pre-norm ViT
with three non-obvious deltas from a generic ViT: an asymmetric q/v-only QKV bias, a
cls-augmented T5-style relative position bias, and LayerScale plus a linear
stochastic-depth ramp at ``layer_norm_eps=1e-12``.

This package exposes the shared trunk; the masked-image-modeling and classification
heads that consume it are added alongside it.
"""

from .model import (
    BACKBONE_NAME,
    SCALE_CONFIGS,
    MODEL_VARIANTS,
    BeitModel,
    create_beit_backbone,
)

__all__ = [
    "BACKBONE_NAME",
    "SCALE_CONFIGS",
    "MODEL_VARIANTS",
    "BeitModel",
    "create_beit_backbone",
]
