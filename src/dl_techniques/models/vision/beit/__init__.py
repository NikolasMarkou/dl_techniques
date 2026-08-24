"""BEiT vision model public API.

BEiT (*BERT Pre-Training of Image Transformers*, arXiv:2106.08254) is a pre-norm ViT
with three non-obvious deltas from a generic ViT: an asymmetric q/v-only QKV bias, a
cls-augmented T5-style relative position bias, and LayerScale plus a linear
stochastic-depth ramp at ``layer_norm_eps=1e-12``.

One shared trunk (:class:`BeitModel`) is consumed by two heads with DISJOINT layer-name
prefixes — ``decoder_`` for masked image modeling, ``head_`` for classification — so a
classifier warm-starts from an MIM checkpoint via
``load_weights_from_checkpoint(target, ckpt, skip_prefixes=("decoder_", "head_"))``.
"""

from .model import (
    BACKBONE_NAME,
    DEFAULT_VOCAB_SIZE,
    SCALE_CONFIGS,
    MODEL_VARIANTS,
    BeitModel,
    BeitForMaskedImageModeling,
    BeitForImageClassification,
    create_beit_backbone,
    create_beit_mim,
    create_beit_classifier,
)

__all__ = [
    "BACKBONE_NAME",
    "DEFAULT_VOCAB_SIZE",
    "SCALE_CONFIGS",
    "MODEL_VARIANTS",
    "BeitModel",
    "BeitForMaskedImageModeling",
    "BeitForImageClassification",
    "create_beit_backbone",
    "create_beit_mim",
    "create_beit_classifier",
]
