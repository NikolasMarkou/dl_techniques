"""Energy Transformer image models public API.

Two consumers of the ``EnergyTransformer`` block (arXiv:2302.07253) sharing one trunk:
a masked-image-completion model (the paper's §3 model) and a classifier that can warm-start
its trunk from an MIM checkpoint.
"""

from .model import (
    BACKBONE_NAME,
    SCALE_CONFIGS,
    MODEL_VARIANTS,
    EnergyTransformerBackbone,
    EnergyTransformerMIM,
    EnergyTransformerClassifier,
    create_energy_transformer_backbone,
    create_energy_transformer_mim,
    create_energy_transformer_classifier,
)

__all__ = [
    "BACKBONE_NAME",
    "SCALE_CONFIGS",
    "MODEL_VARIANTS",
    "EnergyTransformerBackbone",
    "EnergyTransformerMIM",
    "EnergyTransformerClassifier",
    "create_energy_transformer_backbone",
    "create_energy_transformer_mim",
    "create_energy_transformer_classifier",
]
