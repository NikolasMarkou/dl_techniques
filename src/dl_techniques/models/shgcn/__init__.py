"""sHGCN (Simplified Hyperbolic GCN) — public API re-exports.

Three models share one backbone: ``SHGCNModel`` (feature extractor),
``SHGCNNodeClassifier`` and ``SHGCNLinkPredictor``, each with its own factory.

There is no ``MODEL_VARIANTS`` table and none was invented: sHGCN is sized by a
per-layer hidden-dimension list chosen for the dataset at hand, and the paper
publishes no named scale family, so the factories construct the classes
directly rather than delegating to a ``from_variant``.
"""
from dl_techniques.models.shgcn.model import (
    SHGCNModel,
    SHGCNNodeClassifier,
    SHGCNLinkPredictor,
    create_shgcn,
    create_shgcn_node_classifier,
    create_shgcn_link_predictor,
)

__all__ = [
    "SHGCNModel",
    "SHGCNNodeClassifier",
    "SHGCNLinkPredictor",
    "create_shgcn",
    "create_shgcn_node_classifier",
    "create_shgcn_link_predictor",
]
