"""Graph Energy Transformer — a graph-domain trunk for the ``EnergyTransformer`` block.

Public surface:
    * :class:`GraphEnergyTransformerBackbone` — the shared graph trunk (node projection,
      optional Laplacian-PE add, always-created node mask token, optional CLS token, and a
      configurable stack of ``EnergyTransformer`` blocks). Exposes ``embed`` / ``call`` /
      ``descend_capture`` for the two heads.
    * :func:`create_graph_energy_transformer_backbone` — factory for the backbone.
    * :data:`GRAPH_BACKBONE_NAME` — the stable trunk name (warm-start matches by name).

Variant heads (``GraphAnomalyDetector`` / ``GraphClassifier``) are added to ``model.py`` in
later plan steps.
"""

from .model import (
    GRAPH_BACKBONE_NAME,
    GraphEnergyTransformerBackbone,
    create_graph_energy_transformer_backbone,
)

__all__ = [
    "GRAPH_BACKBONE_NAME",
    "GraphEnergyTransformerBackbone",
    "create_graph_energy_transformer_backbone",
]
