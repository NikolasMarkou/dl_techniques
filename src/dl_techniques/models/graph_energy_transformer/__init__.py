"""Graph Energy Transformer — a graph-domain trunk for the ``EnergyTransformer`` block.

Public surface:
    * :class:`GraphEnergyTransformerBackbone` — the shared graph trunk (node projection,
      optional Laplacian-PE add, always-created node mask token, optional CLS token, and a
      configurable stack of ``EnergyTransformer`` blocks). Exposes ``embed`` / ``call`` /
      ``descend_capture`` for the two heads.
    * :func:`create_graph_energy_transformer_backbone` — factory for the backbone.
    * :data:`GRAPH_BACKBONE_NAME` — the stable trunk name (warm-start matches by name).
    * :class:`GraphAnomalyDetector` — variant B (node anomaly) head: trunk -> target-node
      ``g_1 || g_T`` readout -> MLP logit.
    * :func:`create_graph_anomaly_detector` — factory for the variant-B detector.

The variant-C head (``GraphClassifier``) is added to ``model.py`` in a later plan step.
"""

from .model import (
    GRAPH_BACKBONE_NAME,
    GraphEnergyTransformerBackbone,
    GraphAnomalyDetector,
    create_graph_energy_transformer_backbone,
    create_graph_anomaly_detector,
)

__all__ = [
    "GRAPH_BACKBONE_NAME",
    "GraphEnergyTransformerBackbone",
    "GraphAnomalyDetector",
    "create_graph_energy_transformer_backbone",
    "create_graph_anomaly_detector",
]
