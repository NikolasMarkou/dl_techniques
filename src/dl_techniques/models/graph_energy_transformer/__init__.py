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
    * :class:`GraphClassifier` — variant C-lite (graph classification) head: trunk (CLS token +
      Laplacian PE + saddle-escape noise) -> CLS-token readout -> class logits.
    * :func:`create_graph_classifier` — factory for the variant-C-lite classifier.
"""

from .model import (
    GRAPH_BACKBONE_NAME,
    GraphEnergyTransformerBackbone,
    GraphAnomalyDetector,
    GraphClassifier,
    create_graph_energy_transformer_backbone,
    create_graph_anomaly_detector,
    create_graph_classifier,
)

__all__ = [
    "GRAPH_BACKBONE_NAME",
    "GraphEnergyTransformerBackbone",
    "GraphAnomalyDetector",
    "GraphClassifier",
    "create_graph_energy_transformer_backbone",
    "create_graph_anomaly_detector",
    "create_graph_classifier",
]
