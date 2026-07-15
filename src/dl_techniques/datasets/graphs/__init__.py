"""Graph benchmark dataset loaders for the Graph Energy Transformer.

Stdlib + numpy/scipy loaders for graph-domain datasets, with no dependency on
the heavy graph-learning stacks (``torch_geometric`` / ``dgl`` / ``spektral``).

Public surface:
    * :func:`load_tudataset` — parse a TUDataset benchmark (MUTAG/PROTEINS/...)
      into dense per-graph records + metadata.
    * :func:`download_tudataset` — idempotent download + unzip into an on-disk
      cache outside the repository.
    * :class:`GraphSample` — a single parsed graph record.
    * :data:`DEFAULT_GRAPH_CACHE` — default on-disk cache root.
    * :func:`compute_laplacian_pe` — Laplacian-eigenvector positional encoding.
    * :func:`sign_flip_pe` — per-epoch PE sign-flip augmentation.
    * :func:`collate_graph_batch` — dense batch/pad + node-validity mask.
    * :func:`build_tudataset_graph_dataset` — ``tf.data`` builder emitting the
      ``({node_features, adjacency, pe, node_mask}, label)`` contract.

Variant-B (node-anomaly) fraud loaders:
    * :class:`FraudGraph` — one large graph with per-node binary labels
      (whole-graph adjacency kept sparse).
    * :func:`load_fraud_graph` — download + parse a CARE-GNN ``.mat`` benchmark.
    * :func:`sample_subgraph` — bounded k-hop neighbour-sampling subgraph.
    * :func:`make_synthetic_fraud_graph` — network-free synthetic fallback.
    * :func:`build_fraud_subgraph_dataset` — ``tf.data`` builder emitting the
      ``({node_features, adjacency, node_mask, target_index}, label)`` contract.
"""

from .tudataset import (
    DEFAULT_GRAPH_CACHE,
    GraphSample,
    build_tudataset_graph_dataset,
    collate_graph_batch,
    compute_laplacian_pe,
    download_tudataset,
    load_tudataset,
    sign_flip_pe,
)
from .fraud import (
    FraudGraph,
    build_fraud_subgraph_dataset,
    download_fraud_dataset,
    load_fraud_graph,
    make_synthetic_fraud_graph,
    sample_subgraph,
)

__all__ = [
    "DEFAULT_GRAPH_CACHE",
    "GraphSample",
    "build_tudataset_graph_dataset",
    "collate_graph_batch",
    "compute_laplacian_pe",
    "download_tudataset",
    "load_tudataset",
    "sign_flip_pe",
    "FraudGraph",
    "build_fraud_subgraph_dataset",
    "download_fraud_dataset",
    "load_fraud_graph",
    "make_synthetic_fraud_graph",
    "sample_subgraph",
]
