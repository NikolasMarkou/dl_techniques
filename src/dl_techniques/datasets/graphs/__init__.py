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
"""

from .tudataset import (
    DEFAULT_GRAPH_CACHE,
    GraphSample,
    download_tudataset,
    load_tudataset,
)

__all__ = [
    "DEFAULT_GRAPH_CACHE",
    "GraphSample",
    "download_tudataset",
    "load_tudataset",
]
