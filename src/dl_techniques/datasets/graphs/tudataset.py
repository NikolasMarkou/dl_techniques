"""TUDataset loader (parser + on-disk cache) for the Graph Energy Transformer.

This module reads the plain-text TUDataset benchmark format
(https://chrsmrrs.github.io/datasets/) for whole-graph classification
datasets such as ``MUTAG`` and ``PROTEINS`` using **only** the Python
standard library plus ``numpy``/``scipy``. It deliberately avoids the heavy
graph-learning stacks (``torch_geometric`` / ``dgl`` / ``spektral``) so the
dependency surface stays tiny and import-clean.

Scope of this module (Phase-0 step 1 of the graph-ET plan):

    * ``download_tudataset`` — idempotent download + unzip into an on-disk
      cache that lives OUTSIDE the repository (never on the repo SSD).
    * ``load_tudataset`` — parse the text files into a list of
      :class:`GraphSample` records plus a metadata dictionary.

Batching, padding, the rank-2 node-validity mask, Laplacian positional
encodings and the ``tf.data`` builder are intentionally **not** implemented
here; they are added by a later step to this same module. The parser below
returns per-graph dense ``numpy`` records with clean seams for that work.

Design choices (documented per the plan's requirements):

    * **Dense adjacency.** Each graph's adjacency is a dense
      ``np.ndarray`` of shape ``(n, n)`` and dtype ``float32``. Downstream
      the Energy Transformer consumes a dense ``(B, N, N)`` rank-3 attention
      mask, so a dense per-graph adjacency is the natural representation and
      avoids a sparse->dense conversion at batch time. TUDataset graphs are
      small (avg ~18 nodes for MUTAG, ~39 for PROTEINS), so density is cheap.
    * **Symmetrization.** The Energy Transformer rank-3 mask is applied
      literally ``(key, query)`` and is NOT auto-symmetrized, while the
      attention energy requires ``A == A.T``. TUDataset edge lists may or may
      not list both directions, so every adjacency is symmetrized here via
      ``A = ((A + A.T) > 0)`` cast to float. Symmetry is a caller obligation
      the loader satisfies up front (plan invariant 4).
    * **Self-loops.** Enabled by default (``add_self_loops=True``). One-hop
      neighbour attention should let a node attend to itself (the paper's
      neighbourhood includes the node), so the diagonal is set to 1 after
      symmetrization. Toggle off to get the raw adjacency.
    * **Node-feature fallback order.** ``node_labels`` one-hot (across the
      global label vocabulary) if present, else ``node_attributes`` as-is,
      else a single-column node-degree feature computed from the symmetrized
      adjacency (before self-loops are added).
"""

import os
import zipfile
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------
# module constants
# ---------------------------------------------------------------------

# On-disk cache root for graph benchmark datasets. This MUST live on the
# large data volume, never inside the repository or on the repo SSD
# (`/media/arxwn/data_fast`). Overridable per call.
DEFAULT_GRAPH_CACHE: str = "/media/arxwn/data0_4tb/datasets/graphs"

# TUDataset download host (verified reachable). The archive for a dataset
# `{name}` lives at `{TUDATASET_URL_TEMPLATE.format(name=name)}`.
TUDATASET_URL_TEMPLATE: str = (
    "https://www.chrsmrrs.com/graphkerneldatasets/{name}.zip"
)


# ---------------------------------------------------------------------
# data record
# ---------------------------------------------------------------------

@dataclass
class GraphSample:
    """A single parsed graph.

    Attributes:
        adjacency: Symmetric dense adjacency of shape ``(num_nodes,
            num_nodes)`` and dtype ``float32``. Guaranteed ``A == A.T``.
            Includes self-loops on the diagonal when the loader was called
            with ``add_self_loops=True``.
        node_features: Per-node feature matrix of shape ``(num_nodes, F)``
            and dtype ``float32``. ``F`` is constant across all samples in a
            dataset (the global node-label vocabulary size, the node
            attribute dimension, or ``1`` for the degree fallback).
        label: Integer graph-level class label remapped to the contiguous
            range ``0 .. num_classes - 1``.
        num_nodes: Number of nodes in this graph (``adjacency.shape[0]``).
    """

    adjacency: np.ndarray
    node_features: np.ndarray
    label: int
    num_nodes: int


# ---------------------------------------------------------------------
# download / cache
# ---------------------------------------------------------------------

def download_tudataset(
        name: str,
        cache_root: str = DEFAULT_GRAPH_CACHE,
) -> str:
    """Download and extract a TUDataset archive into the on-disk cache.

    The archive ``{name}.zip`` is fetched from
    ``https://www.chrsmrrs.com/graphkerneldatasets/`` and unzipped into
    ``cache_root``. The archive already contains a top-level ``{name}/``
    directory, so the extracted data lands at ``cache_root/{name}/``.

    The operation is idempotent: if the extracted data directory already
    contains ``{name}_A.txt`` the download and extraction are skipped.

    Args:
        name: Dataset name, e.g. ``"MUTAG"`` or ``"PROTEINS"`` (case matters;
            it must match the archive name on the host).
        cache_root: Directory under which ``{name}/`` is created. Defaults to
            :data:`DEFAULT_GRAPH_CACHE`. Never point this at the repository or
            the repo SSD.

    Returns:
        Absolute path to the extracted dataset directory (``cache_root/{name}``).

    Raises:
        urllib.error.URLError: If the download host is unreachable.
        FileNotFoundError: If extraction completed but the expected
            ``{name}_A.txt`` is missing (malformed archive).
    """
    cache_root = os.path.abspath(os.path.expanduser(cache_root))
    data_dir = os.path.join(cache_root, name)
    edge_file = os.path.join(data_dir, f"{name}_A.txt")

    if os.path.isfile(edge_file):
        logger.info(f"TUDataset '{name}' already cached at {data_dir}")
        return data_dir

    os.makedirs(cache_root, exist_ok=True)
    url = TUDATASET_URL_TEMPLATE.format(name=name)
    zip_path = os.path.join(cache_root, f"{name}.zip")

    logger.info(f"Downloading TUDataset '{name}' from {url}")
    request = urllib.request.Request(
        url, headers={"User-Agent": "dl-techniques/graph-loader"}
    )
    with urllib.request.urlopen(request) as response, open(zip_path, "wb") as fh:
        fh.write(response.read())

    logger.info(f"Extracting {zip_path} into {cache_root}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(cache_root)

    # The archive's top-level directory name is expected to equal `name`.
    if not os.path.isfile(edge_file):
        raise FileNotFoundError(
            f"Extraction of '{name}' did not produce {edge_file}. "
            f"Archive layout may differ from the expected TUDataset format."
        )

    # The zip is redundant once extracted; keep the cache tidy.
    try:
        os.remove(zip_path)
    except OSError:
        pass

    logger.info(f"TUDataset '{name}' ready at {data_dir}")
    return data_dir


# ---------------------------------------------------------------------
# text-format helpers
# ---------------------------------------------------------------------

def _read_int_column(path: str) -> np.ndarray:
    """Read a single-column integer text file into a 1-D int64 array."""
    return np.loadtxt(path, dtype=np.int64, ndmin=1)


def _read_edge_list(path: str) -> np.ndarray:
    """Read the comma-separated ``{name}_A.txt`` edge list.

    Returns an ``(E, 2)`` int64 array of 1-indexed global node id pairs.
    """
    edges = np.loadtxt(path, dtype=np.int64, delimiter=",", ndmin=2)
    if edges.shape[1] != 2:
        raise ValueError(
            f"Edge file {path} expected 2 columns, got {edges.shape[1]}"
        )
    return edges


def _read_node_attributes(path: str) -> np.ndarray:
    """Read ``{name}_node_attributes.txt`` into a ``(num_nodes, F)`` float32."""
    attrs = np.loadtxt(path, dtype=np.float32, delimiter=",", ndmin=2)
    return attrs


# ---------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------

def _parse_tudataset(
        name: str,
        data_dir: str,
        add_self_loops: bool,
) -> Tuple[List[GraphSample], Dict[str, Any]]:
    """Parse the extracted TUDataset text files into per-graph records.

    See :func:`load_tudataset` for the public contract. This function assumes
    the files are already present on disk.
    """
    def _p(suffix: str) -> str:
        return os.path.join(data_dir, f"{name}_{suffix}.txt")

    # --- required files -------------------------------------------------
    node_graph_raw = _read_int_column(_p("graph_indicator"))  # (Nn,) graph id
    graph_labels_raw = _read_int_column(_p("graph_labels"))   # (G,) label
    edges = _read_edge_list(_p("A"))                          # (E, 2) 1-indexed

    num_nodes_total = node_graph_raw.shape[0]

    # --- optional node features ----------------------------------------
    node_labels: Optional[np.ndarray] = None
    node_attributes: Optional[np.ndarray] = None
    node_labels_path = _p("node_labels")
    node_attrs_path = _p("node_attributes")
    if os.path.isfile(node_labels_path):
        node_labels = _read_int_column(node_labels_path)
    if os.path.isfile(node_attrs_path):
        node_attributes = _read_node_attributes(node_attrs_path)

    # --- map raw graph ids -> contiguous 0-based indices ----------------
    graph_id_values = np.unique(node_graph_raw)
    gid_to_idx = {int(g): i for i, g in enumerate(graph_id_values)}
    num_graphs = len(graph_id_values)

    # For each global node, its 0-based graph index and its within-graph
    # local index (assigned in ascending global-id order). Also collect the
    # global node ids per graph in that same local order for feature slicing.
    node_gidx = np.empty(num_nodes_total, dtype=np.int64)
    node_local = np.empty(num_nodes_total, dtype=np.int64)
    graph_node_globals: List[List[int]] = [[] for _ in range(num_graphs)]
    for gi in range(num_nodes_total):
        g = gid_to_idx[int(node_graph_raw[gi])]
        node_gidx[gi] = g
        node_local[gi] = len(graph_node_globals[g])
        graph_node_globals[g].append(gi)

    graph_num_nodes = np.array(
        [len(g) for g in graph_node_globals], dtype=np.int64
    )

    # --- allocate per-graph dense adjacencies and fill edges ------------
    adjacencies = [
        np.zeros((int(n), int(n)), dtype=np.float32) for n in graph_num_nodes
    ]
    for a, b in edges:
        ai = int(a) - 1
        bi = int(b) - 1
        g = int(node_gidx[ai])
        adjacencies[g][node_local[ai], node_local[bi]] = 1.0

    # --- build a global node-feature matrix (or degree fallback) --------
    global_features: Optional[np.ndarray] = None
    if node_labels is not None:
        label_values = np.unique(node_labels)
        lbl_to_idx = {int(v): i for i, v in enumerate(label_values)}
        feat_dim = len(label_values)
        global_features = np.zeros(
            (num_nodes_total, feat_dim), dtype=np.float32
        )
        for i in range(num_nodes_total):
            global_features[i, lbl_to_idx[int(node_labels[i])]] = 1.0
        feature_source = "node_labels_onehot"
    elif node_attributes is not None:
        global_features = node_attributes.astype(np.float32, copy=False)
        feat_dim = global_features.shape[1]
        feature_source = "node_attributes"
    else:
        feat_dim = 1  # single-column degree feature, filled per graph below
        feature_source = "node_degree"

    # --- remap graph labels to 0..C-1 ----------------------------------
    label_values = np.unique(graph_labels_raw)
    glbl_to_idx = {int(v): i for i, v in enumerate(label_values)}
    num_classes = len(label_values)

    # --- assemble per-graph samples ------------------------------------
    samples: List[GraphSample] = []
    label_distribution: Dict[int, int] = {c: 0 for c in range(num_classes)}
    for g in range(num_graphs):
        adj = adjacencies[g]
        # Symmetrize: A == A.T is a hard requirement of the ET rank-3 mask.
        adj = ((adj + adj.T) > 0.0).astype(np.float32)

        globals_arr = np.array(graph_node_globals[g], dtype=np.int64)
        if global_features is not None:
            node_feats = global_features[globals_arr].astype(
                np.float32, copy=True
            )
        else:
            # Degree fallback: row-sum of the symmetrized adjacency BEFORE
            # self-loops are added, as a single float32 column.
            deg = adj.sum(axis=1, keepdims=True).astype(np.float32)
            node_feats = deg

        # Self-loops on the diagonal (after degree is computed).
        if add_self_loops:
            np.fill_diagonal(adj, 1.0)

        label = glbl_to_idx[int(graph_labels_raw[g])]
        label_distribution[label] += 1
        samples.append(
            GraphSample(
                adjacency=adj,
                node_features=node_feats,
                label=int(label),
                num_nodes=int(graph_num_nodes[g]),
            )
        )

    avg_nodes = float(graph_num_nodes.mean()) if num_graphs else 0.0
    metadata: Dict[str, Any] = {
        "name": name,
        "num_graphs": num_graphs,
        "num_classes": num_classes,
        "num_node_features": int(feat_dim),
        "avg_nodes": avg_nodes,
        "label_distribution": label_distribution,
        "feature_source": feature_source,
        "add_self_loops": add_self_loops,
    }
    return samples, metadata


# ---------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------

def load_tudataset(
        name: str,
        cache_root: str = DEFAULT_GRAPH_CACHE,
        add_self_loops: bool = True,
        download: bool = True,
) -> Tuple[List[GraphSample], Dict[str, Any]]:
    """Load a TUDataset benchmark as a list of dense per-graph records.

    Args:
        name: Dataset name, e.g. ``"MUTAG"`` or ``"PROTEINS"``.
        cache_root: On-disk cache root. Defaults to
            :data:`DEFAULT_GRAPH_CACHE`. Must not be the repo or repo SSD.
        add_self_loops: If ``True`` (default), set the adjacency diagonal to
            ``1`` after symmetrization so a node attends to itself.
        download: If ``True`` (default), download+extract the archive when it
            is not already cached. If ``False``, the data must already exist
            under ``cache_root/{name}/`` or a ``FileNotFoundError`` is raised.

    Returns:
        A tuple ``(samples, metadata)`` where ``samples`` is a list of
        :class:`GraphSample` and ``metadata`` is a dict with keys
        ``name``, ``num_graphs``, ``num_classes``, ``num_node_features``,
        ``avg_nodes``, ``label_distribution``, ``feature_source`` and
        ``add_self_loops``.

    Raises:
        FileNotFoundError: If ``download=False`` and the data is not cached,
            or the extracted archive is missing required files.
    """
    cache_root = os.path.abspath(os.path.expanduser(cache_root))
    data_dir = os.path.join(cache_root, name)
    edge_file = os.path.join(data_dir, f"{name}_A.txt")

    if not os.path.isfile(edge_file):
        if download:
            data_dir = download_tudataset(name, cache_root=cache_root)
        else:
            raise FileNotFoundError(
                f"TUDataset '{name}' not found at {data_dir} and "
                f"download=False."
            )

    logger.info(f"Parsing TUDataset '{name}' from {data_dir}")
    samples, metadata = _parse_tudataset(
        name=name, data_dir=data_dir, add_self_loops=add_self_loops
    )
    logger.info(
        f"Parsed '{name}': {metadata['num_graphs']} graphs, "
        f"{metadata['num_classes']} classes, "
        f"{metadata['num_node_features']} node features "
        f"({metadata['feature_source']}), "
        f"avg_nodes={metadata['avg_nodes']:.2f}"
    )
    return samples, metadata

# ---------------------------------------------------------------------
