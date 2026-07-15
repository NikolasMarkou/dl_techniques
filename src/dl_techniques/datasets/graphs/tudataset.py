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
# Laplacian-eigenvector positional encoding (PE)
# ---------------------------------------------------------------------

def compute_laplacian_pe(adjacency: np.ndarray, k: int = 15) -> np.ndarray:
    """Compute Laplacian-eigenvector positional encodings for one graph.

    The positional encoding is the set of eigenvectors of the symmetric
    normalized graph Laplacian associated with the ``k`` smallest **non-zero**
    eigenvalues (Dwivedi & Bresson 2021). The trivial constant eigenvector at
    index ``0`` (eigenvalue ``0`` for a connected graph) carries no positional
    information and is skipped, so columns ``1 .. k`` of the eigenbasis are
    returned.

    The Laplacian is::

        L = I - D̃^{-1/2} Ã D̃^{-1/2}

    where ``Ã`` is the adjacency **without self-loops** (the diagonal is zeroed
    before the Laplacian is formed, regardless of whether the caller loaded the
    graph with ``add_self_loops=True``) and ``D̃`` is its degree matrix. This is
    deliberate: self-loops shift every eigenvalue by a constant and blur the
    normalized-Laplacian spectrum, so they are stripped here even though the
    attention adjacency downstream keeps them. Degree-0 (isolated) nodes are
    guarded by setting their ``D^{-1/2}`` entry to ``0`` (rather than dividing
    by zero), which leaves their Laplacian row/column as a unit diagonal.

    Sign ambiguity (each eigenvector is defined up to a global sign, and
    degenerate/disconnected spectra add rotational ambiguity within an
    eigenspace) is **not** resolved here; it is absorbed downstream by the
    per-epoch random sign-flip augmentation (:func:`sign_flip_pe`).

    Small-graph handling: if the number of available non-trivial eigenvectors
    (``n - 1``) is smaller than ``k``, the returned matrix is **zero-padded** on
    the right to width ``k``. This keeps the PE width constant across a dataset
    with variable graph sizes (needed for a fixed downstream projection).

    Args:
        adjacency: Dense ``(n, n)`` adjacency (symmetric). Self-loops, if
            present on the diagonal, are ignored for the Laplacian.
        k: Number of positional-encoding columns to return (default ``15``).

    Returns:
        A ``(n, k)`` ``float32`` array of positional-encoding features. Columns
        corresponding to available non-trivial eigenvectors are orthonormal
        across the graph's nodes; any trailing padding columns are all-zero.
    """
    # Imported lazily: the text parser above does not need scipy.linalg, so
    # keeping this out of the module import path keeps the loader import-light.
    from scipy.linalg import eigh

    n = int(adjacency.shape[0])
    if n == 0:
        return np.zeros((0, k), dtype=np.float32)

    # Strip self-loops for the Laplacian (see docstring).
    a = np.asarray(adjacency, dtype=np.float64).copy()
    np.fill_diagonal(a, 0.0)

    deg = a.sum(axis=1)
    # Guarded D^{-1/2}: isolated nodes get 0 instead of inf.
    with np.errstate(divide="ignore"):
        d_inv_sqrt = np.where(deg > 0.0, 1.0 / np.sqrt(deg), 0.0)

    # L = I - D^{-1/2} A D^{-1/2}
    lap = np.eye(n, dtype=np.float64) - (d_inv_sqrt[:, None] * a * d_inv_sqrt[None, :])
    # Numerically symmetrize to protect eigh from tiny asymmetries.
    lap = 0.5 * (lap + lap.T)

    # Ascending eigenvalues; eigenvectors are columns, orthonormal.
    _, eigvecs = eigh(lap)

    # Skip the trivial index-0 eigenvector; take the next k columns.
    take = min(k, max(n - 1, 0))
    pe = np.zeros((n, k), dtype=np.float32)
    if take > 0:
        pe[:, :take] = eigvecs[:, 1:1 + take].astype(np.float32)
    return pe


def sign_flip_pe(pe: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Randomly flip the sign of each PE column (per-epoch augmentation).

    Each of the ``k`` eigenvector columns is multiplied by an independent
    random ``±1``. This is the standard Laplacian-PE augmentation that makes a
    model invariant to the arbitrary sign of each eigenvector (and, together
    over epochs, robust to the sign/rotation ambiguity of degenerate spectra).
    It is a **separate** function so a ``tf.data`` pipeline can re-apply it
    per-epoch from its own seeded ``numpy`` generator (see
    :func:`build_tudataset_graph_dataset`).

    Zero-valued padding columns are unaffected in magnitude (``±0 == 0``).

    Args:
        pe: A ``(n, k)`` positional-encoding matrix.
        rng: A seeded ``numpy.random.Generator`` used to draw the signs.

    Returns:
        A new ``(n, k)`` ``float32`` array with per-column signs flipped;
        ``abs(result) == abs(pe)`` element-wise.
    """
    k = int(pe.shape[1])
    signs = rng.integers(0, 2, size=k).astype(np.float32) * 2.0 - 1.0  # {-1, +1}
    return (np.asarray(pe, dtype=np.float32) * signs[None, :]).astype(np.float32)


# ---------------------------------------------------------------------
# dense batch / pad + node-validity mask
# ---------------------------------------------------------------------

def collate_graph_batch(
        samples: List[GraphSample],
        pe_list: List[np.ndarray],
        k: int,
        max_nodes: Optional[int] = None,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Dense-pad a list of variable-size graphs into a single batch.

    Pads every graph in the batch to a common node dimension ``N`` and emits a
    rank-2 node-validity mask marking real vs padding nodes. ``N`` is the
    per-batch maximum node count when ``max_nodes is None``, otherwise the fixed
    ``max_nodes`` cap. Padding rows/columns are **all-zero** (a clean,
    all-zero PAD token — the ``fp16`` hazard surface that the downstream model
    excludes from the energy via the node mask; that is a model-side concern,
    handled in a later step, not here).

    If a graph has more than ``max_nodes`` nodes it is truncated to the first
    ``max_nodes`` nodes (defensive; the ``tf.data`` builder filters oversized
    graphs before batching so truncation does not normally occur).

    Args:
        samples: The graphs in this batch (variable ``n`` each).
        pe_list: One ``(n, k)`` positional-encoding array per graph, aligned
            with ``samples`` (already sign-flipped for this epoch if desired).
        k: Positional-encoding width (columns of every ``pe_list`` entry).
        max_nodes: Fixed node cap, or ``None`` for per-batch dynamic padding.

    Returns:
        A tuple ``(inputs, labels)`` where ``inputs`` is a dict with keys:

            * ``"node_features"`` — ``(B, N, F)`` ``float32``
            * ``"adjacency"``     — ``(B, N, N)`` ``float32`` (0/1, symmetric,
              self-loops as loaded)
            * ``"pe"``            — ``(B, N, k)`` ``float32``
            * ``"node_mask"``     — ``(B, N)`` ``float32`` (1 = real, 0 = PAD)

        and ``labels`` is a ``(B,)`` ``int32`` array of graph class labels.
    """
    b = len(samples)
    feat_dim = int(samples[0].node_features.shape[1])
    if max_nodes is not None:
        n_max = int(max_nodes)
    else:
        n_max = max(int(s.adjacency.shape[0]) for s in samples)

    adjacency = np.zeros((b, n_max, n_max), dtype=np.float32)
    node_features = np.zeros((b, n_max, feat_dim), dtype=np.float32)
    pe = np.zeros((b, n_max, k), dtype=np.float32)
    node_mask = np.zeros((b, n_max), dtype=np.float32)
    labels = np.zeros((b,), dtype=np.int32)

    for i, sample in enumerate(samples):
        n = min(int(sample.adjacency.shape[0]), n_max)
        adjacency[i, :n, :n] = sample.adjacency[:n, :n]
        node_features[i, :n, :] = sample.node_features[:n, :]
        pe[i, :n, :] = pe_list[i][:n, :]
        node_mask[i, :n] = 1.0
        labels[i] = int(sample.label)

    inputs = {
        "node_features": node_features,
        "adjacency": adjacency,
        "pe": pe,
        "node_mask": node_mask,
    }
    return inputs, labels


# ---------------------------------------------------------------------
# deterministic stratified split
# ---------------------------------------------------------------------

def _stratified_split(
        labels: np.ndarray,
        rng: np.random.Generator,
        ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic, class-stratified train/val/test index split.

    Within each class the indices are shuffled with ``rng`` and partitioned by
    ``ratios`` (train, val, test). The three returned index arrays are disjoint
    and together cover every input index exactly once.

    Args:
        labels: ``(G,)`` integer class labels.
        rng: Seeded generator controlling the per-class shuffle.
        ratios: ``(train, val, test)`` fractions; must sum to ``1.0``.

    Returns:
        ``(train_idx, val_idx, test_idx)`` sorted int64 arrays.
    """
    train_r, val_r, _ = ratios
    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []
    for c in np.unique(labels):
        cls = np.where(labels == c)[0]
        rng.shuffle(cls)
        n = len(cls)
        n_train = int(round(n * train_r))
        n_val = int(round(n * val_r))
        # Clamp so val does not overrun and test gets the remainder.
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
        train_idx.extend(cls[:n_train].tolist())
        val_idx.extend(cls[n_train:n_train + n_val].tolist())
        test_idx.extend(cls[n_train + n_val:].tolist())
    return (
        np.sort(np.array(train_idx, dtype=np.int64)),
        np.sort(np.array(val_idx, dtype=np.int64)),
        np.sort(np.array(test_idx, dtype=np.int64)),
    )


# ---------------------------------------------------------------------
# tf.data builder
# ---------------------------------------------------------------------

def build_tudataset_graph_dataset(
        name: str,
        split: str = "train",
        batch_size: int = 32,
        k_pe: int = 15,
        max_nodes: Optional[int] = None,
        cache_root: str = DEFAULT_GRAPH_CACHE,
        add_self_loops: bool = True,
        sign_flip: bool = True,
        seed: Optional[int] = None,
        shuffle: bool = True,
) -> Tuple["Any", Dict[str, Any]]:
    """Build a ``tf.data.Dataset`` of dense-batched graphs for the graph ET.

    Loads a TUDataset benchmark (via :func:`load_tudataset`), precomputes the
    Laplacian positional encoding once per graph (``scipy.linalg.eigh`` is
    ``O(n^3)`` — done at load, cached in memory), performs a deterministic
    **stratified 80/10/10** train/val/test split by graph label, and yields
    dense-padded batches shaped for the downstream Graph Energy Transformer.

    **Input contract (consumed verbatim by the graph-ET models/trainers).**
    Each dataset element is a ``(inputs, label)`` tuple where ``inputs`` is a
    dict with these exact keys and shapes::

        inputs = {
            "node_features": (B, N, F) float32,   # per-node features
            "adjacency":     (B, N, N) float32,   # 0/1 symmetric, self-loops as loaded
            "pe":            (B, N, k) float32,    # Laplacian positional encoding
            "node_mask":     (B, N)    float32,    # 1 = real node, 0 = PAD
        }
        label = (B,) int32                          # graph class label

    ``N`` is the per-batch maximum node count when ``max_nodes is None``,
    otherwise the fixed ``max_nodes`` cap (graphs larger than the cap are
    dropped from the split before batching). PAD rows/cols are all-zero.

    **Per-epoch sign-flip (deterministic).** When ``sign_flip=True`` the
    augmentation is applied **numpy-side inside a re-invoked generator**, not
    with ``tf.random``: the dataset is built with
    ``tf.data.Dataset.from_generator`` over a callable that closes over a single
    seeded ``numpy.random.Generator``. That generator is instantiated once and
    its stream advances across epochs, so the whole sequence (shuffle order +
    per-graph column signs) is fully determined by ``seed`` yet differs every
    epoch. This is simpler and more clearly seedable than stateless ``tf``
    random ops in a ``.map``.

    Args:
        name: TUDataset name, e.g. ``"MUTAG"`` or ``"PROTEINS"``.
        split: One of ``"train"``, ``"val"``, ``"test"``.
        batch_size: Graphs per batch (the last batch may be smaller).
        k_pe: Positional-encoding width (columns).
        max_nodes: Fixed node cap, or ``None`` for per-batch dynamic padding.
        cache_root: On-disk cache root for the raw dataset.
        add_self_loops: Passed to :func:`load_tudataset` (adjacency diagonal).
        sign_flip: Apply per-epoch PE sign-flip augmentation (train only makes
            sense, but honored for any split).
        seed: Base seed for the split, shuffle order, and sign-flip stream.
        shuffle: Reshuffle the split's graph order every epoch.

    Returns:
        ``(dataset, metadata)`` where ``dataset`` is a batched
        ``tf.data.Dataset`` yielding the contract above and ``metadata`` extends
        the loader metadata with ``split``, ``split_sizes``, ``k_pe``,
        ``max_nodes``, ``batch_size`` and ``num_node_features``.
    """
    # tf imported lazily to keep the parser/PE/collate path tf-free (loader
    # unit tests exercise those without importing tensorflow).
    import tensorflow as tf

    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be train/val/test, got {split!r}")

    samples, metadata = load_tudataset(
        name=name, cache_root=cache_root, add_self_loops=add_self_loops
    )

    # Optionally drop graphs that exceed the fixed cap (never truncate silently).
    if max_nodes is not None:
        kept = [s for s in samples if int(s.adjacency.shape[0]) <= int(max_nodes)]
        dropped = len(samples) - len(kept)
        if dropped:
            logger.warning(
                f"Dropped {dropped}/{len(samples)} '{name}' graphs exceeding "
                f"max_nodes={max_nodes}"
            )
        samples = kept

    # Precompute PE once per graph (O(n^3) eigh at load, cached in memory).
    logger.info(
        f"Precomputing Laplacian PE (k={k_pe}) for {len(samples)} '{name}' graphs"
    )
    pe_cache = [compute_laplacian_pe(s.adjacency, k=k_pe) for s in samples]

    labels = np.array([s.label for s in samples], dtype=np.int64)
    split_rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = _stratified_split(labels, split_rng)
    split_map = {"train": train_idx, "val": val_idx, "test": test_idx}
    indices = split_map[split]

    feat_dim = int(metadata["num_node_features"])

    # One persistent generator drives shuffle order + sign-flip across epochs.
    # A fixed per-split offset (NOT Python's salted str hash) keeps the three
    # splits' streams distinct yet reproducible for a given ``seed``.
    _split_offset = {"train": 0, "val": 1, "test": 2}[split]
    epoch_rng = np.random.default_rng(
        None if seed is None else int(seed) + _split_offset
    )

    def _generator():
        order = np.array(indices, dtype=np.int64)
        if shuffle:
            epoch_rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            batch_idx = order[start:start + batch_size]
            batch_samples = [samples[j] for j in batch_idx]
            if sign_flip:
                batch_pe = [sign_flip_pe(pe_cache[j], epoch_rng) for j in batch_idx]
            else:
                batch_pe = [pe_cache[j] for j in batch_idx]
            yield collate_graph_batch(
                batch_samples, batch_pe, k=k_pe, max_nodes=max_nodes
            )

    output_signature = (
        {
            "node_features": tf.TensorSpec(
                shape=(None, None, feat_dim), dtype=tf.float32
            ),
            "adjacency": tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
            "pe": tf.TensorSpec(shape=(None, None, k_pe), dtype=tf.float32),
            "node_mask": tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        },
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
    )

    dataset = tf.data.Dataset.from_generator(
        _generator, output_signature=output_signature
    ).prefetch(tf.data.AUTOTUNE)

    metadata = dict(metadata)
    metadata.update(
        {
            "split": split,
            "split_sizes": {
                "train": int(len(train_idx)),
                "val": int(len(val_idx)),
                "test": int(len(test_idx)),
            },
            "k_pe": int(k_pe),
            "max_nodes": max_nodes,
            "batch_size": int(batch_size),
            "num_node_features": feat_dim,
        }
    )
    return dataset, metadata

# ---------------------------------------------------------------------
