"""Node-anomaly fraud-graph loader for the Graph Energy Transformer (variant B).

This module provides the data path for **variant B** (per-node anomaly
detection) of the Graph Energy Transformer: ONE large graph with per-node
binary labels (``0`` = benign, ``1`` = fraud) and heavy class imbalance. It
reads the CARE-GNN ``.mat`` fraud benchmarks (Amazon, YelpChi) with
``scipy.io.loadmat`` (classic MATLAB <7.3) and an ``h5py`` fallback for the
MATLAB v7.3/HDF5 sub-format, using **only** numpy/scipy(+h5py) — no
``torch``/``dgl``/``spektral``/``torch_geometric`` dependency.

Scope of this module (graph-ET plan, Phase-0 step 3):

    * ``load_fraud_graph`` — idempotent download + extract + parse a CARE-GNN
      ``.mat`` benchmark into a :class:`FraudGraph` (whole-graph adjacency kept
      **sparse**, never densified).
    * ``sample_subgraph`` — bounded k-hop neighbour-sampling subgraph extractor.
      A full dense ``N×N`` over the whole graph (Amazon ~11.9k nodes, YelpChi
      ~45k) is infeasible, so variant B feeds the ET a small, size-capped dense
      symmetric subgraph per target node, with the target node pinned at index 0.
    * ``build_fraud_subgraph_dataset`` — ``tf.data`` builder emitting the
      variant-B input contract (see that function's docstring).
    * ``make_synthetic_fraud_graph`` — a network-free synthetic fallback used by
      CI/tests and ``--dataset synthetic``.

Design choices (documented per the plan's requirements):

    * **Whole-graph adjacency stays SPARSE.** The benchmark graphs are far too
      large for a dense ``N×N``. :class:`FraudGraph` keeps the adjacency as a
      ``scipy.sparse`` CSR matrix; only the bounded per-target subgraphs are ever
      densified (to a small ``(max_nodes, max_nodes)`` block).
    * **Which adjacency.** The CARE-GNN ``.mat`` files ship a homogeneous
      adjacency ``homo`` (the union of all relation types) plus the individual
      relation adjacencies (Amazon: ``net_upu``/``net_usu``/``net_uvu``; YelpChi:
      ``net_rur``/``net_rtr``/``net_rsr``). This loader uses ``homo`` as THE
      graph (symmetrized to binary), which is the single-relation homogeneous
      view the CARE-GNN authors provide for exactly this purpose. If ``homo`` is
      absent the loader falls back to the symmetrized binary union of the
      relation adjacencies.
    * **Target at index 0.** :func:`sample_subgraph` always places the target
      node at local index 0 of the returned subgraph, because the variant-B head
      (step 5) reads the target representation from index 0.
    * **Uniform padding.** Every subgraph is zero-padded up to ``max_nodes`` with
      an all-zero PAD token and a rank-1 ``node_mask`` (1 = real, 0 = PAD) so all
      subgraphs share a uniform ``(max_nodes, ...)`` shape. The PAD rows keep the
      adjacency exactly symmetric (all-zero) and are excluded from the energy
      downstream via the node mask.
"""

import os
import zipfile
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# Reuse the shared cache root and the deterministic stratified splitter from the
# TUDataset loader rather than re-deriving them here (DRY: one cache-root fact,
# one split rule for the whole `datasets/graphs/` package).
from .tudataset import DEFAULT_GRAPH_CACHE, _stratified_split

# ---------------------------------------------------------------------
# module constants
# ---------------------------------------------------------------------

# CARE-GNN fraud benchmarks (HTTP-200 verified). Each archive extracts to a
# single top-level `{Mat}` file. Keys are the lowercase dataset names used
# throughout the public API; values carry the download URL and the archived
# `.mat` filename (which is capitalised in the archive).
FRAUD_DATASETS: Dict[str, Dict[str, str]] = {
    "amazon": {
        "url": "https://raw.githubusercontent.com/YingtongDou/CARE-GNN/master/data/Amazon.zip",
        "mat": "Amazon.mat",
    },
    "yelpchi": {
        "url": "https://raw.githubusercontent.com/YingtongDou/CARE-GNN/master/data/YelpChi.zip",
        "mat": "YelpChi.mat",
    },
}

# CARE-GNN relation-adjacency keys per dataset, used only when a `homo` key is
# absent (fallback: symmetrized binary union of relations).
_RELATION_KEYS: Dict[str, Tuple[str, ...]] = {
    "amazon": ("net_upu", "net_usu", "net_uvu"),
    "yelpchi": ("net_rur", "net_rtr", "net_rsr"),
}


# ---------------------------------------------------------------------
# data record
# ---------------------------------------------------------------------

@dataclass
class FraudGraph:
    """A single large node-anomaly graph with per-node binary labels.

    Attributes:
        adjacency: Symmetric binary whole-graph adjacency as a
            ``scipy.sparse`` CSR matrix of shape ``(num_nodes, num_nodes)``.
            Kept **sparse** and never densified whole; only bounded subgraphs
            (see :func:`sample_subgraph`) are densified.
        features: Dense per-node feature matrix of shape
            ``(num_nodes, num_features)`` and dtype ``float32``.
        labels: Per-node integer label vector of shape ``(num_nodes,)`` with
            values in ``{0, 1}`` (``0`` = benign, ``1`` = fraud/anomalous).
        num_nodes: Number of nodes ``N``.
        num_features: Feature dimension ``F``.
        anomaly_ratio: Fraction of nodes with label ``1`` (a small positive
            fraction for the imbalanced fraud benchmarks).
        class_counts: Mapping ``{0: n_benign, 1: n_anomalous}``.
        name: Dataset name (``"amazon"``, ``"yelpchi"`` or ``"synthetic"``).
    """

    adjacency: sp.csr_matrix
    features: np.ndarray
    labels: np.ndarray
    num_nodes: int
    num_features: int
    anomaly_ratio: float
    class_counts: Dict[int, int]
    name: str


# ---------------------------------------------------------------------
# download / cache
# ---------------------------------------------------------------------

def download_fraud_dataset(
        name: str,
        cache_root: str = DEFAULT_GRAPH_CACHE,
) -> str:
    """Download and extract a CARE-GNN fraud ``.mat`` archive into the cache.

    The archive ``{Name}.zip`` is fetched from the CARE-GNN GitHub mirror and
    unzipped into ``cache_root/{name}/`` so the ``.mat`` lands at
    ``cache_root/{name}/{Name}.mat``. The operation is idempotent: if the
    extracted ``.mat`` already exists the download and extraction are skipped.

    Args:
        name: Dataset key, one of :data:`FRAUD_DATASETS` (``"amazon"`` or
            ``"yelpchi"``).
        cache_root: Directory under which ``{name}/`` is created. Defaults to
            :data:`DEFAULT_GRAPH_CACHE`. Never point this at the repository or
            the repo SSD.

    Returns:
        Absolute path to the extracted ``.mat`` file.

    Raises:
        ValueError: If ``name`` is not a known fraud dataset.
        urllib.error.URLError: If the download host is unreachable.
        FileNotFoundError: If extraction completed but the expected ``.mat`` is
            missing (malformed archive).
    """
    if name not in FRAUD_DATASETS:
        raise ValueError(
            f"Unknown fraud dataset {name!r}; known: {sorted(FRAUD_DATASETS)}"
        )

    spec = FRAUD_DATASETS[name]
    cache_root = os.path.abspath(os.path.expanduser(cache_root))
    data_dir = os.path.join(cache_root, name)
    mat_path = os.path.join(data_dir, spec["mat"])

    if os.path.isfile(mat_path):
        logger.info(f"Fraud dataset '{name}' already cached at {mat_path}")
        return mat_path

    os.makedirs(data_dir, exist_ok=True)
    url = spec["url"]
    zip_path = os.path.join(data_dir, f"{name}.zip")

    logger.info(f"Downloading fraud dataset '{name}' from {url}")
    request = urllib.request.Request(
        url, headers={"User-Agent": "dl-techniques/graph-loader"}
    )
    with urllib.request.urlopen(request) as response, open(zip_path, "wb") as fh:
        fh.write(response.read())

    logger.info(f"Extracting {zip_path} into {data_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)

    if not os.path.isfile(mat_path):
        raise FileNotFoundError(
            f"Extraction of '{name}' did not produce {mat_path}. "
            f"Archive layout may differ from the expected CARE-GNN format."
        )

    try:
        os.remove(zip_path)
    except OSError:
        pass

    logger.info(f"Fraud dataset '{name}' ready at {mat_path}")
    return mat_path


# ---------------------------------------------------------------------
# .mat parsing (scipy classic + h5py v7.3 fallback)
# ---------------------------------------------------------------------

def _as_sparse_binary_symmetric(matrix: Any) -> sp.csr_matrix:
    """Coerce an adjacency (sparse or dense) to symmetric binary CSR.

    Diagonal self-loops are stripped here (the subgraph extractor re-adds them
    per subgraph on demand). ``A = ((A + A.T) > 0)`` guarantees ``A == A.T``.
    """
    a = sp.csr_matrix(matrix)
    a = a + a.T
    a.data[:] = 1.0
    a = a.astype(np.float32)
    a.setdiag(0.0)
    a.eliminate_zeros()
    return a.tocsr()


def _read_mat_scipy(path: str) -> Tuple[Dict[str, Any], str]:
    """Load a ``.mat`` with ``scipy.io.loadmat``; signal a v7.3 file.

    Returns ``(container, loader)`` where ``loader`` is ``"scipy"``. Raises
    :class:`NotImplementedError` for MATLAB v7.3 files (caller falls back to
    ``h5py``).
    """
    import scipy.io as sio

    container = sio.loadmat(path)
    return container, "scipy"


def _read_mat_h5py(path: str) -> Tuple[Any, str]:
    """Open a MATLAB v7.3 (HDF5) ``.mat`` with ``h5py`` as a fallback."""
    import h5py

    return h5py.File(path, "r"), "h5py"


def _h5_dense(container: Any, key: str) -> np.ndarray:
    """Read a dense array from an h5py MATLAB-v7.3 container (transpose back)."""
    # MATLAB v7.3 stores arrays column-major; h5py returns them transposed.
    return np.array(container[key]).T


def _h5_sparse(container: Any, key: str, num_nodes: int) -> sp.csr_matrix:
    """Reconstruct a MATLAB-v7.3 sparse adjacency (CSC data/ir/jc) via h5py."""
    grp = container[key]
    data = np.array(grp["data"]).ravel()
    ir = np.array(grp["ir"]).ravel().astype(np.int64)
    jc = np.array(grp["jc"]).ravel().astype(np.int64)
    return sp.csc_matrix((data, ir, jc), shape=(num_nodes, num_nodes)).tocsr()


def _extract_fraud_arrays(
        path: str,
        name: str,
) -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray, List[str], str]:
    """Parse a fraud ``.mat`` into ``(adjacency, features, labels, keys, chosen)``.

    Tries ``scipy.io.loadmat`` first and falls back to ``h5py`` for MATLAB v7.3
    files. Chooses the ``homo`` adjacency when present, otherwise the symmetrized
    binary union of the dataset's relation adjacencies.

    Returns:
        A tuple ``(adjacency, features, labels, mat_keys, chosen_adjacency)``
        where ``adjacency`` is symmetric binary CSR, ``features`` is
        ``(N, F) float32``, ``labels`` is ``(N,) int64`` in ``{0, 1}``,
        ``mat_keys`` is the inspected list of top-level keys, and
        ``chosen_adjacency`` names which key(s) were used for the graph.
    """
    try:
        container, loader = _read_mat_scipy(path)
    except NotImplementedError:
        logger.info(
            f"'{name}' is a MATLAB v7.3 (HDF5) .mat; falling back to h5py"
        )
        container, loader = _read_mat_h5py(path)

    if loader == "scipy":
        mat_keys = [k for k in container.keys() if not k.startswith("__")]
        logger.info(f"Inspected '{name}' .mat keys (scipy): {sorted(mat_keys)}")

        features = container["features"]
        features = (
            features.toarray() if sp.issparse(features) else np.asarray(features)
        )
        features = features.astype(np.float32)
        labels = np.asarray(container["label"]).ravel().astype(np.int64)

        if "homo" in container:
            adjacency = _as_sparse_binary_symmetric(container["homo"])
            chosen = "homo"
        else:
            rels = [
                r for r in _RELATION_KEYS.get(name, ()) if r in container
            ]
            if not rels:
                raise KeyError(
                    f"'{name}' .mat has neither 'homo' nor known relations "
                    f"{_RELATION_KEYS.get(name, ())}; keys={sorted(mat_keys)}"
                )
            union = sp.csr_matrix(container[rels[0]])
            for r in rels[1:]:
                union = union + sp.csr_matrix(container[r])
            adjacency = _as_sparse_binary_symmetric(union)
            chosen = "union(" + "+".join(rels) + ")"
    else:  # h5py v7.3
        mat_keys = [k for k in container.keys()]
        logger.info(f"Inspected '{name}' .mat keys (h5py): {sorted(mat_keys)}")

        labels = _h5_dense(container, "label").ravel().astype(np.int64)
        num_nodes = int(labels.shape[0])
        features = _h5_dense(container, "features").astype(np.float32)
        if features.shape[0] != num_nodes and features.shape[1] == num_nodes:
            features = features.T

        if "homo" in container:
            adjacency = _as_sparse_binary_symmetric(
                _h5_sparse(container, "homo", num_nodes)
            )
            chosen = "homo"
        else:
            rels = [
                r for r in _RELATION_KEYS.get(name, ()) if r in container
            ]
            if not rels:
                raise KeyError(
                    f"'{name}' v7.3 .mat has neither 'homo' nor known relations"
                )
            union = _h5_sparse(container, rels[0], num_nodes)
            for r in rels[1:]:
                union = union + _h5_sparse(container, r, num_nodes)
            adjacency = _as_sparse_binary_symmetric(union)
            chosen = "union(" + "+".join(rels) + ")"
        container.close()

    logger.info(
        f"'{name}': chose adjacency '{chosen}', "
        f"N={adjacency.shape[0]}, F={features.shape[1]}"
    )
    return adjacency, features, labels, sorted(mat_keys), chosen


# ---------------------------------------------------------------------
# public loader
# ---------------------------------------------------------------------

def _finalize_fraud_graph(
        name: str,
        adjacency: sp.csr_matrix,
        features: np.ndarray,
        labels: np.ndarray,
) -> FraudGraph:
    """Assemble a :class:`FraudGraph` and compute class metadata."""
    num_nodes = int(adjacency.shape[0])
    num_features = int(features.shape[1])
    n_anom = int((labels == 1).sum())
    n_benign = int((labels == 0).sum())
    anomaly_ratio = float(n_anom) / float(max(num_nodes, 1))
    return FraudGraph(
        adjacency=adjacency.tocsr(),
        features=features,
        labels=labels,
        num_nodes=num_nodes,
        num_features=num_features,
        anomaly_ratio=anomaly_ratio,
        class_counts={0: n_benign, 1: n_anom},
        name=name,
    )


def load_fraud_graph(
        name: str = "amazon",
        cache_root: str = DEFAULT_GRAPH_CACHE,
        download: bool = True,
) -> FraudGraph:
    """Load a CARE-GNN fraud benchmark as a :class:`FraudGraph`.

    Downloads (idempotently) and parses the ``.mat`` benchmark, choosing the
    ``homo`` homogeneous adjacency (symmetrized to binary) as THE graph. The
    whole-graph adjacency is kept as a ``scipy.sparse`` CSR matrix and never
    densified — use :func:`sample_subgraph` to obtain small dense subgraphs.

    Args:
        name: Dataset key, ``"amazon"`` (primary/smaller) or ``"yelpchi"``.
        cache_root: On-disk cache root. Defaults to :data:`DEFAULT_GRAPH_CACHE`.
            Must not be the repository or the repo SSD.
        download: If ``True`` (default), download+extract when not cached. If
            ``False``, the ``.mat`` must already exist under
            ``cache_root/{name}/``.

    Returns:
        A :class:`FraudGraph`.

    Raises:
        ValueError: If ``name`` is not a known fraud dataset.
        FileNotFoundError: If ``download=False`` and the data is not cached.
    """
    if name not in FRAUD_DATASETS:
        raise ValueError(
            f"Unknown fraud dataset {name!r}; known: {sorted(FRAUD_DATASETS)}"
        )

    cache_root = os.path.abspath(os.path.expanduser(cache_root))
    mat_path = os.path.join(cache_root, name, FRAUD_DATASETS[name]["mat"])

    if not os.path.isfile(mat_path):
        if download:
            mat_path = download_fraud_dataset(name, cache_root=cache_root)
        else:
            raise FileNotFoundError(
                f"Fraud dataset '{name}' not found at {mat_path} and "
                f"download=False."
            )

    logger.info(f"Parsing fraud dataset '{name}' from {mat_path}")
    adjacency, features, labels, _keys, _chosen = _extract_fraud_arrays(
        mat_path, name
    )
    graph = _finalize_fraud_graph(name, adjacency, features, labels)
    logger.info(
        f"Loaded '{name}': N={graph.num_nodes}, F={graph.num_features}, "
        f"anomaly_ratio={graph.anomaly_ratio:.4f}, "
        f"class_counts={graph.class_counts}"
    )
    return graph


# ---------------------------------------------------------------------
# k-hop neighbour-sampling subgraph extractor
# ---------------------------------------------------------------------

def sample_subgraph(
        fraud_graph: FraudGraph,
        target_node: int,
        num_hops: int = 2,
        max_nodes: int = 64,
        rng: Optional[np.random.Generator] = None,
        add_self_loops: bool = True,
) -> Dict[str, Any]:
    """Extract a bounded k-hop subgraph around one target node.

    Performs a breadth-first expansion out from ``target_node``; if a hop's
    frontier would push the selected-node count past ``max_nodes`` it is randomly
    sampled down to fit. The target node is pinned at local index 0 (the
    variant-B head reads its representation from index 0). The returned dense
    adjacency is symmetric and, when ``add_self_loops`` is set, has a unit
    diagonal on the real (non-PAD) nodes. Everything is zero-padded up to
    ``max_nodes`` with a rank-1 ``node_mask`` marking real vs PAD rows, so all
    subgraphs share a uniform ``(max_nodes, ...)`` shape.

    Args:
        fraud_graph: The whole (sparse) graph to sample from.
        target_node: Global node id to centre the subgraph on.
        num_hops: BFS depth ``k`` (default 2).
        max_nodes: Hard cap on subgraph size (default 64); the returned tensors
            are padded to exactly this size.
        rng: Seeded ``numpy.random.Generator`` for frontier down-sampling. A
            fresh default generator is used when ``None``.
        add_self_loops: If ``True`` (default), set the diagonal of the real
            nodes to 1 after symmetrization.

    Returns:
        A dict with keys:

            * ``"adjacency"``     — ``(max_nodes, max_nodes)`` ``float32``,
              symmetric, self-loops on real nodes as requested.
            * ``"node_features"`` — ``(max_nodes, F)`` ``float32`` (PAD rows 0).
            * ``"node_mask"``     — ``(max_nodes,)`` ``float32`` (1 = real, 0 = PAD).
            * ``"target_index"``  — ``int`` local index of the target (always 0).
            * ``"target_label"``  — ``int`` label of the target node (0/1).
    """
    if rng is None:
        rng = np.random.default_rng()

    adj = fraud_graph.adjacency  # CSR
    indptr = adj.indptr
    indices = adj.indices

    def _neighbours(node: int) -> np.ndarray:
        return indices[indptr[node]:indptr[node + 1]]

    selected: List[int] = [int(target_node)]
    seen = {int(target_node)}
    frontier: List[int] = [int(target_node)]

    for _hop in range(num_hops):
        if len(selected) >= max_nodes:
            break
        # Gather all new neighbours of the current frontier.
        cand: List[int] = []
        cand_seen = set()
        for node in frontier:
            for nb in _neighbours(node):
                nb = int(nb)
                if nb not in seen and nb not in cand_seen:
                    cand_seen.add(nb)
                    cand.append(nb)
        if not cand:
            break
        budget = max_nodes - len(selected)
        cand_arr = np.array(cand, dtype=np.int64)
        if cand_arr.shape[0] > budget:
            cand_arr = rng.choice(cand_arr, size=budget, replace=False)
        next_frontier = cand_arr.tolist()
        for nb in next_frontier:
            seen.add(int(nb))
            selected.append(int(nb))
        frontier = next_frontier

    n = len(selected)
    sel = np.array(selected, dtype=np.int64)

    # Densify ONLY this small subgraph block (never the whole graph).
    sub = adj[sel][:, sel].toarray().astype(np.float32)
    # Symmetrize defensively (already symmetric, but padding math below relies
    # on it) and binarize.
    sub = ((sub + sub.T) > 0.0).astype(np.float32)
    if add_self_loops:
        np.fill_diagonal(sub, 1.0)

    max_nodes = int(max_nodes)
    feat_dim = int(fraud_graph.num_features)
    adjacency = np.zeros((max_nodes, max_nodes), dtype=np.float32)
    node_features = np.zeros((max_nodes, feat_dim), dtype=np.float32)
    node_mask = np.zeros((max_nodes,), dtype=np.float32)

    adjacency[:n, :n] = sub
    node_features[:n, :] = fraud_graph.features[sel]
    node_mask[:n] = 1.0

    return {
        "adjacency": adjacency,
        "node_features": node_features,
        "node_mask": node_mask,
        "target_index": 0,
        "target_label": int(fraud_graph.labels[int(target_node)]),
    }


# ---------------------------------------------------------------------
# synthetic fallback
# ---------------------------------------------------------------------

def make_synthetic_fraud_graph(
        n_nodes: int = 2000,
        num_features: int = 25,
        anomaly_ratio: float = 0.1,
        avg_degree: int = 4,
        anomaly_feature_shift: float = 2.0,
        seed: Optional[int] = None,
) -> FraudGraph:
    """Build a network-free synthetic node-anomaly graph.

    Generates a Barabasi-Albert-style scale-free graph (preferential attachment,
    pure numpy, no ``networkx`` in the hot path) and injects anomalies that are
    **learnable** from features alone: anomalous nodes get a mean-shifted feature
    distribution (so even a trivial logistic regression separates the classes
    above chance) and extra intra-anomaly edges (atypical connectivity). Used by
    CI/tests and ``build_fraud_subgraph_dataset(synthetic=True)``.

    Args:
        n_nodes: Number of nodes ``N``.
        num_features: Feature dimension ``F``.
        anomaly_ratio: Target fraction of anomalous (label-1) nodes.
        avg_degree: Approximate attachment degree ``m`` for preferential
            attachment (edges added per new node).
        anomaly_feature_shift: Per-dimension mean shift applied to anomalous
            node features (larger => easier to separate).
        seed: Seed for the ``numpy`` generator (reproducible).

    Returns:
        A :class:`FraudGraph` named ``"synthetic"``.
    """
    rng = np.random.default_rng(seed)
    n = int(n_nodes)
    m = max(1, int(avg_degree))

    # --- Barabasi-Albert preferential attachment (numpy only) ----------
    # `repeated` is the standard BA endpoint pool: sampling from it selects a
    # node with probability proportional to its current degree.
    m0 = m + 1
    rows: List[int] = []
    cols: List[int] = []
    repeated: List[int] = []
    # Seed clique among the first m0 nodes.
    for i in range(m0):
        for j in range(i + 1, m0):
            rows.append(i)
            cols.append(j)
            repeated.append(i)
            repeated.append(j)
    repeated_arr = np.array(repeated, dtype=np.int64)
    for new in range(m0, n):
        chosen: set = set()
        # Draw m distinct existing endpoints proportional to degree.
        while len(chosen) < m:
            pick = int(repeated_arr[rng.integers(repeated_arr.shape[0])])
            chosen.add(pick)
        new_endpoints = [new] * (2 * m)
        for t in chosen:
            rows.append(t)
            cols.append(new)
            new_endpoints.append(t)
        repeated_arr = np.concatenate(
            [repeated_arr, np.array(new_endpoints, dtype=np.int64)]
        )

    # --- assign labels --------------------------------------------------
    labels = np.zeros((n,), dtype=np.int64)
    n_anom = int(round(n * float(anomaly_ratio)))
    anom_idx = rng.choice(n, size=n_anom, replace=False)
    labels[anom_idx] = 1

    # --- inject atypical connectivity among anomalies ------------------
    # A handful of extra intra-anomaly edges gives anomalous nodes a distinct
    # neighbourhood signature (not required for the feature-only learnability
    # check, but makes the structural signal non-trivial too).
    if n_anom >= 2:
        extra = min(2 * n_anom, n_anom * (n_anom - 1) // 2)
        for _ in range(extra):
            a, b = rng.choice(anom_idx, size=2, replace=False)
            rows.append(int(a))
            cols.append(int(b))

    data = np.ones((len(rows),), dtype=np.float32)
    adjacency = sp.csr_matrix(
        (data, (np.array(rows), np.array(cols))), shape=(n, n)
    )
    adjacency = _as_sparse_binary_symmetric(adjacency)

    # --- features: mean-shifted for anomalies --------------------------
    features = rng.standard_normal((n, int(num_features))).astype(np.float32)
    features[labels == 1] += np.float32(anomaly_feature_shift)

    graph = _finalize_fraud_graph("synthetic", adjacency, features, labels)
    logger.info(
        f"Synthetic fraud graph: N={graph.num_nodes}, F={graph.num_features}, "
        f"anomaly_ratio={graph.anomaly_ratio:.4f}, "
        f"class_counts={graph.class_counts}"
    )
    return graph


# ---------------------------------------------------------------------
# tf.data builder
# ---------------------------------------------------------------------

def build_fraud_subgraph_dataset(
        name: str = "amazon",
        split: str = "train",
        batch_size: int = 32,
        num_hops: int = 2,
        max_nodes: int = 64,
        cache_root: str = DEFAULT_GRAPH_CACHE,
        add_self_loops: bool = True,
        synthetic: bool = False,
        n_synth_nodes: int = 2000,
        synth_num_features: int = 25,
        synth_anomaly_ratio: float = 0.1,
        seed: Optional[int] = None,
        shuffle: bool = True,
) -> Tuple["Any", Dict[str, Any]]:
    """Build a ``tf.data.Dataset`` of bounded k-hop subgraphs for variant B.

    Loads a fraud graph (real CARE-GNN benchmark, or the synthetic fallback when
    ``synthetic=True``), performs a deterministic **stratified** train/val/test
    node split by label, and yields one bounded subgraph per target node.

    **Input contract (consumed verbatim by the variant-B model/trainer).**
    Each dataset element is a ``(inputs, label)`` tuple where ``inputs`` is a
    dict with these exact keys and shapes::

        inputs = {
            "node_features": (B, N, F) float32,   # per-node features (N = max_nodes)
            "adjacency":     (B, N, N) float32,   # 0/1 symmetric, self-loops as loaded
            "node_mask":     (B, N)    float32,   # 1 = real node, 0 = PAD
            "target_index":  (B,)      int32,      # local index of the target (0)
        }
        label = (B,) int32                          # per-target binary label (0/1)

    There is **no** ``"pe"`` key: Laplacian positional encoding is a variant-C
    concern only. The ``node_features``/``adjacency``/``node_mask`` keys match
    the TUDataset builder's contract so the shared backbone (step 4) consumes
    both.

    Args:
        name: Fraud dataset key (``"amazon"`` or ``"yelpchi"``); ignored when
            ``synthetic=True``.
        split: One of ``"train"``, ``"val"``, ``"test"``.
        batch_size: Target nodes per batch (last batch may be smaller).
        num_hops: BFS depth for :func:`sample_subgraph`.
        max_nodes: Subgraph node cap ``N`` (padded to exactly this).
        cache_root: On-disk cache root for the real dataset.
        add_self_loops: Passed to :func:`sample_subgraph`.
        synthetic: If ``True``, route to :func:`make_synthetic_fraud_graph`
            (network-free) instead of downloading a real benchmark.
        n_synth_nodes: Node count for the synthetic graph.
        synth_num_features: Feature dim for the synthetic graph.
        synth_anomaly_ratio: Anomaly fraction for the synthetic graph.
        seed: Base seed for the split, shuffle order and subgraph sampling.
        shuffle: Reshuffle the split's target order every epoch.

    Returns:
        ``(dataset, metadata)``. ``metadata`` carries ``name``, ``split``,
        ``split_sizes``, ``max_nodes``, ``num_hops``, ``batch_size``,
        ``num_features``, ``anomaly_ratio`` and ``pos_weight`` (= benign /
        anomalous ratio ω, for the trainer's class-weighted BCE).
    """
    import tensorflow as tf

    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be train/val/test, got {split!r}")

    if synthetic:
        graph = make_synthetic_fraud_graph(
            n_nodes=n_synth_nodes,
            num_features=synth_num_features,
            anomaly_ratio=synth_anomaly_ratio,
            seed=seed,
        )
    else:
        graph = load_fraud_graph(name=name, cache_root=cache_root)

    feat_dim = int(graph.num_features)

    # Deterministic stratified node split (reuse the package splitter).
    split_rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = _stratified_split(graph.labels, split_rng)
    split_map = {"train": train_idx, "val": val_idx, "test": test_idx}
    indices = split_map[split]

    # pos_weight ω = benign / anomalous (guard divide-by-zero).
    n_anom = max(int(graph.class_counts.get(1, 0)), 1)
    n_benign = int(graph.class_counts.get(0, 0))
    pos_weight = float(n_benign) / float(n_anom)

    # One persistent generator drives shuffle + subgraph sampling across epochs.
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
            b = len(batch_idx)
            node_features = np.zeros((b, max_nodes, feat_dim), dtype=np.float32)
            adjacency = np.zeros((b, max_nodes, max_nodes), dtype=np.float32)
            node_mask = np.zeros((b, max_nodes), dtype=np.float32)
            target_index = np.zeros((b,), dtype=np.int32)
            labels = np.zeros((b,), dtype=np.int32)
            for i, tgt in enumerate(batch_idx):
                sub = sample_subgraph(
                    graph,
                    int(tgt),
                    num_hops=num_hops,
                    max_nodes=max_nodes,
                    rng=epoch_rng,
                    add_self_loops=add_self_loops,
                )
                node_features[i] = sub["node_features"]
                adjacency[i] = sub["adjacency"]
                node_mask[i] = sub["node_mask"]
                target_index[i] = sub["target_index"]
                labels[i] = sub["target_label"]
            inputs = {
                "node_features": node_features,
                "adjacency": adjacency,
                "node_mask": node_mask,
                "target_index": target_index,
            }
            yield inputs, labels

    output_signature = (
        {
            "node_features": tf.TensorSpec(
                shape=(None, max_nodes, feat_dim), dtype=tf.float32
            ),
            "adjacency": tf.TensorSpec(
                shape=(None, max_nodes, max_nodes), dtype=tf.float32
            ),
            "node_mask": tf.TensorSpec(
                shape=(None, max_nodes), dtype=tf.float32
            ),
            "target_index": tf.TensorSpec(shape=(None,), dtype=tf.int32),
        },
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
    )

    dataset = tf.data.Dataset.from_generator(
        _generator, output_signature=output_signature
    ).prefetch(tf.data.AUTOTUNE)

    metadata: Dict[str, Any] = {
        "name": graph.name,
        "split": split,
        "split_sizes": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "max_nodes": int(max_nodes),
        "num_hops": int(num_hops),
        "batch_size": int(batch_size),
        "num_features": feat_dim,
        "anomaly_ratio": float(graph.anomaly_ratio),
        "pos_weight": pos_weight,
        "class_counts": dict(graph.class_counts),
    }
    return dataset, metadata

# ---------------------------------------------------------------------
