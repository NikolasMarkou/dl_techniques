"""Network-free tests for the Graph Energy Transformer dataset loaders.

Every test here is **self-contained and offline**: the TUDataset parser is
exercised on a tiny hand-built text fixture written into ``tmp_path`` (never a
download), the Laplacian PE / sign-flip / collate paths run on numpy graphs
built in-process, and the fraud subgraph sampler uses the network-free
synthetic generator. The ONE optional test that reads the real cached MUTAG is
skip-guarded on the cache directory's presence AND calls the loader with
``download=False``, so it can never reach the network.

Network-freedom is additionally *enforced* by the module-level ``_block_network``
autouse fixture, which replaces ``urllib.request.urlopen`` with a function that
raises: any accidental download attempt fails the suite instead of silently
hitting the wire (CI runs offline).
"""

import os

import numpy as np
import pytest

from dl_techniques.datasets.graphs import (
    GraphSample,
    collate_graph_batch,
    compute_laplacian_pe,
    load_tudataset,
    make_synthetic_fraud_graph,
    sample_subgraph,
    sign_flip_pe,
)
from dl_techniques.datasets.graphs.tudataset import DEFAULT_GRAPH_CACHE

# Published MUTAG statistics (chrsmrrs.github.io/datasets) for the optional
# real-cache check.
_MUTAG_NUM_GRAPHS = 188
_MUTAG_NUM_CLASSES = 2
_MUTAG_AVG_NODES = 17.93


# ---------------------------------------------------------------------
# network kill-switch (proves offline execution)
# ---------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _block_network(monkeypatch):
    """Fail loudly if any test tries to open a URL (guarantees offline)."""

    def _no_network(*_args, **_kwargs):
        raise RuntimeError(
            "network access attempted in a network-free test suite"
        )

    # Both loaders call ``urllib.request.urlopen``; patching the shared module
    # attribute covers tudataset.py and fraud.py at once.
    monkeypatch.setattr("urllib.request.urlopen", _no_network)


# ---------------------------------------------------------------------
# TUDataset text fixture (hand-built, 3 known graphs)
# ---------------------------------------------------------------------

def _write_tudataset_fixture(root, name="FOO"):
    """Write a tiny 3-graph TUDataset-format fixture under ``root/{name}/``.

    Graphs (global 1-indexed node ids, edges listed in ONE direction only so
    the parser's symmetrization is actually exercised):

        * G1: nodes 1,2,3   triangle          edges (1,2)(2,3)(1,3)  -> label -1
        * G2: nodes 4,5     single edge        edge  (4,5)            -> label  1
        * G3: nodes 6,7,8,9 path               edges (6,7)(7,8)(8,9)  -> label -1

    Graph labels {-1, 1} force a 0..C-1 remap. Node labels drawn from the
    vocabulary {0, 2, 7} force a 3-wide one-hot.
    """
    data_dir = os.path.join(str(root), name)
    os.makedirs(data_dir, exist_ok=True)

    def _p(suffix):
        return os.path.join(data_dir, f"{name}_{suffix}.txt")

    # One direction per edge (symmetrization must recover the other).
    with open(_p("A"), "w") as fh:
        fh.write("1, 2\n2, 3\n1, 3\n4, 5\n6, 7\n7, 8\n8, 9\n")

    # node -> graph id (1-indexed graphs).
    with open(_p("graph_indicator"), "w") as fh:
        fh.write("\n".join(["1", "1", "1", "2", "2", "3", "3", "3", "3"]) + "\n")

    # per-graph label (non-contiguous -> exercises remap).
    with open(_p("graph_labels"), "w") as fh:
        fh.write("-1\n1\n-1\n")

    # per-node label from vocab {0, 2, 7} -> one-hot width 3.
    with open(_p("node_labels"), "w") as fh:
        fh.write("\n".join(["0", "2", "0", "7", "2", "0", "2", "7", "0"]) + "\n")

    return data_dir


class TestTUDatasetParser:
    """Parse the hand-built fixture and assert known structure (no download)."""

    @pytest.fixture
    def parsed(self, tmp_path):
        _write_tudataset_fixture(tmp_path, name="FOO")
        # download=False + cache_root=tmp_path => pure local parse, no network.
        return load_tudataset(
            "FOO", cache_root=str(tmp_path), download=False, add_self_loops=True
        )

    def test_graph_count_and_node_counts(self, parsed):
        samples, meta = parsed
        assert meta["num_graphs"] == 3
        assert len(samples) == 3
        assert [s.num_nodes for s in samples] == [3, 2, 4]
        assert [s.adjacency.shape[0] for s in samples] == [3, 2, 4]

    def test_label_remap_contiguous(self, parsed):
        samples, meta = parsed
        assert meta["num_classes"] == 2
        # {-1, 1} -> {0, 1}: G1->0, G2->1, G3->0.
        assert [s.label for s in samples] == [0, 1, 0]
        assert set(meta["label_distribution"].keys()) == {0, 1}

    def test_onehot_node_features(self, parsed):
        samples, meta = parsed
        assert meta["feature_source"] == "node_labels_onehot"
        assert meta["num_node_features"] == 3  # vocab {0, 2, 7}
        for s in samples:
            assert s.node_features.shape == (s.num_nodes, 3)
            # Every real node row is a one-hot vector.
            assert np.all(np.isin(s.node_features, [0.0, 1.0]))
            assert np.allclose(s.node_features.sum(axis=1), 1.0)

    def test_adjacency_symmetric_square_selfloops(self, parsed):
        samples, _ = parsed
        for s in samples:
            a = s.adjacency
            assert a.shape[0] == a.shape[1]          # square
            assert np.array_equal(a, a.T)            # symmetric
            assert np.all(np.diag(a) == 1.0)         # self-loops on
        # Triangle G1 with self-loops is the all-ones 3x3 matrix.
        assert np.array_equal(samples[0].adjacency, np.ones((3, 3), np.float32))

    @pytest.mark.skipif(
        not os.path.isfile(
            os.path.join(DEFAULT_GRAPH_CACHE, "MUTAG", "MUTAG_A.txt")
        ),
        reason="real MUTAG cache absent (offline/CI) — skip published-stats check",
    )
    def test_real_mutag_matches_published_stats(self):
        # download=False: reads ONLY the on-disk cache, never the network.
        samples, meta = load_tudataset(
            "MUTAG", cache_root=DEFAULT_GRAPH_CACHE, download=False
        )
        assert meta["num_graphs"] == _MUTAG_NUM_GRAPHS
        assert meta["num_classes"] == _MUTAG_NUM_CLASSES
        assert len(samples) == _MUTAG_NUM_GRAPHS
        assert abs(meta["avg_nodes"] - _MUTAG_AVG_NODES) < 0.5
        for s in samples:
            assert np.array_equal(s.adjacency, s.adjacency.T)


# ---------------------------------------------------------------------
# Laplacian positional encoding
# ---------------------------------------------------------------------

def _cycle_adjacency(n, extra_chords=0, seed=0):
    """Symmetric 0/1 adjacency of an n-cycle plus optional random chords."""
    a = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        j = (i + 1) % n
        a[i, j] = 1.0
        a[j, i] = 1.0
    if extra_chords:
        rng = np.random.default_rng(seed)
        for _ in range(extra_chords):
            u, v = rng.integers(0, n, size=2)
            if u != v:
                a[u, v] = 1.0
                a[v, u] = 1.0
    return a


class TestLaplacianPE:
    """Shape, orthonormality, zero-padding and sign-flip invariance."""

    K = 15

    def test_shape_and_zero_padding_small_graph(self):
        n = 6  # n <= K forces trailing zero-pad columns.
        pe = compute_laplacian_pe(_cycle_adjacency(n), k=self.K)
        assert pe.shape == (n, self.K)
        assert pe.dtype == np.float32
        take = min(self.K, n - 1)  # = 5 available non-trivial eigenvectors
        # Trailing columns beyond the available eigenvectors are all-zero.
        assert np.all(pe[:, take:] == 0.0)
        assert np.any(pe[:, :take] != 0.0)

    def test_columns_orthonormal_small(self):
        n = 6
        pe = compute_laplacian_pe(_cycle_adjacency(n), k=self.K)
        take = min(self.K, n - 1)
        gram = pe[:, :take].T @ pe[:, :take]
        off = np.max(np.abs(gram - np.eye(take)))
        assert off < 1e-5, f"small-graph PE not orthonormal, max off={off}"

    def test_columns_orthonormal_realistic(self):
        # Realistic size (n=64 >> K): do NOT trust a decomposition only at toy n.
        n = 64
        adj = _cycle_adjacency(n, extra_chords=40, seed=3)
        pe = compute_laplacian_pe(adj, k=self.K)
        assert pe.shape == (n, self.K)
        # All K columns are real non-trivial eigenvectors here (n-1 >> K).
        gram = pe.T @ pe
        off = np.max(np.abs(gram - np.eye(self.K)))
        assert off < 1e-5, f"realistic PE not orthonormal, max off={off}"
        # expose the number for the report
        self._last_off = off

    def test_sign_flip_preserves_magnitude(self):
        n = 64
        pe = compute_laplacian_pe(_cycle_adjacency(n, 40, seed=3), k=self.K)
        rng = np.random.default_rng(11)
        flipped = sign_flip_pe(pe, rng)
        assert flipped.shape == pe.shape
        assert flipped.dtype == np.float32
        # abs() is invariant to per-column sign flips.
        assert np.allclose(np.abs(flipped), np.abs(pe))

    def test_sign_flip_actually_changes_sign(self):
        n = 64
        pe = compute_laplacian_pe(_cycle_adjacency(n, 40, seed=3), k=self.K)
        # Deterministically find a seed that flips at least one column.
        changed = False
        for s in range(8):
            flipped = sign_flip_pe(pe, np.random.default_rng(s))
            if not np.allclose(flipped, pe):
                changed = True
                # magnitude still preserved even when signs change
                assert np.allclose(np.abs(flipped), np.abs(pe))
                break
        assert changed, "sign_flip_pe never changed a sign across 8 seeds"


# ---------------------------------------------------------------------
# Fraud subgraph sampler (variant B, synthetic => network-free)
# ---------------------------------------------------------------------

class TestFraudSubgraphSampler:
    """Bounded k-hop subgraph: cap, symmetry, target@0, PAD masking."""

    @pytest.fixture(scope="class")
    def graph(self):
        return make_synthetic_fraud_graph(
            n_nodes=300,
            num_features=16,
            anomaly_ratio=0.1,
            avg_degree=4,
            seed=0,
        )

    def test_graph_metadata_sensible(self, graph):
        assert 0.0 < graph.anomaly_ratio < 1.0
        assert graph.num_nodes == 300
        assert graph.num_features == 16
        assert graph.class_counts[0] > graph.class_counts[1] > 0

    def test_subgraph_bounds_symmetry_and_padding(self, graph):
        max_nodes = 32
        rng = np.random.default_rng(1)
        for tgt in [0, 5, 17, 123, 299]:
            sub = sample_subgraph(
                graph, tgt, num_hops=2, max_nodes=max_nodes, rng=rng
            )
            adj = sub["adjacency"]
            feats = sub["node_features"]
            mask = sub["node_mask"]

            assert adj.shape == (max_nodes, max_nodes)      # dense, capped
            assert np.array_equal(adj, adj.T)               # symmetric
            assert sub["target_index"] == 0                 # target pinned @ 0
            assert sub["target_label"] in (0, 1)

            n_real = int(mask.sum())
            assert 1 <= n_real <= max_nodes                 # <= cap, target present
            assert np.all((mask == 0.0) | (mask == 1.0))
            assert mask[0] == 1.0                           # index 0 is real

            pad = mask == 0.0
            # PAD rows and columns of the adjacency are all-zero.
            assert np.all(adj[pad, :] == 0.0)
            assert np.all(adj[:, pad] == 0.0)
            # PAD feature rows are all-zero.
            assert np.all(feats[pad] == 0.0)

    def test_builder_metadata_pos_weight(self):
        # tf.data builder on the synthetic graph (network-free).
        from dl_techniques.datasets.graphs import build_fraud_subgraph_dataset

        ds, meta = build_fraud_subgraph_dataset(
            synthetic=True,
            split="train",
            batch_size=4,
            max_nodes=32,
            n_synth_nodes=300,
            synth_num_features=16,
            synth_anomaly_ratio=0.1,
            seed=0,
        )
        assert meta["name"] == "synthetic"
        assert 0.0 < meta["anomaly_ratio"] < 1.0
        # pos_weight = benign/anomalous; imbalance => > 1.
        assert meta["pos_weight"] > 1.0
        assert meta["max_nodes"] == 32

        # One batch honors the dense (B, N, ...) contract with a valid mask.
        (inputs, labels) = next(iter(ds.take(1)))
        adj = inputs["adjacency"].numpy()
        mask = inputs["node_mask"].numpy()
        b = adj.shape[0]
        assert adj.shape == (b, 32, 32)
        assert inputs["node_features"].shape[:2] == (b, 32)
        assert inputs["target_index"].shape == (b,)
        assert labels.shape == (b,)
        for i in range(b):
            assert np.array_equal(adj[i], adj[i].T)
            pad = mask[i] == 0.0
            assert np.all(adj[i][pad, :] == 0.0)
            assert np.all(adj[i][:, pad] == 0.0)


# ---------------------------------------------------------------------
# Dense pad + node-mask collate (variant C path, numpy)
# ---------------------------------------------------------------------

class TestCollateGraphBatch:
    """Variable-size graphs -> uniform (B, N, ...) with a correct node mask."""

    @pytest.fixture
    def samples(self, tmp_path):
        _write_tudataset_fixture(tmp_path, name="FOO")
        s, _ = load_tudataset(
            "FOO", cache_root=str(tmp_path), download=False, add_self_loops=True
        )
        return s  # node counts [3, 2, 4]

    def test_dense_pad_shapes_mask_and_symmetry(self, samples):
        k = 15
        pe_list = [compute_laplacian_pe(s.adjacency, k=k) for s in samples]
        inputs, labels = collate_graph_batch(samples, pe_list, k=k)

        b = len(samples)
        n = max(s.num_nodes for s in samples)  # dynamic pad -> N = 4
        f = samples[0].node_features.shape[1]

        assert inputs["node_features"].shape == (b, n, f)
        assert inputs["adjacency"].shape == (b, n, n)
        assert inputs["pe"].shape == (b, n, k)
        assert inputs["node_mask"].shape == (b, n)
        assert labels.shape == (b,)

        # node_mask row sums == true per-graph node counts.
        true_counts = np.array([s.num_nodes for s in samples], dtype=np.float32)
        assert np.array_equal(inputs["node_mask"].sum(axis=1), true_counts)

        for i, s in enumerate(samples):
            pad = inputs["node_mask"][i] == 0.0
            adj = inputs["adjacency"][i]
            assert np.array_equal(adj, adj.T)            # per-sample symmetric
            assert np.all(adj[pad, :] == 0.0)            # PAD rows zero
            assert np.all(adj[:, pad] == 0.0)            # PAD cols zero
            assert np.all(inputs["node_features"][i][pad] == 0.0)
            assert np.all(inputs["pe"][i][pad] == 0.0)

    def test_fixed_max_nodes_cap(self, samples):
        k = 15
        pe_list = [compute_laplacian_pe(s.adjacency, k=k) for s in samples]
        inputs, _ = collate_graph_batch(samples, pe_list, k=k, max_nodes=8)
        assert inputs["adjacency"].shape == (3, 8, 8)
        assert inputs["node_mask"].shape == (3, 8)
        # Cap does not change the true node counts recorded by the mask.
        assert np.array_equal(
            inputs["node_mask"].sum(axis=1),
            np.array([3, 2, 4], dtype=np.float32),
        )
