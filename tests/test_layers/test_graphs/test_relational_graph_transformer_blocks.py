"""Tests for the RELGT building blocks.

Covers ``LightweightGNNLayer``, ``RELGTTokenEncoder`` and
``RELGTTransformerBlock``: construction (incl. ``ValueError`` paths), forward
pass, ``compute_output_shape`` agreement, and ``.keras`` round-trips.
"""

import os
import keras
import numpy as np
import pytest

from dl_techniques.layers.graphs.relational_graph_transformer_blocks import (
    LightweightGNNLayer,
    RELGTTokenEncoder,
    RELGTTransformerBlock,
)

B, N, D, U = 2, 5, 8, 6


# ---------------------------------------------------------------------
# LightweightGNNLayer
# ---------------------------------------------------------------------

class TestLightweightGNNLayer:

    @pytest.fixture
    def inputs(self):
        rng = np.random.default_rng(3)
        feats = rng.standard_normal((B, N, D)).astype("float32")
        adj = (rng.uniform(size=(B, N, N)) > 0.5).astype("float32")
        return feats, adj

    def test_invalid_units(self):
        with pytest.raises(ValueError):
            LightweightGNNLayer(units=0)

    def test_forward_pass(self, inputs):
        feats, adj = inputs
        layer = LightweightGNNLayer(units=U)
        out = layer([feats, adj])
        assert tuple(out.shape) == (B, N, U)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(out)))

    def test_compute_output_shape(self):
        layer = LightweightGNNLayer(units=U)
        assert layer.compute_output_shape([(B, N, D), (B, N, N)]) == (B, N, U)

    def test_serialization_round_trip(self, inputs, tmp_path):
        feats, adj = inputs
        feat_in = keras.Input(shape=(N, D), name="feats")
        adj_in = keras.Input(shape=(N, N), name="adj")
        out = LightweightGNNLayer(units=U, name="gnn")([feat_in, adj_in])
        model = keras.Model([feat_in, adj_in], out)
        y0 = model([feats, adj])
        path = os.path.join(tmp_path, "lgnn.keras")
        model.save(path)
        loaded = keras.models.load_model(
            path, custom_objects={"LightweightGNNLayer": LightweightGNNLayer}
        )
        y1 = loaded([feats, adj])
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y0), keras.ops.convert_to_numpy(y1),
            rtol=1e-6, atol=1e-6,
        )


# ---------------------------------------------------------------------
# RELGTTokenEncoder
# ---------------------------------------------------------------------

K, F = 5, 10
EMB, NTYPES, MAXHOPS, PEDIM = 16, 4, 2, 8


def _token_inputs():
    rng = np.random.default_rng(4)
    return {
        "node_features": rng.standard_normal((B, K, F)).astype("float32"),
        "node_types": rng.integers(0, NTYPES, size=(B, K)).astype("int32"),
        "hop_distances": rng.integers(0, MAXHOPS + 1, size=(B, K)).astype("int32"),
        "relative_times": rng.standard_normal((B, K, 1)).astype("float32"),
        "subgraph_adjacency": (rng.uniform(size=(B, K, K)) > 0.5).astype("float32"),
    }


def _token_input_shapes():
    return {
        "node_features": (B, K, F),
        "node_types": (B, K),
        "hop_distances": (B, K),
        "relative_times": (B, K, 1),
        "subgraph_adjacency": (B, K, K),
    }


class TestRELGTTokenEncoder:

    def test_invalid_args(self):
        with pytest.raises(ValueError):
            RELGTTokenEncoder(embedding_dim=0, num_node_types=NTYPES)
        with pytest.raises(ValueError):
            RELGTTokenEncoder(embedding_dim=EMB, num_node_types=0)

    def test_forward_pass(self):
        layer = RELGTTokenEncoder(
            embedding_dim=EMB, num_node_types=NTYPES, max_hops=MAXHOPS,
            gnn_pe_dim=PEDIM, gnn_pe_layers=2,
        )
        out = layer(_token_inputs())
        assert tuple(out.shape) == (B, K, EMB)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(out)))

    def test_compute_output_shape(self):
        layer = RELGTTokenEncoder(
            embedding_dim=EMB, num_node_types=NTYPES, max_hops=MAXHOPS,
            gnn_pe_dim=PEDIM, gnn_pe_layers=2,
        )
        assert layer.compute_output_shape(_token_input_shapes()) == (B, K, EMB)

    def test_serialization_round_trip(self, tmp_path):
        inputs = {
            k: keras.Input(shape=v[1:], dtype=("int32" if "type" in k or "hop" in k else "float32"), name=k)
            for k, v in _token_input_shapes().items()
        }
        out = RELGTTokenEncoder(
            embedding_dim=EMB, num_node_types=NTYPES, max_hops=MAXHOPS,
            gnn_pe_dim=PEDIM, gnn_pe_layers=2, name="tok",
        )(inputs)
        model = keras.Model(inputs, out)
        data = _token_inputs()
        y0 = model(data)
        path = os.path.join(tmp_path, "tok.keras")
        model.save(path)
        loaded = keras.models.load_model(
            path, custom_objects={"RELGTTokenEncoder": RELGTTokenEncoder}
        )
        y1 = loaded(data)
        # GNN PE uses random features internally; only shape stability is asserted.
        assert tuple(y0.shape) == tuple(y1.shape) == (B, K, EMB)


# ---------------------------------------------------------------------
# RELGTTransformerBlock
# ---------------------------------------------------------------------

class TestRELGTTransformerBlock:

    @pytest.fixture
    def inputs(self):
        rng = np.random.default_rng(5)
        local = rng.standard_normal((B, K, EMB)).astype("float32")
        seed = rng.standard_normal((B, 1, EMB)).astype("float32")
        return local, seed

    def test_invalid_args(self):
        with pytest.raises(ValueError):
            RELGTTransformerBlock(embedding_dim=EMB, num_heads=3, num_global_centroids=3, ffn_dim=32)

    def test_forward_pass(self, inputs):
        local, seed = inputs
        layer = RELGTTransformerBlock(
            embedding_dim=EMB, num_heads=4, num_global_centroids=3, ffn_dim=32,
        )
        out = layer([local, seed])
        assert tuple(out.shape) == (B, EMB)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(out)))

    def test_compute_output_shape(self):
        layer = RELGTTransformerBlock(
            embedding_dim=EMB, num_heads=4, num_global_centroids=3, ffn_dim=32,
        )
        assert layer.compute_output_shape([(B, K, EMB), (B, 1, EMB)]) == (B, EMB)

    def test_serialization_round_trip(self, inputs, tmp_path):
        local, seed = inputs
        local_in = keras.Input(shape=(K, EMB), name="local")
        seed_in = keras.Input(shape=(1, EMB), name="seed")
        out = RELGTTransformerBlock(
            embedding_dim=EMB, num_heads=4, num_global_centroids=3, ffn_dim=32,
            name="block",
        )([local_in, seed_in])
        model = keras.Model([local_in, seed_in], out)
        y0 = model([local, seed])
        path = os.path.join(tmp_path, "block.keras")
        model.save(path)
        loaded = keras.models.load_model(
            path, custom_objects={"RELGTTransformerBlock": RELGTTransformerBlock}
        )
        y1 = loaded([local, seed])
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y0), keras.ops.convert_to_numpy(y1),
            rtol=1e-6, atol=1e-6,
        )
