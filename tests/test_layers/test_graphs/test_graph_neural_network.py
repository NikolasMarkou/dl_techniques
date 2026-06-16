"""Tests for the configurable multi-paradigm GraphNeuralNetworkLayer."""

import os
import keras
import numpy as np
import pytest

from dl_techniques.layers.graphs.graph_neural_network import GraphNeuralNetworkLayer

B, N, D = 2, 5, 16


@pytest.fixture
def graph_inputs():
    rng = np.random.default_rng(1)
    nodes = rng.standard_normal((B, N, D)).astype("float32")
    adj = (rng.uniform(size=(B, N, N)) > 0.5).astype("float32")
    return nodes, adj


class TestGraphNeuralNetworkLayer:

    def test_construction(self):
        layer = GraphNeuralNetworkLayer(concept_dim=D, num_layers=2)
        assert layer.concept_dim == D
        assert layer.num_layers == 2

    @pytest.mark.parametrize("bad", [
        {"concept_dim": 0},
        {"concept_dim": D, "num_layers": 0},
        {"concept_dim": D, "dropout_rate": 1.5},
        {"concept_dim": D, "num_attention_heads": 0},
        {"concept_dim": D, "message_passing": "bogus"},
        {"concept_dim": D, "aggregation": "bogus"},
    ])
    def test_invalid_args_raise(self, bad):
        with pytest.raises(ValueError):
            GraphNeuralNetworkLayer(**bad)

    @pytest.mark.parametrize("mp", ["gcn", "graphsage", "gat", "gin"])
    def test_forward_pass(self, graph_inputs, mp):
        nodes, adj = graph_inputs
        layer = GraphNeuralNetworkLayer(
            concept_dim=D, num_layers=2, message_passing=mp, aggregation="none",
            num_attention_heads=4,
        )
        out = layer((nodes, adj))
        assert tuple(out.shape) == (B, N, D)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(out)))

    @pytest.mark.parametrize("agg,expected_nodes", [
        ("none", N), ("mean", 1), ("max", 1), ("sum", 1), ("attention", N),
    ])
    def test_compute_output_shape(self, agg, expected_nodes):
        layer = GraphNeuralNetworkLayer(
            concept_dim=D, num_layers=1, aggregation=agg, num_attention_heads=4,
        )
        shape = layer.compute_output_shape([(B, N, D), (B, N, N)])
        assert shape == (B, expected_nodes, D)

    def test_compute_output_shape_matches_call(self, graph_inputs):
        nodes, adj = graph_inputs
        layer = GraphNeuralNetworkLayer(
            concept_dim=D, num_layers=2, aggregation="mean", num_attention_heads=4,
        )
        out = layer((nodes, adj))
        computed = layer.compute_output_shape([nodes.shape, adj.shape])
        assert tuple(out.shape) == tuple(computed)

    def test_serialization_round_trip(self, graph_inputs, tmp_path):
        nodes, adj = graph_inputs
        node_in = keras.Input(shape=(N, D), name="nodes")
        adj_in = keras.Input(shape=(N, N), name="adj")
        out = GraphNeuralNetworkLayer(
            concept_dim=D, num_layers=2, message_passing="gcn",
            aggregation="none", num_attention_heads=4, name="gnn",
        )((node_in, adj_in))
        model = keras.Model([node_in, adj_in], out)
        y0 = model([nodes, adj])

        path = os.path.join(tmp_path, "gnn.keras")
        model.save(path)
        loaded = keras.models.load_model(
            path, custom_objects={"GraphNeuralNetworkLayer": GraphNeuralNetworkLayer}
        )
        y1 = loaded([nodes, adj])
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y0),
            keras.ops.convert_to_numpy(y1),
            rtol=1e-6, atol=1e-6,
        )

    def test_get_config_round_trip(self):
        layer = GraphNeuralNetworkLayer(concept_dim=D, num_layers=2, message_passing="gin")
        config = layer.get_config()
        rebuilt = GraphNeuralNetworkLayer.from_config(config)
        assert rebuilt.concept_dim == D
        assert rebuilt.message_passing == "gin"
