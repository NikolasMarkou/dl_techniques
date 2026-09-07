"""Tests for the configurable multi-paradigm GraphNeuralNetworkLayer."""

import os
import keras
import numpy as np
import pytest
import tensorflow as tf

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


# ---------------------------------------------------------------------
# The stack must run at a hidden width that differs from the input width.
#
# Every test above pins concept_dim == D == 16, so `node_shape` and the
# running block width coincide and the build-time shape contract is
# structurally invisible. These cases separate the two.
# ---------------------------------------------------------------------

MESSAGE_PASSING = ["gcn", "graphsage", "gat", "gin"]
NORMALIZATION = ["none", "layer", "rms", "batch"]
# 8 < D, 16 == D (the currently-passing arm), 32 > D.
HIDDEN_WIDTHS = [8, 16, 32]


class TestTheGnnStackRunsAtAHiddenWidth:
    """The layer must run for any ``concept_dim``, not only ``concept_dim == D``."""

    @pytest.mark.parametrize("norm", NORMALIZATION)
    @pytest.mark.parametrize("mp", MESSAGE_PASSING)
    @pytest.mark.parametrize("num_layers", [1, 2])
    @pytest.mark.parametrize("concept_dim", HIDDEN_WIDTHS)
    def test_forward_pass_at_any_hidden_width(
            self, graph_inputs, concept_dim, num_layers, mp, norm
    ):
        nodes, adj = graph_inputs
        layer = GraphNeuralNetworkLayer(
            concept_dim=concept_dim,
            num_layers=num_layers,
            message_passing=mp,
            normalization=norm,
            aggregation="none",
            num_attention_heads=4,
        )
        out = layer((nodes, adj))
        assert tuple(out.shape) == (B, N, concept_dim)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(out)))

    @pytest.mark.parametrize("norm", NORMALIZATION)
    @pytest.mark.parametrize("mp", MESSAGE_PASSING)
    @pytest.mark.parametrize("concept_dim", HIDDEN_WIDTHS)
    def test_gradient_step_at_any_hidden_width(self, graph_inputs, concept_dim, mp, norm):
        nodes, adj = graph_inputs
        layer = GraphNeuralNetworkLayer(
            concept_dim=concept_dim,
            num_layers=2,
            message_passing=mp,
            normalization=norm,
            aggregation="none",
            num_attention_heads=4,
        )
        with tf.GradientTape() as tape:
            out = layer((nodes, adj), training=True)
            loss = keras.ops.mean(keras.ops.square(out))
        grads = tape.gradient(loss, layer.trainable_variables)
        assert len(layer.trainable_variables) > 0
        assert any(g is not None for g in grads)

    @pytest.mark.parametrize("agg,expected_nodes", [
        ("none", N), ("mean", 1), ("max", 1), ("sum", 1), ("attention", N),
    ])
    @pytest.mark.parametrize("mp", MESSAGE_PASSING)
    @pytest.mark.parametrize("concept_dim", HIDDEN_WIDTHS)
    def test_compute_output_shape_agrees_with_call_at_any_hidden_width(
            self, graph_inputs, concept_dim, mp, agg, expected_nodes
    ):
        nodes, adj = graph_inputs
        layer = GraphNeuralNetworkLayer(
            concept_dim=concept_dim,
            num_layers=2,
            message_passing=mp,
            aggregation=agg,
            num_attention_heads=4,
        )
        # UNBUILT: compute_output_shape must answer before any call.
        computed_unbuilt = layer.compute_output_shape([(B, N, D), (B, N, N)])
        assert tuple(computed_unbuilt) == (B, expected_nodes, concept_dim)

        out = layer((nodes, adj))
        assert tuple(out.shape) == tuple(computed_unbuilt)
        assert tuple(out.shape) == tuple(
            layer.compute_output_shape([nodes.shape, adj.shape])
        )
