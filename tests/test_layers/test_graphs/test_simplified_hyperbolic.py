"""Tests for the simplified Hyperbolic GCN layer (sHGCN)."""

import os
import keras
import numpy as np
import pytest

from dl_techniques.layers.graphs.simplified_hyperbolic_graph_convolutional_neural_layer import (
    SHGCNLayer,
)

B, N, D, U = 2, 5, 8, 6


@pytest.fixture
def graph_inputs():
    rng = np.random.default_rng(2)
    feats = rng.standard_normal((B, N, D)).astype("float32")
    adj = (rng.uniform(size=(B, N, N)) > 0.5).astype("float32")
    return feats, adj


class TestSHGCNLayer:

    def test_construction(self):
        layer = SHGCNLayer(units=U)
        assert layer.units == U

    def test_invalid_units(self):
        with pytest.raises(ValueError):
            SHGCNLayer(units=0)

    def test_invalid_dropout(self):
        with pytest.raises(ValueError):
            SHGCNLayer(units=U, dropout_rate=1.0)

    @pytest.mark.parametrize("use_curvature", [True, False])
    def test_forward_pass(self, graph_inputs, use_curvature):
        feats, adj = graph_inputs
        layer = SHGCNLayer(units=U, use_curvature=use_curvature)
        out = layer([feats, adj])
        assert tuple(out.shape) == (B, N, U)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(out)))

    def test_compute_output_shape(self):
        layer = SHGCNLayer(units=U)
        assert layer.compute_output_shape([(B, N, D), (B, N, N)]) == (B, N, U)

    def test_compute_output_shape_matches_call(self, graph_inputs):
        feats, adj = graph_inputs
        layer = SHGCNLayer(units=U)
        out = layer([feats, adj])
        assert tuple(out.shape) == tuple(layer.compute_output_shape([feats.shape, adj.shape]))

    def test_serialization_round_trip(self, graph_inputs, tmp_path):
        feats, adj = graph_inputs
        feat_in = keras.Input(shape=(N, D), name="feats")
        adj_in = keras.Input(shape=(N, N), name="adj")
        out = SHGCNLayer(units=U, name="shgcn")([feat_in, adj_in])
        model = keras.Model([feat_in, adj_in], out)
        y0 = model([feats, adj])

        path = os.path.join(tmp_path, "shgcn.keras")
        model.save(path)
        loaded = keras.models.load_model(
            path, custom_objects={"SHGCNLayer": SHGCNLayer}
        )
        y1 = loaded([feats, adj])
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y0),
            keras.ops.convert_to_numpy(y1),
            rtol=1e-6, atol=1e-6,
        )

    def test_get_config_round_trip(self):
        layer = SHGCNLayer(units=U, activation="gelu", use_bias=False)
        config = layer.get_config()
        rebuilt = SHGCNLayer.from_config(config)
        assert rebuilt.units == U
        assert rebuilt.use_bias is False
