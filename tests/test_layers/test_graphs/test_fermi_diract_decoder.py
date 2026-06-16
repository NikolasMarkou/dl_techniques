"""Tests for the Fermi-Dirac decoder layer (link prediction)."""

import os
import keras
import numpy as np
import pytest

from dl_techniques.layers.graphs.fermi_diract_decoder import FermiDiracDecoder

B, D = 4, 8


@pytest.fixture
def sample_inputs():
    rng = np.random.default_rng(0)
    u = rng.standard_normal((B, D)).astype("float32")
    v = rng.standard_normal((B, D)).astype("float32")
    return u, v


class TestFermiDiracDecoder:

    def test_construction(self):
        layer = FermiDiracDecoder()
        assert layer.r_initializer is not None
        assert layer.t_initializer is not None

    def test_mismatched_dims_raises(self):
        layer = FermiDiracDecoder()
        with pytest.raises(ValueError):
            layer.build([(None, 8), (None, 4)])

    def test_invalid_input_structure_raises(self):
        layer = FermiDiracDecoder()
        with pytest.raises(ValueError):
            layer.build([(None, 8), (None, 8), (None, 8)])  # not a 2-element list

    def test_forward_pass(self, sample_inputs):
        u, v = sample_inputs
        layer = FermiDiracDecoder()
        out = layer([u, v])
        assert tuple(out.shape) == (B,)
        probs = keras.ops.convert_to_numpy(out)
        assert np.all((probs >= 0.0) & (probs <= 1.0))

    def test_compute_output_shape(self):
        layer = FermiDiracDecoder()
        assert layer.compute_output_shape([(B, D), (B, D)]) == (B,)

    def test_compute_output_shape_matches_call(self, sample_inputs):
        u, v = sample_inputs
        layer = FermiDiracDecoder()
        out = layer([u, v])
        assert tuple(out.shape) == tuple(layer.compute_output_shape([u.shape, v.shape]))

    def test_serialization_round_trip(self, sample_inputs, tmp_path):
        u, v = sample_inputs
        u_in = keras.Input(shape=(D,), name="u")
        v_in = keras.Input(shape=(D,), name="v")
        out = FermiDiracDecoder(name="fd")([u_in, v_in])
        model = keras.Model([u_in, v_in], out)
        y0 = model([u, v])

        path = os.path.join(tmp_path, "fd.keras")
        model.save(path)
        loaded = keras.models.load_model(
            path, custom_objects={"FermiDiracDecoder": FermiDiracDecoder}
        )
        y1 = loaded([u, v])
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y0),
            keras.ops.convert_to_numpy(y1),
            rtol=1e-6, atol=1e-6,
        )

    def test_get_config(self):
        layer = FermiDiracDecoder()
        config = layer.get_config()
        assert "r_initializer" in config
        assert "t_initializer" in config
        FermiDiracDecoder.from_config(config)
