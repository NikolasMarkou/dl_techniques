"""Tests for the RestrictedBoltzmannMachine layer."""

import os
import keras
import numpy as np
import pytest

from dl_techniques.layers.restricted_boltzmann_machine import RestrictedBoltzmannMachine

B, NV, NH = 4, 10, 5


@pytest.fixture
def sample():
    return np.random.default_rng(0).uniform(0, 1, size=(B, NV)).astype("float32")


class TestRestrictedBoltzmannMachine:

    def test_construction(self):
        layer = RestrictedBoltzmannMachine(n_hidden=NH)
        assert layer.n_hidden == NH

    def test_forward_pass(self, sample):
        out = RestrictedBoltzmannMachine(n_hidden=NH)(sample)
        assert tuple(out.shape) == (B, NH)
        probs = keras.ops.convert_to_numpy(out)
        assert np.all((probs >= 0.0) & (probs <= 1.0))  # sigmoid probabilities

    def test_compute_output_shape(self):
        assert RestrictedBoltzmannMachine(n_hidden=NH).compute_output_shape((B, NV)) == (B, NH)

    def test_compute_output_shape_matches_call(self, sample):
        layer = RestrictedBoltzmannMachine(n_hidden=NH)
        out = layer(sample)
        assert tuple(out.shape) == tuple(layer.compute_output_shape(sample.shape))

    def test_serialization_round_trip(self, sample, tmp_path):
        inp = keras.Input(shape=(NV,))
        out = RestrictedBoltzmannMachine(n_hidden=NH, name="rbm")(inp)
        model = keras.Model(inp, out)
        y0 = model(sample)
        path = os.path.join(tmp_path, "rbm.keras")
        model.save(path)
        loaded = keras.models.load_model(
            path, custom_objects={"RestrictedBoltzmannMachine": RestrictedBoltzmannMachine}
        )
        y1 = loaded(sample)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y0), keras.ops.convert_to_numpy(y1),
            rtol=1e-5, atol=1e-5,
        )

    def test_get_config_round_trip(self):
        layer = RestrictedBoltzmannMachine(n_hidden=NH, use_bias=False, n_gibbs_steps=3)
        rebuilt = RestrictedBoltzmannMachine.from_config(layer.get_config())
        assert rebuilt.n_hidden == NH and rebuilt.use_bias is False
