"""Tests for the RFFKernelLayer (Random Fourier Features)."""

import os
import keras
import numpy as np
import pytest

from dl_techniques.layers.random_fourier_features import RFFKernelLayer

B, D, OUT = 3, 8, 4


@pytest.fixture
def sample():
    return np.random.default_rng(0).standard_normal((B, D)).astype("float32")


class TestRFFKernelLayer:

    def test_construction(self):
        layer = RFFKernelLayer(input_dim=D, output_dim=OUT, n_features=64)
        assert layer.output_dim == OUT

    @pytest.mark.parametrize("bad", [
        {"input_dim": 0},
        {"input_dim": D, "n_features": 0},
        {"input_dim": D, "gamma": 0.0},
        {"input_dim": D, "output_dim": -1},
    ])
    def test_invalid_args_raise(self, bad):
        with pytest.raises(ValueError):
            RFFKernelLayer(**bad)

    def test_forward_pass(self, sample):
        out = RFFKernelLayer(input_dim=D, output_dim=OUT, n_features=64)(sample)
        assert tuple(out.shape) == (B, OUT)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(out)))

    def test_default_output_dim(self, sample):
        out = RFFKernelLayer(input_dim=D, n_features=32)(sample)
        assert tuple(out.shape) == (B, D)

    def test_compute_output_shape(self):
        assert RFFKernelLayer(input_dim=D, output_dim=OUT).compute_output_shape((B, D)) == (B, OUT)

    def test_serialization_round_trip(self, sample, tmp_path):
        inp = keras.Input(shape=(D,))
        out = RFFKernelLayer(input_dim=D, output_dim=OUT, n_features=64, name="rff")(inp)
        model = keras.Model(inp, out)
        y0 = model(sample)
        path = os.path.join(tmp_path, "rff.keras")
        model.save(path)
        loaded = keras.models.load_model(
            path, custom_objects={"RFFKernelLayer": RFFKernelLayer}
        )
        y1 = loaded(sample)
        # omega/b are frozen non-trainable weights, restored from the checkpoint.
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y0), keras.ops.convert_to_numpy(y1),
            rtol=1e-5, atol=1e-5,
        )

    def test_get_config_round_trip(self):
        layer = RFFKernelLayer(input_dim=D, output_dim=OUT, gamma=2.0, activation="relu")
        rebuilt = RFFKernelLayer.from_config(layer.get_config())
        assert rebuilt.gamma == 2.0 and rebuilt.output_dim == OUT
