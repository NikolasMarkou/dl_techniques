"""Tests for the trainable CLAHE layer.

CLAHE is a TensorFlow-backend-only layer (``tf.histogram_fixed_width`` +
``@tf.function``) with a per-image ``call`` contract — it operates on a single
``(H, W, 1)`` image rather than a batched tensor. The serialization round-trip is
therefore exercised via ``get_config``/``from_config`` plus an explicit weight
transfer (rather than a batched functional ``.keras`` model).
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.clahe import CLAHE

HH, WW = 32, 32
N_BINS = 256


@pytest.fixture
def sample_image():
    # CLAHE expects pixel values in [0, 255] (histogram value_range).
    rng = np.random.default_rng(0)
    return (rng.uniform(0, 255, size=(HH, WW, 1))).astype("float32")


class TestCLAHE:

    def test_construction(self):
        layer = CLAHE()
        assert layer.n_bins == 256
        assert layer.tile_size == 16

    @pytest.mark.parametrize("bad", [
        {"clip_limit": 0.0},
        {"n_bins": 1},
        {"tile_size": 0},
    ])
    def test_invalid_args_raise(self, bad):
        with pytest.raises(ValueError):
            CLAHE(**bad)

    def test_build_invalid_shape_raises(self):
        layer = CLAHE()
        with pytest.raises(ValueError):
            layer.build((HH, WW, 3))  # last dim must be 1

    def test_forward_pass(self, sample_image):
        layer = CLAHE(n_bins=N_BINS, tile_size=16)
        out = layer(keras.ops.convert_to_tensor(sample_image))
        assert tuple(out.shape) == (HH, WW, 1)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(out)))

    def test_compute_output_shape(self):
        layer = CLAHE()
        assert layer.compute_output_shape((HH, WW, 1)) == (HH, WW, 1)

    def test_serialization_round_trip(self, sample_image):
        """Config + weight round-trip yields identical output (per-image contract)."""
        x = keras.ops.convert_to_tensor(sample_image)
        layer = CLAHE(n_bins=N_BINS, tile_size=16, clip_limit=3.0)
        y0 = keras.ops.convert_to_numpy(layer(x))  # builds the layer

        rebuilt = CLAHE.from_config(layer.get_config())
        rebuilt.build((HH, WW, 1))
        rebuilt.set_weights(layer.get_weights())
        y1 = keras.ops.convert_to_numpy(rebuilt(x))

        np.testing.assert_allclose(y0, y1, rtol=1e-5, atol=1e-5)

    def test_get_config(self):
        layer = CLAHE(clip_limit=2.5, n_bins=128, tile_size=8)
        config = layer.get_config()
        assert config["clip_limit"] == 2.5
        assert config["n_bins"] == 128
        assert config["tile_size"] == 8
