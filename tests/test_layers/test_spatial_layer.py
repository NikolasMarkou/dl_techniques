"""Tests for the SpatialLayer (CoordConv coordinate generator)."""

import os
import keras
import numpy as np
import pytest

from dl_techniques.layers.spatial_layer import SpatialLayer

B, H, W, C = 2, 8, 8, 3


@pytest.fixture
def sample():
    return np.random.default_rng(0).standard_normal((B, H, W, C)).astype("float32")


class TestSpatialLayer:

    def test_construction(self):
        layer = SpatialLayer(resolution=(4, 4))
        assert layer.resolution == (4, 4)

    @pytest.mark.parametrize("bad", [
        {"resolution": (4,)},
        {"resolution": (0, 4)},
        {"resize_method": "bogus"},
    ])
    def test_invalid_args_raise(self, bad):
        with pytest.raises(ValueError):
            SpatialLayer(**bad)

    def test_forward_pass(self, sample):
        out = SpatialLayer(resolution=(4, 4))(sample)
        assert tuple(out.shape) == (B, H, W, 2)

    def test_build_wrong_rank_raises(self):
        with pytest.raises(ValueError):
            SpatialLayer().build((B, H))

    def test_compute_output_shape(self):
        assert SpatialLayer().compute_output_shape((B, H, W, C)) == (B, H, W, 2)

    def test_serialization_round_trip(self, sample, tmp_path):
        inp = keras.Input(shape=(H, W, C))
        out = SpatialLayer(resolution=(4, 4), resize_method="bilinear", name="spatial")(inp)
        model = keras.Model(inp, out)
        y0 = model(sample)
        path = os.path.join(tmp_path, "spatial.keras")
        model.save(path)
        loaded = keras.models.load_model(
            path, custom_objects={"SpatialLayer": SpatialLayer}
        )
        y1 = loaded(sample)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y0), keras.ops.convert_to_numpy(y1),
            rtol=1e-5, atol=1e-5,
        )
