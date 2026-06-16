"""Tests for the PatchMerging (Swin) downsampling layer."""

import os
import keras
import numpy as np
import pytest

from dl_techniques.layers.patch_merging import PatchMerging

B, H, W, C = 2, 8, 8, 6


@pytest.fixture
def sample():
    return np.random.default_rng(0).standard_normal((B, H, W, C)).astype("float32")


class TestPatchMerging:

    def test_invalid_dim(self):
        with pytest.raises(ValueError):
            PatchMerging(dim=0)

    def test_forward_pass(self, sample):
        out = PatchMerging(dim=C)(sample)
        assert tuple(out.shape) == (B, H // 2, W // 2, 2 * C)

    def test_forward_odd_dims(self):
        x = np.random.default_rng(1).standard_normal((B, 7, 7, C)).astype("float32")
        out = PatchMerging(dim=C)(x)
        assert tuple(out.shape) == (B, 4, 4, 2 * C)

    def test_compute_output_shape(self):
        assert PatchMerging(dim=C).compute_output_shape((B, H, W, C)) == (B, 4, 4, 2 * C)

    def test_compute_output_shape_matches_call(self, sample):
        layer = PatchMerging(dim=C)
        out = layer(sample)
        assert tuple(out.shape) == tuple(layer.compute_output_shape(sample.shape))

    def test_serialization_round_trip(self, sample, tmp_path):
        inp = keras.Input(shape=(H, W, C))
        out = PatchMerging(dim=C, name="pm")(inp)
        model = keras.Model(inp, out)
        y0 = model(sample)
        path = os.path.join(tmp_path, "pm.keras")
        model.save(path)
        loaded = keras.models.load_model(
            path, custom_objects={"PatchMerging": PatchMerging}
        )
        y1 = loaded(sample)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y0), keras.ops.convert_to_numpy(y1),
            rtol=1e-5, atol=1e-5,
        )

    def test_get_config_round_trip(self):
        layer = PatchMerging(dim=C, use_bias=True)
        rebuilt = PatchMerging.from_config(layer.get_config())
        assert rebuilt.dim == C and rebuilt.use_bias is True
