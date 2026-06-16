"""Tests for the HierarchicalMLPStem ViT patch-embedding stem."""

import os
import keras
import numpy as np
import pytest

from dl_techniques.layers.hierarchical_mlp_stem import HierarchicalMLPStem

B, IMG, PATCH, CH, EMB = 2, 32, 16, 3, 16
NUM_PATCHES = (IMG // PATCH) ** 2  # 4


@pytest.fixture
def sample():
    return np.random.default_rng(0).standard_normal((B, IMG, IMG, CH)).astype("float32")


class TestHierarchicalMLPStem:

    def _make(self, **kw):
        defaults = dict(embed_dim=EMB, img_size=(IMG, IMG), patch_size=(PATCH, PATCH),
                        in_channels=CH)
        defaults.update(kw)
        return HierarchicalMLPStem(**defaults)

    @pytest.mark.parametrize("bad", [
        {"embed_dim": 7},                       # not divisible by 4
        {"patch_size": (16, 8)},                # unequal
        {"patch_size": (3, 3)},                 # < 4 / not power of two
        {"norm_layer": "bogus"},
        {"in_channels": 0},
    ])
    def test_invalid_args_raise(self, bad):
        with pytest.raises(ValueError):
            self._make(**bad)

    @pytest.mark.parametrize("norm", ["batch", "layer"])
    def test_forward_pass(self, sample, norm):
        out = self._make(norm_layer=norm)(sample)
        assert tuple(out.shape) == (B, NUM_PATCHES, EMB)

    def test_compute_output_shape(self):
        assert self._make().compute_output_shape((B, IMG, IMG, CH)) == (B, NUM_PATCHES, EMB)

    def test_serialization_round_trip(self, sample, tmp_path):
        inp = keras.Input(shape=(IMG, IMG, CH))
        out = self._make(norm_layer="layer", name="stem")(inp)
        model = keras.Model(inp, out)
        y0 = model(sample, training=False)
        path = os.path.join(tmp_path, "stem.keras")
        model.save(path)
        loaded = keras.models.load_model(
            path, custom_objects={"HierarchicalMLPStem": HierarchicalMLPStem}
        )
        y1 = loaded(sample, training=False)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y0), keras.ops.convert_to_numpy(y1),
            rtol=1e-5, atol=1e-5,
        )

    def test_get_config_round_trip(self):
        rebuilt = HierarchicalMLPStem.from_config(self._make().get_config())
        assert rebuilt.embed_dim == EMB and rebuilt.patch_size == (PATCH, PATCH)
