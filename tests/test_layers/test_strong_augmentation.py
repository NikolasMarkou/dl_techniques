"""Tests for the StrongAugmentation layer."""

import os
import keras
import numpy as np
import pytest

from dl_techniques.layers.strong_augmentation import StrongAugmentation

B, H, W, C = 4, 8, 8, 3


@pytest.fixture
def sample():
    return np.random.default_rng(0).uniform(0, 1, size=(B, H, W, C)).astype("float32")


class TestStrongAugmentation:

    def test_construction(self):
        layer = StrongAugmentation(cutmix_prob=0.5)
        assert layer.cutmix_prob == 0.5

    @pytest.mark.parametrize("bad", [
        {"cutmix_prob": 1.5},
        {"color_jitter_strength": -0.1},
        {"cutmix_ratio_range": (0.5, 0.1)},
    ])
    def test_invalid_args_raise(self, bad):
        with pytest.raises(ValueError):
            StrongAugmentation(**bad)

    def test_inference_is_identity(self, sample):
        out = StrongAugmentation()(sample, training=False)
        np.testing.assert_allclose(keras.ops.convert_to_numpy(out), sample, atol=1e-6)

    def test_training_preserves_shape(self, sample):
        out = StrongAugmentation(cutmix_prob=1.0)(sample, training=True)
        assert tuple(out.shape) == (B, H, W, C)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(out)))

    def test_compute_output_shape(self):
        assert StrongAugmentation().compute_output_shape((B, H, W, C)) == (B, H, W, C)

    def test_serialization_round_trip(self, sample, tmp_path):
        inp = keras.Input(shape=(H, W, C))
        out = StrongAugmentation(cutmix_prob=0.5, name="aug")(inp)
        model = keras.Model(inp, out)
        y0 = model(sample, training=False)
        path = os.path.join(tmp_path, "aug.keras")
        model.save(path)
        loaded = keras.models.load_model(
            path, custom_objects={"StrongAugmentation": StrongAugmentation}
        )
        y1 = loaded(sample, training=False)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y0), keras.ops.convert_to_numpy(y1), atol=1e-6
        )

    def test_get_config(self):
        layer = StrongAugmentation(cutmix_prob=0.3, color_jitter_strength=0.4)
        rebuilt = StrongAugmentation.from_config(layer.get_config())
        assert rebuilt.cutmix_prob == 0.3 and rebuilt.color_jitter_strength == 0.4
