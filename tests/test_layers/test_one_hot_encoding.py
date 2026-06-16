"""Tests for the OneHotEncoding layer."""

import os
import keras
import numpy as np
import pytest

from dl_techniques.layers.one_hot_encoding import OneHotEncoding

B = 4
CARDS = [3, 4, 2]
TOTAL = sum(CARDS)


@pytest.fixture
def cat_input():
    rng = np.random.default_rng(0)
    cols = [rng.integers(0, c, size=(B,)) for c in CARDS]
    return np.stack(cols, axis=1).astype("int32")


class TestOneHotEncoding:

    def test_construction(self):
        layer = OneHotEncoding(cardinalities=CARDS)
        assert layer.total_dim == TOTAL

    def test_invalid_cardinalities(self):
        with pytest.raises(ValueError):
            OneHotEncoding(cardinalities=[3, 0, 2])

    def test_empty_cardinalities_allowed(self):
        layer = OneHotEncoding(cardinalities=[])
        assert layer.total_dim == 0

    def test_forward_pass(self, cat_input):
        out = OneHotEncoding(cardinalities=CARDS)(cat_input)
        assert tuple(out.shape) == (B, TOTAL)
        # Each feature contributes exactly one hot bit.
        assert np.allclose(keras.ops.convert_to_numpy(out).sum(axis=1), len(CARDS))

    def test_compute_output_shape(self):
        assert OneHotEncoding(cardinalities=CARDS).compute_output_shape((B, 3)) == (B, TOTAL)

    def test_serialization_round_trip(self, cat_input, tmp_path):
        inp = keras.Input(shape=(3,), dtype="int32")
        out = OneHotEncoding(cardinalities=CARDS, name="ohe")(inp)
        model = keras.Model(inp, out)
        y0 = model(cat_input)
        path = os.path.join(tmp_path, "ohe.keras")
        model.save(path)
        loaded = keras.models.load_model(
            path, custom_objects={"OneHotEncoding": OneHotEncoding}
        )
        y1 = loaded(cat_input)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y0), keras.ops.convert_to_numpy(y1), atol=1e-6
        )
