"""Tests for PolarInitializer (exact per-vector norm, uniform direction)."""

import os
import tempfile

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.initializers import PolarInitializer


class TestPolarInitializer:
    @pytest.mark.parametrize(
        "shape,axis",
        [((64, 32), 0), ((48, 16), 0), ((10, 20), 1), ((7, 5, 8), 2)],
    )
    def test_exact_norm(self, shape, axis):
        """SC6: every vector along `axis` has L2 norm exactly `norm`."""
        init = PolarInitializer(norm=2.5, axis=axis, seed=1)
        w = ops.convert_to_numpy(init(shape))
        norms = np.sqrt(np.sum(np.square(w), axis=axis))
        np.testing.assert_allclose(norms, 2.5, rtol=1e-5, atol=1e-5)

    def test_reproducible_with_seed(self):
        a = ops.convert_to_numpy(PolarInitializer(norm=1.0, seed=7)((32, 16)))
        b = ops.convert_to_numpy(PolarInitializer(norm=1.0, seed=7)((32, 16)))
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        a = ops.convert_to_numpy(PolarInitializer(seed=1)((32, 16)))
        b = ops.convert_to_numpy(PolarInitializer(seed=2)((32, 16)))
        assert not np.allclose(a, b)

    def test_auto_norm_matches_he_energy(self):
        w = ops.convert_to_numpy(PolarInitializer(seed=0)((200, 8)))
        norms = np.linalg.norm(w, axis=0)
        np.testing.assert_allclose(norms, np.sqrt(2.0), rtol=1e-5)

    def test_gain(self):
        w = ops.convert_to_numpy(PolarInitializer(norm=1.0, gain=3.0, seed=0)((32, 8)))
        np.testing.assert_allclose(np.linalg.norm(w, axis=0), 3.0, rtol=1e-5)

    def test_invalid_norm(self):
        with pytest.raises(ValueError):
            PolarInitializer(norm=-1.0)

    def test_invalid_axis(self):
        with pytest.raises(ValueError):
            PolarInitializer(axis=5)((4, 4))

    def test_serialization_roundtrip(self):
        init = PolarInitializer(norm=1.5, axis=0, gain=2.0, seed=9)
        cfg = init.get_config()
        restored = PolarInitializer.from_config(cfg)
        assert restored.norm == 1.5
        assert restored.axis == 0
        assert restored.gain == 2.0
        assert restored.seed == 9
        np.testing.assert_allclose(
            ops.convert_to_numpy(init((16, 8))),
            ops.convert_to_numpy(restored((16, 8))),
            atol=1e-7,
        )

    def test_keras_serialize_deserialize(self):
        init = PolarInitializer(norm=1.0, seed=3)
        restored = keras.initializers.deserialize(keras.initializers.serialize(init))
        assert isinstance(restored, PolarInitializer)
        np.testing.assert_allclose(
            ops.convert_to_numpy(init((8, 8))),
            ops.convert_to_numpy(restored((8, 8))),
            atol=1e-7,
        )

    def test_use_in_dense_save_load(self):
        inputs = keras.Input(shape=(16,))
        out = keras.layers.Dense(
            8, kernel_initializer=PolarInitializer(norm=1.0, seed=2)
        )(inputs)
        model = keras.Model(inputs, out)
        x = np.random.randn(3, 16).astype("float32")
        before = ops.convert_to_numpy(model(x))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
            after = ops.convert_to_numpy(loaded(x))
        np.testing.assert_allclose(before, after, rtol=1e-6, atol=1e-7)
