"""Tests for LinearUpInitializer (THERA heat-field uniform-on-disk frequency init)."""

import os
import tempfile

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.initializers import LinearUpInitializer

PI = np.pi


class TestLinearUpInitializer:
    def test_output_shape(self):
        """SC1: a (2, N) request yields a (2, N) tensor (x/y rows)."""
        w = ops.convert_to_numpy(LinearUpInitializer(scale=1.0, seed=0)((2, 64)))
        assert w.shape == (2, 64)

    def test_leading_dims(self):
        """Leading dims are supported: (..., 2, N) -> (..., 2, N)."""
        w = ops.convert_to_numpy(LinearUpInitializer(scale=1.0, seed=0)((3, 2, 16)))
        assert w.shape == (3, 2, 16)

    def test_norm_distribution(self):
        """SC1: r = sqrt(x^2+y^2) is uniform-on-disk: r^2/(pi*scale)^2 ~ U(0,1)."""
        scale = 2.0
        n = 100000
        w = ops.convert_to_numpy(LinearUpInitializer(scale=scale, seed=42)((2, n)))
        x, y = w[0], w[1]
        r = np.sqrt(x ** 2 + y ** 2)
        radius = PI * scale

        # max radius bounded by pi*scale (uniform-on-disk).
        assert r.max() <= radius * (1.0 + 1e-4)

        # r^2 / radius^2 ~ Uniform(0,1) -> mean ~ 0.5.
        normalized = (r ** 2) / (radius ** 2)
        assert 0.45 <= normalized.mean() <= 0.55

    def test_angle_isotropy(self):
        """Angles span the full circle: x and y are both ~zero-mean (isotropic)."""
        w = ops.convert_to_numpy(LinearUpInitializer(scale=1.0, seed=7)((2, 100000)))
        radius = PI * 1.0
        assert abs(w[0].mean()) < 0.05 * radius
        assert abs(w[1].mean()) < 0.05 * radius

    def test_reproducible_with_seed(self):
        a = ops.convert_to_numpy(LinearUpInitializer(scale=1.0, seed=11)((2, 32)))
        b = ops.convert_to_numpy(LinearUpInitializer(scale=1.0, seed=11)((2, 32)))
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        a = ops.convert_to_numpy(LinearUpInitializer(scale=1.0, seed=1)((2, 32)))
        b = ops.convert_to_numpy(LinearUpInitializer(scale=1.0, seed=2)((2, 32)))
        assert not np.allclose(a, b)

    def test_invalid_scale(self):
        with pytest.raises(ValueError):
            LinearUpInitializer(scale=-1.0)

    def test_invalid_shape_second_axis(self):
        with pytest.raises(ValueError):
            LinearUpInitializer(scale=1.0)((3, 8))

    def test_invalid_rank(self):
        with pytest.raises(ValueError):
            LinearUpInitializer(scale=1.0)((8,))

    def test_get_config_roundtrip(self):
        init = LinearUpInitializer(scale=1.5, seed=9)
        cfg = init.get_config()
        restored = LinearUpInitializer.from_config(cfg)
        assert restored.scale == 1.5
        assert restored.seed == 9
        # same seed -> identical draws.
        np.testing.assert_allclose(
            ops.convert_to_numpy(init((2, 16))),
            ops.convert_to_numpy(restored((2, 16))),
            atol=1e-7,
        )

    def test_keras_serialize_deserialize(self):
        init = LinearUpInitializer(scale=3.0, seed=3)
        restored = keras.saving.deserialize_keras_object(
            keras.saving.serialize_keras_object(init)
        )
        assert isinstance(restored, LinearUpInitializer)
        assert restored.scale == 3.0
        assert restored.seed == 3
        np.testing.assert_allclose(
            ops.convert_to_numpy(init((2, 16))),
            ops.convert_to_numpy(restored((2, 16))),
            atol=1e-7,
        )

    def test_use_in_layer_save_load(self):
        """A custom layer holding a (2, N) weight initialized by LinearUp
        round-trips through .keras save/load (the initializer is meant for
        (2, N) frequency matrices, not (fan_in, units) Dense kernels)."""

        @keras.saving.register_keras_serializable()
        class FreqLayer(keras.layers.Layer):
            def __init__(self, n=8, **kwargs):
                super().__init__(**kwargs)
                self.n = n

            def build(self, input_shape):
                self.freq = self.add_weight(
                    name="freq",
                    shape=(2, self.n),
                    initializer=LinearUpInitializer(scale=1.0, seed=2),
                    trainable=True,
                )
                super().build(input_shape)

            def call(self, inputs):
                return inputs + ops.sum(self.freq)

            def get_config(self):
                cfg = super().get_config()
                cfg.update({"n": self.n})
                return cfg

        inputs = keras.Input(shape=(4,))
        out = FreqLayer(n=8)(inputs)
        model = keras.Model(inputs, out)
        x = np.random.randn(3, 4).astype("float32")
        before = ops.convert_to_numpy(model(x))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
            after = ops.convert_to_numpy(loaded(x))
        np.testing.assert_allclose(before, after, rtol=1e-6, atol=1e-7)
