"""Tests for PolarWeightNorm layer and the recursive polar transform."""

import os
import tempfile

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import ops

from dl_techniques.layers.norms.polar_weight_norm import (
    PolarWeightNorm,
    polar_encode,
    polar_decode,
    _next_power_of_two,
    _level_sizes,
)


class TestPolarTransform:
    """The differentiable Cartesian<->polar transform (paper Def 1 / Alg 1)."""

    @pytest.mark.parametrize("d", [2, 4, 8, 16, 64])
    def test_roundtrip(self, d):
        np.random.seed(d)
        x = np.random.randn(11, d).astype("float32")
        radius, angles = polar_encode(ops.convert_to_tensor(x))
        recon = ops.convert_to_numpy(polar_decode(radius, angles))
        np.testing.assert_allclose(recon, x, rtol=1e-5, atol=1e-5)

    def test_radius_equals_norm(self):
        np.random.seed(0)
        x = np.random.randn(6, 16).astype("float32")
        radius, _ = polar_encode(ops.convert_to_tensor(x))
        np.testing.assert_allclose(
            ops.convert_to_numpy(radius)[:, 0],
            np.linalg.norm(x, axis=1),
            rtol=1e-5,
            atol=1e-5,
        )

    @pytest.mark.parametrize("d", [2, 4, 8, 16])
    def test_angle_dim_is_d_minus_one(self, d):
        _, angles = polar_encode(
            ops.convert_to_tensor(np.random.randn(3, d).astype("float32"))
        )
        assert ops.convert_to_numpy(angles).shape[1] == d - 1

    def test_next_power_of_two(self):
        assert [_next_power_of_two(n) for n in [1, 2, 3, 5, 8, 17, 48]] == [
            1, 2, 4, 8, 8, 32, 64,
        ]

    def test_level_sizes(self):
        assert _level_sizes(8) == [4, 2, 1]
        assert sum(_level_sizes(64)) == 63

    def test_encode_rejects_non_power_of_two(self):
        with pytest.raises(ValueError):
            polar_encode(ops.convert_to_tensor(np.random.randn(2, 3).astype("float32")))


class TestPolarWeightNorm:
    @pytest.fixture
    def x2d(self) -> np.ndarray:
        np.random.seed(42)
        return np.random.randn(4, 64).astype("float32")

    def test_build_and_forward(self, x2d):
        layer = PolarWeightNorm(32)
        y = ops.convert_to_numpy(layer(x2d))
        assert y.shape == (4, 32)
        assert layer.built

    @pytest.mark.parametrize("fan_in,units", [(64, 32), (48, 16), (8, 4), (16, 16)])
    def test_exact_norm_at_init(self, fan_in, units):
        """SC2: ||kernel[:, j]|| == |radius[j]| (pow2 and non-pow2 fan_in)."""
        np.random.seed(fan_in)
        x = np.random.randn(3, fan_in).astype("float32")
        layer = PolarWeightNorm(units, use_bias=False)
        _ = layer(x)
        kernel = ops.convert_to_numpy(layer._reconstruct_kernel())
        colnorm = np.linalg.norm(kernel, axis=0)
        radius = np.abs(ops.convert_to_numpy(layer.radius))
        np.testing.assert_allclose(colnorm, radius, rtol=1e-5, atol=1e-5)

    def test_exact_norm_after_optimizer_step(self):
        """SC2: exactness survives a gradient update (non-pow2 fan_in)."""
        fan_in, units = 48, 12
        np.random.seed(1)
        x = np.random.randn(20, fan_in).astype("float32")
        t = np.random.randn(20, units).astype("float32")
        inp = keras.Input(shape=(fan_in,))
        out = PolarWeightNorm(units)(inp)
        model = keras.Model(inp, out)
        model.compile(optimizer=keras.optimizers.SGD(0.1), loss="mse")
        model.fit(x, t, epochs=1, verbose=0)
        layer = model.layers[1]
        kernel = ops.convert_to_numpy(layer._reconstruct_kernel())
        colnorm = np.linalg.norm(kernel, axis=0)
        radius = np.abs(ops.convert_to_numpy(layer.radius))
        np.testing.assert_allclose(colnorm, radius, rtol=1e-5, atol=1e-5)

    def test_init_reproduces_seed_kernel(self):
        """SC: a freshly built layer reconstructs the seed kernel exactly."""
        fan_in, units = 64, 8
        np.random.seed(3)
        w0 = np.random.randn(fan_in, units).astype("float32")
        seed_init = lambda shape, dtype=None: ops.convert_to_tensor(
            w0, dtype=dtype or "float32"
        )
        layer = PolarWeightNorm(units, use_bias=False, kernel_initializer=seed_init)
        _ = layer(np.random.randn(3, fan_in).astype("float32"))
        kernel = ops.convert_to_numpy(layer._reconstruct_kernel())
        np.testing.assert_allclose(kernel, w0, rtol=1e-4, atol=1e-5)

    def test_3d_input_and_compute_output_shape(self):
        layer = PolarWeightNorm(10)
        x = np.random.randn(2, 7, 16).astype("float32")
        y = ops.convert_to_numpy(layer(x))
        assert y.shape == (2, 7, 10)
        assert layer.compute_output_shape((None, 7, 16)) == (None, 7, 10)

    def test_no_bias(self):
        layer = PolarWeightNorm(8, use_bias=False)
        _ = layer(np.random.randn(2, 16).astype("float32"))
        assert layer.bias is None

    def test_invalid_args(self):
        with pytest.raises(ValueError):
            PolarWeightNorm(0)
        with pytest.raises(ValueError):
            PolarWeightNorm(8, epsilon=-1.0)

    def test_gradient_flow(self):
        layer = PolarWeightNorm(8)
        x = tf.convert_to_tensor(np.random.randn(4, 16).astype("float32"))
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.square(layer(x)))
        grads = tape.gradient(loss, layer.trainable_variables)
        assert all(g is not None for g in grads)
        names = [v.name for v in layer.trainable_variables]
        assert any("radius" in n for n in names)
        assert any("angles" in n for n in names)

    def test_get_config_keys_and_from_config(self):
        layer = PolarWeightNorm(
            32, activation="relu", use_bias=False, epsilon=1e-10,
            radius_regularizer=keras.regularizers.L2(1e-4),
        )
        cfg = layer.get_config()
        for k in [
            "units", "activation", "use_bias", "kernel_initializer",
            "bias_initializer", "radius_regularizer", "angle_regularizer",
            "bias_regularizer", "epsilon",
        ]:
            assert k in cfg
        restored = PolarWeightNorm.from_config(cfg)
        assert restored.units == 32
        assert restored.use_bias is False
        assert restored.epsilon == 1e-10

    def test_save_load_keras_format(self, x2d):
        """SC4: full .keras round-trip reproduces outputs."""
        inputs = keras.Input(shape=x2d.shape[1:])
        outputs = PolarWeightNorm(32, activation="relu")(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)
        original = ops.convert_to_numpy(model(x2d, training=False))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "pwn.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
            reloaded = ops.convert_to_numpy(loaded(x2d, training=False))
        np.testing.assert_allclose(original, reloaded, rtol=1e-6, atol=1e-7)

    def test_save_load_weights(self, x2d):
        inputs = keras.Input(shape=x2d.shape[1:])
        outputs = PolarWeightNorm(16)(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)
        before = ops.convert_to_numpy(model(x2d, training=False))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "pwn.weights.h5")
            model.save_weights(path)
            model.load_weights(path)
            after = ops.convert_to_numpy(model(x2d, training=False))
        np.testing.assert_allclose(before, after, rtol=1e-6, atol=1e-7)


class TestPolarWeightNormIntegration:
    def test_small_net_fit(self):
        """SC5: a 2-layer PolarWeightNorm MLP fits with decreasing finite loss."""
        np.random.seed(0)
        x = np.random.randn(200, 32).astype("float32")
        w = np.random.randn(32, 1).astype("float32")
        y = (x @ w + 0.1 * np.random.randn(200, 1)).astype("float32")

        inp = keras.Input(shape=(32,))
        h = PolarWeightNorm(16, activation="relu")(inp)
        out = PolarWeightNorm(1)(h)
        model = keras.Model(inp, out)
        model.compile(optimizer=keras.optimizers.Adam(1e-2), loss="mse")
        hist = model.fit(x, y, epochs=3, batch_size=32, verbose=0)

        losses = hist.history["loss"]
        assert np.all(np.isfinite(losses))
        assert losses[-1] < losses[0]
