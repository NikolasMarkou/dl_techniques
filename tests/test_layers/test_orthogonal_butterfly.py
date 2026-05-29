"""Tests for OrthogonalButterfly: exactly-orthogonal butterfly Givens layer."""

import os
import tempfile

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import ops

from dl_techniques.layers.orthogonal_butterfly import OrthogonalButterfly


def _random_angle_layer(num_blocks, seed):
    return OrthogonalButterfly(
        num_blocks=num_blocks,
        angle_initializer=keras.initializers.RandomUniform(-3.14, 3.14, seed=seed),
    )


class TestOrthogonalButterfly:
    @pytest.mark.parametrize("d", [2, 4, 8, 16, 64])
    @pytest.mark.parametrize("num_blocks", [1, 3])
    def test_norm_preservation(self, d, num_blocks):
        """SC1: ||layer(x)|| == ||x|| per row for arbitrary angles."""
        np.random.seed(d + num_blocks)
        x = np.random.randn(5, d).astype("float32")
        layer = _random_angle_layer(num_blocks, seed=d + num_blocks)
        y = ops.convert_to_numpy(layer(x))
        np.testing.assert_allclose(
            np.linalg.norm(y, axis=1), np.linalg.norm(x, axis=1), rtol=1e-5, atol=1e-5
        )

    @pytest.mark.parametrize("d", [2, 4, 8, 16])
    @pytest.mark.parametrize("num_blocks", [1, 2])
    def test_materialized_matrix_orthogonal(self, d, num_blocks):
        """SC2: the materialized transform W satisfies WᵀW == I."""
        layer = _random_angle_layer(num_blocks, seed=7 * d + num_blocks)
        # Rows of layer(I) are images of the basis vectors => that array is Wᵀ.
        wt = ops.convert_to_numpy(layer(np.eye(d, dtype="float32")))
        gram = wt @ wt.T
        np.testing.assert_allclose(gram, np.eye(d), atol=1e-5)

    def test_identity_at_zero_init(self):
        """SC3: zeros angle init => identity transform."""
        layer = OrthogonalButterfly()  # default angle_initializer='zeros'
        x = np.random.randn(4, 16).astype("float32")
        np.testing.assert_allclose(ops.convert_to_numpy(layer(x)), x, atol=1e-6)

    def test_non_power_of_two_raises(self):
        with pytest.raises(ValueError):
            OrthogonalButterfly()(np.random.randn(2, 6).astype("float32"))

    def test_invalid_num_blocks(self):
        with pytest.raises(ValueError):
            OrthogonalButterfly(num_blocks=0)

    def test_2d_and_3d_shapes(self):
        layer = OrthogonalButterfly(num_blocks=2)
        x2d = np.random.randn(3, 8).astype("float32")
        x3d = np.random.randn(3, 5, 8).astype("float32")
        assert ops.convert_to_numpy(layer(x2d)).shape == (3, 8)
        assert ops.convert_to_numpy(layer(x3d)).shape == (3, 5, 8)
        assert layer.compute_output_shape((None, 5, 8)) == (None, 5, 8)

    def test_3d_norm_preserved_per_vector(self):
        layer = _random_angle_layer(2, seed=11)
        x = np.random.randn(2, 4, 16).astype("float32")
        y = ops.convert_to_numpy(layer(x))
        np.testing.assert_allclose(
            np.linalg.norm(y, axis=-1), np.linalg.norm(x, axis=-1), rtol=1e-5, atol=1e-5
        )

    def test_bias(self):
        layer = OrthogonalButterfly(use_bias=True)
        _ = layer(np.random.randn(2, 8).astype("float32"))
        assert layer.bias is not None
        layer2 = OrthogonalButterfly(use_bias=False)
        _ = layer2(np.random.randn(2, 8).astype("float32"))
        assert layer2.bias is None

    def test_gradient_flow(self):
        """SC5: gradients reach the angle parameters."""
        layer = _random_angle_layer(2, seed=1)
        x = tf.convert_to_tensor(np.random.randn(4, 16).astype("float32"))
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(layer(x) * tf.constant(np.random.randn(4, 16).astype("float32")))
        grads = tape.gradient(loss, layer.trainable_variables)
        assert all(g is not None for g in grads)
        assert any("angles" in v.name for v in layer.trainable_variables)

    def test_get_config_and_from_config(self):
        layer = OrthogonalButterfly(
            num_blocks=3, use_bias=True,
            angle_regularizer=keras.regularizers.L2(1e-5),
        )
        cfg = layer.get_config()
        for k in [
            "num_blocks", "use_bias", "angle_initializer", "angle_regularizer",
            "bias_initializer", "bias_regularizer",
        ]:
            assert k in cfg
        restored = OrthogonalButterfly.from_config(cfg)
        assert restored.num_blocks == 3
        assert restored.use_bias is True

    def test_save_load_keras_format(self):
        """SC6: full .keras round-trip reproduces outputs (with non-trivial angles)."""
        x = np.random.randn(4, 32).astype("float32")
        inputs = keras.Input(shape=(32,))
        outputs = _random_angle_layer(2, seed=5)(inputs)
        model = keras.Model(inputs, outputs)
        original = ops.convert_to_numpy(model(x, training=False))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ob.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
            reloaded = ops.convert_to_numpy(loaded(x, training=False))
        np.testing.assert_allclose(original, reloaded, rtol=1e-6, atol=1e-7)

    def test_save_load_weights(self):
        x = np.random.randn(4, 16).astype("float32")
        inputs = keras.Input(shape=(16,))
        outputs = _random_angle_layer(2, seed=9)(inputs)
        model = keras.Model(inputs, outputs)
        before = ops.convert_to_numpy(model(x, training=False))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ob.weights.h5")
            model.save_weights(path)
            model.load_weights(path)
            after = ops.convert_to_numpy(model(x, training=False))
        np.testing.assert_allclose(before, after, rtol=1e-6, atol=1e-7)


class TestOrthogonalButterflyInverse:
    @pytest.mark.parametrize("d", [2, 8, 16, 64])
    @pytest.mark.parametrize("num_blocks", [1, 3])
    @pytest.mark.parametrize("use_bias", [False, True])
    def test_inverse_roundtrip(self, d, num_blocks, use_bias):
        """inverse(forward(x)) == x and forward(inverse(x)) == x."""
        layer = OrthogonalButterfly(
            num_blocks=num_blocks, use_bias=use_bias,
            angle_initializer=keras.initializers.RandomUniform(-3.14, 3.14, seed=d + num_blocks),
            bias_initializer=keras.initializers.RandomNormal(seed=1),
        )
        x = np.random.randn(6, d).astype("float32")
        y = layer(x, inverse=False)
        x_rec = ops.convert_to_numpy(layer(y, inverse=True))
        np.testing.assert_allclose(x_rec, x, rtol=1e-5, atol=1e-5)
        # other direction
        z = ops.convert_to_numpy(layer(layer(x, inverse=True), inverse=False))
        np.testing.assert_allclose(z, x, rtol=1e-5, atol=1e-5)

    def test_inverse_alias(self):
        layer = _random_angle_layer(2, seed=4)
        x = np.random.randn(4, 16).astype("float32")
        via_alias = ops.convert_to_numpy(layer.inverse(layer(x)))
        np.testing.assert_allclose(via_alias, x, rtol=1e-5, atol=1e-5)

    def test_inverse_3d(self):
        layer = _random_angle_layer(2, seed=6)
        x = np.random.randn(2, 5, 16).astype("float32")
        x_rec = ops.convert_to_numpy(layer(layer(x), inverse=True))
        np.testing.assert_allclose(x_rec, x, rtol=1e-5, atol=1e-5)

    def test_log_det_jacobian_is_zero(self):
        layer = _random_angle_layer(3, seed=2)
        x = np.random.randn(7, 64).astype("float32")
        ldj = ops.convert_to_numpy(layer.log_det_jacobian(x))
        assert ldj.shape == (7,)
        np.testing.assert_allclose(ldj, 0.0, atol=1e-12)

    def test_inverse_after_save_load(self):
        x = np.random.randn(4, 16).astype("float32")
        inputs = keras.Input(shape=(16,))
        layer = _random_angle_layer(2, seed=8)
        model = keras.Model(inputs, layer(inputs))
        y = ops.convert_to_numpy(model(x))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ob.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
        # the loaded layer's inverse must undo the saved forward output
        loaded_layer = loaded.layers[1]
        x_rec = ops.convert_to_numpy(loaded_layer(y, inverse=True))
        np.testing.assert_allclose(x_rec, x, rtol=1e-5, atol=1e-5)


class TestOrthogonalButterflyIntegration:
    def test_small_net_fit(self):
        """SC5: a net using OrthogonalButterfly trains (finite, decreasing loss)."""
        np.random.seed(0)
        x = np.random.randn(200, 16).astype("float32")
        # Target: a fixed orthogonal-ish rotation of the input (learnable by butterfly).
        rng = np.random.default_rng(0)
        q, _ = np.linalg.qr(rng.standard_normal((16, 16)))
        y = (x @ q.astype("float32"))

        inp = keras.Input(shape=(16,))
        out = OrthogonalButterfly(num_blocks=4)(inp)
        model = keras.Model(inp, out)
        model.compile(optimizer=keras.optimizers.Adam(1e-2), loss="mse")
        hist = model.fit(x, y, epochs=3, batch_size=32, verbose=0)
        losses = hist.history["loss"]
        assert np.all(np.isfinite(losses))
        assert losses[-1] < losses[0]
