"""Tests for the SGLD (Stochastic Gradient Langevin Dynamics) optimizer."""

import os
import tempfile

import numpy as np
import pytest
import keras
from keras import ops

from dl_techniques.optimization.sgld_optimizer import SGLD


class TestSGLDInstantiation:
    """Tests for optimizer instantiation and configuration."""

    def test_default_instantiation(self):
        optimizer = SGLD()
        assert optimizer.learning_rate == 1e-2
        assert optimizer.noise_scale == 1.0
        assert optimizer.seed is None

    def test_custom_instantiation(self):
        optimizer = SGLD(
            learning_rate=5e-3,
            noise_scale=0.5,
            weight_decay=1e-4,
            seed=123,
        )
        assert optimizer.learning_rate == 5e-3
        assert optimizer.noise_scale == 0.5
        assert optimizer.seed == 123

    def test_negative_noise_scale_raises(self):
        with pytest.raises(ValueError, match="noise_scale must be non-negative"):
            SGLD(noise_scale=-0.1)

    def test_learning_rate_schedule(self):
        schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-2,
            decay_steps=1000,
            decay_rate=0.9,
        )
        optimizer = SGLD(learning_rate=schedule)
        assert isinstance(
            optimizer._learning_rate,
            keras.optimizers.schedules.LearningRateSchedule,
        )


class TestSGLDBuild:
    """Tests for optimizer build process."""

    def test_build_idempotent(self):
        optimizer = SGLD(seed=0)
        var = keras.Variable(np.random.randn(4, 8).astype(np.float32))
        optimizer.build([var])
        sg_first = optimizer._seed_generator
        optimizer.build([var])
        # Idempotency: subsequent build() is a no-op (does not replace state).
        assert optimizer._seed_generator is sg_first

    def test_build_creates_seed_generator(self):
        optimizer = SGLD(seed=42)
        var = keras.Variable(np.zeros((3, 3), dtype=np.float32))
        optimizer.build([var])
        assert isinstance(optimizer._seed_generator, keras.random.SeedGenerator)

    def test_build_no_per_variable_buffers(self):
        # SGLD must remain stateless: no momentum / second-moment slots.
        optimizer = SGLD()
        var1 = keras.Variable(np.zeros((4, 4), dtype=np.float32))
        var2 = keras.Variable(np.zeros((4,), dtype=np.float32))
        optimizer.build([var1, var2])
        # No SGLD-specific buffers should have been added.
        assert not hasattr(optimizer, "_sgld_velocities")
        assert not hasattr(optimizer, "_sgld_m")


class TestSGLDUpdate:
    """Tests for the SGLD update rule."""

    def test_update_modifies_variable(self):
        optimizer = SGLD(learning_rate=0.1, noise_scale=1.0, seed=0)
        var = keras.Variable(np.zeros((8, 8), dtype=np.float32))
        initial = ops.convert_to_numpy(var).copy()
        grad = ops.convert_to_tensor(
            np.random.randn(8, 8).astype(np.float32)
        )
        optimizer.build([var])
        optimizer.update_step(grad, var, ops.convert_to_tensor(0.1))
        assert not np.allclose(initial, ops.convert_to_numpy(var))

    def test_zero_lr_no_change(self):
        """With lr=0 the drift is zero and noise std sqrt(2*0)=0 — no update."""
        optimizer = SGLD(learning_rate=0.0, noise_scale=1.0, seed=1)
        var = keras.Variable(np.ones((4, 4), dtype=np.float32))
        initial = ops.convert_to_numpy(var).copy()
        grad = ops.convert_to_tensor(
            np.random.randn(4, 4).astype(np.float32)
        )
        optimizer.build([var])
        optimizer.update_step(grad, var, ops.convert_to_tensor(0.0))
        np.testing.assert_allclose(
            initial, ops.convert_to_numpy(var), atol=1e-7
        )

    def test_noise_only_when_grad_zero(self):
        """With zero gradient, the variable still moves due to Langevin noise."""
        optimizer = SGLD(learning_rate=0.1, noise_scale=1.0, seed=7)
        var = keras.Variable(np.zeros((16, 16), dtype=np.float32))
        initial = ops.convert_to_numpy(var).copy()
        grad = ops.zeros((16, 16), dtype="float32")
        optimizer.build([var])
        optimizer.update_step(grad, var, ops.convert_to_tensor(0.1))
        diff = ops.convert_to_numpy(var) - initial
        # Noise should produce a non-trivial perturbation.
        assert np.std(diff) > 0.0

    def test_zero_noise_scale_is_pure_sgd(self):
        """noise_scale=0 collapses SGLD to SGD: w -= lr * grad exactly."""
        optimizer = SGLD(learning_rate=0.1, noise_scale=0.0, seed=0)
        var = keras.Variable(np.zeros((4, 4), dtype=np.float32))
        grad_np = np.random.randn(4, 4).astype(np.float32)
        grad = ops.convert_to_tensor(grad_np)
        optimizer.build([var])
        optimizer.update_step(grad, var, ops.convert_to_tensor(0.1))
        np.testing.assert_allclose(
            ops.convert_to_numpy(var), -0.1 * grad_np, rtol=1e-6, atol=1e-7
        )

    def test_seed_reproducibility(self):
        """Same seed + same inputs => identical post-update variables."""
        def run_once(seed):
            optimizer = SGLD(learning_rate=0.1, noise_scale=1.0, seed=seed)
            var = keras.Variable(np.zeros((8, 8), dtype=np.float32))
            grad = ops.convert_to_tensor(
                np.zeros((8, 8), dtype=np.float32)
            )
            optimizer.build([var])
            optimizer.update_step(grad, var, ops.convert_to_tensor(0.1))
            return ops.convert_to_numpy(var)

        a = run_once(seed=2026)
        b = run_once(seed=2026)
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)

    def test_different_seeds_differ(self):
        def run_once(seed):
            optimizer = SGLD(learning_rate=0.1, noise_scale=1.0, seed=seed)
            var = keras.Variable(np.zeros((8, 8), dtype=np.float32))
            grad = ops.convert_to_tensor(np.zeros((8, 8), dtype=np.float32))
            optimizer.build([var])
            optimizer.update_step(grad, var, ops.convert_to_tensor(0.1))
            return ops.convert_to_numpy(var)

        a = run_once(seed=1)
        b = run_once(seed=2)
        assert not np.allclose(a, b)


class TestSGLDSerialization:
    """Tests for optimizer serialization and deserialization."""

    def test_get_config_complete(self):
        optimizer = SGLD(
            learning_rate=5e-3,
            noise_scale=0.7,
            weight_decay=1e-4,
            seed=99,
        )
        config = optimizer.get_config()
        assert config["learning_rate"] == pytest.approx(5e-3)
        assert config["noise_scale"] == 0.7
        assert config["weight_decay"] == pytest.approx(1e-4)
        assert config["seed"] == 99

    def test_from_config_roundtrip(self):
        original = SGLD(
            learning_rate=2e-3, noise_scale=0.3, seed=7
        )
        restored = SGLD.from_config(original.get_config())
        assert restored.learning_rate == original.learning_rate
        assert restored.noise_scale == original.noise_scale
        assert restored.seed == original.seed

    def test_keras_serialization(self):
        optimizer = SGLD(learning_rate=1e-3, noise_scale=2.0, seed=42)
        serialized = keras.saving.serialize_keras_object(optimizer)
        restored = keras.saving.deserialize_keras_object(serialized)
        assert isinstance(restored, SGLD)
        assert restored.learning_rate == 1e-3
        assert restored.noise_scale == 2.0
        assert restored.seed == 42


class TestSGLDIntegration:
    """Integration tests against the Keras training loop."""

    def test_simple_model_training(self):
        model = keras.Sequential([
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(10),
        ])
        model.compile(
            optimizer=SGLD(learning_rate=1e-3, noise_scale=1.0, seed=0),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        x = np.random.randn(64, 16).astype(np.float32)
        y = np.random.randint(0, 10, size=(64,))
        history = model.fit(x, y, epochs=2, batch_size=16, verbose=0)
        assert len(history.history["loss"]) == 2

    def test_model_save_load(self):
        model = keras.Sequential([
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(4),
        ])
        model.compile(
            optimizer=SGLD(learning_rate=1e-3, noise_scale=0.5, seed=11),
            loss="mse",
        )
        x = np.random.randn(20, 8).astype(np.float32)
        y = np.random.randn(20, 4).astype(np.float32)
        model.fit(x, y, epochs=1, verbose=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sgld_model.keras")
            model.save(path)
            loaded = keras.models.load_model(path)

        assert isinstance(loaded.optimizer, SGLD)
        assert loaded.optimizer.noise_scale == 0.5
        assert loaded.optimizer.seed == 11

        # Predictions are deterministic (no noise at inference).
        pred_a = model.predict(x, verbose=0)
        pred_b = loaded.predict(x, verbose=0)
        np.testing.assert_allclose(pred_a, pred_b, rtol=1e-5, atol=1e-5)

    def test_gradient_clipping_compatible(self):
        model = keras.Sequential([
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(4),
        ])
        model.compile(
            optimizer=SGLD(learning_rate=1e-3, clipnorm=1.0, seed=0),
            loss="mse",
        )
        x = np.random.randn(16, 8).astype(np.float32)
        y = np.random.randn(16, 4).astype(np.float32)
        model.fit(x, y, epochs=1, batch_size=8, verbose=0)


class TestSGLDEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_variable(self):
        optimizer = SGLD(seed=0)
        var = keras.Variable(np.zeros((3,), dtype=np.float32))
        grad = ops.zeros((3,), dtype="float32")
        optimizer.build([var])
        optimizer.update_step(grad, var, ops.convert_to_tensor(0.1))

    def test_rank1_variable(self):
        optimizer = SGLD(learning_rate=0.1, seed=0)
        var = keras.Variable(np.zeros((8,), dtype=np.float32))
        grad = ops.convert_to_tensor(np.ones((8,), dtype=np.float32))
        optimizer.build([var])
        optimizer.update_step(grad, var, ops.convert_to_tensor(0.1))
        assert not np.any(np.isnan(ops.convert_to_numpy(var)))

    def test_high_rank_variable(self):
        optimizer = SGLD(learning_rate=0.01, seed=0)
        var = keras.Variable(np.zeros((2, 3, 4, 5), dtype=np.float32))
        grad = ops.convert_to_tensor(
            np.random.randn(2, 3, 4, 5).astype(np.float32)
        )
        optimizer.build([var])
        optimizer.update_step(grad, var, ops.convert_to_tensor(0.01))
        assert not np.any(np.isnan(ops.convert_to_numpy(var)))

    def test_no_nans_after_many_steps(self):
        optimizer = SGLD(learning_rate=1e-3, noise_scale=1.0, seed=0)
        var = keras.Variable(np.zeros((8, 8), dtype=np.float32))
        optimizer.build([var])
        for _ in range(50):
            grad = ops.convert_to_tensor(
                np.random.randn(8, 8).astype(np.float32)
            )
            optimizer.update_step(grad, var, ops.convert_to_tensor(1e-3))
        result = ops.convert_to_numpy(var)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
