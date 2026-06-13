"""
Tests for the VSGD optimizer implementation.
"""

import math
import tempfile
import os
import numpy as np
import pytest
import keras
from keras import ops

# Top-level import so @register_keras_serializable fires before any load_model call.
from dl_techniques.optimization.vsgd_optimizer import VSGD


class TestVSGDInstantiation:
    """Tests for optimizer instantiation and configuration."""

    def test_default_instantiation(self):
        """Test optimizer creates with correct default parameters and derived constants."""
        optimizer = VSGD()

        assert optimizer._ghattg == 30.0
        assert optimizer._ps == 1e-8
        assert optimizer._tau1 == 0.81
        assert optimizer._tau2 == 0.90
        assert optimizer.learning_rate == 0.1
        assert optimizer._weight_decay == 0.0
        assert optimizer._eps == 1e-8

        # Derived constants
        # _pa2 = 2*1e-8 + 1.0 + 1e-4
        expected_pa2 = 2.0 * 1e-8 + 1.0 + 1e-4
        assert abs(optimizer._pa2 - expected_pa2) < 1e-15

        # _pbg2 = 2*1e-8
        expected_pbg2 = 2.0 * 1e-8
        assert abs(optimizer._pbg2 - expected_pbg2) < 1e-20

        # _pbhg2 = 2*30.0*1e-8
        expected_pbhg2 = 2.0 * 30.0 * 1e-8
        assert abs(optimizer._pbhg2 - expected_pbhg2) < 1e-15

    def test_custom_instantiation(self):
        """Test optimizer creates with non-default values that round-trip through attrs."""
        optimizer = VSGD(
            ghattg=50.0,
            ps=1e-6,
            tau1=0.75,
            tau2=0.85,
            learning_rate=0.01,
            weight_decay=0.001,
            eps=1e-7,
        )

        assert optimizer._ghattg == 50.0
        assert optimizer._ps == 1e-6
        assert optimizer._tau1 == 0.75
        assert optimizer._tau2 == 0.85
        assert optimizer.learning_rate == 0.01
        assert optimizer._weight_decay == 0.001
        assert optimizer._eps == 1e-7

    def test_learning_rate_schedule(self):
        """Test optimizer accepts a Keras learning rate schedule."""
        schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.1,
            decay_steps=1000,
            decay_rate=0.9,
        )
        optimizer = VSGD(learning_rate=schedule)

        assert isinstance(
            optimizer._learning_rate,
            keras.optimizers.schedules.LearningRateSchedule,
        )


class TestVSGDBuild:
    """Tests for optimizer build process."""

    def test_build_creates_state_variables(self):
        """Test build allocates _mug, _bg, _bhg for each variable."""
        optimizer = VSGD()

        var1 = keras.Variable(np.ones((8, 16), dtype=np.float32))
        var2 = keras.Variable(np.ones((16,), dtype=np.float32))

        optimizer.build([var1, var2])

        assert optimizer.built is True
        assert len(optimizer._mug) == 2
        assert len(optimizer._bg) == 2
        assert len(optimizer._bhg) == 2

    def test_build_idempotent(self):
        """Test calling build twice does not duplicate state."""
        optimizer = VSGD()

        var1 = keras.Variable(np.ones((4, 8), dtype=np.float32))
        optimizer.build([var1])

        count_after_first = len(optimizer._mug)
        optimizer.build([var1])

        assert len(optimizer._mug) == count_after_first

    def test_variable_index_mapping(self):
        """Test _get_variable_index returns correct index for each variable."""
        optimizer = VSGD()

        var1 = keras.Variable(np.ones((4, 8), dtype=np.float32))
        var2 = keras.Variable(np.ones((8, 16), dtype=np.float32))
        var3 = keras.Variable(np.ones((32,), dtype=np.float32))

        optimizer.build([var1, var2, var3])

        assert optimizer._get_variable_index(var1) == 0
        assert optimizer._get_variable_index(var2) == 1
        assert optimizer._get_variable_index(var3) == 2


class TestVSGDUpdate:
    """Tests for the actual optimization updates."""

    def test_update_step_modifies_variable(self):
        """Test a single update step changes the variable value."""
        optimizer = VSGD(learning_rate=0.1, weight_decay=0.0)

        var = keras.Variable(np.ones((4, 8), dtype=np.float32))
        initial_value = keras.ops.convert_to_numpy(var).copy()

        grad = ops.convert_to_tensor(
            np.random.randn(4, 8).astype(np.float32)
        )

        optimizer.build([var])
        optimizer.update_step(grad, var, ops.convert_to_tensor(0.1, dtype="float32"))

        final_value = keras.ops.convert_to_numpy(var)
        assert not np.allclose(initial_value, final_value)

    def test_weight_decay_reduces_magnitude(self):
        """Test that weight_decay reduces the variable norm when gradient is zero."""
        # lr is non-zero but gradient is zero; weight decay still fires via:
        #   variable *= (1 - lr * weight_decay)
        optimizer = VSGD(learning_rate=0.1, weight_decay=0.5, eps=1e-8)

        var = keras.Variable(np.ones((4,), dtype=np.float32))
        initial_norm = float(ops.norm(var))

        grad = ops.zeros((4,), dtype="float32")

        optimizer.build([var])
        optimizer.update_step(grad, var, ops.convert_to_tensor(0.1, dtype="float32"))

        final_norm = float(ops.norm(var))
        assert final_norm < initial_norm

    def test_step1_branch_fires(self):
        """Test that _mug, _bg, _bhg become non-zero after first step with non-zero gradient."""
        optimizer = VSGD(learning_rate=0.01, weight_decay=0.0)

        var = keras.Variable(np.ones((4,), dtype=np.float32))
        grad = ops.convert_to_tensor(np.ones((4,), dtype=np.float32))

        optimizer.build([var])
        optimizer.update_step(grad, var, ops.convert_to_tensor(0.01, dtype="float32"))

        mug_vals = keras.ops.convert_to_numpy(optimizer._mug[0])
        bg_vals = keras.ops.convert_to_numpy(optimizer._bg[0])
        bhg_vals = keras.ops.convert_to_numpy(optimizer._bhg[0])

        assert np.any(mug_vals != 0.0), "_mug should be non-zero after first step"
        assert np.any(bg_vals != 0.0), "_bg should be non-zero after first step"
        assert np.any(bhg_vals != 0.0), "_bhg should be non-zero after first step"


class TestVSGDSerialization:
    """Tests for optimizer serialization and deserialization."""

    def test_get_config_complete(self):
        """Test get_config returns all 7 custom hyperparameters."""
        optimizer = VSGD(
            ghattg=50.0,
            ps=1e-6,
            tau1=0.75,
            tau2=0.85,
            learning_rate=0.01,
            weight_decay=0.001,
            eps=1e-7,
        )

        config = optimizer.get_config()

        assert "ghattg" in config
        assert "ps" in config
        assert "tau1" in config
        assert "tau2" in config
        assert "learning_rate" in config
        assert "weight_decay" in config
        assert "eps" in config

        assert abs(config["ghattg"] - 50.0) < 1e-10
        assert abs(config["ps"] - 1e-6) < 1e-15
        assert abs(config["tau1"] - 0.75) < 1e-10
        assert abs(config["tau2"] - 0.85) < 1e-10
        assert abs(config["weight_decay"] - 0.001) < 1e-10
        assert abs(config["eps"] - 1e-7) < 1e-15

    def test_from_config_roundtrip(self):
        """Test optimizer can be reconstructed from config with identical hyperparams."""
        original = VSGD(
            ghattg=50.0,
            ps=1e-6,
            tau1=0.75,
            tau2=0.85,
            learning_rate=0.01,
            weight_decay=0.001,
            eps=1e-7,
        )

        config = original.get_config()
        restored = VSGD.from_config(config)

        assert abs(restored._ghattg - original._ghattg) < 1e-10
        assert abs(restored._ps - original._ps) < 1e-15
        assert abs(restored._tau1 - original._tau1) < 1e-10
        assert abs(restored._tau2 - original._tau2) < 1e-10
        assert abs(restored._weight_decay - original._weight_decay) < 1e-10
        assert abs(restored._eps - original._eps) < 1e-15

    def test_keras_serialization(self):
        """Test optimizer serializes through Keras saving utilities."""
        optimizer = VSGD(ghattg=40.0, ps=1e-7, learning_rate=0.05)

        serialized = keras.saving.serialize_keras_object(optimizer)
        restored = keras.saving.deserialize_keras_object(serialized)

        assert isinstance(restored, VSGD)
        assert abs(restored._ghattg - 40.0) < 1e-10
        assert abs(restored._ps - 1e-7) < 1e-15


class TestVSGDIntegration:
    """Integration tests with Keras models."""

    def test_simple_model_training(self):
        """Test VSGD works with a simple Sequential model for 2 epochs."""
        model = keras.Sequential([
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(10),
        ])

        optimizer = VSGD(learning_rate=0.01)
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
        )

        x = np.random.randn(100, 16).astype(np.float32)
        y = np.random.randint(0, 10, size=(100,))

        history = model.fit(x, y, epochs=2, batch_size=32, verbose=0)

        assert len(history.history["loss"]) == 2

    def test_model_save_load(self):
        """Test model with VSGD optimizer can be saved and loaded."""
        model = keras.Sequential([
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(4),
        ])

        optimizer = VSGD(learning_rate=0.01, weight_decay=0.0)
        model.compile(optimizer=optimizer, loss="mse")

        x = np.random.randn(20, 8).astype(np.float32)
        y = np.random.randn(20, 4).astype(np.float32)
        model.fit(x, y, epochs=1, verbose=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "vsgd_model.keras")
            model.save(path)
            loaded_model = keras.models.load_model(path)

        assert isinstance(loaded_model.optimizer, VSGD)

        pred_original = model.predict(x, verbose=0)
        pred_loaded = loaded_model.predict(x, verbose=0)

        np.testing.assert_allclose(
            pred_original, pred_loaded,
            atol=1e-5,
            err_msg="Predictions should match after save/load",
        )


class TestVSGDEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_gradient(self):
        """Test zero gradient does not produce NaN in variable after one step."""
        optimizer = VSGD(learning_rate=0.01)

        var = keras.Variable(np.ones((8,), dtype=np.float32))
        grad = ops.zeros((8,), dtype="float32")

        optimizer.build([var])
        optimizer.update_step(grad, var, ops.convert_to_tensor(0.01, dtype="float32"))

        result = keras.ops.convert_to_numpy(var)
        assert not np.any(np.isnan(result)), "NaN detected after zero-gradient step"
        assert not np.any(np.isinf(result)), "Inf detected after zero-gradient step"

    def test_extreme_hyperparams(self):
        """Test extreme hyperparams (large ghattg, tiny ps) do not produce NaN/Inf."""
        optimizer = VSGD(ghattg=1000.0, ps=1e-12, learning_rate=0.01)

        var = keras.Variable(np.ones((8,), dtype=np.float32))
        grad = ops.convert_to_tensor(np.random.randn(8).astype(np.float32))

        optimizer.build([var])
        optimizer.update_step(grad, var, ops.convert_to_tensor(0.01, dtype="float32"))

        result = keras.ops.convert_to_numpy(var)
        assert not np.any(np.isnan(result)), "NaN detected with extreme hyperparams"
        assert not np.any(np.isinf(result)), "Inf detected with extreme hyperparams"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
