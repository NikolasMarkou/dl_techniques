"""
Tests for the Muon optimizer implementation.
"""

import numpy as np
import pytest
import keras
from keras import ops

from dl_techniques.optimization.muon_optimizer import Muon


class TestMuonInstantiation:
    """Tests for optimizer instantiation and configuration."""

    def test_default_instantiation(self):
        """Test optimizer creates with default parameters."""
        optimizer = Muon()

        assert optimizer.learning_rate == 0.02
        assert optimizer.momentum == 0.95
        assert optimizer.nesterov is True
        assert optimizer.ns_steps == 5
        assert optimizer.adam_learning_rate == 1e-3
        assert optimizer.adam_beta_1 == 0.9
        assert optimizer.adam_beta_2 == 0.999
        assert optimizer.adam_epsilon == 1e-7
        assert optimizer._weight_decay == 0.0

    def test_custom_instantiation(self):
        """Test optimizer creates with custom parameters."""
        optimizer = Muon(
            learning_rate=0.01,
            momentum=0.9,
            nesterov=False,
            ns_steps=3,
            adam_learning_rate=5e-4,
            adam_beta_1=0.85,
            adam_beta_2=0.99,
            adam_epsilon=1e-8,
            weight_decay=0.01,
            exclude_embedding_names=["embed", "token"],
        )

        assert optimizer.learning_rate == 0.01
        assert optimizer.momentum == 0.9
        assert optimizer.nesterov is False
        assert optimizer.ns_steps == 3
        assert optimizer.adam_learning_rate == 5e-4
        assert optimizer.adam_beta_1 == 0.85
        assert optimizer.adam_beta_2 == 0.99
        assert optimizer.adam_epsilon == 1e-8
        assert optimizer._weight_decay == 0.01
        assert optimizer.exclude_embedding_names == ["embed", "token"]

    def test_learning_rate_schedule(self):
        """Test optimizer accepts learning rate schedule."""
        schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.02,
            decay_steps=1000,
            decay_rate=0.9,
        )
        optimizer = Muon(learning_rate=schedule)

        # Keras 3 stores schedule in _learning_rate
        assert isinstance(
            optimizer._learning_rate,
            keras.optimizers.schedules.LearningRateSchedule
        )


class TestMuonBuild:
    """Tests for optimizer build process."""

    def test_build_creates_state_variables(self):
        """Test build creates momentum and Adam state variables."""
        optimizer = Muon()

        var1 = keras.Variable(np.random.randn(64, 128).astype(np.float32))
        var2 = keras.Variable(np.random.randn(128,).astype(np.float32))

        optimizer.build([var1, var2])

        assert optimizer.built is True
        assert len(optimizer._muon_velocities) == 2
        assert len(optimizer._adam_m) == 2
        assert len(optimizer._adam_v) == 2

    def test_build_idempotent(self):
        """Test build can be called multiple times safely."""
        optimizer = Muon()

        var1 = keras.Variable(np.random.randn(64, 128).astype(np.float32))
        optimizer.build([var1])

        num_vars_after_first = len(optimizer._muon_velocities)

        optimizer.build([var1])

        assert len(optimizer._muon_velocities) == num_vars_after_first

    def test_variable_index_mapping(self):
        """Test variable indexing works correctly via base class."""
        optimizer = Muon()

        var1 = keras.Variable(np.random.randn(64, 128).astype(np.float32))
        var2 = keras.Variable(np.random.randn(128, 256).astype(np.float32))
        var3 = keras.Variable(np.random.randn(256,).astype(np.float32))

        optimizer.build([var1, var2, var3])

        assert optimizer._get_variable_index(var1) == 0
        assert optimizer._get_variable_index(var2) == 1
        assert optimizer._get_variable_index(var3) == 2


class TestMuonRouting:
    """Tests for variable routing between Muon and Adam."""

    def test_rank2_uses_muon(self):
        """Test rank-2 tensors use Muon."""
        optimizer = Muon()

        var = keras.Variable(
            np.random.randn(64, 128).astype(np.float32),
            name="dense_kernel"
        )

        assert optimizer._should_use_muon(var) is True

    def test_rank1_uses_adam(self):
        """Test rank-1 tensors (biases) use Adam."""
        optimizer = Muon()

        var = keras.Variable(
            np.random.randn(128,).astype(np.float32),
            name="dense_bias"
        )

        assert optimizer._should_use_muon(var) is False

    def test_rank4_uses_muon(self):
        """Test rank-4 tensors (conv kernels) use Muon."""
        optimizer = Muon()

        var = keras.Variable(
            np.random.randn(3, 3, 64, 128).astype(np.float32),
            name="conv2d_kernel"
        )

        assert optimizer._should_use_muon(var) is True

    def test_embedding_uses_adam(self):
        """Test embedding layers are excluded from Muon."""
        optimizer = Muon()

        var = keras.Variable(
            np.random.randn(10000, 512).astype(np.float32),
            name="embedding_embeddings"
        )

        assert optimizer._should_use_muon(var) is False

    def test_token_embedding_uses_adam(self):
        """Test token_emb pattern is excluded from Muon."""
        optimizer = Muon()

        var = keras.Variable(
            np.random.randn(10000, 512).astype(np.float32),
            name="token_emb_kernel"
        )

        assert optimizer._should_use_muon(var) is False

    def test_custom_exclude_patterns(self):
        """Test custom exclusion patterns work."""
        optimizer = Muon(exclude_embedding_names=["special_layer", "custom"])

        var1 = keras.Variable(
            np.random.randn(64, 128).astype(np.float32),
            name="special_layer_kernel"
        )
        var2 = keras.Variable(
            np.random.randn(64, 128).astype(np.float32),
            name="custom_projection_kernel"
        )
        var3 = keras.Variable(
            np.random.randn(64, 128).astype(np.float32),
            name="dense_kernel"
        )

        assert optimizer._should_use_muon(var1) is False
        assert optimizer._should_use_muon(var2) is False
        assert optimizer._should_use_muon(var3) is True


class TestNewtonSchulz:
    """Tests for Newton-Schulz orthogonalization."""

    def test_output_shape_preserved(self):
        """Test Newton-Schulz preserves input shape."""
        optimizer = Muon()

        # Wide matrix
        G_wide = ops.convert_to_tensor(
            np.random.randn(32, 64).astype(np.float32)
        )
        result_wide = optimizer._newton_schulz5(G_wide, steps=5)
        assert ops.shape(result_wide)[0] == 32
        assert ops.shape(result_wide)[1] == 64

        # Tall matrix
        G_tall = ops.convert_to_tensor(
            np.random.randn(64, 32).astype(np.float32)
        )
        result_tall = optimizer._newton_schulz5(G_tall, steps=5)
        assert ops.shape(result_tall)[0] == 64
        assert ops.shape(result_tall)[1] == 32

        # Square matrix
        G_square = ops.convert_to_tensor(
            np.random.randn(32, 32).astype(np.float32)
        )
        result_square = optimizer._newton_schulz5(G_square, steps=5)
        assert ops.shape(result_square)[0] == 32
        assert ops.shape(result_square)[1] == 32

    def test_near_orthogonality(self):
        """Test output is approximately orthogonal after iterations."""
        optimizer = Muon(ns_steps=10)

        G = ops.convert_to_tensor(
            np.random.randn(32, 64).astype(np.float32)
        )
        result = optimizer._newton_schulz5(G, steps=10)

        # For a wide matrix, U @ U.T should be close to identity
        product = ops.matmul(result, ops.transpose(result))
        identity = ops.eye(32)

        # Check approximate orthogonality (not exact due to finite iterations)
        diff = ops.mean(ops.abs(product - identity))
        assert float(diff) < 0.5  # Reasonable threshold

    def test_handles_zero_matrix(self):
        """Test Newton-Schulz handles near-zero input gracefully."""
        optimizer = Muon()

        G = ops.convert_to_tensor(
            np.zeros((16, 32), dtype=np.float32) + 1e-10
        )
        result = optimizer._newton_schulz5(G, steps=5)

        # Should not produce NaN or Inf
        assert not np.any(np.isnan(keras.ops.convert_to_numpy(result)))
        assert not np.any(np.isinf(keras.ops.convert_to_numpy(result)))


class TestMuonUpdate:
    """Tests for the actual optimization updates."""

    def test_muon_update_modifies_variable(self):
        """Test Muon update actually modifies the variable."""
        optimizer = Muon(learning_rate=0.1, weight_decay=0.0)

        var = keras.Variable(
            np.random.randn(64, 128).astype(np.float32),
            name="dense_kernel"
        )
        initial_value = keras.ops.convert_to_numpy(var).copy()

        grad = ops.convert_to_tensor(
            np.random.randn(64, 128).astype(np.float32)
        )

        optimizer.build([var])
        optimizer.update_step(grad, var, ops.convert_to_tensor(0.1))

        final_value = keras.ops.convert_to_numpy(var)

        assert not np.allclose(initial_value, final_value)

    def test_adam_update_modifies_variable(self):
        """Test Adam update actually modifies the variable."""
        optimizer = Muon(adam_learning_rate=0.1, weight_decay=0.0)

        var = keras.Variable(
            np.random.randn(128,).astype(np.float32),
            name="dense_bias"
        )
        initial_value = keras.ops.convert_to_numpy(var).copy()

        grad = ops.convert_to_tensor(
            np.random.randn(128,).astype(np.float32)
        )

        optimizer.build([var])
        optimizer.update_step(grad, var, ops.convert_to_tensor(0.1))

        final_value = keras.ops.convert_to_numpy(var)

        assert not np.allclose(initial_value, final_value)

    def test_weight_decay_applied(self):
        """Test weight decay reduces variable magnitude."""
        optimizer = Muon(learning_rate=0.0, weight_decay=0.5)

        var = keras.Variable(
            np.ones((64, 128), dtype=np.float32),
            name="dense_kernel"
        )
        initial_norm = float(ops.norm(var))

        grad = ops.zeros((64, 128), dtype="float32")

        optimizer.build([var])
        optimizer.update_step(grad, var, ops.convert_to_tensor(1.0))

        final_norm = float(ops.norm(var))

        # Weight decay should reduce magnitude
        assert final_norm < initial_norm

    def test_momentum_accumulates(self):
        """Test momentum buffer accumulates across steps."""
        optimizer = Muon(learning_rate=0.1, momentum=0.9, nesterov=False)

        var = keras.Variable(
            np.zeros((32, 64), dtype=np.float32),
            name="dense_kernel"
        )

        grad = ops.ones((32, 64), dtype="float32")

        optimizer.build([var])
        idx = optimizer._get_variable_index(var)

        # First step
        optimizer.update_step(grad, var, ops.convert_to_tensor(0.1))
        velocity_1 = keras.ops.convert_to_numpy(
            optimizer._muon_velocities[idx]
        ).copy()

        # Second step
        optimizer.update_step(grad, var, ops.convert_to_tensor(0.1))
        velocity_2 = keras.ops.convert_to_numpy(
            optimizer._muon_velocities[idx]
        )

        # Velocity should increase with momentum
        assert np.mean(velocity_2) > np.mean(velocity_1)


class TestMuonSerialization:
    """Tests for optimizer serialization and deserialization."""

    def test_get_config_complete(self):
        """Test get_config returns all parameters."""
        optimizer = Muon(
            learning_rate=0.01,
            momentum=0.9,
            nesterov=False,
            ns_steps=3,
            adam_learning_rate=5e-4,
            adam_beta_1=0.85,
            adam_beta_2=0.99,
            adam_epsilon=1e-8,
            weight_decay=0.01,
            exclude_embedding_names=["embed", "token"],
        )

        config = optimizer.get_config()

        assert config["learning_rate"] == pytest.approx(0.01)
        assert config["momentum"] == 0.9
        assert config["nesterov"] is False
        assert config["ns_steps"] == 3
        assert config["adam_learning_rate"] == 5e-4
        assert config["adam_beta_1"] == 0.85
        assert config["adam_beta_2"] == 0.99
        assert config["adam_epsilon"] == 1e-8
        assert config["weight_decay"] == 0.01
        assert config["exclude_embedding_names"] == ["embed", "token"]

    def test_from_config_roundtrip(self):
        """Test optimizer can be reconstructed from config."""
        original = Muon(
            learning_rate=0.01,
            momentum=0.9,
            nesterov=False,
            ns_steps=3,
            weight_decay=0.05,
        )

        config = original.get_config()
        restored = Muon.from_config(config)

        assert restored.learning_rate == original.learning_rate
        assert restored.momentum == original.momentum
        assert restored.nesterov == original.nesterov
        assert restored.ns_steps == original.ns_steps
        assert restored._weight_decay == original._weight_decay

    def test_keras_serialization(self):
        """Test optimizer serializes through Keras utilities."""
        optimizer = Muon(learning_rate=0.01, momentum=0.9)

        serialized = keras.saving.serialize_keras_object(optimizer)
        restored = keras.saving.deserialize_keras_object(serialized)

        assert isinstance(restored, Muon)
        assert restored.learning_rate == 0.01
        assert restored.momentum == 0.9


class TestMuonIntegration:
    """Integration tests with Keras models."""

    def test_simple_model_training(self):
        """Test optimizer works with a simple model."""
        model = keras.Sequential([
            keras.layers.Dense(64, activation="relu", name="dense1"),
            keras.layers.Dense(32, activation="relu", name="dense2"),
            keras.layers.Dense(10, name="output"),
        ])

        optimizer = Muon(learning_rate=0.01, adam_learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Dummy data
        x = np.random.randn(100, 128).astype(np.float32)
        y = np.random.randint(0, 10, size=(100,))

        # Should train without errors
        history = model.fit(x, y, epochs=2, batch_size=32, verbose=0)

        assert len(history.history["loss"]) == 2
        assert history.history["loss"][1] <= history.history["loss"][0] * 1.5

    def test_model_with_embeddings(self):
        """Test optimizer correctly routes embedding vs dense layers."""
        inputs = keras.Input(shape=(10,), dtype="int32")
        x = keras.layers.Embedding(1000, 64, name="embedding")(inputs)
        x = keras.layers.GlobalAveragePooling1D()(x)
        x = keras.layers.Dense(32, activation="relu", name="dense")(x)
        outputs = keras.layers.Dense(5, name="output")(x)
        model = keras.Model(inputs, outputs)

        optimizer = Muon(learning_rate=0.01)
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
        )

        # Dummy data
        x = np.random.randint(0, 1000, size=(50, 10))
        y = np.random.randint(0, 5, size=(50,))

        # Should train without errors
        model.fit(x, y, epochs=1, batch_size=16, verbose=0)

    def test_model_save_load(self):
        """Test model with Muon optimizer can be saved and loaded."""
        import tempfile
        import os

        model = keras.Sequential([
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(10),
        ])

        optimizer = Muon(learning_rate=0.01, momentum=0.9)
        model.compile(optimizer=optimizer, loss="mse")

        # Build model
        x = np.random.randn(10, 64).astype(np.float32)
        y = np.random.randn(10, 10).astype(np.float32)
        model.fit(x, y, epochs=1, verbose=0)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.keras")
            model.save(path)
            loaded_model = keras.models.load_model(path)

        # Verify optimizer restored
        assert isinstance(loaded_model.optimizer, Muon)
        assert loaded_model.optimizer.momentum == 0.9

        # Verify model still works
        pred_original = model.predict(x, verbose=0)
        pred_loaded = loaded_model.predict(x, verbose=0)

        np.testing.assert_allclose(
            pred_original, pred_loaded,
            rtol=1e-5, atol=1e-5,
            err_msg="Predictions should match after save/load"
        )

    def test_gradient_clipping_compatible(self):
        """Test optimizer works with gradient clipping."""
        model = keras.Sequential([
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(10),
        ])

        optimizer = Muon(
            learning_rate=0.01,
            clipnorm=1.0,
        )
        model.compile(optimizer=optimizer, loss="mse")

        x = np.random.randn(20, 64).astype(np.float32)
        y = np.random.randn(20, 10).astype(np.float32)

        # Should train without errors
        model.fit(x, y, epochs=2, batch_size=10, verbose=0)


class TestMuonEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_variable(self):
        """Test optimizer works with single variable."""
        optimizer = Muon()

        var = keras.Variable(np.random.randn(32, 64).astype(np.float32))
        optimizer.build([var])

        assert len(optimizer._muon_velocities) == 1

    def test_mixed_dtypes(self):
        """Test optimizer handles mixed precision variables."""
        optimizer = Muon()

        var_f32 = keras.Variable(
            np.random.randn(32, 64).astype(np.float32),
            name="float32_var"
        )
        var_f16 = keras.Variable(
            np.random.randn(64,).astype(np.float16),
            name="float16_var"
        )

        optimizer.build([var_f32, var_f16])

        grad_f32 = ops.convert_to_tensor(
            np.random.randn(32, 64).astype(np.float32)
        )
        grad_f16 = ops.convert_to_tensor(
            np.random.randn(64,).astype(np.float16)
        )

        # Should not raise dtype errors
        optimizer.update_step(grad_f32, var_f32, ops.convert_to_tensor(0.01))
        optimizer.update_step(grad_f16, var_f16, ops.convert_to_tensor(0.01))

    def test_very_small_matrix(self):
        """Test optimizer handles very small matrices."""
        optimizer = Muon()

        var = keras.Variable(
            np.random.randn(2, 3).astype(np.float32),
            name="tiny_kernel"
        )

        grad = ops.convert_to_tensor(
            np.random.randn(2, 3).astype(np.float32)
        )

        optimizer.build([var])
        optimizer.update_step(grad, var, ops.convert_to_tensor(0.1))

        # Should not produce NaN
        assert not np.any(np.isnan(keras.ops.convert_to_numpy(var)))

    def test_zero_learning_rate(self):
        """Test optimizer with zero learning rate doesn't modify variables."""
        optimizer = Muon(learning_rate=0.0, adam_learning_rate=0.0, weight_decay=0.0)

        var = keras.Variable(
            np.ones((32, 64), dtype=np.float32),
            name="dense_kernel"
        )
        initial_value = keras.ops.convert_to_numpy(var).copy()

        grad = ops.convert_to_tensor(
            np.random.randn(32, 64).astype(np.float32)
        )

        optimizer.build([var])
        optimizer.update_step(grad, var, ops.convert_to_tensor(0.0))

        np.testing.assert_allclose(
            initial_value,
            keras.ops.convert_to_numpy(var),
            rtol=1e-6, atol=1e-6,
            err_msg="Variable should not change with zero learning rate"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])