"""
Tests for NTM Model wrapper module.

Tests cover:
    - NTMModel initialization
    - NTMModel.from_variant classmethod
    - create_ntm_variant factory function
    - Forward pass with various configurations
    - Serialization and model saving
"""

import pytest
import numpy as np
import keras
from keras import ops
import tensorflow as tf
import tempfile
import os

from dl_techniques.layers.ntm.ntm_interface import NTMConfig
from dl_techniques.models.ntm.model import NTMModel, create_ntm_variant


# ---------------------------------------------------------------------
# NTMModel Initialization Tests
# ---------------------------------------------------------------------


class TestNTMModelInit:
    """Tests for NTMModel initialization."""

    def test_init_with_config_object(self):
        """Test initialization with NTMConfig object."""
        config = NTMConfig(
            memory_size=32,
            memory_dim=16,
            controller_dim=64,
            num_read_heads=1,
            num_write_heads=1,
        )
        model = NTMModel(
            input_shape=(10, 8),
            output_dim=4,
            config=config,
        )

        assert model.output_dim == 4
        assert model.return_sequences is True
        assert model.use_projection is True
        assert model.config_obj.memory_size == 32

    def test_init_with_dict_config(self):
        """Test initialization with dict config."""
        config_dict = {
            "memory_size": 64,
            "memory_dim": 32,
            "controller_dim": 128,
            "num_read_heads": 2,
            "num_write_heads": 1,
            "controller_type": "lstm",
            "shift_range": 3,
        }
        model = NTMModel(
            input_shape=(None, 16),
            output_dim=8,
            config=config_dict,
        )

        assert model.config_obj.memory_size == 64
        assert model.config_obj.num_read_heads == 2

    def test_init_return_sequences_false(self):
        """Test initialization with return_sequences=False."""
        config = NTMConfig(memory_size=32, memory_dim=16, controller_dim=64)
        model = NTMModel(
            input_shape=(10, 8),
            output_dim=4,
            config=config,
            return_sequences=False,
        )

        assert model.return_sequences is False

    def test_init_use_projection_false(self):
        """Test initialization with use_projection=False."""
        config = NTMConfig(memory_size=32, memory_dim=16, controller_dim=64)
        model = NTMModel(
            input_shape=(10, 8),
            output_dim=4,
            config=config,
            use_projection=False,
        )

        assert model.use_projection is False
        assert model.projection is None

    def test_init_creates_layers(self):
        """Test that initialization creates all required layers."""
        config = NTMConfig(memory_size=32, memory_dim=16, controller_dim=64)
        model = NTMModel(
            input_shape=(10, 8),
            output_dim=4,
            config=config,
        )

        assert model.cell is not None
        assert model.rnn is not None
        assert model.projection is not None


# ---------------------------------------------------------------------
# NTMModel.from_variant Tests
# ---------------------------------------------------------------------


class TestNTMModelFromVariant:
    """Tests for NTMModel.from_variant classmethod."""

    def test_tiny_variant(self):
        """Test creating model from 'tiny' variant."""
        model = NTMModel.from_variant(
            variant="tiny",
            input_shape=(10, 8),
            output_dim=4,
        )

        assert model.config_obj.memory_size == 32
        assert model.config_obj.memory_dim == 16
        assert model.config_obj.controller_dim == 64
        assert model.config_obj.num_read_heads == 1
        assert model.config_obj.num_write_heads == 1

    def test_base_variant(self):
        """Test creating model from 'base' variant."""
        model = NTMModel.from_variant(
            variant="base",
            input_shape=(20, 16),
            output_dim=8,
        )

        assert model.config_obj.memory_size == 128
        assert model.config_obj.memory_dim == 32
        assert model.config_obj.controller_dim == 256

    def test_large_variant(self):
        """Test creating model from 'large' variant."""
        model = NTMModel.from_variant(
            variant="large",
            input_shape=(50, 32),
            output_dim=16,
        )

        assert model.config_obj.memory_size == 256
        assert model.config_obj.memory_dim == 64
        assert model.config_obj.controller_dim == 512
        assert model.config_obj.num_read_heads == 2
        assert model.config_obj.num_write_heads == 2

    def test_invalid_variant(self):
        """Test that invalid variant raises ValueError."""
        try:
            NTMModel.from_variant(
                variant="invalid_variant",
                input_shape=(10, 8),
                output_dim=4,
            )
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Unknown variant" in str(e)

    def test_variant_with_overrides(self):
        """Test variant with config overrides."""
        model = NTMModel.from_variant(
            variant="tiny",
            input_shape=(10, 8),
            output_dim=4,
            controller_type="gru",
            num_read_heads=2,
        )

        # Overridden values
        assert model.config_obj.controller_type == "gru"
        assert model.config_obj.num_read_heads == 2

        # Default tiny values preserved
        assert model.config_obj.memory_size == 32
        assert model.config_obj.memory_dim == 16

    def test_variant_with_model_kwargs(self):
        """Test variant with model-level kwargs."""
        model = NTMModel.from_variant(
            variant="tiny",
            input_shape=(10, 8),
            output_dim=4,
            return_sequences=False,
            use_projection=False,
        )

        assert model.return_sequences is False
        assert model.use_projection is False


# ---------------------------------------------------------------------
# NTMModel Forward Pass Tests
# ---------------------------------------------------------------------


class TestNTMModelCall:
    """Tests for NTMModel forward pass."""

    def test_call_return_sequences_true(self):
        """Test forward pass with return_sequences=True."""
        batch_size = 4
        seq_len = 10
        input_dim = 8
        output_dim = 4

        model = NTMModel.from_variant(
            variant="tiny",
            input_shape=(seq_len, input_dim),
            output_dim=output_dim,
            return_sequences=True,
        )

        inputs = keras.random.normal((batch_size, seq_len, input_dim), seed=42)
        outputs = model(inputs)

        assert ops.shape(outputs) == (batch_size, seq_len, output_dim)

    def test_call_return_sequences_false(self):
        """Test forward pass with return_sequences=False."""
        batch_size = 4
        seq_len = 10
        input_dim = 8
        output_dim = 4

        model = NTMModel.from_variant(
            variant="tiny",
            input_shape=(seq_len, input_dim),
            output_dim=output_dim,
            return_sequences=False,
        )

        inputs = keras.random.normal((batch_size, seq_len, input_dim), seed=42)
        outputs = model(inputs)

        assert ops.shape(outputs) == (batch_size, output_dim)

    def test_call_without_projection(self):
        """Test forward pass without output projection."""
        batch_size = 4
        seq_len = 10
        input_dim = 8

        config = NTMConfig(
            memory_size=32,
            memory_dim=16,
            controller_dim=64,
            num_read_heads=1,
        )
        model = NTMModel(
            input_shape=(seq_len, input_dim),
            output_dim=4,  # ignored when use_projection=False
            config=config,
            use_projection=False,
        )

        inputs = keras.random.normal((batch_size, seq_len, input_dim), seed=42)
        outputs = model(inputs)

        # Output dim = controller_dim + num_read_heads * memory_dim
        expected_dim = 64 + 1 * 16
        assert ops.shape(outputs) == (batch_size, seq_len, expected_dim)

    def test_call_variable_sequence_length(self):
        """Test forward pass with variable sequence length."""
        batch_size = 4
        input_dim = 8
        output_dim = 4

        model = NTMModel.from_variant(
            variant="tiny",
            input_shape=(None, input_dim),
            output_dim=output_dim,
        )

        # Different sequence lengths
        inputs_5 = keras.random.normal((batch_size, 5, input_dim), seed=42)
        inputs_15 = keras.random.normal((batch_size, 15, input_dim), seed=43)

        outputs_5 = model(inputs_5)
        outputs_15 = model(inputs_15)

        assert ops.shape(outputs_5) == (batch_size, 5, output_dim)
        assert ops.shape(outputs_15) == (batch_size, 15, output_dim)

    def test_call_training_mode(self):
        """Test forward pass with training=True and training=False."""
        batch_size = 4
        seq_len = 10
        input_dim = 8
        output_dim = 4

        model = NTMModel.from_variant(
            variant="tiny",
            input_shape=(seq_len, input_dim),
            output_dim=output_dim,
        )

        inputs = keras.random.normal((batch_size, seq_len, input_dim), seed=42)

        # Both modes should work
        outputs_train = model(inputs, training=True)
        outputs_eval = model(inputs, training=False)

        assert ops.shape(outputs_train) == (batch_size, seq_len, output_dim)
        assert ops.shape(outputs_eval) == (batch_size, seq_len, output_dim)

    def test_call_with_different_controller_types(self):
        """Test forward pass with different controller types."""
        batch_size = 2
        seq_len = 5
        input_dim = 8
        output_dim = 4

        for controller_type in ["lstm", "gru", "feedforward"]:
            model = NTMModel.from_variant(
                variant="tiny",
                input_shape=(seq_len, input_dim),
                output_dim=output_dim,
                controller_type=controller_type,
            )

            inputs = keras.random.normal((batch_size, seq_len, input_dim), seed=42)
            outputs = model(inputs)

            assert ops.shape(outputs) == (batch_size, seq_len, output_dim), (
                f"Failed for controller_type={controller_type}"
            )


# ---------------------------------------------------------------------
# NTMModel Shape Tests
# ---------------------------------------------------------------------


class TestNTMModelComputeOutputShape:
    """Tests for NTMModel.compute_output_shape method."""

    def test_compute_output_shape_with_sequences(self):
        """Test compute_output_shape with return_sequences=True."""
        model = NTMModel.from_variant(
            variant="tiny",
            input_shape=(10, 8),
            output_dim=4,
            return_sequences=True,
        )

        output_shape = model.compute_output_shape((None, 10, 8))
        assert output_shape == (None, 10, 4)

    def test_compute_output_shape_without_sequences(self):
        """Test compute_output_shape with return_sequences=False."""
        model = NTMModel.from_variant(
            variant="tiny",
            input_shape=(10, 8),
            output_dim=4,
            return_sequences=False,
        )

        output_shape = model.compute_output_shape((None, 10, 8))
        assert output_shape == (None, 4)

    def test_compute_output_shape_without_projection(self):
        """Test compute_output_shape without output projection."""
        config = NTMConfig(
            memory_size=32,
            memory_dim=16,
            controller_dim=64,
            num_read_heads=2,
        )
        model = NTMModel(
            input_shape=(10, 8),
            output_dim=4,
            config=config,
            use_projection=False,
        )

        output_shape = model.compute_output_shape((None, 10, 8))
        # output = controller_dim + num_read_heads * memory_dim
        expected_dim = 64 + 2 * 16
        assert output_shape == (None, 10, expected_dim)


# ---------------------------------------------------------------------
# NTMModel Serialization Tests
# ---------------------------------------------------------------------


class TestNTMModelSerialization:
    """Tests for NTMModel serialization."""

    def test_get_config(self):
        """Test get_config returns correct configuration."""
        config = NTMConfig(
            memory_size=32,
            memory_dim=16,
            controller_dim=64,
        )
        model = NTMModel(
            input_shape=(10, 8),
            output_dim=4,
            config=config,
            return_sequences=False,
            use_projection=True,
        )

        model_config = model.get_config()

        assert model_config["input_shape"] == (10, 8)
        assert model_config["output_dim"] == 4
        assert model_config["return_sequences"] is False
        assert model_config["use_projection"] is True
        assert "config" in model_config
        assert model_config["config"]["memory_size"] == 32

    def test_from_config(self):
        """Test from_config reconstructs model correctly."""
        config = NTMConfig(
            memory_size=32,
            memory_dim=16,
            controller_dim=64,
            num_read_heads=2,
        )
        model = NTMModel(
            input_shape=(10, 8),
            output_dim=4,
            config=config,
            return_sequences=False,
        )

        model_config = model.get_config()
        model_restored = NTMModel.from_config(model_config)

        assert model_restored.output_dim == model.output_dim
        assert model_restored.return_sequences == model.return_sequences
        assert model_restored.config_obj.memory_size == model.config_obj.memory_size
        assert model_restored.config_obj.num_read_heads == model.config_obj.num_read_heads

    def test_save_and_load(self):
        """Test model saving and loading."""
        model = NTMModel.from_variant(
            variant="tiny",
            input_shape=(5, 8),
            output_dim=4,
        )

        # Build the model
        inputs = keras.random.normal((2, 5, 8), seed=42)
        output_before = model(inputs)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "ntm_model.keras")
            model.save(model_path)

            model_loaded = keras.models.load_model(model_path)

        output_after = model_loaded(inputs)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output_before),
            keras.ops.convert_to_numpy(output_after),
            rtol=1e-5,
            atol=1e-5,
            err_msg="Loaded model should produce same output",
        )

    def test_save_and_load_variant(self):
        """Test saving and loading model created from variant."""
        model = NTMModel.from_variant(
            variant="base",
            input_shape=(10, 16),
            output_dim=8,
            return_sequences=False,
        )

        inputs = keras.random.normal((2, 10, 16), seed=42)
        output_before = model(inputs)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "ntm_variant.keras")
            model.save(model_path)

            model_loaded = keras.models.load_model(model_path)

        output_after = model_loaded(inputs)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output_before),
            keras.ops.convert_to_numpy(output_after),
            rtol=1e-5,
            atol=1e-5,
        )


# ---------------------------------------------------------------------
# create_ntm_variant Factory Tests
# ---------------------------------------------------------------------


class TestCreateNTMVariant:
    """Tests for create_ntm_variant factory function."""

    def test_create_tiny_variant(self):
        """Test creating tiny variant with factory."""
        model = create_ntm_variant(
            variant="tiny",
            input_shape=(10, 8),
            output_dim=4,
        )

        assert isinstance(model, NTMModel)
        assert model.config_obj.memory_size == 32

    def test_create_base_variant(self):
        """Test creating base variant with factory."""
        model = create_ntm_variant(
            variant="base",
            input_shape=(20, 16),
            output_dim=8,
        )

        assert model.config_obj.memory_size == 128
        assert model.config_obj.controller_dim == 256

    def test_create_large_variant(self):
        """Test creating large variant with factory."""
        model = create_ntm_variant(
            variant="large",
            input_shape=(50, 32),
            output_dim=16,
        )

        assert model.config_obj.memory_size == 256
        assert model.config_obj.num_read_heads == 2

    def test_factory_with_overrides(self):
        """Test factory with config overrides."""
        model = create_ntm_variant(
            variant="tiny",
            input_shape=(10, 8),
            output_dim=4,
            memory_size=64,
            controller_type="gru",
        )

        assert model.config_obj.memory_size == 64
        assert model.config_obj.controller_type == "gru"

    def test_factory_return_sequences_false(self):
        """Test factory with return_sequences=False."""
        model = create_ntm_variant(
            variant="tiny",
            input_shape=(10, 8),
            output_dim=4,
            return_sequences=False,
        )

        assert model.return_sequences is False

    def test_factory_produces_working_model(self):
        """Test that factory produces a working model."""
        batch_size = 2
        seq_len = 5
        input_dim = 8
        output_dim = 4

        model = create_ntm_variant(
            variant="tiny",
            input_shape=(seq_len, input_dim),
            output_dim=output_dim,
        )

        inputs = keras.random.normal((batch_size, seq_len, input_dim), seed=42)
        outputs = model(inputs)

        assert ops.shape(outputs) == (batch_size, seq_len, output_dim)


# ---------------------------------------------------------------------
# Training and Gradient Tests
# ---------------------------------------------------------------------


class TestNTMModelTraining:
    """Tests for NTMModel training."""

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        batch_size = 2
        seq_len = 5
        input_dim = 8
        output_dim = 4

        model = NTMModel.from_variant(
            variant="tiny",
            input_shape=(seq_len, input_dim),
            output_dim=output_dim,
        )

        inputs = tf.Variable(
            keras.random.normal((batch_size, seq_len, input_dim), seed=42)
        )

        with tf.GradientTape() as tape:
            outputs = model(inputs, training=True)
            loss = ops.sum(outputs)

        grads = tape.gradient(loss, inputs)
        assert grads is not None

    def test_training_step(self):
        """Test a simple training step."""
        batch_size = 2
        seq_len = 5
        input_dim = 8
        output_dim = 4

        model = NTMModel.from_variant(
            variant="tiny",
            input_shape=(seq_len, input_dim),
            output_dim=output_dim,
        )

        model.compile(optimizer="adam", loss="mse")

        # Generate dummy data
        x = np.random.randn(batch_size, seq_len, input_dim).astype(np.float32)
        y = np.random.randn(batch_size, seq_len, output_dim).astype(np.float32)

        # Train for one step
        history = model.fit(x, y, epochs=1, verbose=0)

        assert "loss" in history.history
        assert len(history.history["loss"]) == 1

    def test_training_multiple_epochs(self):
        """Test training for multiple epochs."""
        batch_size = 4
        seq_len = 5
        input_dim = 8
        output_dim = 4

        model = NTMModel.from_variant(
            variant="tiny",
            input_shape=(seq_len, input_dim),
            output_dim=output_dim,
        )

        model.compile(optimizer="adam", loss="mse")

        x = np.random.randn(batch_size * 2, seq_len, input_dim).astype(np.float32)
        y = np.random.randn(batch_size * 2, seq_len, output_dim).astype(np.float32)

        history = model.fit(x, y, epochs=3, batch_size=batch_size, verbose=0)

        assert len(history.history["loss"]) == 3
        # Loss should generally decrease (not guaranteed but likely)

    def test_training_return_sequences_false(self):
        """Test training with return_sequences=False."""
        batch_size = 4
        seq_len = 5
        input_dim = 8
        output_dim = 4

        model = NTMModel.from_variant(
            variant="tiny",
            input_shape=(seq_len, input_dim),
            output_dim=output_dim,
            return_sequences=False,
        )

        model.compile(optimizer="adam", loss="mse")

        x = np.random.randn(batch_size, seq_len, input_dim).astype(np.float32)
        y = np.random.randn(batch_size, output_dim).astype(np.float32)

        history = model.fit(x, y, epochs=1, verbose=0)

        assert "loss" in history.history

    def test_model_weights_update(self):
        """Test that model weights are updated during training."""
        model = NTMModel.from_variant(
            variant="tiny",
            input_shape=(5, 8),
            output_dim=4,
        )

        # Build model
        x = np.random.randn(2, 5, 8).astype(np.float32)
        _ = model(x)

        # Get initial weights
        initial_weights = [w.numpy().copy() for w in model.trainable_weights[:3]]

        model.compile(optimizer="adam", loss="mse")

        y = np.random.randn(2, 5, 4).astype(np.float32)
        model.fit(x, y, epochs=5, verbose=0)

        # Get updated weights
        updated_weights = [w.numpy() for w in model.trainable_weights[:3]]

        # At least some weights should have changed
        weights_changed = False
        for initial, updated in zip(initial_weights, updated_weights):
            if not np.allclose(initial, updated):
                weights_changed = True
                break

        assert weights_changed, "Some weights should have been updated during training"


# ---------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------


class TestNTMModelIntegration:
    """Integration tests for NTMModel."""

    def test_copy_task_structure(self):
        """Test model structure suitable for copy task."""
        batch_size = 4
        seq_len = 10
        dim = 8

        model = create_ntm_variant(
            variant="base",
            input_shape=(seq_len, dim),
            output_dim=dim,
        )

        inputs = keras.random.normal((batch_size, seq_len, dim), seed=42)
        outputs = model(inputs)

        assert ops.shape(outputs) == (batch_size, seq_len, dim)

    def test_classification_structure(self):
        """Test model structure suitable for sequence classification."""
        batch_size = 4
        seq_len = 20
        input_dim = 32
        num_classes = 10

        model = create_ntm_variant(
            variant="base",
            input_shape=(seq_len, input_dim),
            output_dim=num_classes,
            return_sequences=False,
        )

        inputs = keras.random.normal((batch_size, seq_len, input_dim), seed=42)
        outputs = model(inputs)

        assert ops.shape(outputs) == (batch_size, num_classes)

    def test_all_variants_work(self):
        """Test that all predefined variants work correctly."""
        batch_size = 2
        seq_len = 5
        input_dim = 8
        output_dim = 4

        for variant in ["tiny", "base", "large"]:
            model = create_ntm_variant(
                variant=variant,
                input_shape=(seq_len, input_dim),
                output_dim=output_dim,
            )

            inputs = keras.random.normal((batch_size, seq_len, input_dim), seed=42)
            outputs = model(inputs)

            assert ops.shape(outputs) == (batch_size, seq_len, output_dim), (
                f"Failed for variant={variant}"
            )

    def test_functional_api_wrapping(self):
        """Test that NTMModel can be used in Keras Functional API."""
        seq_len = 10
        input_dim = 8
        output_dim = 4

        # Create NTMModel
        ntm_model = NTMModel.from_variant(
            variant="tiny",
            input_shape=(seq_len, input_dim),
            output_dim=output_dim,
        )

        # Wrap in Functional API
        inputs = keras.Input(shape=(seq_len, input_dim))
        outputs = ntm_model(inputs)
        functional_model = keras.Model(inputs, outputs)

        # Test
        batch_size = 2
        x = keras.random.normal((batch_size, seq_len, input_dim), seed=42)
        y = functional_model(x)

        assert ops.shape(y) == (batch_size, seq_len, output_dim)

    def test_model_summary(self):
        """Test that model.summary() works."""
        model = NTMModel.from_variant(
            variant="tiny",
            input_shape=(10, 8),
            output_dim=4,
        )

        # Build the model first
        _ = model(keras.random.normal((1, 10, 8), seed=42))

        # This should not raise an error
        model.summary()


# ---------------------------------------------------------------------
# Edge Cases Tests
# ---------------------------------------------------------------------


class TestNTMModelEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_timestep(self):
        """Test with sequence length of 1."""
        model = NTMModel.from_variant(
            variant="tiny",
            input_shape=(1, 8),
            output_dim=4,
        )

        inputs = keras.random.normal((2, 1, 8), seed=42)
        outputs = model(inputs)

        assert ops.shape(outputs) == (2, 1, 4)

    def test_batch_size_one(self):
        """Test with batch size of 1."""
        model = NTMModel.from_variant(
            variant="tiny",
            input_shape=(10, 8),
            output_dim=4,
        )

        inputs = keras.random.normal((1, 10, 8), seed=42)
        outputs = model(inputs)

        assert ops.shape(outputs) == (1, 10, 4)

    def test_large_batch(self):
        """Test with large batch size."""
        model = NTMModel.from_variant(
            variant="tiny",
            input_shape=(5, 8),
            output_dim=4,
        )

        inputs = keras.random.normal((32, 5, 8), seed=42)
        outputs = model(inputs)

        assert ops.shape(outputs) == (32, 5, 4)

    def test_long_sequence(self):
        """Test with long sequence."""
        model = NTMModel.from_variant(
            variant="tiny",
            input_shape=(100, 8),
            output_dim=4,
        )

        inputs = keras.random.normal((2, 100, 8), seed=42)
        outputs = model(inputs)

        assert ops.shape(outputs) == (2, 100, 4)

    def test_high_dimensional_input(self):
        """Test with high-dimensional input."""
        model = NTMModel.from_variant(
            variant="tiny",
            input_shape=(10, 256),
            output_dim=4,
        )

        inputs = keras.random.normal((2, 10, 256), seed=42)
        outputs = model(inputs)

        assert ops.shape(outputs) == (2, 10, 4)

    def test_output_dim_equals_one(self):
        """Test with output dimension of 1."""
        model = NTMModel.from_variant(
            variant="tiny",
            input_shape=(10, 8),
            output_dim=1,
        )

        inputs = keras.random.normal((2, 10, 8), seed=42)
        outputs = model(inputs)

        assert ops.shape(outputs) == (2, 10, 1)


# ---------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------


if __name__ == "__main__":
    pytest.main([__file__, "-v"])