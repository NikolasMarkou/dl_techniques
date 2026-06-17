"""
Comprehensive Test Suite for PowerMLP Model
==========================================

This test suite follows the patterns from "Complete Guide to Modern Keras 3
Custom Layers and Models - Refined.md" to ensure robust, production-ready
testing of the PowerMLP model implementation.

Test Coverage:
- Initialization and configuration validation
- Forward pass and building
- Full serialization cycle (CRITICAL)
- Configuration completeness
- Gradient computation
- Training vs inference modes
- Edge cases and error handling
- Model variants
- Helper functions
- Integration with Keras workflows

Run with:
    pytest test_power_mlp.py -v
    pytest test_power_mlp.py -v --cov=power_mlp --cov-report=html
"""

import pytest
import tempfile
import os
import numpy as np
import keras
from keras import ops
from typing import Dict, Any, List
import tensorflow as tf

# ---------------------------------------------------------------------
# Import the PowerMLP model and related components
# ---------------------------------------------------------------------

from dl_techniques.models.power_mlp.model import (
    PowerMLP,
    create_power_mlp,
    create_power_mlp_regressor,
    create_power_mlp_binary_classifier
)
from dl_techniques.layers.ffn.power_mlp_layer import PowerMLPLayer

# ---------------------------------------------------------------------


class TestPowerMLPInitialization:
    """Test suite for PowerMLP model initialization and validation."""

    @pytest.fixture
    def basic_config(self) -> Dict[str, Any]:
        """Standard configuration for testing."""
        return {
            'hidden_units': [100, 64, 32, 10],
            'k': 3,
            'dropout_rate': 0.2,
            'batch_normalization': True
        }

    def test_basic_initialization(self, basic_config: Dict[str, Any]) -> None:
        """Test basic model initialization with valid parameters."""
        model = PowerMLP(**basic_config)

        # Verify configuration is stored
        assert model.hidden_units == basic_config['hidden_units']
        assert model.k == basic_config['k']
        assert model.dropout_rate == basic_config['dropout_rate']
        assert model.batch_normalization == basic_config['batch_normalization']

        # Verify model is not yet built
        assert not model.built

        # Verify layer types
        assert all(isinstance(layer, PowerMLPLayer) for layer in model.hidden_layers)
        assert model.output_layer.units == 10

    def test_minimal_configuration(self) -> None:
        """Test initialization with minimal configuration."""
        model = PowerMLP(hidden_units=[10, 5])

        assert model.hidden_units == [10, 5]
        assert model.k == 3  # Default value
        assert model.dropout_rate == 0.0  # Default value
        assert not model.batch_normalization  # Default value
        assert len(model.hidden_layers) == 0  # No hidden layers, just input->output

    def test_complex_architecture(self) -> None:
        """Test initialization with complex architecture."""
        model = PowerMLP(
            hidden_units=[784, 512, 256, 128, 64, 10],
            k=4,
            dropout_rate=0.3,
            batch_normalization=True,
            kernel_regularizer='l2',
            output_activation='softmax'
        )

        assert len(model.hidden_layers) == 4
        assert model.k == 4
        assert model.kernel_regularizer is not None

    def test_invalid_hidden_units_empty(self) -> None:
        """Test that empty hidden_units raises ValueError."""
        with pytest.raises(ValueError, match="at least an input and output size"):
            PowerMLP(hidden_units=[])

    def test_invalid_hidden_units_single(self) -> None:
        """Test that single element hidden_units raises ValueError."""
        with pytest.raises(ValueError, match="at least an input and output size"):
            PowerMLP(hidden_units=[10])

    def test_invalid_hidden_units_negative(self) -> None:
        """Test that negative units raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            PowerMLP(hidden_units=[10, -5, 10])

    def test_invalid_hidden_units_zero(self) -> None:
        """Test that zero units raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            PowerMLP(hidden_units=[10, 0, 10])

    def test_invalid_k_type(self) -> None:
        """Test that non-integer k raises TypeError."""
        with pytest.raises(TypeError, match="must be an integer"):
            PowerMLP(hidden_units=[10, 5], k=3.5)

    def test_invalid_k_negative(self) -> None:
        """Test that negative k raises ValueError."""
        with pytest.raises(ValueError, match="must be a positive integer"):
            PowerMLP(hidden_units=[10, 5], k=-2)

    def test_invalid_k_zero(self) -> None:
        """Test that zero k raises ValueError."""
        with pytest.raises(ValueError, match="must be a positive integer"):
            PowerMLP(hidden_units=[10, 5], k=0)

    def test_invalid_dropout_rate_negative(self) -> None:
        """Test that negative dropout_rate raises ValueError."""
        with pytest.raises(ValueError, match="must be in"):
            PowerMLP(hidden_units=[10, 5], dropout_rate=-0.1)

    def test_invalid_dropout_rate_too_high(self) -> None:
        """Test that dropout_rate > 1 raises ValueError."""
        with pytest.raises(ValueError, match="must be in"):
            PowerMLP(hidden_units=[10, 5], dropout_rate=1.5)


class TestPowerMLPForwardPass:
    """Test suite for PowerMLP forward pass and building."""

    @pytest.fixture
    def model(self) -> PowerMLP:
        """Create a standard model for testing."""
        return PowerMLP(
            hidden_units=[20, 32, 16, 5],
            k=3,
            dropout_rate=0.1,
            batch_normalization=False
        )

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Create sample input tensor."""
        return keras.random.normal(shape=(8, 20))

    def test_forward_pass_builds_model(
        self,
        model: PowerMLP,
        sample_input: keras.KerasTensor
    ) -> None:
        """Test that forward pass builds the model."""
        assert not model.built

        output = model(sample_input, training=False)

        assert model.built
        assert output.shape == (8, 5)

    def test_output_shape_correct(
        self,
        model: PowerMLP,
        sample_input: keras.KerasTensor
    ) -> None:
        """Test that output shape matches expected dimensions."""
        output = model(sample_input)

        assert output.shape[0] == sample_input.shape[0]  # Batch size preserved
        assert output.shape[-1] == model.hidden_units[-1]  # Output dim correct

    def test_different_batch_sizes(self, model: PowerMLP) -> None:
        """Test that model handles different batch sizes."""
        for batch_size in [1, 4, 16, 32]:
            x = keras.random.normal(shape=(batch_size, 20))
            output = model(x)
            assert output.shape == (batch_size, 5)

    def test_compute_output_shape(self, model: PowerMLP) -> None:
        """Test compute_output_shape method."""
        input_shape = (None, 20)
        output_shape = model.compute_output_shape(input_shape)

        assert output_shape == (None, 5)

    def test_forward_pass_with_softmax(self) -> None:
        """Test forward pass with softmax output activation."""
        model = PowerMLP(
            hidden_units=[10, 8, 3],
            output_activation='softmax'
        )
        x = keras.random.normal(shape=(4, 10))
        output = model(x)

        # Check softmax properties
        output_np = ops.convert_to_numpy(output)
        assert np.allclose(np.sum(output_np, axis=-1), 1.0, atol=1e-6)
        assert np.all(output_np >= 0.0)
        assert np.all(output_np <= 1.0)

    def test_forward_pass_with_dropout(self) -> None:
        """Test forward pass with dropout enabled."""
        model = PowerMLP(
            hidden_units=[10, 8, 5],
            dropout_rate=0.5
        )
        x = keras.random.normal(shape=(4, 10))

        # Training mode should apply dropout
        output_train = model(x, training=True)
        assert output_train.shape == (4, 5)

        # Inference mode should not apply dropout
        output_infer = model(x, training=False)
        assert output_infer.shape == (4, 5)

    def test_forward_pass_with_batch_norm(self) -> None:
        """Test forward pass with batch normalization."""
        model = PowerMLP(
            hidden_units=[10, 8, 5],
            batch_normalization=True
        )
        x = keras.random.normal(shape=(4, 10))

        output_train = model(x, training=True)
        output_infer = model(x, training=False)

        assert output_train.shape == (4, 5)
        assert output_infer.shape == (4, 5)


class TestPowerMLPSerialization:
    """
    CRITICAL TEST SUITE: Serialization cycle testing.

    This is the most important test suite as it validates that the model
    can be saved and loaded correctly, preserving all weights and behavior.
    """

    @pytest.fixture
    def model_config(self) -> Dict[str, Any]:
        """Configuration for serialization tests."""
        return {
            'hidden_units': [50, 32, 16, 10],
            'k': 3,
            'dropout_rate': 0.2,
            'batch_normalization': True,
            'output_activation': 'softmax',
            'kernel_initializer': 'he_normal',
            'use_bias': True
        }

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Sample input for serialization tests."""
        return keras.random.normal(shape=(4, 50))

    def test_serialization_cycle_inference_mode(
        self,
        model_config: Dict[str, Any],
        sample_input: keras.KerasTensor
    ) -> None:
        """
        CRITICAL TEST: Full serialization cycle in inference mode.

        This test verifies that a model can be saved and loaded, and that
        the loaded model produces identical predictions to the original.
        """
        # Create and use original model
        model = PowerMLP(**model_config)
        original_prediction = model(sample_input, training=False)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_power_mlp.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input, training=False)

            # Verify identical predictions
            np.testing.assert_allclose(
                ops.convert_to_numpy(original_prediction),
                ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6,
                atol=1e-6,
                err_msg="Predictions differ after serialization in inference mode"
            )

    def test_serialization_cycle_with_training(
        self,
        model_config: Dict[str, Any]
    ) -> None:
        """Test serialization cycle after training."""
        # Create simple dataset
        x_train = keras.random.normal(shape=(32, 50))
        y_train = ops.convert_to_tensor(np.random.randint(0, 10, size=(32,)))

        # Create, compile, and train model
        model = PowerMLP(**model_config)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        model.fit(x_train, y_train, epochs=2, verbose=0)

        # Get prediction before save
        x_test = keras.random.normal(shape=(4, 50))
        original_prediction = model(x_test, training=False)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'trained_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_prediction = loaded_model(x_test, training=False)

            # Verify identical predictions
            np.testing.assert_allclose(
                ops.convert_to_numpy(original_prediction),
                ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6,
                atol=1e-6,
                err_msg="Predictions differ after serialization of trained model"
            )

    def test_save_model_method(self, model_config: Dict[str, Any]) -> None:
        """Test the save_model convenience method."""
        model = PowerMLP(**model_config)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'model_via_method.keras')
            model.save_model(filepath)

            assert os.path.exists(filepath)

            # Verify it can be loaded
            loaded_model = keras.models.load_model(filepath)
            assert isinstance(loaded_model, PowerMLP)

    def test_load_model_method(self, model_config: Dict[str, Any]) -> None:
        """Test the load_model class method."""
        model = PowerMLP(**model_config)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'model_for_loading.keras')
            model.save(filepath)

            loaded_model = PowerMLP.load_model(filepath)
            assert isinstance(loaded_model, PowerMLP)
            assert loaded_model.hidden_units == model.hidden_units


class TestPowerMLPConfiguration:
    """Test suite for configuration and serialization."""

    @pytest.fixture
    def model_config(self) -> Dict[str, Any]:
        """Complete configuration for testing."""
        return {
            'hidden_units': [100, 64, 32, 10],
            'k': 4,
            'dropout_rate': 0.3,
            'batch_normalization': True,
            'output_activation': 'relu',
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'use_bias': False
        }

    def test_get_config_completeness(self, model_config: Dict[str, Any]) -> None:
        """Test that get_config contains all constructor parameters."""
        model = PowerMLP(**model_config)
        config = model.get_config()

        # Check all important parameters are present
        assert 'hidden_units' in config
        assert 'k' in config
        assert 'dropout_rate' in config
        assert 'batch_normalization' in config
        assert 'output_activation' in config
        assert 'kernel_initializer' in config
        assert 'bias_initializer' in config
        assert 'use_bias' in config

        # Verify values match
        assert config['hidden_units'] == model_config['hidden_units']
        assert config['k'] == model_config['k']
        assert config['dropout_rate'] == model_config['dropout_rate']
        assert config['batch_normalization'] == model_config['batch_normalization']
        assert config['use_bias'] == model_config['use_bias']

    def test_from_config(self, model_config: Dict[str, Any]) -> None:
        """Test reconstruction from configuration."""
        original_model = PowerMLP(**model_config)
        config = original_model.get_config()

        # Reconstruct model from config
        reconstructed_model = PowerMLP.from_config(config)

        assert reconstructed_model.hidden_units == original_model.hidden_units
        assert reconstructed_model.k == original_model.k
        assert reconstructed_model.dropout_rate == original_model.dropout_rate
        assert reconstructed_model.batch_normalization == original_model.batch_normalization

    def test_config_with_regularizers(self) -> None:
        """Test configuration with regularizers."""
        model = PowerMLP(
            hidden_units=[10, 8, 5],
            kernel_regularizer='l2',
            bias_regularizer='l1'
        )
        config = model.get_config()

        assert 'kernel_regularizer' in config
        assert 'bias_regularizer' in config


class TestPowerMLPGradients:
    """Test suite for gradient computation and backpropagation."""

    @pytest.fixture
    def model(self) -> PowerMLP:
        """Create model for gradient tests."""
        return PowerMLP(
            hidden_units=[10, 16, 8, 3],
            k=3
        )

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Create sample input."""
        return ops.convert_to_tensor(
            np.random.randn(4, 10).astype('float32')
        )

    def test_gradients_flow(
        self,
        model: PowerMLP,
        sample_input: keras.KerasTensor
    ) -> None:
        """Test that gradients flow through the model."""
        with tf.GradientTape() as tape:
            output = model(sample_input, training=True)
            loss = ops.mean(ops.square(output))

        gradients = tape.gradient(loss, model.trainable_variables)

        # Check all gradients exist
        assert all(g is not None for g in gradients)
        assert len(gradients) > 0

        # Check gradients have proper shapes
        for var, grad in zip(model.trainable_variables, gradients):
            assert grad.shape == var.shape

    def test_gradients_with_regularization(self) -> None:
        """Test gradients with kernel regularization."""
        model = PowerMLP(
            hidden_units=[10, 8, 5],
            kernel_regularizer='l2'
        )
        x = ops.convert_to_tensor(np.random.randn(4, 10).astype('float32'))

        with tf.GradientTape() as tape:
            output = model(x, training=True)
            loss = ops.mean(ops.square(output))

        gradients = tape.gradient(loss, model.trainable_variables)
        assert all(g is not None for g in gradients)

    def test_training_updates_weights(self) -> None:
        """Test that training actually updates weights."""
        model = PowerMLP(hidden_units=[10, 8, 5])
        model.compile(optimizer='sgd', loss='mse')

        # Get initial weights
        initial_weights = [w.numpy().copy() for w in model.trainable_variables]

        # Train for one step
        x = keras.random.normal(shape=(8, 10))
        y = keras.random.normal(shape=(8, 5))
        model.fit(x, y, epochs=1, verbose=0)

        # Get updated weights
        updated_weights = [w.numpy() for w in model.trainable_variables]

        # Verify weights changed
        for initial, updated in zip(initial_weights, updated_weights):
            assert not np.allclose(initial, updated)


class TestPowerMLPTrainingModes:
    """Test suite for different training modes."""

    @pytest.fixture
    def model(self) -> PowerMLP:
        """Create model with dropout and batch norm for mode testing."""
        return PowerMLP(
            hidden_units=[10, 16, 5],
            dropout_rate=0.5,
            batch_normalization=True
        )

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Create sample input."""
        return keras.random.normal(shape=(8, 10))

    @pytest.mark.parametrize("training", [True, False, None])
    def test_different_training_modes(
        self,
        model: PowerMLP,
        sample_input: keras.KerasTensor,
        training: bool
    ) -> None:
        """Test model behavior in different training modes."""
        output = model(sample_input, training=training)

        assert output.shape == (8, 5)
        assert not ops.any(ops.isnan(output))
        assert not ops.any(ops.isinf(output))

    def test_dropout_affects_training(self) -> None:
        """Test that dropout behaves differently in training vs inference."""
        model = PowerMLP(
            hidden_units=[50, 32, 10],
            dropout_rate=0.5
        )
        x = keras.random.normal(shape=(1, 50))

        # Get multiple predictions in training mode
        train_outputs = [model(x, training=True) for _ in range(10)]

        # Get multiple predictions in inference mode
        infer_outputs = [model(x, training=False) for _ in range(10)]

        # Training outputs should vary (due to dropout)
        train_np = [ops.convert_to_numpy(o) for o in train_outputs]
        assert not all(np.allclose(train_np[0], o) for o in train_np[1:])

        # Inference outputs should be identical (no dropout)
        infer_np = [ops.convert_to_numpy(o) for o in infer_outputs]
        assert all(np.allclose(infer_np[0], o, atol=1e-6) for o in infer_np[1:])


class TestPowerMLPVariants:
    """Test suite for pre-configured model variants."""

    @pytest.mark.parametrize("variant", ["micro", "tiny", "small", "base", "large", "xlarge"])
    def test_variant_creation(self, variant: str) -> None:
        """Test that all variants can be created."""
        model = PowerMLP.from_variant(
            variant=variant,
            num_classes=10,
            input_dim=784
        )

        assert isinstance(model, PowerMLP)
        assert model.hidden_units[0] == 784  # Input dim
        assert model.hidden_units[-1] == 10  # Num classes
        assert model.k in [2, 3, 4]  # Valid k values

    def test_variant_micro(self) -> None:
        """Test micro variant specifically."""
        model = PowerMLP.from_variant("micro", num_classes=10, input_dim=100)

        expected_hidden = [32, 16]
        assert model.hidden_units == [100] + expected_hidden + [10]
        assert model.k == 2

    def test_variant_base(self) -> None:
        """Test base variant specifically."""
        model = PowerMLP.from_variant("base", num_classes=5, input_dim=50)

        expected_hidden = [256, 128, 64]
        assert model.hidden_units == [50] + expected_hidden + [5]
        assert model.k == 3

    def test_variant_with_override(self) -> None:
        """Test variant with parameter override."""
        model = PowerMLP.from_variant(
            variant="small",
            num_classes=10,
            input_dim=784,
            k=5,  # Override k
            dropout_rate=0.3,  # Add dropout
            batch_normalization=True  # Add batch norm
        )

        assert model.k == 5  # Overridden value
        assert model.dropout_rate == 0.3
        assert model.batch_normalization

    def test_invalid_variant(self) -> None:
        """Test that invalid variant raises ValueError."""
        with pytest.raises(ValueError, match="Unknown variant"):
            PowerMLP.from_variant("invalid", num_classes=10, input_dim=100)


class TestPowerMLPHelperFunctions:
    """Test suite for helper factory functions."""

    def test_create_power_mlp(self) -> None:
        """Test create_power_mlp helper function."""
        model = create_power_mlp(
            hidden_units=[100, 64, 10],
            k=3,
            optimizer='adam',
            learning_rate=0.001,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        assert isinstance(model, PowerMLP)
        assert model.compiled
        assert model.optimizer is not None

    def test_create_power_mlp_regressor(self) -> None:
        """Test create_power_mlp_regressor helper function."""
        model = create_power_mlp_regressor(
            hidden_units=[100, 64, 1],
            k=3,
            learning_rate=0.001
        )

        assert isinstance(model, PowerMLP)
        assert model.compiled
        assert model.hidden_units[-1] == 1  # Regression output

        # Check loss is appropriate for regression
        assert hasattr(model, 'loss')

    def test_create_power_mlp_binary_classifier(self) -> None:
        """Test create_power_mlp_binary_classifier helper function."""
        model = create_power_mlp_binary_classifier(
            hidden_units=[100, 64, 1],
            k=3,
            learning_rate=0.001
        )

        assert isinstance(model, PowerMLP)
        assert model.compiled
        assert model.hidden_units[-1] == 1  # Binary output

        # Check activation is sigmoid
        assert model.output_layer.activation == keras.activations.sigmoid

    def test_compiled_model_can_fit(self) -> None:
        """Test that compiled models can be trained."""
        model = create_power_mlp(
            hidden_units=[20, 16, 5],
            k=2,
            dropout_rate=0.1,
            loss='sparse_categorical_crossentropy',
        )

        # Create dummy data
        x = keras.random.normal(shape=(32, 20))
        y = ops.convert_to_tensor(np.random.randint(0, 5, size=(32,)))

        # Train should work
        history = model.fit(x, y, epochs=2, verbose=0)
        assert 'loss' in history.history


class TestPowerMLPIntegration:
    """Integration tests for PowerMLP with Keras workflows."""

    def test_fit_and_evaluate(self) -> None:
        """Test full training and evaluation workflow."""
        model = PowerMLP(
            hidden_units=[20, 16, 8, 3],
            k=3,
            dropout_rate=0.1,
            output_activation='softmax'
        )
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Create dummy dataset
        x_train = keras.random.normal(shape=(64, 20))
        y_train = ops.convert_to_tensor(np.random.randint(0, 3, size=(64,)))
        x_val = keras.random.normal(shape=(16, 20))
        y_val = ops.convert_to_tensor(np.random.randint(0, 3, size=(16,)))

        # Train
        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=3,
            verbose=0
        )

        # Evaluate
        loss, accuracy = model.evaluate(x_val, y_val, verbose=0)

        assert 'loss' in history.history
        assert 'val_loss' in history.history
        assert isinstance(loss, float)
        assert isinstance(accuracy, float)

    def test_predict(self) -> None:
        """Test prediction functionality."""
        model = PowerMLP(
            hidden_units=[10, 8, 3],
            output_activation='softmax'
        )

        x = keras.random.normal(shape=(5, 10))
        predictions = model.predict(x, verbose=0)

        assert predictions.shape == (5, 3)
        # Check softmax properties
        assert np.allclose(np.sum(predictions, axis=-1), 1.0, atol=1e-6)

    def test_summary(self) -> None:
        """Test model summary generation."""
        model = PowerMLP(hidden_units=[100, 64, 10], k=3)

        # Build the model first
        model.build((None, 100))

        # Should not raise an error
        model.summary()

    def test_with_callbacks(self) -> None:
        """Test training with Keras callbacks."""
        model = create_power_mlp(
            hidden_units=[20, 16, 5],
            k=2,
            loss='sparse_categorical_crossentropy'
        )

        x = keras.random.normal(shape=(32, 20))
        y = ops.convert_to_tensor(np.random.randint(0, 5, size=(32,)))

        # Train with early stopping callback
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=2,
            restore_best_weights=True
        )

        history = model.fit(
            x, y,
            epochs=10,
            callbacks=[early_stopping],
            verbose=0
        )

        assert len(history.history['loss']) <= 10


class TestPowerMLPEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    def test_single_hidden_layer(self) -> None:
        """Test model with only one hidden layer."""
        model = PowerMLP(hidden_units=[10, 16, 5])

        x = keras.random.normal(shape=(4, 10))
        output = model(x)

        assert output.shape == (4, 5)

    def test_large_k_value(self) -> None:
        """Test model with large k value."""
        model = PowerMLP(hidden_units=[10, 8, 5], k=10)

        x = keras.random.normal(shape=(4, 10))
        output = model(x)

        assert output.shape == (4, 5)
        assert not ops.any(ops.isnan(output))

    def test_no_dropout(self) -> None:
        """Test model without dropout."""
        model = PowerMLP(
            hidden_units=[10, 8, 5],
            dropout_rate=0.0
        )

        assert all(layer is None for layer in model.dropout_layers)

    def test_no_batch_norm(self) -> None:
        """Test model without batch normalization."""
        model = PowerMLP(
            hidden_units=[10, 8, 5],
            batch_normalization=False
        )

        assert all(layer is None for layer in model.batch_norm_layers)

    def test_batch_size_one(self) -> None:
        """Test model with batch size of 1."""
        model = PowerMLP(hidden_units=[10, 8, 5])

        x = keras.random.normal(shape=(1, 10))
        output = model(x)

        assert output.shape == (1, 5)

    def test_large_batch_size(self) -> None:
        """Test model with large batch size."""
        model = PowerMLP(hidden_units=[10, 8, 5])

        x = keras.random.normal(shape=(1000, 10))
        output = model(x)

        assert output.shape == (1000, 5)

    def test_repr(self) -> None:
        """Test string representation."""
        model = PowerMLP(hidden_units=[10, 8, 5], k=3, dropout_rate=0.2)

        repr_str = repr(model)

        assert 'PowerMLP' in repr_str
        assert 'hidden_units' in repr_str
        assert 'k=3' in repr_str


class TestPowerMLPNumericalStability:
    """Test suite for numerical stability."""

    def test_no_nans_in_output(self) -> None:
        """Test that output doesn't contain NaN values."""
        model = PowerMLP(hidden_units=[10, 8, 5], k=5)

        x = keras.random.normal(shape=(4, 10))
        output = model(x)

        assert not ops.any(ops.isnan(output))

    def test_no_infs_in_output(self) -> None:
        """Test that output doesn't contain inf values."""
        model = PowerMLP(hidden_units=[10, 8, 5], k=5)

        x = keras.random.normal(shape=(4, 10))
        output = model(x)

        assert not ops.any(ops.isinf(output))

    def test_gradients_not_nan(self) -> None:
        """Test that gradients don't become NaN."""
        model = PowerMLP(hidden_units=[10, 8, 5], k=4)

        x = ops.convert_to_tensor(np.random.randn(4, 10).astype('float32'))

        with tf.GradientTape() as tape:
            output = model(x, training=True)
            loss = ops.mean(ops.square(output))

        gradients = tape.gradient(loss, model.trainable_variables)

        for grad in gradients:
            assert not ops.any(ops.isnan(grad))
            assert not ops.any(ops.isinf(grad))


# ---------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])