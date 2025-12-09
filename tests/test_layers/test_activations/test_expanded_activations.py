"""
Comprehensive test suite for expanded activation functions.

This module provides thorough testing of all activation layers following
the modern Keras 3 testing patterns from the dl-techniques framework guide.
"""

import pytest
import tempfile
import os
import numpy as np
import keras
from typing import Any, Dict, Type
import tensorflow as tf

from dl_techniques.layers.activations.expanded_activations import (
    BaseActivation, GELU, SiLU, ExpandedActivation,
    xATLU, xGELU, xSiLU, EluPlusOne, get_activation,
    elu_plus_one_plus_epsilon
)


class TestBaseActivation:
    """Test suite for BaseActivation class."""

    @pytest.fixture
    def base_activation(self) -> BaseActivation:
        """Create a BaseActivation instance for testing."""
        return BaseActivation()

    def test_initialization(self, base_activation):
        """Test BaseActivation initialization."""
        assert hasattr(base_activation, 'trainable')
        assert base_activation.trainable is True
        assert not base_activation.built

    def test_config_serialization(self, base_activation):
        """Test configuration serialization."""
        config = base_activation.get_config()
        assert isinstance(config, dict)
        assert 'name' in config
        assert 'trainable' in config


class TestSimpleActivations:
    """Test suite for simple activation functions (GELU, SiLU)."""

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Create sample input tensor for testing."""
        return keras.random.normal(shape=(4, 32))

    @pytest.mark.parametrize("activation_class", [GELU, SiLU, EluPlusOne])
    def test_initialization(self, activation_class: Type[BaseActivation]):
        """Test activation initialization."""
        layer = activation_class()
        assert hasattr(layer, 'trainable')
        assert not layer.built

    @pytest.mark.parametrize("activation_class", [GELU, SiLU, EluPlusOne])
    def test_forward_pass(self, activation_class: Type[BaseActivation], sample_input):
        """Test forward pass and building."""
        layer = activation_class()
        output = layer(sample_input)

        assert layer.built
        assert output.shape == sample_input.shape
        assert keras.ops.all(keras.ops.isfinite(output))

    @pytest.mark.parametrize("activation_class", [GELU, SiLU, EluPlusOne])
    def test_serialization_cycle(self, activation_class: Type[BaseActivation], sample_input):
        """CRITICAL TEST: Full serialization cycle."""
        # Create model with custom layer
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = activation_class()(inputs)
        model = keras.Model(inputs, outputs)

        # Get original prediction
        original_pred = model(sample_input)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input)

            # Verify identical predictions
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization"
            )

    @pytest.mark.parametrize("activation_class", [GELU, SiLU, EluPlusOne])
    def test_config_completeness(self, activation_class: Type[BaseActivation]):
        """Test that get_config contains all necessary parameters."""
        layer = activation_class()
        config = layer.get_config()

        # Basic Layer configuration should be present
        assert 'name' in config
        assert 'trainable' in config
        assert 'dtype' in config


class TestExpandedActivations:
    """Test suite for expanded activation functions with trainable alpha."""

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Create sample input tensor for testing."""
        return keras.random.normal(shape=(4, 32))

    @pytest.fixture
    def activation_configs(self) -> Dict[str, Dict[str, Any]]:
        """Configuration for different expanded activations."""
        return {
            'default': {},
            'with_regularizer': {
                'alpha_regularizer': keras.regularizers.L2(1e-4)
            },
            'with_constraint': {
                'alpha_constraint': keras.constraints.NonNeg()
            },
            'custom_init': {
                'alpha_initializer': keras.initializers.Constant(0.1)
            }
        }

    @pytest.mark.parametrize("activation_class", [xATLU, xGELU, xSiLU])
    def test_initialization(self, activation_class: Type[ExpandedActivation]):
        """Test expanded activation initialization."""
        layer = activation_class()

        assert hasattr(layer, 'alpha_initializer')
        assert hasattr(layer, 'alpha_regularizer')
        assert hasattr(layer, 'alpha_constraint')
        assert layer.alpha is None  # Not built yet
        assert not layer.built

    @pytest.mark.parametrize("activation_class", [xATLU, xGELU, xSiLU])
    def test_forward_pass(self, activation_class: Type[ExpandedActivation], sample_input):
        """Test forward pass and building."""
        layer = activation_class()
        output = layer(sample_input)

        assert layer.built
        assert layer.alpha is not None  # Alpha weight created
        assert output.shape == sample_input.shape
        assert keras.ops.all(keras.ops.isfinite(output))

    @pytest.mark.parametrize("activation_class", [xATLU, xGELU, xSiLU])
    @pytest.mark.parametrize("config_name", ['default', 'with_regularizer', 'custom_init'])
    def test_serialization_cycle(
            self,
            activation_class: Type[ExpandedActivation],
            activation_configs,
            config_name: str,
            sample_input
    ):
        """CRITICAL TEST: Full serialization cycle with different configurations."""
        config = activation_configs[config_name]

        # Create model with custom layer
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = activation_class(**config)(inputs)
        model = keras.Model(inputs, outputs)

        # Get original prediction
        original_pred = model(sample_input)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input)

            # Verify identical predictions
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg=f"Predictions differ after serialization for config {config_name}"
            )

    @pytest.mark.parametrize("activation_class", [xATLU, xGELU, xSiLU])
    def test_config_completeness(self, activation_class: Type[ExpandedActivation]):
        """Test that get_config contains all __init__ parameters."""
        layer = activation_class(
            alpha_regularizer=keras.regularizers.L2(1e-4),
            alpha_constraint=keras.constraints.NonNeg()
        )
        config = layer.get_config()

        # Check all custom parameters are present
        assert 'alpha_initializer' in config
        assert 'alpha_regularizer' in config
        assert 'alpha_constraint' in config

    @pytest.mark.parametrize("activation_class", [xATLU, xGELU, xSiLU])
    def test_alpha_learning(self, activation_class: Type[ExpandedActivation], sample_input):
        """Test that alpha parameter can be updated during training."""
        layer = activation_class()

        # Build layer
        output = layer(sample_input)

        # Get initial alpha value
        initial_alpha = keras.ops.convert_to_numpy(layer.alpha)

        # Create a simple training setup
        with tf.GradientTape() as tape:
            output = layer(sample_input, training=True)
            loss = keras.ops.mean(keras.ops.square(output))

        # Compute and apply gradients
        gradients = tape.gradient(loss, [layer.alpha])
        assert gradients[0] is not None

        # Apply a simple gradient step manually
        layer.alpha.assign_sub(0.01 * gradients[0])

        # Verify alpha changed
        updated_alpha = keras.ops.convert_to_numpy(layer.alpha)
        assert not np.allclose(initial_alpha, updated_alpha, rtol=1e-7, atol=1e-7)

    @pytest.mark.parametrize("activation_class", [xATLU, xGELU, xSiLU])
    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(
            self,
            activation_class: Type[ExpandedActivation],
            sample_input,
            training: bool
    ):
        """Test behavior in different training modes."""
        layer = activation_class()

        output = layer(sample_input, training=training)
        assert output.shape == sample_input.shape


class TestFactoryFunction:
    """Test suite for get_activation factory function."""

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Create sample input tensor for testing."""
        return keras.random.normal(shape=(4, 32))

    @pytest.mark.parametrize("activation_name,expected_class", [
        ('gelu', GELU),
        ('silu', SiLU),
        ('xatlu', xATLU),
        ('xgelu', xGELU),
        ('xsilu', xSiLU),
        ('elu_plus_one', EluPlusOne),
    ])
    def test_factory_creation(self, activation_name: str, expected_class: Type[BaseActivation]):
        """Test factory function creates correct activation."""
        activation = get_activation(activation_name)
        assert isinstance(activation, expected_class)

    @pytest.mark.parametrize("activation_name", [
        'GELU', 'SiLU', 'xGELU',  # Test case insensitive
        ' gelu ', ' silu '  # Test whitespace handling
    ])
    def test_case_insensitive_and_whitespace(self, activation_name: str):
        """Test factory handles case insensitive input and whitespace."""
        activation = get_activation(activation_name)
        assert isinstance(activation, BaseActivation)

    def test_unknown_activation_raises_error(self):
        """Test that unknown activation names raise ValueError."""
        with pytest.raises(ValueError, match="Unknown activation"):
            get_activation("unknown_activation")

    def test_factory_serialization(self, sample_input):
        """Test that factory-created activations can be serialized."""
        activation = get_activation("xgelu")

        # Create model
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = activation(inputs)
        model = keras.Model(inputs, outputs)

        # Test serialization
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'factory_test.keras')
            model.save(filepath)
            loaded_model = keras.models.load_model(filepath)

            original_pred = model(sample_input)
            loaded_pred = loaded_model(sample_input)

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6
            )


class TestHelperFunctions:
    """Test suite for helper functions."""

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Create sample input tensor for testing."""
        return keras.random.normal(shape=(4, 32))

    def test_elu_plus_one_plus_epsilon_positive_output(self, sample_input):
        """Test that elu_plus_one_plus_epsilon always produces positive outputs."""
        output = elu_plus_one_plus_epsilon(sample_input)

        # All outputs should be positive
        assert keras.ops.all(output > 0)

    def test_elu_plus_one_plus_epsilon_shape(self, sample_input):
        """Test that elu_plus_one_plus_epsilon preserves input shape."""
        output = elu_plus_one_plus_epsilon(sample_input)
        assert output.shape == sample_input.shape


class TestEdgeCases:
    """Test suite for edge cases and error conditions."""

    def test_large_input_values(self):
        """Test activations with very large input values."""
        large_input = keras.ops.ones((2, 16)) * 100.0

        activations = [GELU(), SiLU(), xGELU(), xSiLU(), xATLU()]

        for activation in activations:
            output = activation(large_input)
            assert keras.ops.all(keras.ops.isfinite(output))

    def test_small_input_values(self):
        """Test activations with very small input values."""
        small_input = keras.ops.ones((2, 16)) * 1e-8

        activations = [GELU(), SiLU(), xGELU(), xSiLU(), xATLU()]

        for activation in activations:
            output = activation(small_input)
            assert keras.ops.all(keras.ops.isfinite(output))

    def test_negative_input_values(self):
        """Test activations with negative input values."""
        negative_input = keras.ops.ones((2, 16)) * -10.0

        activations = [GELU(), SiLU(), xGELU(), xSiLU(), xATLU()]

        for activation in activations:
            output = activation(negative_input)
            assert keras.ops.all(keras.ops.isfinite(output))

    def test_zero_input_values(self):
        """Test activations with zero input values."""
        zero_input = keras.ops.zeros((2, 16))

        activations = [GELU(), SiLU(), xGELU(), xSiLU(), xATLU()]

        for activation in activations:
            output = activation(zero_input)
            assert keras.ops.all(keras.ops.isfinite(output))


class TestIntegrationWithModels:
    """Integration tests with common model architectures."""

    @pytest.fixture
    def sample_data(self) -> tuple:
        """Create sample classification data."""
        x = keras.random.normal(shape=(100, 32))
        y = keras.ops.cast(
            keras.random.uniform(shape=(100,), minval=0, maxval=2),
            dtype='int32'
        )
        return x, y

    @pytest.mark.parametrize("activation_name", ['gelu', 'xgelu', 'silu', 'xsilu'])
    def test_in_dense_model(self, activation_name: str, sample_data):
        """Test activations in a simple dense model."""
        x, y = sample_data
        activation = get_activation(activation_name)

        model = keras.Sequential([
            keras.layers.Dense(64),
            activation,
            keras.layers.Dense(32),
            activation,
            keras.layers.Dense(2, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Test training
        history = model.fit(x, y, epochs=1, verbose=0)
        assert len(history.history['loss']) == 1

        # Test prediction
        predictions = model.predict(x[:10], verbose=0)
        assert predictions.shape == (10, 2)
        assert keras.ops.all(keras.ops.sum(predictions, axis=-1) - 1.0 < 1e-5)  # Softmax sums to 1


def debug_layer_serialization(layer_class, layer_config, sample_input):
    """
    Debug helper for layer serialization issues.

    This function helps identify issues with custom layer serialization
    by testing each step of the process.
    """
    from dl_techniques.utils.logger import logger

    try:
        # Test basic functionality
        layer = layer_class(**layer_config)
        output = layer(sample_input)
        logger.info(f"✅ Forward pass successful: {output.shape}")

        # Test configuration
        config = layer.get_config()
        logger.info(f"✅ Configuration keys: {list(config.keys())}")

        # Test serialization
        inputs = keras.Input(shape=sample_input.shape[1:])
        outputs = layer_class(**layer_config)(inputs)
        model = keras.Model(inputs, outputs)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(os.path.join(tmpdir, 'test.keras'))
            loaded = keras.models.load_model(os.path.join(tmpdir, 'test.keras'))
            logger.info("✅ Serialization test passed")

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])