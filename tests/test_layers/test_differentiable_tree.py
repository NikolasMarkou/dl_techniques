"""
Comprehensive test suite for the DifferentiableTreeDense layer.

This test suite follows modern Keras 3 testing patterns, ensuring robust
validation of all layer functionality including initialization, forward pass
with different combination strategies, serialization, gradient flow, and
integration into a full model.
"""

import pytest
import tempfile
import os
import numpy as np
import keras
from typing import Any, Dict

# Import the layer to test
# You may need to adjust this import path based on your project structure
from dl_techniques.layers.differentiable_tree import DifferentiableTreeDense

# TensorFlow is used as the backend for gradient testing
import tensorflow as tf


class TestDifferentiableTreeDense:
    """Comprehensive unit tests for the DifferentiableTreeDense layer."""

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard configuration for testing."""
        return {
            'output_dim': 128,
            'leaf_output_dim': 32,
            'activation': 'relu',
            'kernel_initializer': 'orthogonal'
        }

    @pytest.fixture
    def sample_inputs(self) -> keras.KerasTensor:
        """Sample features tensor for testing."""
        batch_size = 8
        feature_dim = 64
        return keras.random.normal(shape=(batch_size, feature_dim))

    def test_initialization(self, layer_config):
        """Test layer initialization with valid parameters."""
        layer = DifferentiableTreeDense(**layer_config, leaf_combination='concat')
        assert layer.output_dim == layer_config['output_dim']
        assert layer.leaf_output_dim == layer_config['leaf_output_dim']
        assert layer.leaf_combination == 'concat'
        assert layer.num_leaves == 4  # 128 / 32
        assert layer.num_internal_nodes == 3
        # FIX: Check function name, not class instance
        assert layer.activation.__name__ == 'relu'
        assert isinstance(layer.kernel_initializer, keras.initializers.Orthogonal)
        assert not layer.built

    def test_initialization_defaults(self):
        """Test layer initialization with default parameters."""
        layer = DifferentiableTreeDense(output_dim=100)
        assert layer.leaf_output_dim == 128
        assert layer.leaf_combination == 'concat'
        # FIX: Default activation is 'linear', not None
        assert layer.activation.__name__ == 'linear'
        assert isinstance(layer.kernel_initializer, keras.initializers.GlorotUniform)

    @pytest.mark.parametrize("leaf_combination", ['concat', 'dense'])
    def test_forward_pass_and_build(self, layer_config, sample_inputs, leaf_combination):
        """Test forward pass, building, and output shape for both strategies."""
        layer = DifferentiableTreeDense(**layer_config, leaf_combination=leaf_combination)
        output = layer(sample_inputs)

        # Test build status and weight creation
        assert layer.built
        assert layer.routing_weights is not None
        assert layer.leaf_weights is not None
        assert layer.leaf_biases is not None
        if leaf_combination == 'dense':
            assert layer.combination_layer is not None
            assert layer.combination_layer.built
        else:
            assert layer.combination_layer is None

        # Test output shape
        expected_shape = layer.compute_output_shape(sample_inputs.shape)
        assert output.shape == expected_shape

        # Test output content
        output_np = keras.ops.convert_to_numpy(output)
        assert np.all(np.isfinite(output_np))
        if layer_config['activation'] == 'relu':
            assert np.all(output_np >= 0)

    @pytest.mark.parametrize("leaf_combination", ['concat', 'dense'])
    def test_serialization_cycle(self, layer_config, sample_inputs, leaf_combination):
        """CRITICAL TEST: Full serialization and deserialization cycle."""
        feature_dim = sample_inputs.shape[-1]

        inputs = keras.Input(shape=(feature_dim,))
        outputs = DifferentiableTreeDense(
            **layer_config, leaf_combination=leaf_combination
        )(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)

        original_output = model(sample_inputs)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)
            loaded_model = keras.models.load_model(filepath)

            loaded_output = loaded_model(sample_inputs)

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_output),
                keras.ops.convert_to_numpy(loaded_output),
                rtol=1e-6, atol=1e-6
            )

    def test_config_completeness(self, layer_config):
        """Test that get_config contains all __init__ parameters."""
        layer = DifferentiableTreeDense(**layer_config)
        config = layer.get_config()
        for key in layer_config:
            assert key in config
        # Check nested serialization
        assert config['kernel_initializer']['class_name'].lower() == 'orthogonal'

    @pytest.mark.parametrize("leaf_combination", ['concat', 'dense'])
    def test_gradients_flow(self, layer_config, sample_inputs, leaf_combination):
        """Test that gradients flow through all trainable weights."""
        layer = DifferentiableTreeDense(**layer_config, leaf_combination=leaf_combination)
        features = tf.convert_to_tensor(keras.ops.convert_to_numpy(sample_inputs))

        with tf.GradientTape() as tape:
            tape.watch(features)
            output = layer(features)
            # Use sum as a dummy loss for gradient calculation
            loss = tf.reduce_sum(output)

        sources_to_grad = layer.trainable_variables + [features]
        all_grads = tape.gradient(loss, sources_to_grad)
        trainable_grads = all_grads[:-1]
        input_grads = all_grads[-1]

        # Assertions for trainable_variables' gradients
        expected_num_weights = 3 if leaf_combination == 'concat' else 5  # +2 for dense layer
        assert len(trainable_grads) == expected_num_weights

        for grad in trainable_grads:
            assert grad is not None
            grad_np = grad.numpy()
            assert np.all(np.isfinite(grad_np))
            assert np.any(grad_np != 0)

        # Assertions for input features' gradients
        assert input_grads is not None
        assert np.any(input_grads.numpy() != 0)

    def test_edge_cases_initialization(self):
        """Test error conditions during layer initialization."""
        with pytest.raises(ValueError, match="output_dim must be positive"):
            DifferentiableTreeDense(output_dim=0)
        with pytest.raises(ValueError, match="leaf_output_dim must be positive"):
            DifferentiableTreeDense(output_dim=10, leaf_output_dim=-1)
        with pytest.raises(ValueError, match="leaf_combination must be one of"):
            DifferentiableTreeDense(output_dim=10, leaf_combination='invalid')

    def test_deterministic_behavior(self, layer_config, sample_inputs):
        """Test that the layer produces consistent outputs for the same input."""
        layer = DifferentiableTreeDense(**layer_config)
        output1 = layer(sample_inputs)
        output2 = layer(sample_inputs)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output1),
            keras.ops.convert_to_numpy(output2),
            rtol=1e-6, atol=1e-6
        )

    def test_numerical_stability(self, layer_config):
        """Test with extreme feature values."""
        layer = DifferentiableTreeDense(**layer_config)
        batch_size, feature_dim = 4, 64

        large_features = keras.ops.ones((batch_size, feature_dim)) * 1000
        output_large = layer(large_features)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(output_large)))

        small_features = keras.ops.ones((batch_size, feature_dim)) * -1000
        output_small = layer(small_features)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(output_small)))


class TestDifferentiableTreeDenseIntegration:
    """Integration tests for the layer in a complete model."""

    @pytest.mark.parametrize("leaf_combination", ['concat', 'dense'])
    def test_in_model_compilation_and_training(self, leaf_combination):
        """Test the layer in a complete, trainable model."""
        input_dim = 256
        feature_dim = 128
        output_dim = 512

        inputs = keras.Input(shape=(input_dim,))
        x = keras.layers.Dense(feature_dim, activation='relu')(inputs)
        tree_output = DifferentiableTreeDense(
            output_dim=output_dim,
            leaf_output_dim=64,
            leaf_combination=leaf_combination
        )(x)
        final_output = keras.layers.Dense(10, name="final_dense")(tree_output)  # A final head

        model = keras.Model(inputs=inputs, outputs=final_output)
        model.compile(optimizer='adam', loss='mse')

        batch_size = 16
        dummy_x = np.random.rand(batch_size, input_dim)
        dummy_y = np.random.rand(batch_size, 10)

        history = model.fit(dummy_x, dummy_y, epochs=1, verbose=0)

        assert 'loss' in history.history
        assert len(history.history['loss']) == 1
        assert history.history['loss'][0] > 0

        # Verify output shape is as expected after training
        pred = model.predict(dummy_x, verbose=0)
        assert pred.shape == (batch_size, 10)