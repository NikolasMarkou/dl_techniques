"""
Comprehensive test suite for the DifferentiableSoftmaxTree layer.

This test suite follows modern Keras 3 testing patterns, ensuring robust
validation of all layer functionality including initialization, forward pass,
serialization, gradient flow, and integration into a full model.
"""

import pytest
import tempfile
import os
import numpy as np
import keras
from typing import Any, Dict

# Import the layer to test
from dl_techniques.layers.activations.softmax_tree import DifferentiableSoftmaxTree

# TensorFlow is used as the backend for gradient testing
import tensorflow as tf


class TestDifferentiableSoftmaxTree:
    """Comprehensive test suite for the DifferentiableSoftmaxTree layer."""

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard configuration for testing."""
        return {
            'num_classes': 128,
            'feature_dim': 64,
            'initializer': 'orthogonal'
        }

    @pytest.fixture
    def sample_inputs(self, layer_config) -> Dict[str, keras.KerasTensor]:
        """Sample features and targets for testing."""
        batch_size = 8
        features = keras.random.normal(shape=(batch_size, layer_config['feature_dim']))
        targets = keras.random.randint(
            shape=(batch_size, 1),
            minval=0,
            maxval=layer_config['num_classes']
        )
        return {'features': features, 'targets': targets}

    def test_initialization(self, layer_config):
        """Test layer initialization with valid parameters."""
        layer = DifferentiableSoftmaxTree(**layer_config)
        assert layer.num_classes == layer_config['num_classes']
        assert layer.feature_dim == layer_config['feature_dim']
        assert isinstance(layer.initializer, keras.initializers.Orthogonal)
        assert hasattr(layer, 'path_nodes_map')
        assert hasattr(layer, 'path_directions_map')
        assert not layer.built
        assert layer.node_weights is None

    def test_initialization_defaults(self):
        """Test layer initialization with default parameters."""
        layer = DifferentiableSoftmaxTree(num_classes=100, feature_dim=32)
        assert layer.num_classes == 100
        assert layer.feature_dim == 32
        assert isinstance(layer.initializer, keras.initializers.GlorotUniform)

    def test_forward_pass_and_build(self, layer_config, sample_inputs):
        """Test forward pass, building, and output shape."""
        layer = DifferentiableSoftmaxTree(**layer_config)
        features, targets = sample_inputs['features'], sample_inputs['targets']
        loss = layer([features, targets])
        assert layer.built
        assert layer.node_weights is not None
        assert loss.shape == (features.shape[0],)
        loss_np = keras.ops.convert_to_numpy(loss)
        assert np.all(loss_np >= 0)
        assert np.all(np.isfinite(loss_np))

    def test_serialization_cycle(self, layer_config, sample_inputs):
        """CRITICAL TEST: Full serialization and deserialization cycle."""
        features, targets = sample_inputs['features'], sample_inputs['targets']
        input_features = keras.Input(shape=(layer_config['feature_dim'],), name="features")
        input_targets = keras.Input(shape=(1,), dtype="int32", name="targets")
        outputs = DifferentiableSoftmaxTree(**layer_config)([input_features, input_targets])
        model = keras.Model(inputs=[input_features, input_targets], outputs=outputs)
        original_loss = model([features, targets])

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)
            loaded_model = keras.models.load_model(filepath)
            loaded_loss = loaded_model([features, targets])
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_loss),
                keras.ops.convert_to_numpy(loaded_loss),
                rtol=1e-6, atol=1e-6
            )

    def test_config_completeness(self, layer_config):
        """Test that get_config contains all __init__ parameters."""
        layer = DifferentiableSoftmaxTree(**layer_config)
        config = layer.get_config()
        for key in layer_config:
            assert key in config
        assert config['num_classes'] == layer_config['num_classes']
        assert config['feature_dim'] == layer_config['feature_dim']
        assert config['initializer']['class_name'].lower() == layer_config['initializer']

    def test_gradients_flow(self, layer_config, sample_inputs):
        """Test that gradients flow through the layer's trainable weights."""
        layer = DifferentiableSoftmaxTree(**layer_config)
        features = tf.convert_to_tensor(keras.ops.convert_to_numpy(sample_inputs['features']))
        targets = tf.convert_to_tensor(keras.ops.convert_to_numpy(sample_inputs['targets']))

        with tf.GradientTape() as tape:
            # Need to watch the features to check gradient flow back to inputs
            tape.watch(features)
            loss_output = layer([features, targets])
            total_loss = tf.reduce_sum(loss_output)

        # --- START FIX: Calculate all gradients in a single, efficient call ---
        # Define all sources for the gradient calculation
        sources_to_grad = layer.trainable_variables + [features]

        # Calculate all gradients at once
        all_grads = tape.gradient(total_loss, sources_to_grad)

        # Unpack the results
        trainable_grads = all_grads[:-1]
        input_grads = all_grads[-1]
        # --- END FIX ---

        # Assertions for trainable_variables' gradients
        assert len(trainable_grads) == 1
        assert trainable_grads[0] is not None

        gradient_object = trainable_grads[0]
        if isinstance(gradient_object, tf.IndexedSlices):
            grad_values = gradient_object.values
            assert grad_values is not None, "IndexedSlices should have values."
            grad_np = grad_values.numpy()
        else:
            grad_np = gradient_object.numpy()

        assert np.all(np.isfinite(grad_np)), "All gradients should be finite"
        assert np.any(grad_np != 0), "At least some gradients should be non-zero"

        # Assertions for input features' gradients
        assert input_grads is not None
        assert np.any(input_grads.numpy() != 0)


    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, layer_config, sample_inputs, training):
        """Test behavior in different training modes (should be identical)."""
        layer = DifferentiableSoftmaxTree(**layer_config)
        features, targets = sample_inputs['features'], sample_inputs['targets']
        loss = layer([features, targets], training=training)
        assert loss.shape == (features.shape[0],)
        assert np.all(keras.ops.convert_to_numpy(loss) >= 0)

    def test_edge_cases_initialization(self):
        """Test error conditions during layer initialization."""
        with pytest.raises(ValueError, match="num_classes must be > 1"):
            DifferentiableSoftmaxTree(num_classes=1, feature_dim=32)
        with pytest.raises(ValueError, match="feature_dim must be positive"):
            DifferentiableSoftmaxTree(num_classes=10, feature_dim=0)

    def test_edge_cases_build(self, layer_config):
        """Test error conditions during the build process."""
        layer = DifferentiableSoftmaxTree(**layer_config)
        wrong_features = keras.random.normal(shape=(4, layer_config['feature_dim'] + 1))
        targets = keras.random.randint(shape=(4, 1), minval=0, maxval=layer_config['num_classes'])
        with pytest.raises(ValueError, match="does not match configured feature_dim"):
            layer([wrong_features, targets])
        with pytest.raises(ValueError, match="expects a list/tuple of two input shapes"):
            layer(wrong_features)

    def test_deterministic_behavior(self, layer_config, sample_inputs):
        """Test that the layer produces consistent outputs for the same input."""
        layer = DifferentiableSoftmaxTree(**layer_config)
        features, targets = sample_inputs['features'], sample_inputs['targets']
        layer([features, targets]) # Build
        loss1 = layer([features, targets])
        loss2 = layer([features, targets])
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(loss1),
            keras.ops.convert_to_numpy(loss2),
            rtol=1e-6, atol=1e-6
        )

    def test_numerical_stability(self, layer_config):
        """Test with extreme feature values."""
        layer = DifferentiableSoftmaxTree(**layer_config)
        batch_size = 4
        targets = keras.random.randint(shape=(batch_size, 1), minval=0, maxval=layer_config['num_classes'])
        large_features = keras.ops.ones((batch_size, layer_config['feature_dim'])) * 100
        loss_large = layer([large_features, targets])
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(loss_large)))
        small_features = keras.ops.ones((batch_size, layer_config['feature_dim'])) * -100
        loss_small = layer([small_features, targets])
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(loss_small)))

class TestDifferentiableSoftmaxTreeIntegration:
    """Integration tests for the layer in a complete model."""

    def test_in_classification_model_compilation_and_training(self):
        """Test the layer in a complete, trainable classification model."""
        num_classes, feature_dim, input_dim = 512, 128, 256

        input_data = keras.Input(shape=(input_dim,), name="input_data")
        target_indices = keras.Input(shape=(1,), dtype="int32", name="target_class")

        x = keras.layers.Dense(feature_dim * 2, activation='relu')(input_data)
        features = keras.layers.Dense(feature_dim, name="feature_extractor")(x)

        loss_output = DifferentiableSoftmaxTree(num_classes=num_classes, feature_dim=feature_dim)([features, target_indices])
        model = keras.Model(inputs=[input_data, target_indices], outputs=loss_output)

        model.compile(optimizer='adam', loss=lambda y_true, y_pred: y_pred)

        batch_size = 16
        dummy_x = keras.random.normal((batch_size, input_dim))
        dummy_y = keras.random.randint(shape=(batch_size, 1), minval=0, maxval=num_classes)
        dummy_y_placeholder = np.zeros(batch_size)

        history = model.fit([dummy_x, dummy_y], dummy_y_placeholder, epochs=1, verbose=0)

        assert 'loss' in history.history
        assert len(history.history['loss']) == 1
        assert history.history['loss'][0] > 0