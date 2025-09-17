"""
Comprehensive test suite for MobileNetV1 model implementation.

Following the Complete Guide to Modern Keras 3 Custom Layers and Models,
this test suite ensures a robust, serializable, and production-ready implementation
with thorough validation of all model functionality.

Test Categories:
- Serialization cycle tests (CRITICAL)
- Model initialization and building
- Forward pass validation
- Configuration completeness
- Gradient flow verification
- Training mode behavior
- Variant creation and validation
- Edge cases and error handling
- Performance benchmarks
"""

import pytest
import tempfile
import os
import numpy as np
from typing import Dict, Any

import keras
from keras import ops
import tensorflow as tf

from dl_techniques.models.mobilenet.mobilenet_v1 import MobileNetV1, create_mobilenetv1


class TestMobileNetV1:
    """Comprehensive test suite for MobileNetV1 model."""

    @pytest.fixture
    def default_config(self) -> Dict[str, Any]:
        """Standard configuration for testing."""
        return {
            'num_classes': 10,
            'width_multiplier': 1.0,
            'dropout_rate': 0.001,
            'weight_decay': 1e-5,
            'input_shape': (32, 32, 3),
            'include_top': True
        }

    @pytest.fixture
    def small_config(self) -> Dict[str, Any]:
        """Configuration for a smaller model."""
        return {
            'num_classes': 100,
            'width_multiplier': 0.25,
            'dropout_rate': 0.0,
            'weight_decay': 0.0,
            'input_shape': (64, 64, 3),
            'include_top': True
        }

    @pytest.fixture
    def sample_inputs(self) -> Dict[str, keras.KerasTensor]:
        """Sample inputs for different test scenarios."""
        return {
            'cifar': keras.random.normal(shape=(4, 32, 32, 3)),
            'imagenet': keras.random.normal(shape=(2, 224, 224, 3)),
            'custom': keras.random.normal(shape=(3, 64, 64, 3)),
            'grayscale': keras.random.normal(shape=(2, 28, 28, 1))
        }

    # ============================================================================
    # Core Tests - Initialization and Building
    # ============================================================================

    def test_initialization(self, default_config):
        """Test model initialization with standard parameters."""
        model = MobileNetV1(**default_config)
        assert model.num_classes == default_config['num_classes']
        assert model.width_multiplier == default_config['width_multiplier']
        assert model.include_top == default_config['include_top']
        assert model._input_shape == default_config['input_shape']

    def test_width_multiplier_scaling(self):
        """Test width multiplier correctly scales filter dimensions."""
        base_filters = 32  # From the initial conv layer
        multiplier = 0.5

        model = MobileNetV1(width_multiplier=multiplier, input_shape=(32, 32, 3))
        expected_filters = int(base_filters * multiplier)
        assert model.initial_conv.filters == expected_filters

        # Check a later block
        # Block 1 in ARCHITECTURE has 64 filters
        base_filters_block1 = 64
        expected_filters_block1 = int(base_filters_block1 * multiplier)
        assert model.depthwise_blocks[0].pointwise_conv.filters == expected_filters_block1

    def test_model_building_on_call(self, default_config, sample_inputs):
        """Test that model builds correctly on first call."""
        config_no_shape = default_config.copy()
        del config_no_shape['input_shape']
        model = MobileNetV1(**config_no_shape)
        assert not model.built

        # First call should trigger building
        output = model(sample_inputs['imagenet'])
        assert model.built
        assert output.shape == (2, default_config['num_classes'])

    # ============================================================================
    # Critical Serialization Tests
    # ============================================================================

    def test_serialization_cycle_standard(self, default_config, sample_inputs):
        """CRITICAL TEST: Full serialization cycle for a standard model."""
        model = MobileNetV1(**default_config)
        original_pred = model(sample_inputs['cifar'])

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'mobilenetv1_standard.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_inputs['cifar'])

            np.testing.assert_allclose(
                ops.convert_to_numpy(original_pred),
                ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Standard model predictions differ after serialization"
            )

    def test_serialization_cycle_small_variant(self, small_config, sample_inputs):
        """CRITICAL TEST: Full serialization cycle for a small variant model."""
        model = MobileNetV1(**small_config)
        original_pred = model(sample_inputs['custom'])

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'mobilenetv1_small.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_inputs['custom'])

            np.testing.assert_allclose(
                ops.convert_to_numpy(original_pred),
                ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Small variant model predictions differ after serialization"
            )

    def test_serialization_without_top(self, sample_inputs):
        """Test serialization of feature extractor (include_top=False)."""
        model = MobileNetV1(include_top=False, input_shape=(32, 32, 3))
        original_features = model(sample_inputs['cifar'])

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'mobilenetv1_features.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_features = loaded_model(sample_inputs['cifar'])

            np.testing.assert_allclose(
                ops.convert_to_numpy(original_features),
                ops.convert_to_numpy(loaded_features),
                rtol=1e-6, atol=1e-6,
                err_msg="Feature extractor outputs differ after serialization"
            )

    # ============================================================================
    # Configuration and Variant Tests
    # ============================================================================

    def test_config_completeness(self, default_config):
        """Test that get_config contains all initialization parameters."""
        model = MobileNetV1(**default_config)
        config = model.get_config()

        required_keys = [
            'num_classes', 'width_multiplier', 'dropout_rate',
            'weight_decay', 'kernel_initializer', 'include_top', 'input_shape'
        ]

        for key in required_keys:
            assert key in config, f"Missing '{key}' in get_config()"

    def test_from_config_reconstruction(self, default_config):
        """Test model reconstruction from configuration."""
        model = MobileNetV1(**default_config)
        config = model.get_config()

        reconstructed_model = MobileNetV1.from_config(config)

        assert reconstructed_model.num_classes == model.num_classes
        assert reconstructed_model.width_multiplier == model.width_multiplier
        assert reconstructed_model.dropout_rate == model.dropout_rate
        assert reconstructed_model.include_top == model.include_top

    @pytest.mark.parametrize("variant", ["1.0", "0.75", "0.5", "0.25"])
    def test_variant_creation(self, variant):
        """Test creation of all predefined variants."""
        model = MobileNetV1.from_variant(variant, num_classes=10, input_shape=(32, 32, 3))
        expected_multiplier = float(variant)
        assert model.width_multiplier == expected_multiplier

    def test_convenience_function(self):
        """Test create_mobilenetv1 convenience function."""
        model = create_mobilenetv1(
            variant="0.75",
            num_classes=10,
            input_shape=(32, 32, 3)
        )
        assert model.num_classes == 10
        assert model.width_multiplier == 0.75
        assert model._input_shape == (32, 32, 3)

    # ============================================================================
    # Forward Pass and Output Shape Tests
    # ============================================================================

    def test_forward_pass_shapes(self, sample_inputs):
        """Test forward pass produces correct output shapes."""
        # With classification head
        model = MobileNetV1.from_variant("1.0", num_classes=10, input_shape=(32, 32, 3))
        output = model(sample_inputs['cifar'])
        assert output.shape == (4, 10)

        # As a feature extractor
        model_no_top = MobileNetV1.from_variant("1.0", include_top=False, input_shape=(32, 32, 3))
        features = model_no_top(sample_inputs['cifar'])
        assert len(features.shape) == 2  # (batch, channels) after GlobalAveragePooling

    @pytest.mark.parametrize("input_shape, sample_shape", [
        ((28, 28, 1), (2, 28, 28, 1)),   # Grayscale
        ((64, 64, 3), (3, 64, 64, 3)),   # Custom size
        ((224, 224, 3), (1, 224, 224, 3)) # ImageNet size
    ])
    def test_different_input_sizes(self, input_shape, sample_shape):
        """Test model with different input sizes."""
        model = MobileNetV1.from_variant("0.5", num_classes=5, input_shape=input_shape)
        sample_input = keras.random.normal(shape=sample_shape)
        output = model(sample_input)
        assert output.shape == (sample_shape[0], 5)

    def test_batch_size_handling(self):
        """Test model handles different batch sizes correctly."""
        model = MobileNetV1.from_variant("0.25", num_classes=10, input_shape=(32, 32, 3))
        for batch_size in [1, 4, 16]:
            sample_input = keras.random.normal(shape=(batch_size, 32, 32, 3))
            output = model(sample_input)
            assert output.shape == (batch_size, 10)

    # ============================================================================
    # Training and Gradient Tests
    # ============================================================================

    def test_gradients_flow(self, default_config, sample_inputs):
        """Test gradient computation and backpropagation."""
        model = MobileNetV1(**default_config)
        with tf.GradientTape() as tape:
            output = model(sample_inputs['cifar'], training=True)
            loss = ops.mean(ops.square(output))
        gradients = tape.gradient(loss, model.trainable_variables)
        assert len(gradients) > 0
        assert all(g is not None for g in gradients)
        non_zero_grads = [g for g in gradients if ops.max(ops.abs(g)) > 1e-8]
        assert len(non_zero_grads) > 0

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, default_config, sample_inputs, training):
        """Test model behavior in different training modes."""
        model = MobileNetV1(**default_config)
        output = model(sample_inputs['cifar'], training=training)
        assert output.shape == (4, default_config['num_classes'])

    def test_model_compilation_and_fit(self, sample_inputs):
        """Test model compiles and can run a training step."""
        model = MobileNetV1.from_variant("0.25", num_classes=10, input_shape=(32, 32, 3))
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        x_train = sample_inputs['cifar']
        y_train = np.random.randint(0, 10, size=(4,))
        history = model.fit(x_train, y_train, epochs=1, verbose=0)
        assert 'loss' in history.history
        assert len(history.history['loss']) == 1

    # ============================================================================
    # Edge Cases and Error Handling
    # ============================================================================

    def test_invalid_configurations(self):
        """Test that invalid configurations raise appropriate errors."""
        with pytest.raises(ValueError, match="width_multiplier must be positive"):
            MobileNetV1(width_multiplier=0)
        with pytest.raises(ValueError, match="width_multiplier must be positive"):
            MobileNetV1(width_multiplier=-1.0)

    def test_unknown_variant(self):
        """Test error handling for unknown variants."""
        with pytest.raises(ValueError, match="Unknown variant 'invalid_variant'"):
            MobileNetV1.from_variant("invalid_variant")

    def test_invalid_input_shape(self):
        """Test error handling for invalid input shapes."""
        with pytest.raises(ValueError, match="input_shape must be 3D"):
            MobileNetV1(input_shape=(32, 32))

    # ============================================================================
    # Performance and Sanity Tests
    # ============================================================================

    def test_model_parameter_counts(self):
        """Test parameter counts are reasonable for different variants."""
        # Expected params for ImageNet (num_classes=1000, input_shape=(224, 224, 3))
        # From paper: 1.0 -> ~4.2M, 0.75 -> ~2.6M, 0.5 -> ~1.3M, 0.25 -> ~0.5M
        params_ranges = {
            "1.0": (4.0e6, 4.5e6),
            "0.75": (2.4e6, 2.8e6),
            "0.5": (1.2e6, 1.5e6),
            "0.25": (0.4e6, 0.6e6),
        }
        for variant, (min_p, max_p) in params_ranges.items():
            model = MobileNetV1.from_variant(
                variant,
                num_classes=1000,
                input_shape=(224, 224, 3)
            )
            # FIX: Build the model by calling it on a dummy input before counting params
            model(keras.random.normal(shape=(1, 224, 224, 3)))

            param_count = model.count_params()
            assert min_p <= param_count <= max_p, (
                f"{variant} has {param_count} parameters, "
                f"expected between {min_p} and {max_p}"
            )

    def test_model_summary_execution(self):
        """Test that model summary executes without errors."""
        model = MobileNetV1.from_variant("0.25", num_classes=10, input_shape=(32, 32, 3))
        try:
            model.summary()
        except Exception as e:
            pytest.fail(f"Model summary failed with error: {e}")

    # ============================================================================
    # Integration Tests
    # ============================================================================

    def test_model_evaluation_and_prediction(self):
        """Test model evaluation and prediction methods."""
        model = MobileNetV1.from_variant("0.25", num_classes=5, input_shape=(32, 32, 3))
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        x_test = keras.random.normal(shape=(10, 32, 32, 3))
        y_test = np.random.randint(0, 5, size=(10,))

        results = model.evaluate(x_test, y_test, verbose=0)
        assert len(results) == 2  # loss and accuracy

        predictions = model.predict(x_test, verbose=0)
        assert predictions.shape == (10, 5)
        pred_sums = ops.sum(predictions, axis=1)
        np.testing.assert_allclose(
            ops.convert_to_numpy(pred_sums),
            np.ones(10),
            rtol=1e-6, atol=1e-6,
            err_msg="Softmax predictions don't sum to 1"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])