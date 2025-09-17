"""
Comprehensive test suite for MobileNetV2 model implementation.

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

from dl_techniques.models.mobilenet.mobilenet_v2 import MobileNetV2, create_mobilenetv2


def make_divisible(v: float, divisor: int = 8) -> int:
    """Helper function to replicate width multiplier logic for testing."""
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class TestMobileNetV2:
    """Comprehensive test suite for MobileNetV2 model."""

    @pytest.fixture
    def default_config(self) -> Dict[str, Any]:
        """Standard configuration for testing (alpha=1.0)."""
        return {
            'num_classes': 10,
            'width_multiplier': 1.0,
            'dropout_rate': 0.1,
            'weight_decay': 1e-5,
            'input_shape': (32, 32, 3),
            'include_top': True
        }

    @pytest.fixture
    def small_config(self) -> Dict[str, Any]:
        """Configuration with a smaller width multiplier (alpha=0.5)."""
        return {
            'num_classes': 100,
            'width_multiplier': 0.5,
            'dropout_rate': 0.1,
            'weight_decay': 1e-5,
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

    def test_width_multiplier_scaling(self):
        """Test width multiplier correctly scales layer dimensions."""
        multiplier = 0.75
        model = MobileNetV2(
            width_multiplier=multiplier,
            input_shape=(32, 32, 3)
        )

        # Check initial convolution layer
        expected_initial_filters = make_divisible(32 * multiplier)
        assert model.initial_conv.filters == expected_initial_filters

        # Check a specific bottleneck block (first block of the second stage)
        # Architecture: (6, 24, 2, 2)
        expected_stage2_filters = make_divisible(24 * multiplier)
        assert model.blocks[1].filters == expected_stage2_filters

        # Check last convolution layer
        expected_last_filters = make_divisible(1280 * multiplier)
        assert model.last_conv.filters == expected_last_filters

    def test_model_building_on_call(self, default_config, sample_inputs):
        """Test that model builds correctly on first call."""
        config_no_shape = default_config.copy()
        del config_no_shape['input_shape']
        model = MobileNetV2(**config_no_shape)
        assert not model.built

        # First call should trigger building with default shape
        output = model(sample_inputs['imagenet'])
        assert model.built
        assert output.shape == (2, default_config['num_classes'])

    # ============================================================================
    # Critical Serialization Tests
    # ============================================================================

    def test_serialization_cycle_default(self, default_config, sample_inputs):
        """CRITICAL TEST: Full serialization cycle for standard MobileNetV2."""
        model = MobileNetV2(**default_config)

        original_pred = model(sample_inputs['cifar'])

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'mobilenetv2_default.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_inputs['cifar'])

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Default model predictions differ after serialization"
            )

    def test_serialization_cycle_small(self, small_config, sample_inputs):
        """CRITICAL TEST: Full serialization cycle for small variant."""
        model = MobileNetV2(**small_config)

        original_pred = model(sample_inputs['custom'])

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'mobilenetv2_small.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_inputs['custom'])

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Small variant predictions differ after serialization"
            )

    def test_serialization_without_top(self, sample_inputs):
        """Test serialization of feature extractor (include_top=False)."""
        model = MobileNetV2(
            width_multiplier=1.0,
            include_top=False,
            input_shape=(32, 32, 3)
        )

        original_features = model(sample_inputs['cifar'])

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'mobilenetv2_features.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_features = loaded_model(sample_inputs['cifar'])

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_features),
                keras.ops.convert_to_numpy(loaded_features),
                rtol=1e-6, atol=1e-6,
                err_msg="Feature extractor outputs differ after serialization"
            )

    # ============================================================================
    # Configuration and Variant Tests
    # ============================================================================

    def test_config_completeness(self, default_config):
        """Test that get_config contains all initialization parameters."""
        model = MobileNetV2(**default_config)
        config = model.get_config()

        required_keys = [
            'num_classes', 'width_multiplier', 'dropout_rate', 'weight_decay',
            'kernel_initializer', 'include_top', 'input_shape'
        ]

        for key in required_keys:
            assert key in config, f"Missing {key} in get_config()"

    def test_from_config_reconstruction(self, default_config):
        """Test model reconstruction from configuration."""
        model = MobileNetV2(**default_config)
        config = model.get_config()

        reconstructed_model = MobileNetV2.from_config(config)

        assert reconstructed_model.num_classes == model.num_classes
        assert reconstructed_model.width_multiplier == model.width_multiplier
        assert reconstructed_model.dropout_rate == model.dropout_rate

    @pytest.mark.parametrize("variant", ["1.4", "1.0", "0.75", "0.5", "0.35"])
    def test_variant_creation(self, variant):
        """Test creation of all predefined variants."""
        model = MobileNetV2.from_variant(variant, num_classes=10, input_shape=(32, 32, 3))
        expected_multiplier = MobileNetV2.MODEL_VARIANTS[variant]
        assert model.width_multiplier == expected_multiplier

    def test_convenience_function(self):
        """Test create_mobilenetv2 convenience function."""
        model = create_mobilenetv2(
            variant="0.75",
            num_classes=10,
            input_shape=(32, 32, 3),
            dropout_rate=0.5
        )

        assert model.num_classes == 10
        assert model.width_multiplier == 0.75
        assert model.dropout_rate == 0.5
        assert model.input_shape_config == (32, 32, 3)

    # ============================================================================
    # Forward Pass and Output Shape Tests
    # ============================================================================

    def test_forward_pass_shapes(self, sample_inputs):
        """Test forward pass produces correct output shapes."""
        model = MobileNetV2.from_variant("1.0", num_classes=10, input_shape=(32, 32, 3))
        output = model(sample_inputs['cifar'])
        assert output.shape == (4, 10)

        model = MobileNetV2.from_variant("0.5", include_top=False, input_shape=(32, 32, 3))
        features = model(sample_inputs['cifar'])
        assert len(features.shape) == 4

    def test_different_input_sizes(self):
        """Test model with different input sizes."""
        test_configs = [
            ((28, 28, 1), (2, 28, 28, 1)),
            ((64, 64, 3), (3, 64, 64, 3)),
            ((224, 224, 3), (1, 224, 224, 3)),
        ]

        for input_shape, sample_shape in test_configs:
            model = MobileNetV2.from_variant("0.5", num_classes=5, input_shape=input_shape)
            sample_input = keras.random.normal(shape=sample_shape)
            output = model(sample_input)
            assert output.shape == (sample_shape[0], 5)

    def test_batch_size_handling(self):
        """Test model handles different batch sizes correctly."""
        model = MobileNetV2.from_variant("1.0", num_classes=10, input_shape=(32, 32, 3))
        for batch_size in [1, 4, 8]:
            sample_input = keras.random.normal(shape=(batch_size, 32, 32, 3))
            output = model(sample_input)
            assert output.shape == (batch_size, 10)

    # ============================================================================
    # Training and Gradient Tests
    # ============================================================================

    def test_gradients_flow(self, default_config, sample_inputs):
        """Test gradient computation and backpropagation."""
        model = MobileNetV2(**default_config)
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
        """Test behavior in different training modes."""
        model = MobileNetV2(**default_config)
        output = model(sample_inputs['cifar'], training=training)
        assert output.shape == (4, default_config['num_classes'])

    def test_model_compilation_and_fit(self, sample_inputs):
        """Test model compiles and can run a training step."""
        model = MobileNetV2.from_variant("0.35", num_classes=10, input_shape=(32, 32, 3))
        # FIX: Use keyword arguments to prevent type promotion errors.
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
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
            MobileNetV2(width_multiplier=0)

        with pytest.raises(ValueError, match="input_shape must be a 3D tuple"):
            MobileNetV2(input_shape=(32, 32))

    def test_unknown_variant(self):
        """Test error handling for unknown variants."""
        with pytest.raises(ValueError, match="Unknown variant"):
            MobileNetV2.from_variant("nonexistent_variant")

    # ============================================================================
    # Performance and Memory Tests
    # ============================================================================

    def test_model_parameter_counts(self):
        """Test parameter counts are reasonable for different variants."""
        variants_expected_range = {
            '1.0': (3.0e6, 4.0e6),
            '0.75': (2.0e6, 3.0e6),
            '0.5': (1.5e6, 2.5e6),
            '0.35': (1.0e6, 2.0e6),
        }
        for variant, (min_p, max_p) in variants_expected_range.items():
            model = MobileNetV2.from_variant(variant, num_classes=1000)
            model(keras.random.normal(shape=(1, 224, 224, 3)))
            param_count = model.count_params()
            assert min_p <= param_count <= max_p, f"{variant} params out of range"

    def test_model_summary_execution(self):
        """Test that model summary executes without errors."""
        model = MobileNetV2.from_variant("0.5", num_classes=10, input_shape=(32, 32, 3))
        model(keras.random.normal(shape=(1, 32, 32, 3)))
        try:
            model.summary()
        except Exception as e:
            pytest.fail(f"Model summary failed with error: {e}")

    # ============================================================================
    # Integration Tests
    # ============================================================================

    def test_model_in_training_loop(self):
        """Test model in a realistic training scenario."""
        model = MobileNetV2.from_variant("0.5", num_classes=2, input_shape=(32, 32, 3))
        # FIX: Use keyword arguments to prevent type promotion errors.
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        x_train = keras.random.normal(shape=(20, 32, 32, 3))
        y_train = np.random.randint(0, 2, size=(20,))
        x_val = keras.random.normal(shape=(10, 32, 32, 3))
        y_val = np.random.randint(0, 2, size=(10,))
        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=2, batch_size=4, verbose=0
        )
        assert len(history.history['loss']) == 2
        assert 'val_loss' in history.history

    def test_model_evaluation_and_prediction(self):
        """Test model evaluation and prediction methods."""
        model = MobileNetV2.from_variant("0.5", num_classes=5, input_shape=(32, 32, 3))
        # FIX: Use keyword arguments to prevent type promotion errors.
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        x_test = keras.random.normal(shape=(10, 32, 32, 3))
        y_test = np.random.randint(0, 5, size=(10,))
        results = model.evaluate(x_test, y_test, verbose=0)
        assert len(results) == 2
        predictions = model.predict(x_test, verbose=0)
        assert predictions.shape == (10, 5)
        pred_sums = ops.sum(predictions, axis=1)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(pred_sums),
            np.ones(10),
            rtol=1e-6, atol=1e-6
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])