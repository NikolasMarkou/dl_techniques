"""
Comprehensive test suite for MobileNetV3 model implementation.

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

from dl_techniques.models.mobilenet.mobilenet_v3 import MobileNetV3, create_mobilenetv3


def make_divisible(value: float, divisor: int = 8) -> int:
    """Helper function to replicate width multiplier logic."""
    new_value = max(divisor, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


class TestMobileNetV3:
    """Comprehensive test suite for MobileNetV3 model."""

    @pytest.fixture
    def large_config(self) -> Dict[str, Any]:
        """Standard configuration for MobileNetV3-Large."""
        return {
            'num_classes': 10,
            'variant': 'large',
            'width_multiplier': 1.0,
            'dropout_rate': 0.1,
            'weight_decay': 1e-5,
            'input_shape': (32, 32, 3),
            'include_top': True
        }

    @pytest.fixture
    def small_config(self) -> Dict[str, Any]:
        """Configuration for MobileNetV3-Small."""
        return {
            'num_classes': 100,
            'variant': 'small',
            'width_multiplier': 0.75,
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
        model = MobileNetV3(
            variant='large',
            width_multiplier=multiplier,
            input_shape=(32, 32, 3)
        )

        # Check stem convolution
        expected_stem_filters = make_divisible(16 * multiplier)
        assert model.stem_conv.filters == expected_stem_filters

        # Check a specific block's output channels (e.g., first block of Large)
        expected_block0_out = make_divisible(16 * multiplier)
        assert model.blocks[0].filters == expected_block0_out

        # Check last convolution block
        expected_last_conv = make_divisible(960 * multiplier)
        assert model.last_conv.filters == expected_last_conv

    def test_model_building_on_call(self, large_config, sample_inputs):
        """Test that model builds correctly on first call."""
        config_no_shape = large_config.copy()
        del config_no_shape['input_shape']
        model = MobileNetV3(**config_no_shape)
        assert not model.built

        # First call should trigger building with default shape
        output = model(sample_inputs['imagenet'])
        assert model.built
        assert output.shape == (2, large_config['num_classes'])

    # ============================================================================
    # Critical Serialization Tests
    # ============================================================================

    def test_serialization_cycle_large(self, large_config, sample_inputs):
        """CRITICAL TEST: Full serialization cycle for MobileNetV3-Large."""
        model = MobileNetV3(**large_config)

        # Get prediction from original model
        original_pred = model(sample_inputs['cifar'])

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'mobilenetv3_large.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_inputs['cifar'])

            # Verify identical predictions
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="MobileNetV3-Large predictions differ after serialization"
            )

    def test_serialization_cycle_small(self, small_config, sample_inputs):
        """CRITICAL TEST: Full serialization cycle for MobileNetV3-Small."""
        model = MobileNetV3(**small_config)

        # Get prediction from original model
        original_pred = model(sample_inputs['custom'])

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'mobilenetv3_small.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_inputs['custom'])

            # Verify identical predictions
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="MobileNetV3-Small predictions differ after serialization"
            )

    def test_serialization_without_top(self, sample_inputs):
        """Test serialization of feature extractor (include_top=False)."""
        model = MobileNetV3(
            variant='small',
            include_top=False,
            input_shape=(32, 32, 3)
        )

        # Get features from original model
        original_features = model(sample_inputs['cifar'])

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'mobilenetv3_features.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_features = loaded_model(sample_inputs['cifar'])

            # Verify identical features
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_features),
                keras.ops.convert_to_numpy(loaded_features),
                rtol=1e-6, atol=1e-6,
                err_msg="Feature extractor outputs differ after serialization"
            )

    # ============================================================================
    # Configuration and Variant Tests
    # ============================================================================

    def test_config_completeness(self, large_config):
        """Test that get_config contains all initialization parameters."""
        model = MobileNetV3(**large_config)
        config = model.get_config()

        required_keys = [
            'num_classes', 'variant', 'width_multiplier', 'dropout_rate',
            'weight_decay', 'kernel_initializer', 'include_top', 'input_shape'
        ]

        for key in required_keys:
            assert key in config, f"Missing {key} in get_config()"

    def test_from_config_reconstruction(self, large_config):
        """Test model reconstruction from configuration."""
        model = MobileNetV3(**large_config)
        config = model.get_config()

        # Reconstruct model from config
        reconstructed_model = MobileNetV3.from_config(config)

        # Verify configurations match
        assert reconstructed_model.num_classes == model.num_classes
        assert reconstructed_model.variant == model.variant
        assert reconstructed_model.width_multiplier == model.width_multiplier
        assert reconstructed_model.dropout_rate == model.dropout_rate

    @pytest.mark.parametrize("variant", ["large", "small"])
    def test_variant_creation(self, variant):
        """Test creation of all predefined variants."""
        model = MobileNetV3.from_variant(variant, num_classes=10, input_shape=(32, 32, 3))

        # Check variant-specific properties
        if variant == "large":
            assert len(model.blocks) == len(MobileNetV3.LARGE_CONFIG)
            assert model.last_block_filters == 960
        else:
            assert len(model.blocks) == len(MobileNetV3.SMALL_CONFIG)
            assert model.last_block_filters == 576

    def test_convenience_function(self):
        """Test create_mobilenetv3 convenience function."""
        model = create_mobilenetv3(
            variant="small",
            num_classes=10,
            input_shape=(32, 32, 3),
            width_multiplier=0.75
        )

        assert model.num_classes == 10
        assert model.variant == "small"
        assert model.width_multiplier == 0.75
        assert model.input_shape_config == (32, 32, 3)

    # ============================================================================
    # Forward Pass and Output Shape Tests
    # ============================================================================

    def test_forward_pass_shapes(self, sample_inputs):
        """Test forward pass produces correct output shapes."""
        # Large model
        model = MobileNetV3.from_variant("large", num_classes=10, input_shape=(32, 32, 3))
        output = model(sample_inputs['cifar'])
        assert output.shape == (4, 10)

        # Feature extractor
        model = MobileNetV3.from_variant("small", include_top=False, input_shape=(32, 32, 3))
        features = model(sample_inputs['cifar'])
        assert len(features.shape) == 4  # (batch, height, width, channels)

    def test_different_input_sizes(self):
        """Test model with different input sizes."""
        test_configs = [
            ((28, 28, 1), (2, 28, 28, 1)),  # MNIST-like
            ((64, 64, 3), (3, 64, 64, 3)),  # Medium resolution
            ((224, 224, 3), (1, 224, 224, 3)),  # ImageNet-like
        ]

        for input_shape, sample_shape in test_configs:
            model = MobileNetV3.from_variant("small", num_classes=5, input_shape=input_shape)
            sample_input = keras.random.normal(shape=sample_shape)

            output = model(sample_input)
            assert output.shape == (sample_shape[0], 5)

    def test_batch_size_handling(self):
        """Test model handles different batch sizes correctly."""
        model = MobileNetV3.from_variant("small", num_classes=10, input_shape=(32, 32, 3))

        batch_sizes = [1, 4, 8, 16]
        for batch_size in batch_sizes:
            sample_input = keras.random.normal(shape=(batch_size, 32, 32, 3))
            output = model(sample_input)
            assert output.shape == (batch_size, 10)

    # ============================================================================
    # Training and Gradient Tests
    # ============================================================================

    def test_gradients_flow(self, large_config, sample_inputs):
        """Test gradient computation and backpropagation."""
        model = MobileNetV3(**large_config)

        with tf.GradientTape() as tape:
            output = model(sample_inputs['cifar'], training=True)
            loss = ops.mean(ops.square(output))

        gradients = tape.gradient(loss, model.trainable_variables)

        # Check all gradients exist and are non-zero
        assert len(gradients) > 0
        assert all(g is not None for g in gradients)

        # Check some gradients are non-zero (model is learning)
        non_zero_grads = [g for g in gradients if ops.max(ops.abs(g)) > 1e-8]
        assert len(non_zero_grads) > 0

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, large_config, sample_inputs, training):
        """Test behavior in different training modes."""
        model = MobileNetV3(**large_config)
        output = model(sample_inputs['cifar'], training=training)
        assert output.shape == (4, large_config['num_classes'])

    def test_model_compilation_and_fit(self, sample_inputs):
        """Test model compiles and can run a training step."""
        model = MobileNetV3.from_variant("small", num_classes=10, input_shape=(32, 32, 3))

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
        # Invalid variant name
        with pytest.raises(ValueError, match="Unknown variant"):
            MobileNetV3(variant="medium", input_shape=(32, 32, 3))

        # Invalid input shape
        with pytest.raises(ValueError, match="input_shape must be 3D"):
            MobileNetV3(input_shape=(32, 32))

    def test_unknown_variant(self):
        """Test error handling for unknown variants in factory method."""
        with pytest.raises(ValueError, match="Unknown variant"):
            MobileNetV3.from_variant("nonexistent_variant")

    # ============================================================================
    # Performance and Memory Tests
    # ============================================================================

    def test_model_parameter_counts(self):
        """Test parameter counts are reasonable for different variants."""
        variants_expected_range = {
            'small': (1.5e6, 10e6),   # ~2.5M params
            'large': (3.5e6, 7e6),   # ~5.4M params
        }

        for variant, (min_params, max_params) in variants_expected_range.items():
            model = MobileNetV3.from_variant(variant, num_classes=1000)
            model(keras.random.normal(shape=(1, 224, 224, 3)))

            param_count = model.count_params()
            assert min_params <= param_count <= max_params, (
                f"{variant} has {param_count} parameters, "
                f"expected between {min_params} and {max_params}"
            )

    def test_model_summary_execution(self):
        """Test that model summary executes without errors."""
        model = MobileNetV3.from_variant("small", num_classes=10, input_shape=(32, 32, 3))
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
        model = MobileNetV3.from_variant("small", num_classes=2, input_shape=(32, 32, 3))
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        x_train = keras.random.normal(shape=(20, 32, 32, 3))
        y_train = np.random.randint(0, 2, size=(20,))
        x_val = keras.random.normal(shape=(10, 32, 32, 3))
        y_val = np.random.randint(0, 2, size=(10,))

        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=2,
            batch_size=4,
            verbose=0
        )

        assert len(history.history['loss']) == 2
        assert 'val_loss' in history.history

    def test_model_evaluation_and_prediction(self):
        """Test model evaluation and prediction methods."""
        model = MobileNetV3.from_variant("small", num_classes=5, input_shape=(32, 32, 3))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        x_test = keras.random.normal(shape=(10, 32, 32, 3))
        y_test = np.random.randint(0, 5, size=(10,))

        results = model.evaluate(x_test, y_test, verbose=0)
        assert len(results) == 2  # loss and accuracy

        predictions = model.predict(x_test, verbose=0)
        assert predictions.shape == (10, 5)

        pred_sums = ops.sum(predictions, axis=1)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(pred_sums),
            np.ones(10),
            rtol=1e-6, atol=1e-6,
            err_msg="Softmax predictions don't sum to 1"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
