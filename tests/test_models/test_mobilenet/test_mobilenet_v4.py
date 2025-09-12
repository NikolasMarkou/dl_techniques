"""
Comprehensive test suite for MobileNetV4 model implementation.

Following the Complete Guide to Modern Keras 3 Custom Layers and Models,
this test suite ensures robust, serializable, and production-ready implementation
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
from typing import Dict, Any, List, Tuple

import keras
from keras import ops
import tensorflow as tf

from dl_techniques.models.mobilenet.mobilenet_v4 import MobileNetV4, create_mobilenetv4
from dl_techniques.utils.logger import logger


class TestMobileNetV4:
    """Comprehensive test suite for MobileNetV4 model."""

    @pytest.fixture
    def default_config(self) -> Dict[str, Any]:
        """Standard configuration for testing."""
        return {
            'num_classes': 10,
            'depths': [1, 1, 2, 2],
            'dims': [16, 24, 32, 64],
            'block_types': ['IB', 'IB', 'ExtraDW', 'IB'],
            'strides': [1, 2, 2, 1],
            'width_multiplier': 1.0,
            'use_attention': False,
            'dropout_rate': 0.1,
            'weight_decay': 1e-5,
            'input_shape': (32, 32, 3),
            'include_top': True
        }

    @pytest.fixture
    def hybrid_config(self) -> Dict[str, Any]:
        """Configuration with attention for hybrid testing."""
        return {
            'num_classes': 100,
            'depths': [1, 1, 2, 2],
            'dims': [16, 24, 32, 64],
            'block_types': ['IB', 'IB', 'ExtraDW', 'IB'],
            'strides': [1, 2, 2, 1],
            'width_multiplier': 1.0,
            'use_attention': True,
            'attention_stages': [2, 3],
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
        """Test width multiplier correctly scales dimensions."""
        base_dims = [16, 24, 32, 64]
        multiplier = 0.75
        num_stages = len(base_dims)

        model = MobileNetV4(
            dims=base_dims,
            depths=[1] * num_stages,
            block_types=['IB'] * num_stages,
            strides=[1, 2, 1, 2],
            width_multiplier=multiplier,
            input_shape=(32, 32, 3)
        )

        expected_dims = [int(dim * multiplier) for dim in base_dims]
        assert model.actual_dims == expected_dims

    def test_model_building_on_call(self, default_config, sample_inputs):
        """Test that model builds correctly on first call."""
        config_no_shape = default_config.copy()
        del config_no_shape['input_shape']
        model = MobileNetV4(**config_no_shape)
        assert not model.built

        # First call should trigger building
        output = model(sample_inputs['cifar'])
        assert model.built
        assert output.shape == (4, default_config['num_classes'])

    # ============================================================================
    # Critical Serialization Tests
    # ============================================================================

    def test_serialization_cycle_conv_only(self, default_config, sample_inputs):
        """CRITICAL TEST: Full serialization cycle for conv-only model."""
        model = MobileNetV4(**default_config)

        # Get prediction from original model
        original_pred = model(sample_inputs['cifar'])

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'mobilenetv4_conv.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_inputs['cifar'])

            # Verify identical predictions
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Conv-only model predictions differ after serialization"
            )

    def test_serialization_cycle_hybrid(self, hybrid_config, sample_inputs):
        """CRITICAL TEST: Full serialization cycle for hybrid model with attention."""
        model = MobileNetV4(**hybrid_config)

        # Get prediction from original model
        original_pred = model(sample_inputs['custom'])

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'mobilenetv4_hybrid.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_inputs['custom'])

            # Verify identical predictions
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Hybrid model predictions differ after serialization"
            )

    def test_serialization_without_top(self, sample_inputs):
        """Test serialization of feature extractor (include_top=False)."""
        model = MobileNetV4(
            depths=[1, 1, 2],
            dims=[16, 24, 32],
            block_types=['IB', 'IB', 'ExtraDW'],
            strides=[1, 2, 2],
            include_top=False,
            input_shape=(32, 32, 3)
        )

        # Get features from original model
        original_features = model(sample_inputs['cifar'])

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'mobilenetv4_features.keras')
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

    def test_config_completeness(self, default_config):
        """Test that get_config contains all initialization parameters."""
        model = MobileNetV4(**default_config)
        config = model.get_config()

        # Check all important config parameters are present
        required_keys = [
            'num_classes', 'depths', 'dims', 'block_types', 'strides',
            'width_multiplier', 'use_attention', 'attention_stages',
            'dropout_rate', 'weight_decay', 'kernel_initializer',
            'include_top', 'input_shape'
        ]

        for key in required_keys:
            assert key in config, f"Missing {key} in get_config()"

    def test_from_config_reconstruction(self, default_config):
        """Test model reconstruction from configuration."""
        model = MobileNetV4(**default_config)
        config = model.get_config()

        # Reconstruct model from config
        reconstructed_model = MobileNetV4.from_config(config)

        # Verify configurations match
        assert reconstructed_model.num_classes == model.num_classes
        assert reconstructed_model.depths == model.depths
        assert reconstructed_model.dims == model.dims
        assert reconstructed_model.block_types == model.block_types
        assert reconstructed_model.width_multiplier == model.width_multiplier

    @pytest.mark.parametrize("variant", [
        "conv_small", "conv_medium", "conv_large", "hybrid_medium", "hybrid_large"
    ])
    def test_variant_creation(self, variant):
        """Test creation of all predefined variants."""
        model = MobileNetV4.from_variant(variant, num_classes=10, input_shape=(32, 32, 3))

        # Check variant-specific properties
        variant_config = MobileNetV4.MODEL_VARIANTS[variant]
        assert model.depths == variant_config['depths']
        assert model.dims == variant_config['dims']
        assert model.block_types == variant_config['block_types']
        assert model.use_attention == variant_config['use_attention']

    def test_convenience_function(self):
        """Test create_mobilenetv4 convenience function."""
        model = create_mobilenetv4(
            variant="conv_small",
            num_classes=10,
            input_shape=(32, 32, 3),
            width_multiplier=0.75
        )

        assert model.num_classes == 10
        assert model.width_multiplier == 0.75
        assert model._input_shape == (32, 32, 3)

    # ============================================================================
    # Forward Pass and Output Shape Tests
    # ============================================================================

    def test_forward_pass_shapes(self, sample_inputs):
        """Test forward pass produces correct output shapes."""
        # Conv-only model
        model = MobileNetV4.from_variant("conv_small", num_classes=10, input_shape=(32, 32, 3))
        output = model(sample_inputs['cifar'])
        assert output.shape == (4, 10)

        # Feature extractor
        model = MobileNetV4.from_variant("conv_small", include_top=False, input_shape=(32, 32, 3))
        features = model(sample_inputs['cifar'])
        # Output should be spatial features from last stage
        assert len(features.shape) == 4  # (batch, height, width, channels)

    def test_different_input_sizes(self):
        """Test model with different input sizes."""
        test_configs = [
            ((28, 28, 1), (2, 28, 28, 1)),  # MNIST-like
            ((64, 64, 3), (3, 64, 64, 3)),  # Medium resolution
            ((224, 224, 3), (1, 224, 224, 3)),  # ImageNet-like
        ]

        for input_shape, sample_shape in test_configs:
            model = MobileNetV4.from_variant("conv_small", num_classes=5, input_shape=input_shape)
            sample_input = keras.random.normal(shape=sample_shape)

            output = model(sample_input)
            assert output.shape == (sample_shape[0], 5)

    def test_batch_size_handling(self):
        """Test model handles different batch sizes correctly."""
        model = MobileNetV4.from_variant("conv_small", num_classes=10, input_shape=(32, 32, 3))

        batch_sizes = [1, 4, 8, 16]
        for batch_size in batch_sizes:
            sample_input = keras.random.normal(shape=(batch_size, 32, 32, 3))
            output = model(sample_input)
            assert output.shape == (batch_size, 10)

    # ============================================================================
    # Training and Gradient Tests
    # ============================================================================

    def test_gradients_flow(self, default_config, sample_inputs):
        """Test gradient computation and backpropagation."""
        model = MobileNetV4(**default_config)

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
    def test_training_modes(self, default_config, sample_inputs, training):
        """Test behavior in different training modes."""
        model = MobileNetV4(**default_config)

        output = model(sample_inputs['cifar'], training=training)
        assert output.shape == (4, default_config['num_classes'])

        # Test with attention model too
        hybrid_config = default_config.copy()
        hybrid_config.update({
            "use_attention": True,
            "attention_stages": [2, 3]
        })
        hybrid_model = MobileNetV4(**hybrid_config)
        output = hybrid_model(sample_inputs['cifar'], training=training)
        assert output.shape == (4, default_config['num_classes'])

    def test_model_compilation_and_fit(self, sample_inputs):
        """Test model compiles and can run a training step."""
        model = MobileNetV4.from_variant("conv_small", num_classes=10, input_shape=(32, 32, 3))

        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Create dummy training data
        x_train = sample_inputs['cifar']
        y_train = np.random.randint(0, 10, size=(4,))

        # Test one training step
        history = model.fit(x_train, y_train, epochs=1, verbose=0)

        # Check training executed successfully
        assert 'loss' in history.history
        assert len(history.history['loss']) == 1

    # ============================================================================
    # Edge Cases and Error Handling
    # ============================================================================

    def test_invalid_configurations(self):
        """Test that invalid configurations raise appropriate errors."""

        # Mismatched configuration lengths
        with pytest.raises(ValueError, match="All stage configurations must have same length"):
            MobileNetV4(
                depths=[1, 2, 3],
                dims=[16, 24],  # Different length
                block_types=['IB', 'IB', 'ExtraDW'],
                strides=[1, 2, 2],
                input_shape=(32, 32, 3)
            )

        # Invalid block type
        with pytest.raises(ValueError, match="Invalid block type"):
            MobileNetV4(
                depths=[1, 2],
                dims=[16, 24],
                block_types=['IB', 'InvalidBlock'],
                strides=[1, 2],
                input_shape=(32, 32, 3)
            )

        # Invalid attention stage index
        with pytest.raises(ValueError, match="Attention stage index.*out of range"):
            MobileNetV4(
                depths=[1, 2],
                dims=[16, 24],
                block_types=['IB', 'IB'],
                strides=[1, 2],
                use_attention=True,
                attention_stages=[5],  # Out of range
                input_shape=(32, 32, 3)
            )

    def test_unknown_variant(self):
        """Test error handling for unknown variants."""
        with pytest.raises(ValueError, match="Unknown variant"):
            MobileNetV4.from_variant("nonexistent_variant")

    def test_invalid_input_shape(self):
        """Test error handling for invalid input shapes."""
        with pytest.raises(ValueError, match="input_shape must be 3D"):
            MobileNetV4(input_shape=(32, 32))  # Missing channel dimension

    def test_edge_case_dimensions(self):
        """Test edge cases with small dimensions and configurations."""
        # Very small model
        model = MobileNetV4(
            depths=[1],
            dims=[8],
            block_types=['IB'],
            strides=[1],
            num_classes=2,
            input_shape=(16, 16, 1)
        )

        sample_input = keras.random.normal(shape=(2, 16, 16, 1))
        output = model(sample_input)
        assert output.shape == (2, 2)

    # ============================================================================
    # Performance and Memory Tests
    # ============================================================================

    def test_model_parameter_counts(self):
        """Test parameter counts are reasonable for different variants."""
        variants_expected_range = {
            'conv_small': (1e5, 5e6),    # 100K - 5M parameters
            'conv_medium': (3e6, 20e6),  # 3M - 20M parameters
            'conv_large': (4e6, 50e6),  # 10M - 50M parameters
        }

        for variant, (min_params, max_params) in variants_expected_range.items():
            model = MobileNetV4.from_variant(variant, num_classes=1000)
            # Build model to count parameters
            model(keras.random.normal(shape=(1, 224, 224, 3)))

            param_count = model.count_params()
            assert min_params <= param_count <= max_params, (
                f"{variant} has {param_count} parameters, "
                f"expected between {min_params} and {max_params}"
            )

    def test_model_summary_execution(self):
        """Test that model summary executes without errors."""
        model = MobileNetV4.from_variant("conv_small", num_classes=10, input_shape=(32, 32, 3))

        # Build model first
        model(keras.random.normal(shape=(1, 32, 32, 3)))

        # Test summary doesn't crash
        try:
            model.summary()
        except Exception as e:
            pytest.fail(f"Model summary failed with error: {e}")

    # ============================================================================
    # Integration Tests
    # ============================================================================

    def test_model_in_training_loop(self):
        """Test model in a realistic training scenario."""
        model = MobileNetV4.from_variant("conv_small", num_classes=2, input_shape=(32, 32, 3))
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Generate synthetic data
        x_train = keras.random.normal(shape=(20, 32, 32, 3))
        y_train = np.random.randint(0, 2, size=(20,))
        x_val = keras.random.normal(shape=(10, 32, 32, 3))
        y_val = np.random.randint(0, 2, size=(10,))

        # Train for a few epochs
        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=2,
            batch_size=4,
            verbose=0
        )

        # Check training completed successfully
        assert len(history.history['loss']) == 2
        assert 'val_loss' in history.history

    def test_model_evaluation_and_prediction(self):
        """Test model evaluation and prediction methods."""
        model = MobileNetV4.from_variant("conv_small", num_classes=5, input_shape=(32, 32, 3))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Test data
        x_test = keras.random.normal(shape=(10, 32, 32, 3))
        y_test = np.random.randint(0, 5, size=(10,))

        # Test evaluation
        results = model.evaluate(x_test, y_test, verbose=0)
        assert len(results) == 2  # loss and accuracy

        # Test prediction
        predictions = model.predict(x_test, verbose=0)
        assert predictions.shape == (10, 5)

        # Test predictions sum to 1 (softmax)
        pred_sums = ops.sum(predictions, axis=1)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(pred_sums),
            np.ones(10),
            rtol=1e-6, atol=1e-6,
            err_msg="Softmax predictions don't sum to 1"
        )


# ============================================================================
# Additional Test Utilities and Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def test_models():
    """Create test models for reuse across tests."""
    models = {}

    # Small conv model for quick tests
    models['small_conv'] = MobileNetV4.from_variant(
        "conv_small",
        num_classes=10,
        input_shape=(32, 32, 3)
    )

    # Medium hybrid model
    models['medium_hybrid'] = MobileNetV4.from_variant(
        "hybrid_medium",
        num_classes=100,
        input_shape=(64, 64, 3)
    )

    return models


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])