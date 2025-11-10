"""
Comprehensive test suite for the SqueezeNet V1 model.

This module contains all tests for the SqueezeNet V1 model implementation,
covering initialization, forward pass, training, serialization,
model variants, and edge cases.
"""

import pytest
import numpy as np
import tensorflow as tf
import keras
import tempfile
import os
from typing import Tuple, Dict, Any

from dl_techniques.models.squeezenet.squeezenet_v1 import SqueezeNetV1, FireModule, create_squeezenet_v1


class TestSqueezeNetV1:
    """Test suite for SqueezeNet V1 model implementation."""

    @pytest.fixture
    def input_shape(self) -> Tuple[int, int, int]:
        """Create test input shape for standard images."""
        return (224, 224, 3)

    @pytest.fixture
    def small_input_shape(self) -> Tuple[int, int, int]:
        """Create test input shape for smaller images."""
        return (64, 64, 3)

    @pytest.fixture
    def cifar_input_shape(self) -> Tuple[int, int, int]:
        """Create test input shape for CIFAR-10 like images."""
        return (32, 32, 3)

    @pytest.fixture
    def num_classes(self) -> int:
        """Create test number of classes."""
        return 10

    @pytest.fixture
    def sample_data(self, input_shape):
        """Create sample input data."""
        batch_size = 2
        return tf.random.uniform([batch_size] + list(input_shape), 0, 1)

    @pytest.fixture
    def small_sample_data(self, small_input_shape):
        """Create small sample input data."""
        batch_size = 4
        return tf.random.uniform([batch_size] + list(small_input_shape), 0, 1)

    @pytest.fixture
    def cifar_sample_data(self, cifar_input_shape):
        """Create CIFAR-like sample data."""
        batch_size = 8
        return tf.random.uniform([batch_size] + list(cifar_input_shape), 0, 1)

    def test_initialization_defaults(self, num_classes, input_shape):
        """Test initialization with default parameters."""
        model = SqueezeNetV1(num_classes=num_classes, input_shape=input_shape)

        default_config = SqueezeNetV1.MODEL_VARIANTS["1.0"]
        assert model.num_classes == num_classes
        assert model.include_top is True
        assert model.dropout_rate == 0.5
        assert model.use_bypass is False
        assert model.conv1_filters == default_config["conv1_filters"]
        assert model.conv1_kernel == default_config["conv1_kernel"]
        assert model.pool_indices == default_config["pool_indices"]
        assert len(model.fire_modules) == len(default_config["fire_configs"])

    def test_initialization_custom(self, num_classes, small_input_shape):
        """Test initialization with custom parameters."""
        regularizer = keras.regularizers.L1(0.01)
        initializer = keras.initializers.HeNormal()

        model = SqueezeNetV1(
            num_classes=num_classes,
            use_bypass="simple",
            dropout_rate=0.2,
            kernel_regularizer=regularizer,
            kernel_initializer=initializer,
            include_top=False,
            input_shape=small_input_shape
        )

        assert model.num_classes == num_classes
        assert model.use_bypass == "simple"
        assert model.dropout_rate == 0.2
        assert model.kernel_regularizer == regularizer
        # Note: Keras might serialize/deserialize initializers, so direct object comparison may fail
        assert isinstance(model.kernel_initializer, type(initializer))
        assert model.include_top is False

    def test_invalid_initialization(self):
        """Test initialization with invalid parameters raises errors."""
        with pytest.raises(ValueError, match="num_classes must be a positive integer"):
            SqueezeNetV1(num_classes=0)
        with pytest.raises(ValueError, match="dropout_rate must be in range"):
            SqueezeNetV1(dropout_rate=1.1)
        with pytest.raises(ValueError, match="dropout_rate must be in range"):
            SqueezeNetV1(dropout_rate=-0.1)

    @pytest.mark.parametrize(
        "variant, expected_conv1_filters, expected_bypass, expected_pool_indices",
        [
            ("1.0", 96, False, [1, 4, 8]),
            ("1.1", 64, False, [1, 3, 5]),
            ("1.0_bypass", 96, "simple", [1, 4, 8]),
        ],
    )
    def test_model_variants(self, variant, expected_conv1_filters, expected_bypass, expected_pool_indices, num_classes):
        """Test all predefined model variants."""
        model = SqueezeNetV1.from_variant(variant, num_classes=num_classes)

        assert model.num_classes == num_classes
        assert model.conv1_filters == expected_conv1_filters
        assert model.use_bypass == expected_bypass
        assert model.pool_indices == expected_pool_indices

    def test_invalid_variant(self, num_classes):
        """Test invalid variant raises error."""
        with pytest.raises(ValueError, match="Unknown variant"):
            SqueezeNetV1.from_variant("invalid_variant", num_classes=num_classes)

    def test_forward_pass_with_top(self, num_classes, sample_data):
        """Test forward pass with classification head."""
        model = SqueezeNetV1.from_variant("1.0", num_classes=num_classes)
        outputs = model(sample_data)

        # Check output shape
        expected_shape = (sample_data.shape[0], num_classes)
        assert outputs.shape == expected_shape

        # Check for valid values (probabilities should sum to 1)
        assert not np.any(np.isnan(outputs.numpy()))
        assert not np.any(np.isinf(outputs.numpy()))
        assert np.allclose(np.sum(outputs.numpy(), axis=-1), 1.0)

    def test_forward_pass_without_top(self, small_input_shape, small_sample_data):
        """Test forward pass without classification head (feature extraction)."""
        model = SqueezeNetV1(
            include_top=False,
            input_shape=small_input_shape
        )

        features = model(small_sample_data)

        # Output should be 4D tensor: (batch, height, width, channels)
        assert len(features.shape) == 4
        # The number of output channels is e1x1 + e3x3 of the last Fire module (fire9)
        last_fire_config = SqueezeNetV1.MODEL_VARIANTS["1.0"]["fire_configs"][-1]
        expected_channels = last_fire_config['e1x1'] + last_fire_config['e3x3']
        assert features.shape[-1] == expected_channels

        # Check for valid values
        assert not np.any(np.isnan(features.numpy()))
        assert not np.any(np.isinf(features.numpy()))

    def test_different_input_shapes(self, num_classes):
        """Test with different input shapes."""
        test_shapes = [
            (32, 32, 3),
            (64, 64, 3),
            (96, 96, 1),  # Grayscale
        ]

        for shape in test_shapes:
            # Use the less aggressive v1.1 for small inputs to avoid pooling errors
            variant = "1.1" if shape[0] < 64 else "1.0"

            model = SqueezeNetV1.from_variant(
                variant,
                num_classes=num_classes,
                input_shape=shape
            )
            batch_data = tf.random.uniform([2] + list(shape), 0, 1)
            outputs = model(batch_data)
            assert outputs.shape == (2, num_classes)

    def test_bypass_configurations(self, num_classes, small_input_shape, small_sample_data):
        """Test different bypass configurations build and run."""
        bypass_options = [False, "simple", "complex"]

        for bypass_type in bypass_options:
            model = SqueezeNetV1(
                num_classes=num_classes,
                use_bypass=bypass_type,
                input_shape=small_input_shape
            )
            outputs = model(small_sample_data, training=True)
            assert outputs.shape == (small_sample_data.shape[0], num_classes)

    def test_model_compilation(self, num_classes, cifar_input_shape):
        """Test model compilation with different optimizers and losses."""
        model = SqueezeNetV1(num_classes=num_classes, input_shape=cifar_input_shape, variant_config=SqueezeNetV1.MODEL_VARIANTS["1.1"])

        optimizers = [keras.optimizers.Adam(), keras.optimizers.SGD()]
        for optimizer in optimizers:
            model.compile(
                optimizer=optimizer,
                loss=keras.losses.SparseCategoricalCrossentropy(),
                metrics=["accuracy"]
            )
            assert model.optimizer == optimizer

    def test_serialization(self, num_classes):
        """Test serialization and deserialization of the model."""
        original_model = SqueezeNetV1.from_variant(
            "1.0_bypass",
            num_classes=num_classes,
            dropout_rate=0.3,
            kernel_regularizer=keras.regularizers.L2(0.02)
        )

        config = original_model.get_config()
        recreated_model = SqueezeNetV1.from_config(config)

        # Check configuration matches
        assert recreated_model.num_classes == original_model.num_classes
        assert recreated_model.use_bypass == original_model.use_bypass
        assert recreated_model.dropout_rate == original_model.dropout_rate
        assert recreated_model.get_config()['kernel_regularizer']['config'] == \
               original_model.get_config()['kernel_regularizer']['config']

    def test_model_save_load(self, num_classes, cifar_input_shape, cifar_sample_data):
        """Test saving and loading a SqueezeNet model."""
        model = SqueezeNetV1(
            num_classes=num_classes,
            input_shape=cifar_input_shape,
            name="test_squeezenet",
            variant_config=SqueezeNetV1.MODEL_VARIANTS["1.1"]
        )
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

        original_outputs = model(cifar_sample_data, training=False)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "squeezenet_model.keras")
            model.save(model_path)

            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={"SqueezeNetV1": SqueezeNetV1, "FireModule": FireModule}
            )

            loaded_outputs = loaded_model(cifar_sample_data, training=False)

            assert original_outputs.shape == loaded_outputs.shape
            np.testing.assert_allclose(original_outputs.numpy(), loaded_outputs.numpy(), rtol=1e-6)
            assert loaded_model.count_params() == model.count_params()

    def test_training_integration(self, num_classes, small_input_shape, small_sample_data):
        """Test training integration with a small dataset."""
        model = SqueezeNetV1(num_classes=num_classes, input_shape=small_input_shape)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        labels = tf.random.uniform([small_sample_data.shape[0]], 0, num_classes, dtype=tf.int32)
        dataset = tf.data.Dataset.from_tensor_slices((small_sample_data, labels)).batch(4)

        history = model.fit(dataset, epochs=2, verbose=0)

        assert "loss" in history.history
        assert "accuracy" in history.history
        assert len(history.history["loss"]) == 2

        initial_loss = history.history["loss"][0]
        final_loss = history.history["loss"][-1]
        assert not np.isnan(final_loss)
        assert final_loss < initial_loss * 2

    def test_gradient_flow(self, num_classes, small_input_shape, small_sample_data):
        """Test gradient flow through the model."""
        model = SqueezeNetV1(num_classes=num_classes, input_shape=small_input_shape)
        labels = tf.random.uniform([small_sample_data.shape[0]], 0, num_classes, dtype=tf.int32)

        with tf.GradientTape() as tape:
            outputs = model(small_sample_data, training=True)
            loss = keras.losses.sparse_categorical_crossentropy(labels, outputs)
            loss = tf.reduce_mean(loss)

        grads = tape.gradient(loss, model.trainable_variables)

        assert all(g is not None for g in grads)
        assert np.any(grads[0].numpy() != 0)
        assert np.any(grads[-2].numpy() != 0)

    def test_create_squeezenet_v1_factory(self, num_classes):
        """Test the create_squeezenet_v1 factory function."""
        model = create_squeezenet_v1(
            variant="1.1",
            num_classes=num_classes,
            dropout_rate=0.4
        )
        assert isinstance(model, SqueezeNetV1)
        assert model.num_classes == num_classes
        assert model.conv1_filters == SqueezeNetV1.MODEL_VARIANTS["1.1"]["conv1_filters"]
        assert model.dropout_rate == 0.4

    def test_factory_with_weights_warning(self, num_classes, caplog):
        """Test factory function with weights warning."""
        _ = create_squeezenet_v1(variant="1.0", num_classes=num_classes, weights="imagenet")
        assert "Pretrained weights are not yet implemented" in caplog.text

    def test_numerical_stability(self, num_classes, small_input_shape):
        """Test model stability with extreme input values."""
        # MODIFIED: Use a larger input shape to prevent pooling errors.
        model = SqueezeNetV1(num_classes=num_classes, input_shape=small_input_shape)

        test_cases = [
            tf.zeros((2,) + small_input_shape),
            tf.ones((2,) + small_input_shape),
            tf.random.uniform((2,) + small_input_shape, 0, 100.0), # More reasonable large values
        ]

        for i, test_input in enumerate(test_cases):
            outputs = model(test_input)
            # MODIFIED: Add a check for infinity.
            assert not np.any(np.isnan(outputs.numpy())), f"NaN in outputs for test case {i}"
            assert not np.any(np.isinf(outputs.numpy())), f"Inf in outputs for test case {i}"

    def test_batch_size_independence(self, num_classes, cifar_input_shape):
        """Test that model works with different batch sizes."""
        # Using the less aggressive v1.1 since the input is small
        model = SqueezeNetV1(num_classes=num_classes, input_shape=cifar_input_shape, variant_config=SqueezeNetV1.MODEL_VARIANTS["1.1"])

        batch_sizes = [1, 4, 8]
        for batch_size in batch_sizes:
            test_data = tf.random.uniform((batch_size,) + cifar_input_shape, 0, 1)
            outputs = model(test_data)
            assert outputs.shape == (batch_size, num_classes)

    def test_model_summary_with_details(self, capsys):
        """Test that model summary provides useful information."""
        model = SqueezeNetV1(num_classes=10)
        try:
            model.summary_with_details()
        except Exception as e:
            pytest.fail(f"summary_with_details() raised an exception: {e}")


# Tests for the FireModule itself
class TestFireModule:
    """Test suite for the FireModule layer."""

    @pytest.fixture
    def fire_module_config(self) -> Dict[str, Any]:
        return {'s1x1': 16, 'e1x1': 64, 'e3x3': 64}

    def test_initialization(self, fire_module_config):
        """Test FireModule initialization."""
        module = FireModule(**fire_module_config)
        assert module.s1x1 == fire_module_config['s1x1']
        assert module.e1x1 == fire_module_config['e1x1']
        assert module.e3x3 == fire_module_config['e3x3']
        assert module.squeeze.filters == fire_module_config['s1x1']
        assert module.expand_1x1.filters == fire_module_config['e1x1']
        assert module.expand_3x3.filters == fire_module_config['e3x3']

    def test_invalid_initialization(self):
        """Test that invalid filter counts raise an error."""
        with pytest.raises(ValueError):
            FireModule(s1x1=0, e1x1=1, e3x3=1)
        with pytest.raises(ValueError):
            FireModule(s1x1=1, e1x1=-1, e3x3=1)

    def test_forward_pass_and_shape(self, fire_module_config):
        """Test the forward pass and output shape of the FireModule."""
        input_shape = (4, 32, 32, 96) # batch, h, w, c
        inputs = tf.random.normal(input_shape)

        module = FireModule(**fire_module_config)
        outputs = module(inputs)

        expected_channels = fire_module_config['e1x1'] + fire_module_config['e3x3']
        expected_shape = (input_shape[0], input_shape[1], input_shape[2], expected_channels)

        assert outputs.shape == expected_shape

    def test_serialization(self, fire_module_config):
        """Test serialization and deserialization of the FireModule."""
        module = FireModule(**fire_module_config)
        config = module.get_config()

        recreated_module = FireModule.from_config(config)

        assert recreated_module.s1x1 == module.s1x1
        assert recreated_module.e1x1 == module.e1x1
        assert recreated_module.e3x3 == module.e3x3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])