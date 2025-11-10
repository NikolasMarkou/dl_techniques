"""
Comprehensive test suite for the SqueezeNodule-Net V2 model.

This module contains all tests for the SqueezeNoduleNetV2 model implementation,
covering the SimplifiedFireModule, 2D and 3D variants, initialization,
forward pass, training, serialization, and edge cases.
"""

import pytest
import numpy as np
import tensorflow as tf
import keras
import tempfile
import os
from typing import Tuple, Dict, Any

from dl_techniques.models.squeezenet.squeezenet_v2 import SqueezeNoduleNetV2, SimplifiedFireModule, create_squeezenodule_net_v2


class TestSqueezeNoduleNetV2:
    """Test suite for SqueezeNodule-Net V2 model implementation."""

    @pytest.fixture
    def input_shape_2d(self) -> Tuple[int, int, int]:
        """Create test input shape for 2D images."""
        return (64, 64, 3)

    @pytest.fixture
    def input_shape_3d(self) -> Tuple[int, int, int, int]:
        """Create test input shape for 3D volumes."""
        # MODIFIED: Increased size from 32 to 64 to prevent NaN from pooling.
        return (64, 64, 64, 1)

    @pytest.fixture
    def num_classes(self) -> int:
        """Create test number of classes."""
        return 10

    @pytest.fixture
    def num_classes_binary(self) -> int:
        """Create test number of classes for binary classification."""
        return 2

    @pytest.fixture
    def sample_data_2d(self, input_shape_2d):
        """Create sample 2D input data."""
        batch_size = 2
        return tf.random.uniform([batch_size] + list(input_shape_2d), 0, 1)

    @pytest.fixture
    def sample_data_3d(self, input_shape_3d):
        """Create sample 3D input data."""
        batch_size = 2
        return tf.random.uniform([batch_size] + list(input_shape_3d), 0, 1)

    def test_initialization_defaults(self, num_classes, input_shape_2d):
        """Test initialization with default parameters (should be variant 'v2')."""
        model = SqueezeNoduleNetV2(num_classes=num_classes, input_shape=input_shape_2d)

        default_config = SqueezeNoduleNetV2.MODEL_VARIANTS["v2"]
        assert model.num_classes == num_classes
        assert model.use_3d is False
        assert model.dropout_rate == 0.5
        assert model.conv1_filters == default_config["conv1_filters"]
        # Check first fire module config to confirm it's v2
        assert model.fire_configs[0]['s1x1'] == 32
        assert model.fire_configs[0]['e3x3'] == 64

    def test_initialization_custom(self, num_classes, input_shape_2d):
        """Test initialization with custom parameters."""
        regularizer = keras.regularizers.L2(0.01)
        initializer = keras.initializers.RandomNormal()

        model = SqueezeNoduleNetV2(
            num_classes=num_classes,
            dropout_rate=0.25,
            kernel_regularizer=regularizer,
            kernel_initializer=initializer,
            include_top=False,
            input_shape=input_shape_2d
        )

        assert model.num_classes == num_classes
        assert model.dropout_rate == 0.25
        assert model.kernel_regularizer == regularizer
        assert isinstance(model.kernel_initializer, type(initializer))
        assert model.include_top is False

    @pytest.mark.parametrize(
        "variant, expected_s1x1_fire2, use_3d",
        [
            ("v1", 16, False),
            ("v2", 32, False),
            ("v1_3d", 16, True),
            ("v2_3d", 32, True),
        ],
    )
    def test_model_variants(self, variant, expected_s1x1_fire2, use_3d, num_classes, input_shape_2d, input_shape_3d):
        """Test all predefined model variants."""
        # MODIFIED: Use fixtures for input shapes for robustness.
        input_shape = input_shape_3d if use_3d else input_shape_2d
        model = SqueezeNoduleNetV2.from_variant(variant, num_classes=num_classes, input_shape=input_shape)

        assert model.use_3d is use_3d
        assert model.fire_configs[0]['s1x1'] == expected_s1x1_fire2

    def test_invalid_variant(self, num_classes):
        """Test invalid variant raises error."""
        with pytest.raises(ValueError, match="Unknown variant"):
            SqueezeNoduleNetV2.from_variant("invalid_variant", num_classes=num_classes)

    def test_forward_pass_2d_with_top(self, num_classes, sample_data_2d):
        """Test 2D forward pass with classification head."""
        model = SqueezeNoduleNetV2(num_classes=num_classes, input_shape=sample_data_2d.shape[1:])
        outputs = model(sample_data_2d)

        expected_shape = (sample_data_2d.shape[0], num_classes)
        assert outputs.shape == expected_shape
        assert not np.any(np.isnan(outputs.numpy()))

    def test_forward_pass_3d_with_top(self, num_classes, sample_data_3d):
        """Test 3D forward pass with classification head."""
        model = SqueezeNoduleNetV2.from_variant("v2_3d", num_classes=num_classes, input_shape=sample_data_3d.shape[1:])
        outputs = model(sample_data_3d)

        expected_shape = (sample_data_3d.shape[0], num_classes)
        assert outputs.shape == expected_shape
        assert not np.any(np.isnan(outputs.numpy()))

    def test_forward_pass_without_top(self, input_shape_2d, sample_data_2d):
        """Test forward pass without classification head (feature extraction)."""
        model = SqueezeNoduleNetV2(include_top=False, input_shape=input_shape_2d)
        features = model(sample_data_2d)

        assert len(features.shape) == 4
        last_fire_config = SqueezeNoduleNetV2.MODEL_VARIANTS["v2"]["fire_configs"][-1]
        expected_channels = last_fire_config['e3x3']
        assert features.shape[-1] == expected_channels

    def test_binary_classification_head(self, num_classes_binary, sample_data_2d):
        """Test that binary classification uses sigmoid activation."""
        model = SqueezeNoduleNetV2(num_classes=num_classes_binary, input_shape=sample_data_2d.shape[1:])

        # The last layer should be an Activation layer with sigmoid
        assert isinstance(model.head_layers[-1], keras.layers.Activation)
        assert model.head_layers[-1].activation.__name__ == 'sigmoid'

        outputs = model(sample_data_2d)
        assert outputs.shape == (sample_data_2d.shape[0], num_classes_binary)
        assert np.all(outputs.numpy() >= 0) and np.all(outputs.numpy() <= 1)

    def test_model_compilation(self, num_classes, input_shape_2d):
        """Test model compilation with different optimizers and losses."""
        model = SqueezeNoduleNetV2(num_classes=num_classes, input_shape=input_shape_2d)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        assert model.optimizer is not None

    def test_serialization_2d(self, num_classes, input_shape_2d):
        """Test serialization and deserialization of the 2D model."""
        original_model = SqueezeNoduleNetV2(num_classes=num_classes, input_shape=input_shape_2d)
        config = original_model.get_config()
        recreated_model = SqueezeNoduleNetV2.from_config(config)

        assert recreated_model.num_classes == original_model.num_classes
        assert recreated_model.use_3d == original_model.use_3d
        assert recreated_model.get_config()['variant_config'] == original_model.get_config()['variant_config']

    def test_model_save_load_2d(self, num_classes, sample_data_2d):
        """Test saving and loading a 2D SqueezeNodule-Net model."""
        model = SqueezeNoduleNetV2(num_classes=num_classes, input_shape=sample_data_2d.shape[1:])
        original_outputs = model(sample_data_2d)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "squeezenodule_model.keras")
            model.save(model_path)

            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={"SqueezeNoduleNetV2": SqueezeNoduleNetV2, "SimplifiedFireModule": SimplifiedFireModule}
            )
            loaded_outputs = loaded_model(sample_data_2d)
            np.testing.assert_allclose(original_outputs.numpy(), loaded_outputs.numpy(), rtol=1e-6)

    def test_model_save_load_3d(self, num_classes, sample_data_3d):
        """Test saving and loading a 3D SqueezeNodule-Net model."""
        model = SqueezeNoduleNetV2.from_variant("v2_3d", num_classes=num_classes, input_shape=sample_data_3d.shape[1:])
        original_outputs = model(sample_data_3d)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "squeezenodule_3d_model.keras")
            model.save(model_path)

            # 3D models with sequential fire modules don't require custom objects if built-in layers are used
            loaded_model = keras.models.load_model(model_path)
            loaded_outputs = loaded_model(sample_data_3d)
            np.testing.assert_allclose(original_outputs.numpy(), loaded_outputs.numpy(), rtol=1e-6)

    def test_training_integration(self, num_classes, input_shape_2d, sample_data_2d):
        """Test training integration with a small dataset."""
        model = SqueezeNoduleNetV2(num_classes=num_classes, input_shape=input_shape_2d)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        labels = tf.random.uniform([sample_data_2d.shape[0]], 0, num_classes, dtype=tf.int32)
        history = model.fit(sample_data_2d, labels, epochs=2, verbose=0)

        assert "loss" in history.history
        assert len(history.history["loss"]) == 2

    def test_gradient_flow(self, num_classes, input_shape_2d, sample_data_2d):
        """Test gradient flow through the model."""
        model = SqueezeNoduleNetV2(num_classes=num_classes, input_shape=input_shape_2d)
        labels = tf.random.uniform([sample_data_2d.shape[0]], 0, num_classes, dtype=tf.int32)

        with tf.GradientTape() as tape:
            outputs = model(sample_data_2d, training=True)
            loss = keras.losses.sparse_categorical_crossentropy(labels, outputs)

        grads = tape.gradient(loss, model.trainable_variables)
        assert all(g is not None for g in grads)
        assert np.any(grads[0].numpy() != 0) # Check conv1 kernel gradients

    def test_create_squeezenodule_net_v2_factory(self, num_classes, input_shape_2d):
        """Test the convenience factory function."""
        model = create_squeezenodule_net_v2(variant="v1", num_classes=num_classes, input_shape=input_shape_2d)
        assert isinstance(model, SqueezeNoduleNetV2)
        assert model.fire_configs[0]['s1x1'] == 16 # Check for V1 config

    def test_numerical_stability(self, num_classes, input_shape_2d):
        """Test model stability with extreme input values."""
        model = SqueezeNoduleNetV2(num_classes=num_classes, input_shape=input_shape_2d)
        test_cases = [
            tf.zeros((2,) + input_shape_2d),
            tf.ones((2,) + input_shape_2d),
        ]
        for i, test_input in enumerate(test_cases):
            outputs = model(test_input)
            assert not np.any(np.isnan(outputs.numpy())), f"NaN in outputs for test case {i}"
            assert not np.any(np.isinf(outputs.numpy())), f"Inf in outputs for test case {i}"

class TestSimplifiedFireModule:
    """Test suite for the SimplifiedFireModule layer."""

    @pytest.fixture
    def fire_module_config(self) -> Dict[str, Any]:
        return {'s1x1': 16, 'e3x3': 64}

    def test_initialization(self, fire_module_config):
        """Test SimplifiedFireModule initialization."""
        module = SimplifiedFireModule(**fire_module_config)
        assert module.s1x1 == fire_module_config['s1x1']
        assert module.e3x3 == fire_module_config['e3x3']
        assert module.squeeze.filters == fire_module_config['s1x1']
        assert module.expand_3x3.filters == fire_module_config['e3x3']

    def test_invalid_initialization(self):
        """Test that invalid filter counts raise an error."""
        with pytest.raises(ValueError, match="must be positive integers"):
            SimplifiedFireModule(s1x1=0, e3x3=1)
        with pytest.raises(ValueError, match="Squeeze filters should be less than expand filters"):
            SimplifiedFireModule(s1x1=16, e3x3=16)
        with pytest.raises(ValueError, match="Squeeze filters should be less than expand filters"):
            SimplifiedFireModule(s1x1=32, e3x3=16)

    def test_forward_pass_and_shape(self, fire_module_config):
        """Test the forward pass and output shape of the SimplifiedFireModule."""
        input_shape = (4, 32, 32, 96)
        inputs = tf.random.normal(input_shape)

        module = SimplifiedFireModule(**fire_module_config)
        outputs = module(inputs)

        expected_shape = (input_shape[0], input_shape[1], input_shape[2], fire_module_config['e3x3'])
        assert outputs.shape == expected_shape

    def test_serialization(self, fire_module_config):
        """Test serialization and deserialization of the SimplifiedFireModule."""
        module = SimplifiedFireModule(**fire_module_config)
        config = module.get_config()
        recreated_module = SimplifiedFireModule.from_config(config)

        assert recreated_module.s1x1 == module.s1x1
        assert recreated_module.e3x3 == module.e3x3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])