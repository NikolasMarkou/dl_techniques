"""
Comprehensive test suite for the FractalNet model and its recursive blocks.

This module contains all tests for the FractalNet implementation, covering
the FractalBlock, the full FractalNet model, and the create_fractal_net
factory function. Tests include initialization, forward pass, shape
correctness, serialization, model variants, and edge cases.
"""

import pytest
import numpy as np
import tensorflow as tf
import keras
import tempfile
import os
from typing import Tuple, Dict, Any


from dl_techniques.layers.fractal_block import FractalBlock
from dl_techniques.layers.standard_blocks import ConvBlock
from dl_techniques.layers.stochastic_depth import StochasticDepth
from dl_techniques.models.fractalnet.model import FractalNet, create_fractal_net

class TestFractalNet:
    """Test suite for the full FractalNet model implementation."""

    @pytest.fixture
    def input_shape(self) -> Tuple[int, int, int]:
        """Provide a standard input shape."""
        return (32, 32, 3)

    @pytest.fixture
    def num_classes(self) -> int:
        """Provide a standard number of classes."""
        return 10

    @pytest.fixture
    def sample_data(self, input_shape: Tuple[int, int, int]):
        """Create sample batch of image data."""
        batch_size = 4
        return tf.random.normal([batch_size] + list(input_shape))

    @pytest.fixture
    def model_config(self, num_classes: int, input_shape: Tuple[int, int, int]) -> Dict[str, Any]:
        """Provide a standard model configuration."""
        return {
            "num_classes": num_classes,
            "input_shape": input_shape,
            "depths": [1, 2, 2],
            "filters": [32, 64, 128],
            "drop_path_rate": 0.1
        }

    def test_initialization_defaults(self, num_classes: int):
        """Test model initialization with default parameters."""
        model = FractalNet(num_classes=num_classes)
        assert model.num_classes == num_classes
        assert model.depths == [2, 3, 3]  # Default for 'small' variant
        assert model.filters == [32, 64, 128]
        assert model.include_top

    def test_initialization_custom(self, model_config: Dict[str, Any]):
        """Test model initialization with custom parameters."""
        model = FractalNet(**model_config)
        assert model.num_classes == model_config["num_classes"]
        assert model.depths == model_config["depths"]
        assert model.filters == model_config["filters"]
        assert model.drop_path_rate == model_config["drop_path_rate"]
        assert model._input_shape == model_config["input_shape"]

    def test_parameter_validation(self):
        """Test that invalid parameters raise ValueErrors."""
        with pytest.raises(ValueError, match="Length of depths .* must equal length of filters"):
            FractalNet(num_classes=10, depths=[1, 2], filters=[32, 64, 128])

        # FIX: Added strides=[] to properly test the "at least one stage" check
        with pytest.raises(ValueError, match="At least one stage is required"):
            FractalNet(num_classes=10, depths=[], filters=[], strides=[])

        with pytest.raises(ValueError, match="input_shape must be 3D"):
            FractalNet(num_classes=10, input_shape=(32, 32))

    def test_forward_pass_with_top(self, model_config: Dict[str, Any], sample_data: tf.Tensor):
        """Test a full forward pass with the classification head."""
        model = FractalNet(**model_config)
        output = model(sample_data, training=False)
        assert model.built
        assert output.shape == (sample_data.shape[0], model_config["num_classes"])

    def test_forward_pass_without_top(self, model_config: Dict[str, Any], sample_data: tf.Tensor):
        """Test a forward pass as a feature extractor (headless)."""
        config = model_config.copy()
        config["include_top"] = False
        model = FractalNet(**config)

        output = model(sample_data, training=False)

        # Check output shape
        # Input: 32x32 -> Stride 2 -> 16x16 -> Stride 2 -> 8x8 -> Stride 2 -> 4x4
        final_dim = config["filters"][-1]
        downsample_factor = 2 ** len(config["depths"])
        expected_size = sample_data.shape[1] // downsample_factor
        expected_shape = (sample_data.shape[0], expected_size, expected_size, final_dim)
        assert output.shape == expected_shape

    def test_from_variant_creation(self, input_shape: Tuple[int, int, int], num_classes: int):
        """Test creating models from all predefined variants."""
        for variant in FractalNet.MODEL_VARIANTS.keys():
            model = FractalNet.from_variant(variant, num_classes=num_classes, input_shape=input_shape)
            assert isinstance(model, FractalNet)
            assert model.depths == FractalNet.MODEL_VARIANTS[variant]["depths"]
            assert model.filters == FractalNet.MODEL_VARIANTS[variant]["filters"]
            print(f"✓ FractalNet-{variant} created successfully")

    def test_config_completeness(self, model_config: Dict[str, Any]):
        """Test that get_config contains all __init__ parameters."""
        model = FractalNet(**model_config)
        config = model.get_config()

        # Check all custom config parameters are present
        for key in model_config:
            # input_shape is stored as _input_shape
            if key == "input_shape":
                assert config['input_shape'] == model_config['input_shape']
                continue
            assert key in config, f"Missing {key} in get_config()"
            # Special check for serialized objects
            if "regularizer" in key or "initializer" in key:
                 continue
            assert config[key] == model_config[key]

    def test_serialization_cycle(self, model_config: Dict[str, Any], sample_data: tf.Tensor):
        """CRITICAL TEST: Test the full save and load cycle."""
        # 1. Create original model
        model = FractalNet(**model_config)

        # 2. Get prediction from original
        original_prediction = model(sample_data, training=False)

        # 3. Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_fractalnet_model.keras')
            model.save(filepath)

            # The custom objects are needed for Keras to recognize these custom layers
            # upon loading the model.
            loaded_model = keras.models.load_model(filepath, custom_objects={
                'FractalNet': FractalNet,
                'FractalBlock': FractalBlock,
                'ConvBlock': ConvBlock,
                'StochasticDepth': StochasticDepth
            })

            loaded_prediction = loaded_model(sample_data, training=False)

            # 4. Verify identical outputs
            np.testing.assert_allclose(
                original_prediction.numpy(),
                loaded_prediction.numpy(),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization"
            )

            # 5. Verify configuration is identical
            # FIX: Normalize input_shape from list to tuple for consistent comparison
            original_config = model.get_config()
            loaded_config = loaded_model.get_config()
            if 'input_shape' in loaded_config and isinstance(loaded_config['input_shape'], list):
                loaded_config['input_shape'] = tuple(loaded_config['input_shape'])

            assert original_config == loaded_config


    def test_gradients_flow(self, model_config: Dict[str, Any], sample_data: tf.Tensor):
        """Test that gradients can be computed for all trainable variables."""
        model = FractalNet(**model_config)

        with tf.GradientTape() as tape:
            # Use training=True to ensure all paths (like dropout, batchnorm) are active
            output = model(sample_data, training=True)
            # Use a simple loss for gradient checking
            loss = tf.reduce_mean(tf.square(output))

        gradients = tape.gradient(loss, model.trainable_variables)

        assert len(model.trainable_variables) > 0, "No trainable variables found"
        assert all(g is not None for g in gradients), "Some gradients are None"

    def test_training_integration(self, model_config: Dict[str, Any], sample_data: tf.Tensor, num_classes: int):
        """Test model integration with model.fit()."""
        model = FractalNet(**model_config)
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        # Create a dummy dataset
        labels = tf.random.uniform([sample_data.shape[0]], 0, num_classes, dtype=tf.int32)
        dataset = tf.data.Dataset.from_tensor_slices((sample_data, labels)).batch(2)

        # Train for one epoch
        history = model.fit(dataset, epochs=1, verbose=0)

        # Check that training metrics are recorded
        assert "loss" in history.history
        assert "accuracy" in history.history
        assert history.history["loss"][0] is not None

    def test_create_fractal_net_factory(self, input_shape: Tuple[int, int, int], num_classes: int):
        """Test the create_fractal_net factory function."""
        model = create_fractal_net(
            variant="micro",
            num_classes=num_classes,
            input_shape=input_shape
        )

        assert isinstance(model, FractalNet)
        assert model.num_classes == num_classes
        assert model._input_shape == input_shape
        assert model.depths == FractalNet.MODEL_VARIANTS["micro"]["depths"]
        assert model.filters == FractalNet.MODEL_VARIANTS["micro"]["filters"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])