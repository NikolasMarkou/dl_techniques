"""
Comprehensive test suite for the CBAMNet model and its attention layers.

This module contains all tests for the CBAMNet implementation, covering
the ChannelAttention, SpatialAttention, and CBAM layers, as well as the
full CBAMNet model. Tests include initialization, forward pass, shape
correctness, serialization, model variants, and edge cases.
"""

import pytest
import numpy as np
import tensorflow as tf
import keras
import tempfile
import os
from typing import Tuple, Dict, Any

from dl_techniques.models.cbam.model import CBAMNet, create_cbam_net
from dl_techniques.layers.attention.channel_attention import ChannelAttention
from dl_techniques.layers.attention.spatial_attention import SpatialAttention
from dl_techniques.layers.attention.convolutional_block_attention import CBAM


class TestCBAMNet:
    """Test suite for the full CBAMNet model implementation."""

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
            "dims": [32, 64],
            "attention_ratio": 8,
            "attention_kernel_size": 7
        }

    def test_initialization_defaults(self, num_classes: int):
        """Test model initialization with default parameters."""
        model = CBAMNet(num_classes=num_classes)
        assert model.num_classes == num_classes
        assert model.dims == [64, 128]
        assert model.include_top

    def test_initialization_custom(self, model_config: Dict[str, Any]):
        """Test model initialization with custom parameters."""
        model = CBAMNet(**model_config)
        assert model.num_classes == model_config["num_classes"]
        assert model.dims == model_config["dims"]
        assert model.input_shape_arg == model_config["input_shape"]

    def test_parameter_validation(self):
        """Test that invalid parameters raise ValueErrors."""
        with pytest.raises(ValueError, match="num_classes must be positive"):
            CBAMNet(num_classes=0)
        with pytest.raises(ValueError, match="dims must be a non-empty list"):
            CBAMNet(num_classes=10, dims=[])
        with pytest.raises(ValueError, match="dims must be a non-empty list"):
            CBAMNet(num_classes=10, dims=[64, 0])
        with pytest.raises(ValueError, match="attention_ratio must be positive"):
            CBAMNet(num_classes=10, attention_ratio=-1)
        with pytest.raises(ValueError, match="attention_kernel_size must be positive"):
            CBAMNet(num_classes=10, attention_kernel_size=0)

    def test_forward_pass_with_top(self, model_config: Dict[str, Any], sample_data: tf.Tensor):
        """Test a full forward pass with the classification head."""
        model = CBAMNet(**model_config)
        output = model(sample_data)
        assert model.built
        assert output.shape == (sample_data.shape[0], model_config["num_classes"])
        # Output should be probabilities from softmax
        assert tf.reduce_all(output >= 0.0) and tf.reduce_all(output <= 1.0)
        assert np.allclose(tf.reduce_sum(output, axis=-1), 1.0)

    def test_forward_pass_without_top(self, model_config: Dict[str, Any], sample_data: tf.Tensor):
        """Test a forward pass as a feature extractor (headless)."""
        config = model_config.copy()
        config["include_top"] = False
        model = CBAMNet(**config)

        output = model(sample_data)

        # Check output shape
        # Input: 32x32 -> Pool -> 16x16 -> Pool -> 8x8
        final_dim = config["dims"][-1]
        expected_shape = (sample_data.shape[0], 8, 8, final_dim)
        assert output.shape == expected_shape

    def test_from_variant_creation(self, input_shape: Tuple[int, int, int], num_classes: int):
        """Test creating models from all predefined variants."""
        for variant in CBAMNet.MODEL_VARIANTS.keys():
            model = CBAMNet.from_variant(variant, num_classes=num_classes, input_shape=input_shape)
            assert isinstance(model, CBAMNet)
            assert model.dims == CBAMNet.MODEL_VARIANTS[variant]["dims"]
            print(f"✓ CBAMNet-{variant} created successfully")

    def test_config_completeness(self, model_config: Dict[str, Any]):
        """Test that get_config contains all __init__ parameters."""
        model = CBAMNet(**model_config)
        config = model.get_config()

        # Check all custom config parameters are present
        for key in model_config:
            # Skip input_shape as it's stored differently
            if key == "input_shape":
                assert config['input_shape'] == model_config['input_shape']
                continue
            assert key in config, f"Missing {key} in get_config()"

    def test_serialization_cycle(self, model_config: Dict[str, Any], sample_data: tf.Tensor):
        """CRITICAL TEST: Test the full save and load cycle."""
        # 1. Create original model
        model = CBAMNet(**model_config)

        # 2. Get prediction from original
        original_prediction = model(sample_data, training=False)

        # 3. Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_cbamnet_model.keras')
            model.save(filepath)

            # The custom objects are needed if the layers are not globally registered
            # in the test environment.
            loaded_model = keras.models.load_model(filepath, custom_objects={
                'CBAMNet': CBAMNet,
                'CBAM': CBAM,
                'ChannelAttention': ChannelAttention,
                'SpatialAttention': SpatialAttention
            })

            loaded_prediction = loaded_model(sample_data, training=False)

            # 4. Verify identical outputs
            np.testing.assert_allclose(
                original_prediction.numpy(),
                loaded_prediction.numpy(),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization"
            )

    def test_gradients_flow(self, model_config: Dict[str, Any], sample_data: tf.Tensor):
        """Test that gradients can be computed for all trainable variables."""
        model = CBAMNet(**model_config)

        with tf.GradientTape() as tape:
            tape.watch(sample_data)
            output = model(sample_data, training=True)
            # Use a simple loss for gradient checking
            loss = tf.reduce_mean(tf.square(output))

        gradients = tape.gradient(loss, model.trainable_variables)

        assert all(g is not None for g in gradients), "Some gradients are None"
        assert len(gradients) > 0, "No trainable variables found"

    def test_training_integration(self, model_config: Dict[str, Any], sample_data: tf.Tensor, num_classes: int):
        """Test model integration with model.fit()."""
        model = CBAMNet(**model_config)
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

    def test_create_cbam_net_factory(self, input_shape: Tuple[int, int, int], num_classes: int):
        """Test the create_cbam_net factory function."""
        model = create_cbam_net(
            variant="tiny",
            num_classes=num_classes,
            input_shape=input_shape
        )

        assert isinstance(model, CBAMNet)
        assert model.num_classes == num_classes
        assert model.input_shape_arg == input_shape
        assert model.dims == CBAMNet.MODEL_VARIANTS["tiny"]["dims"]
        assert not model.built


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

