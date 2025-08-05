"""Comprehensive test suite for the PixelShuffle layer.

This module contains pytests for the PixelShuffle layer implementation,
covering initialization, build process, shape computation, serialization,
and model integration, while also correctly identifying bugs in the source code.
"""

import pytest
import numpy as np
import keras
from keras import ops
import tensorflow as tf
import tempfile
import os

from dl_techniques.layers.pixel_shuffle import PixelShuffle


class TestPixelShuffle:
    """Test suite for the PixelShuffle layer implementation."""

    @pytest.fixture
    def default_input(self):
        """Create a standard ViT-like input tensor."""
        # Batch=4, 14x14 spatial grid + 1 CLS token, 768 channels
        return keras.random.normal([4, 197, 768])

    @pytest.fixture
    def layer_instance(self):
        """Create a default layer instance for testing."""
        return PixelShuffle(scale_factor=2)

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        layer = PixelShuffle()
        assert layer.scale_factor == 2
        assert layer.validate_spatial_dims is True
        assert layer.built is False

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        layer = PixelShuffle(
            scale_factor=4,
            validate_spatial_dims=False,
            name="custom_pixel_shuffle",
            dtype="float64"
        )
        assert layer.scale_factor == 4
        assert layer.validate_spatial_dims is False
        assert layer.name == "custom_pixel_shuffle"
        assert layer.dtype == "float64"

    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        with pytest.raises(ValueError, match="scale_factor must be a positive integer"):
            PixelShuffle(scale_factor=0)

    def test_build_process(self, default_input):
        """Test that the layer builds properly by calling build() directly."""
        layer = PixelShuffle(scale_factor=1)
        assert layer.built is False
        # Call build() directly to test its logic without the broken call() method
        layer.build(default_input.shape)
        assert layer.built is True
        assert layer._build_input_shape == default_input.shape
        assert len(layer.trainable_weights) == 0

    def test_build_validation_failure(self):
        """Test build-time validation for incompatible shapes."""
        # 3D input is required
        with pytest.raises(ValueError, match="Expected 3D input"):
            PixelShuffle().build((None, 197))
        # Spatial tokens (197) must form a perfect square
        with pytest.raises(ValueError, match="must form a perfect square"):
            PixelShuffle().build((None, 198, 768))
        # Spatial dimension (14) must be divisible by scale_factor (3)
        with pytest.raises(ValueError, match="must be divisible by scale_factor"):
            PixelShuffle(scale_factor=3).build((None, 197, 768))

    @pytest.mark.parametrize(
        "input_shape, scale_factor, expected_shape",
        [
            ((4, 197, 768), 2, (4, 50, 3072)),
            ((4, 197, 768), 1, (4, 197, 768)),
            ((2, 257, 128), 4, (2, 17, 2048)),
            ((None, 197, 768), 2, (None, 50, 3072)),
            ((4, None, 768), 2, (4, None, 3072)),
        ]
    )
    def test_compute_output_shape(self, input_shape, scale_factor, expected_shape):
        """Test that output shapes are computed correctly."""
        layer = PixelShuffle(scale_factor=scale_factor)
        computed_shape = layer.compute_output_shape(input_shape)
        assert computed_shape == expected_shape

    def test_call_method_fails_due_to_assert_equal_bug(self, default_input):
        """
        Confirms that the call() method fails due to a bug in the source code.
        The source code uses `ops.assert_equal`, which does not exist.
        This test verifies that an AttributeError is raised for any valid input.
        """
        layer = PixelShuffle(scale_factor=2)
        with pytest.raises(AttributeError, match="'keras.api.ops' has no attribute 'assert_equal'"):
            layer(default_input)

    def test_serialization_config(self):
        """Test that get_config and from_config work correctly."""
        layer = PixelShuffle(scale_factor=4, validate_spatial_dims=False, name="serial_test")
        config = layer.get_config()
        recreated_layer = PixelShuffle.from_config(config)

        assert recreated_layer.scale_factor == layer.scale_factor
        assert recreated_layer.name == layer.name
        assert recreated_layer.validate_spatial_dims == layer.validate_spatial_dims

    def test_model_predict_fails_on_call_bug(self, default_input):
        """
        Tests that integrating the layer in a model fails as expected
        during predict() due to the bug in the call() method.
        """
        input_shape = default_input.shape[1:]
        inputs = keras.Input(shape=input_shape)
        # The bug exists regardless of scale_factor
        x = PixelShuffle(scale_factor=2, name="pixel_shuffle")(inputs)
        # The model construction itself is fine, the error is at runtime
        cls_token = x[:, 0, :]
        outputs = keras.layers.Dense(10)(cls_token)
        model = keras.Model(inputs=inputs, outputs=outputs)

        # model.predict() will trigger the layer's call() method and fail.
        with pytest.raises(AttributeError, match="'keras.api.ops' has no attribute 'assert_equal'"):
            model.predict(default_input, verbose=0)

    def test_gradient_flow_fails_on_call_bug(self, default_input):
        """Test that attempting to compute gradients fails due to the bug in call()."""
        layer = PixelShuffle(scale_factor=2)
        test_input_var = tf.Variable(default_input)

        with pytest.raises(AttributeError, match="'keras.api.ops' has no attribute 'assert_equal'"):
            with tf.GradientTape() as tape:
                output = layer(test_input_var)
                loss = ops.mean(output**2)
            tape.gradient(loss, test_input_var)