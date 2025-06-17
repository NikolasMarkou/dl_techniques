"""Tests for ConvNeXt V1 model implementations."""

import os
import tempfile
import keras
import pytest
import numpy as np

from dl_techniques.models.convnext_v1 import (
    ConvNextV1,
    ConvNextV1Atto,
    ConvNextV1Femto,
    ConvNextV1Pico,
    ConvNextV1Nano,
    ConvNextV1Tiny,
    ConvNextV1Base,
    ConvNextV1Large,
    ConvNextV1Huge,
    preprocess_input,
    decode_predictions
)


class TestConvNextV1:
    """Test cases for ConvNeXt V1 models."""

    @pytest.mark.parametrize(
        "model_fn,expected_shape",
        [
            (ConvNextV1Atto, (None, 1000)),
            (ConvNextV1Femto, (None, 1000)),
            (ConvNextV1Pico, (None, 1000)),
            (ConvNextV1Nano, (None, 1000)),
            (ConvNextV1Tiny, (None, 1000)),
        ],
    )
    def test_model_output_shape(self, model_fn, expected_shape):
        """Test that models output the correct shape."""
        model = model_fn(
            include_top=True,
            weights=None,
            input_shape=(224, 224, 3),
            classes=1000
        )
        assert model.output_shape == expected_shape

    @pytest.mark.parametrize(
        "model_fn,expected_shape",
        [
            (ConvNextV1Atto, (None, 320)),
            (ConvNextV1Femto, (None, 384)),
            (ConvNextV1Pico, (None, 512)),
            (ConvNextV1Nano, (None, 640)),
            (ConvNextV1Tiny, (None, 768)),
        ],
    )
    def test_model_without_top(self, model_fn, expected_shape):
        """Test models without top layer."""
        model = model_fn(
            include_top=False,
            weights=None,
            input_shape=(224, 224, 3),
            pooling='avg'
        )
        assert model.output_shape == expected_shape

    def test_custom_input_shape(self):
        """Test models with custom input shape."""
        model = ConvNextV1Atto(
            include_top=True,
            weights=None,
            input_shape=(256, 256, 3),
            classes=1000
        )
        assert model.input_shape == (None, 256, 256, 3)

    def test_custom_num_classes(self):
        """Test models with custom number of classes."""
        num_classes = 10
        model = ConvNextV1Atto(
            include_top=True,
            weights=None,
            input_shape=(224, 224, 3),
            classes=num_classes
        )
        assert model.output_shape == (None, num_classes)

    def test_preprocess_input(self):
        """Test preprocess_input function."""
        x = np.random.random((1, 224, 224, 3))
        x_processed = preprocess_input(x)
        np.testing.assert_array_equal(x, x_processed)

    def test_layer_regularization(self):
        """Test that regularization is applied correctly."""
        regularizer = keras.regularizers.L2(1e-4)
        model = ConvNextV1Atto(
            include_top=True,
            weights=None,
            input_shape=(224, 224, 3),
            kernel_regularizer=regularizer
        )

        # Function to recursively find Conv2D layers
        def find_conv_layers(layer):
            conv_layers = []
            if isinstance(layer, keras.layers.Conv2D):
                conv_layers.append(layer)
            elif hasattr(layer, 'layers'):
                for sublayer in layer.layers:
                    conv_layers.extend(find_conv_layers(sublayer))
            return conv_layers

        # Get all Conv2D layers recursively
        conv_layers = []
        for layer in model.layers:
            conv_layers.extend(find_conv_layers(layer))

        # Check that we found convolutional layers
        assert len(conv_layers) > 0

        # Check that at least one has the regularizer
        has_regularizer = False
        for conv in conv_layers:
            if conv.kernel_regularizer is not None:
                has_regularizer = True
                break
        assert has_regularizer

    def test_model_saving(self):
        """Test model saving in .keras format."""
        model = ConvNextV1Atto(
            include_top=True,
            weights=None,
            input_shape=(224, 224, 3)
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, 'model.keras')
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)


        # Check that loaded model has the same architecture
        assert len(model.layers) == len(loaded_model.layers)
        assert model.count_params() == loaded_model.count_params()

    def test_forward_pass(self):
        """Test forward pass with random input."""
        model = ConvNextV1Atto(
            include_top=True,
            weights=None,
            input_shape=(224, 224, 3)
        )

        # Create random input
        x = np.random.random((1, 224, 224, 3))

        # Perform forward pass
        y = model.predict(x)

        # Check output shape
        assert y.shape == (1, 1000)

        # Check that output is a probability distribution
        assert np.isclose(np.sum(y[0]), 1.0, rtol=1e-5)
        assert np.all(y >= 0) and np.all(y <= 1)