"""
Comprehensive test suite for the ConvUNext Model.

Expanded test coverage including edge cases, serialization,
gradient flow, regularization, and integration scenarios.
"""

import pytest
import numpy as np
import tensorflow as tf
import keras
from keras import ops
import tempfile
import os
from typing import Tuple, List

from dl_techniques.models.convunext.model import (
    ConvUNextModel,
    ConvUNextStem,
    create_convunext_variant,
    create_inference_model_from_training_model
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def input_shape() -> Tuple[int, int, int]:
    """Standard input shape for testing."""
    return (128, 128, 3)


@pytest.fixture
def odd_input_shape() -> Tuple[int, int, int]:
    """Odd-sized input shape to test resize logic."""
    return (129, 129, 3)


@pytest.fixture
def small_input_shape() -> Tuple[int, int, int]:
    """Small input shape for faster tests."""
    return (32, 32, 3)


@pytest.fixture
def tiny_input_shape() -> Tuple[int, int, int]:
    """Tiny input shape for minimal tests."""
    return (16, 16, 3)


@pytest.fixture
def small_sample_data(small_input_shape):
    """Generate small sample data batch."""
    batch_size = 2
    return tf.random.uniform([batch_size] + list(small_input_shape), 0, 1)


@pytest.fixture
def tiny_sample_data(tiny_input_shape):
    """Generate tiny sample data batch."""
    batch_size = 2
    return tf.random.uniform([batch_size] + list(tiny_input_shape), 0, 1)


@pytest.fixture
def grayscale_input_shape() -> Tuple[int, int, int]:
    """Grayscale input shape."""
    return (32, 32, 1)


@pytest.fixture
def multichannel_input_shape() -> Tuple[int, int, int]:
    """Multi-channel input shape."""
    return (32, 32, 8)


# ============================================================================
# ConvUNextStem Tests
# ============================================================================

class TestConvUNextStem:
    """Test suite for ConvUNextStem layer."""

    def test_stem_initialization(self):
        """Test stem layer initialization with default parameters."""
        stem = ConvUNextStem(filters=64)
        assert stem.filters == 64
        assert stem.kernel_size == 7

    def test_stem_custom_kernel_size(self):
        """Test stem layer with custom kernel size."""
        stem = ConvUNextStem(filters=64, kernel_size=3)
        assert stem.kernel_size == 3

    def test_stem_tuple_kernel_size(self):
        """Test stem layer with tuple kernel size."""
        stem = ConvUNextStem(filters=64, kernel_size=(5, 5))
        assert stem.kernel_size == (5, 5)

    def test_stem_forward_pass(self, small_input_shape):
        """Test stem layer forward pass."""
        stem = ConvUNextStem(filters=64, kernel_size=7)
        inputs = tf.random.uniform([2] + list(small_input_shape))
        outputs = stem(inputs)
        assert outputs.shape == (2, 32, 32, 64)

    def test_stem_output_shape_computation(self, small_input_shape):
        """Test stem compute_output_shape method."""
        stem = ConvUNextStem(filters=48)
        input_shape = (None,) + small_input_shape
        output_shape = stem.compute_output_shape(input_shape)
        assert output_shape == (None, 32, 32, 48)

    def test_stem_serialization(self):
        """Test stem layer serialization and deserialization."""
        original = ConvUNextStem(filters=64, kernel_size=5)
        config = original.get_config()
        recreated = ConvUNextStem.from_config(config)
        assert recreated.filters == original.filters
        assert recreated.kernel_size == original.kernel_size

    def test_stem_with_regularization(self, small_input_shape):
        """Test stem layer with kernel regularization."""
        stem = ConvUNextStem(
            filters=64,
            kernel_regularizer='l2'
        )
        inputs = tf.random.uniform([1] + list(small_input_shape))
        outputs = stem(inputs)
        assert outputs.shape == (1, 32, 32, 64)

    def test_stem_preserves_spatial_dimensions(self):
        """Test that stem preserves spatial dimensions with 'same' padding."""
        stem = ConvUNextStem(filters=32, kernel_size=7)
        for h, w in [(32, 32), (64, 64), (128, 128)]:
            inputs = tf.random.uniform([1, h, w, 3])
            outputs = stem(inputs)
            assert outputs.shape[1:3] == (h, w)


# ============================================================================
# ConvUNextModel Initialization Tests
# ============================================================================

class TestConvUNextModelInitialization:
    """Test suite for ConvUNextModel initialization."""

    def test_initialization_defaults(self, input_shape):
        """Test model initialization with default parameters."""
        model = ConvUNextModel(input_shape=input_shape)
        assert model.depth == 4
        assert model.output_channels == 3
        assert model.initial_filters == 64
        assert model.convnext_version == 'v2'

    def test_initialization_custom(self, small_input_shape):
        """Test model initialization with custom parameters."""
        model = ConvUNextModel(
            input_shape=small_input_shape,
            depth=3,
            initial_filters=32,
            output_channels=1,
            convnext_version='v1'
        )
        assert model.depth == 3
        assert model.output_channels == 1
        assert model.initial_filters == 32
        assert model.convnext_version == 'v1'

    def test_initialization_minimum_depth(self, small_input_shape):
        """Test model with minimum allowed depth."""
        model = ConvUNextModel(input_shape=small_input_shape, depth=2)
        assert model.depth == 2

    def test_initialization_invalid_depth(self, small_input_shape):
        """Test that depth < 2 raises ValueError."""
        with pytest.raises(ValueError, match="Depth must be >= 2"):
            ConvUNextModel(input_shape=small_input_shape, depth=1)

    def test_initialization_grayscale_input(self, grayscale_input_shape):
        """Test model initialization with grayscale input."""
        model = ConvUNextModel(input_shape=grayscale_input_shape, depth=2)
        assert model.input_channels == 1
        assert model.output_channels == 1

    def test_initialization_multichannel_input(self, multichannel_input_shape):
        """Test model initialization with multi-channel input."""
        model = ConvUNextModel(input_shape=multichannel_input_shape, depth=2)
        assert model.input_channels == 8
        assert model.output_channels == 8

    def test_filter_sizes_computation(self, small_input_shape):
        """Test filter sizes are computed correctly."""
        model = ConvUNextModel(
            input_shape=small_input_shape,
            depth=3,
            initial_filters=32,
            filter_multiplier=2
        )
        expected_filters = [32, 64, 128, 256]  # depth+1 levels
        assert model.filter_sizes == expected_filters

    def test_initialization_with_regularization(self, small_input_shape):
        """Test model initialization with kernel regularization."""
        model = ConvUNextModel(
            input_shape=small_input_shape,
            depth=2,
            kernel_regularizer='l2'
        )
        assert model.kernel_regularizer is not None


# ============================================================================
# Model Variants Tests
# ============================================================================

class TestModelVariants:
    """Test suite for predefined model variants."""

    @pytest.mark.parametrize(
        "variant,expected_depth,expected_filters",
        [
            ("tiny", 3, 32),
            ("small", 3, 48),
            ("base", 4, 64),
            ("large", 4, 96),
            ("xlarge", 5, 128),
        ],
    )
    def test_model_variants(self, variant, expected_depth, expected_filters):
        """Test all predefined model variants."""
        model = ConvUNextModel.from_variant(
            variant,
            input_shape=(64, 64, 3)
        )
        assert model.depth == expected_depth
        assert model.initial_filters == expected_filters

    def test_invalid_variant(self):
        """Test that invalid variant name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown variant"):
            ConvUNextModel.from_variant("invalid_variant")

    @pytest.mark.parametrize("variant", ["tiny", "small", "base", "large", "xlarge"])
    def test_variant_with_custom_output_channels(self, variant):
        """Test variants with custom output channels."""
        model = ConvUNextModel.from_variant(
            variant,
            input_shape=(64, 64, 3),
            output_channels=10
        )
        assert model.output_channels == 10

    @pytest.mark.parametrize("variant", ["tiny", "small", "base"])
    def test_variant_with_deep_supervision(self, variant):
        """Test variants with deep supervision enabled."""
        model = ConvUNextModel.from_variant(
            variant,
            input_shape=(32, 32, 3),
            enable_deep_supervision=True
        )
        assert model.enable_deep_supervision is True

    def test_variant_override_parameters(self):
        """Test that variant parameters can be overridden."""
        model = ConvUNextModel.from_variant(
            'tiny',
            input_shape=(64, 64, 3),
            blocks_per_level=4,  # Override default
            drop_path_rate=0.2   # Override default
        )
        assert model.blocks_per_level == 4
        assert model.drop_path_rate == 0.2


# ============================================================================
# Forward Pass Tests
# ============================================================================

class TestForwardPass:
    """Test suite for model forward pass."""

    def test_forward_pass_no_supervision(self, small_input_shape, small_sample_data):
        """Test forward pass without deep supervision."""
        model = ConvUNextModel(
            input_shape=small_input_shape,
            depth=3,
            initial_filters=32,
            enable_deep_supervision=False,
            output_channels=3
        )
        outputs = model(small_sample_data)
        assert isinstance(outputs, (tf.Tensor, keras.KerasTensor))
        assert outputs.shape == small_sample_data.shape

    def test_forward_pass_deep_supervision(self, small_input_shape, small_sample_data):
        """Test forward pass with deep supervision enabled."""
        depth = 3
        model = ConvUNextModel(
            input_shape=small_input_shape,
            depth=depth,
            initial_filters=32,
            enable_deep_supervision=True,
            output_channels=3
        )
        outputs = model(small_sample_data)
        assert isinstance(outputs, list)
        expected_outputs = 1 + (depth - 1)  # Main + aux outputs
        assert len(outputs) == expected_outputs

    def test_forward_pass_training_mode(self, small_input_shape, small_sample_data):
        """Test forward pass in training mode."""
        model = ConvUNextModel(
            input_shape=small_input_shape,
            depth=2,
            initial_filters=32
        )
        outputs = model(small_sample_data, training=True)
        assert outputs.shape == small_sample_data.shape

    def test_forward_pass_inference_mode(self, small_input_shape, small_sample_data):
        """Test forward pass in inference mode."""
        model = ConvUNextModel(
            input_shape=small_input_shape,
            depth=2,
            initial_filters=32
        )
        outputs = model(small_sample_data, training=False)
        assert outputs.shape == small_sample_data.shape

    def test_odd_input_shapes_resize_logic(self, odd_input_shape):
        """Test that model handles odd input shapes correctly."""
        model = ConvUNextModel(
            input_shape=odd_input_shape,
            depth=2,
            initial_filters=32,
            output_channels=3
        )
        batch_data = tf.random.uniform([1] + list(odd_input_shape), 0, 1)
        outputs = model(batch_data)
        assert outputs.shape == (1, 129, 129, 3)

    def test_output_channel_configuration(self, small_input_shape):
        """Test model with custom output channels."""
        model = ConvUNextModel(
            input_shape=small_input_shape,
            output_channels=10,
            depth=2,
            initial_filters=32
        )
        batch_data = tf.random.uniform([2] + list(small_input_shape), 0, 1)
        outputs = model(batch_data)
        assert outputs.shape[-1] == 10

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_variable_batch_sizes(self, small_input_shape, batch_size):
        """Test model handles various batch sizes."""
        model = ConvUNextModel(
            input_shape=small_input_shape,
            depth=2,
            initial_filters=32
        )
        batch_data = tf.random.uniform([batch_size] + list(small_input_shape))
        outputs = model(batch_data)
        assert outputs.shape[0] == batch_size

    def test_variable_input_size(self):
        """Test model with variable spatial input dimensions."""
        model = ConvUNextModel(
            input_shape=(None, None, 3),
            depth=2,
            initial_filters=32
        )
        out1 = model(tf.random.uniform((1, 64, 64, 3)))
        assert out1.shape == (1, 64, 64, 3)
        out2 = model(tf.random.uniform((1, 32, 32, 3)))
        assert out2.shape == (1, 32, 32, 3)

    def test_multiple_forward_passes(self, small_input_shape, small_sample_data):
        """Test multiple consecutive forward passes."""
        model = ConvUNextModel(
            input_shape=small_input_shape,
            depth=2,
            initial_filters=32
        )
        for _ in range(3):
            outputs = model(small_sample_data)
            assert outputs.shape == small_sample_data.shape

    @pytest.mark.parametrize("output_channels", [1, 3, 10, 64])
    def test_different_output_channels(self, small_input_shape, output_channels):
        """Test model with different output channel configurations."""
        model = ConvUNextModel(
            input_shape=small_input_shape,
            depth=2,
            initial_filters=32,
            output_channels=output_channels
        )
        inputs = tf.random.uniform([1] + list(small_input_shape))
        outputs = model(inputs)
        assert outputs.shape[-1] == output_channels


# ============================================================================
# Gradient Flow Tests
# ============================================================================

class TestGradientFlow:
    """Test suite for gradient flow and backpropagation."""

    def test_gradient_flow_basic(self, small_input_shape, small_sample_data):
        """Test basic gradient flow through model."""
        model = ConvUNextModel(
            input_shape=small_input_shape,
            depth=2,
            initial_filters=32,
            enable_deep_supervision=False
        )

        inputs = tf.ones_like(small_sample_data)
        targets = tf.zeros_like(small_sample_data)

        with tf.GradientTape() as tape:
            outputs = model(inputs, training=True)
            loss = keras.losses.mean_squared_error(targets, outputs)
            loss = tf.reduce_mean(loss)

        grads = tape.gradient(loss, model.trainable_weights)

        assert len(grads) > 0
        assert all(g is not None for g in grads)

    def test_gradient_flow_deep_supervision(self, small_input_shape):
        """Test gradient flow with deep supervision."""
        model = ConvUNextModel(
            input_shape=small_input_shape,
            depth=3,
            initial_filters=32,
            enable_deep_supervision=True
        )

        inputs = tf.random.uniform([2] + list(small_input_shape))
        targets = [
            tf.random.uniform([2] + list(small_input_shape)),
            tf.random.uniform([2, 16, 16, 3]),
            tf.random.uniform([2, 8, 8, 3])
        ]

        with tf.GradientTape() as tape:
            outputs = model(inputs, training=True)
            losses = [
                tf.reduce_mean(keras.losses.mean_squared_error(t, o))
                for t, o in zip(targets, outputs)
            ]
            total_loss = tf.reduce_sum(losses)

        grads = tape.gradient(total_loss, model.trainable_weights)

        assert len(grads) > 0
        assert all(g is not None for g in grads)

    def test_no_gradient_explosion(self, small_input_shape):
        """Test that gradients don't explode."""
        model = ConvUNextModel(
            input_shape=small_input_shape,
            depth=3,
            initial_filters=32
        )

        inputs = tf.random.normal([2] + list(small_input_shape))
        targets = tf.random.normal([2] + list(small_input_shape))

        with tf.GradientTape() as tape:
            outputs = model(inputs, training=True)
            loss = tf.reduce_mean(keras.losses.mean_squared_error(targets, outputs))

        grads = tape.gradient(loss, model.trainable_weights)

        for grad in grads:
            if grad is not None:
                assert not tf.reduce_any(tf.math.is_inf(grad))
                assert not tf.reduce_any(tf.math.is_nan(grad))

    def test_gradient_magnitude_reasonable(self, tiny_input_shape):
        """Test that gradient magnitudes are reasonable."""
        model = ConvUNextModel(
            input_shape=tiny_input_shape,
            depth=2,
            initial_filters=16
        )

        inputs = tf.random.normal([2] + list(tiny_input_shape))
        targets = tf.random.normal([2] + list(tiny_input_shape))

        with tf.GradientTape() as tape:
            outputs = model(inputs, training=True)
            loss = tf.reduce_mean(keras.losses.mean_squared_error(targets, outputs))

        grads = tape.gradient(loss, model.trainable_weights)

        for grad in grads:
            if grad is not None:
                grad_norm = tf.norm(grad)
                assert grad_norm < 1000.0  # Reasonable upper bound


# ============================================================================
# Serialization Tests
# ============================================================================

class TestSerialization:
    """Test suite for model serialization and deserialization."""

    def test_get_config(self, small_input_shape):
        """Test get_config returns complete configuration."""
        model = ConvUNextModel(
            input_shape=small_input_shape,
            depth=3,
            initial_filters=32,
            output_channels=5
        )
        config = model.get_config()

        assert 'depth' in config
        assert 'initial_filters' in config
        assert 'output_channels' in config
        assert config['depth'] == 3
        assert config['output_channels'] == 5

    def test_from_config(self, small_input_shape):
        """Test model reconstruction from config."""
        original_model = ConvUNextModel(
            input_shape=small_input_shape,
            depth=3,
            initial_filters=32
        )
        config = original_model.get_config()
        recreated_model = ConvUNextModel.from_config(config)

        assert recreated_model.depth == original_model.depth
        assert recreated_model.initial_filters == original_model.initial_filters

    def test_save_load_keras_format(self, small_input_shape):
        """Test save and load in .keras format."""
        model = ConvUNextModel(
            input_shape=small_input_shape,
            depth=2,
            initial_filters=32,
            enable_deep_supervision=False
        )

        dummy = tf.random.uniform([1] + list(small_input_shape))
        original_output = model(dummy)

        with tempfile.TemporaryDirectory() as tmpdirname:
            filepath = os.path.join(tmpdirname, "model.keras")
            model.save(filepath)
            loaded_model = keras.models.load_model(filepath)

            loaded_output = loaded_model(dummy)

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_output),
                keras.ops.convert_to_numpy(loaded_output),
                rtol=1e-6, atol=1e-6,
                err_msg="Outputs should match after serialization"
            )

    def test_save_load_with_deep_supervision(self, small_input_shape):
        """Test save and load with deep supervision enabled."""
        model = ConvUNextModel(
            input_shape=small_input_shape,
            depth=2,
            initial_filters=32,
            enable_deep_supervision=True
        )

        dummy = tf.random.uniform([1] + list(small_input_shape))
        _ = model(dummy)

        with tempfile.TemporaryDirectory() as tmpdirname:
            filepath = os.path.join(tmpdirname, "model.keras")
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)

            out = loaded_model(dummy)
            assert isinstance(out, list)
            assert len(out) == model.depth

    def test_save_weights_load_weights(self, small_input_shape):
        """Test saving and loading weights only."""
        model = ConvUNextModel(
            input_shape=small_input_shape,
            depth=2,
            initial_filters=32
        )

        dummy = tf.random.uniform([1] + list(small_input_shape))
        _ = model(dummy)

        with tempfile.TemporaryDirectory() as tmpdirname:
            weights_path = os.path.join(tmpdirname, "weights.weights.h5")
            model.save_weights(weights_path)

            new_model = ConvUNextModel(
                input_shape=small_input_shape,
                depth=2,
                initial_filters=32
            )
            _ = new_model(dummy)
            new_model.load_weights(weights_path)

            original_out = model(dummy)
            loaded_out = new_model(dummy)

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_out),
                keras.ops.convert_to_numpy(loaded_out),
                rtol=1e-6, atol=1e-6
            )

    def test_serialization_with_regularization(self, small_input_shape):
        """Test serialization with regularization enabled."""
        original = ConvUNextModel(
            input_shape=small_input_shape,
            depth=2,
            initial_filters=32,
            kernel_regularizer='l2'
        )

        config = original.get_config()
        recreated = ConvUNextModel.from_config(config)

        assert recreated.kernel_regularizer is not None


# ============================================================================
# Factory Function Tests
# ============================================================================

class TestFactoryFunctions:
    """Test suite for factory functions."""

    def test_create_convunext_variant(self):
        """Test create_convunext_variant factory function."""
        model = create_convunext_variant(
            'base',
            input_shape=(64, 64, 3),
            output_channels=10
        )
        assert isinstance(model, ConvUNextModel)
        assert model.depth == 4
        assert model.output_channels == 10

    @pytest.mark.parametrize("variant", ["tiny", "small", "base", "large", "xlarge"])
    def test_factory_all_variants(self, variant):
        """Test factory function with all variants."""
        model = create_convunext_variant(
            variant,
            input_shape=(64, 64, 3)
        )
        assert isinstance(model, ConvUNextModel)

    def test_create_inference_from_training(self, small_input_shape):
        """Test creating inference model from training model."""
        training_model = ConvUNextModel(
            input_shape=small_input_shape,
            depth=3,
            initial_filters=32,
            enable_deep_supervision=True
        )
        data = tf.random.uniform([1] + list(small_input_shape))
        _ = training_model(data)

        inference_model = create_inference_model_from_training_model(training_model)

        assert inference_model.enable_deep_supervision is False

        inf_out = inference_model(data, training=False)
        train_out = training_model(data, training=False)[0]

        np.testing.assert_allclose(
            inf_out.numpy(),
            train_out.numpy(),
            rtol=1e-5, atol=1e-5
        )

    def test_create_inference_from_already_inference(self, small_input_shape):
        """Test that creating inference from inference model returns same model."""
        model = ConvUNextModel(
            input_shape=small_input_shape,
            depth=2,
            initial_filters=32,
            enable_deep_supervision=False
        )

        same_model = create_inference_model_from_training_model(model)
        assert same_model is model

    def test_inference_model_weight_transfer(self, small_input_shape):
        """Test that weights are correctly transferred to inference model."""
        training_model = ConvUNextModel(
            input_shape=small_input_shape,
            depth=2,
            initial_filters=32,
            enable_deep_supervision=True
        )

        # Initialize weights
        dummy = tf.random.uniform([1] + list(small_input_shape))
        _ = training_model(dummy)

        # Get initial stem weights
        stem_weights_before = training_model.stem.get_weights()

        # Create inference model
        inference_model = create_inference_model_from_training_model(training_model)

        # Check stem weights transferred
        stem_weights_after = inference_model.stem.get_weights()

        for w_before, w_after in zip(stem_weights_before, stem_weights_after):
            np.testing.assert_allclose(w_before, w_after, rtol=1e-6, atol=1e-6)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Test suite for model integration scenarios."""

    def test_training_loop_integration(self, small_input_shape, small_sample_data):
        """Test model in full training loop."""
        model = ConvUNextModel(
            input_shape=small_input_shape,
            depth=2,
            initial_filters=32,
            enable_deep_supervision=True
        )
        model.compile(optimizer='adam', loss=['mse', 'mse'])

        y_true = [small_sample_data, tf.image.resize(small_sample_data, (16, 16))]
        history = model.fit(small_sample_data, y_true, epochs=1, verbose=0)

        assert "loss" in history.history

    def test_training_with_validation(self, small_input_shape, small_sample_data):
        """Test model training with validation data."""
        model = ConvUNextModel(
            input_shape=small_input_shape,
            depth=2,
            initial_filters=32
        )
        model.compile(optimizer='adam', loss='mse')

        history = model.fit(
            small_sample_data,
            small_sample_data,
            validation_data=(small_sample_data, small_sample_data),
            epochs=2,
            verbose=0
        )

        assert "val_loss" in history.history

    def test_prediction_api(self, small_input_shape, small_sample_data):
        """Test model.predict() API."""
        model = ConvUNextModel(
            input_shape=small_input_shape,
            depth=2,
            initial_filters=32
        )

        predictions = model.predict(small_sample_data, verbose=0)
        assert predictions.shape == small_sample_data.shape

    def test_evaluate_api(self, small_input_shape, small_sample_data):
        """Test model.evaluate() API."""
        model = ConvUNextModel(
            input_shape=small_input_shape,
            depth=2,
            initial_filters=32
        )
        model.compile(optimizer='adam', loss='mse')

        loss = model.evaluate(small_sample_data, small_sample_data, verbose=0)
        assert isinstance(loss, float)

    def test_callbacks_integration(self, small_input_shape, small_sample_data):
        """Test model with Keras callbacks."""
        model = ConvUNextModel(
            input_shape=small_input_shape,
            depth=2,
            initial_filters=32
        )
        model.compile(optimizer='adam', loss='mse')

        early_stop = keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=3,
            restore_best_weights=True
        )

        history = model.fit(
            small_sample_data,
            small_sample_data,
            epochs=2,
            callbacks=[early_stop],
            verbose=0
        )

        assert len(history.history['loss']) > 0

    def test_custom_training_loop(self, small_input_shape):
        """Test model in custom training loop."""
        model = ConvUNextModel(
            input_shape=small_input_shape,
            depth=2,
            initial_filters=32
        )

        optimizer = keras.optimizers.Adam()
        loss_fn = keras.losses.MeanSquaredError()

        x = tf.random.uniform([2] + list(small_input_shape))
        y = tf.random.uniform([2] + list(small_input_shape))

        with tf.GradientTape() as tape:
            predictions = model(x, training=True)
            loss = loss_fn(y, predictions)

        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        assert loss is not None


# ============================================================================
# Numerical Stability Tests
# ============================================================================

class TestNumericalStability:
    """Test suite for numerical stability."""

    def test_numerical_stability_zeros(self, small_input_shape):
        """Test model with zero inputs."""
        model = ConvUNextModel(
            input_shape=small_input_shape,
            depth=2,
            initial_filters=32
        )
        zeros = tf.zeros((1,) + small_input_shape)
        out = model(zeros)

        assert not np.any(np.isnan(out.numpy()))
        assert not np.any(np.isinf(out.numpy()))

    def test_numerical_stability_ones(self, small_input_shape):
        """Test model with ones inputs."""
        model = ConvUNextModel(
            input_shape=small_input_shape,
            depth=2,
            initial_filters=32
        )
        ones = tf.ones((1,) + small_input_shape)
        out = model(ones)

        assert not np.any(np.isnan(out.numpy()))
        assert not np.any(np.isinf(out.numpy()))

    def test_numerical_stability_large_values(self, small_input_shape):
        """Test model with large input values."""
        model = ConvUNextModel(
            input_shape=small_input_shape,
            depth=2,
            initial_filters=32
        )
        large_inputs = tf.ones((1,) + small_input_shape) * 100.0
        out = model(large_inputs)

        assert not np.any(np.isnan(out.numpy()))
        assert not np.any(np.isinf(out.numpy()))

    def test_numerical_stability_negative_values(self, small_input_shape):
        """Test model with negative input values."""
        model = ConvUNextModel(
            input_shape=small_input_shape,
            depth=2,
            initial_filters=32
        )
        negative_inputs = -tf.random.uniform((1,) + small_input_shape)
        out = model(negative_inputs)

        assert not np.any(np.isnan(out.numpy()))
        assert not np.any(np.isinf(out.numpy()))

    def test_output_range_reasonable(self, small_input_shape):
        """Test that output values are in reasonable range."""
        model = ConvUNextModel(
            input_shape=small_input_shape,
            depth=2,
            initial_filters=32,
            final_activation='tanh'  # Bounded activation
        )
        inputs = tf.random.normal((2,) + small_input_shape)
        outputs = model(inputs)

        # With tanh, outputs should be in [-1, 1]
        assert tf.reduce_max(outputs) <= 1.5
        assert tf.reduce_min(outputs) >= -1.5


# ============================================================================
# Activation Function Tests
# ============================================================================

class TestActivations:
    """Test suite for different activation functions."""

    @pytest.mark.parametrize(
        "activation",
        ['linear', 'relu', 'sigmoid', 'tanh', 'softmax']
    )
    def test_different_final_activations(self, small_input_shape, activation):
        """Test model with different final activations."""
        model = ConvUNextModel(
            input_shape=small_input_shape,
            depth=2,
            initial_filters=32,
            final_activation=activation
        )
        inputs = tf.random.uniform([1] + list(small_input_shape))
        outputs = model(inputs)

        assert outputs.shape == (1,) + small_input_shape

    def test_sigmoid_activation_range(self, small_input_shape):
        """Test that sigmoid activation produces values in [0, 1]."""
        model = ConvUNextModel(
            input_shape=small_input_shape,
            depth=2,
            initial_filters=32,
            final_activation='sigmoid'
        )
        inputs = tf.random.uniform([2] + list(small_input_shape))
        outputs = model(inputs)

        assert tf.reduce_all(outputs >= 0.0)
        assert tf.reduce_all(outputs <= 1.0)

    def test_softmax_activation_sums_to_one(self, small_input_shape):
        """Test that softmax activation sums to one along channels."""
        model = ConvUNextModel(
            input_shape=small_input_shape,
            depth=2,
            initial_filters=32,
            output_channels=5,
            final_activation='softmax'
        )
        inputs = tf.random.uniform([2] + list(small_input_shape))
        outputs = model(inputs)

        # Sum along channel dimension should be ~1
        channel_sums = tf.reduce_sum(outputs, axis=-1)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(channel_sums),
            np.ones_like(channel_sums.numpy()),
            rtol=1e-5, atol=1e-5
        )


# ============================================================================
# Architecture Configuration Tests
# ============================================================================

class TestArchitectureConfiguration:
    """Test suite for different architecture configurations."""

    @pytest.mark.parametrize("depth", [2, 3, 4, 5])
    def test_different_depths(self, small_input_shape, depth):
        """Test model with different depths."""
        model = ConvUNextModel(
            input_shape=small_input_shape,
            depth=depth,
            initial_filters=32
        )
        inputs = tf.random.uniform([1] + list(small_input_shape))
        outputs = model(inputs)

        assert outputs.shape == (1,) + small_input_shape

    @pytest.mark.parametrize("initial_filters", [16, 32, 64, 96])
    def test_different_initial_filters(self, small_input_shape, initial_filters):
        """Test model with different initial filter counts."""
        model = ConvUNextModel(
            input_shape=small_input_shape,
            depth=2,
            initial_filters=initial_filters
        )
        inputs = tf.random.uniform([1] + list(small_input_shape))
        outputs = model(inputs)

        assert outputs.shape == (1,) + small_input_shape

    @pytest.mark.parametrize("blocks_per_level", [1, 2, 3, 4])
    def test_different_blocks_per_level(self, tiny_input_shape, blocks_per_level):
        """Test model with different blocks per level."""
        model = ConvUNextModel(
            input_shape=tiny_input_shape,
            depth=2,
            initial_filters=16,
            blocks_per_level=blocks_per_level
        )
        inputs = tf.random.uniform([1] + list(tiny_input_shape))
        outputs = model(inputs)

        assert outputs.shape == (1,) + tiny_input_shape

    @pytest.mark.parametrize("filter_multiplier", [2, 3, 4])
    def test_different_filter_multipliers(self, tiny_input_shape, filter_multiplier):
        """Test model with different filter multipliers."""
        model = ConvUNextModel(
            input_shape=tiny_input_shape,
            depth=2,
            initial_filters=16,
            filter_multiplier=filter_multiplier
        )
        inputs = tf.random.uniform([1] + list(tiny_input_shape))
        outputs = model(inputs)

        assert outputs.shape == (1,) + tiny_input_shape

    @pytest.mark.parametrize("version", ['v1', 'v2'])
    def test_convnext_versions(self, small_input_shape, version):
        """Test both ConvNeXt v1 and v2 versions."""
        model = ConvUNextModel(
            input_shape=small_input_shape,
            depth=2,
            initial_filters=32,
            convnext_version=version
        )
        inputs = tf.random.uniform([1] + list(small_input_shape))
        outputs = model(inputs)

        assert outputs.shape == (1,) + small_input_shape

    @pytest.mark.parametrize("kernel_size", [3, 5, 7, 9])
    def test_different_kernel_sizes(self, tiny_input_shape, kernel_size):
        """Test model with different kernel sizes."""
        model = ConvUNextModel(
            input_shape=tiny_input_shape,
            depth=2,
            initial_filters=16,
            stem_kernel_size=kernel_size,
            block_kernel_size=kernel_size
        )
        inputs = tf.random.uniform([1] + list(tiny_input_shape))
        outputs = model(inputs)

        assert outputs.shape == (1,) + tiny_input_shape

    @pytest.mark.parametrize("drop_path_rate", [0.0, 0.1, 0.2, 0.3])
    def test_different_drop_path_rates(self, small_input_shape, drop_path_rate):
        """Test model with different drop path rates."""
        model = ConvUNextModel(
            input_shape=small_input_shape,
            depth=2,
            initial_filters=32,
            drop_path_rate=drop_path_rate
        )
        inputs = tf.random.uniform([1] + list(small_input_shape))

        # In training mode, drop path should be active
        outputs_train = model(inputs, training=True)
        # In inference mode, drop path should be inactive
        outputs_infer = model(inputs, training=False)

        assert outputs_train.shape == (1,) + small_input_shape
        assert outputs_infer.shape == (1,) + small_input_shape


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """Test suite for edge cases and error conditions."""

    def test_minimum_spatial_size(self):
        """Test model with minimum viable spatial dimensions."""
        # 16x16 is about the minimum for depth=2 with 2x downsampling
        model = ConvUNextModel(
            input_shape=(16, 16, 3),
            depth=2,
            initial_filters=16
        )
        inputs = tf.random.uniform([1, 16, 16, 3])
        outputs = model(inputs)

        assert outputs.shape == (1, 16, 16, 3)

    def test_single_channel_input_output(self):
        """Test model with single-channel input and output."""
        model = ConvUNextModel(
            input_shape=(32, 32, 1),
            depth=2,
            initial_filters=32,
            output_channels=1
        )
        inputs = tf.random.uniform([1, 32, 32, 1])
        outputs = model(inputs)

        assert outputs.shape == (1, 32, 32, 1)

    def test_many_output_channels(self):
        """Test model with many output channels."""
        model = ConvUNextModel(
            input_shape=(32, 32, 3),
            depth=2,
            initial_filters=32,
            output_channels=128
        )
        inputs = tf.random.uniform([1, 32, 32, 3])
        outputs = model(inputs)

        assert outputs.shape == (1, 32, 32, 128)

    def test_rectangular_input(self):
        """Test model with non-square input dimensions."""
        model = ConvUNextModel(
            input_shape=(64, 32, 3),
            depth=2,
            initial_filters=32
        )
        inputs = tf.random.uniform([1, 64, 32, 3])
        outputs = model(inputs)

        assert outputs.shape == (1, 64, 32, 3)

    def test_very_rectangular_input(self):
        """Test model with highly rectangular input dimensions."""
        model = ConvUNextModel(
            input_shape=(128, 32, 3),
            depth=2,
            initial_filters=32
        )
        inputs = tf.random.uniform([1, 128, 32, 3])
        outputs = model(inputs)

        assert outputs.shape == (1, 128, 32, 3)

    def test_output_shape_preservation(self, small_input_shape):
        """Test that output spatial dimensions match input."""
        model = ConvUNextModel(
            input_shape=small_input_shape,
            depth=3,
            initial_filters=32
        )

        for h, w in [(32, 32), (64, 64), (128, 128)]:
            inputs = tf.random.uniform([1, h, w, 3])
            outputs = model(inputs)
            assert outputs.shape[1:3] == (h, w)


# ============================================================================
# Performance and Memory Tests
# ============================================================================

class TestPerformanceMemory:
    """Test suite for performance and memory characteristics."""

    def test_parameter_count_reasonable(self, small_input_shape):
        """Test that parameter count is reasonable."""
        model = ConvUNextModel(
            input_shape=small_input_shape,
            depth=3,
            initial_filters=32
        )

        # Build model first
        dummy = tf.random.uniform([1] + list(small_input_shape))
        _ = model(dummy)

        total_params = sum([
            keras.ops.size(w).numpy() for w in model.trainable_weights
        ])

        # Should be reasonable for the architecture
        assert total_params > 0
        assert total_params < 10_000_000  # Less than 10M parameters

    def test_trainable_vs_non_trainable(self, small_input_shape):
        """Test ratio of trainable vs non-trainable parameters."""
        model = ConvUNextModel(
            input_shape=small_input_shape,
            depth=2,
            initial_filters=32
        )

        # Build model
        _ = model(tf.random.uniform([1] + list(small_input_shape)))

        trainable = len(model.trainable_weights)
        non_trainable = len(model.non_trainable_weights)

        # Should have mostly trainable weights
        assert trainable > 0
        assert trainable > non_trainable

    def test_inference_vs_training_consistency(self, small_input_shape):
        """Test that training and inference modes produce similar outputs."""
        model = ConvUNextModel(
            input_shape=small_input_shape,
            depth=2,
            initial_filters=32,
            drop_path_rate=0.0  # Disable stochastic depth
        )

        inputs = tf.random.uniform([1] + list(small_input_shape))

        out_train = model(inputs, training=True)
        out_infer = model(inputs, training=False)

        # With drop_path=0, should be identical
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(out_train),
            keras.ops.convert_to_numpy(out_infer),
            rtol=1e-6, atol=1e-6
        )

    def test_memory_cleanup(self, small_input_shape):
        """Test that model can be deleted and memory freed."""
        model = ConvUNextModel(
            input_shape=small_input_shape,
            depth=2,
            initial_filters=32
        )

        inputs = tf.random.uniform([1] + list(small_input_shape))
        _ = model(inputs)

        # Delete model and check no errors
        del model

        # Create new model to verify no issues
        new_model = ConvUNextModel(
            input_shape=small_input_shape,
            depth=2,
            initial_filters=32
        )
        _ = new_model(inputs)


# ============================================================================
# Deep Supervision Specific Tests
# ============================================================================

class TestDeepSupervision:
    """Test suite specifically for deep supervision functionality."""

    def test_deep_supervision_output_count(self, small_input_shape):
        """Test correct number of deep supervision outputs."""
        for depth in [2, 3, 4]:
            model = ConvUNextModel(
                input_shape=small_input_shape,
                depth=depth,
                initial_filters=32,
                enable_deep_supervision=True
            )
            inputs = tf.random.uniform([1] + list(small_input_shape))
            outputs = model(inputs)

            expected_count = depth  # 1 main + (depth-1) aux
            assert len(outputs) == expected_count

    def test_deep_supervision_output_resolutions(self, small_input_shape):
        """Test that deep supervision outputs have correct resolutions."""
        model = ConvUNextModel(
            input_shape=small_input_shape,
            depth=3,
            initial_filters=32,
            enable_deep_supervision=True
        )
        inputs = tf.random.uniform([1] + list(small_input_shape))
        outputs = model(inputs)

        # Main output should be full resolution
        assert outputs[0].shape == (1,) + small_input_shape

        # Auxiliary outputs should be at intermediate resolutions
        for aux_out in outputs[1:]:
            # Should be smaller than main output
            assert aux_out.shape[1] <= small_input_shape[0]
            assert aux_out.shape[2] <= small_input_shape[1]

    def test_deep_supervision_channels(self, small_input_shape):
        """Test that all deep supervision outputs have correct channels."""
        output_channels = 5
        model = ConvUNextModel(
            input_shape=small_input_shape,
            depth=3,
            initial_filters=32,
            enable_deep_supervision=True,
            output_channels=output_channels
        )
        inputs = tf.random.uniform([1] + list(small_input_shape))
        outputs = model(inputs)

        # All outputs should have same number of channels
        for out in outputs:
            assert out.shape[-1] == output_channels

    def test_deep_supervision_training(self, small_input_shape):
        """Test training with deep supervision outputs."""
        model = ConvUNextModel(
            input_shape=small_input_shape,
            depth=2,
            initial_filters=32,
            enable_deep_supervision=True
        )

        x = tf.random.uniform([2] + list(small_input_shape))
        y = [
            tf.random.uniform([2] + list(small_input_shape)),
            tf.random.uniform([2, 16, 16, 3])
        ]

        model.compile(optimizer='adam', loss=['mse', 'mse'])
        history = model.fit(x, y, epochs=1, verbose=0)

        assert 'loss' in history.history

    def test_deep_supervision_vs_no_supervision_consistency(self, small_input_shape):
        """Test that main output is consistent between modes."""
        # Create training model with deep supervision
        train_model = ConvUNextModel(
            input_shape=small_input_shape,
            depth=2,
            initial_filters=32,
            enable_deep_supervision=True
        )

        inputs = tf.random.uniform([1] + list(small_input_shape))
        _ = train_model(inputs)

        # Create inference model
        infer_model = create_inference_model_from_training_model(train_model)

        # Compare outputs
        train_out = train_model(inputs, training=False)[0]
        infer_out = infer_model(inputs, training=False)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(train_out),
            keras.ops.convert_to_numpy(infer_out),
            rtol=1e-5, atol=1e-5
        )