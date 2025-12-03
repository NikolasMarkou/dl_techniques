import pytest
import tempfile
import os
from typing import Any, Dict, Tuple
import numpy as np
import keras
import tensorflow as tf

from dl_techniques.layers.squeeze_excitation import SqueezeExcitation


class TestSqueezeExcitation:
    """Comprehensive test suite for SqueezeExcitation layer following modern patterns."""

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard configuration for testing."""
        return {
            'reduction_ratio': 0.25,
            'activation': 'relu',
            'use_bias': False,
            'kernel_initializer': 'glorot_normal'
        }

    @pytest.fixture
    def custom_layer_config(self) -> Dict[str, Any]:
        """Custom configuration with all parameters."""
        return {
            'reduction_ratio': 0.125,
            'activation': 'swish',
            'use_bias': True,
            'kernel_initializer': 'he_normal',
            'kernel_regularizer': keras.regularizers.L2(1e-4),
            'bias_initializer': 'ones',
            'bias_regularizer': keras.regularizers.L1(1e-5),
        }

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Sample input for testing."""
        return keras.random.normal(shape=(4, 32, 32, 64))

    @pytest.fixture
    def input_shape(self) -> Tuple[int, int, int, int]:
        """Standard input shape for testing."""
        return (4, 32, 32, 64)  # (batch_size, height, width, channels)

    def test_initialization(self, layer_config):
        """Test layer initialization with default and custom parameters."""
        # Test with default parameters
        layer = SqueezeExcitation()

        assert layer.reduction_ratio == 0.25
        assert layer.use_bias is False
        assert isinstance(layer.kernel_initializer, keras.initializers.GlorotNormal)
        assert layer.kernel_regularizer is None
        assert not layer.built

        # Sub-layers should be None before building
        assert layer.global_pool is None
        assert layer.conv_reduce is None
        assert layer.conv_restore is None

        # Test with custom parameters
        custom_layer = SqueezeExcitation(**layer_config)
        assert custom_layer.reduction_ratio == 0.25
        assert custom_layer.use_bias is False

    def test_forward_pass(self, layer_config, sample_input):
        """Test forward pass and building process."""
        layer = SqueezeExcitation(**layer_config)

        # Layer should not be built initially
        assert not layer.built
        assert layer.global_pool is None

        # Forward pass triggers building
        output = layer(sample_input)

        # Verify layer is now built
        assert layer.built
        assert layer.global_pool is not None
        assert layer.conv_reduce is not None
        assert layer.conv_restore is not None

        # Verify output properties
        assert output.shape == sample_input.shape
        assert output.dtype == sample_input.dtype
        assert not keras.ops.any(keras.ops.isnan(output))
        assert not keras.ops.any(keras.ops.isinf(output))

        # SE block should scale features (multiply by attention weights)
        # Output values should be in reasonable range relative to input
        input_abs_max = keras.ops.max(keras.ops.abs(sample_input))
        output_abs_max = keras.ops.max(keras.ops.abs(output))
        assert output_abs_max <= input_abs_max * 2  # Reasonable scaling bound

    def test_serialization_cycle(self, layer_config, sample_input):
        """CRITICAL TEST: Full serialization cycle following modern patterns."""
        # Create model with custom layer
        inputs = keras.Input(shape=sample_input.shape[1:])
        layer_output = SqueezeExcitation(**layer_config)(inputs)
        model = keras.Model(inputs, layer_output)

        # Get original prediction
        original_prediction = model(sample_input)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            # Load without custom_objects (thanks to registration decorator)
            loaded_model = keras.models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input)

            # Verify identical predictions
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_prediction),
                keras.ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization cycle"
            )

    def test_config_completeness(self, custom_layer_config):
        """Test that get_config contains all __init__ parameters."""
        layer = SqueezeExcitation(**custom_layer_config)
        config = layer.get_config()

        # Check all custom config parameters are present
        expected_keys = {
            'reduction_ratio', 'activation', 'use_bias',
            'kernel_initializer', 'kernel_regularizer',
            'bias_initializer', 'bias_regularizer'
        }

        config_keys = set(config.keys())
        for key in expected_keys:
            assert key in config_keys, f"Missing {key} in get_config()"

        # Verify specific values
        assert config['reduction_ratio'] == 0.125
        assert config['use_bias'] is True

    def test_gradients_flow(self, layer_config, sample_input):
        """Test gradient computation and flow."""
        layer = SqueezeExcitation(**layer_config)

        with tf.GradientTape() as tape:
            tape.watch(sample_input)
            output = layer(sample_input)
            loss = keras.ops.mean(keras.ops.square(output))

        # Compute gradients
        gradients = tape.gradient(loss, layer.trainable_variables)

        # Check that all gradients exist
        assert all(g is not None for g in gradients)
        assert len(gradients) > 0

        # Check gradient shapes match variable shapes
        for grad, var in zip(gradients, layer.trainable_variables):
            assert grad.shape == var.shape
            assert not keras.ops.any(keras.ops.isnan(grad))

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, layer_config, sample_input, training):
        """Test behavior in different training modes."""
        layer = SqueezeExcitation(**layer_config)

        output = layer(sample_input, training=training)

        assert output.shape == sample_input.shape
        assert not keras.ops.any(keras.ops.isnan(output))
        assert not keras.ops.any(keras.ops.isinf(output))

    def test_edge_cases(self):
        """Test error conditions and parameter validation."""
        # Invalid reduction ratios
        with pytest.raises(ValueError, match="reduction_ratio must be in range \\(0, 1\\]"):
            SqueezeExcitation(reduction_ratio=0.0)

        with pytest.raises(ValueError, match="reduction_ratio must be in range \\(0, 1\\]"):
            SqueezeExcitation(reduction_ratio=1.5)

        with pytest.raises(ValueError, match="reduction_ratio must be in range \\(0, 1\\]"):
            SqueezeExcitation(reduction_ratio=-0.1)

        # Test with minimal reduction ratio
        layer_minimal = SqueezeExcitation(reduction_ratio=1e-6)
        test_input = keras.random.normal([2, 16, 16, 64])
        output = layer_minimal(test_input)
        assert output.shape == test_input.shape

    def test_compute_output_shape(self):
        """Test output shape computation."""
        layer = SqueezeExcitation(reduction_ratio=0.5)

        test_shapes = [
            ((None, 32, 32, 64), (None, 32, 32, 64)),
            ((4, 16, 16, 128), (4, 16, 16, 128)),
            ((1, 224, 224, 3), (1, 224, 224, 3)),
        ]

        for input_shape, expected_output_shape in test_shapes:
            output_shape = layer.compute_output_shape(input_shape)
            assert output_shape == expected_output_shape

    def test_different_activations(self, sample_input):
        """Test layer with various activation functions."""
        activations = ['relu', 'swish', 'gelu', 'tanh', 'sigmoid']

        for activation in activations:
            layer = SqueezeExcitation(
                reduction_ratio=0.25,
                activation=activation
            )
            output = layer(sample_input)

            assert output.shape == sample_input.shape
            assert not keras.ops.any(keras.ops.isnan(output))
            assert not keras.ops.any(keras.ops.isinf(output))

    def test_custom_activation_callable(self, sample_input):
        """Test with custom activation as callable."""

        def custom_activation(x):
            return keras.ops.relu(x) * 0.5

        layer = SqueezeExcitation(
            reduction_ratio=0.25,
            activation=custom_activation
        )
        output = layer(sample_input)

        assert output.shape == sample_input.shape
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_bottleneck_dimensions(self):
        """Test bottleneck dimension calculation."""
        test_cases = [
            # (input_channels, reduction_ratio, expected_bottleneck)
            (64, 0.25, 16),
            (128, 0.125, 16),
            (32, 0.5, 16),
            (8, 0.25, 2),
            (4, 0.25, 1),  # Minimum bottleneck size
            (1000, 0.001, 1),  # Very small ratio
        ]

        for input_channels, ratio, expected_bottleneck in test_cases:
            layer = SqueezeExcitation(reduction_ratio=ratio)
            test_input = keras.random.normal([2, 8, 8, input_channels])

            # Build the layer
            layer(test_input)

            assert layer.bottleneck_channels == expected_bottleneck

            # Verify conv_reduce has correct number of filters
            assert layer.conv_reduce.filters == expected_bottleneck
            assert layer.conv_restore.filters == input_channels

    def test_regularization_losses(self, sample_input):
        """Test regularization loss computation."""
        layer = SqueezeExcitation(
            reduction_ratio=0.25,
            kernel_regularizer=keras.regularizers.L2(0.01),
            bias_regularizer=keras.regularizers.L1(0.01),
            use_bias=True
        )

        # No losses before forward pass
        initial_losses = len(layer.losses)

        # Forward pass
        output = layer(sample_input)

        # Should have regularization losses
        assert len(layer.losses) > initial_losses

        # Verify losses are positive
        total_loss = sum(layer.losses)
        assert keras.ops.convert_to_numpy(total_loss) > 0

    def test_model_integration(self, sample_input):
        """Test integration within a complete model."""
        # Create CNN with SE blocks
        inputs = keras.layers.Input(shape=sample_input.shape[1:])
        x = keras.layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = SqueezeExcitation(reduction_ratio=0.25)(x)

        x = keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = SqueezeExcitation(reduction_ratio=0.125)(x)

        x = keras.layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(10, activation='softmax')(x)

        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Test forward pass
        predictions = model(sample_input)
        assert predictions.shape == (sample_input.shape[0], 10)

        # Verify softmax probabilities sum to 1
        prob_sums = keras.ops.sum(predictions, axis=1)
        expected_sums = keras.ops.ones_like(prob_sums)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(prob_sums),
            keras.ops.convert_to_numpy(expected_sums),
            rtol=1e-6, atol=1e-6,
            err_msg="Softmax probabilities should sum to 1"
        )

    def test_numerical_stability(self):
        """Test numerical stability with extreme input values."""
        layer = SqueezeExcitation(reduction_ratio=0.25)

        test_cases = [
            keras.ops.zeros((2, 16, 16, 32)),  # All zeros
            keras.random.normal((2, 16, 16, 32)) * 1e-10,  # Very small
            keras.random.normal((2, 16, 16, 32)) * 1e5,  # Large values
            keras.ops.ones((2, 16, 16, 32)) * -100,  # Large negative
        ]

        for test_input in test_cases:
            output = layer(test_input)

            assert not keras.ops.any(keras.ops.isnan(output)), "NaN values detected"
            assert not keras.ops.any(keras.ops.isinf(output)), "Inf values detected"
            assert output.shape == test_input.shape

    def test_attention_weights_properties(self, sample_input):
        """Test properties of attention weights generated by SE block."""
        layer = SqueezeExcitation(reduction_ratio=0.25)

        # Create a custom forward pass to access attention weights
        with tf.GradientTape() as tape:
            tape.watch(sample_input)

            # Manual forward pass to access intermediate values
            squeezed = layer.global_pool(sample_input) if layer.built else None
            if squeezed is None:
                # Build layer first
                _ = layer(sample_input)
                squeezed = layer.global_pool(sample_input)

            excited = layer.conv_reduce(squeezed)
            excited = layer.reduction_activation(excited)
            excited = layer.conv_restore(excited)
            attention_weights = keras.activations.sigmoid(excited)

            # Verify attention weights properties
            assert attention_weights.shape == (sample_input.shape[0], 1, 1, sample_input.shape[-1])

            # Sigmoid output should be in [0, 1]
            assert keras.ops.all(attention_weights >= 0.0)
            assert keras.ops.all(attention_weights <= 1.0)

            # Should not be all the same value (unless input is pathological)
            if not keras.ops.all(keras.ops.equal(sample_input, sample_input[0:1])):
                attention_std = keras.ops.std(attention_weights)
                assert keras.ops.convert_to_numpy(attention_std) > 1e-8

    def test_deterministic_behavior(self):
        """Test deterministic behavior with fixed inputs."""
        layer = SqueezeExcitation(
            reduction_ratio=0.5,
            kernel_initializer='ones',
            bias_initializer='zeros',
            use_bias=False
        )

        # Fixed input
        fixed_input = keras.ops.ones((2, 4, 4, 8))

        # Multiple forward passes should be identical
        output1 = layer(fixed_input)
        output2 = layer(fixed_input)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output1),
            keras.ops.convert_to_numpy(output2),
            rtol=1e-6, atol=1e-6,
            err_msg="Deterministic inputs should produce identical outputs"
        )

    def test_layer_weights_structure(self, sample_input):
        """Test weight structure and shapes."""
        layer = SqueezeExcitation(reduction_ratio=0.25, use_bias=True)

        # Build layer
        _ = layer(sample_input)

        # Should have 4 weights: 2 kernels + 2 biases
        assert len(layer.weights) == 4
        assert len(layer.trainable_variables) == 4

        # Verify weight shapes (more reliable than name checking)
        input_channels = sample_input.shape[-1]
        bottleneck_channels = max(1, int(round(input_channels * 0.25)))

        expected_shapes = [
            (1, 1, input_channels, bottleneck_channels),  # conv_reduce kernel
            (bottleneck_channels,),  # conv_reduce bias
            (1, 1, bottleneck_channels, input_channels),  # conv_restore kernel
            (input_channels,)  # conv_restore bias
        ]

        # Sort both lists by shape for comparison
        actual_shapes = sorted([tuple(w.shape) for w in layer.weights])
        expected_shapes = sorted(expected_shapes)

        assert actual_shapes == expected_shapes

        # Verify we have both kernel and bias weights
        kernel_shapes = [s for s in actual_shapes if len(s) == 4]  # 4D tensors are kernels
        bias_shapes = [s for s in actual_shapes if len(s) == 1]  # 1D tensors are biases

        assert len(kernel_shapes) == 2, "Should have 2 kernel weights"
        assert len(bias_shapes) == 2, "Should have 2 bias weights"


    def test_minimal_input_size(self):
        """Test with minimal spatial dimensions."""
        layer = SqueezeExcitation(reduction_ratio=0.5)

        # Minimum spatial size (1x1)
        minimal_input = keras.random.normal([2, 1, 1, 16])
        output = layer(minimal_input)

        assert output.shape == minimal_input.shape
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_single_channel_input(self):
        """Test with single channel input."""
        layer = SqueezeExcitation(reduction_ratio=0.5)

        single_channel_input = keras.random.normal([2, 8, 8, 1])
        output = layer(single_channel_input)

        assert output.shape == single_channel_input.shape
        assert layer.bottleneck_channels == 1  # max(1, int(round(1 * 0.5)))

    def test_very_high_channel_count(self):
        """Test with very high channel count."""
        layer = SqueezeExcitation(reduction_ratio=0.01)  # Very small ratio

        high_channel_input = keras.random.normal([1, 4, 4, 1024])
        output = layer(high_channel_input)

        assert output.shape == high_channel_input.shape
        assert layer.bottleneck_channels >= 1

    def test_extreme_reduction_ratios(self):
        """Test with extreme but valid reduction ratios."""
        # Very small ratio
        layer_small = SqueezeExcitation(reduction_ratio=1e-6)
        test_input = keras.random.normal([2, 8, 8, 1000])
        output = layer_small(test_input)
        assert output.shape == test_input.shape
        assert layer_small.bottleneck_channels == 1

        # Maximum ratio
        layer_max = SqueezeExcitation(reduction_ratio=1.0)
        test_input = keras.random.normal([2, 8, 8, 64])
        output = layer_max(test_input)
        assert output.shape == test_input.shape
        assert layer_max.bottleneck_channels == 64

    def test_layer_reuse_same_channels(self):
        """Test reusing layer instance with compatible input shapes."""
        layer = SqueezeExcitation(reduction_ratio=0.25)

        # First input
        input1 = keras.random.normal([2, 16, 16, 32])
        output1 = layer(input1)
        assert output1.shape == input1.shape

        # Different batch size - should work
        input2 = keras.random.normal([4, 16, 16, 32])
        output2 = layer(input2)
        assert output2.shape == input2.shape

        # Different spatial size - should work
        input3 = keras.random.normal([2, 8, 8, 32])
        output3 = layer(input3)
        assert output3.shape == input3.shape

        # All outputs should be valid
        for output in [output1, output2, output3]:
            assert not keras.ops.any(keras.ops.isnan(output))

    def test_different_channel_counts_separate_layers(self):
        """Test different channel counts with separate layer instances."""
        channel_counts = [1, 8, 32, 128, 512]

        for channels in channel_counts:
            layer = SqueezeExcitation(reduction_ratio=0.25)
            test_input = keras.random.normal([2, 8, 8, channels])

            output = layer(test_input)

            assert output.shape == test_input.shape
            assert not keras.ops.any(keras.ops.isnan(output))

            # Verify bottleneck calculation
            expected_bottleneck = max(1, int(round(channels * 0.25)))
            assert layer.bottleneck_channels == expected_bottleneck

    def test_model_save_load(self, sample_input):
        """Test saving and loading a model with the SqueezeExcitation layer."""
        # Create a model with the SE layer
        inputs = keras.layers.Input(shape=sample_input.shape[1:])
        x = keras.layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
        x = SqueezeExcitation(reduction_ratio=0.25, name="se_block")(x)
        x = keras.layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(10, activation='softmax')(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

        # Generate prediction before saving
        original_prediction = model.predict(sample_input, verbose=0)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'model.keras')

            # Save the model
            model.save(model_path)

            # Load without custom_objects (thanks to registration decorator)
            loaded_model = keras.models.load_model(model_path)

            # Generate prediction with loaded model
            loaded_prediction = loaded_model.predict(sample_input, verbose=0)

            # Check predictions match
            np.testing.assert_allclose(
                original_prediction,
                loaded_prediction,
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions should match after model save/load"
            )

            # Check layer type is preserved
            se_layer = loaded_model.get_layer("se_block")
            assert isinstance(se_layer, SqueezeExcitation)
            assert se_layer.reduction_ratio == 0.25

    def test_training_behavior_detailed(self, sample_input):
        """Test detailed behavior differences in training vs inference mode."""
        layer = SqueezeExcitation(reduction_ratio=0.25)

        # Test training mode multiple times (should be deterministic for SE)
        training_outputs = []
        for _ in range(3):
            output = layer(sample_input, training=True)
            training_outputs.append(keras.ops.convert_to_numpy(output))

        # SE layer should be deterministic even in training mode
        for i in range(1, len(training_outputs)):
            np.testing.assert_allclose(
                training_outputs[0],
                training_outputs[i],
                rtol=1e-6, atol=1e-6,
                err_msg="SE layer should be deterministic in training mode"
            )

        # Test inference mode
        inference_output1 = layer(sample_input, training=False)
        inference_output2 = layer(sample_input, training=False)

        # Inference should also be deterministic
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(inference_output1),
            keras.ops.convert_to_numpy(inference_output2),
            rtol=1e-6, atol=1e-6,
            err_msg="SE layer should be deterministic in inference mode"
        )

        # Training and inference should produce same result for SE layer
        np.testing.assert_allclose(
            training_outputs[0],
            keras.ops.convert_to_numpy(inference_output1),
            rtol=1e-6, atol=1e-6,
            err_msg="SE layer should produce same output in training and inference"
        )

    def test_different_input_sizes_comprehensive(self):
        """Test layer with comprehensive range of input sizes."""
        # Various input configurations
        input_configs = [
            (1, 8, 8, 16),  # Small spatial, few channels
            (2, 32, 32, 64),  # Medium spatial, medium channels
            (1, 224, 224, 3),  # Large spatial, few channels (like RGB)
            (4, 16, 16, 256),  # Small spatial, many channels
            (2, 64, 128, 32),  # Non-square spatial dimensions
            (3, 1, 1, 512),  # Minimal spatial dimensions
            (1, 512, 512, 1),  # Very large spatial, single channel
        ]

        for batch, height, width, channels in input_configs:
            # Create new layer instance for each configuration
            layer = SqueezeExcitation(reduction_ratio=0.5)
            test_input = keras.random.normal([batch, height, width, channels])
            output = layer(test_input)

            assert output.shape == test_input.shape
            assert not keras.ops.any(keras.ops.isnan(output))
            assert not keras.ops.any(keras.ops.isinf(output))

            # Verify bottleneck calculation
            expected_bottleneck = max(1, int(round(channels * 0.5)))
            assert layer.bottleneck_channels == expected_bottleneck

    def test_regularization_comprehensive(self, sample_input):
        """Test comprehensive regularization behavior."""
        # Test kernel regularization
        layer_kernel_reg = SqueezeExcitation(
            reduction_ratio=0.25,
            kernel_regularizer=keras.regularizers.L2(0.01)
        )

        initial_losses_kernel = len(layer_kernel_reg.losses)
        _ = layer_kernel_reg(sample_input)
        assert len(layer_kernel_reg.losses) > initial_losses_kernel

        # Test bias regularization
        layer_bias_reg = SqueezeExcitation(
            reduction_ratio=0.25,
            use_bias=True,
            bias_regularizer=keras.regularizers.L1(0.01)
        )

        initial_losses_bias = len(layer_bias_reg.losses)
        _ = layer_bias_reg(sample_input)
        assert len(layer_bias_reg.losses) > initial_losses_bias

        # Test combined regularization
        layer_combined = SqueezeExcitation(
            reduction_ratio=0.25,
            use_bias=True,
            kernel_regularizer=keras.regularizers.L2(0.01),
            bias_regularizer=keras.regularizers.L1(0.01)
        )

        initial_losses_combined = len(layer_combined.losses)
        _ = layer_combined(sample_input)
        final_losses_combined = len(layer_combined.losses)

        # Should have regularization losses from both kernel and bias
        assert final_losses_combined > initial_losses_combined

        # Verify regularization losses are positive
        total_reg_loss = sum(layer_combined.losses)
        assert keras.ops.convert_to_numpy(total_reg_loss) > 0

    def test_attention_mechanism_properties(self, sample_input):
        """Test detailed properties of the SE attention mechanism."""
        layer = SqueezeExcitation(reduction_ratio=0.25)

        # Test that different channels get different attention weights
        # Create input with distinct channel patterns
        channels = sample_input.shape[-1]

        # Create patterned input more simply
        patterned_input = keras.ops.zeros_like(sample_input)
        for c in range(min(channels, 8)):  # Test first 8 channels
            channel_value = (c + 1) * 0.5
            # Set specific channel to specific value
            channel_slice = patterned_input[:, :, :, c:c + 1]
            patterned_input = keras.ops.concatenate([
                patterned_input[:, :, :, :c],
                keras.ops.full_like(channel_slice, channel_value),
                patterned_input[:, :, :, c + 1:]
            ], axis=-1)

        output = layer(patterned_input)

        # Verify output properties
        assert output.shape == patterned_input.shape
        assert not keras.ops.any(keras.ops.isnan(output))

        # SE should modulate the input based on channel importance
        # Check that output is not identical to input (SE is doing something)
        input_mean = keras.ops.mean(keras.ops.abs(patterned_input))
        output_mean = keras.ops.mean(keras.ops.abs(output))

        # SE should produce some modulation (output shouldn't be identical to input)
        # unless all channels have the same statistics (which they don't in our test)
        relative_diff = keras.ops.abs(output_mean - input_mean) / (input_mean + 1e-8)
        assert keras.ops.convert_to_numpy(relative_diff) > 1e-6

    def test_gradient_flow_comprehensive(self, sample_input):
        """Test comprehensive gradient flow analysis."""
        layer = SqueezeExcitation(reduction_ratio=0.25, use_bias=True)

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(sample_input)
            output = layer(sample_input)

            # Test different loss functions
            losses = {
                'mse': keras.ops.mean(keras.ops.square(output)),
                'mae': keras.ops.mean(keras.ops.abs(output)),
                'sum': keras.ops.sum(output)
            }

        for loss_name, loss_value in losses.items():
            # Input gradients
            input_grads = tape.gradient(loss_value, sample_input)
            assert input_grads is not None, f"No input gradients for {loss_name}"
            assert input_grads.shape == sample_input.shape
            assert not keras.ops.any(keras.ops.isnan(input_grads))

            # Layer weight gradients
            weight_grads = tape.gradient(loss_value, layer.trainable_variables)
            assert all(g is not None for g in weight_grads), f"Missing weight gradients for {loss_name}"

            # Verify gradient shapes and properties
            for grad, var in zip(weight_grads, layer.trainable_variables):
                assert grad.shape == var.shape, f"Gradient shape mismatch for {loss_name}"
                assert not keras.ops.any(keras.ops.isnan(grad)), f"NaN gradients for {loss_name}"

                # Gradients should not all be zero (unless input is pathological)
                grad_magnitude = keras.ops.sqrt(keras.ops.sum(keras.ops.square(grad)))
                assert keras.ops.convert_to_numpy(grad_magnitude) > 1e-10, f"Zero gradients for {loss_name}"

    def test_compatibility_with_other_layers(self, sample_input):
        """Test compatibility and integration with various other layers."""
        # Test with different normalization layers
        normalizations = [
            keras.layers.BatchNormalization(),
            keras.layers.LayerNormalization(),
        ]

        for norm_layer in normalizations:
            inputs = keras.layers.Input(shape=sample_input.shape[1:])
            x = keras.layers.Conv2D(64, 3, padding='same')(inputs)
            x = norm_layer(x)
            x = SqueezeExcitation(reduction_ratio=0.25)(x)
            x = keras.layers.GlobalAveragePooling2D()(x)
            outputs = keras.layers.Dense(10)(x)

            model = keras.Model(inputs, outputs)
            result = model(sample_input)

            assert result.shape == (sample_input.shape[0], 10)
            assert not keras.ops.any(keras.ops.isnan(result))

        # Test with different activation patterns
        activations = ['relu', 'swish', 'gelu']
        for activation in activations:
            inputs = keras.layers.Input(shape=sample_input.shape[1:])
            x = keras.layers.Conv2D(64, 3, padding='same', activation=activation)(inputs)
            x = SqueezeExcitation(reduction_ratio=0.25)(x)
            x = keras.layers.GlobalAveragePooling2D()(x)
            outputs = keras.layers.Dense(10)(x)

            model = keras.Model(inputs, outputs)
            result = model(sample_input)

            assert result.shape == (sample_input.shape[0], 10)
            assert not keras.ops.any(keras.ops.isnan(result))

    def test_memory_efficiency(self):
        """Test memory usage with large inputs."""
        # Test with progressively larger inputs to verify memory efficiency
        base_channels = 32

        input_sizes = [
            (1, 64, 64, base_channels),
            (1, 128, 128, base_channels),
            (1, 256, 256, base_channels),
        ]

        for size in input_sizes:
            # Create new layer instance for each size test
            layer = SqueezeExcitation(reduction_ratio=0.25)
            test_input = keras.random.normal(size)

            # Should handle large inputs without issues
            output = layer(test_input)

            assert output.shape == test_input.shape
            assert not keras.ops.any(keras.ops.isnan(output))

            # Clean up
            del layer, test_input, output

    def test_performance_characteristics(self, sample_input):
        """Test performance-related characteristics."""
        import time

        # Compare performance with different reduction ratios
        ratios = [0.0625, 0.125, 0.25, 0.5]
        times = []

        for ratio in ratios:
            layer = SqueezeExcitation(reduction_ratio=ratio)

            # Warm up
            _ = layer(sample_input)

            # Measure time for multiple iterations with same layer
            start_time = time.time()
            for _ in range(10):
                _ = layer(sample_input, training=False)
            end_time = time.time()

            avg_time = (end_time - start_time) / 10
            times.append(avg_time)

        # Times should be reasonable (not testing exact values due to hardware variation)
        for t in times:
            assert t < 1.0  # Should complete in under 1 second for 10 iterations

        # All times should be positive and finite
        for t in times:
            assert t > 0
            assert not np.isnan(t)
            assert not np.isinf(t)