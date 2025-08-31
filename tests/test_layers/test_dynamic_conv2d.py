import pytest
import tempfile
import os
import numpy as np
import tensorflow as tf
import keras
from typing import Dict, Any, Tuple

# Import the layer to test
from dl_techniques.layers.dynamic_conv2d import DynamicConv2D


class TestDynamicConv2D:
    """Comprehensive test suite for DynamicConv2D layer."""

    @pytest.fixture
    def basic_layer_config(self) -> Dict[str, Any]:
        """Basic configuration for testing."""
        return {
            'filters': 64,
            'kernel_size': 3,
            'num_kernels': 4,
            'temperature': 30.0,
            'attention_reduction_ratio': 4,
            'strides': (1, 1),
            'padding': 'valid',
            'activation': 'relu'
        }

    @pytest.fixture
    def advanced_layer_config(self) -> Dict[str, Any]:
        """Advanced configuration with all parameters."""
        return {
            'filters': 128,
            'kernel_size': (5, 5),
            'num_kernels': 6,
            'temperature': 15.0,
            'attention_reduction_ratio': 8,
            'strides': (2, 2),
            'padding': 'same',
            'dilation_rate': (1, 1),
            'groups': 1,
            'activation': 'gelu',
            'use_bias': True,
            'kernel_initializer': 'he_normal',
            'bias_initializer': 'zeros',
        }

    @pytest.fixture
    def sample_input_channels_last(self) -> keras.KerasTensor:
        """Sample input tensor with channels_last format."""
        return keras.random.normal(shape=(4, 32, 32, 16))

    def test_initialization(self, basic_layer_config):
        """Test layer initialization."""
        layer = DynamicConv2D(**basic_layer_config)

        # Check basic attributes
        assert layer.filters == 64
        assert layer.kernel_size == (3, 3)
        assert layer.num_kernels == 4
        assert layer.temperature == 30.0
        assert layer.attention_reduction_ratio == 4
        assert not layer.built

        # Check sub-layers are created
        assert layer.gap is not None
        assert isinstance(layer.gap, keras.layers.GlobalAveragePooling2D)
        assert isinstance(layer.conv_layers, list)
        assert len(layer.conv_layers) == 0  # Not built yet

    def test_advanced_initialization(self, advanced_layer_config):
        """Test layer initialization with advanced config."""
        layer = DynamicConv2D(**advanced_layer_config)

        assert layer.filters == 128
        assert layer.kernel_size == (5, 5)
        assert layer.num_kernels == 6
        assert layer.strides == (2, 2)
        assert layer.padding == 'same'

    def test_forward_pass_channels_last(self, basic_layer_config, sample_input_channels_last):
        """Test forward pass with channels_last format."""
        layer = DynamicConv2D(**basic_layer_config)

        output = layer(sample_input_channels_last)

        # Check layer is built
        assert layer.built
        assert len(layer.conv_layers) == 4
        assert layer.attention_dense1 is not None
        assert layer.attention_dense2 is not None

        # Check output shape - valid padding reduces size
        batch_size, height, width, channels = sample_input_channels_last.shape
        expected_height = height - 2  # kernel_size=3, padding='valid'
        expected_width = width - 2
        expected_shape = (batch_size, expected_height, expected_width, 64)

        assert output.shape == expected_shape

    def test_different_padding_modes(self, sample_input_channels_last):
        """Test different padding modes."""
        batch_size, height, width, channels = sample_input_channels_last.shape

        # Test 'valid' padding
        layer_valid = DynamicConv2D(filters=32, kernel_size=5, padding='valid')
        output_valid = layer_valid(sample_input_channels_last)
        expected_valid = (batch_size, height - 4, width - 4, 32)  # 5x5 kernel reduces by 4
        assert output_valid.shape == expected_valid

        # Test 'same' padding
        layer_same = DynamicConv2D(filters=32, kernel_size=5, padding='same')
        output_same = layer_same(sample_input_channels_last)
        expected_same = (batch_size, height, width, 32)  # Same size preserved
        assert output_same.shape == expected_same

    def test_different_strides(self, sample_input_channels_last):
        """Test different stride configurations."""
        batch_size, height, width, channels = sample_input_channels_last.shape

        # Test stride=2
        layer = DynamicConv2D(filters=32, kernel_size=3, strides=2, padding='same')
        output = layer(sample_input_channels_last)
        expected_shape = (batch_size, height // 2, width // 2, 32)
        assert output.shape == expected_shape

    def test_grouped_convolution(self, sample_input_channels_last):
        """Test grouped convolution."""
        # Create input with 32 channels for group=4
        input_32_channels = keras.random.normal(shape=(4, 32, 32, 32))

        layer = DynamicConv2D(
            filters=64,
            kernel_size=3,
            groups=4,  # Each group processes 8 input channels -> 16 output channels
            padding='same'
        )
        output = layer(input_32_channels)

        batch_size, height, width, _ = input_32_channels.shape
        expected_shape = (batch_size, height, width, 64)
        assert output.shape == expected_shape

    def test_serialization_cycle(self, basic_layer_config, sample_input_channels_last):
        """CRITICAL TEST: Full serialization cycle."""
        # Create model with custom layer
        inputs = keras.Input(shape=sample_input_channels_last.shape[1:])
        outputs = DynamicConv2D(**basic_layer_config)(inputs)
        model = keras.Model(inputs, outputs)

        # Get original prediction
        original_pred = model(sample_input_channels_last)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input_channels_last)

            # Verify identical predictions
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization"
            )

    def test_config_completeness(self, basic_layer_config):
        """Test that get_config contains all __init__ parameters."""
        layer = DynamicConv2D(**basic_layer_config)
        config = layer.get_config()

        # Check all basic config parameters are present
        essential_keys = [
            'filters', 'kernel_size', 'num_kernels', 'temperature',
            'attention_reduction_ratio', 'strides', 'padding',
            'dilation_rate', 'groups', 'activation', 'use_bias'
        ]

        for key in essential_keys:
            assert key in config, f"Missing {key} in get_config()"

        # Check serialized values are correct types
        assert isinstance(config['filters'], int)
        assert isinstance(config['kernel_size'], tuple)
        assert isinstance(config['num_kernels'], int)
        assert isinstance(config['temperature'], float)

    def test_advanced_config_completeness(self, advanced_layer_config):
        """Test config completeness with advanced parameters."""
        layer = DynamicConv2D(**advanced_layer_config)
        config = layer.get_config()

        # Check advanced parameters
        advanced_keys = [
            'kernel_initializer', 'bias_initializer', 'kernel_regularizer',
            'bias_regularizer', 'activity_regularizer', 'kernel_constraint',
            'bias_constraint'
        ]

        for key in advanced_keys:
            assert key in config, f"Missing {key} in get_config()"

    def test_gradients_flow(self, basic_layer_config, sample_input_channels_last):
        """Test gradient computation."""
        layer = DynamicConv2D(**basic_layer_config)

        with tf.GradientTape() as tape:
            tape.watch(sample_input_channels_last)
            output = layer(sample_input_channels_last)
            loss = keras.ops.mean(keras.ops.square(output))

        gradients = tape.gradient(loss, layer.trainable_variables)

        assert gradients is not None
        assert len(gradients) > 0
        assert all(g is not None for g in gradients), "Some gradients are None"

        # Check we have gradients for all kernels + attention mechanism
        # Each conv layer has 2 weights (kernel, bias) * num_kernels + 4 attention weights
        expected_num_gradients = (2 * layer.num_kernels) + 4  # 2 dense layers, each with kernel & bias
        assert len(gradients) == expected_num_gradients

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, basic_layer_config, sample_input_channels_last, training):
        """Test behavior in different training modes."""
        layer = DynamicConv2D(**basic_layer_config)

        output = layer(sample_input_channels_last, training=training)
        assert output.shape[0] == sample_input_channels_last.shape[0]

        # Output should be deterministic (no dropout in this layer)
        output2 = layer(sample_input_channels_last, training=training)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output),
            keras.ops.convert_to_numpy(output2),
            rtol=1e-6, atol=1e-6,
            err_msg="Outputs should be identical for same input"
        )

    def test_attention_weights_properties(self, basic_layer_config, sample_input_channels_last):
        """Test that attention weights have correct properties."""
        layer = DynamicConv2D(**basic_layer_config)

        # Build layer first
        _ = layer(sample_input_channels_last)

        # Access attention computation manually
        pooled = layer.gap(sample_input_channels_last)
        attention_hidden = layer.attention_dense1(pooled)
        attention_logits = layer.attention_dense2(attention_hidden)
        attention_weights = keras.ops.softmax(attention_logits / layer.temperature)

        attention_weights_np = keras.ops.convert_to_numpy(attention_weights)

        # Check attention weights sum to 1 (softmax property)
        attention_sums = np.sum(attention_weights_np, axis=1)
        np.testing.assert_allclose(
            attention_sums,
            np.ones_like(attention_sums),
            rtol=1e-6, atol=1e-6,
            err_msg="Attention weights should sum to 1"
        )

        # Check all weights are positive
        assert np.all(attention_weights_np >= 0), "Attention weights should be non-negative"

        # Check shape
        batch_size = sample_input_channels_last.shape[0]
        assert attention_weights_np.shape == (batch_size, layer.num_kernels)

    def test_temperature_effect(self, sample_input_channels_last):
        """Test effect of temperature on attention distribution."""
        # Low temperature should create more peaked distribution
        layer_low_temp = DynamicConv2D(
            filters=32, kernel_size=3, num_kernels=4,
            temperature=1.0, padding='same'
        )

        # High temperature should create more uniform distribution
        layer_high_temp = DynamicConv2D(
            filters=32, kernel_size=3, num_kernels=4,
            temperature=100.0, padding='same'
        )

        # Build layers
        _ = layer_low_temp(sample_input_channels_last)
        _ = layer_high_temp(sample_input_channels_last)

        # Get attention weights
        pooled = layer_low_temp.gap(sample_input_channels_last)

        # Low temperature attention
        attention_hidden_low = layer_low_temp.attention_dense1(pooled)
        attention_logits_low = layer_low_temp.attention_dense2(attention_hidden_low)
        attention_low = keras.ops.softmax(attention_logits_low / 1.0)

        # High temperature attention
        attention_hidden_high = layer_high_temp.attention_dense1(pooled)
        attention_logits_high = layer_high_temp.attention_dense2(attention_hidden_high)
        attention_high = keras.ops.softmax(attention_logits_high / 100.0)

        # Convert to numpy
        attention_low_np = keras.ops.convert_to_numpy(attention_low)
        attention_high_np = keras.ops.convert_to_numpy(attention_high)

        # High temperature should have lower variance (more uniform)
        var_low = np.var(attention_low_np, axis=1).mean()
        var_high = np.var(attention_high_np, axis=1).mean()

        # Generally, high temperature leads to more uniform distribution (lower variance)
        # But this is probabilistic, so we just check they're different
        assert not np.allclose(var_low, var_high), "Temperature should affect attention distribution"

    def test_edge_cases(self):
        """Test error conditions."""
        # Invalid filters
        with pytest.raises(ValueError, match="filters must be positive"):
            DynamicConv2D(filters=0, kernel_size=3)

        with pytest.raises(ValueError, match="filters must be positive"):
            DynamicConv2D(filters=-5, kernel_size=3)

        # Invalid num_kernels
        with pytest.raises(ValueError, match="num_kernels must be at least 2"):
            DynamicConv2D(filters=32, kernel_size=3, num_kernels=1)

        with pytest.raises(ValueError, match="num_kernels must be at least 2"):
            DynamicConv2D(filters=32, kernel_size=3, num_kernels=0)

        # Invalid temperature
        with pytest.raises(ValueError, match="temperature must be positive"):
            DynamicConv2D(filters=32, kernel_size=3, temperature=0.0)

        with pytest.raises(ValueError, match="temperature must be positive"):
            DynamicConv2D(filters=32, kernel_size=3, temperature=-1.0)

        # Invalid attention_reduction_ratio
        with pytest.raises(ValueError, match="attention_reduction_ratio must be positive"):
            DynamicConv2D(filters=32, kernel_size=3, attention_reduction_ratio=0)

        # Invalid groups
        with pytest.raises(ValueError, match="groups must be positive"):
            DynamicConv2D(filters=32, kernel_size=3, groups=0)

        # Invalid padding
        with pytest.raises(ValueError, match="padding must be 'valid' or 'same'"):
            DynamicConv2D(filters=32, kernel_size=3, padding='invalid')

    def test_incompatible_groups(self):
        """Test group compatibility validation during build."""
        # Input channels not divisible by groups
        layer = DynamicConv2D(filters=32, kernel_size=3, groups=4)
        input_15_channels = keras.random.normal((4, 16, 16, 15))  # 15 not divisible by 4

        with pytest.raises(ValueError, match="Input channels .* must be divisible by groups"):
            layer(input_15_channels)

        # Filters not divisible by groups
        layer2 = DynamicConv2D(filters=30, kernel_size=3, groups=4)  # 30 not divisible by 4
        input_16_channels = keras.random.normal((4, 16, 16, 16))

        with pytest.raises(ValueError, match="Filters .* must be divisible by groups"):
            layer2(input_16_channels)

    def test_compute_output_shape(self, basic_layer_config):
        """Test output shape computation."""
        layer = DynamicConv2D(**basic_layer_config)

        # Test channels_last
        input_shape = (None, 32, 32, 16)
        output_shape = layer.compute_output_shape(input_shape)

        # Valid padding reduces size by (kernel_size - 1)
        expected_shape = (None, 30, 30, 64)  # 32 - 2 = 30 for kernel_size=3
        assert output_shape == expected_shape

        # Test with same padding
        layer_same = DynamicConv2D(filters=64, kernel_size=3, padding='same')
        output_shape_same = layer_same.compute_output_shape(input_shape)
        expected_shape_same = (None, 32, 32, 64)
        assert output_shape_same == expected_shape_same


    def test_different_kernel_sizes(self, sample_input_channels_last):
        """Test different kernel size configurations."""
        # Test integer kernel_size
        layer_int = DynamicConv2D(filters=32, kernel_size=5, padding='same')
        output_int = layer_int(sample_input_channels_last)

        # Test tuple kernel_size
        layer_tuple = DynamicConv2D(filters=32, kernel_size=(5, 5), padding='same')
        output_tuple = layer_tuple(sample_input_channels_last)

        # Should produce same shape
        assert output_int.shape == output_tuple.shape

        # Test asymmetric kernel
        layer_asym = DynamicConv2D(filters=32, kernel_size=(3, 5), padding='same')
        output_asym = layer_asym(sample_input_channels_last)

        batch_size, height, width, _ = sample_input_channels_last.shape
        expected_shape = (batch_size, height, width, 32)
        assert output_asym.shape == expected_shape

    def test_regularizers_and_constraints_integration(self):
        """Test that regularizers and constraints are properly handled."""
        layer = DynamicConv2D(
            filters=32,
            kernel_size=3,
            kernel_regularizer=keras.regularizers.L2(0.01),
            bias_regularizer=keras.regularizers.L1(0.01),
            kernel_constraint=keras.constraints.MaxNorm(2.0),
            bias_constraint=keras.constraints.NonNeg()
        )

        # Build layer
        inputs = keras.random.normal((4, 16, 16, 8))
        output = layer(inputs)

        # Check that layer has losses (from regularizers)
        assert len(layer.losses) > 0, "Layer should have regularization losses"

    def test_activation_functions(self, sample_input_channels_last):
        """Test different activation functions."""
        activations_to_test = ['relu', 'gelu', 'tanh', 'swish']

        for activation in activations_to_test:
            layer = DynamicConv2D(
                filters=32, kernel_size=3, padding='same',
                activation=activation
            )

            output = layer(sample_input_channels_last)

            # Check output shape is correct
            expected_shape = sample_input_channels_last.shape[:-1] + (32,)
            assert output.shape == expected_shape

            # Test that activation is applied (output statistics should differ from linear)
            output_np = keras.ops.convert_to_numpy(output)

            if activation == 'relu':
                # ReLU should have no negative values
                assert np.all(output_np >= 0), f"ReLU output should be non-negative"

    def test_no_bias(self, sample_input_channels_last):
        """Test layer without bias."""
        layer = DynamicConv2D(
            filters=32, kernel_size=3, padding='same',
            use_bias=False
        )

        output = layer(sample_input_channels_last)

        # Check that convolution layers don't have bias
        for conv_layer in layer.conv_layers:
            assert not conv_layer.use_bias, "Conv layers should not use bias when use_bias=False"

        # Output shape should still be correct
        expected_shape = sample_input_channels_last.shape[:-1] + (32,)
        assert output.shape == expected_shape


# Additional integration tests
class TestDynamicConv2DIntegration:
    """Integration tests with other layers and training scenarios."""

    def test_in_sequential_model(self):
        """Test DynamicConv2D in a Sequential model."""
        model = keras.Sequential([
            keras.layers.Input(shape=(32, 32, 3)),
            DynamicConv2D(filters=64, kernel_size=3, activation='relu', padding='same'),
            keras.layers.MaxPooling2D(2),
            DynamicConv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(10, activation='softmax')
        ])

        # Test model compilation
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Test model prediction
        dummy_input = keras.random.normal((1, 32, 32, 3))
        prediction = model(dummy_input)
        assert prediction.shape == (1, 10)

        # Test model summary (shouldn't crash)
        model.summary()

    def test_gradient_accumulation(self):
        """Test gradient accumulation across multiple forward passes."""
        layer = DynamicConv2D(filters=32, kernel_size=3, padding='same')
        inputs = keras.random.normal((2, 16, 16, 8))

        # Accumulate gradients over multiple forward passes
        total_loss = 0
        with tf.GradientTape() as tape:
            for _ in range(3):
                output = layer(inputs)
                loss = keras.ops.mean(keras.ops.square(output))
                total_loss += loss

        gradients = tape.gradient(total_loss, layer.trainable_variables)
        assert all(g is not None for g in gradients), "All gradients should be computed"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])