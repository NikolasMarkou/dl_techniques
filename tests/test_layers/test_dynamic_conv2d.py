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
            'kernel_initializer', 'bias_initializer',
            'kernel_regularizer', 'bias_regularizer',
            'activity_regularizer', 'kernel_constraint', 'bias_constraint'
        ]

        for key in advanced_keys:
            assert key in config, f"Missing {key} in get_config()"

    def test_from_config_reconstruction(self, basic_layer_config):
        """Test layer can be reconstructed from config."""
        original_layer = DynamicConv2D(**basic_layer_config)
        config = original_layer.get_config()

        # Reconstruct layer from config
        reconstructed_layer = DynamicConv2D.from_config(config)

        # Check key attributes match
        assert reconstructed_layer.filters == original_layer.filters
        assert reconstructed_layer.kernel_size == original_layer.kernel_size
        assert reconstructed_layer.num_kernels == original_layer.num_kernels
        assert reconstructed_layer.temperature == original_layer.temperature
        assert reconstructed_layer.padding == original_layer.padding

    def test_trainable_weights(self, basic_layer_config, sample_input_channels_last):
        """Test that layer has correct number of trainable weights."""
        layer = DynamicConv2D(**basic_layer_config)

        # Build layer
        output = layer(sample_input_channels_last)

        # Count trainable weights
        # Expected: (K conv kernels + K conv biases) + (2 attention dense layers with weights and biases)
        num_kernels = basic_layer_config['num_kernels']
        expected_min_weights = (num_kernels * 2) + 4  # At minimum

        assert len(layer.trainable_weights) >= expected_min_weights

    def test_non_trainable_after_freeze(self, basic_layer_config, sample_input_channels_last):
        """Test layer can be frozen."""
        layer = DynamicConv2D(**basic_layer_config)

        # Build layer
        layer(sample_input_channels_last)

        # Freeze layer
        layer.trainable = False

        # Check no trainable weights
        assert len(layer.trainable_weights) == 0
        assert len(layer.non_trainable_weights) > 0

    def test_gradient_flow(self, basic_layer_config, sample_input_channels_last):
        """Test that gradients flow through the layer."""
        layer = DynamicConv2D(**basic_layer_config)

        with tf.GradientTape() as tape:
            output = layer(sample_input_channels_last)
            loss = keras.ops.mean(keras.ops.square(output))

        # Compute gradients
        gradients = tape.gradient(loss, layer.trainable_variables)

        # Check all gradients are non-None
        assert all(g is not None for g in gradients), "Some gradients are None"

        # Check gradients have correct shapes
        for var, grad in zip(layer.trainable_variables, gradients):
            assert var.shape == grad.shape, f"Shape mismatch for {var.name}"

    def test_multiple_num_kernels(self, sample_input_channels_last):
        """Test different numbers of kernels."""
        kernel_counts = [2, 4, 6, 8]

        for num_kernels in kernel_counts:
            layer = DynamicConv2D(
                filters=32,
                kernel_size=3,
                num_kernels=num_kernels,
                padding='same'
            )

            output = layer(sample_input_channels_last)

            # Check correct number of expert convolutions
            assert len(layer.conv_layers) == num_kernels

            # Check output shape is consistent
            expected_shape = sample_input_channels_last.shape[:-1] + (32,)
            assert output.shape == expected_shape

    def test_temperature_effect_on_attention(self, sample_input_channels_last):
        """Test that temperature affects attention distribution."""
        # Low temperature - should create sharper attention
        layer_low_temp = DynamicConv2D(
            filters=32, kernel_size=3, padding='same',
            temperature=1.0, num_kernels=4
        )

        # High temperature - should create more uniform attention
        layer_high_temp = DynamicConv2D(
            filters=32, kernel_size=3, padding='same',
            temperature=100.0, num_kernels=4
        )

        # Build both layers
        layer_low_temp(sample_input_channels_last)
        layer_high_temp(sample_input_channels_last)

        # Get attention weights by doing forward pass and checking internal state
        # We can check if the outputs are different (they should be)
        with tf.GradientTape(persistent=True) as tape:
            output_low = layer_low_temp(sample_input_channels_last)
            output_high = layer_high_temp(sample_input_channels_last)

        # Outputs should be different due to different temperature
        diff = keras.ops.mean(keras.ops.abs(output_low - output_high))
        assert keras.ops.convert_to_numpy(diff) > 0

    def test_attention_weights_sum_to_one(self, sample_input_channels_last):
        """Test that attention weights are properly normalized."""
        layer = DynamicConv2D(filters=32, kernel_size=3, padding='same', num_kernels=4)

        # Build layer first
        _ = layer(sample_input_channels_last)

        # Custom forward to check attention weights
        pooled = layer.gap(sample_input_channels_last)
        attention_hidden = layer.attention_dense1(pooled)
        attention_logits = layer.attention_dense2(attention_hidden)
        attention_weights = keras.ops.softmax(attention_logits / layer.temperature)

        # Check that attention weights sum to 1 for each sample
        attention_sums = keras.ops.sum(attention_weights, axis=-1)
        expected_sums = keras.ops.ones_like(attention_sums)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(attention_sums),
            keras.ops.convert_to_numpy(expected_sums),
            rtol=1e-6, atol=1e-6,
            err_msg="Attention weights should sum to 1"
        )

    def test_attention_reduction_ratio_effect(self, sample_input_channels_last):
        """Test different attention reduction ratios."""
        ratios = [2, 4, 8, 16]

        for ratio in ratios:
            layer = DynamicConv2D(
                filters=32, kernel_size=3, padding='same',
                attention_reduction_ratio=ratio
            )

            # Build and forward pass
            output = layer(sample_input_channels_last)

            # Check that first attention dense layer has correct size
            input_channels = sample_input_channels_last.shape[-1]
            expected_hidden_size = max(1, input_channels // ratio)

            # The weights shape should be (input_channels, expected_hidden_size)
            actual_hidden_size = layer.attention_dense1.units
            assert actual_hidden_size == expected_hidden_size

    def test_statistical_properties(self, sample_input_channels_last):
        """Test statistical properties of attention weights."""
        layer = DynamicConv2D(
            filters=32, kernel_size=3, padding='same',
            temperature=30.0, num_kernels=4
        )

        # Build layer
        layer(sample_input_channels_last)

        # Get attention weights for multiple forward passes
        attention_weights_list = []
        for _ in range(10):
            random_input = keras.random.normal(shape=sample_input_channels_last.shape)
            pooled = layer.gap(random_input)
            attention_hidden = layer.attention_dense1(pooled)
            attention_logits = layer.attention_dense2(attention_hidden)
            attention_weights = keras.ops.softmax(attention_logits / layer.temperature)
            attention_weights_list.append(keras.ops.convert_to_numpy(attention_weights))

        # Stack and check variance
        all_attention_weights = np.stack(attention_weights_list)

        # Check that variance is reasonable (not all kernels get exactly same weight)
        var_across_kernels = np.var(all_attention_weights, axis=2)
        mean_var = np.mean(var_across_kernels)

        # With temperature=30 and random initialization, there should be some variance
        assert mean_var > 0, "Attention weights should have some variance"

    def test_temperature_comparison(self, sample_input_channels_last):
        """Compare attention distributions at different temperatures."""
        # Create two layers with different temperatures
        layer_low = DynamicConv2D(
            filters=32, kernel_size=3, padding='same',
            temperature=5.0, num_kernels=4
        )
        layer_high = DynamicConv2D(
            filters=32, kernel_size=3, padding='same',
            temperature=50.0, num_kernels=4
        )

        # Build layers
        layer_low(sample_input_channels_last)
        layer_high(sample_input_channels_last)

        # Get attention weights
        pooled = layer_low.gap(sample_input_channels_last)

        # Low temperature
        attn_hidden_low = layer_low.attention_dense1(pooled)
        attn_logits_low = layer_low.attention_dense2(attn_hidden_low)
        attn_weights_low = keras.ops.softmax(attn_logits_low / layer_low.temperature)

        # High temperature
        attn_hidden_high = layer_high.attention_dense1(pooled)
        attn_logits_high = layer_high.attention_dense2(attn_hidden_high)
        attn_weights_high = keras.ops.softmax(attn_logits_high / layer_high.temperature)

        # Convert to numpy
        attn_low_np = keras.ops.convert_to_numpy(attn_weights_low)
        attn_high_np = keras.ops.convert_to_numpy(attn_weights_high)

        # Calculate variance across kernels for each sample
        var_low = np.var(attn_low_np, axis=1).mean()
        var_high = np.var(attn_high_np, axis=1).mean()

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