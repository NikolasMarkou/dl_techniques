"""
Comprehensive test suite for MobileMQA layer.

Tests cover initialization, build process, forward pass, serialization,
integration, and edge cases following dl-techniques testing standards
and modern Keras 3 patterns.
"""

import pytest
import numpy as np
import keras
import tempfile
import os
import tensorflow as tf

from dl_techniques.layers.attention.mobile_mqa import MobileMQA


class TestMobileMQA:
    """Test suite for MobileMQA layer."""

    @pytest.fixture
    def input_tensor(self):
        """Create a test input tensor - 4D for image data."""
        return keras.random.normal([4, 32, 32, 64])  # (batch, height, width, channels)

    @pytest.fixture
    def basic_layer(self):
        """Create a basic MobileMQA layer for testing."""
        return MobileMQA(
            dim=64,
            num_heads=8,
            use_downsampling=False
        )

    # =========================================================================
    # Initialization Tests
    # =========================================================================

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        layer = MobileMQA(dim=64, num_heads=8)

        # Check basic parameters
        assert layer.dim == 64
        assert layer.num_heads == 8
        assert layer.head_dim == 8  # 64 // 8
        assert layer.use_downsampling is False

        # Check initializers are set
        assert isinstance(layer.kernel_initializer, keras.initializers.HeNormal)
        assert layer.kernel_regularizer is None

        # Check sub-layers exist (created in __init__)
        assert layer.q_proj is not None
        assert layer.kv_proj is not None
        assert layer.o_proj is not None
        assert layer.downsample is None  # Should be None when use_downsampling=False

        # Layer should not be built yet
        assert not layer.built
        assert layer.lambda_param is None  # Created in build()

    def test_initialization_custom_parameters(self):
        """Test initialization with custom parameters."""
        kernel_reg = keras.regularizers.L2(1e-4)
        layer = MobileMQA(
            dim=32,
            num_heads=16,
            use_downsampling=True,
            kernel_initializer='glorot_uniform',
            kernel_regularizer=kernel_reg
        )

        assert layer.dim == 32
        assert layer.num_heads == 16
        assert layer.head_dim == 2  # 32 // 16
        assert layer.use_downsampling is True
        # FIX: Test scale against the correct head_dim
        assert layer.scale == (layer.head_dim ** -0.5)

        # Check initializers are set correctly
        assert isinstance(layer.kernel_initializer, keras.initializers.GlorotUniform)
        assert layer.kernel_regularizer is not None

        # Check sub-layers exist including downsample
        assert layer.q_proj is not None
        assert layer.kv_proj is not None
        assert layer.o_proj is not None
        assert layer.downsample is not None  # Should exist when use_downsampling=True

    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        # Negative dim
        with pytest.raises(ValueError, match="dim must be positive"):
            MobileMQA(dim=-64, num_heads=8)

        # Zero dim
        with pytest.raises(ValueError, match="dim must be positive"):
            MobileMQA(dim=0, num_heads=8)

        # Negative num_heads
        with pytest.raises(ValueError, match="num_heads must be positive"):
            MobileMQA(dim=64, num_heads=0)

        # dim not divisible by num_heads
        with pytest.raises(ValueError, match="dim.*must be divisible by num_heads"):
            MobileMQA(dim=257, num_heads=8)

    # =========================================================================
    # Build Process Tests
    # =========================================================================

    def test_build_process(self, basic_layer, input_tensor):
        """Test that the layer builds properly."""
        # Sub-layers should exist but layer should not be built
        assert basic_layer.q_proj is not None
        assert basic_layer.kv_proj is not None
        assert basic_layer.o_proj is not None
        assert basic_layer.lambda_param is None  # Not created yet
        assert not basic_layer.built

        # Trigger build by calling the layer
        output = basic_layer(input_tensor)

        # After building, layer should be built
        assert basic_layer.built is True
        assert basic_layer.lambda_param is not None  # Should be created in build()

        # Check sublayer types
        assert isinstance(basic_layer.q_proj, keras.layers.Dense)
        assert isinstance(basic_layer.kv_proj, keras.layers.Dense)
        assert isinstance(basic_layer.o_proj, keras.layers.Dense)

    def test_sublayer_dimensions(self, basic_layer, input_tensor):
        """Test that sublayers have correct dimensions."""
        # Build the layer
        basic_layer(input_tensor)

        # Check projection dimensions
        assert basic_layer.q_proj.units == basic_layer.dim  # 64
        # FIX: Test against correct MQA logic (2 * head_dim)
        assert basic_layer.kv_proj.units == 2 * basic_layer.head_dim
        assert basic_layer.o_proj.units == basic_layer.dim  # 64

    def test_downsample_layer_creation(self):
        """Test that downsample layer is created when use_downsampling=True."""
        # Test with downsampling disabled
        layer_no_downsample = MobileMQA(dim=64, num_heads=8, use_downsampling=False)
        assert layer_no_downsample.downsample is None

        # Test with downsampling enabled (no regularizer)
        layer_with_downsample = MobileMQA(dim=64, num_heads=8, use_downsampling=True)
        assert layer_with_downsample.downsample is not None
        assert isinstance(layer_with_downsample.downsample, keras.layers.DepthwiseConv2D)

        # Test with downsampling and regularizer
        kernel_reg = keras.regularizers.L2(1e-4)
        layer_with_reg = MobileMQA(
            dim=64,
            num_heads=8,
            use_downsampling=True,
            kernel_regularizer=kernel_reg
        )
        assert layer_with_reg.downsample is not None

        # Check downsample layer properties
        inputs = keras.random.normal([2, 16, 16, 64])
        layer_with_downsample(inputs)
        layer_with_reg(inputs)

        # Check basic properties
        downsample = layer_with_downsample.downsample
        assert downsample.kernel_size == (3, 3)
        assert downsample.strides == (2, 2)
        assert downsample.padding == 'same'

    def test_regularizers_and_initializers(self):
        """Test that regularizers and initializers are properly configured."""
        kernel_reg = keras.regularizers.L2(1e-4)

        # Test with regularizers
        layer_with_reg = MobileMQA(
            dim=64,
            num_heads=8,
            kernel_initializer='he_normal',
            kernel_regularizer=kernel_reg
        )

        inputs = keras.random.normal([2, 16, 16, 64])
        layer_with_reg(inputs)

        # Check initializers
        assert isinstance(layer_with_reg.kernel_initializer, keras.initializers.HeNormal)

        # Check regularizers are applied to Dense sub-layers
        assert layer_with_reg.q_proj.kernel_regularizer is not None
        assert layer_with_reg.kv_proj.kernel_regularizer is not None
        assert layer_with_reg.o_proj.kernel_regularizer is not None

        # Test without regularizers
        layer_no_reg = MobileMQA(
            dim=64,
            num_heads=8,
            kernel_initializer='he_normal'
        )
        layer_no_reg(inputs)

        # Without regularizers, they should be None
        assert layer_no_reg.q_proj.kernel_regularizer is None
        assert layer_no_reg.kv_proj.kernel_regularizer is None
        assert layer_no_reg.o_proj.kernel_regularizer is None

    def test_lambda_parameter_creation(self, basic_layer, input_tensor):
        """Test that lambda parameter is created correctly in build()."""
        # Before build
        assert basic_layer.lambda_param is None

        # After build
        basic_layer(input_tensor)
        assert basic_layer.lambda_param is not None

        # Check lambda parameter properties
        assert basic_layer.lambda_param.shape == ()  # Scalar
        assert basic_layer.lambda_param.trainable is True

    # =========================================================================
    # Forward Pass Tests
    # =========================================================================

    def test_forward_pass_basic(self, basic_layer, input_tensor):
        """Test basic forward pass functionality."""
        output = basic_layer(input_tensor)

        # Check output shape matches input shape
        assert output.shape == input_tensor.shape

        # Check output contains no NaN or Inf values
        output_np = keras.ops.convert_to_numpy(output)
        assert not np.any(np.isnan(output_np))
        assert not np.any(np.isinf(output_np))

    def test_forward_pass_different_shapes(self, basic_layer):
        """Test forward pass with different input shapes."""
        test_shapes = [
            (1, 8, 8, 64),     # Small spatial dimensions
            (2, 16, 16, 64),   # Medium spatial dimensions
            (4, 32, 32, 64),   # Larger spatial dimensions
            (2, 64, 64, 64),   # Large spatial dimensions
        ]

        for batch_size, height, width, channels in test_shapes:
            inputs = keras.random.normal([batch_size, height, width, channels])
            output = basic_layer(inputs)
            assert output.shape == (batch_size, height, width, channels)

    def test_training_vs_inference_mode(self, basic_layer, input_tensor):
        """Test behavior in different training modes."""
        # Training mode
        output_train = basic_layer(input_tensor, training=True)

        # Inference mode
        output_infer = basic_layer(input_tensor, training=False)

        # Shapes should be the same
        assert output_train.shape == output_infer.shape
        assert output_train.shape == input_tensor.shape

    def test_downsampling_functionality(self):
        """Test downsampling behavior when enabled."""
        # Layer without downsampling
        layer_no_downsample = MobileMQA(dim=64, num_heads=8, use_downsampling=False)

        # Layer with downsampling
        layer_with_downsample = MobileMQA(dim=64, num_heads=8, use_downsampling=True)

        inputs = keras.random.normal([2, 32, 32, 64])

        # Both should produce same output shape (downsampling is internal)
        output_no_downsample = layer_no_downsample(inputs)
        output_with_downsample = layer_with_downsample(inputs)

        assert output_no_downsample.shape == inputs.shape
        assert output_with_downsample.shape == inputs.shape

        # But outputs should be different due to different computation
        no_down_np = keras.ops.convert_to_numpy(output_no_downsample)
        with_down_np = keras.ops.convert_to_numpy(output_with_downsample)

        # They should be different (not exactly equal)
        assert not np.allclose(no_down_np, with_down_np, rtol=1e-3)

    def test_different_channel_dimensions(self):
        """Test layer with different channel dimensions."""
        test_configs = [
            (128, 8),   # dim=128, num_heads=8
            (64, 8),   # dim=64, num_heads=8
            (512, 16),  # dim=512, num_heads=16
            (384, 12),  # dim=384, num_heads=12
        ]

        for dim, num_heads in test_configs:
            layer = MobileMQA(dim=dim, num_heads=num_heads)
            inputs = keras.random.normal([2, 16, 16, dim])
            output = layer(inputs)

            assert output.shape == (2, 16, 16, dim)
            assert layer.head_dim == dim // num_heads

    # =========================================================================
    # Shape Computation Tests
    # =========================================================================

    def test_compute_output_shape(self, basic_layer):
        """Test output shape computation."""
        input_shapes = [
            (None, 32, 32, 64),
            (4, None, 32, 64),
            (4, 32, None, 64),
            (4, 32, 32, 64),
        ]

        for input_shape in input_shapes:
            output_shape = basic_layer.compute_output_shape(input_shape)
            assert output_shape == input_shape

    # =========================================================================
    # Mathematical Properties Tests
    # =========================================================================

    def test_multi_query_attention_property(self):
        """Test that MobileMQA correctly implements multi-query attention."""
        layer = MobileMQA(dim=64, num_heads=8)
        inputs = keras.random.normal([1, 16, 16, 64])

        # Build layer
        layer(inputs)

        # The key property: K and V projections are small (head_dim) and shared
        # across all query heads.
        # FIX: Test against correct MQA logic (2 * head_dim)
        assert layer.kv_proj.units == 2 * layer.head_dim
        # While Q projection outputs dim for all heads
        assert layer.q_proj.units == layer.dim

    def test_spatial_flattening_and_reshaping(self):
        """Test that spatial dimensions are correctly flattened and reshaped."""
        layer = MobileMQA(dim=64, num_heads=8)

        # Test with different spatial sizes
        test_cases = [
            (2, 8, 8, 64),    # Small
            (2, 16, 16, 64),  # Medium
            (2, 32, 32, 64),  # Large
        ]

        for batch, height, width, channels in test_cases:
            inputs = keras.random.normal([batch, height, width, channels])
            output = layer(inputs)

            # Output should maintain original 4D shape
            assert output.shape == (batch, height, width, channels)

    def test_attention_scale_factor(self, basic_layer, input_tensor):
        """Test that attention scaling is applied correctly."""
        basic_layer(input_tensor)  # Build layer

        expected_scale = basic_layer.head_dim ** -0.5
        assert abs(basic_layer.scale - expected_scale) < 1e-6

    def test_downsampling_spatial_reduction(self):
        """Test that downsampling reduces spatial dimensions of K,V internally."""
        layer = MobileMQA(dim=64, num_heads=8, use_downsampling=True)

        # Create inputs with known spatial dimensions
        inputs = keras.random.normal([1, 32, 32, 64])
        output = layer(inputs)

        # Output shape should remain the same
        assert output.shape == (1, 32, 32, 64)

        # But downsample layer should exist and be configured correctly
        assert layer.downsample is not None
        assert layer.downsample.strides == (2, 2)

    # =========================================================================
    # Serialization Tests (Modern Keras 3 Pattern)
    # =========================================================================

    def test_get_config(self, basic_layer):
        """Test configuration serialization."""
        config = basic_layer.get_config()

        expected_keys = {
            'dim', 'num_heads', 'use_downsampling',
            'kernel_initializer', 'kernel_regularizer'
        }

        # Check all expected keys are present
        assert expected_keys.issubset(set(config.keys()))

        # Check values match initialization
        assert config['dim'] == 64
        assert config['num_heads'] == 8
        assert config['use_downsampling'] is False

    def test_serialization_cycle(self, input_tensor):
        """Test complete serialization cycle using modern Keras 3 pattern."""
        # Create model for serialization testing
        inputs = keras.Input(shape=input_tensor.shape[1:])
        outputs = MobileMQA(
            dim=64,
            num_heads=8,
            use_downsampling=True,
            name='mobile_mqa_layer'
        )(inputs)
        model = keras.Model(inputs, outputs)

        # Get prediction from original model
        original_prediction = model(input_tensor)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_prediction = loaded_model(input_tensor)

            # Verify identical predictions
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_prediction),
                keras.ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization"
            )

    def test_config_completeness(self):
        """Test that get_config contains all __init__ parameters."""
        layer_config = {
            'dim': 384,
            'num_heads': 12,
            'use_downsampling': True,
            'kernel_initializer': 'glorot_uniform'
        }

        layer = MobileMQA(**layer_config)
        config = layer.get_config()

        # Check all config parameters are present
        for key in layer_config:
            assert key in config, f"Missing {key} in get_config()"

    # =========================================================================
    # Model Integration Tests
    # =========================================================================

    def test_model_integration(self, input_tensor):
        """Test the layer in a complete model context."""
        # Create a simple model using MobileMQA
        inputs = keras.layers.Input(shape=(32, 32, 64))
        x = MobileMQA(dim=64, num_heads=8)(inputs)
        x = keras.layers.LayerNormalization()(x)
        x = keras.layers.GlobalAveragePooling2D()(x)  # Pool spatial dimensions
        x = keras.layers.Dense(128, activation='relu')(x)
        outputs = keras.layers.Dense(10, activation='softmax')(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Compile and test forward pass
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

        # Test prediction
        predictions = model(input_tensor)
        assert predictions.shape == (input_tensor.shape[0], 10)

    def test_model_save_load(self, input_tensor):
        """Test saving and loading a model with MobileMQA layer."""
        # Create model with MobileMQA
        inputs = keras.layers.Input(shape=(32, 32, 64))
        x = MobileMQA(dim=64, num_heads=8, use_downsampling=True, name='mobile_mqa')(inputs)
        x = keras.layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(5)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Generate prediction before saving
        original_prediction = model(input_tensor)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "model.keras")
            model.save(model_path)

            loaded_model = keras.models.load_model(model_path)

            # Test prediction with loaded model
            loaded_prediction = loaded_model(input_tensor)

            # Predictions should match
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_prediction),
                keras.ops.convert_to_numpy(loaded_prediction),
                rtol=1e-5, atol=1e-5
            )

            # Check layer type is preserved
            mqa_layer = loaded_model.get_layer('mobile_mqa')
            assert isinstance(mqa_layer, MobileMQA)
            assert mqa_layer.dim == 64
            assert mqa_layer.num_heads == 8
            assert mqa_layer.use_downsampling is True

    def test_multiple_mqa_layers(self):
        """Test model with multiple MobileMQA layers."""
        inputs = keras.Input(shape=(16, 16, 128))

        # Stack multiple MobileMQA layers
        x = MobileMQA(dim=128, num_heads=4, name='mqa1')(inputs)
        x = keras.layers.LayerNormalization()(x)
        x = MobileMQA(dim=128, num_heads=8, use_downsampling=True, name='mqa2')(x)
        x = keras.layers.LayerNormalization()(x)
        x = keras.layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(10)(x)

        model = keras.Model(inputs, outputs)

        test_input = keras.random.normal([2, 16, 16, 128])
        prediction = model(test_input)
        assert prediction.shape == (2, 10)

    # =========================================================================
    # Edge Cases and Error Handling
    # =========================================================================

    def test_numerical_stability(self):
        """Test layer stability with extreme input values."""
        layer = MobileMQA(dim=128, num_heads=4)

        # Test with different input magnitudes
        test_cases = [
            keras.ops.zeros((2, 16, 16, 128)),                    # Zeros
            keras.ops.ones((2, 16, 16, 128)) * 1e-10,            # Very small values
            keras.ops.ones((2, 16, 16, 128)) * 1e3,              # Large values
            keras.random.normal((2, 16, 16, 128)) * 10,          # Large random values
        ]

        for test_input in test_cases:
            output = layer(test_input)

            # Check for NaN/Inf values
            output_np = keras.ops.convert_to_numpy(output)
            assert not np.any(np.isnan(output_np)), "NaN values detected"
            assert not np.any(np.isinf(output_np)), "Inf values detected"

    def test_gradient_flow(self, basic_layer, input_tensor):
        """Test gradient flow through the layer."""
        with tf.GradientTape() as tape:
            inputs = tf.Variable(keras.ops.convert_to_numpy(input_tensor))
            tape.watch(inputs)
            outputs = basic_layer(inputs)
            loss = tf.reduce_mean(tf.square(outputs))

        # Get gradients
        grads = tape.gradient(loss, basic_layer.trainable_variables)

        # Check gradients exist and are not None
        assert all(g is not None for g in grads)

        # Check gradients have reasonable values
        for grad in grads:
            grad_np = keras.ops.convert_to_numpy(grad)
            assert not np.any(np.isnan(grad_np))
            assert not np.any(np.isinf(grad_np))

    def test_variable_spatial_dimensions(self):
        """Test handling of different spatial dimensions."""
        layer = MobileMQA(dim=64, num_heads=8)

        spatial_sizes = [(8, 8), (16, 16), (32, 32), (64, 64)]

        for height, width in spatial_sizes:
            inputs = keras.random.normal([2, height, width, 64])
            output = layer(inputs)
            assert output.shape == (2, height, width, 64)

    def test_minimum_spatial_dimensions(self):
        """Test with very small spatial dimensions."""
        layer = MobileMQA(dim=64, num_heads=8)

        # Test with 1x1 spatial dimensions
        inputs = keras.random.normal([2, 1, 1, 64])
        output = layer(inputs)
        assert output.shape == (2, 1, 1, 64)

        # Test with 2x2 spatial dimensions
        inputs = keras.random.normal([2, 2, 2, 64])
        output = layer(inputs)
        assert output.shape == (2, 2, 2, 64)

    # =========================================================================
    # Performance and Memory Tests
    # =========================================================================

    def test_memory_efficiency_comparison(self):
        """Test that MobileMQA uses shared K,V projections efficiently."""
        # Create layer
        layer = MobileMQA(dim=64, num_heads=8)
        inputs = keras.random.normal([2, 16, 16, 64])

        # Build layer
        layer(inputs)

        # Check projection dimensions - key insight of MQA
        assert layer.q_proj.units == 64  # Full dim for queries (all heads)
        # FIX: Test against correct MQA logic (2 * head_dim)
        assert layer.kv_proj.units == 2 * layer.head_dim
        assert layer.o_proj.units == 64  # Full dim for output

    def test_computational_efficiency_with_downsampling(self):
        """Test computational benefits of downsampling option."""
        # Layer without downsampling
        layer_no_downsample = MobileMQA(dim=64, num_heads=8, use_downsampling=False)

        # Layer with downsampling
        layer_with_downsample = MobileMQA(dim=64, num_heads=8, use_downsampling=True)

        # Large spatial input to see downsampling benefit
        inputs = keras.random.normal([1, 64, 64, 64])

        # Both should work, but downsampling version should handle larger inputs more efficiently
        output_no_downsample = layer_no_downsample(inputs)
        output_with_downsample = layer_with_downsample(inputs)

        assert output_no_downsample.shape == (1, 64, 64, 64)
        assert output_with_downsample.shape == (1, 64, 64, 64)

    # =========================================================================
    # Regression Tests
    # =========================================================================

    def test_output_determinism(self):
        """Test that layer behavior is consistent and deterministic."""
        layer = MobileMQA(dim=64, num_heads=8)

        # Use fixed inputs
        inputs = keras.ops.ones([2, 16, 16, 64])

        # Multiple calls with same input should give same output
        output1 = layer(inputs, training=False)
        output2 = layer(inputs, training=False)

        # Should be exactly the same (deterministic computation)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output1),
            keras.ops.convert_to_numpy(output2),
            rtol=1e-6, atol=1e-6,
            err_msg="Outputs should be deterministic"
        )

        # Test that layer produces different outputs for different inputs
        inputs_different = keras.ops.ones([2, 16, 16, 64]) * 0.5
        output_different = layer(inputs_different, training=False)

        # Different inputs should produce different outputs
        assert not np.allclose(
            keras.ops.convert_to_numpy(output1),
            keras.ops.convert_to_numpy(output_different),
            rtol=1e-3
        )

    def test_consistency_across_builds(self):
        """Test that rebuilt layers produce consistent outputs."""
        config = {
            'dim': 32,
            'num_heads': 4,
            'use_downsampling': True,
            'kernel_initializer': 'he_normal'
        }

        # Create two identical layers
        layer1 = MobileMQA(**config)
        layer2 = MobileMQA(**config)

        inputs = keras.random.normal([2, 16, 16, 32])

        # Build both layers
        output1 = layer1(inputs)
        output2 = layer2(inputs)

        # They should have the same shape
        assert output1.shape == output2.shape

        # But different outputs due to different weight initialization
        assert not np.allclose(
            keras.ops.convert_to_numpy(output1),
            keras.ops.convert_to_numpy(output2),
            rtol=1e-3
        )

if __name__ == "__main__":
    # Run specific tests
    pytest.main([__file__, "-v"])