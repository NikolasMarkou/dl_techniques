"""
Comprehensive test suite for MobileMQA layer.

Tests cover initialization, build process, forward pass, serialization,
integration, and edge cases following dl-techniques testing standards
and modern Keras 3 patterns.

Updated to reflect MobileMQA's inheritance from GroupedQueryAttention.
"""

import pytest
import numpy as np
import keras
import tempfile
import os
import tensorflow as tf

from dl_techniques.layers.attention.mobile_mqa import MobileMQA
from dl_techniques.layers.attention.group_query_attention import GroupedQueryAttention


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

    def test_inheritance_and_defaults(self):
        """Test inheritance structure and default parameters."""
        layer = MobileMQA(dim=64, num_heads=8)
        
        # Verify inheritance
        assert isinstance(layer, MobileMQA)
        assert isinstance(layer, GroupedQueryAttention)

        # Check basic parameters
        assert layer.dim == 64
        assert layer.num_heads == 8
        assert layer.head_dim == 8  # 64 // 8
        assert layer.use_downsampling is False
        
        # Verify MQA enforcement (Hardcoded in __init__)
        assert layer.num_kv_heads == 1
        assert layer.rope_percentage == 0.0
        assert layer.num_groups == 8  # num_heads // num_kv_heads

        # Check initializers are set
        assert isinstance(layer.kernel_initializer, keras.initializers.HeNormal)
        assert layer.kernel_regularizer is None

        # Check sub-layers exist (inherited from GQA)
        assert layer.w_q is not None
        assert layer.w_k is not None
        assert layer.w_v is not None
        assert layer.w_o is not None
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
        
        # Check initializers are set correctly
        assert isinstance(layer.kernel_initializer, keras.initializers.GlorotUniform)
        assert layer.kernel_regularizer is not None

        # Check sub-layers exist including downsample
        assert layer.w_q is not None
        assert layer.w_k is not None
        assert layer.w_v is not None
        assert layer.w_o is not None
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
        assert basic_layer.w_q is not None
        assert basic_layer.w_k is not None
        assert basic_layer.w_o is not None
        assert basic_layer.lambda_param is None
        assert not basic_layer.built

        # Trigger build by calling the layer
        output = basic_layer(input_tensor)

        # After building, layer should be built
        assert basic_layer.built is True
        assert basic_layer.lambda_param is not None

        # Check sublayer types
        assert isinstance(basic_layer.w_q, keras.layers.Dense)
        assert isinstance(basic_layer.w_k, keras.layers.Dense)
        assert isinstance(basic_layer.w_v, keras.layers.Dense)
        assert isinstance(basic_layer.w_o, keras.layers.Dense)

    def test_sublayer_dimensions(self, basic_layer, input_tensor):
        """Test that sublayers have correct dimensions for MQA."""
        # Build the layer
        basic_layer(input_tensor)

        # Check projection dimensions
        # Query: projects to all heads
        assert basic_layer.w_q.units == basic_layer.dim  # 64
        
        # Key/Value: projects to single head (MQA)
        # Note: In GQA refactor, K and V are separate layers
        assert basic_layer.w_k.units == basic_layer.head_dim  # 8
        assert basic_layer.w_v.units == basic_layer.head_dim  # 8
        
        # Output: projects back to dim
        assert basic_layer.w_o.units == basic_layer.dim  # 64

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
        assert layer_with_reg.w_q.kernel_regularizer is not None
        assert layer_with_reg.w_k.kernel_regularizer is not None
        assert layer_with_reg.w_v.kernel_regularizer is not None
        assert layer_with_reg.w_o.kernel_regularizer is not None

        # Test without regularizers
        layer_no_reg = MobileMQA(
            dim=64,
            num_heads=8,
            kernel_initializer='he_normal'
        )
        layer_no_reg(inputs)

        # Without regularizers, they should be None
        assert layer_no_reg.w_q.kernel_regularizer is None
        assert layer_no_reg.w_k.kernel_regularizer is None
        assert layer_no_reg.w_v.kernel_regularizer is None
        assert layer_no_reg.w_o.kernel_regularizer is None

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

        # Both should produce same output shape (downsampling is internal to KV)
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
            (64, 8),    # dim=64, num_heads=8
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
        # across all query heads (via broadcasting in call()).
        # In GQA/MobileMQA, w_k and w_v units equal num_kv_heads * head_dim.
        # Since MobileMQA forces num_kv_heads=1, units should equal head_dim.
        assert layer.w_k.units == layer.head_dim
        assert layer.w_v.units == layer.head_dim
        
        # While Q projection outputs dim for all heads
        assert layer.w_q.units == layer.dim

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

    def test_downsampling_spatial_reduction(self):
        """Test that downsampling layer is configured for reduction."""
        layer = MobileMQA(dim=64, num_heads=8, use_downsampling=True)

        # Create inputs with known spatial dimensions
        inputs = keras.random.normal([1, 32, 32, 64])
        output = layer(inputs)

        # Output shape should remain the same
        assert output.shape == (1, 32, 32, 64)

        # But downsample layer should exist and be configured correctly
        # The reduction happens inside call(), we verify the component exists
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
        
        # We also expect filter keys NOT to be present (MQA specific forces)
        forbidden_keys = {'num_kv_heads', 'rope_percentage'}

        # Check all expected keys are present
        assert expected_keys.issubset(set(config.keys()))
        
        # Check filtered keys are absent (as implemented in MobileMQA.get_config)
        for key in forbidden_keys:
            assert key not in config

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
        # Q projects to full dim (all heads)
        assert layer.w_q.units == 64
        
        # K and V project to single head dim each (num_kv_heads=1)
        # This is 1/8th of the params of a standard attention Key/Value projection if heads=8
        assert layer.w_k.units == layer.head_dim
        assert layer.w_v.units == layer.head_dim
        
        # Verify large reduction in params: (Q + K + V) vs (Q + K*8 + V*8)
        total_proj_units = layer.w_q.units + layer.w_k.units + layer.w_v.units
        assert total_proj_units == 64 + 8 + 8  # 80 units
        
        # In standard attention, it would be 64 + 64 + 64 = 192 units
        assert total_proj_units < (layer.dim * 3)

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
