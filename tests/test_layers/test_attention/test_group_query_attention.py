"""
Comprehensive test suite for GroupedQueryAttention layer.

Tests cover initialization, build process, forward pass, serialization,
integration, and edge cases following dl-techniques testing standards
and modern Keras 3 patterns.

Updated to match the current GroupedQueryAttention implementation with
correct parameter names and attributes.
"""

import pytest
import numpy as np
import keras
import tempfile
import os
import tensorflow as tf
from typing import List

# Import the layer to test
from dl_techniques.layers.attention.group_query_attention import GroupedQueryAttention


class TestGroupedQueryAttention:
    """Test suite for GroupedQueryAttention layer."""

    @pytest.fixture
    def input_tensor(self):
        """Create a test input tensor."""
        return keras.random.normal([4, 32, 512])  # (batch, seq_len, dim)

    @pytest.fixture
    def basic_layer(self):
        """Create a basic GQA layer for testing."""
        return GroupedQueryAttention(
            dim=512,
            num_heads=8,
            num_kv_heads=2,
            max_seq_len=128
        )

    # =========================================================================
    # Initialization Tests
    # =========================================================================

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        layer = GroupedQueryAttention(
            dim=512,
            num_heads=8,
            num_kv_heads=2,
            max_seq_len=128
        )

        # Check basic parameters
        assert layer.dim == 512
        assert layer.num_heads == 8
        assert layer.num_kv_heads == 2
        assert layer.max_seq_len == 128
        assert layer.dropout_rate == 0.0
        assert layer.rope_percentage == 1.0
        assert layer.rope_theta == 10000.0
        assert layer.use_bias is False

        # Check derived parameters
        assert layer.head_dim == 64  # 512 // 8
        assert layer.num_groups == 4    # 8 // 2

        # Check sub-layers exist (created in __init__)
        assert layer.w_q is not None
        assert layer.w_k is not None
        assert layer.w_v is not None
        assert layer.w_o is not None
        assert layer.dropout is not None
        assert layer.rope is not None

        # But layer should not be built yet
        assert not layer.built

    def test_initialization_custom_parameters(self):
        """Test initialization with custom parameters."""
        layer = GroupedQueryAttention(
            dim=768,
            num_heads=12,
            num_kv_heads=4,
            max_seq_len=256,
            dropout_rate=0.1,
            rope_percentage=0.5,
            rope_theta=50000.0,
            use_bias=True,
            kernel_initializer='he_normal',
            bias_initializer='ones'
        )

        assert layer.dim == 768
        assert layer.num_heads == 12
        assert layer.num_kv_heads == 4
        assert layer.max_seq_len == 256
        assert layer.dropout_rate == 0.1
        assert layer.rope_percentage == 0.5
        assert layer.rope_theta == 50000.0
        assert layer.use_bias is True
        assert layer.head_dim == 64  # 768 // 12
        assert layer.num_groups == 3    # 12 // 4

        # Check initializers are set correctly
        assert isinstance(layer.kernel_initializer, keras.initializers.HeNormal)
        assert isinstance(layer.bias_initializer, keras.initializers.Ones)

    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        # dim not divisible by num_heads
        with pytest.raises(ValueError, match="dim.*must be divisible by num_heads"):
            GroupedQueryAttention(dim=513, num_heads=8, num_kv_heads=2, max_seq_len=128)

        # num_heads not divisible by num_kv_heads
        with pytest.raises(ValueError, match="num_heads.*must be divisible by num_kv_heads"):
            GroupedQueryAttention(dim=504, num_heads=7, num_kv_heads=2, max_seq_len=128)

        # Negative dim
        with pytest.raises(ValueError, match="dim must be positive"):
            GroupedQueryAttention(dim=-512, num_heads=8, num_kv_heads=2, max_seq_len=128)

        # Negative num_heads
        with pytest.raises(ValueError, match="num_heads must be positive"):
            GroupedQueryAttention(dim=512, num_heads=0, num_kv_heads=2, max_seq_len=128)

        # Negative num_kv_heads
        with pytest.raises(ValueError, match="num_kv_heads must be positive"):
            GroupedQueryAttention(dim=512, num_heads=8, num_kv_heads=0, max_seq_len=128)

        # Negative max_seq_len
        with pytest.raises(ValueError, match="max_seq_len must be positive"):
            GroupedQueryAttention(dim=512, num_heads=8, num_kv_heads=2, max_seq_len=-1)

        # Negative rope_theta
        with pytest.raises(ValueError, match="rope_theta must be positive"):
            GroupedQueryAttention(
                dim=512, num_heads=8, num_kv_heads=2, max_seq_len=128, rope_theta=-1.0
            )

        # Invalid dropout rate
        with pytest.raises(ValueError, match="dropout_rate must be in"):
            GroupedQueryAttention(
                dim=512, num_heads=8, num_kv_heads=2, max_seq_len=128, dropout_rate=1.5
            )

        # Invalid rope_percentage (too high)
        with pytest.raises(ValueError, match="rope_percentage must be in"):
            GroupedQueryAttention(
                dim=512, num_heads=8, num_kv_heads=2, max_seq_len=128, rope_percentage=2.0
            )

        # rope_percentage = 0 (should be > 0)
        with pytest.raises(ValueError, match="rope_percentage must be in"):
            GroupedQueryAttention(
                dim=512, num_heads=8, num_kv_heads=2, max_seq_len=128, rope_percentage=0.0
            )

    # =========================================================================
    # Build Process Tests
    # =========================================================================

    def test_build_process(self, basic_layer, input_tensor):
        """Test that the layer builds properly."""
        # Sub-layers should exist but layer should not be built
        assert basic_layer.w_q is not None
        assert basic_layer.w_k is not None
        assert basic_layer.w_v is not None
        assert basic_layer.w_o is not None
        assert basic_layer.dropout is not None
        assert basic_layer.rope is not None
        assert not basic_layer.built

        # Trigger build by calling the layer
        output = basic_layer(input_tensor)

        # After building, layer should be built
        assert basic_layer.built is True

        # Check sublayer types
        assert isinstance(basic_layer.w_q, keras.layers.Dense)
        assert isinstance(basic_layer.w_k, keras.layers.Dense)
        assert isinstance(basic_layer.w_v, keras.layers.Dense)
        assert isinstance(basic_layer.w_o, keras.layers.Dense)
        assert isinstance(basic_layer.dropout, keras.layers.Dropout)

    def test_sublayer_dimensions(self, basic_layer, input_tensor):
        """Test that sublayers have correct dimensions."""
        # Build the layer
        basic_layer(input_tensor)

        # Check projection dimensions
        assert basic_layer.w_q.units == basic_layer.num_heads * basic_layer.head_dim  # 8 * 64 = 512
        assert basic_layer.w_k.units == basic_layer.num_kv_heads * basic_layer.head_dim  # 2 * 64 = 128
        assert basic_layer.w_v.units == basic_layer.num_kv_heads * basic_layer.head_dim  # 2 * 64 = 128
        assert basic_layer.w_o.units == basic_layer.dim  # 512

    def test_bias_configuration(self):
        """Test bias configuration in sublayers."""
        # Test with use_bias=False
        layer_no_bias = GroupedQueryAttention(
            dim=512, num_heads=8, num_kv_heads=2, max_seq_len=128, use_bias=False
        )
        inputs = keras.random.normal([2, 16, 512])
        layer_no_bias(inputs)

        assert layer_no_bias.w_q.use_bias is False
        assert layer_no_bias.w_k.use_bias is False
        assert layer_no_bias.w_v.use_bias is False
        assert layer_no_bias.w_o.use_bias is False

        # Test with use_bias=True
        layer_with_bias = GroupedQueryAttention(
            dim=512, num_heads=8, num_kv_heads=2, max_seq_len=128, use_bias=True
        )
        layer_with_bias(inputs)

        assert layer_with_bias.w_q.use_bias is True
        assert layer_with_bias.w_k.use_bias is True
        assert layer_with_bias.w_v.use_bias is True
        assert layer_with_bias.w_o.use_bias is True

    def test_regularizers_and_initializers(self):
        """Test that regularizers and initializers are properly configured."""
        kernel_reg = keras.regularizers.L2(1e-4)
        bias_reg = keras.regularizers.L1(1e-5)

        layer = GroupedQueryAttention(
            dim=256,
            num_heads=4,
            num_kv_heads=2,
            max_seq_len=64,
            kernel_initializer='he_normal',
            bias_initializer='zeros',
            kernel_regularizer=kernel_reg,
            bias_regularizer=bias_reg,
            use_bias=True
        )

        inputs = keras.random.normal([2, 16, 256])
        layer(inputs)

        # Check initializers
        assert isinstance(layer.kernel_initializer, keras.initializers.HeNormal)
        assert isinstance(layer.bias_initializer, keras.initializers.Zeros)

        # Check regularizers are applied to sub-layers
        assert layer.w_q.kernel_regularizer is not None
        assert layer.w_k.kernel_regularizer is not None
        assert layer.w_v.kernel_regularizer is not None
        assert layer.w_o.kernel_regularizer is not None

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
            (1, 10, 512),   # Single sample, short sequence
            (2, 50, 512),   # Small batch, medium sequence
            (8, 128, 512),  # Larger batch, longer sequence
        ]

        for batch_size, seq_len, dim in test_shapes:
            inputs = keras.random.normal([batch_size, seq_len, dim])
            output = basic_layer(inputs)
            assert output.shape == (batch_size, seq_len, dim)

    def test_training_vs_inference_mode(self, basic_layer, input_tensor):
        """Test different behavior in training vs inference mode."""
        # Training mode
        output_train = basic_layer(input_tensor, training=True)

        # Inference mode
        output_infer = basic_layer(input_tensor, training=False)

        # Shapes should be the same
        assert output_train.shape == output_infer.shape

        # With dropout > 0, outputs might be different
        layer_with_dropout = GroupedQueryAttention(
            dim=512, num_heads=8, num_kv_heads=2, max_seq_len=128, dropout_rate=0.5
        )

        train_out = layer_with_dropout(input_tensor, training=True)
        infer_out = layer_with_dropout(input_tensor, training=False)

        assert train_out.shape == infer_out.shape

    def test_attention_mask_handling(self, basic_layer):
        """Test attention mask functionality."""
        batch_size, seq_len = 2, 16
        inputs = keras.random.normal([batch_size, seq_len, 512])

        # Create a simple causal mask (lower triangular)
        mask = np.tril(np.ones((seq_len, seq_len)))
        mask = np.expand_dims(mask, 0)  # Add batch dimension
        mask = np.repeat(mask, batch_size, axis=0)
        mask = keras.ops.convert_to_tensor(mask, dtype=keras.backend.floatx())

        # Test with mask (note: parameter name is 'attention_mask')
        output_masked = basic_layer(inputs, attention_mask=mask)
        output_unmasked = basic_layer(inputs, attention_mask=None)

        assert output_masked.shape == output_unmasked.shape
        # Outputs should be different when mask is applied
        mask_np = keras.ops.convert_to_numpy(output_masked)
        unmask_np = keras.ops.convert_to_numpy(output_unmasked)
        assert not np.allclose(mask_np, unmask_np, rtol=1e-3)

    def test_return_attention_weights(self, basic_layer, input_tensor):
        """Test returning attention weights."""
        # Test with return_attention_weights=False (default)
        output = basic_layer(input_tensor)
        assert not isinstance(output, tuple)

        # Test with return_attention_weights=True
        output, attention_weights = basic_layer(input_tensor, return_attention_weights=True)

        batch_size, seq_len = input_tensor.shape[0], input_tensor.shape[1]
        expected_attn_shape = (batch_size, basic_layer.num_heads, seq_len, seq_len)

        assert output.shape == input_tensor.shape
        assert attention_weights.shape == expected_attn_shape

        # Check attention weights are valid probabilities
        attn_np = keras.ops.convert_to_numpy(attention_weights)
        assert np.all(attn_np >= 0)
        # Check attention weights sum to 1 along last dimension
        attn_sums = np.sum(attn_np, axis=-1)
        assert np.allclose(attn_sums, 1.0, rtol=1e-5)

    # =========================================================================
    # Shape Computation Tests
    # =========================================================================

    def test_compute_output_shape(self, basic_layer):
        """Test output shape computation."""
        input_shapes = [
            (None, 32, 512),
            (4, None, 512),
            (4, 32, 512),
        ]

        for input_shape in input_shapes:
            output_shape = basic_layer.compute_output_shape(input_shape)
            assert output_shape == input_shape

    # =========================================================================
    # Mathematical Properties Tests
    # =========================================================================

    def test_grouped_attention_property(self):
        """Test that GQA correctly groups query heads with shared K,V heads."""
        layer = GroupedQueryAttention(
            dim=512, num_heads=8, num_kv_heads=2, max_seq_len=64
        )

        inputs = keras.random.normal([1, 16, 512])

        # Get attention weights
        _, attention_weights = layer(inputs, return_attention_weights=True)

        # attention_weights shape: (batch, num_heads, seq_len, seq_len)
        assert attention_weights.shape == (1, 8, 16, 16)

        # With num_heads=8 and num_kv_heads=2, we should have 4 groups
        # Each group of 4 query heads should attend to the same K,V
        assert layer.num_groups == 4

    def test_kv_head_reduction_efficiency(self):
        """Test that GQA reduces K,V parameters correctly."""
        # Compare parameter counts
        full_mha = GroupedQueryAttention(
            dim=512, num_heads=8, num_kv_heads=8, max_seq_len=64  # Full MHA
        )
        gqa = GroupedQueryAttention(
            dim=512, num_heads=8, num_kv_heads=2, max_seq_len=64  # GQA
        )

        inputs = keras.random.normal([1, 16, 512])

        # Build both layers
        full_mha(inputs)
        gqa(inputs)

        # Check K,V projection sizes
        assert full_mha.w_k.units == 8 * 64  # 8 heads * 64 dim = 512
        assert full_mha.w_v.units == 8 * 64  # 8 heads * 64 dim = 512

        assert gqa.w_k.units == 2 * 64   # 2 heads * 64 dim = 128
        assert gqa.w_v.units == 2 * 64   # 2 heads * 64 dim = 128

        # GQA should use fewer parameters for K,V projections
        assert gqa.w_k.units < full_mha.w_k.units
        assert gqa.w_v.units < full_mha.w_v.units

    def test_different_gqa_configurations(self):
        """Test various valid GQA configurations."""
        configs = [
            (512, 8, 1),   # Multi-Query Attention (extreme case)
            (512, 8, 2),   # 4 groups
            (512, 8, 4),   # 2 groups
            (768, 12, 3),  # 4 groups
            (768, 12, 6),  # 2 groups
        ]

        for dim, num_heads, num_kv_heads in configs:
            layer = GroupedQueryAttention(
                dim=dim, num_heads=num_heads, num_kv_heads=num_kv_heads, max_seq_len=64
            )

            inputs = keras.random.normal([2, 16, dim])
            output = layer(inputs)

            assert output.shape == (2, 16, dim)
            assert layer.num_groups == num_heads // num_kv_heads

    # =========================================================================
    # Serialization Tests (Modern Keras 3 Pattern)
    # =========================================================================

    def test_get_config(self, basic_layer):
        """Test configuration serialization."""
        config = basic_layer.get_config()

        expected_keys = {
            'dim', 'num_heads', 'num_kv_heads', 'max_seq_len',
            'dropout_rate', 'rope_percentage', 'rope_theta', 'use_bias',
            'kernel_initializer', 'bias_initializer', 'kernel_regularizer', 'bias_regularizer'
        }

        # Check all expected keys are present
        assert expected_keys.issubset(set(config.keys()))

        # Check values match initialization
        assert config['dim'] == 512
        assert config['num_heads'] == 8
        assert config['num_kv_heads'] == 2
        assert config['max_seq_len'] == 128
        assert config['dropout_rate'] == 0.0
        assert config['rope_percentage'] == 1.0
        assert config['rope_theta'] == 10000.0
        assert config['use_bias'] is False

    def test_serialization_cycle(self, input_tensor):
        """Test complete serialization cycle using modern Keras 3 pattern."""
        # Create original layer
        original_layer = GroupedQueryAttention(
            dim=512,
            num_heads=8,
            num_kv_heads=2,
            max_seq_len=128,
            dropout_rate=0.1,
            use_bias=True,
            name='test_gqa'
        )

        # Build the layer
        original_output = original_layer(input_tensor)

        # Create model for serialization testing
        inputs = keras.Input(shape=input_tensor.shape[1:])
        outputs = GroupedQueryAttention(
            dim=512,
            num_heads=8,
            num_kv_heads=2,
            max_seq_len=128,
            dropout_rate=0.1,
            use_bias=True,
            name='gqa_layer'
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
            'dim': 256,
            'num_heads': 4,
            'num_kv_heads': 2,
            'max_seq_len': 64,
            'dropout_rate': 0.1,
            'rope_percentage': 0.8,
            'rope_theta': 50000.0,
            'use_bias': True
        }

        layer = GroupedQueryAttention(**layer_config)
        config = layer.get_config()

        # Check all config parameters are present
        for key in layer_config:
            assert key in config, f"Missing {key} in get_config()"
            if key not in ['kernel_initializer', 'bias_initializer']:  # Skip serialized objects
                assert config[key] == layer_config[key], f"Mismatch for {key}"

    # =========================================================================
    # Model Integration Tests
    # =========================================================================

    def test_model_integration(self, input_tensor):
        """Test the layer in a complete model context."""
        # Create a simple model using GQA
        inputs = keras.layers.Input(shape=(32, 512))
        x = GroupedQueryAttention(
            dim=512, num_heads=8, num_kv_heads=2, max_seq_len=128
        )(inputs)
        x = keras.layers.LayerNormalization()(x)
        x = keras.layers.Dense(256, activation='relu')(x)
        x = keras.layers.GlobalAveragePooling1D()(x)
        outputs = keras.layers.Dense(10, activation='softmax')(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Compile and test forward pass
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

        # Test prediction
        predictions = model(input_tensor)
        assert predictions.shape == (input_tensor.shape[0], 10)

    def test_model_save_load(self, input_tensor):
        """Test saving and loading a model with GQA layer."""
        # Create model with GQA
        inputs = keras.layers.Input(shape=(32, 512))
        x = GroupedQueryAttention(
            dim=512, num_heads=8, num_kv_heads=2, max_seq_len=128, name='gqa'
        )(inputs)
        x = keras.layers.GlobalAveragePooling1D()(x)
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
            gqa_layer = loaded_model.get_layer('gqa')
            assert isinstance(gqa_layer, GroupedQueryAttention)
            assert gqa_layer.dim == 512
            assert gqa_layer.num_heads == 8
            assert gqa_layer.num_kv_heads == 2

    # =========================================================================
    # Edge Cases and Error Handling
    # =========================================================================

    def test_numerical_stability(self):
        """Test layer stability with extreme input values."""
        layer = GroupedQueryAttention(
            dim=128, num_heads=4, num_kv_heads=2, max_seq_len=64
        )

        # Test with different input magnitudes
        test_cases = [
            keras.ops.zeros((2, 16, 128)),                    # Zeros
            keras.ops.ones((2, 16, 128)) * 1e-10,            # Very small values
            keras.ops.ones((2, 16, 128)) * 1e5,              # Large values
            keras.random.normal((2, 16, 128)) * 100,         # Large random values
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

    def test_variable_sequence_lengths(self):
        """Test handling of different sequence lengths."""
        layer = GroupedQueryAttention(
            dim=256, num_heads=4, num_kv_heads=2, max_seq_len=128
        )

        sequence_lengths = [1, 8, 32, 64, 128]

        for seq_len in sequence_lengths:
            inputs = keras.random.normal([2, seq_len, 256])
            output = layer(inputs)
            assert output.shape == (2, seq_len, 256)

    def test_rope_percentage_variations(self):
        """Test different RoPE percentage configurations."""
        rope_percentages = [0.1, 0.25, 0.5, 0.75, 1.0]  # rope_percentage must be > 0.0

        for rope_pct in rope_percentages:
            layer = GroupedQueryAttention(
                dim=256, num_heads=4, num_kv_heads=2,
                max_seq_len=64, rope_percentage=rope_pct
            )

            inputs = keras.random.normal([2, 16, 256])
            output = layer(inputs)

            assert output.shape == (2, 16, 256)
            output_np = keras.ops.convert_to_numpy(output)
            assert not np.any(np.isnan(output_np))

    # =========================================================================
    # Performance and Memory Tests
    # =========================================================================

    def test_memory_efficiency_comparison(self):
        """Test that GQA uses less memory than full MHA for K,V projections."""
        dim, seq_len = 512, 128

        # Full MHA (baseline)
        full_mha = GroupedQueryAttention(
            dim=dim, num_heads=8, num_kv_heads=8, max_seq_len=seq_len
        )

        # GQA variants
        gqa_4_groups = GroupedQueryAttention(
            dim=dim, num_heads=8, num_kv_heads=2, max_seq_len=seq_len
        )
        gqa_8_groups = GroupedQueryAttention(  # Multi-Query Attention
            dim=dim, num_heads=8, num_kv_heads=1, max_seq_len=seq_len
        )

        inputs = keras.random.normal([4, seq_len, dim])

        # Build all layers
        full_mha(inputs)
        gqa_4_groups(inputs)
        gqa_8_groups(inputs)

        # Check K,V projection parameter counts
        full_kv_params = full_mha.w_k.units + full_mha.w_v.units  # 512 + 512 = 1024
        gqa4_kv_params = gqa_4_groups.w_k.units + gqa_4_groups.w_v.units  # 128 + 128 = 256
        gqa8_kv_params = gqa_8_groups.w_k.units + gqa_8_groups.w_v.units  # 64 + 64 = 128

        # GQA should use progressively fewer parameters
        assert gqa4_kv_params < full_kv_params
        assert gqa8_kv_params < gqa4_kv_params
        assert gqa8_kv_params < full_kv_params

    # =========================================================================
    # Regression Tests
    # =========================================================================

    def test_output_determinism(self):
        """Test that layer behavior is consistent and deterministic."""
        # Create layer with dropout_rate=0 to ensure deterministic behavior
        layer = GroupedQueryAttention(
            dim=256, num_heads=4, num_kv_heads=2, max_seq_len=64, dropout_rate=0.0
        )

        # Use fixed inputs
        inputs = keras.ops.ones([2, 16, 256])

        # Multiple calls with same input should give same output (no randomness)
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
        inputs_different = keras.ops.ones([2, 16, 256]) * 0.5
        output_different = layer(inputs_different, training=False)

        # Different inputs should produce different outputs
        assert not np.allclose(
            keras.ops.convert_to_numpy(output1),
            keras.ops.convert_to_numpy(output_different),
            rtol=1e-3
        )


# Additional utility functions for testing
def create_causal_mask(seq_len: int, batch_size: int = 1) -> keras.KerasTensor:
    """Create a causal (lower triangular) attention mask."""
    mask = np.tril(np.ones((seq_len, seq_len)))
    mask = np.expand_dims(mask, 0)
    mask = np.repeat(mask, batch_size, axis=0)
    return keras.ops.convert_to_tensor(mask, dtype=keras.backend.floatx())


def create_padding_mask(lengths: List[int], max_len: int) -> keras.KerasTensor:
    """Create a padding mask for variable length sequences."""
    batch_size = len(lengths)
    mask = np.zeros((batch_size, max_len, max_len))

    for i, length in enumerate(lengths):
        mask[i, :length, :length] = 1

    return keras.ops.convert_to_tensor(mask, dtype=keras.backend.floatx())


if __name__ == "__main__":
    # Run specific tests
    pytest.main([__file__, "-v"])