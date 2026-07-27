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

# ---------------------------------------------------------------------
# Mixed-precision mask tests (plan-2026-07-27-b4ef45f0, step 4)
# ---------------------------------------------------------------------
#
# WHAT IS BEING GUARDED HERE.
#
# `GroupedQueryAttention._apply_mask` used the ARITHMETIC mask form
#
#     additive_mask = (1.0 - ops.cast(mask, scores.dtype)) * -1e9
#
# which `common.MASK_BIAS_VALUE`'s own docstring rules out. Under
# `mixed_precision.set_global_policy('mixed_float16')` the literal `-1e9` is
# materialized in float16, where it is `-inf` (np.float16(-1e9) == -inf). At every
# UNMASKED position `(1.0 - mask) == 0`, so the product is `0 * -inf = NaN` — the
# NaN appears exactly where NOTHING was masked, and the following matmul spreads it
# across the whole batch.
#
# MEASURED on unfixed HEAD (B=2, N=64, D=64, num_heads=4, num_kv_heads=2), GPU 1 / TF 2.18,
# non-finite entries in the layer OUTPUT:
#
#     policy            no mask   all-ones mask   padding mask   causal mask
#     float32            0/8192      0/8192          0/8192        0/8192
#     mixed_float16      0/8192   8192/8192       8192/8192     8192/8192
#
# The all-ones column is the important one: a mask that masks NOTHING destroys the
# entire batch. That is not a pathological input — it is what a caller passes when
# every sequence in the batch happens to be full length.
#
# THE FIX is `common.apply_attention_mask`, which builds the bias with `ops.where`
# inside `common.mask_dtype(...)` (>= float32), so `0 * -inf` cannot be formed at
# all. Each site keeps its own broadcast/cast order and its own polarity spelling.
#
# WHAT THIS FIX DOES **NOT** COVER (assumption A2, measured at step 4): a FULLY-MASKED
# query row. The biased logits are cast back to the compute dtype at this site, so in
# fp16 an all-masked row is all-`-inf` again and `softmax(all -inf)` is `0/0 = NaN`.
# Casting back is not the problem and `out_dtype=None` would not help: the softmax
# here is `self.attn_prob`, a Keras layer with autocasting ON, which drags a float32
# tensor straight back to float16 (pinned by
# `TestGroupedQueryAttentionMaskHazardIsReal::test_the_probability_sublayer_autocasts_a_float32_input`).
# Removing that failure mode needs the predicate-level rescue used in
# `capsule_routing_attention.py` (D-006), which is a SEMANTICS change on a degenerate
# input and is deliberately not part of this step. What IS guaranteed, and asserted
# below, is that the damage no longer spreads: every row that keeps >= 1 key stays
# finite. Before the fix, one degenerate row NaN'd all 8192 outputs.
#
# ANTI-VACUITY. The `N = 7`-hides-an-fp16-`-inf`-at-`N >= 512` trap does not transfer:
# this hazard is a per-ELEMENT dtype overflow of a constant multiplied by an exact
# zero, not a long reduction, so it appears at any N >= 1. It is nevertheless
# asserted reachable rather than assumed — see `TestGroupedQueryAttentionMaskHazardIsReal`, which
# checks that the policy really selects float16 compute, that
# `float16(MASK_BIAS_VALUE)` really is `-inf`, that `float16(0) * that` really is
# `NaN`, and that each mask really has the structure its name claims. N = 64 keys is
# a realistic sequence length for this layer rather than a toy one.

from dl_techniques.layers.attention.common import MASK_BIAS_VALUE

_MP_B, _MP_N, _MP_D, _MP_H = 2, 64, 64, 4
_MP_KEEP = _MP_N // 2            # first half kept, second half masked (padding mask)
_MP_DEG_ROW = 5                  # the query row the 'degenerate' mask blanks entirely
_MP_SEED = 1234

# Absolute tolerance for "this policy's masked forward agrees with the float32 one".
#
# PRE-REGISTERED, not tuned after the fact: sized from the layer's own NO-MASK dtype
# error measured on unfixed HEAD (the only forward that survives fp16 there), which is
# the honest budget — a correct mask fix should leave the masked path no worse
# conditioned than the unmasked one. Measured no-mask max |policy - float32|:
# mixed_float16 0.0127, against an output absmax of 5.99 (up to
# 7.57 once a mask is applied). The entries below carry ~10x headroom on that.
# float32 compares against a control computed the same way and is exact.
_MP_ATOL = {"float32": 1e-6, "mixed_float16": 0.1}

# `float64` is EXCLUDED from the mask tests at this site, and that exclusion is itself
# pinned by `TestGroupedQueryAttentionFloat64IsASeparatePreExistingDefect` below: under a float64
# policy this layer raises inside `RotaryPositionEmbedding` ("cannot compute Mul as
# input #1 was expected to be a double tensor but is a float tensor") with NO mask
# supplied at all. It is a pre-existing RoPE dtype defect, entirely independent of
# masking, and out of scope for this step — but skipping it silently would hide it.


def _mp_input():
    """Deterministic ``(B, N, D)`` float32 input, shared by every test below."""
    return np.random.default_rng(7).standard_normal(
        (_MP_B, _MP_N, _MP_D)
    ).astype("float32")


def _mp_mask(kind):
    """One of the masks these tests need, as a float32 ``1 = keep`` array.

    ``'all_ones'`` masks NOTHING and is the catastrophic case for the arithmetic
    form. ``'padding'`` masks the second half of the keys with a rank-2 ``(B, N)`` mask, exercising this site's rank-2 broadcast branch, ``'causal'`` is
    lower-triangular, and ``'degenerate'`` blanks query row ``_MP_DEG_ROW`` entirely
    (the A2 probe — see the note above; it is NOT part of the finiteness contract
    this step establishes).
    """
    if kind == "all_ones":
        return np.ones((_MP_B, _MP_N, _MP_N), dtype="float32")
    if kind == "padding":
        m = np.ones((_MP_B, _MP_N), dtype="float32")
        m[:, _MP_KEEP:] = 0.0
        return m
    if kind == "causal":
        return np.broadcast_to(
            np.tril(np.ones((_MP_N, _MP_N), dtype="float32")), (_MP_B, _MP_N, _MP_N)
        ).copy()
    if kind == "degenerate":
        m = np.ones((_MP_B, _MP_N, _MP_N), dtype="float32")
        m[:, _MP_DEG_ROW, :] = 0.0
        return m
    raise ValueError(f"unknown mask kind {kind!r}")


def _mp_layer(**kwargs):
    """A built layer whose TRAINABLE weights are identical under every dtype policy.

    Seeding the initializers is NOT sufficient: a ``glorot_uniform`` draw under a
    ``float64`` policy differs from the same-seed draw under ``float32`` (the
    initializer samples in the VARIABLE dtype), so a cross-policy comparison on
    seeded-but-not-assigned weights measures the initializer, not the code under
    test. Explicit values are assigned instead. Non-trainable buffers (e.g. the RoPE
    cos/sin cache) are left as the layer computes them.
    """
    layer = GroupedQueryAttention(dim=_MP_D, num_heads=_MP_H, num_kv_heads=2, max_seq_len=128, **kwargs)
    layer.build((_MP_B, _MP_N, _MP_D))
    rng = np.random.default_rng(_MP_SEED)
    for weight in layer.trainable_weights:
        shape = tuple(weight.shape)
        if len(shape) == 1 and ("bias" in weight.name or "beta" in weight.name):
            value = np.zeros(shape)
        elif len(shape) == 1:                     # a scale / gamma: keep it near 1
            value = 1.0 + 0.1 * rng.standard_normal(shape)
        else:
            value = 0.2 * rng.standard_normal(shape)
        weight.assign(keras.ops.cast(
            keras.ops.convert_to_tensor(value.astype("float32")), weight.dtype
        ))
    return layer


def _mp_forward(layer, array, mask):
    """One masked forward pass, returned as float64 numpy."""
    out = layer(keras.ops.convert_to_tensor(array), attention_mask=(None if mask is None else keras.ops.convert_to_tensor(mask)))
    if isinstance(out, (list, tuple)):
        out = out[0]
    return keras.ops.convert_to_numpy(out).astype("float64")


_F32_REFERENCE = {}


def _float32_reference(kind):
    """Masked float32 output for ``kind``, memoized, under an explicit policy.

    This is the CONTROL every mixed-precision assertion compares against. It sets and
    restores the policy itself, so it is valid whichever parametrization of
    ``dtype_policy`` happens to reach it first.
    """
    if kind not in _F32_REFERENCE:
        previous = keras.mixed_precision.global_policy().name
        keras.mixed_precision.set_global_policy("float32")
        try:
            layer = _mp_layer()
            _F32_REFERENCE[kind] = _mp_forward(
                layer, _mp_input(), _mp_mask(kind)
            )
        finally:
            keras.mixed_precision.set_global_policy(previous)
    return _F32_REFERENCE[kind]


class TestGroupedQueryAttentionMaskHazardIsReal:
    """Anti-vacuity. If these stop holding, every fp16 test below is worthless."""

    def test_policy_really_selects_float16_compute(self, dtype_policy):
        expected = {
            "float32": "float32",
            "mixed_float16": "float16",
            "float64": "float64",
        }[dtype_policy]
        assert keras.mixed_precision.global_policy().compute_dtype == expected

    def test_the_arithmetic_form_really_is_nan_in_the_compute_dtype(self):
        with np.errstate(over="ignore", invalid="ignore"):
            bias = np.float16(MASK_BIAS_VALUE)
            assert np.isneginf(bias), (
                "anti-vacuity FAILED: float16(MASK_BIAS_VALUE) is not -inf, so the "
                "`0 * -inf` hazard this module guards is not reproducible here."
            )
            assert np.isnan(np.float16(0.0) * bias), (
                "anti-vacuity FAILED: float16(0) * float16(MASK_BIAS_VALUE) is not "
                "NaN — the arithmetic mask form would be harmless."
            )
        assert np.isfinite(np.float32(MASK_BIAS_VALUE)), (
            "anti-vacuity FAILED: MASK_BIAS_VALUE is not finite in float32, so "
            "`mask_dtype(...)` would not be a fix."
        )

    def test_the_all_ones_mask_really_masks_nothing(self):
        mask = _mp_mask("all_ones")
        assert int((mask == 0).sum()) == 0, (
            "the 'all_ones' mask masks something; it no longer reproduces the "
            "signature catastrophe (a vacuous mask destroying the batch)"
        )

    def test_the_partial_masks_really_mask_something(self):
        for kind in ("padding", "causal"):
            mask = _mp_mask(kind)
            assert int((mask == 0).sum()) > 0, (
                f"the {kind!r} mask masks nothing; it cannot detect a regression "
                "in the masking code"
            )
            rows = mask if mask.ndim == 3 else mask[:, None, :]
            assert not (rows == 0).all(axis=-1).any(), (
                f"the {kind!r} mask has a fully-masked query row, so it no longer "
                "isolates the covered case from the A2 probe"
            )

    def test_the_degenerate_mask_really_has_exactly_one_fully_masked_row(self):
        mask = _mp_mask("degenerate")
        empty = (mask == 0).all(axis=-1)
        assert int(empty.sum()) == _MP_B, (
            f"expected exactly one fully-masked query row per batch element, got "
            f"{int(empty.sum())} across {_MP_B} batch elements"
        )
        assert empty[:, _MP_DEG_ROW].all()

    def test_the_probability_sublayer_autocasts_a_float32_input(self, dtype_policy):
        """Why ``out_dtype`` cannot rescue a fully-masked row at this site.

        The softmax here is a Keras LAYER with autocasting on, so handing it a
        carefully-promoted float32 tensor changes nothing — it is seen inside its
        own ``call()`` as the compute dtype. This is the same measurement that
        selected the predicate-level rescue in `capsule_routing_attention.py`
        (assumption A4). If Keras ever stops doing this, this test fails and the
        ``out_dtype`` choice at this site can be revisited.
        """
        layer = _mp_layer()
        prob = layer.attn_prob
        assert getattr(prob, "autocast", False) is True

        seen = {}
        original = prob.call

        def spy(x, *args, **kwargs):
            seen["dtype"] = keras.backend.standardize_dtype(x.dtype)
            return original(x, *args, **kwargs)

        prob.call = spy
        try:
            prob(keras.ops.convert_to_tensor(np.zeros((1, _MP_H, 4, 4), dtype="float32")))
        finally:
            prob.call = original

        expected = keras.mixed_precision.global_policy().compute_dtype
        assert seen["dtype"] == expected, (
            f"a float32 tensor entering `attn_prob` was seen inside its call() as "
            f"{seen['dtype']!r}, not the compute dtype {expected!r}"
        )


class TestGroupedQueryAttentionMixedPrecisionMask:
    """SC1 + SC2: finite AND agreeing with float32, for every legal mask."""

    @pytest.mark.parametrize("kind", ["all_ones", "padding", "causal"])
    def test_masked_forward_is_finite_and_matches_float32(self, dtype_policy, kind):
        if dtype_policy == "float64":
            pytest.skip(
                "float64 raises inside RotaryPositionEmbedding regardless of masking "
                "(pre-existing, pinned by "
                "TestGroupedQueryAttentionFloat64IsASeparatePreExistingDefect)"
            )

        layer = _mp_layer()
        out = _mp_forward(layer, _mp_input(), _mp_mask(kind))

        n_bad = int((~np.isfinite(out)).sum())
        assert n_bad == 0, (
            f"{n_bad}/{out.size} non-finite output entries for a {kind!r} mask "
            f"under policy {dtype_policy!r}"
        )

        reference = _float32_reference(kind)
        atol = _MP_ATOL[dtype_policy]
        max_dev = float(np.abs(out - reference).max())
        assert max_dev <= atol, (
            f"{kind!r}-masked forward under {dtype_policy!r} deviates from the "
            f"float32 control by {max_dev:.4g} > {atol:.4g}"
        )
        assert float(np.abs(out).max()) > 0.5 * float(np.abs(reference).max()), (
            f"output absmax {np.abs(out).max():.4g} collapsed relative to the "
            f"float32 control {np.abs(reference).max():.4g}"
        )


class TestGroupedQueryAttentionFullyMaskedRow:
    """Assumption A2 at this site, as an executable statement of what is guaranteed.

    A2 predicted that casting the biased logits back to the compute dtype is safe
    here because every softmax row keeps at least one position. A caller CAN break
    that by masking a whole query row, so the boundary is measured rather than
    assumed. What this step guarantees — and what is asserted — is CONTAINMENT: rows
    that keep >= 1 key stay finite even when a sibling row is degenerate. Before the
    fix, one degenerate row made all 8192 outputs NaN.

    The degenerate row's own value is deliberately NOT compared against the float32
    control: in float32/float64 it is a uniform, meaningless distribution over all
    keys (garbage in, garbage out), and its numeric value is genuinely dtype-
    dependent. Fixing it needs the predicate-level rescue of
    `capsule_routing_attention.py` (decisions.md D-006), which is a semantics change
    outside this step.
    """

    def test_a_fully_masked_row_does_not_poison_the_rest_of_the_batch(
        self, dtype_policy
    ):
        if dtype_policy == "float64":
            pytest.skip(
                "float64 raises inside RotaryPositionEmbedding regardless of masking "
                "(pre-existing, pinned by "
                "TestGroupedQueryAttentionFloat64IsASeparatePreExistingDefect)"
            )

        layer = _mp_layer()
        out = _mp_forward(layer, _mp_input(), _mp_mask("degenerate"))

        kept = np.delete(out, _MP_DEG_ROW, axis=1)
        n_bad = int((~np.isfinite(kept)).sum())
        assert n_bad == 0, (
            f"{n_bad}/{kept.size} non-finite entries in the rows that KEEP keys, "
            f"under policy {dtype_policy!r} — a single degenerate query row is "
            "still poisoning the whole batch"
        )

        reference = np.delete(_float32_reference("degenerate"), _MP_DEG_ROW, axis=1)
        atol = _MP_ATOL[dtype_policy]
        max_dev = float(np.abs(kept - reference).max())
        assert max_dev <= atol, (
            f"the non-degenerate rows under {dtype_policy!r} deviate from the "
            f"float32 control by {max_dev:.4g} > {atol:.4g}"
        )

        if dtype_policy != "mixed_float16":
            assert np.isfinite(out[:, _MP_DEG_ROW]).all(), (
                f"the fully-masked query row is not finite under {dtype_policy!r}, "
                "where MASK_BIAS_VALUE is representable — that is a regression, not "
                "the known fp16 boundary"
            )


class TestGroupedQueryAttentionMaskPolarity:
    """SC6: the mask must suppress the MASKED positions, not the kept ones.

    A polarity inversion at this site — passing the keep predicate where its
    complement is meant, or vice versa — raises nothing, changes no shape and leaves
    the output perfectly finite. Only an influence test can see it. MEASURED on
    unmodified HEAD by handing the layer ``1 - mask``: perturbing a "masked" token
    then moves the kept query rows by 27.9 instead of 0.0, against a
    kept-token influence of 25.9.

    The statement is EXACT here (not approximate): a masked key contributes exactly
    `exp(-1e9) == 0` weight, so a perturbation of a masked token cannot reach a kept
    query row at all. Measured 0.0 in float32 on unfixed HEAD, so a real inversion
    is separated from correct behavior by the full 27.9 signal.
    """

    @staticmethod
    def _influence(layer, mask):
        base_input = _mp_input()
        perturbed_masked = base_input.copy()
        perturbed_masked[:, _MP_KEEP + 3, :] += 5.0      # a MASKED token
        perturbed_kept = base_input.copy()
        perturbed_kept[:, 3, :] += 5.0                   # a KEPT token

        rows = slice(0, _MP_KEEP)
        base = _mp_forward(layer, base_input, mask)
        assert np.isfinite(base[:, rows]).all(), (
            "the kept query rows are not finite; the comparison below would be "
            "meaningless"
        )
        delta_masked = float(
            np.abs(_mp_forward(layer, perturbed_masked, mask)[:, rows]
                   - base[:, rows]).max()
        )
        delta_kept = float(
            np.abs(_mp_forward(layer, perturbed_kept, mask)[:, rows]
                   - base[:, rows]).max()
        )
        return delta_masked, delta_kept

    def test_a_masked_token_has_no_influence_on_the_kept_rows(self, dtype_policy):
        if dtype_policy == "float64":
            pytest.skip(
                "float64 raises inside RotaryPositionEmbedding regardless of masking "
                "(pre-existing, pinned by "
                "TestGroupedQueryAttentionFloat64IsASeparatePreExistingDefect)"
            )

        layer = _mp_layer()
        delta_masked, delta_kept = self._influence(layer, _mp_mask("padding"))

        # Measured EXACTLY 0.0 on unfixed float32. The 1e-3 budget is session-noise
        # headroom (see `test_rpc_attention.py`, where a batched op measured 0.0 in
        # isolation and 1.1e-06 inside the full suite).
        assert delta_masked <= 1e-3, (
            f"perturbing a MASKED token changed the kept query rows by "
            f"{delta_masked:.6g} under policy {dtype_policy!r} — this must be "
            f"exact, so the mask polarity is INVERTED (the layer is attending to "
            f"the padding; measured 27.9 with a deliberately inverted mask)"
        )
        assert delta_kept > 1.0, (
            f"perturbing a KEPT token changed the output by only {delta_kept:.6g}; "
            "the test is vacuous — the layer is ignoring its input"
        )


class TestGroupedQueryAttentionFloat64IsASeparatePreExistingDefect:
    """Pins the reason the mask tests above skip ``float64`` at this site.

    Under a ``float64`` global policy this layer raises inside
    :class:`RotaryPositionEmbedding` — with **no mask supplied** — because the cached
    cos/sin buffer stays float32 while the projected queries are float64. It has
    nothing to do with the attention mask and is not fixed by this step; this test
    exists so the skip above is a DOCUMENTED exclusion rather than a silent one, and
    so it turns red the day someone fixes RoPE.
    """

    def test_float64_policy_raises_in_rope_even_without_a_mask(self):
        previous = keras.mixed_precision.global_policy().name
        keras.mixed_precision.set_global_policy("float64")
        try:
            layer = _mp_layer()
            with pytest.raises(Exception, match="double tensor but is a float tensor"):
                _mp_forward(layer, _mp_input(), None)
        finally:
            keras.mixed_precision.set_global_policy(previous)
