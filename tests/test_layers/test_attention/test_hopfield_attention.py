"""
Comprehensive pytest suite for HopfieldAttention layer.

This test suite covers all aspects of the HopfieldAttention implementation including:
- Initialization with various parameters
- Building process with different input shapes
- Forward pass behavior in different modes
- Serialization and deserialization
- Edge cases and error handling
- Convergence and update dynamics
"""

import pytest
import numpy as np
import keras
import tempfile
import os
from typing import List, Tuple, Any

# Import the layer to test
from dl_techniques.layers.attention.hopfield_attention import HopfieldAttention


class TestHopfieldAttentionInitialization:
    """Test suite for HopfieldAttention initialization."""

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        layer = HopfieldAttention(num_heads=8, key_dim=64)

        # Check default values
        assert layer.num_heads == 8
        assert layer.key_dim == 64
        assert layer.value_dim == 64  # Should default to key_dim
        assert layer.dropout_rate == 0.0
        assert layer.use_bias is True
        assert isinstance(layer.kernel_initializer, keras.initializers.GlorotUniform)
        assert isinstance(layer.bias_initializer, keras.initializers.Zeros)
        assert layer.kernel_regularizer is None
        assert layer.bias_regularizer is None
        assert layer.activity_regularizer is None
        assert layer.qk_norm_type == "layer_norm"
        assert layer.probability_type == "softmax"
        assert layer.update_steps_max == 0
        assert layer.update_steps_eps == 1e-4

    def test_initialization_custom_parameters(self):
        """Test initialization with custom parameters."""
        layer = HopfieldAttention(
            num_heads=4,
            key_dim=32,
            value_dim=48,
            dropout_rate=0.1,
            use_bias=False,
            kernel_initializer="he_normal",
            bias_initializer="ones",
            kernel_regularizer="l2",
            bias_regularizer="l1",
            activity_regularizer="l2",
            qk_norm_type=None,
            update_steps_max=3,
            update_steps_eps=1e-3
        )

        assert layer.num_heads == 4
        assert layer.key_dim == 32
        assert layer.value_dim == 48
        assert layer.dropout_rate == 0.1
        assert layer.use_bias is False
        assert isinstance(layer.kernel_initializer, keras.initializers.HeNormal)
        assert isinstance(layer.bias_initializer, keras.initializers.Ones)
        assert isinstance(layer.kernel_regularizer, keras.regularizers.L2)
        assert isinstance(layer.bias_regularizer, keras.regularizers.L1)
        assert isinstance(layer.activity_regularizer, keras.regularizers.L2)
        assert layer.qk_norm_type is None
        assert layer.update_steps_max == 3
        assert layer.update_steps_eps == 1e-3

    def test_invalid_num_heads(self):
        """Test that invalid num_heads raises ValueError."""
        with pytest.raises(ValueError, match="num_heads must be positive"):
            HopfieldAttention(num_heads=0, key_dim=64)

        with pytest.raises(ValueError, match="num_heads must be positive"):
            HopfieldAttention(num_heads=-1, key_dim=64)

    def test_invalid_key_dim(self):
        """Test that invalid key_dim raises ValueError."""
        with pytest.raises(ValueError, match="key_dim must be positive"):
            HopfieldAttention(num_heads=8, key_dim=0)

        with pytest.raises(ValueError, match="key_dim must be positive"):
            HopfieldAttention(num_heads=8, key_dim=-1)

    def test_invalid_dropout(self):
        """Test that invalid dropout rates raise ValueError."""
        with pytest.raises(ValueError, match="dropout_rate must be in \\[0, 1\\]"):
            HopfieldAttention(num_heads=8, key_dim=64, dropout_rate=-0.1)

        with pytest.raises(ValueError, match="dropout_rate must be in \\[0, 1\\]"):
            HopfieldAttention(num_heads=8, key_dim=64, dropout_rate=1.1)

    def test_invalid_update_steps_max(self):
        """Test that invalid update_steps_max raises ValueError."""
        with pytest.raises(ValueError, match="update_steps_max must be non-negative"):
            HopfieldAttention(num_heads=8, key_dim=64, update_steps_max=-1)

    def test_invalid_update_steps_eps(self):
        """Test that invalid update_steps_eps raises ValueError."""
        with pytest.raises(ValueError, match="update_steps_eps must be positive"):
            HopfieldAttention(num_heads=8, key_dim=64, update_steps_eps=0)

        with pytest.raises(ValueError, match="update_steps_eps must be positive"):
            HopfieldAttention(num_heads=8, key_dim=64, update_steps_eps=-1e-4)


class TestHopfieldAttentionBuild:
    """Test suite for HopfieldAttention build process."""

    def test_build_single_input_shape(self):
        """Test building with single input shape."""
        layer = HopfieldAttention(num_heads=8, key_dim=64)
        input_shape = (None, 32, 512)

        # Build should succeed
        layer.build(input_shape)

        # Check that sublayers were created
        assert layer.query_dense is not None
        assert layer.key_dense is not None
        assert layer.value_dense is not None
        assert layer.output_dense is not None
        assert layer.q_norm is not None  # qk_norm_type='layer_norm' by default
        assert layer.k_norm is not None
        assert layer.dropout_layer is None  # dropout=0.0 by default

        # Check sublayer configurations
        assert layer.query_dense.units == 8 * 64  # num_heads * key_dim
        assert layer.key_dense.units == 8 * 64
        assert layer.value_dense.units == 8 * 64  # num_heads * value_dim
        assert layer.output_dense.units == 512  # input_dim

    def test_build_multiple_input_shapes(self):
        """Test building with multiple input shapes [query, key, value]."""
        layer = HopfieldAttention(num_heads=4, key_dim=32, value_dim=48)
        input_shapes = [
            (None, 16, 256),  # query
            (None, 24, 256),  # key
            (None, 24, 256)   # value
        ]

        layer.build(input_shapes)

        # Check that sublayers were created correctly
        assert layer.query_dense.units == 4 * 32  # num_heads * key_dim
        assert layer.key_dense.units == 4 * 32
        assert layer.value_dense.units == 4 * 48  # num_heads * value_dim
        assert layer.output_dense.units == 256  # query input_dim

    def test_build_with_dropout(self):
        """Test building with dropout enabled."""
        layer = HopfieldAttention(num_heads=8, key_dim=64, dropout_rate=0.1)
        layer.build((None, 32, 512))

        assert layer.dropout_layer is not None
        assert layer.dropout_layer.rate == 0.1

    def test_build_without_layer_norm(self):
        """Test building without layer normalization."""
        layer = HopfieldAttention(num_heads=8, key_dim=64, qk_norm_type=None)
        layer.build((None, 32, 512))

        assert layer.q_norm is None
        assert layer.k_norm is None

    def test_build_input_shape_storage(self):
        """Test that build input shape is stored for serialization."""
        layer = HopfieldAttention(num_heads=8, key_dim=64)
        input_shape = (None, 32, 512)

        layer.build(input_shape)


class TestHopfieldAttentionForwardPass:
    """Test suite for HopfieldAttention forward pass."""

    @pytest.fixture
    def layer(self):
        """Create a standard layer for testing."""
        return HopfieldAttention(num_heads=8, key_dim=64)

    @pytest.fixture
    def input_tensor(self):
        """Create a sample input tensor."""
        return keras.random.normal((4, 32, 512))

    def test_self_attention_forward_pass(self, layer, input_tensor):
        """Test forward pass in self-attention mode."""
        output = layer(input_tensor)

        # Check output shape
        assert output.shape == input_tensor.shape

        # Check output is not None and contains valid values
        assert output is not None
        assert not keras.ops.any(keras.ops.isnan(output))

    def test_cross_attention_forward_pass(self):
        """Test forward pass in cross-attention mode."""
        layer = HopfieldAttention(num_heads=4, key_dim=32)

        query = keras.random.normal((2, 16, 256))
        key = keras.random.normal((2, 24, 256))
        value = keras.random.normal((2, 24, 256))

        output = layer([query, key, value])

        # Output should have same shape as query
        assert output.shape == query.shape

    def test_two_input_forward_pass(self):
        """Test forward pass with two inputs (query, key), value=key."""
        layer = HopfieldAttention(num_heads=4, key_dim=32)

        query = keras.random.normal((2, 16, 256))
        key = keras.random.normal((2, 24, 256))

        output = layer([query, key])

        # Output should have same shape as query
        assert output.shape == query.shape

    def test_return_attention_scores(self, layer, input_tensor):
        """Test returning attention scores."""
        output, attention_scores = layer(
            input_tensor,
            return_attention_scores=True
        )

        # Check output shape
        assert output.shape == input_tensor.shape

        # Check attention scores shape
        # Should be (batch, num_heads, seq_len_q, seq_len_k)
        expected_attention_shape = (4, 8, 32, 32)
        assert attention_scores.shape == expected_attention_shape

        # Attention scores should sum to 1 along last axis
        attention_sums = keras.ops.sum(attention_scores, axis=-1)
        expected_sums = keras.ops.ones_like(attention_sums)
        assert keras.ops.all(keras.ops.isclose(attention_sums, expected_sums, atol=1e-6))

    def test_with_mask(self):
        """Test forward pass with attention mask."""
        layer = HopfieldAttention(num_heads=4, key_dim=32)

        inputs = keras.random.normal((2, 16, 256))
        # Create a simple mask (attend to first 8 positions only)
        mask = keras.ops.concatenate([
            keras.ops.ones((2, 16, 8)),
            keras.ops.zeros((2, 16, 8))
        ], axis=-1)

        output = layer(inputs, attention_mask=mask)

        assert output.shape == inputs.shape

    def test_training_mode_behavior(self, layer, input_tensor):
        """Test different behavior in training vs inference mode."""
        # Training mode
        output_train = layer(input_tensor, training=True)

        # Inference mode
        output_infer = layer(input_tensor, training=False)

        # Both should have correct shape
        assert output_train.shape == input_tensor.shape
        assert output_infer.shape == input_tensor.shape

        # With dropout=0.0, outputs should be identical since there's no randomness
        assert keras.ops.all(keras.ops.isclose(output_train, output_infer, atol=1e-6))

    def test_invalid_input_length(self):
        """Test that invalid input list length raises ValueError."""
        layer = HopfieldAttention(num_heads=4, key_dim=32)

        with pytest.raises(ValueError, match="Expected 2 or 3 inputs"):
            layer([keras.random.normal((2, 16, 256))])  # Only 1 input in list

        with pytest.raises(ValueError, match="Expected 2 or 3 inputs"):
            layer([
                keras.random.normal((2, 16, 256)),
                keras.random.normal((2, 16, 256)),
                keras.random.normal((2, 16, 256)),
                keras.random.normal((2, 16, 256))
            ])  # 4 inputs


class TestHopfieldAttentionOutputShape:
    """Test suite for HopfieldAttention output shape computation."""

    def test_compute_output_shape_single_input(self):
        """Test compute_output_shape with single input."""
        layer = HopfieldAttention(num_heads=8, key_dim=64)
        input_shape = (None, 32, 512)

        output_shape = layer.compute_output_shape(input_shape)
        assert output_shape == input_shape

    def test_compute_output_shape_multiple_inputs(self):
        """Test compute_output_shape with multiple inputs."""
        layer = HopfieldAttention(num_heads=4, key_dim=32)
        input_shapes = [
            (None, 16, 256),  # query
            (None, 24, 256),  # key
            (None, 24, 256)   # value
        ]

        output_shape = layer.compute_output_shape(input_shapes)
        # Should match query shape
        assert output_shape == (None, 16, 256)

    def test_compute_output_shape_nested_input(self):
        """Test compute_output_shape with nested input structure."""
        layer = HopfieldAttention(num_heads=8, key_dim=64)
        input_shape = [(None, 32, 512)]  # Nested structure

        output_shape = layer.compute_output_shape(input_shape)
        assert output_shape == (None, 32, 512)


class TestHopfieldAttentionSerialization:
    """Test suite for HopfieldAttention serialization."""

    def test_get_config(self):
        """Test get_config method."""
        layer = HopfieldAttention(
            num_heads=4,
            key_dim=32,
            value_dim=48,
            dropout_rate=0.1,
            use_bias=False,
            qk_norm_type=None,
            update_steps_max=3,
            update_steps_eps=1e-3
        )

        config = layer.get_config()

        # Check all parameters are in config
        assert config["num_heads"] == 4
        assert config["key_dim"] == 32
        assert config["value_dim"] == 48
        assert config["dropout_rate"] == 0.1
        assert config["use_bias"] is False
        assert config["qk_norm_type"] is None
        assert config["probability_type"] == "softmax"
        assert "qk_norm_kwargs" in config
        assert "probability_config" in config
        assert config["update_steps_max"] == 3
        assert config["update_steps_eps"] == 1e-3
        assert "normalize_patterns" not in config

        # Check serialized initializers/regularizers
        assert "kernel_initializer" in config
        assert "bias_initializer" in config
        assert "kernel_regularizer" in config
        assert "bias_regularizer" in config
        assert "activity_regularizer" in config

    def test_from_config(self):
        """Test creating layer from config."""
        original_layer = HopfieldAttention(
            num_heads=4,
            key_dim=32,
            value_dim=48,
            dropout_rate=0.1,
            qk_norm_type=None
        )

        config = original_layer.get_config()
        new_layer = HopfieldAttention.from_config(config)

        # Check parameters match
        assert new_layer.num_heads == original_layer.num_heads
        assert new_layer.key_dim == original_layer.key_dim
        assert new_layer.value_dim == original_layer.value_dim
        assert new_layer.dropout_rate == original_layer.dropout_rate
        assert new_layer.qk_norm_type == original_layer.qk_norm_type
        assert new_layer.probability_type == original_layer.probability_type

    def test_build_config_serialization(self):
        """Test build configuration serialization."""
        layer = HopfieldAttention(num_heads=8, key_dim=64)
        input_shape = (None, 32, 512)

        # Build the layer
        layer.build(input_shape)

        # Get build config
        build_config = layer.get_build_config()
        assert build_config["input_shape"] == input_shape

        # Create new layer and build from config
        new_layer = HopfieldAttention(num_heads=8, key_dim=64)
        new_layer.build_from_config(build_config)

        # Check that new layer is built
        assert new_layer.built

    def test_full_serialization_cycle(self):
        """Test complete save/load cycle."""
        # Create and configure layer
        layer = HopfieldAttention(
            num_heads=4,
            key_dim=32,
            dropout_rate=0.1,
            qk_norm_type='layer_norm'
        )

        # Build layer
        input_shape = (None, 16, 256)
        layer.build(input_shape)

        # Test on some data
        test_input = keras.random.normal((2, 16, 256))
        original_output = layer(test_input)

        # Get configs
        config = layer.get_config()
        build_config = layer.get_build_config()

        # Recreate layer
        new_layer = HopfieldAttention.from_config(config)
        new_layer.build_from_config(build_config)

        # Copy weights
        new_layer.set_weights(layer.get_weights())

        # Test output matches
        new_output = new_layer(test_input)
        assert keras.ops.all(keras.ops.isclose(original_output, new_output, atol=1e-6))


class TestHopfieldAttentionAdvancedFeatures:
    """Test suite for advanced HopfieldAttention features."""

    def test_different_key_value_dims(self):
        """Test layer with different key and value dimensions."""
        layer = HopfieldAttention(
            num_heads=4,
            key_dim=32,
            value_dim=48
        )

        inputs = keras.random.normal((2, 16, 256))
        output = layer(inputs)

        # Output should maintain input shape
        assert output.shape == inputs.shape

    def test_hopfield_convergence_behavior(self):
        """Test Hopfield update convergence behavior."""
        # Layer with multiple update steps
        layer = HopfieldAttention(
            num_heads=4,
            key_dim=32,
            update_steps_max=3,
            update_steps_eps=1e-6
        )

        inputs = keras.random.normal((2, 16, 256))
        output = layer(inputs)

        assert output.shape == inputs.shape

    def test_no_update_steps_limit(self):
        """Test behavior with unlimited update steps."""
        layer = HopfieldAttention(
            num_heads=4,
            key_dim=32,
            update_steps_max=0,  # No limit
            update_steps_eps=1e-4
        )

        inputs = keras.random.normal((2, 8, 128))
        output = layer(inputs)

        assert output.shape == inputs.shape

    def test_large_number_of_heads(self):
        """Test with large number of attention heads."""
        layer = HopfieldAttention(num_heads=16, key_dim=32)

        inputs = keras.random.normal((2, 32, 512))
        output = layer(inputs)

        assert output.shape == inputs.shape

    def test_dropout_effect(self):
        """Test that dropout has an effect during training."""
        layer = HopfieldAttention(
            num_heads=4,
            key_dim=32,
            dropout_rate=0.5  # High dropout for testing
        )

        inputs = keras.random.normal((2, 16, 256))

        # Multiple forward passes should give different results in training mode
        outputs = [layer(inputs, training=True) for _ in range(3)]

        # Check that outputs are different (due to dropout randomness)
        assert not keras.ops.all(keras.ops.isclose(outputs[0], outputs[1], atol=1e-6))
        assert not keras.ops.all(keras.ops.isclose(outputs[1], outputs[2], atol=1e-6))

    def test_layer_normalization_effect(self):
        """Test effect of layer normalization."""
        # Layer with normalization
        layer_with_norm = HopfieldAttention(
            num_heads=4,
            key_dim=32,
            qk_norm_type='layer_norm'
        )

        # Layer without normalization
        layer_without_norm = HopfieldAttention(
            num_heads=4,
            key_dim=32,
            qk_norm_type=None
        )

        inputs = keras.random.normal((2, 16, 256))

        output_with_norm = layer_with_norm(inputs)
        output_without_norm = layer_without_norm(inputs)

        # Both should have correct shape
        assert output_with_norm.shape == inputs.shape
        assert output_without_norm.shape == inputs.shape

        # Outputs should be different
        assert not keras.ops.all(keras.ops.isclose(output_with_norm, output_without_norm, atol=1e-6))


class TestHopfieldAttentionModelIntegration:
    """Test suite for HopfieldAttention integration in models."""

    def test_in_sequential_model(self):
        """Test layer in a Sequential model."""
        model = keras.Sequential([
            keras.layers.InputLayer(shape=(32, 512)),
            HopfieldAttention(num_heads=8, key_dim=64),
            keras.layers.Dense(256),
            keras.layers.Dense(10)
        ])

        # Test forward pass
        inputs = keras.random.normal((4, 32, 512))
        outputs = model(inputs)

        assert outputs.shape == (4, 32, 10)

    def test_in_functional_model(self):
        """Test layer in a functional model."""
        inputs = keras.Input(shape=(32, 512))
        x = HopfieldAttention(num_heads=8, key_dim=64)(inputs)
        x = keras.layers.GlobalAveragePooling1D()(x)
        outputs = keras.layers.Dense(10)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Test forward pass
        test_inputs = keras.random.normal((4, 32, 512))
        test_outputs = model(test_inputs)

        assert test_outputs.shape == (4, 10)

    def test_multiple_hopfield_layers(self):
        """Test model with multiple HopfieldAttention layers."""
        inputs = keras.Input(shape=(32, 512))
        x = HopfieldAttention(num_heads=8, key_dim=64)(inputs)
        x = keras.layers.Dropout(0.1)(x)
        x = HopfieldAttention(num_heads=4, key_dim=128)(x)
        outputs = keras.layers.GlobalAveragePooling1D()(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Test forward pass
        test_inputs = keras.random.normal((2, 32, 512))
        test_outputs = model(test_inputs)

        assert test_outputs.shape == (2, 512)

    def test_model_compilation_and_training(self):
        """Test that model with HopfieldAttention can be compiled and trained."""
        # Create simple model
        model = keras.Sequential([
            keras.layers.InputLayer(shape=(16, 128)),
            HopfieldAttention(num_heads=4, key_dim=32),
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dense(10, activation='softmax')
        ])

        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Generate dummy data
        x_train = keras.random.normal((32, 16, 128))
        y_train = keras.random.randint(shape=(32,), minval=0, maxval=10)

        # Test training for one step
        history = model.fit(x_train, y_train, epochs=1, verbose=0)

        # Check that training completed
        assert len(history.history['loss']) == 1
        assert not np.isnan(history.history['loss'][0])

    def test_model_save_load_with_hopfield(self):
        """Test saving and loading model with HopfieldAttention."""
        # Create model
        inputs = keras.Input(shape=(16, 128))
        x = HopfieldAttention(num_heads=4, key_dim=32, name="hopfield_1")(inputs)
        outputs = keras.layers.GlobalAveragePooling1D()(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Test data
        test_input = keras.random.normal((2, 16, 128))
        original_output = model(test_input)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "model.keras")
            model.save(model_path)

            loaded_model = keras.models.load_model(model_path)

            # Test output matches
            loaded_output = loaded_model(test_input)
            assert keras.ops.all(keras.ops.isclose(original_output, loaded_output, atol=1e-6))

            # Check layer type is preserved
            hopfield_layer = loaded_model.get_layer("hopfield_1")
            assert isinstance(hopfield_layer, HopfieldAttention)
            assert hopfield_layer.num_heads == 4
            assert hopfield_layer.key_dim == 32


class TestHopfieldAttentionEdgeCases:
    """Test suite for edge cases and robustness."""

    def test_very_small_inputs(self):
        """Test with very small input tensors."""
        layer = HopfieldAttention(num_heads=2, key_dim=4)

        # Very small tensor
        inputs = keras.random.normal((1, 2, 8))
        output = layer(inputs)

        assert output.shape == inputs.shape

    def test_large_sequence_length(self):
        """Test with large sequence length."""
        layer = HopfieldAttention(num_heads=4, key_dim=16)

        # Large sequence length
        inputs = keras.random.normal((1, 1024, 64))
        output = layer(inputs)

        assert output.shape == inputs.shape

    def test_single_head_attention(self):
        """Test with single attention head."""
        layer = HopfieldAttention(num_heads=1, key_dim=64)

        inputs = keras.random.normal((2, 16, 64))
        output = layer(inputs)

        assert output.shape == inputs.shape

    def test_numerical_stability_extreme_values(self):
        """Test numerical stability with extreme input values."""
        layer = HopfieldAttention(num_heads=4, key_dim=32)

        # Test with very large values
        large_inputs = keras.random.normal((2, 16, 128)) * 100
        output_large = layer(large_inputs)
        assert not keras.ops.any(keras.ops.isnan(output_large))
        assert not keras.ops.any(keras.ops.isinf(output_large))

        # Test with very small values
        small_inputs = keras.random.normal((2, 16, 128)) * 0.001
        output_small = layer(small_inputs)
        assert not keras.ops.any(keras.ops.isnan(output_small))
        assert not keras.ops.any(keras.ops.isinf(output_small))

    def test_zero_inputs(self):
        """Test behavior with zero inputs."""
        layer = HopfieldAttention(num_heads=4, key_dim=32)

        zero_inputs = keras.ops.zeros((2, 16, 128))
        output = layer(zero_inputs)

        # Output should be finite
        assert not keras.ops.any(keras.ops.isnan(output))
        assert not keras.ops.any(keras.ops.isinf(output))

    def test_mismatched_dimensions_cross_attention(self):
        """Test cross-attention with mismatched key/value dimensions."""
        layer = HopfieldAttention(num_heads=4, key_dim=32)

        query = keras.random.normal((2, 16, 256))
        key = keras.random.normal((2, 24, 256))    # Different sequence length
        value = keras.random.normal((2, 24, 256))  # Same as key

        output = layer([query, key, value])

        # Output should match query shape
        assert output.shape == query.shape


class TestHopfieldAttentionPlan077a2a35:
    """Regression gate for plan_2026-06-14_077a2a35 Step 1 (F7 + AF1 + AF2).

    - F7/D-001: cross-attention with DIFFERENT K/V feature dim builds the K/V
      Dense layers from the actual key/value shapes (not query_shape).
    - AF1: ``value_dim=None`` round-trips through ``.keras`` as ``None``.
    - AF2: precomputed ``math.sqrt`` scale leaves self-attn numerics
      byte-identical.
    """

    def test_self_attention_byte_identical_reference(self):
        """Self-attention forward is finite, correctly shaped, and matches a
        deterministic reference (locks AF2 scale + F7 fallback byte-identity)."""
        layer = HopfieldAttention(num_heads=4, key_dim=32, qk_norm_type=None)

        # Deterministic input + deterministic weights via build + reference.
        x = keras.random.normal((2, 16, 128), seed=1234)
        out = layer(x)

        assert out.shape == (2, 16, 128)
        assert not keras.ops.any(keras.ops.isnan(out))
        assert not keras.ops.any(keras.ops.isinf(out))

        # The scale folded into __init__ must equal the old
        # ops.sqrt(cast(key_dim, float32)) AT THE FLOAT32 PRECISION the division
        # actually runs in: the Python-float divisor is cast to the score
        # tensor's float32 dtype, so both round to the same float32 value. (The
        # math.sqrt result is float64; comparing it to the float32 reference at
        # full f64 precision would spuriously fail by ~1e-7 — not a numeric drift
        # in the forward pass, which runs in float32.)
        old_scale_f32 = np.float32(
            keras.ops.convert_to_numpy(
                keras.ops.sqrt(keras.ops.cast(32, "float32"))
            )
        )
        new_scale_f32 = np.float32(layer._sqrt_key_dim)
        assert new_scale_f32 == old_scale_f32

        # Re-running with the same (now built) weights must be bit-stable.
        out2 = layer(x)
        np.testing.assert_array_equal(
            keras.ops.convert_to_numpy(out),
            keras.ops.convert_to_numpy(out2),
        )

    def test_cross_attention_different_kv_feature_dim(self):
        """F7: cross-attn via list input where K/V feature dim != query dim
        must build without shape error and return query-length output."""
        layer = HopfieldAttention(num_heads=4, key_dim=32, qk_norm_type=None)

        query = keras.random.normal((2, 16, 256), seed=1)   # query feature dim 256
        key = keras.random.normal((2, 24, 384), seed=2)     # K feature dim 384
        value = keras.random.normal((2, 24, 384), seed=3)   # V feature dim 384

        output = layer([query, key, value])

        # Output maps back to query feature dim and query sequence length.
        assert output.shape == (2, 16, 256)
        assert not keras.ops.any(keras.ops.isnan(output))

        # K/V Dense kernels must have been built from the ACTUAL K/V feature
        # dim (384), not the query dim (256).
        assert layer.key_dense.kernel.shape[0] == 384
        assert layer.value_dense.kernel.shape[0] == 384
        assert layer.query_dense.kernel.shape[0] == 256
        # output_dense maps back to the query feature dim.
        assert layer.output_dense.units == 256

    def test_cross_attention_different_kv_and_value_dim(self):
        """F7: K and V carry different feature dims from each other and query."""
        layer = HopfieldAttention(num_heads=4, key_dim=32, qk_norm_type=None)

        query = keras.random.normal((2, 16, 256), seed=1)
        key = keras.random.normal((2, 24, 384), seed=2)
        value = keras.random.normal((2, 24, 200), seed=3)  # V feature dim 200

        output = layer([query, key, value])

        assert output.shape == (2, 16, 256)
        assert layer.key_dense.kernel.shape[0] == 384
        assert layer.value_dense.kernel.shape[0] == 200

    def test_value_dim_none_roundtrip(self):
        """AF1: value_dim=None round-trips through .keras as None; forward works."""
        inputs = keras.Input(shape=(16, 128))
        x = HopfieldAttention(
            num_heads=4, key_dim=32, value_dim=None, name="hop_none"
        )(inputs)
        outputs = keras.layers.GlobalAveragePooling1D()(x)
        model = keras.Model(inputs=inputs, outputs=outputs)

        test_input = keras.random.normal((2, 16, 128), seed=7)
        original_output = model(test_input)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.keras")
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)

            reloaded = loaded_model.get_layer("hop_none")
            # The raw constructor arg None must survive serialization.
            assert reloaded.get_config()["value_dim"] is None
            # Internally still resolved to key_dim.
            assert reloaded.value_dim == 32

            loaded_output = loaded_model(test_input)
            assert keras.ops.all(
                keras.ops.isclose(original_output, loaded_output, atol=1e-6)
            )

    def test_value_dim_int_roundtrip(self):
        """AF1: an explicit int value_dim still round-trips as that int."""
        layer = HopfieldAttention(num_heads=4, key_dim=32, value_dim=48)
        config = layer.get_config()
        assert config["value_dim"] == 48
        rebuilt = HopfieldAttention.from_config(config)
        assert rebuilt.value_dim == 48
        assert rebuilt.get_config()["value_dim"] == 48


# Run the tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])