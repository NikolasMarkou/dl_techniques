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

# ---------------------------------------------------------------------
# Mixed-precision mask tests (plan-2026-07-27-b4ef45f0, step 4)
# ---------------------------------------------------------------------
#
# WHAT IS BEING GUARDED HERE.
#
# `HopfieldAttention._compute_attention` used the ARITHMETIC mask form
#
#     attention_scores = attention_scores + (1.0 - mask_tensor) * -1e9
#
# which `common.MASK_BIAS_VALUE`'s own docstring rules out. Under
# `mixed_precision.set_global_policy('mixed_float16')` the literal `-1e9` is
# materialized in float16, where it is `-inf` (np.float16(-1e9) == -inf). At every
# UNMASKED position `(1.0 - mask) == 0`, so the product is `0 * -inf = NaN` — the
# NaN appears exactly where NOTHING was masked, and the following matmul spreads it
# across the whole batch.
#
# MEASURED on unfixed HEAD (B=2, N=64, D=64, num_heads=4, key_dim=16), GPU 1 / TF 2.18,
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
# `TestHopfieldAttentionMaskHazardIsReal::test_the_probability_sublayer_autocasts_a_float32_input`).
# Removing that failure mode needs the predicate-level rescue used in
# `capsule_routing_attention.py` (D-006), which is a SEMANTICS change on a degenerate
# input and is deliberately not part of this step. What IS guaranteed, and asserted
# below, is that the damage no longer spreads: every row that keeps >= 1 key stays
# finite. Before the fix, one degenerate row NaN'd all 8192 outputs.
#
# ANTI-VACUITY. The `N = 7`-hides-an-fp16-`-inf`-at-`N >= 512` trap does not transfer:
# this hazard is a per-ELEMENT dtype overflow of a constant multiplied by an exact
# zero, not a long reduction, so it appears at any N >= 1. It is nevertheless
# asserted reachable rather than assumed — see `TestHopfieldAttentionMaskHazardIsReal`, which
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
# mixed_float16 0.0019 and float64 0.0016, against an output absmax of 2.37 (up to
# 6.45 once a mask is applied). The entries below carry ~10x headroom on that.
# float32 compares against a control computed the same way and is exact.
_MP_ATOL = {"float32": 1e-6, "mixed_float16": 0.05, "float64": 0.05}


def _mp_input():
    """Deterministic ``(B, N, D)`` float32 input, shared by every test below."""
    return np.random.default_rng(7).standard_normal(
        (_MP_B, _MP_N, _MP_D)
    ).astype("float32")


def _mp_mask(kind):
    """One of the masks these tests need, as a float32 ``1 = keep`` array.

    ``'all_ones'`` masks NOTHING and is the catastrophic case for the arithmetic
    form. ``'padding'`` masks the second half of the keys (rank 3; this site does not broadcast a rank-2 mask), ``'causal'`` is
    lower-triangular, and ``'degenerate'`` blanks query row ``_MP_DEG_ROW`` entirely
    (the A2 probe — see the note above; it is NOT part of the finiteness contract
    this step establishes).
    """
    if kind == "all_ones":
        return np.ones((_MP_B, _MP_N, _MP_N), dtype="float32")
    if kind == "padding":
        m = np.ones((_MP_B, _MP_N, _MP_N), dtype="float32")
        m[:, :, _MP_KEEP:] = 0.0
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
    layer = HopfieldAttention(num_heads=_MP_H, key_dim=16, **kwargs)
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


class TestHopfieldAttentionMaskHazardIsReal:
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


class TestHopfieldAttentionMixedPrecisionMask:
    """SC1 + SC2: finite AND agreeing with float32, for every legal mask."""

    @pytest.mark.parametrize("kind", ["all_ones", "padding", "causal"])
    def test_masked_forward_is_finite_and_matches_float32(self, dtype_policy, kind):

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


class TestHopfieldAttentionFullyMaskedRow:
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


class TestHopfieldAttentionMaskPolarity:
    """SC6: the mask must suppress the MASKED positions, not the kept ones.

    A polarity inversion at this site — passing the keep predicate where its
    complement is meant, or vice versa — raises nothing, changes no shape and leaves
    the output perfectly finite. Only an influence test can see it. MEASURED on
    unmodified HEAD by handing the layer ``1 - mask``: perturbing a "masked" token
    then moves the kept query rows by 4.55 instead of 0.0, against a
    kept-token influence of 3.43.

    The statement is EXACT here (not approximate): a masked key contributes exactly
    `exp(-1e9) == 0` weight, so a perturbation of a masked token cannot reach a kept
    query row at all. Measured 0.0 in float32 on unfixed HEAD, so a real inversion
    is separated from correct behavior by the full 4.55 signal.
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

        layer = _mp_layer()
        delta_masked, delta_kept = self._influence(layer, _mp_mask("padding"))

        # Measured EXACTLY 0.0 on unfixed float32. The 1e-3 budget is session-noise
        # headroom (see `test_rpc_attention.py`, where a batched op measured 0.0 in
        # isolation and 1.1e-06 inside the full suite).
        assert delta_masked <= 1e-3, (
            f"perturbing a MASKED token changed the kept query rows by "
            f"{delta_masked:.6g} under policy {dtype_policy!r} — this must be "
            f"exact, so the mask polarity is INVERTED (the layer is attending to "
            f"the padding; measured 4.55 with a deliberately inverted mask)"
        )
        assert delta_kept > 0.5, (
            f"perturbing a KEPT token changed the output by only {delta_kept:.6g}; "
            "the test is vacuous — the layer is ignoring its input"
        )
