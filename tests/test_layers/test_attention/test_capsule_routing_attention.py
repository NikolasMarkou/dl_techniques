"""
Refined test suite for CapsuleRoutingSelfAttention layer implementation.

This test suite is specifically designed to validate the new CapsuleRoutingSelfAttention
implementation with proper error message matching, comprehensive routing algorithm testing,
and robust serialization validation.
"""

import pytest
import numpy as np
import keras
import os
import tempfile
import tensorflow as tf
from keras import ops

# Ensure the correct path is used for the import
from dl_techniques.layers.attention.capsule_routing_attention import CapsuleRoutingSelfAttention


class TestCapsuleRoutingSelfAttention:
    """Comprehensive test suite for CapsuleRoutingSelfAttention layer implementation."""

    @pytest.fixture
    def input_tensor(self):
        """Create a test input tensor with proper shape for attention."""
        # Shape: (batch_size, seq_len, embed_dim)
        return keras.random.normal([4, 16, 128])  # 4 batches, 16 tokens, 128 dims

    @pytest.fixture
    def layer_instance(self):
        """Create a default layer instance for testing."""
        return CapsuleRoutingSelfAttention(num_heads=8, key_dim=16)

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        layer = CapsuleRoutingSelfAttention(num_heads=8, key_dim=64)

        # Check default values
        assert layer.num_heads == 8
        assert layer.key_dim == 64
        assert layer.value_dim is None
        assert layer.dropout_rate == 0.0
        assert layer.use_bias is True
        assert isinstance(layer.kernel_initializer, keras.initializers.GlorotUniform)
        assert isinstance(layer.bias_initializer, keras.initializers.Zeros)
        assert layer.kernel_regularizer is None
        assert layer.bias_regularizer is None
        assert layer.routing_iterations == 3
        assert layer.use_vertical_routing is True
        assert layer.use_horizontal_routing is True
        assert layer.use_positional_routing is True
        assert layer.epsilon == 1e-8

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        custom_regularizer = keras.regularizers.L2(1e-4)

        layer = CapsuleRoutingSelfAttention(
            num_heads=12,
            key_dim=32,
            value_dim=48,
            dropout_rate=0.1,
            use_bias=False,
            kernel_initializer="he_normal",
            bias_initializer="ones",
            kernel_regularizer=custom_regularizer,
            bias_regularizer=custom_regularizer,
            routing_iterations=5,
            use_vertical_routing=False,
            use_horizontal_routing=True,
            use_positional_routing=False,
            epsilon=1e-6
        )

        # Check custom values
        assert layer.num_heads == 12
        assert layer.key_dim == 32
        assert layer.value_dim == 48
        assert layer.dropout_rate == 0.1
        assert layer.use_bias is False
        assert isinstance(layer.kernel_initializer, keras.initializers.HeNormal)
        assert isinstance(layer.bias_initializer, keras.initializers.Ones)
        assert layer.kernel_regularizer == custom_regularizer
        assert layer.bias_regularizer == custom_regularizer
        assert layer.routing_iterations == 5
        assert layer.use_vertical_routing is False
        assert layer.use_horizontal_routing is True
        assert layer.use_positional_routing is False
        assert layer.epsilon == 1e-6

    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors with exact message matching."""
        # Test negative or zero num_heads
        with pytest.raises(ValueError, match=r"num_heads must be positive, got -8"):
            CapsuleRoutingSelfAttention(num_heads=-8)

        with pytest.raises(ValueError, match=r"num_heads must be positive, got 0"):
            CapsuleRoutingSelfAttention(num_heads=0)

        # Test negative or zero key_dim
        with pytest.raises(ValueError, match=r"key_dim must be positive, got -16"):
            CapsuleRoutingSelfAttention(num_heads=8, key_dim=-16)

        with pytest.raises(ValueError, match=r"key_dim must be positive, got 0"):
            CapsuleRoutingSelfAttention(num_heads=8, key_dim=0)

        # Test negative or zero value_dim
        with pytest.raises(ValueError, match=r"value_dim must be positive, got -32"):
            CapsuleRoutingSelfAttention(num_heads=8, key_dim=16, value_dim=-32)

        with pytest.raises(ValueError, match=r"value_dim must be positive, got 0"):
            CapsuleRoutingSelfAttention(num_heads=8, key_dim=16, value_dim=0)

        # Test invalid dropout rates (matching exact error message pattern)
        with pytest.raises(ValueError, match=r"dropout_rate must be between 0 and 1, got -0\.1"):
            CapsuleRoutingSelfAttention(num_heads=8, dropout_rate=-0.1)

        with pytest.raises(ValueError, match=r"dropout_rate must be between 0 and 1, got 1\.1"):
            CapsuleRoutingSelfAttention(num_heads=8, dropout_rate=1.1)

        # Test negative or zero routing_iterations
        with pytest.raises(ValueError, match=r"routing_iterations must be positive, got -3"):
            CapsuleRoutingSelfAttention(num_heads=8, routing_iterations=-3)

        with pytest.raises(ValueError, match=r"routing_iterations must be positive, got 0"):
            CapsuleRoutingSelfAttention(num_heads=8, routing_iterations=0)

        # Test negative or zero epsilon
        with pytest.raises(ValueError, match=r"epsilon must be positive, got -1e-08"):
            CapsuleRoutingSelfAttention(num_heads=8, epsilon=-1e-8)

        with pytest.raises(ValueError, match=r"epsilon must be positive, got 0"):
            CapsuleRoutingSelfAttention(num_heads=8, epsilon=0)

    def test_build_process(self, input_tensor, layer_instance):
        """Test that the layer builds properly with all sub-components."""
        # Trigger build through forward pass
        output = layer_instance(input_tensor)

        # Check that layer was built
        assert layer_instance.built is True
        assert len(layer_instance.weights) > 0
        assert hasattr(layer_instance, "query_dense")
        assert hasattr(layer_instance, "key_dense")
        assert hasattr(layer_instance, "value_dense")
        assert hasattr(layer_instance, "output_dense")
        assert hasattr(layer_instance, "dropout_layer")

        # Check embedding dimension was set correctly
        assert layer_instance.embed_dim == input_tensor.shape[-1]
        assert layer_instance.actual_key_dim == layer_instance.key_dim
        assert layer_instance.actual_value_dim == layer_instance.key_dim  # defaults to key_dim

        # Check weight shapes for projection layers
        embed_dim = input_tensor.shape[-1]
        num_heads = layer_instance.num_heads
        key_dim = layer_instance.actual_key_dim

        assert layer_instance.query_dense.kernel.shape == (embed_dim, num_heads * key_dim)
        assert layer_instance.key_dense.kernel.shape == (embed_dim, num_heads * key_dim)
        assert layer_instance.value_dense.kernel.shape == (embed_dim, num_heads * key_dim)
        assert layer_instance.output_dense.kernel.shape == (num_heads * key_dim, embed_dim)

    def test_build_with_embed_dim_not_divisible_by_heads(self):
        """Test build fails when embed_dim is not divisible by num_heads."""
        # Create input with embed_dim = 127 (not divisible by 8)
        input_tensor = keras.random.normal([2, 16, 127])
        layer = CapsuleRoutingSelfAttention(num_heads=8)

        # Match exact error message pattern from implementation
        with pytest.raises(ValueError, match=r"embed_dim \(127\) must be divisible by num_heads \(8\)"):
            layer(input_tensor)

    def test_build_with_invalid_input_dimensions(self):
        """Test build fails with invalid input shapes."""
        layer = CapsuleRoutingSelfAttention(num_heads=8, key_dim=16)

        # Test non-3D input
        with pytest.raises(ValueError, match=r"Expected 3D input, got shape"):
            layer.build((None, 128))  # 2D shape

        # Test 4D input
        with pytest.raises(ValueError, match=r"Expected 3D input, got shape"):
            layer.build((None, 16, 16, 128))  # 4D shape

    def test_output_shapes(self, input_tensor):
        """Test that output shapes are computed correctly for various configurations."""
        configs_to_test = [
            {"num_heads": 4, "key_dim": 32},
            {"num_heads": 8, "key_dim": 16, "value_dim": 24},
            {"num_heads": 16, "key_dim": 8, "value_dim": 8},  # Fixed divisibility issue
        ]

        for config in configs_to_test:
            layer = CapsuleRoutingSelfAttention(**config)
            output = layer(input_tensor)

            # Check output shape matches input shape
            assert output.shape == input_tensor.shape

            # Test compute_output_shape separately
            computed_shape = layer.compute_output_shape(input_tensor.shape)
            assert computed_shape == input_tensor.shape

    def test_forward_pass_comprehensive(self, input_tensor, layer_instance):
        """Comprehensive test of forward pass functionality."""
        output = layer_instance(input_tensor)

        # Basic sanity checks
        assert not np.any(np.isnan(ops.convert_to_numpy(output)))
        assert not np.any(np.isinf(ops.convert_to_numpy(output)))

        # Check output shape
        assert output.shape == input_tensor.shape

        # Test with training=False
        output_inference = layer_instance(input_tensor, training=False)
        assert output_inference.shape == input_tensor.shape

        # Test with training=True
        output_training = layer_instance(input_tensor, training=True)
        assert output_training.shape == input_tensor.shape

        # Test deterministic behavior in inference mode
        output_inference_2 = layer_instance(input_tensor, training=False)
        np.testing.assert_allclose(
            ops.convert_to_numpy(output_inference),
            ops.convert_to_numpy(output_inference_2),
            rtol=1e-6, atol=1e-6,
            err_msg="Inference should be deterministic"
        )

    def test_routing_configurations(self):
        """Test layer with different routing configurations systematically."""
        configurations = [
            # Test individual routing mechanisms
            {"num_heads": 8, "use_vertical_routing": True, "use_horizontal_routing": False},
            {"num_heads": 8, "use_vertical_routing": False, "use_horizontal_routing": True},
            {"num_heads": 8, "use_vertical_routing": False, "use_horizontal_routing": False},

            # Test positional routing variations
            {"num_heads": 8, "use_positional_routing": False},
            {"num_heads": 8, "use_positional_routing": True},

            # Test routing iterations
            {"num_heads": 8, "routing_iterations": 1},
            {"num_heads": 8, "routing_iterations": 10},

            # Combined configurations
            {
                "num_heads": 6,
                "use_vertical_routing": True,
                "use_horizontal_routing": True,
                "use_positional_routing": True,
                "routing_iterations": 5
            },
        ]

        for config in configurations:
            layer = CapsuleRoutingSelfAttention(**config)

            # Create test input
            test_input = keras.random.normal([2, 16, 120])  # 120 is divisible by 6

            # Test forward pass
            output = layer(test_input)

            # Check output is valid
            assert not np.any(np.isnan(ops.convert_to_numpy(output)))
            assert output.shape == test_input.shape

    def test_attention_mask_comprehensive(self, layer_instance):
        """Comprehensive test of attention mask functionality."""
        batch_size, seq_len, embed_dim = 2, 16, 128
        input_tensor = keras.random.normal([batch_size, seq_len, embed_dim])

        # Test with 2D mask (padding mask)
        mask_2d = keras.ops.ones((batch_size, seq_len), dtype='bool')
        # Mask out last 4 positions
        mask_2d = keras.ops.concatenate([
            mask_2d[:, :-4],
            keras.ops.zeros((batch_size, 4), dtype='bool')
        ], axis=1)

        output_with_mask = layer_instance(input_tensor, attention_mask=mask_2d)
        assert output_with_mask.shape == input_tensor.shape
        assert not np.any(np.isnan(ops.convert_to_numpy(output_with_mask)))

        # Test with 3D mask (causal mask)
        #
        # DECISION plan-2026-07-30T140922-8af1028f/D-032
        # This causal mask is built from an `ops.arange` index comparison
        # (`row >= col`), NOT from `keras.ops.tril`. Do NOT "simplify" it back to
        # `keras.ops.tril(keras.ops.ones(...))` -- nor to `ops.triu`, which shares
        # the same implementation and therefore the same trap. `ops.tril` routes
        # through a `tf.cond` whose predicate rejects a Python bool once traced,
        # raising `TypeError: pred must not be a Python bool` on every graph path
        # (`tf.function`, `Model.predict`, `jit_compile=True`, `.keras`
        # save/load), for BOTH Python-int and symbolic sizes; re-verified by
        # execution this cycle. `src/` has zero live call sites of either op and
        # this was the last live call anywhere in the repo.
        #
        # The old call here was EAGER and never traced, so it never triggered the
        # trap: this migration is hygiene, not a bug fix. It is done anyway
        # because the repo's standing preference is to REMOVE a raw trap rather
        # than document an accepted exception, and because an eager fixture is
        # exactly the shape that gets copied into a traced path later.
        #
        # The `arange` form is element-wise identical to what `ops.tril` produced
        # (measured on this shape: 0 mismatches out of batch_size*seq_len**2).
        # An `arange` comparison is used rather than `MaskFactory.create_causal_mask`
        # because that helper returns BLOCK polarity (`True` where a position must
        # be suppressed) and this layer wants a KEEP mask; adapting it would have
        # added a third copy of the `logical_not(...)` block->keep adapter, and
        # the repo currently has exactly 2.
        row = keras.ops.arange(seq_len)[:, None]
        col = keras.ops.arange(seq_len)[None, :]
        mask_3d = keras.ops.broadcast_to(
            (row >= col)[None, :, :], (batch_size, seq_len, seq_len)
        )
        # Pin the identity against a NumPy oracle so a future edit cannot drift
        # the mask's meaning while keeping the NaN/shape assertions green.
        expected_causal = np.broadcast_to(
            np.tril(np.ones((seq_len, seq_len), dtype=bool)),
            (batch_size, seq_len, seq_len),
        )
        assert np.array_equal(ops.convert_to_numpy(mask_3d), expected_causal), (
            "the arange-built causal mask must be element-wise identical to the "
            "lower-triangular mask ops.tril used to produce"
        )

        output_with_3d_mask = layer_instance(input_tensor, attention_mask=mask_3d)
        assert output_with_3d_mask.shape == input_tensor.shape
        assert not np.any(np.isnan(ops.convert_to_numpy(output_with_3d_mask)))

        # Test that masking actually affects output
        output_without_mask = layer_instance(input_tensor)

        # Outputs should be different when mask is applied
        assert not np.allclose(
            ops.convert_to_numpy(output_without_mask),
            ops.convert_to_numpy(output_with_mask),
            rtol=1e-5, atol=1e-5
        )

    def test_different_sequence_lengths(self):
        """Test layer with different sequence lengths."""
        seq_lengths = [1, 4, 8, 16, 32, 64]
        layer = CapsuleRoutingSelfAttention(num_heads=8, key_dim=16)

        for seq_len in seq_lengths:
            test_input = keras.random.normal([2, seq_len, 128])

            output = layer(test_input)

            assert output.shape == test_input.shape
            assert not np.any(np.isnan(ops.convert_to_numpy(output)))

    def test_serialization_cycle_comprehensive(self):
        """Comprehensive test of serialization cycle with all parameters."""
        original_layer = CapsuleRoutingSelfAttention(
            num_heads=12,
            key_dim=32,
            value_dim=48,
            dropout_rate=0.1,
            use_bias=True,
            kernel_initializer="he_normal",
            bias_initializer="ones",
            kernel_regularizer=keras.regularizers.L2(1e-4),
            bias_regularizer=keras.regularizers.L1(1e-5),
            routing_iterations=5,
            use_vertical_routing=True,
            use_horizontal_routing=False,
            use_positional_routing=True,
            epsilon=1e-6
        )

        # Get config
        config = original_layer.get_config()

        # Check that all parameters are included in config
        expected_keys = {
            'num_heads', 'key_dim', 'value_dim', 'dropout_rate', 'use_bias',
            'kernel_initializer', 'bias_initializer', 'kernel_regularizer',
            'bias_regularizer', 'routing_iterations', 'use_vertical_routing',
            'use_horizontal_routing', 'use_positional_routing', 'epsilon'
        }

        assert expected_keys.issubset(
            set(config.keys())), f"Missing keys in config: {expected_keys - set(config.keys())}"

        # Recreate the layer
        recreated_layer = CapsuleRoutingSelfAttention.from_config(config)

        # Check configuration matches
        assert recreated_layer.num_heads == original_layer.num_heads
        assert recreated_layer.key_dim == original_layer.key_dim
        assert recreated_layer.value_dim == original_layer.value_dim
        assert recreated_layer.dropout_rate == original_layer.dropout_rate
        assert recreated_layer.use_bias == original_layer.use_bias
        assert recreated_layer.routing_iterations == original_layer.routing_iterations
        assert recreated_layer.use_vertical_routing == original_layer.use_vertical_routing
        assert recreated_layer.use_horizontal_routing == original_layer.use_horizontal_routing
        assert recreated_layer.use_positional_routing == original_layer.use_positional_routing
        assert recreated_layer.epsilon == original_layer.epsilon

    def test_model_integration_complete(self, input_tensor):
        """Complete test of the layer in a model context with compilation and training."""
        # Create a model with the capsule attention layer
        inputs = keras.Input(shape=input_tensor.shape[1:])
        x = CapsuleRoutingSelfAttention(
            num_heads=8,
            key_dim=16,
            routing_iterations=3,
            name="capsule_attention"
        )(inputs)
        x = keras.layers.LayerNormalization()(x)
        x = keras.layers.Dense(64, activation='relu')(x)
        x = keras.layers.GlobalAveragePooling1D()(x)
        outputs = keras.layers.Dense(10, activation='softmax')(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Compile the model
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        # Test forward pass
        y_pred = model(input_tensor, training=False)
        assert y_pred.shape == (input_tensor.shape[0], 10)

        # Test with dummy labels for training
        labels = np.random.randint(0, 10, size=(input_tensor.shape[0],))
        loss = model.test_on_batch(input_tensor, labels)
        assert not np.isnan(loss[0])  # loss should be finite

        # Test model summary doesn't crash
        model.summary()

    def test_model_save_load_cycle_comprehensive(self, input_tensor):
        """CRITICAL TEST: Complete model save/load serialization cycle with predictions."""
        # Create a model with the capsule attention layer
        inputs = keras.Input(shape=input_tensor.shape[1:])
        x = CapsuleRoutingSelfAttention(
            num_heads=8,
            key_dim=16,
            routing_iterations=3,
            use_vertical_routing=True,
            use_horizontal_routing=True,
            dropout_rate=0.1,
            name="capsule_attention"
        )(inputs)
        x = keras.layers.LayerNormalization()(x)
        x = keras.layers.Dense(64, activation='relu')(x)
        x = keras.layers.GlobalAveragePooling1D()(x)
        outputs = keras.layers.Dense(10, activation='softmax')(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Generate a prediction before saving
        original_prediction = model.predict(input_tensor, verbose=0)

        # Create temporary directory for model
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "model.keras")

            # Save the model
            model.save(model_path)

            # Load the model
            loaded_model = keras.models.load_model(model_path)

            # Generate prediction with loaded model
            loaded_prediction = loaded_model.predict(input_tensor, verbose=0)

            # Check predictions match (this is the critical test)
            np.testing.assert_allclose(
                original_prediction,
                loaded_prediction,
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions should match after serialization"
            )

            # Check layer type is preserved
            capsule_layer = loaded_model.get_layer("capsule_attention")
            assert isinstance(capsule_layer, CapsuleRoutingSelfAttention)

            # Check layer configuration is preserved
            assert capsule_layer.num_heads == 8
            assert capsule_layer.key_dim == 16
            assert capsule_layer.routing_iterations == 3
            assert capsule_layer.use_vertical_routing is True
            assert capsule_layer.use_horizontal_routing is True

    def test_numerical_stability_extreme_cases(self):
        """Test layer stability with extreme input values and configurations."""
        layer = CapsuleRoutingSelfAttention(num_heads=4, key_dim=16, epsilon=1e-8)

        # Create inputs with different magnitudes
        batch_size = 2
        seq_len = 16
        embed_dim = 64

        test_cases = [
            ("zeros", keras.ops.zeros((batch_size, seq_len, embed_dim))),
            ("tiny_values", keras.ops.ones((batch_size, seq_len, embed_dim)) * 1e-10),
            ("large_values", keras.ops.ones((batch_size, seq_len, embed_dim)) * 1e3),
            ("large_random", keras.random.normal((batch_size, seq_len, embed_dim)) * 100),
            ("mixed_scale", keras.ops.concatenate([
                keras.ops.ones((batch_size, seq_len // 2, embed_dim)) * 1e-6,
                keras.ops.ones((batch_size, seq_len // 2, embed_dim)) * 1e6
            ], axis=1))
        ]

        for case_name, test_input in test_cases:
            output = layer(test_input)

            # Check for NaN/Inf values
            output_numpy = ops.convert_to_numpy(output)
            assert not np.any(np.isnan(output_numpy)), f"NaN values detected in output for {case_name}"
            assert not np.any(np.isinf(output_numpy)), f"Inf values detected in output for {case_name}"

    def test_regularization_losses_comprehensive(self, input_tensor):
        """Test that regularization losses are properly applied and accessible."""
        # Create layer with regularization
        layer = CapsuleRoutingSelfAttention(
            num_heads=8,
            key_dim=16,
            kernel_regularizer=keras.regularizers.L2(0.1),
            bias_regularizer=keras.regularizers.L1(0.1)
        )

        # Build layer and call it
        _ = layer(input_tensor)

        # Check that regularization losses have been added
        assert len(layer.losses) > 0

        # Test that losses are finite
        total_loss = sum(layer.losses)
        assert not np.isnan(ops.convert_to_numpy(total_loss))
        assert not np.isinf(ops.convert_to_numpy(total_loss))

    def test_squash_function_properties_detailed(self):
        """Detailed test of the squashing function properties."""
        layer = CapsuleRoutingSelfAttention(num_heads=4, key_dim=16)

        # Test squash function with various input scales
        test_cases = [
            keras.random.normal([2, 3, 4]),  # Standard normal
            keras.ops.ones([2, 3, 4]) * 0.1,  # Small vectors
            keras.ops.ones([2, 3, 4]) * 10.0,  # Large vectors
            keras.ops.zeros([2, 3, 4]),  # Zero vectors
        ]

        for test_vectors in test_cases:
            squashed = layer._squash(test_vectors)

            # Check output shape is preserved
            assert squashed.shape == test_vectors.shape

            # Check no NaN/Inf values
            squashed_numpy = ops.convert_to_numpy(squashed)
            assert not np.any(np.isnan(squashed_numpy))
            assert not np.any(np.isinf(squashed_numpy))

            # Check squashing properties
            original_norms = ops.sqrt(ops.sum(ops.square(test_vectors), axis=-1))
            squashed_norms = ops.sqrt(ops.sum(ops.square(squashed), axis=-1))

            # For non-zero vectors, squashed norm should be <= 1
            non_zero_mask = original_norms > 1e-6
            if ops.any(non_zero_mask):
                masked_squashed_norms = ops.where(non_zero_mask, squashed_norms, 0.0)
                assert ops.all(masked_squashed_norms <= 1.0 + 1e-6)  # Allow small numerical error

    def test_dynamic_routing_algorithm_properties(self):
        """Test properties of the dynamic routing algorithm."""
        layer = CapsuleRoutingSelfAttention(
            num_heads=4,
            key_dim=16,
            routing_iterations=5
        )

        test_input = keras.random.normal([2, 8, 64])

        # Test consistency across multiple calls (with dropout disabled)
        outputs = []
        for _ in range(3):
            output = layer(test_input, training=False)
            outputs.append(ops.convert_to_numpy(output))

        # Check that outputs are consistent (deterministic routing)
        for i in range(1, len(outputs)):
            np.testing.assert_allclose(
                outputs[0],
                outputs[i],
                rtol=1e-5, atol=1e-5,
                err_msg="Dynamic routing should be deterministic in inference mode"
            )

    def test_routing_iterations_effect(self):
        """Test that different routing iterations produce different results."""
        test_input = keras.random.normal([2, 8, 64])

        # Create layers with different routing iterations
        layer_1_iter = CapsuleRoutingSelfAttention(
            num_heads=4, key_dim=16, routing_iterations=1
        )
        layer_5_iter = CapsuleRoutingSelfAttention(
            num_heads=4, key_dim=16, routing_iterations=5
        )

        output_1 = layer_1_iter(test_input, training=False)
        output_5 = layer_5_iter(test_input, training=False)

        # Outputs should be different (routing should converge differently)
        assert not np.allclose(
            ops.convert_to_numpy(output_1),
            ops.convert_to_numpy(output_5),
            rtol=1e-3, atol=1e-3
        )

    def test_routing_disabled_fallback_comprehensive(self):
        """Test that layer works correctly when routing methods are disabled."""
        configurations = [
            # No routing at all
            {"use_vertical_routing": False, "use_horizontal_routing": False},
            # Only vertical routing
            {"use_vertical_routing": True, "use_horizontal_routing": False},
            # Only horizontal routing
            {"use_vertical_routing": False, "use_horizontal_routing": True},
        ]

        test_input = keras.random.normal([2, 16, 128])

        for config in configurations:
            layer = CapsuleRoutingSelfAttention(num_heads=8, key_dim=16, **config)
            output = layer(test_input)

            # Should still work and produce valid output
            assert output.shape == test_input.shape
            output_numpy = ops.convert_to_numpy(output)
            assert not np.any(np.isnan(output_numpy))
            assert not np.any(np.isinf(output_numpy))

    def test_different_key_value_dims_comprehensive(self):
        """Test layer with different key and value dimensions extensively."""
        test_configs = [
            {"key_dim": 32, "value_dim": 16},
            {"key_dim": 16, "value_dim": 32},
            {"key_dim": 64, "value_dim": 8},
        ]

        for config in test_configs:
            layer = CapsuleRoutingSelfAttention(num_heads=8, **config)
            test_input = keras.random.normal([2, 16, 128])
            output = layer(test_input)

            # Output should still match input shape
            assert output.shape == test_input.shape
            assert not np.any(np.isnan(ops.convert_to_numpy(output)))

            # Check internal dimensions are set correctly
            assert layer.actual_key_dim == config["key_dim"]
            assert layer.actual_value_dim == config["value_dim"]

    def test_gradients_flow_comprehensive(self, input_tensor):
        """Comprehensive test that gradients flow properly through all components."""
        layer = CapsuleRoutingSelfAttention(
            num_heads=8,
            key_dim=16,
            use_vertical_routing=True,
            use_horizontal_routing=True
        )

        with tf.GradientTape() as tape:
            input_tensor_tf = tf.convert_to_tensor(input_tensor)
            tape.watch(input_tensor_tf)
            output = layer(input_tensor_tf)
            loss = keras.ops.mean(keras.ops.square(output))

        gradients = tape.gradient(loss, layer.trainable_variables)

        # Check that gradients exist and are finite
        assert all(g is not None for g in gradients), "Some gradients are None"
        assert len(gradients) > 0, "No trainable variables found"

        for i, grad in enumerate(gradients):
            grad_numpy = grad.numpy()
            assert not np.any(np.isnan(grad_numpy)), f"NaN gradients detected in variable {i}"
            assert not np.any(np.isinf(grad_numpy)), f"Infinite gradients detected in variable {i}"

            # Check that gradients have reasonable magnitude
            grad_norm = np.linalg.norm(grad_numpy)
            assert grad_norm < 1e6, f"Gradient norm too large: {grad_norm}"

    def test_different_batch_sizes_comprehensive(self):
        """Test layer with comprehensive range of batch sizes."""
        layer = CapsuleRoutingSelfAttention(num_heads=8, key_dim=16)

        batch_sizes = [1, 2, 4, 8, 16, 32]
        seq_len, embed_dim = 16, 128

        for batch_size in batch_sizes:
            test_input = keras.random.normal([batch_size, seq_len, embed_dim])
            output = layer(test_input)

            assert output.shape == test_input.shape
            output_numpy = ops.convert_to_numpy(output)
            assert not np.any(np.isnan(output_numpy))

            # Test that output has reasonable scale
            output_std = np.std(output_numpy)
            assert 0.001 < output_std < 100, f"Output std {output_std} seems unreasonable"

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes_comprehensive(self, input_tensor, layer_instance, training):
        """Comprehensive test of behavior in different training modes."""
        output = layer_instance(input_tensor, training=training)

        # Should always produce valid output
        assert output.shape == input_tensor.shape
        output_numpy = ops.convert_to_numpy(output)
        assert not np.any(np.isnan(output_numpy))
        assert not np.any(np.isinf(output_numpy))

        # Test that training mode affects dropout behavior
        if hasattr(layer_instance, 'dropout_rate') and layer_instance.dropout_rate > 0:
            if training is True:
                # In training mode with dropout, outputs should vary slightly
                output2 = layer_instance(input_tensor, training=True)
                # Note: Due to routing determinism, difference might be small
                # We just check that it doesn't crash
                assert output2.shape == input_tensor.shape
            elif training is False:
                # In inference mode, outputs should be deterministic
                output2 = layer_instance(input_tensor, training=False)
                np.testing.assert_allclose(
                    output_numpy,
                    ops.convert_to_numpy(output2),
                    rtol=1e-6, atol=1e-6,
                    err_msg="Inference mode should be deterministic"
                )

    def test_capsule_routing_weight_initialization(self):
        """Test that capsule routing weights are properly initialized."""
        layer = CapsuleRoutingSelfAttention(
            num_heads=8,
            key_dim=16,
            use_vertical_routing=True
        )

        # Build the layer
        test_input = keras.random.normal([2, 16, 128])
        _ = layer(test_input)

        # Check vertical routing weights exist when enabled
        assert hasattr(layer, 'vertical_aggregation_weights')
        assert layer.vertical_aggregation_weights is not None

        # Check weight shapes
        assert layer.vertical_aggregation_weights.shape == (8, 8)  # (num_heads, num_heads)

    def test_positional_routing_behavior(self):
        """Test that positional routing behaves differently from non-positional."""
        test_input = keras.random.normal([2, 8, 64])

        layer_with_pos = CapsuleRoutingSelfAttention(
            num_heads=4,
            key_dim=16,
            use_vertical_routing=False,  # Focus on horizontal routing
            use_horizontal_routing=True,
            use_positional_routing=True
        )

        layer_without_pos = CapsuleRoutingSelfAttention(
            num_heads=4,
            key_dim=16,
            use_vertical_routing=False,  # Focus on horizontal routing
            use_horizontal_routing=True,
            use_positional_routing=False
        )

        output_with_pos = layer_with_pos(test_input, training=False)
        output_without_pos = layer_without_pos(test_input, training=False)

        # Outputs should be different due to different routing constraints
        assert not np.allclose(
            ops.convert_to_numpy(output_with_pos),
            ops.convert_to_numpy(output_without_pos),
            rtol=1e-3, atol=1e-3
        )

    def test_epsilon_parameter_numerical_effect(self):
        """Test that epsilon parameter affects numerical stability appropriately."""
        # Test with different epsilon values
        epsilons = [1e-12, 1e-8, 1e-4]

        # Create input that might cause numerical issues (very small values)
        test_input = keras.ops.ones([2, 8, 64]) * 1e-10

        for eps in epsilons:
            layer = CapsuleRoutingSelfAttention(
                num_heads=4,
                key_dim=16,
                epsilon=eps
            )

            output = layer(test_input)

            # Should handle small values gracefully
            output_numpy = ops.convert_to_numpy(output)
            assert not np.any(np.isnan(output_numpy)), f"NaN values with epsilon={eps}"
            assert not np.any(np.isinf(output_numpy)), f"Inf values with epsilon={eps}"

    def test_graph_mode_positional_routing_symbolic_seqlen_raises(self):
        """Locks A1: a tf.function trace with UNKNOWN seq-len must fail LOUD.

        This is the exact bug shape. Pre-fix, `_horizontal_routing` did
        `seq_len = ops.shape(...)[2]` then `for l in range(seq_len)`; tracing
        with an unknown-N input signature made `range(symbolic_tensor)` raise a
        cryptic `TypeError("'SymbolicTensor' object cannot be interpreted as an
        integer")`. Post-fix the static `.shape[2]` is None under a symbolic
        signature, so we raise a clear `ValueError` (documented static-N
        contract) BEFORE the loop. We assert the message, which guarantees the
        new code path (not the old TypeError) is what fires.
        """
        layer = CapsuleRoutingSelfAttention(
            num_heads=4,
            key_dim=16,
            use_vertical_routing=False,
            use_horizontal_routing=True,
            use_positional_routing=True,  # the DEFAULT — the crashing path
        )
        # Build eagerly with a concrete seq_len so the Dense layers exist;
        # the symbolic trace below exercises only the routing loop.
        _ = layer(keras.random.normal([2, 8, 64]))

        @tf.function(input_signature=[tf.TensorSpec([None, None, 64], tf.float32)])
        def fwd(t):
            return layer(t, training=False)

        with pytest.raises(ValueError, match=r"statically-known sequence length"):
            fwd(tf.convert_to_tensor(keras.random.normal([2, 8, 64])))

    def test_graph_mode_positional_routing_concrete_seqlen(self):
        """Locks A1: a tf.function trace with a CONCRETE seq-len must trace and
        match eager output (the static `.shape[2]` unrolls the loop).

        Tolerance note: eager and graph (XLA) kernels differ at the ~1e-6 level
        on GPU; the unseeded input previously made the worst-case delta straddle
        atol=1e-6, so this test flaked only under full-suite GPU memory pressure
        (passed in isolation). The seed makes the input deterministic and the
        1e-5 tolerance reflects the real eager-vs-graph float delta — a broken
        loop unroll would diverge by orders of magnitude, not 1e-6, so the A1
        invariant is still fully locked.
        """
        batch_size, seq_len, embed_dim = 2, 8, 64
        test_input = keras.random.normal(
            [batch_size, seq_len, embed_dim], seed=1337
        )

        layer = CapsuleRoutingSelfAttention(
            num_heads=4,
            key_dim=16,
            use_vertical_routing=False,
            use_horizontal_routing=True,
            use_positional_routing=True,
        )

        # Eager reference (also builds the layer with a concrete seq_len).
        eager_output = ops.convert_to_numpy(layer(test_input, training=False))

        # Concrete-signature tf.function: static shape known at trace time.
        @tf.function(input_signature=[tf.TensorSpec([None, seq_len, embed_dim], tf.float32)])
        def graph_forward(x):
            return layer(x, training=False)

        graph_output = ops.convert_to_numpy(graph_forward(test_input))

        assert graph_output.shape == (batch_size, seq_len, embed_dim)
        np.testing.assert_allclose(
            eager_output,
            graph_output,
            rtol=1e-5, atol=1e-5,
            err_msg="Graph-mode positional routing must match eager output",
        )

    def test_graph_mode_positional_routing_in_compiled_model(self):
        """Locks A1 via the compiled-model path (predict() traces a tf.function)."""
        batch_size, seq_len, embed_dim = 2, 8, 64
        test_input = keras.random.normal([batch_size, seq_len, embed_dim])

        inputs = keras.Input(shape=(seq_len, embed_dim))
        x = CapsuleRoutingSelfAttention(
            num_heads=4,
            key_dim=16,
            use_vertical_routing=False,
            use_horizontal_routing=True,
            use_positional_routing=True,
            name="capsule_pos",
        )(inputs)
        model = keras.Model(inputs=inputs, outputs=x)

        # model.predict runs the forward inside a traced tf.function — pre-fix
        # this crashed with a TypeError from range(symbolic).
        preds = model.predict(test_input, verbose=0)
        assert preds.shape == (batch_size, seq_len, embed_dim)
        assert not np.any(np.isnan(preds))

    def test_keras_roundtrip_positional_routing(self):
        """Locks A2: .keras save/load of a positional-routing model reloads weights.

        The build() idempotency guard ensures a second build() (on reload /
        functional reuse) does not re-create and discard the four Dense
        projections. Reloaded forward must match the original bit-for-bit.
        """
        batch_size, seq_len, embed_dim = 2, 8, 64
        test_input = keras.random.normal([batch_size, seq_len, embed_dim])

        inputs = keras.Input(shape=(seq_len, embed_dim))
        x = CapsuleRoutingSelfAttention(
            num_heads=4,
            key_dim=16,
            use_vertical_routing=True,
            use_horizontal_routing=True,
            use_positional_routing=True,  # explicitly exercise the A1/A2 path
            name="capsule_pos",
        )(inputs)
        model = keras.Model(inputs=inputs, outputs=x)

        original_prediction = model.predict(test_input, verbose=0)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "model.keras")
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)
            loaded_prediction = loaded_model.predict(test_input, verbose=0)

            np.testing.assert_allclose(
                original_prediction,
                loaded_prediction,
                rtol=1e-6, atol=1e-6,
                err_msg="Positional-routing predictions must match after .keras reload",
            )

            capsule_layer = loaded_model.get_layer("capsule_pos")
            assert isinstance(capsule_layer, CapsuleRoutingSelfAttention)
            assert capsule_layer.use_positional_routing is True

    def test_build_idempotent_preserves_weights(self):
        """Locks A2 directly: a second build() must NOT replace the Dense layers."""
        layer = CapsuleRoutingSelfAttention(num_heads=4, key_dim=16)
        test_input = keras.random.normal([2, 8, 64])
        _ = layer(test_input)

        # Capture the Dense object identities after first build.
        q_id = id(layer.query_dense)
        k_id = id(layer.key_dense)
        v_id = id(layer.value_dense)
        o_id = id(layer.output_dense)

        # Re-invoking build() must be a no-op (guard) — same objects, no
        # weight discard.
        layer.build(test_input.shape)

        assert id(layer.query_dense) == q_id
        assert id(layer.key_dense) == k_id
        assert id(layer.value_dense) == v_id
        assert id(layer.output_dense) == o_id

# ---------------------------------------------------------------------
# Mixed-precision mask tests (plan-2026-07-27-b4ef45f0, step 3)
# ---------------------------------------------------------------------
#
# WHAT IS BEING GUARDED HERE, and why it is a DIFFERENT mechanism from the
# `0 * -inf = NaN` family fixed at the other nine mask sites in this package.
#
# `_apply_attention_mask` already used the structurally-safe `ops.where` form:
# there is no `(1 - keep) * -1e9` product anywhere, so no unmasked position can
# ever become `0 * -inf`. What it did NOT survive is a FULLY-MASKED QUERY ROW.
# `ops.cast(-1e9, 'float16')` is `-inf` (np.float16(-1e9) == -inf), so a row whose
# mask is all-False becomes all-`-inf`, and `softmax(all -inf)` is `0/0 = NaN`.
# In float32 the same row is finite: `-1e9` is representable, every entry is equal,
# and the softmax returns a uniform — meaningless, but finite. Garbage in, garbage
# out rather than garbage in, whole-tensor NaN out.
#
# MEASURED on unfixed HEAD (B=2, N=32, D=64, H=4, key_dim=16, one fully-masked
# query row), GPU 1 / TF 2.18:
#
#     policy           degenerate mask      padding mask     causal mask
#     float32              0/4096 NaN         0/4096           0/4096
#     mixed_float16      128/4096 NaN         0/4096           0/4096
#
# 128 = 2 batches x 1 query row x 64 features: the NaN is confined to the
# degenerate row's own output (the earlier in-source note's "384/384, the whole
# batch" figure came from an entirely-masked input, not a single bad row). It is
# still fatal in training — one NaN row NaNs every gradient.
#
# WHY A DTYPE-ONLY FIX CANNOT WORK HERE (assumption A4, MEASURED not argued).
# The softmax at this site is `self.attn_prob_attention`, a Keras LAYER, and Keras
# autocasting is on: probed under `mixed_float16`, a float32 tensor handed to
# `attn_prob_attention.__call__` is seen INSIDE its `call()` as float16, and a
# fully-masked float32 `-1e9` row fed straight to it still returns 8/8 NaN. So
# keeping the biased logits in `mask_dtype(...)` — which is what fixes
# `rpc_attention.py` — is silently undone at this layer boundary.
# `TestCapsuleRoutingMaskHazardIsReal::test_the_probability_sublayer_autocasts_a_
# float32_input` pins that measurement as an executable assertion, so the
# justification for the predicate-level rescue cannot rot.
#
# Anti-vacuity note on sizes. The reduction-size trap (`plans/LESSONS.md`: `N = 7`
# once hid an fp16 `-inf` that only appeared at `N >= 512`) does not transfer: the
# hazard here is a per-ELEMENT dtype overflow of a constant followed by a
# degenerate softmax ROW, neither of which needs a long reduction to appear. It is
# nevertheless asserted reachable rather than assumed — see
# `TestCapsuleRoutingMaskHazardIsReal`, which checks that the policy really selects
# float16 compute, that `float16(MASK_BIAS_VALUE)` really is `-inf`, and that the
# degenerate mask really contains exactly one all-False query row while the other
# two masks contain none.

from dl_techniques.layers.attention.common import MASK_BIAS_VALUE

_MP_B, _MP_N, _MP_D, _MP_H, _MP_KD = 2, 32, 64, 4, 16
_MP_DEG_ROW = 5                  # the query row the degenerate mask blanks entirely
_MP_KEEP = _MP_N // 2            # first half kept, second half masked (padding mask)
_MP_SEED = 1234

# Absolute tolerance for "this policy's forward agrees with the float32 control".
#
# These are NOT an fp16 error budget. MEASURED on unmodified HEAD (i.e. BEFORE the
# degenerate-row fix, so they describe the layer, not the change), with
# byte-identical weights, max |policy - float32| against an output absmax of ~7:
#
#     mask          mixed_float16      float64      float32 (re-run)
#     degenerate       0.01473          0.008319          0.0
#     padding          0.01232          0.005866          0.0
#     causal           0.00981          0.007013          0.0
#
# THE FLOAT64 COLUMN IS A TF32 MEASUREMENT, and that is not a footnote — it is a
# factor of ~1500. The numbers above were taken running THIS FILE ALONE, where
# TensorFloat-32 tensor-core matmul is enabled (the Ampere+ default, ~1e-3 relative
# precision for a float32 matmul). Running the whole `tests/test_layers/
# test_attention/` directory, the SAME measurement USED TO give 3e-06 / 5.5e-06 /
# 6e-06, because `test_linear_attention.py` called
# `tf.config.experimental.enable_tensor_float_32_execution(False)` AT IMPORT TIME —
# process-globally, for the rest of the session, for every test file collected
# alongside it. That leak is GONE: all four such disables are now scoped to their own
# modules by the `tf32_disabled` fixture in `tests/test_layers/conftest.py`, so the
# directory-scoped run now sees the same ambient TF32 state as the file-scoped one.
# The historical divergence is kept here because it is what these tolerances are sized
# against, and because `test_the_float32_float64_divergence_justifies_the_tolerance`
# below still reads the LIVE flag rather than assuming either regime.
# So this layer is ~0.1% "dtype-sensitive" under TF32 and ~1e-6
# dtype-sensitive in true fp32; the routing loop (three iterations of `_squash`,
# which divides by `sqrt(squared_norm + epsilon)` with `epsilon = 1e-8`, added back
# onto the logits before a softmax) amplifies whatever the matmul gave it.
#
# The tolerances must therefore hold in BOTH regimes, so they are set from the
# WORSE (TF32-on) one. `TestCapsuleRoutingConditioning` pins the justification and
# is itself TF32-aware, so it cannot be satisfied by silently widening a number.
# The load-bearing guards for this step are finiteness of the degenerate row and
# the polarity tests, not this comparison. `float32` compares against a control
# computed the same way and is exact (measured 0.0 in both regimes), so its entry
# stays at 1e-6.
_MP_ATOL = {"float32": 1e-6, "mixed_float16": 0.05, "float64": 0.05}


def _mp_input():
    """Deterministic ``(B, N, D)`` float32 input, shared by every test below."""
    return np.random.default_rng(7).standard_normal(
        (_MP_B, _MP_N, _MP_D)
    ).astype("float32")


def _mp_mask(kind):
    """One of the boolean masks these tests need, as a numpy ``bool`` array.

    ``'degenerate'`` is a rank-3 ``(B, N, N)`` mask that keeps everything EXCEPT
    query row ``_MP_DEG_ROW``, which is masked entirely — the fully-masked row this
    step exists to rescue. ``'padding'`` is a rank-2 ``(B, N)`` key-axis mask
    (exercises the layer's rank-2 expand branch) and ``'causal'`` is a rank-3
    lower-triangular mask; NEITHER of those two has a fully-masked row, so they are
    the no-regression controls — the fix must leave them alone.
    """
    if kind == "degenerate":
        m = np.ones((_MP_B, _MP_N, _MP_N), dtype=bool)
        m[:, _MP_DEG_ROW, :] = False
        return m
    if kind == "padding":
        m = np.ones((_MP_B, _MP_N), dtype=bool)
        m[:, _MP_KEEP:] = False
        return m
    if kind == "causal":
        return np.broadcast_to(
            np.tril(np.ones((_MP_N, _MP_N), dtype=bool)), (_MP_B, _MP_N, _MP_N)
        ).copy()
    raise ValueError(f"unknown mask kind {kind!r}")


def _mp_layer(**kwargs):
    """A built layer whose weights are byte-identical under every dtype policy.

    Seeding the initializers is NOT sufficient: a ``glorot_uniform`` draw under a
    ``float64`` policy differs from the same-seed draw under ``float32`` (the
    initializer samples in the VARIABLE dtype), so a float64-vs-float32 comparison
    on seeded-but-not-set weights measures the initializer, not the code under
    test. Explicit float32 arrays are assigned instead.
    """
    layer = CapsuleRoutingSelfAttention(num_heads=_MP_H, key_dim=_MP_KD, **kwargs)
    layer.build((_MP_B, _MP_N, _MP_D))
    rng = np.random.default_rng(_MP_SEED)
    layer.set_weights(
        [(rng.standard_normal(w.shape) * 0.2).astype("float32") for w in layer.weights]
    )
    return layer


_F32_REFERENCE = {}


def _float32_reference(kind):
    """Masked float32 output for ``kind``, memoized, under an explicit policy.

    This is the CONTROL every mixed-precision assertion compares against. It sets
    and restores the policy itself, so it is valid whichever parametrization of
    ``dtype_policy`` happens to reach it first.
    """
    if kind not in _F32_REFERENCE:
        previous = keras.mixed_precision.global_policy().name
        keras.mixed_precision.set_global_policy("float32")
        try:
            layer = _mp_layer()
            out = layer(
                ops.convert_to_tensor(_mp_input()),
                attention_mask=ops.convert_to_tensor(_mp_mask(kind)),
            )
            _F32_REFERENCE[kind] = ops.convert_to_numpy(out).astype("float32")
        finally:
            keras.mixed_precision.set_global_policy(previous)
    return _F32_REFERENCE[kind]


def _numpy(tensor):
    return ops.convert_to_numpy(tensor).astype("float32")


class TestCapsuleRoutingMaskHazardIsReal:
    """Anti-vacuity. If these stop holding, every fp16 test below is worthless."""

    def test_policy_really_selects_float16_compute(self, dtype_policy):
        expected = {
            "float32": "float32",
            "mixed_float16": "float16",
            "float64": "float64",
        }[dtype_policy]
        assert keras.mixed_precision.global_policy().compute_dtype == expected

    def test_mask_bias_value_overflows_in_the_compute_dtype(self):
        with np.errstate(over="ignore"):
            assert np.isneginf(np.float16(MASK_BIAS_VALUE)), (
                "anti-vacuity FAILED: float16(MASK_BIAS_VALUE) is not -inf, so the "
                "degenerate-row hazard this module guards is not reproducible here."
            )

    def test_the_degenerate_mask_really_has_exactly_one_fully_masked_row(self):
        degenerate = _mp_mask("degenerate")
        empty_rows = (~degenerate).all(axis=-1)
        assert int(empty_rows.sum()) == _MP_B, (
            f"expected exactly one fully-masked query row per batch element, got "
            f"{int(empty_rows.sum())} across {_MP_B} batch elements"
        )
        assert empty_rows[:, _MP_DEG_ROW].all()

    def test_the_control_masks_have_no_fully_masked_row(self):
        for kind in ("padding", "causal"):
            mask = _mp_mask(kind)
            if mask.ndim == 2:
                mask = mask[:, None, :]
            assert not (~mask).all(axis=-1).any(), (
                f"the {kind!r} control mask contains a fully-masked row, so it no "
                "longer isolates the no-regression case from the degenerate one"
            )
            assert (~mask).sum() > 0, (
                f"the {kind!r} control mask masks nothing; it cannot detect a "
                "regression in the masking code"
            )

    def test_the_probability_sublayer_autocasts_a_float32_input(self, dtype_policy):
        """Assumption A4, as an executable assertion rather than a claim.

        The whole reason this site needs a rescue IN THE PREDICATE — rather than
        the dtype-only fix that works at ``rpc_attention.py`` — is that the softmax
        here lives inside a Keras layer with autocasting enabled, which drags a
        carefully-promoted float32 tensor straight back down to float16. If Keras
        ever stops doing that, this test fails and the rescue can be revisited.
        """
        layer = _mp_layer()
        prob = layer.attn_prob_attention
        assert getattr(prob, "autocast", False) is True

        seen = {}
        original = prob.call

        def spy(x, *args, **kwargs):
            seen["dtype"] = keras.backend.standardize_dtype(x.dtype)
            return original(x, *args, **kwargs)

        prob.call = spy
        try:
            prob(ops.convert_to_tensor(
                np.zeros((1, _MP_H, 4, 4), dtype="float32")
            ))
        finally:
            prob.call = original

        expected = keras.mixed_precision.global_policy().compute_dtype
        assert seen["dtype"] == expected, (
            f"a float32 tensor entering `attn_prob_attention` was seen inside its "
            f"call() as {seen['dtype']!r}, not the compute dtype {expected!r}"
        )


class TestCapsuleRoutingDegenerateRow:
    """SC4: a FULLY-masked query row must stay finite under ``mixed_float16``."""

    def test_fully_masked_row_is_finite_and_matches_float32(self, dtype_policy):
        layer = _mp_layer()
        out = _numpy(
            layer(
                ops.convert_to_tensor(_mp_input()),
                attention_mask=ops.convert_to_tensor(_mp_mask("degenerate")),
            )
        )

        n_bad = int((~np.isfinite(out)).sum())
        assert n_bad == 0, (
            f"{n_bad}/{out.size} non-finite output entries for a mask with one "
            f"fully-masked query row under policy {dtype_policy!r}"
        )
        # The row itself, not merely the tensor total: a whole-tensor count could be
        # satisfied by a fix that happened to zero the row instead of computing it.
        assert np.isfinite(out[:, _MP_DEG_ROW]).all(), (
            f"the fully-masked query row {_MP_DEG_ROW} is not finite under policy "
            f"{dtype_policy!r}"
        )
        assert float(np.abs(out[:, _MP_DEG_ROW]).max()) > 0.0, (
            "the degenerate row is finite but identically zero — that is a different "
            "convention from the float32 behavior this criterion asks for"
        )

        reference = _float32_reference("degenerate")
        atol = _MP_ATOL[dtype_policy]
        max_dev = float(np.abs(out - reference).max())
        assert max_dev <= atol, (
            f"degenerate-row forward under {dtype_policy!r} deviates from the "
            f"float32 control by {max_dev:.4g} > {atol:.4g}"
        )
        assert float(np.abs(out).max()) > 0.5 * float(np.abs(reference).max()), (
            f"output absmax {np.abs(out).max():.4g} collapsed relative to the "
            f"float32 control {np.abs(reference).max():.4g}"
        )


class TestCapsuleRoutingPartialMaskNoRegression:
    """The masks that were ALREADY fine must stay bit-for-bit fine.

    The degenerate-row rescue changes the keep predicate, so it could in principle
    alter rows that were never degenerate. These two masks have no fully-masked row
    (asserted above), so the rescue must be completely inert for them.
    """

    @pytest.mark.parametrize("kind", ["padding", "causal"])
    def test_partial_mask_is_finite_and_matches_float32(self, dtype_policy, kind):
        layer = _mp_layer()
        out = _numpy(
            layer(
                ops.convert_to_tensor(_mp_input()),
                attention_mask=ops.convert_to_tensor(_mp_mask(kind)),
            )
        )

        n_bad = int((~np.isfinite(out)).sum())
        assert n_bad == 0, (
            f"{n_bad}/{out.size} non-finite output entries for a {kind!r} mask under "
            f"policy {dtype_policy!r}"
        )

        reference = _float32_reference(kind)
        atol = _MP_ATOL[dtype_policy]
        max_dev = float(np.abs(out - reference).max())
        assert max_dev <= atol, (
            f"{kind!r} mask under {dtype_policy!r} deviates from the float32 control "
            f"by {max_dev:.4g} > {atol:.4g}"
        )


class TestCapsuleRoutingConditioning:
    """Pins the JUSTIFICATION for the loose entries in :data:`_MP_ATOL`.

    This test manages its own policies (it needs two) and touches no fp16. It
    exists so the masked-path tolerances above cannot silently become either
    unnecessary (someone tightens the routing loop's conditioning) or insufficient
    (it gets worse): it re-measures the float32-vs-float64 divergence of the same
    forward pass and asserts the tolerance brackets it.

    It is TF32-AWARE, because the quantity it measures is not a property of this
    layer alone. `tf.config.experimental.enable_tensor_float_32_execution(False)`
    is a PROCESS-GLOBAL switch that `test_linear_attention.py` used to flip at
    import time, so the identical measurement read 0.0083 when this file ran alone
    (TF32 on, the GPU default) and 6e-06 when the attention directory ran as a
    session (TF32 off). A single hard-coded lower bound here would therefore pass in
    isolation and fail in the gate — which is exactly what happened when it was
    first written.

    That leak has since been scoped away (`tf32_disabled` in
    `tests/test_layers/conftest.py`), so both runs now see the same ambient state —
    but the branch below is KEPT and still reads the live flag. It is what makes
    this test correct under a future toggle, on CPU (where TF32 is inert), and under
    any wider pytest invocation that sets the flag itself; a test that hard-codes the
    regime it expects is the defect this one exists to avoid.
    """

    @pytest.mark.parametrize("kind", ["degenerate", "padding", "causal"])
    def test_the_float32_float64_divergence_justifies_the_tolerance(self, kind):
        def forward(policy):
            previous = keras.mixed_precision.global_policy().name
            keras.mixed_precision.set_global_policy(policy)
            try:
                layer = _mp_layer()
                return ops.convert_to_numpy(
                    layer(
                        ops.convert_to_tensor(_mp_input()),
                        attention_mask=ops.convert_to_tensor(_mp_mask(kind)),
                    )
                ).astype("float64")
            finally:
                keras.mixed_precision.set_global_policy(previous)

        divergence = float(np.abs(forward("float32") - forward("float64")).max())
        budget = _MP_ATOL["float64"]

        assert divergence <= budget, (
            f"the float32-vs-float64 divergence of a {kind!r}-masked forward is "
            f"{divergence:.4g}, which EXCEEDS the tolerance {budget:.4g} the "
            "agreement tests rely on; re-derive the tolerances"
        )
        # DECISION plan-2026-07-30T140922-8af1028f/D-031
        # The FLAG is not the REGIME: TF32 is a tensor-core path, so the flag being
        # enabled changes nothing without an Ampere+ GPU present. Reading the flag
        # alone made this test RED on CPU at floor=1e-4 (measured: 3.1e-06 - 6.4e-06,
        # true-fp32 numbers under an "enabled" flag). It only ever passed on CPU
        # because `test_linear_attention.py`'s import-time process-global disable
        # leaked the flag to False for whole-directory runs -- i.e. this test was
        # itself a consumer of the collection-order coupling that leak created, and
        # it was ALREADY RED when this file was run alone on CPU, before that leak
        # was scoped away. Requiring a GPU here restores the intended meaning on both
        # devices and leaves the GPU behaviour (flag on + GPU present -> 1e-4)
        # unchanged. plan-2026-07-30T140922-8af1028f D-031.
        tf32_in_effect = bool(
            tf.config.experimental.tensor_float_32_execution_enabled()
            and tf.config.list_physical_devices("GPU")
        )
        if tf32_in_effect:
            # TF32 matmul (~1e-3 relative). Measured 0.0059 - 0.0083 across the
            # three masks; this is the regime `_MP_ATOL['float64']` is sized for.
            floor = 1e-4
            regime = "TF32-enabled"
        else:
            # True fp32 matmul. Measured 2.9e-06 - 6.0e-06. The floor is only
            # asserting that the two policies really did compute something
            # different — a divergence of exactly 0.0 would mean the float64
            # policy never took effect and this test is measuring nothing.
            floor = 1e-7
            regime = "true-fp32 (TF32 disabled, or enabled with no GPU present)"
        assert divergence > floor, (
            f"a {kind!r}-masked forward is now better conditioned than expected in "
            f"the {regime} regime ({divergence:.4g} <= {floor:.4g}); either the "
            f"float64 policy is not taking effect, or the loose {budget:.4g} float64 "
            "tolerance is no longer justified and must be tightened"
        )


class TestCapsuleRoutingMaskPolarity:
    """SC6: the mask must suppress the MASKED positions, not the kept ones.

    A polarity inversion at this site — passing ``~mask`` where ``mask`` is meant —
    raises nothing, changes no shape and leaves the output perfectly finite. Only
    an influence test can see it. MEASURED on unmodified HEAD by handing the layer
    ``~mask``: perturbing a "masked" token then moves the kept query rows by 27.0
    (no-routing config) / 30.0 (default config) instead of 0.0 / 0.48 — so the
    assertions below have a 156x / 63x margin against a real inversion.

    The two tests differ in ONE thing, deliberately:

    *   ``..._with_routing_disabled`` is the exact statement: with both routing
        paths off, ``routing_output`` IS the raw logits, so masking key column ``p``
        removes token ``p``'s influence on every kept query row exactly. Measured
        0.0 under all three policies.
    *   ``..._in_the_default_routing_config`` is the shipped configuration, where
        the same statement can only hold approximately — and that residual is a
        property of capsule routing, NOT of the mask. Both routing paths run BEFORE
        the mask is applied (``call()``: routing is computed from the unmasked
        logits and added to them), and ``_squash`` normalizes over the KEY axis, so
        a masked column contributes to the routed value of every kept column.
        Measured 0.48 against a kept-token influence of 27.9.
    """

    @staticmethod
    def _influence(layer, mask):
        base_input = _mp_input()
        perturbed_masked = base_input.copy()
        perturbed_masked[:, _MP_KEEP + 3, :] += 5.0      # a MASKED token
        perturbed_kept = base_input.copy()
        perturbed_kept[:, 3, :] += 5.0                   # a KEPT token

        mask_tensor = ops.convert_to_tensor(mask)

        def forward(array):
            return ops.convert_to_numpy(
                layer(ops.convert_to_tensor(array), attention_mask=mask_tensor)
            ).astype("float64")

        rows = slice(0, _MP_KEEP)
        base = forward(base_input)
        assert np.isfinite(base[:, rows]).all(), (
            "the kept query rows are not finite; the comparison below would be "
            "meaningless"
        )
        delta_masked = float(np.abs(forward(perturbed_masked)[:, rows] - base[:, rows]).max())
        delta_kept = float(np.abs(forward(perturbed_kept)[:, rows] - base[:, rows]).max())
        return delta_masked, delta_kept

    @staticmethod
    def _padding_mask():
        mask = np.ones((_MP_B, _MP_N), dtype=bool)
        mask[:, _MP_KEEP:] = False
        return mask

    def test_a_masked_token_has_no_influence_with_routing_disabled(self, dtype_policy):
        layer = _mp_layer(
            use_vertical_routing=False, use_horizontal_routing=False
        )
        delta_masked, delta_kept = self._influence(layer, self._padding_mask())

        # Measured EXACTLY 0.0 under all three policies. The 1e-3 budget is
        # session-noise headroom (see `test_rpc_attention.py`, where a batched op
        # measured 0.0 in isolation and 1.1e-06 inside the full suite); it still
        # sits four orders of magnitude below the 27.0 signal.
        assert delta_masked <= 1e-3, (
            f"perturbing a MASKED token changed the kept query rows by "
            f"{delta_masked:.6g} under policy {dtype_policy!r} — with routing "
            "disabled this must be exact, so the mask polarity is INVERTED (the "
            "layer is attending to the padding)"
        )
        assert delta_kept > 1.0, (
            f"perturbing a KEPT token changed the output by only {delta_kept:.6g}; "
            "the test is vacuous — the layer is ignoring its input"
        )

    def test_a_masked_token_barely_influences_the_default_routing_config(
        self, dtype_policy
    ):
        layer = _mp_layer()
        delta_masked, delta_kept = self._influence(layer, self._padding_mask())

        assert delta_masked <= 1.0, (
            f"perturbing a MASKED token changed the kept query rows by "
            f"{delta_masked:.6g} under policy {dtype_policy!r} (measured 0.48 on "
            "correct code, 30.0 with an INVERTED mask)"
        )
        assert delta_kept > 20.0 * delta_masked, (
            f"masked influence {delta_masked:.6g} is not decisively smaller than "
            f"kept influence {delta_kept:.6g}"
        )


# ==============================================================================
# Step 7 (b) — the static-`seq_len` guard fires only where the static length is
# actually needed.
#
# `plan-2026-07-27T183600-b4ef45f0` D-014. The guard used to sit OUTSIDE the
# `if self.use_positional_routing:` branch, so it rejected a dynamic sequence
# length for EVERY `use_horizontal_routing=True` layer — including the
# non-positional path, which is pure transpose / expand_dims / repeat and reads
# `seq_len` nowhere. The message even advised "set use_positional_routing=False",
# advice the guard's own placement made useless.
#
# PROVEN RED on the unfixed code (`04caa0e7`):
# `test_dynamic_seq_len_with_non_positional_horizontal_routing_succeeds` failed
# with `ValueError: CapsuleRoutingSelfAttention positional routing
# (use_positional_routing=True) requires a statically-known sequence length; got
# None. ...` raised from `_horizontal_routing` — i.e. the layer refused an input
# it can compute, quoting a flag that was NOT set.
#
# This WIDENS the accepted input set. The other half of the pair — the positional
# case, which genuinely needs the static length — must still raise, and the
# pre-existing
# `test_graph_mode_positional_routing_symbolic_seqlen_raises` (which matches
# r"statically-known sequence length") must stay green; the reworded message keeps
# that phrase verbatim for exactly that reason.
# ==============================================================================


class TestCapsuleRoutingStaticSeqLenGuardPlacement:
    """D-014: the guard belongs to positional routing, not to horizontal routing."""

    @staticmethod
    def _layer(positional):
        return CapsuleRoutingSelfAttention(
            num_heads=4,
            key_dim=16,
            use_vertical_routing=False,
            use_horizontal_routing=True,
            use_positional_routing=positional,
        )

    @staticmethod
    def _traced(layer):
        """A `tf.function` whose sequence dimension is statically UNKNOWN."""

        @tf.function(input_signature=[tf.TensorSpec([None, None, 64], tf.float32)])
        def fwd(t):
            return layer(t, training=False)

        return fwd

    def test_dynamic_seq_len_with_non_positional_horizontal_routing_succeeds(self):
        """THE widening. Raises on unfixed code; must run and produce real numbers."""
        layer = self._layer(positional=False)
        concrete = keras.random.normal([2, 8, 64], seed=99)
        _ = layer(concrete)  # build eagerly so the Dense sub-layers exist

        out = self._traced(layer)(tf.convert_to_tensor(concrete))
        out = ops.convert_to_numpy(out)

        assert out.shape == (2, 8, 64)
        assert np.isfinite(out).all(), (
            "the newly-accepted dynamic-length path produced non-finite output; "
            "widening the contract must not ship a broken path"
        )
        # ANTI-VACUITY: a layer that silently returned its input, or zeros, would
        # satisfy the shape+finite assertions above.
        assert float(np.abs(out).max()) > 1e-3

    def test_the_dynamic_trace_really_has_an_unknown_sequence_length(self):
        """ANTI-VACUITY for the test above: if the trace kept a static length, it
        would exercise the ordinary path and prove nothing about the guard."""
        seen = {}
        layer = self._layer(positional=False)
        _ = layer(keras.random.normal([2, 8, 64], seed=99))

        original = layer._horizontal_routing

        def spy(attention_weights):
            seen["static_len"] = attention_weights.shape[2]
            return original(attention_weights)

        layer._horizontal_routing = spy
        try:
            self._traced(layer)(tf.convert_to_tensor(
                keras.random.normal([2, 8, 64], seed=99)
            ))
        finally:
            layer._horizontal_routing = original

        assert seen.get("static_len", "missing") is None, (
            "`_horizontal_routing` saw a STATIC sequence length "
            f"({seen.get('static_len')!r}) under the symbolic trace, so the guard "
            "under test was never reachable"
        )

    def test_dynamic_seq_len_with_positional_routing_still_raises(self):
        """The half that must NOT change: positional routing unrolls
        `for l in range(seq_len)` and genuinely cannot run without a static N."""
        layer = self._layer(positional=True)
        _ = layer(keras.random.normal([2, 8, 64], seed=99))

        with pytest.raises(ValueError, match=r"statically-known sequence length"):
            self._traced(layer)(tf.convert_to_tensor(
                keras.random.normal([2, 8, 64], seed=99)
            ))

    def test_the_message_names_the_flags_that_actually_gate_it(self):
        """The original message advised a flag the guard's placement ignored.

        `plans/LESSONS.md` I:3 — grep the target test file's existing
        `pytest.raises(match=...)` strings before choosing message text. The only
        pre-existing matcher on this raise is
        r"statically-known sequence length" (used by
        `test_graph_mode_positional_routing_symbolic_seqlen_raises` above), which
        the reworded text preserves verbatim.
        """
        layer = self._layer(positional=True)
        _ = layer(keras.random.normal([2, 8, 64], seed=99))

        with pytest.raises(ValueError) as excinfo:
            self._traced(layer)(tf.convert_to_tensor(
                keras.random.normal([2, 8, 64], seed=99)
            ))
        message = str(excinfo.value)

        assert "use_horizontal_routing=True" in message, (
            "the diagnostic does not name `use_horizontal_routing`, the flag that "
            "puts the caller on this code path at all"
        )
        assert "use_positional_routing=True" in message, (
            "the diagnostic does not name `use_positional_routing`, the flag that "
            "actually gates the guard"
        )
        assert "set use_positional_routing=False" in message, (
            "the diagnostic no longer offers the remedy that now genuinely works"
        )
        assert "statically-known sequence length" in message
