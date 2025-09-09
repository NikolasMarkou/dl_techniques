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
        mask_3d = keras.ops.ones((batch_size, seq_len, seq_len), dtype='bool')
        # Create causal mask
        mask_3d = keras.ops.tril(mask_3d)

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