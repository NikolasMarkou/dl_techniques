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
    """Test suite for CapsuleRoutingSelfAttention layer implementation."""

    @pytest.fixture
    def input_tensor(self):
        """Create a test input tensor."""
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
        assert layer.dropout == 0.0
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
            dropout=0.1,
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
        assert layer.dropout == 0.1
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
        """Test that invalid parameters raise appropriate errors."""
        # Test negative or zero num_heads
        with pytest.raises(ValueError, match="num_heads must be positive"):
            CapsuleRoutingSelfAttention(num_heads=-8)

        with pytest.raises(ValueError, match="num_heads must be positive"):
            CapsuleRoutingSelfAttention(num_heads=0)

        # Test negative or zero key_dim
        with pytest.raises(ValueError, match="key_dim must be positive"):
            CapsuleRoutingSelfAttention(num_heads=8, key_dim=-16)

        with pytest.raises(ValueError, match="key_dim must be positive"):
            CapsuleRoutingSelfAttention(num_heads=8, key_dim=0)

        # Test negative or zero value_dim
        with pytest.raises(ValueError, match="value_dim must be positive"):
            CapsuleRoutingSelfAttention(num_heads=8, key_dim=16, value_dim=-32)

        with pytest.raises(ValueError, match="value_dim must be positive"):
            CapsuleRoutingSelfAttention(num_heads=8, key_dim=16, value_dim=0)

        # Test invalid dropout rates
        with pytest.raises(ValueError, match="dropout must be between 0 and 1"):
            CapsuleRoutingSelfAttention(num_heads=8, dropout=-0.1)

        with pytest.raises(ValueError, match="dropout must be between 0 and 1"):
            CapsuleRoutingSelfAttention(num_heads=8, dropout=1.1)

        # Test negative or zero routing_iterations
        with pytest.raises(ValueError, match="routing_iterations must be positive"):
            CapsuleRoutingSelfAttention(num_heads=8, routing_iterations=-3)

        with pytest.raises(ValueError, match="routing_iterations must be positive"):
            CapsuleRoutingSelfAttention(num_heads=8, routing_iterations=0)

        # Test negative or zero epsilon
        with pytest.raises(ValueError, match="epsilon must be positive"):
            CapsuleRoutingSelfAttention(num_heads=8, epsilon=-1e-8)

        with pytest.raises(ValueError, match="epsilon must be positive"):
            CapsuleRoutingSelfAttention(num_heads=8, epsilon=0)

    def test_build_process(self, input_tensor, layer_instance):
        """Test that the layer builds properly."""
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

        # Check embedding dimension was set
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

        with pytest.raises(ValueError, match="embed_dim .* must be divisible by num_heads"):
            layer(input_tensor)

    def test_output_shapes(self, input_tensor):
        """Test that output shapes are computed correctly."""
        configs_to_test = [
            {"num_heads": 4, "key_dim": 32},
            {"num_heads": 8, "key_dim": 16, "value_dim": 24},
            # Corrected this config: 128 is not divisible by 12. Use 16 instead.
            {"num_heads": 16, "key_dim": 8, "value_dim": 8},
        ]

        for config in configs_to_test:
            # Use a custom input tensor if embed_dim doesn't match fixture
            current_embed_dim = input_tensor.shape[-1]
            if current_embed_dim % config["num_heads"] != 0:
                # This case is avoided by correcting the config above but is good practice
                continue

            layer = CapsuleRoutingSelfAttention(**config)
            output = layer(input_tensor)

            # Check output shape matches input shape
            assert output.shape == input_tensor.shape

            # Test compute_output_shape separately
            computed_shape = layer.compute_output_shape(input_tensor.shape)
            assert computed_shape == input_tensor.shape

    def test_forward_pass(self, input_tensor, layer_instance):
        """Test that forward pass produces expected values."""
        output = layer_instance(input_tensor)

        # Basic sanity checks
        assert not np.any(np.isnan(output.numpy()))
        assert not np.any(np.isinf(output.numpy()))

        # Check output shape
        assert output.shape == input_tensor.shape

        # Test with training=False
        output_inference = layer_instance(input_tensor, training=False)
        assert output_inference.shape == input_tensor.shape

        # Test with training=True
        output_training = layer_instance(input_tensor, training=True)
        assert output_training.shape == input_tensor.shape

    def test_different_routing_configurations(self):
        """Test layer with different routing configurations."""
        configurations = [
            {"num_heads": 8, "use_vertical_routing": True, "use_horizontal_routing": False},
            {"num_heads": 8, "use_vertical_routing": False, "use_horizontal_routing": True},
            {"num_heads": 8, "use_vertical_routing": False, "use_horizontal_routing": False},
            {"num_heads": 8, "use_positional_routing": False},
            {"num_heads": 8, "routing_iterations": 1},
            {"num_heads": 8, "routing_iterations": 10},
        ]

        for config in configurations:
            layer = CapsuleRoutingSelfAttention(**config)

            # Create test input
            test_input = keras.random.normal([2, 16, 128])

            # Test forward pass
            output = layer(test_input)

            # Check output is valid
            assert not np.any(np.isnan(output.numpy()))
            assert output.shape == test_input.shape

    def test_attention_mask_support(self, layer_instance):
        """Test attention mask functionality."""
        batch_size, seq_len, embed_dim = 2, 16, 128
        input_tensor = keras.random.normal([batch_size, seq_len, embed_dim])

        # Test with 2D mask (padding mask)
        mask_2d = keras.ops.ones((batch_size, seq_len))
        # Mask out last 4 positions
        mask_2d = keras.ops.concatenate([
            mask_2d[:, :-4],
            keras.ops.zeros((batch_size, 4))
        ], axis=1)

        output_with_mask = layer_instance(input_tensor, attention_mask=mask_2d)
        assert output_with_mask.shape == input_tensor.shape
        assert not np.any(np.isnan(output_with_mask.numpy()))

        # Test with 3D mask (attention mask)
        mask_3d = keras.ops.ones((batch_size, seq_len, seq_len))
        # Create causal mask
        mask_3d = keras.ops.tril(mask_3d)

        output_with_3d_mask = layer_instance(input_tensor, attention_mask=mask_3d)
        assert output_with_3d_mask.shape == input_tensor.shape
        assert not np.any(np.isnan(output_with_3d_mask.numpy()))

    def test_different_sequence_lengths(self):
        """Test layer with different sequence lengths."""
        seq_lengths = [4, 8, 16, 32, 64]
        layer = CapsuleRoutingSelfAttention(num_heads=8, key_dim=16)

        for seq_len in seq_lengths:
            test_input = keras.random.normal([2, seq_len, 128])

            output = layer(test_input)

            assert output.shape == test_input.shape
            assert not np.any(np.isnan(output.numpy()))

    def test_serialization_cycle(self):
        """Test complete serialization cycle."""
        original_layer = CapsuleRoutingSelfAttention(
            num_heads=12,
            key_dim=32,
            value_dim=48,
            dropout=0.1,
            use_bias=True,
            kernel_initializer="he_normal",
            routing_iterations=5,
            use_vertical_routing=True,
            use_horizontal_routing=False,
            use_positional_routing=True,
            epsilon=1e-6
        )

        # Get config
        config = original_layer.get_config()

        # Recreate the layer
        recreated_layer = CapsuleRoutingSelfAttention.from_config(config)

        # Check configuration matches
        assert recreated_layer.num_heads == original_layer.num_heads
        assert recreated_layer.key_dim == original_layer.key_dim
        assert recreated_layer.value_dim == original_layer.value_dim
        assert recreated_layer.dropout == original_layer.dropout
        assert recreated_layer.use_bias == original_layer.use_bias
        assert recreated_layer.routing_iterations == original_layer.routing_iterations
        assert recreated_layer.use_vertical_routing == original_layer.use_vertical_routing
        assert recreated_layer.use_horizontal_routing == original_layer.use_horizontal_routing
        assert recreated_layer.use_positional_routing == original_layer.use_positional_routing
        assert recreated_layer.epsilon == original_layer.epsilon

    def test_model_integration(self, input_tensor):
        """Test the layer in a model context."""
        # Create a simple model with the capsule attention layer
        inputs = keras.Input(shape=input_tensor.shape[1:])
        x = CapsuleRoutingSelfAttention(
            num_heads=8,
            key_dim=16,
            routing_iterations=3
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
        # FIX: Use numpy to generate random integers, which is more stable in tests.
        labels = np.random.randint(0, 10, size=(input_tensor.shape[0],))
        loss = model.test_on_batch(input_tensor, labels)
        assert not np.isnan(loss[0])  # loss should be finite

    def test_model_save_load_cycle(self, input_tensor):
        """CRITICAL TEST: Full model save/load serialization cycle."""
        # Create a model with the capsule attention layer
        inputs = keras.Input(shape=input_tensor.shape[1:])
        x = CapsuleRoutingSelfAttention(
            num_heads=8,
            key_dim=16,
            routing_iterations=3,
            use_vertical_routing=True,
            use_horizontal_routing=True,
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

    def test_numerical_stability(self):
        """Test layer stability with extreme input values."""
        layer = CapsuleRoutingSelfAttention(num_heads=4, key_dim=16)

        # Create inputs with different magnitudes
        batch_size = 2
        seq_len = 16
        embed_dim = 64

        test_cases = [
            keras.ops.zeros((batch_size, seq_len, embed_dim)),  # Zeros
            keras.ops.ones((batch_size, seq_len, embed_dim)) * 1e-10,  # Very small values
            keras.ops.ones((batch_size, seq_len, embed_dim)) * 1e3,  # Large values
            keras.random.normal((batch_size, seq_len, embed_dim)) * 100  # Large random values
        ]

        for test_input in test_cases:
            output = layer(test_input)

            # Check for NaN/Inf values
            assert not np.any(np.isnan(output.numpy())), "NaN values detected in output"
            assert not np.any(np.isinf(output.numpy())), "Inf values detected in output"

    def test_regularization_losses(self, input_tensor):
        """Test that regularization losses are properly applied."""
        # Create layer with regularization
        layer = CapsuleRoutingSelfAttention(
            num_heads=8,
            key_dim=16,
            kernel_regularizer=keras.regularizers.L2(0.1),
            bias_regularizer=keras.regularizers.L1(0.1)
        )

        # Build layer and call it
        _ = layer(input_tensor)

        # Check that regularization losses have been added.
        # In Keras 3, losses are typically added during build.
        assert len(layer.losses) > 0

    def test_squash_function_properties(self):
        """Test properties of the squashing function."""
        layer = CapsuleRoutingSelfAttention(num_heads=4, key_dim=16)

        # Test squash function directly
        # Create test vectors
        test_vectors = keras.random.normal([2, 3, 4])  # arbitrary shape
        squashed = layer._squash(test_vectors)

        # Check output shape is preserved
        assert squashed.shape == test_vectors.shape

        # Check no NaN/Inf values
        assert not np.any(np.isnan(squashed.numpy()))
        assert not np.any(np.isinf(squashed.numpy()))

        # Check squashing properties: ||squashed|| should be less than ||original|| for large vectors
        original_norms = keras.ops.sqrt(keras.ops.sum(keras.ops.square(test_vectors), axis=-1))
        squashed_norms = keras.ops.sqrt(keras.ops.sum(keras.ops.square(squashed), axis=-1))

        # For vectors with norm > 1, squashed norm should be smaller
        large_norm_mask = original_norms > 1.0
        if keras.ops.any(large_norm_mask):
            are_squashed_norms_smaller = squashed_norms < original_norms
            final_check = ops.logical_or(ops.logical_not(large_norm_mask), are_squashed_norms_smaller)
            assert ops.all(final_check)

    def test_dynamic_routing_convergence(self):
        """Test that dynamic routing produces consistent results."""
        layer = CapsuleRoutingSelfAttention(
            num_heads=4,
            key_dim=16,
            routing_iterations=10  # More iterations for stability
        )

        test_input = keras.random.normal([2, 8, 64])

        # Run multiple times and check consistency
        outputs = []
        for _ in range(3):
            output = layer(test_input, training=False)  # Disable dropout for consistency
            outputs.append(output.numpy())

        # Check that outputs are very similar (deterministic routing)
        for i in range(1, len(outputs)):
            np.testing.assert_allclose(
                outputs[0],
                outputs[i],
                rtol=1e-5, atol=1e-5,
                err_msg="Dynamic routing should be deterministic in inference mode"
            )

    def test_routing_disabled_fallback(self):
        """Test that layer works when both routing methods are disabled."""
        layer = CapsuleRoutingSelfAttention(
            num_heads=8,
            key_dim=16,
            use_vertical_routing=False,
            use_horizontal_routing=False
        )

        test_input = keras.random.normal([2, 16, 128])
        output = layer(test_input)

        # Should still work and produce valid output
        assert output.shape == test_input.shape
        assert not np.any(np.isnan(output.numpy()))
        assert not np.any(np.isinf(output.numpy()))

    def test_different_key_value_dims(self):
        """Test layer with different key and value dimensions."""
        layer = CapsuleRoutingSelfAttention(
            num_heads=8,
            key_dim=32,
            value_dim=16  # Different from key_dim
        )

        test_input = keras.random.normal([2, 16, 128])
        output = layer(test_input)

        # Output should still match input shape
        assert output.shape == test_input.shape
        assert not np.any(np.isnan(output.numpy()))

        # Check internal dimensions are set correctly
        assert layer.actual_key_dim == 32
        assert layer.actual_value_dim == 16

    def test_gradients_flow(self, input_tensor):
        """Test that gradients flow properly through the layer."""
        layer = CapsuleRoutingSelfAttention(num_heads=8, key_dim=16)

        with tf.GradientTape() as tape:
            # KerasTensors need to be converted to TF tensors for gradient tape
            input_tensor_tf = tf.convert_to_tensor(input_tensor)
            tape.watch(input_tensor_tf)
            output = layer(input_tensor_tf)
            loss = keras.ops.mean(keras.ops.square(output))

        gradients = tape.gradient(loss, layer.trainable_variables)

        # Check that gradients exist and are finite
        assert all(g is not None for g in gradients)
        assert len(gradients) > 0

        for grad in gradients:
            assert not np.any(np.isnan(grad.numpy())), "NaN gradients detected"
            assert not np.any(np.isinf(grad.numpy())), "Infinite gradients detected"

    def test_different_batch_sizes(self):
        """Test layer with different batch sizes."""
        layer = CapsuleRoutingSelfAttention(num_heads=8, key_dim=16)

        batch_sizes = [1, 4, 8, 16]
        seq_len, embed_dim = 16, 128

        for batch_size in batch_sizes:
            test_input = keras.random.normal([batch_size, seq_len, embed_dim])
            output = layer(test_input)

            assert output.shape == test_input.shape
            assert not np.any(np.isnan(output.numpy()))

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, input_tensor, layer_instance, training):
        """Test behavior in different training modes."""
        output = layer_instance(input_tensor, training=training)

        # Should always produce valid output
        assert output.shape == input_tensor.shape
        assert not np.any(np.isnan(output.numpy()))
        assert not np.any(np.isinf(output.numpy()))

    def test_invalid_input_shapes(self):
        """Test that invalid input shapes raise appropriate errors."""
        layer = CapsuleRoutingSelfAttention(num_heads=8, key_dim=16)

        # Test 2D input (missing sequence dimension)
        with pytest.raises(ValueError, match="Expected 3D input"):
            layer.build((None, 128))  # Should be (None, seq_len, embed_dim)

        # Test 4D input (too many dimensions)
        with pytest.raises(ValueError, match="Expected 3D input"):
            layer.build((None, 16, 16, 128))  # Should be 3D

    def test_very_small_sequence_length(self):
        """Test layer with very small sequence lengths."""
        layer = CapsuleRoutingSelfAttention(num_heads=4, key_dim=16)

        # Test with sequence length of 1
        test_input = keras.random.normal([2, 1, 64])
        output = layer(test_input)

        assert output.shape == test_input.shape
        assert not np.any(np.isnan(output.numpy()))

    def test_epsilon_parameter_effect(self):
        """Test that epsilon parameter affects numerical stability."""
        # Test with different epsilon values
        epsilons = [1e-12, 1e-8, 1e-4]

        for eps in epsilons:
            layer = CapsuleRoutingSelfAttention(
                num_heads=4,
                key_dim=16,
                epsilon=eps
            )

            # Create input that might cause numerical issues
            test_input = keras.ops.ones([2, 8, 64]) * 1e-10
            output = layer(test_input)

            # Should handle small values gracefully
            assert not np.any(np.isnan(output.numpy()))
            assert not np.any(np.isinf(output.numpy()))