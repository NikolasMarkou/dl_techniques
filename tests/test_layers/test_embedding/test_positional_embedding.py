import pytest
import numpy as np
import tensorflow as tf
import keras
import tempfile
import os


from dl_techniques.layers.embedding.positional_embedding import PositionalEmbedding


class TestPositionalEmbedding:
    """Comprehensive test suite for PositionalEmbedding layer."""

    @pytest.fixture
    def input_tensor(self) -> tf.Tensor:
        """Create a test input tensor."""
        return tf.random.normal([4, 32, 128])  # batch_size=4, seq_len=32, dim=128

    @pytest.fixture
    def layer_instance(self) -> PositionalEmbedding:
        """Create a default layer instance for testing."""
        return PositionalEmbedding(max_seq_len=64, dim=128)

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        layer = PositionalEmbedding(max_seq_len=512, dim=256)

        # Check default values
        assert layer.max_seq_len == 512
        assert layer.dim == 256
        assert layer.dropout_rate == 0.0
        assert layer.scale == 0.02
        assert isinstance(layer.pos_initializer, keras.initializers.TruncatedNormal)
        assert layer.pos_embedding is None

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        custom_initializer = keras.initializers.GlorotUniform()

        layer = PositionalEmbedding(
            max_seq_len=256,
            dim=512,
            dropout=0.1,
            pos_initializer=custom_initializer,
            scale=0.01
        )

        # Check custom values
        assert layer.max_seq_len == 256
        assert layer.dim == 512
        assert layer.dropout_rate == 0.1
        assert layer.scale == 0.01
        assert isinstance(layer.pos_initializer, keras.initializers.GlorotUniform)

    def test_initialization_string_initializer(self):
        """Test initialization with string initializer."""
        layer = PositionalEmbedding(
            max_seq_len=128,
            dim=64,
            pos_initializer="glorot_uniform"
        )

        assert isinstance(layer.pos_initializer, keras.initializers.GlorotUniform)

    def test_initialization_truncated_normal_with_scale(self):
        """Test that truncated normal initializer uses custom scale."""
        layer = PositionalEmbedding(
            max_seq_len=128,
            dim=64,
            scale=0.05
        )

        assert isinstance(layer.pos_initializer, keras.initializers.TruncatedNormal)
        assert layer.pos_initializer.stddev == 0.05

    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        with pytest.raises(ValueError, match="max_seq_len must be positive"):
            PositionalEmbedding(max_seq_len=0, dim=128)

        with pytest.raises(ValueError, match="dim must be positive"):
            PositionalEmbedding(max_seq_len=128, dim=-1)

        with pytest.raises(ValueError, match="dropout must be in"):
            PositionalEmbedding(max_seq_len=128, dim=64, dropout=1.5)

        with pytest.raises(ValueError, match="scale must be positive"):
            PositionalEmbedding(max_seq_len=128, dim=64, scale=-0.1)

    def test_build_process(self, input_tensor: tf.Tensor):
        """Test that the layer builds properly."""
        layer = PositionalEmbedding(max_seq_len=64, dim=128)

        # Build by calling the layer
        output = layer(input_tensor)

        # Check that layer was built
        assert layer.built is True
        assert layer.pos_embedding is not None
        assert layer.dropout is not None
        assert layer.pos_embedding.shape == (1, 64, 128)

    def test_build_invalid_input_shape(self):
        """Test build with invalid input shapes."""
        layer = PositionalEmbedding(max_seq_len=64, dim=128)

        # 2D input should fail
        with pytest.raises(ValueError, match="Input must be 3D"):
            layer.build((32, 128))

        # Wrong dimension should fail
        with pytest.raises(ValueError, match="Input dimension .* does not match expected dim"):
            layer.build((4, 32, 64))  # dim=64 but expected 128

    def test_output_shapes(self, input_tensor: tf.Tensor):
        """Test that output shapes are computed correctly."""
        test_configs = [
            (64, 128),
            (128, 256),
            (256, 512),
        ]

        for max_seq_len, dim in test_configs:
            layer = PositionalEmbedding(max_seq_len=max_seq_len, dim=dim)

            # Create appropriate input
            test_input = tf.random.normal([2, min(32, max_seq_len), dim])
            output = layer(test_input)

            # Check output shape matches input shape
            assert output.shape == test_input.shape

            # Test compute_output_shape separately
            computed_shape = layer.compute_output_shape(test_input.shape)
            assert computed_shape == test_input.shape

    def test_forward_pass_basic(self, input_tensor: tf.Tensor):
        """Test basic forward pass functionality."""
        layer = PositionalEmbedding(max_seq_len=64, dim=128, dropout=0.0)
        output = layer(input_tensor, training=False)

        # Basic sanity checks
        assert output.shape == input_tensor.shape
        assert not np.any(np.isnan(output.numpy()))
        assert not np.any(np.isinf(output.numpy()))

    def test_forward_pass_deterministic(self):
        """Test forward pass with deterministic inputs."""
        layer = PositionalEmbedding(
            max_seq_len=8,
            dim=4,
            dropout=0.0,
            pos_initializer="zeros"
        )

        # Input of ones
        test_input = tf.ones([1, 4, 4])
        output = layer(test_input, training=False)

        # With zero initialization, output should equal input
        assert np.allclose(output.numpy(), test_input.numpy())

    def test_positional_embedding_addition(self):
        """Test that positional embeddings are correctly added."""
        layer = PositionalEmbedding(
            max_seq_len=4,
            dim=3,
            dropout=0.0,
            pos_initializer="ones"
        )

        # Build the layer
        layer.build((None, 4, 3))

        # Zero input
        zero_input = tf.zeros([1, 4, 3])
        output = layer(zero_input, training=False)

        # Output should be the positional embeddings (all ones)
        expected = tf.ones([1, 4, 3])
        assert np.allclose(output.numpy(), expected.numpy())

    def test_sequence_length_variation(self):
        """Test with different sequence lengths."""
        layer = PositionalEmbedding(max_seq_len=32, dim=64, dropout=0.0)

        # Test with different sequence lengths
        for seq_len in [8, 16, 24, 32]:
            test_input = tf.random.normal([2, seq_len, 64])
            output = layer(test_input, training=False)

            assert output.shape == test_input.shape
            assert not np.any(np.isnan(output.numpy()))

    def test_dropout_behavior(self):
        """Test dropout behavior during training and inference."""
        layer = PositionalEmbedding(max_seq_len=16, dim=32, dropout=0.5)
        test_input = tf.random.normal([4, 8, 32])

        # Training mode - dropout should be applied
        output_train = layer(test_input, training=True)
        assert output_train.shape == test_input.shape

        # Inference mode - no dropout
        output_inference = layer(test_input, training=False)
        assert output_inference.shape == test_input.shape

        # With high dropout, training and inference outputs should differ
        # This test is probabilistic but should pass most of the time
        assert not np.allclose(output_train.numpy(), output_inference.numpy(), atol=1e-5)

    def test_serialization(self):
        """Test layer serialization and deserialization."""
        original_layer = PositionalEmbedding(
            max_seq_len=128,
            dim=64,
            dropout=0.1,
            pos_initializer="glorot_uniform",
            scale=0.05
        )

        # Build the layer
        input_shape = (None, 32, 64)
        original_layer.build(input_shape)

        # Test data
        test_input = tf.random.normal([2, 32, 64])
        original_output = original_layer(test_input, training=False)

        # Get configs
        config = original_layer.get_config()
        build_config = original_layer.get_build_config()

        # Recreate the layer
        recreated_layer = PositionalEmbedding.from_config(config)
        recreated_layer.build_from_config(build_config)

        # Copy weights to ensure identical output
        recreated_layer.pos_embedding.assign(original_layer.pos_embedding)

        # Test configuration matches
        assert recreated_layer.max_seq_len == original_layer.max_seq_len
        assert recreated_layer.dim == original_layer.dim
        assert recreated_layer.dropout_rate == original_layer.dropout_rate
        assert recreated_layer.scale == original_layer.scale

        # Test output matches
        recreated_output = recreated_layer(test_input, training=False)
        assert np.allclose(original_output.numpy(), recreated_output.numpy())

    def test_model_integration(self, input_tensor: tf.Tensor):
        """Test the layer in a model context."""
        # Create a simple model with the positional embedding layer
        inputs = keras.Input(shape=(32, 128))
        x = PositionalEmbedding(max_seq_len=64, dim=128)(inputs)
        x = keras.layers.Dense(64, activation="relu")(x)
        x = keras.layers.GlobalAveragePooling1D()(x)
        outputs = keras.layers.Dense(10, activation="softmax")(x)

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

    def test_model_save_load(self, input_tensor: tf.Tensor):
        """Test saving and loading a model with the custom layer."""
        # Create a model with the positional embedding layer
        inputs = keras.Input(shape=(32, 128))
        x = PositionalEmbedding(max_seq_len=64, dim=128, name="pos_embed")(inputs)
        x = keras.layers.Dense(32, activation="relu")(x)
        x = keras.layers.GlobalAveragePooling1D()(x)
        outputs = keras.layers.Dense(5)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Generate prediction before saving
        original_prediction = model.predict(input_tensor, verbose=0)

        # Create temporary directory for model
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "test_model.keras")

            # Save the model
            model.save(model_path)

            # Load the model
            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={"PositionalEmbedding": PositionalEmbedding}
            )

            # Generate prediction with loaded model
            loaded_prediction = loaded_model.predict(input_tensor, verbose=0)

            # Check predictions match
            assert np.allclose(original_prediction, loaded_prediction, rtol=1e-5)

            # Check layer type is preserved
            assert isinstance(loaded_model.get_layer("pos_embed"), PositionalEmbedding)

    def test_gradient_flow(self, input_tensor: tf.Tensor):
        """Test gradient flow through the layer."""
        layer = PositionalEmbedding(max_seq_len=64, dim=128)

        # Watch the variables
        with tf.GradientTape() as tape:
            inputs = tf.Variable(input_tensor)
            outputs = layer(inputs, training=True)
            loss = tf.reduce_mean(tf.square(outputs))

        # Get gradients
        grads = tape.gradient(loss, layer.trainable_variables)

        # Check gradients exist and are not None
        assert all(g is not None for g in grads)

        # Check gradients have values (not all zeros)
        assert all(tf.reduce_any(g != 0) for g in grads)

    def test_training_mode_differences(self):
        """Test differences between training and inference modes."""
        layer = PositionalEmbedding(max_seq_len=32, dim=64, dropout=0.3)
        test_input = tf.random.normal([4, 16, 64])

        # Multiple calls in training mode should give different results due to dropout
        outputs_train = [layer(test_input, training=True) for _ in range(3)]

        # Multiple calls in inference mode should give identical results
        outputs_inference = [layer(test_input, training=False) for _ in range(3)]

        # Training outputs should differ (probabilistic test)
        assert not np.allclose(outputs_train[0].numpy(), outputs_train[1].numpy(), atol=1e-5)

        # Inference outputs should be identical
        assert np.allclose(outputs_inference[0].numpy(), outputs_inference[1].numpy())

    def test_numerical_stability(self):
        """Test layer stability with extreme input values."""
        layer = PositionalEmbedding(max_seq_len=16, dim=32)

        # Test with different input magnitudes
        test_cases = [
            tf.zeros((2, 8, 32)),  # Zeros
            tf.ones((2, 8, 32)) * 1e-10,  # Very small values
            tf.ones((2, 8, 32)) * 1e10,  # Very large values
            tf.random.normal((2, 8, 32)) * 1e5,  # Large random values
        ]

        for test_input in test_cases:
            output = layer(test_input, training=False)

            # Check for NaN/Inf values
            assert not np.any(np.isnan(output.numpy())), "NaN values detected in output"
            assert not np.any(np.isinf(output.numpy())), "Inf values detected in output"

    def test_different_initializers(self):
        """Test layer with different initializers."""
        initializers_to_test = [
            "zeros",
            "ones",
            "glorot_uniform",
            "glorot_normal",
            keras.initializers.RandomUniform(-0.1, 0.1),
            keras.initializers.RandomNormal(0.0, 0.05),
        ]

        for init in initializers_to_test:
            layer = PositionalEmbedding(
                max_seq_len=16,
                dim=32,
                pos_initializer=init
            )

            test_input = tf.random.normal([2, 8, 32])
            output = layer(test_input, training=False)

            # Check output is valid
            assert output.shape == test_input.shape
            assert not np.any(np.isnan(output.numpy()))

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with sequence length equal to max_seq_len
        layer = PositionalEmbedding(max_seq_len=16, dim=32)
        test_input = tf.random.normal([1, 16, 32])  # seq_len == max_seq_len
        output = layer(test_input, training=False)
        assert output.shape == test_input.shape

        # Test with minimum sequence length
        test_input_min = tf.random.normal([1, 1, 32])  # seq_len = 1
        output_min = layer(test_input_min, training=False)
        assert output_min.shape == test_input_min.shape

    def test_batch_size_independence(self):
        """Test that layer works with different batch sizes."""
        layer = PositionalEmbedding(max_seq_len=32, dim=64)

        # Test with different batch sizes
        for batch_size in [1, 2, 8, 16]:
            test_input = tf.random.normal([batch_size, 16, 64])
            output = layer(test_input, training=False)
            assert output.shape == test_input.shape

    def test_multiple_calls_consistency(self):
        """Test that multiple calls with same input give consistent results."""
        layer = PositionalEmbedding(max_seq_len=32, dim=64, dropout=0.0)
        test_input = tf.random.normal([2, 16, 64])

        # Multiple calls should give identical results (no dropout)
        outputs = [layer(test_input, training=False) for _ in range(3)]

        for i in range(1, len(outputs)):
            assert np.allclose(outputs[0].numpy(), outputs[i].numpy())

    def test_weight_sharing(self):
        """Test that positional embeddings are shared across batch dimension."""
        layer = PositionalEmbedding(max_seq_len=8, dim=4, dropout=0.0)

        # Create inputs with different values for each batch item
        test_input = tf.stack([
            tf.ones([4, 4]),  # First batch item: all ones
            tf.zeros([4, 4]),  # Second batch item: all zeros
        ])

        output = layer(test_input, training=False)

        # The difference between outputs should be the same as difference between inputs
        # because the same positional embeddings are added to both
        input_diff = test_input[0] - test_input[1]
        output_diff = output[0] - output[1]

        assert np.allclose(input_diff.numpy(), output_diff.numpy())


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])