import pytest
import numpy as np
import tensorflow as tf
import keras
import tempfile
import os

from dl_techniques.layers.embedding.patch_embedding import (
    PatchEmbedding2D, PatchEmbedding1D)


class TestPatchEmbedding2D:
    """Comprehensive test suite for PatchEmbedding2D layer."""

    @pytest.fixture
    def input_tensor(self) -> tf.Tensor:
        """Create a test input tensor for 2D patches."""
        return tf.random.normal([4, 224, 224, 3])  # batch_size=4, height=224, width=224, channels=3

    @pytest.fixture
    def layer_instance(self) -> PatchEmbedding2D:
        """Create a default layer instance for testing."""
        return PatchEmbedding2D(patch_size=16, embed_dim=768)

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        layer = PatchEmbedding2D(patch_size=16, embed_dim=512)

        # Check default values
        assert layer.patch_size == (16, 16)
        assert layer.embed_dim == 512
        assert isinstance(layer.kernel_initializer, keras.initializers.GlorotNormal)
        assert layer.kernel_regularizer is None
        assert isinstance(layer.bias_initializer, keras.initializers.Zeros)
        assert layer.bias_regularizer is None
        assert layer.use_bias is True
        assert layer.proj is None

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        custom_kernel_reg = keras.regularizers.L2(1e-4)
        custom_bias_reg = keras.regularizers.L1(1e-5)

        layer = PatchEmbedding2D(
            patch_size=(8, 16),
            embed_dim=256,
            kernel_initializer="he_normal",
            kernel_regularizer=custom_kernel_reg,
            bias_initializer="ones",
            bias_regularizer=custom_bias_reg,
            activation="relu",
            use_bias=False
        )

        # Check custom values
        assert layer.patch_size == (8, 16)
        assert layer.embed_dim == 256
        assert isinstance(layer.kernel_initializer, keras.initializers.HeNormal)
        assert layer.kernel_regularizer == custom_kernel_reg
        assert isinstance(layer.bias_initializer, keras.initializers.Ones)
        assert layer.bias_regularizer == custom_bias_reg
        assert layer.use_bias is False

    def test_patch_size_handling(self):
        """Test different patch size specifications."""
        # Integer patch size
        layer1 = PatchEmbedding2D(patch_size=16, embed_dim=512)
        assert layer1.patch_size == (16, 16)

        # Tuple patch size
        layer2 = PatchEmbedding2D(patch_size=(8, 16), embed_dim=512)
        assert layer2.patch_size == (8, 16)

    def test_build_process(self, input_tensor: tf.Tensor):
        """Test that the layer builds properly."""
        layer = PatchEmbedding2D(patch_size=16, embed_dim=768)

        # Build by calling the layer
        output = layer(input_tensor)

        # Check that layer was built
        assert layer.built is True
        assert layer.proj is not None
        assert isinstance(layer.proj, keras.layers.Conv2D)

    def test_build_invalid_input_shape(self):
        """Test build with invalid input shapes."""
        layer = PatchEmbedding2D(patch_size=16, embed_dim=768)

        # 3D input should fail
        with pytest.raises(ValueError, match="Expected 4D input"):
            layer.build((32, 224, 224))

        # Non-divisible dimensions should fail
        with pytest.raises(ValueError, match="Input height .* must be divisible"):
            layer.build((4, 225, 224, 3))  # 225 not divisible by 16

        with pytest.raises(ValueError, match="Input width .* must be divisible"):
            layer.build((4, 224, 225, 3))  # 225 not divisible by 16

    def test_output_shapes(self, input_tensor: tf.Tensor):
        """Test that output shapes are computed correctly."""
        test_configs = [
            (16, 768, 224, 224),  # Standard ViT-Base
            (32, 512, 224, 224),  # Larger patches
            (8, 256, 224, 224),  # Smaller patches
        ]

        for patch_size, embed_dim, height, width in test_configs:
            layer = PatchEmbedding2D(patch_size=patch_size, embed_dim=embed_dim)

            # Create appropriate input
            test_input = tf.random.normal([2, height, width, 3])
            output = layer(test_input)

            # Calculate expected number of patches
            expected_num_patches = (height // patch_size) * (width // patch_size)
            expected_shape = (2, expected_num_patches, embed_dim)

            # Check output shape
            assert output.shape == expected_shape

            # Test compute_output_shape separately
            computed_shape = layer.compute_output_shape(test_input.shape)
            assert computed_shape == expected_shape

    def test_rectangular_patches(self):
        """Test with rectangular patches."""
        layer = PatchEmbedding2D(patch_size=(8, 16), embed_dim=256)

        # Input divisible by patch dimensions
        test_input = tf.random.normal([2, 64, 128, 3])  # 64/8=8, 128/16=8
        output = layer(test_input)

        # Expected: 8 * 8 = 64 patches
        expected_shape = (2, 64, 256)
        assert output.shape == expected_shape

    def test_forward_pass_basic(self, input_tensor: tf.Tensor):
        """Test basic forward pass functionality."""
        layer = PatchEmbedding2D(patch_size=16, embed_dim=768)
        output = layer(input_tensor, training=False)

        # Basic sanity checks
        expected_num_patches = (224 // 16) * (224 // 16)  # 14 * 14 = 196
        assert output.shape == (4, expected_num_patches, 768)
        assert not np.any(np.isnan(output.numpy()))
        assert not np.any(np.isinf(output.numpy()))

    def test_forward_pass_with_activation(self):
        """Test forward pass with different activations."""
        activations = ["relu", "gelu", "swish", "linear"]

        for activation in activations:
            layer = PatchEmbedding2D(patch_size=16, embed_dim=256, activation=activation)
            test_input = tf.random.normal([2, 64, 64, 3])
            output = layer(test_input, training=False)

            # Check output is valid
            assert output.shape == (2, 16, 256)  # 4*4 patches
            assert not np.any(np.isnan(output.numpy()))

    def test_num_patches_property(self):
        """Test the num_patches property."""
        layer = PatchEmbedding2D(patch_size=16, embed_dim=768)

        # Before build
        assert layer.num_patches is None

        # After build
        layer.build((None, 224, 224, 3))
        assert layer.num_patches == 196  # 14 * 14

    def test_get_patch_grid_shape(self):
        """Test the get_patch_grid_shape method."""
        layer = PatchEmbedding2D(patch_size=16, embed_dim=768)

        # Before build
        assert layer.get_patch_grid_shape() is None

        # After build
        layer.build((None, 224, 224, 3))
        assert layer.get_patch_grid_shape() == (14, 14)

        # Test with rectangular patches
        layer_rect = PatchEmbedding2D(patch_size=(8, 16), embed_dim=256)
        layer_rect.build((None, 64, 128, 3))
        assert layer_rect.get_patch_grid_shape() == (8, 8)

    def test_serialization(self):
        """Test layer serialization and deserialization."""
        original_layer = PatchEmbedding2D(
            patch_size=(8, 16),
            embed_dim=256,
            kernel_initializer="he_normal",
            activation="relu",
            use_bias=False
        )

        # Build the layer
        input_shape = (None, 64, 128, 3)
        original_layer.build(input_shape)

        # Test data
        test_input = tf.random.normal([2, 64, 128, 3])
        original_output = original_layer(test_input, training=False)

        # Get configs
        config = original_layer.get_config()
        build_config = original_layer.get_build_config()

        # Recreate the layer
        recreated_layer = PatchEmbedding2D.from_config(config)
        recreated_layer.build_from_config(build_config)

        # Copy weights to ensure identical output
        for orig_weight, recreated_weight in zip(original_layer.weights, recreated_layer.weights):
            recreated_weight.assign(orig_weight)

        # Test configuration matches
        assert recreated_layer.patch_size == original_layer.patch_size
        assert recreated_layer.embed_dim == original_layer.embed_dim
        assert recreated_layer.use_bias == original_layer.use_bias

        # Test output matches
        recreated_output = recreated_layer(test_input, training=False)
        assert np.allclose(original_output.numpy(), recreated_output.numpy())

    def test_model_integration(self, input_tensor: tf.Tensor):
        """Test the layer in a model context."""
        # Create a simple Vision Transformer-like model
        inputs = keras.Input(shape=(224, 224, 3))

        # Patch embedding
        patches = PatchEmbedding2D(patch_size=16, embed_dim=768)(inputs)

        # Add positional embeddings (simplified)
        x = keras.layers.Dense(768)(patches)

        # Global average pooling and classification
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

    def test_layer_serialization_only(self):
        """Test just the layer serialization without full model save/load."""
        # Create and configure layer
        layer = PatchEmbedding2D(
            patch_size=16,
            embed_dim=256,
            kernel_initializer="glorot_normal",
            activation="relu"
        )

        # Build the layer
        layer.build((None, 64, 64, 3))

        # Test serialization
        config = layer.get_config()
        build_config = layer.get_build_config()

        # Recreate layer
        new_layer = PatchEmbedding2D.from_config(config)
        new_layer.build_from_config(build_config)

        # Test that configs match
        assert new_layer.patch_size == layer.patch_size
        assert new_layer.embed_dim == layer.embed_dim

        # Test forward pass with both layers
        test_input = tf.random.normal([2, 64, 64, 3])

        # Copy weights to ensure identical outputs
        for orig_weight, new_weight in zip(layer.weights, new_layer.weights):
            new_weight.assign(orig_weight)

        output1 = layer(test_input)
        output2 = new_layer(test_input)

        assert np.allclose(output1.numpy(), output2.numpy())

    def test_model_save_load(self, input_tensor: tf.Tensor):
        """Test saving and loading a model with the custom layer."""
        # Create a model with the patch embedding layer
        inputs = keras.Input(shape=(224, 224, 3))
        x = PatchEmbedding2D(patch_size=16, embed_dim=256, name="patch_embed")(inputs)
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

            # Load the model with comprehensive custom objects
            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={
                    "PatchEmbedding2D": PatchEmbedding2D,
                    "PatchEmbedding1D": PatchEmbedding1D
                }
            )

            # Generate prediction with loaded model
            loaded_prediction = loaded_model.predict(input_tensor, verbose=0)

            # Check predictions match
            assert np.allclose(original_prediction, loaded_prediction, rtol=1e-5)

            # Check layer type is preserved
            assert isinstance(loaded_model.get_layer("patch_embed"), PatchEmbedding2D)

    def test_gradient_flow(self, input_tensor: tf.Tensor):
        """Test gradient flow through the layer."""
        layer = PatchEmbedding2D(patch_size=16, embed_dim=768)

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

    def test_numerical_stability(self):
        """Test layer stability with extreme input values."""
        layer = PatchEmbedding2D(patch_size=16, embed_dim=256)

        # Test with different input magnitudes
        test_cases = [
            tf.zeros((2, 64, 64, 3)),  # Zeros
            tf.ones((2, 64, 64, 3)) * 1e-10,  # Very small values
            tf.ones((2, 64, 64, 3)) * 1e5,  # Large values
        ]

        for test_input in test_cases:
            output = layer(test_input, training=False)

            # Check for NaN/Inf values
            assert not np.any(np.isnan(output.numpy())), "NaN values detected in output"
            assert not np.any(np.isinf(output.numpy())), "Inf values detected in output"

    def test_different_input_sizes(self):
        """Test with different input sizes."""
        layer = PatchEmbedding2D(patch_size=16, embed_dim=256)

        # Test with different sizes (all divisible by 16)
        sizes = [(32, 32), (64, 64), (128, 128), (224, 224)]

        for height, width in sizes:
            test_input = tf.random.normal([1, height, width, 3])
            output = layer(test_input, training=False)

            expected_patches = (height // 16) * (width // 16)
            assert output.shape == (1, expected_patches, 256)


class TestPatchEmbedding1D:
    """Comprehensive test suite for PatchEmbedding1D layer."""

    @pytest.fixture
    def input_tensor(self) -> tf.Tensor:
        """Create a test input tensor for 1D patches."""
        return tf.random.normal([4, 128, 64])  # batch_size=4, seq_len=128, features=64

    @pytest.fixture
    def layer_instance(self) -> PatchEmbedding1D:
        """Create a default layer instance for testing."""
        return PatchEmbedding1D(patch_size=16, embed_dim=256)

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        layer = PatchEmbedding1D(patch_size=16, embed_dim=256)

        # Check default values
        assert layer.patch_size == 16
        assert layer.embed_dim == 256
        assert layer.stride == 16  # Should default to patch_size
        assert layer.padding == 'causal'
        assert layer.use_bias is True
        assert isinstance(layer.kernel_initializer, keras.initializers.GlorotUniform)
        assert isinstance(layer.bias_initializer, keras.initializers.Zeros)
        assert layer.embedding is None

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        layer = PatchEmbedding1D(
            patch_size=8,
            embed_dim=128,
            stride=4,
            padding='same',
            use_bias=False,
            kernel_initializer="he_normal",
            bias_initializer="ones"
        )

        # Check custom values
        assert layer.patch_size == 8
        assert layer.embed_dim == 128
        assert layer.stride == 4
        assert layer.padding == 'same'
        assert layer.use_bias is False
        assert isinstance(layer.kernel_initializer, keras.initializers.HeNormal)
        assert isinstance(layer.bias_initializer, keras.initializers.Ones)

    def test_build_process(self, input_tensor: tf.Tensor):
        """Test that the layer builds properly."""
        layer = PatchEmbedding1D(patch_size=16, embed_dim=256)

        # Build by calling the layer
        output = layer(input_tensor)

        # Check that layer was built
        assert layer.built is True
        assert layer.embedding is not None
        assert isinstance(layer.embedding, keras.layers.Conv1D)

    def test_output_shapes_different_padding(self):
        """Test output shapes with different padding modes."""
        seq_len = 128
        patch_size = 16
        embed_dim = 256
        test_input = tf.random.normal([2, seq_len, 32])

        # Test 'valid' padding
        layer_valid = PatchEmbedding1D(patch_size=patch_size, embed_dim=embed_dim, padding='valid')
        output_valid = layer_valid(test_input)
        expected_len_valid = (seq_len - patch_size) // patch_size + 1
        assert output_valid.shape == (2, expected_len_valid, embed_dim)

        # Test 'same' padding
        layer_same = PatchEmbedding1D(patch_size=patch_size, embed_dim=embed_dim, padding='same')
        output_same = layer_same(test_input)
        expected_len_same = (seq_len + patch_size - 1) // patch_size
        assert output_same.shape == (2, expected_len_same, embed_dim)

        # Test 'causal' padding
        layer_causal = PatchEmbedding1D(patch_size=patch_size, embed_dim=embed_dim, padding='causal')
        output_causal = layer_causal(test_input)
        expected_len_causal = seq_len // patch_size
        assert output_causal.shape == (2, expected_len_causal, embed_dim)

    def test_overlapping_patches(self):
        """Test with overlapping patches (stride < patch_size)."""
        layer = PatchEmbedding1D(patch_size=16, embed_dim=256, stride=8, padding='valid')

        test_input = tf.random.normal([2, 128, 32])
        output = layer(test_input)

        # With stride=8, patch_size=16, we get overlapping patches
        expected_len = (128 - 16) // 8 + 1
        assert output.shape == (2, expected_len, 256)

    def test_nan_handling(self):
        """Test that NaN values are handled correctly."""
        layer = PatchEmbedding1D(patch_size=8, embed_dim=128)

        # Create input with NaN values
        test_input = tf.random.normal([2, 64, 16])
        test_input_with_nan = tf.concat([
            test_input[:, :32, :],
            tf.fill([2, 32, 16], float('nan'))
        ], axis=1)

        output = layer(test_input_with_nan, training=False)

        # Check that output doesn't contain NaN
        assert not np.any(np.isnan(output.numpy()))

    def test_forward_pass_basic(self, input_tensor: tf.Tensor):
        """Test basic forward pass functionality."""
        layer = PatchEmbedding1D(patch_size=16, embed_dim=256)
        output = layer(input_tensor, training=False)

        # Basic sanity checks
        assert len(output.shape) == 3
        assert output.shape[0] == input_tensor.shape[0]  # batch size preserved
        assert output.shape[2] == 256  # embed_dim
        assert not np.any(np.isnan(output.numpy()))
        assert not np.any(np.isinf(output.numpy()))

    def test_compute_output_shape(self):
        """Test compute_output_shape method."""
        # Test with different padding modes
        test_cases = [
            (PatchEmbedding1D(patch_size=8, embed_dim=128, padding='valid'), (2, 64, 16)),
            (PatchEmbedding1D(patch_size=8, embed_dim=128, padding='same'), (2, 64, 16)),
            (PatchEmbedding1D(patch_size=8, embed_dim=128, padding='causal'), (2, 64, 16)),
        ]

        for layer, input_shape in test_cases:
            output_shape = layer.compute_output_shape(input_shape)
            assert output_shape[0] == input_shape[0]  # batch size
            assert output_shape[2] == 128  # embed_dim
            assert output_shape[1] is not None  # sequence length computed

    def test_serialization(self):
        """Test layer serialization and deserialization."""
        original_layer = PatchEmbedding1D(
            patch_size=8,
            embed_dim=128,
            stride=4,
            padding='same',
            use_bias=False,
            kernel_initializer="he_normal"
        )

        # Build the layer
        input_shape = (None, 64, 32)
        original_layer.build(input_shape)

        # Test data
        test_input = tf.random.normal([2, 64, 32])
        original_output = original_layer(test_input, training=False)

        # Get configs
        config = original_layer.get_config()
        build_config = original_layer.get_build_config()

        # Recreate the layer
        recreated_layer = PatchEmbedding1D.from_config(config)
        recreated_layer.build_from_config(build_config)

        # Copy weights to ensure identical output
        for orig_weight, recreated_weight in zip(original_layer.weights, recreated_layer.weights):
            recreated_weight.assign(orig_weight)

        # Test configuration matches
        assert recreated_layer.patch_size == original_layer.patch_size
        assert recreated_layer.embed_dim == original_layer.embed_dim
        assert recreated_layer.stride == original_layer.stride
        assert recreated_layer.padding == original_layer.padding
        assert recreated_layer.use_bias == original_layer.use_bias

        # Test output matches
        recreated_output = recreated_layer(test_input, training=False)
        assert np.allclose(original_output.numpy(), recreated_output.numpy())

    def test_model_integration(self, input_tensor: tf.Tensor):
        """Test the layer in a model context."""
        # Create a simple model with 1D patch embedding
        inputs = keras.Input(shape=(128, 64))

        # Patch embedding
        patches = PatchEmbedding1D(patch_size=16, embed_dim=256)(inputs)

        # Simple processing
        x = keras.layers.Dense(128, activation="relu")(patches)
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
        # Create a model with the 1D patch embedding layer
        inputs = keras.Input(shape=(128, 64))
        x = PatchEmbedding1D(patch_size=16, embed_dim=128, name="patch_embed_1d")(inputs)
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
                custom_objects={"PatchEmbedding1D": PatchEmbedding1D}
            )

            # Generate prediction with loaded model
            loaded_prediction = loaded_model.predict(input_tensor, verbose=0)

            # Check predictions match
            assert np.allclose(original_prediction, loaded_prediction, rtol=1e-5)

            # Check layer type is preserved
            assert isinstance(loaded_model.get_layer("patch_embed_1d"), PatchEmbedding1D)

    def test_gradient_flow(self, input_tensor: tf.Tensor):
        """Test gradient flow through the layer."""
        layer = PatchEmbedding1D(patch_size=16, embed_dim=256)

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

    def test_numerical_stability(self):
        """Test layer stability with extreme input values."""
        layer = PatchEmbedding1D(patch_size=8, embed_dim=128)

        # Test with different input magnitudes
        test_cases = [
            tf.zeros((2, 64, 32)),  # Zeros
            tf.ones((2, 64, 32)) * 1e-10,  # Very small values
            tf.ones((2, 64, 32)) * 1e5,  # Large values
        ]

        for test_input in test_cases:
            output = layer(test_input, training=False)

            # Check for NaN/Inf values
            assert not np.any(np.isnan(output.numpy())), "NaN values detected in output"
            assert not np.any(np.isinf(output.numpy())), "Inf values detected in output"

    def test_different_sequence_lengths(self):
        """Test with different sequence lengths."""
        layer = PatchEmbedding1D(patch_size=8, embed_dim=128, padding='causal')

        # Test with different sequence lengths
        lengths = [32, 64, 128, 256]

        for length in lengths:
            test_input = tf.random.normal([2, length, 16])
            output = layer(test_input, training=False)

            expected_patches = length // 8
            assert output.shape == (2, expected_patches, 128)

    def test_training_vs_inference(self):
        """Test differences between training and inference modes."""
        layer = PatchEmbedding1D(patch_size=16, embed_dim=256)
        test_input = tf.random.normal([2, 128, 32])

        # Training mode
        output_train = layer(test_input, training=True)

        # Inference mode
        output_inference = layer(test_input, training=False)

        # Shapes should be the same
        assert output_train.shape == output_inference.shape

        # Since there's no dropout in this layer, outputs should be identical
        # for the same input (assuming no randomness in Conv1D)
        assert np.allclose(output_train.numpy(), output_inference.numpy())


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])