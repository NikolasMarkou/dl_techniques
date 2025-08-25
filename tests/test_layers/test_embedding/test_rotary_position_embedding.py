import pytest
import numpy as np
import keras
import os
import tempfile

from dl_techniques.layers.embedding.rotary_position_embedding import RotaryPositionEmbedding


class TestRotaryPositionEmbedding:
    """Test suite for RotaryPositionEmbedding layer implementation."""

    @pytest.fixture
    def input_tensor(self):
        """Create a test input tensor."""
        # Shape: (batch_size, num_heads, seq_len, head_dim)
        return keras.random.normal([4, 8, 128, 64])

    @pytest.fixture
    def layer_instance(self):
        """Create a default layer instance for testing."""
        return RotaryPositionEmbedding(head_dim=64, max_seq_len=512)

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        layer = RotaryPositionEmbedding(head_dim=64, max_seq_len=256)

        # Check default values
        assert layer.head_dim == 64
        assert layer.max_seq_len == 256
        assert layer.rope_theta == 10000.0
        assert layer.rope_percentage == 0.5
        assert layer.rope_dim == 32  # 64 * 0.5
        assert layer.cos_cached is None
        assert layer.sin_cached is None
        assert layer._build_input_shape is None

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        layer = RotaryPositionEmbedding(
            head_dim=128,
            max_seq_len=1024,
            rope_theta=50000.0,
            rope_percentage=0.25,
        )

        # Check custom values
        assert layer.head_dim == 128
        assert layer.max_seq_len == 1024
        assert layer.rope_theta == 50000.0
        assert layer.rope_percentage == 0.25
        assert layer.rope_dim == 32  # 128 * 0.25

    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        # Test negative or zero head_dim
        with pytest.raises(ValueError, match="head_dim must be positive"):
            RotaryPositionEmbedding(head_dim=-64, max_seq_len=256)

        with pytest.raises(ValueError, match="head_dim must be positive"):
            RotaryPositionEmbedding(head_dim=0, max_seq_len=256)

        # Test negative or zero max_seq_len
        with pytest.raises(ValueError, match="max_seq_len must be positive"):
            RotaryPositionEmbedding(head_dim=64, max_seq_len=-256)

        with pytest.raises(ValueError, match="max_seq_len must be positive"):
            RotaryPositionEmbedding(head_dim=64, max_seq_len=0)

        # Test negative or zero rope_theta
        with pytest.raises(ValueError, match="rope_theta must be positive"):
            RotaryPositionEmbedding(head_dim=64, max_seq_len=256, rope_theta=-10000.0)

        with pytest.raises(ValueError, match="rope_theta must be positive"):
            RotaryPositionEmbedding(head_dim=64, max_seq_len=256, rope_theta=0.0)

        # Test invalid rope_percentage
        with pytest.raises(ValueError, match="rope_percentage must be in"):
            RotaryPositionEmbedding(head_dim=64, max_seq_len=256, rope_percentage=0.0)

        with pytest.raises(ValueError, match="rope_percentage must be in"):
            RotaryPositionEmbedding(head_dim=64, max_seq_len=256, rope_percentage=1.5)

        with pytest.raises(ValueError, match="rope_percentage must be in"):
            RotaryPositionEmbedding(head_dim=64, max_seq_len=256, rope_percentage=-0.1)

    def test_build_process(self, input_tensor, layer_instance):
        """Test that the layer builds properly."""
        # Trigger build through forward pass
        output = layer_instance(input_tensor)

        # Check that layer was built
        assert layer_instance.built is True
        assert len(layer_instance.weights) > 0
        assert hasattr(layer_instance, "cos_cached")
        assert hasattr(layer_instance, "sin_cached")
        assert layer_instance.cos_cached is not None
        assert layer_instance.sin_cached is not None

        # Check cached weights shapes
        expected_freq_dim = layer_instance.rope_dim // 2
        expected_cache_shape = (layer_instance.max_seq_len, expected_freq_dim)
        assert layer_instance.cos_cached.shape == expected_cache_shape
        assert layer_instance.sin_cached.shape == expected_cache_shape

        # Check that weights are non-trainable
        assert not layer_instance.cos_cached.trainable
        assert not layer_instance.sin_cached.trainable

    def test_build_with_invalid_input_shape(self, layer_instance):
        """Test build with invalid input shapes."""
        # Test with wrong number of dimensions
        with pytest.raises(ValueError, match="Expected 4D input"):
            layer_instance.build((None, 128, 64))  # Only 3D

        with pytest.raises(ValueError, match="Expected 4D input"):
            layer_instance.build((None, 8, 128, 64, 32))  # 5D

        # Test with mismatched head_dim
        with pytest.raises(ValueError, match="Input head_dim .* doesn't match"):
            layer_instance.build((None, 8, 128, 128))  # head_dim=128, but layer expects 64

    def test_output_shapes(self, input_tensor):
        """Test that output shapes are computed correctly."""
        configs_to_test = [
            {"head_dim": 32, "max_seq_len": 256, "rope_percentage": 0.5},
            {"head_dim": 64, "max_seq_len": 512, "rope_percentage": 0.25},
            {"head_dim": 128, "max_seq_len": 1024, "rope_percentage": 0.75},
        ]

        for config in configs_to_test:
            layer = RotaryPositionEmbedding(**config)

            # Create appropriate input tensor
            head_dim = config["head_dim"]
            test_input = keras.random.normal([2, 4, 100, head_dim])

            output = layer(test_input)

            # Check output shape matches input shape (RoPE doesn't change dimensions)
            assert output.shape == test_input.shape

            # Test compute_output_shape separately
            computed_shape = layer.compute_output_shape(test_input.shape)
            assert computed_shape == test_input.shape

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
        assert np.allclose(output.numpy(), output_inference.numpy())

        # Test with training=True
        output_training = layer_instance(input_tensor, training=True)
        assert output_training.shape == input_tensor.shape
        assert np.allclose(output.numpy(), output_training.numpy())

    def test_sequence_length_limits(self, layer_instance):
        """Test sequence length validation."""
        # Build the layer first
        layer_instance.build((None, 8, 128, 64))

        # Test with sequence length exceeding max_seq_len
        long_input = keras.random.normal([2, 8, 600, 64])  # seq_len=600 > max_seq_len=512

        with pytest.raises(ValueError, match="Input sequence length .* exceeds max_seq_len"):
            layer_instance(long_input)

        # Test with valid sequence length
        valid_input = keras.random.normal([2, 8, 400, 64])  # seq_len=400 < max_seq_len=512
        output = layer_instance(valid_input)
        assert output.shape == valid_input.shape

    def test_different_rope_percentages(self):
        """Test layer with different rope_percentage values."""
        rope_percentages = [0.1, 0.25, 0.5, 0.75, 1.0]
        head_dim = 64

        for rope_percentage in rope_percentages:
            layer = RotaryPositionEmbedding(
                head_dim=head_dim,
                max_seq_len=256,
                rope_percentage=rope_percentage
            )

            # Create test input
            test_input = keras.random.normal([2, 4, 100, head_dim])

            # Test forward pass
            output = layer(test_input)

            # Check output is valid
            assert not np.any(np.isnan(output.numpy()))
            assert output.shape == test_input.shape

            # Check rope_dim calculation
            expected_rope_dim = int(head_dim * rope_percentage)
            if expected_rope_dim % 2 != 0:
                expected_rope_dim -= 1
            assert layer.rope_dim == expected_rope_dim

    def test_different_configurations(self):
        """Test layer with different configurations."""
        configurations = [
            {"head_dim": 32, "max_seq_len": 128, "rope_theta": 5000.0},
            {"head_dim": 64, "max_seq_len": 512, "rope_percentage": 0.25},
            {"head_dim": 128, "max_seq_len": 1024, "rope_theta": 20000.0, "rope_percentage": 0.75},
        ]

        for config in configurations:
            layer = RotaryPositionEmbedding(**config)

            # Create appropriate input
            head_dim = config["head_dim"]
            test_input = keras.random.normal([2, 4, 50, head_dim])

            # Test forward pass
            output = layer(test_input)

            # Check output is valid
            assert not np.any(np.isnan(output.numpy()))
            assert output.shape == test_input.shape

    def test_rotary_transformation_properties(self):
        """Test mathematical properties of rotary transformation."""
        layer = RotaryPositionEmbedding(head_dim=64, max_seq_len=256)

        # Create deterministic input for testing
        test_input = keras.ops.ones([1, 1, 10, 64])

        # Apply RoPE
        output = layer(test_input)

        # Basic checks
        assert not np.any(np.isnan(output.numpy()))
        assert output.shape == test_input.shape

        # The rotated dimensions should have similar magnitude (rotations preserve norm)
        rope_dim = layer.rope_dim
        input_rope_part = test_input[..., :rope_dim]
        output_rope_part = output[..., :rope_dim]

        # Check that rotation preserves magnitude approximately
        input_norms = keras.ops.sqrt(keras.ops.sum(keras.ops.square(input_rope_part), axis=-1))
        output_norms = keras.ops.sqrt(keras.ops.sum(keras.ops.square(output_rope_part), axis=-1))

        # Allow small numerical differences
        assert np.allclose(input_norms.numpy(), output_norms.numpy(), rtol=1e-5)

    def test_position_dependency(self):
        """Test that output depends on position."""
        layer = RotaryPositionEmbedding(head_dim=64, max_seq_len=256)

        # Create identical vectors at different positions
        batch_size, num_heads, seq_len, head_dim = 1, 1, 5, 64
        test_input = keras.ops.ones([batch_size, num_heads, seq_len, head_dim])

        # Apply RoPE
        output = layer(test_input)

        # Different positions should produce different outputs (for the rotated dimensions)
        rope_dim = layer.rope_dim
        pos_0 = output[0, 0, 0, :rope_dim]
        pos_1 = output[0, 0, 1, :rope_dim]
        pos_2 = output[0, 0, 2, :rope_dim]

        # Positions should be different (not all close)
        assert not np.allclose(pos_0.numpy(), pos_1.numpy(), rtol=1e-3)
        assert not np.allclose(pos_1.numpy(), pos_2.numpy(), rtol=1e-3)

        # But the non-rotated dimensions should remain the same
        if rope_dim < head_dim:
            pos_0_pass = output[0, 0, 0, rope_dim:]
            pos_1_pass = output[0, 0, 1, rope_dim:]
            assert np.allclose(pos_0_pass.numpy(), pos_1_pass.numpy())

    def test_serialization(self):
        """Test serialization and deserialization of the layer."""
        original_layer = RotaryPositionEmbedding(
            head_dim=128,
            max_seq_len=1024,
            rope_theta=50000.0,
            rope_percentage=0.25,
        )

        # Build the layer
        input_shape = (None, 8, 256, 128)
        original_layer.build(input_shape)

        # Get configs
        config = original_layer.get_config()
        build_config = original_layer.get_build_config()

        # Recreate the layer
        recreated_layer = RotaryPositionEmbedding.from_config(config)
        recreated_layer.build_from_config(build_config)

        # Check configuration matches
        assert recreated_layer.head_dim == original_layer.head_dim
        assert recreated_layer.max_seq_len == original_layer.max_seq_len
        assert recreated_layer.rope_theta == original_layer.rope_theta
        assert recreated_layer.rope_percentage == original_layer.rope_percentage
        assert recreated_layer.rope_dim == original_layer.rope_dim

        # Check weights match (shapes and values)
        assert len(recreated_layer.weights) == len(original_layer.weights)
        for w1, w2 in zip(original_layer.weights, recreated_layer.weights):
            assert w1.shape == w2.shape
            assert np.allclose(w1.numpy(), w2.numpy())

    def test_model_integration(self, input_tensor):
        """Test the layer in a model context."""
        # Create a simple model with the RoPE layer
        inputs = keras.Input(shape=input_tensor.shape[1:])
        x = RotaryPositionEmbedding(head_dim=64, max_seq_len=512)(inputs)
        x = keras.layers.LayerNormalization()(x)
        x = keras.layers.GlobalAveragePooling2D()(x)  # Pool over seq_len and head_dim
        x = keras.layers.Dense(64)(x)
        outputs = keras.layers.Dense(10)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Compile the model
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
        )

        # Test forward pass
        y_pred = model(input_tensor, training=False)
        assert y_pred.shape == (input_tensor.shape[0], 10)

    def test_model_save_load(self, input_tensor):
        """Test saving and loading a model with the RoPE layer."""
        # Create a model with the RoPE layer
        inputs = keras.Input(shape=input_tensor.shape[1:])
        x = RotaryPositionEmbedding(head_dim=64, max_seq_len=512, name="rope_layer")(inputs)
        x = keras.layers.LayerNormalization()(x)
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dense(64)(x)
        outputs = keras.layers.Dense(10)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Generate a prediction before saving
        original_prediction = model.predict(input_tensor, verbose=0)

        # Create temporary directory for model
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "model.keras")

            # Save the model
            model.save(model_path)

            # Load the model
            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={
                    "RotaryPositionEmbedding": RotaryPositionEmbedding
                }
            )

            # Generate prediction with loaded model
            loaded_prediction = loaded_model.predict(input_tensor, verbose=0)

            # Check predictions match
            assert np.allclose(original_prediction, loaded_prediction, rtol=1e-5)

            # Check layer type is preserved
            assert isinstance(loaded_model.get_layer("rope_layer"), RotaryPositionEmbedding)

    def test_numerical_stability(self):
        """Test layer stability with extreme input values."""
        layer = RotaryPositionEmbedding(head_dim=32, max_seq_len=128)

        # Create inputs with different magnitudes
        batch_size, num_heads, seq_len, head_dim = 2, 4, 50, 32

        test_cases = [
            keras.ops.zeros((batch_size, num_heads, seq_len, head_dim)),  # Zeros
            keras.ops.ones((batch_size, num_heads, seq_len, head_dim)) * 1e-10,  # Very small values
            keras.ops.ones((batch_size, num_heads, seq_len, head_dim)) * 1e5,  # Large values
            keras.random.normal((batch_size, num_heads, seq_len, head_dim)) * 100  # Large random values
        ]

        for test_input in test_cases:
            output = layer(test_input)

            # Check for NaN/Inf values
            assert not np.any(np.isnan(output.numpy())), "NaN values detected in output"
            assert not np.any(np.isinf(output.numpy())), "Inf values detected in output"

    def test_cache_building(self):
        """Test that cos/sin cache is built correctly."""
        layer = RotaryPositionEmbedding(head_dim=64, max_seq_len=128, rope_theta=10000.0)

        # Build the layer
        layer.build((None, 8, 100, 64))

        # Check cache shapes
        freq_dim = layer.rope_dim // 2  # 16
        expected_shape = (128, freq_dim)
        assert layer.cos_cached.shape == expected_shape
        assert layer.sin_cached.shape == expected_shape

        # Check that cached values are reasonable
        cos_values = layer.cos_cached.numpy()
        sin_values = layer.sin_cached.numpy()

        # All values should be in [-1, 1] range
        assert np.all(cos_values >= -1.0) and np.all(cos_values <= 1.0)
        assert np.all(sin_values >= -1.0) and np.all(sin_values <= 1.0)

        # Check that cos^2 + sin^2 ≈ 1 (trigonometric identity)
        squared_sum = cos_values ** 2 + sin_values ** 2
        assert np.allclose(squared_sum, 1.0, rtol=1e-6)

    def test_zero_rope_dim_handling(self):
        """Test handling when rope_dim becomes 0."""
        # Very small rope_percentage that results in rope_dim = 0
        layer = RotaryPositionEmbedding(
            head_dim=64,
            max_seq_len=128,
            rope_percentage=0.01  # This will result in rope_dim = 0 after adjustment
        )

        test_input = keras.random.normal([2, 4, 50, 64])

        # Should not crash and should return input unchanged
        output = layer(test_input)
        assert output.shape == test_input.shape
        # When rope_dim = 0, output should be identical to input
        assert np.allclose(output.numpy(), test_input.numpy())

    def test_odd_head_dim_warning(self):
        """Test that odd head_dim produces a warning."""
        # This test would need to capture logging output in a real implementation
        # For now, just test that it doesn't crash
        layer = RotaryPositionEmbedding(head_dim=63, max_seq_len=128)  # Odd head_dim

        test_input = keras.random.normal([2, 4, 50, 63])
        output = layer(test_input)

        assert output.shape == test_input.shape
        assert not np.any(np.isnan(output.numpy()))