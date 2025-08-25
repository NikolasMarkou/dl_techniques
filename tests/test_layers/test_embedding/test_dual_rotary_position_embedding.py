import pytest
import numpy as np
import keras
import os
import tempfile

from dl_techniques.layers.embedding.dual_rotary_position_embedding import DualRotaryPositionEmbedding


class TestDualRotaryPositionEmbedding:
    """Test suite for DualRotaryPositionEmbedding layer implementation."""

    @pytest.fixture
    def input_tensor(self):
        """Create a test input tensor."""
        # Shape: (batch_size, num_heads, seq_len, head_dim)
        return keras.random.normal([4, 8, 128, 64])

    @pytest.fixture
    def layer_instance(self):
        """Create a default layer instance for testing."""
        return DualRotaryPositionEmbedding(head_dim=64, max_seq_len=512)

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        layer = DualRotaryPositionEmbedding(head_dim=64, max_seq_len=256)

        # Check default values
        assert layer.head_dim == 64
        assert layer.max_seq_len == 256
        assert layer.global_theta_base == 1_000_000.0
        assert layer.local_theta_base == 10_000.0
        assert layer.cos_global_cached is None
        assert layer.sin_global_cached is None
        assert layer.cos_local_cached is None
        assert layer.sin_local_cached is None
        assert layer._build_input_shape is None

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        layer = DualRotaryPositionEmbedding(
            head_dim=128,
            max_seq_len=1024,
            global_theta_base=2_000_000.0,
            local_theta_base=5_000.0,
        )

        # Check custom values
        assert layer.head_dim == 128
        assert layer.max_seq_len == 1024
        assert layer.global_theta_base == 2_000_000.0
        assert layer.local_theta_base == 5_000.0

    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        # Test negative or zero head_dim
        with pytest.raises(ValueError, match="head_dim must be positive"):
            DualRotaryPositionEmbedding(head_dim=-64, max_seq_len=256)

        with pytest.raises(ValueError, match="head_dim must be positive"):
            DualRotaryPositionEmbedding(head_dim=0, max_seq_len=256)

        # Test odd head_dim (must be even for RoPE)
        with pytest.raises(ValueError, match="head_dim must be even for RoPE"):
            DualRotaryPositionEmbedding(head_dim=63, max_seq_len=256)

        # Test negative or zero max_seq_len
        with pytest.raises(ValueError, match="max_seq_len must be positive"):
            DualRotaryPositionEmbedding(head_dim=64, max_seq_len=-256)

        with pytest.raises(ValueError, match="max_seq_len must be positive"):
            DualRotaryPositionEmbedding(head_dim=64, max_seq_len=0)

        # Test negative or zero global_theta_base
        with pytest.raises(ValueError, match="global_theta_base must be positive"):
            DualRotaryPositionEmbedding(head_dim=64, max_seq_len=256, global_theta_base=-1000.0)

        with pytest.raises(ValueError, match="global_theta_base must be positive"):
            DualRotaryPositionEmbedding(head_dim=64, max_seq_len=256, global_theta_base=0.0)

        # Test negative or zero local_theta_base
        with pytest.raises(ValueError, match="local_theta_base must be positive"):
            DualRotaryPositionEmbedding(head_dim=64, max_seq_len=256, local_theta_base=-1000.0)

        with pytest.raises(ValueError, match="local_theta_base must be positive"):
            DualRotaryPositionEmbedding(head_dim=64, max_seq_len=256, local_theta_base=0.0)

    def test_build_process(self, input_tensor, layer_instance):
        """Test that the layer builds properly."""
        # Trigger build through forward pass
        output = layer_instance(input_tensor, rope_type='global')

        # Check that layer was built
        assert layer_instance.built is True
        assert len(layer_instance.weights) > 0

        # Check that both global and local caches exist
        assert hasattr(layer_instance, "cos_global_cached")
        assert hasattr(layer_instance, "sin_global_cached")
        assert hasattr(layer_instance, "cos_local_cached")
        assert hasattr(layer_instance, "sin_local_cached")
        assert layer_instance.cos_global_cached is not None
        assert layer_instance.sin_global_cached is not None
        assert layer_instance.cos_local_cached is not None
        assert layer_instance.sin_local_cached is not None

        # Check cached weights shapes
        expected_freq_dim = layer_instance.head_dim
        expected_cache_shape = (layer_instance.max_seq_len, expected_freq_dim)
        assert layer_instance.cos_global_cached.shape == expected_cache_shape
        assert layer_instance.sin_global_cached.shape == expected_cache_shape
        assert layer_instance.cos_local_cached.shape == expected_cache_shape
        assert layer_instance.sin_local_cached.shape == expected_cache_shape

        # Check that weights are non-trainable
        assert not layer_instance.cos_global_cached.trainable
        assert not layer_instance.sin_global_cached.trainable
        assert not layer_instance.cos_local_cached.trainable
        assert not layer_instance.sin_local_cached.trainable

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
            {"head_dim": 32, "max_seq_len": 256},
            {"head_dim": 64, "max_seq_len": 512},
            {"head_dim": 128, "max_seq_len": 1024},
        ]

        for config in configs_to_test:
            layer = DualRotaryPositionEmbedding(**config)

            # Create appropriate input tensor
            head_dim = config["head_dim"]
            test_input = keras.random.normal([2, 4, 100, head_dim])

            # Test both rope types
            for rope_type in ['global', 'local']:
                output = layer(test_input, rope_type=rope_type)

                # Check output shape matches input shape (RoPE doesn't change dimensions)
                assert output.shape == test_input.shape

            # Test compute_output_shape separately
            computed_shape = layer.compute_output_shape(test_input.shape)
            assert computed_shape == test_input.shape

    def test_forward_pass_global_rope(self, input_tensor, layer_instance):
        """Test forward pass with global RoPE."""
        output = layer_instance(input_tensor, rope_type='global')

        # Basic sanity checks
        assert not np.any(np.isnan(output.numpy()))
        assert not np.any(np.isinf(output.numpy()))

        # Check output shape
        assert output.shape == input_tensor.shape

        # Test with training=False
        output_inference = layer_instance(input_tensor, rope_type='global', training=False)
        assert output_inference.shape == input_tensor.shape
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output),
            keras.ops.convert_to_numpy(output_inference),
            rtol=1e-6, atol=1e-6,
            err_msg="Global RoPE outputs should match between training modes"
        )

        # Test with training=True
        output_training = layer_instance(input_tensor, rope_type='global', training=True)
        assert output_training.shape == input_tensor.shape
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output),
            keras.ops.convert_to_numpy(output_training),
            rtol=1e-6, atol=1e-6,
            err_msg="Global RoPE outputs should match between training modes"
        )

    def test_forward_pass_local_rope(self, input_tensor, layer_instance):
        """Test forward pass with local RoPE."""
        output = layer_instance(input_tensor, rope_type='local')

        # Basic sanity checks
        assert not np.any(np.isnan(output.numpy()))
        assert not np.any(np.isinf(output.numpy()))

        # Check output shape
        assert output.shape == input_tensor.shape

        # Test with training=False
        output_inference = layer_instance(input_tensor, rope_type='local', training=False)
        assert output_inference.shape == input_tensor.shape
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output),
            keras.ops.convert_to_numpy(output_inference),
            rtol=1e-6, atol=1e-6,
            err_msg="Local RoPE outputs should match between training modes"
        )

    def test_rope_type_differences(self, input_tensor, layer_instance):
        """Test that global and local RoPE produce different outputs."""
        output_global = layer_instance(input_tensor, rope_type='global')
        output_local = layer_instance(input_tensor, rope_type='local')

        # Outputs should be different due to different theta_base values
        assert not np.allclose(output_global.numpy(), output_local.numpy(), rtol=1e-3)

        # But shapes should be the same
        assert output_global.shape == output_local.shape == input_tensor.shape

    def test_invalid_rope_type(self, input_tensor, layer_instance):
        """Test that invalid rope_type raises error."""
        with pytest.raises(ValueError, match="rope_type must be 'global' or 'local'"):
            layer_instance(input_tensor, rope_type='invalid')

        with pytest.raises(ValueError, match="rope_type must be 'global' or 'local'"):
            layer_instance(input_tensor, rope_type='both')

    def test_sequence_length_limits(self, layer_instance):
        """Test sequence length validation."""
        # Build the layer first
        layer_instance.build((None, 8, 128, 64))

        # Test with sequence length exceeding max_seq_len
        long_input = keras.random.normal([2, 8, 600, 64])  # seq_len=600 > max_seq_len=512

        with pytest.raises(ValueError, match="Input sequence length .* exceeds max_seq_len"):
            layer_instance(long_input, rope_type='global')

        with pytest.raises(ValueError, match="Input sequence length .* exceeds max_seq_len"):
            layer_instance(long_input, rope_type='local')

        # Test with valid sequence length
        valid_input = keras.random.normal([2, 8, 400, 64])  # seq_len=400 < max_seq_len=512

        output_global = layer_instance(valid_input, rope_type='global')
        assert output_global.shape == valid_input.shape

        output_local = layer_instance(valid_input, rope_type='local')
        assert output_local.shape == valid_input.shape

    def test_different_theta_bases(self):
        """Test layer with different theta_base configurations."""
        configurations = [
            {"global_theta_base": 500_000.0, "local_theta_base": 5_000.0},
            {"global_theta_base": 2_000_000.0, "local_theta_base": 20_000.0},
            {"global_theta_base": 100_000.0, "local_theta_base": 1_000.0},
        ]

        head_dim = 64
        test_input = keras.random.normal([2, 4, 100, head_dim])

        for config in configurations:
            layer = DualRotaryPositionEmbedding(
                head_dim=head_dim,
                max_seq_len=256,
                **config
            )

            # Test both rope types
            for rope_type in ['global', 'local']:
                output = layer(test_input, rope_type=rope_type)

                # Check output is valid
                assert not np.any(np.isnan(output.numpy()))
                assert output.shape == test_input.shape

    def test_different_configurations(self):
        """Test layer with different configurations."""
        configurations = [
            {"head_dim": 32, "max_seq_len": 128, "global_theta_base": 500_000.0},
            {"head_dim": 64, "max_seq_len": 512, "local_theta_base": 5_000.0},
            {"head_dim": 128, "max_seq_len": 1024, "global_theta_base": 2_000_000.0, "local_theta_base": 20_000.0},
        ]

        for config in configurations:
            layer = DualRotaryPositionEmbedding(**config)

            # Create appropriate input
            head_dim = config["head_dim"]
            test_input = keras.random.normal([2, 4, 50, head_dim])

            # Test both rope types
            for rope_type in ['global', 'local']:
                output = layer(test_input, rope_type=rope_type)

                # Check output is valid
                assert not np.any(np.isnan(output.numpy()))
                assert output.shape == test_input.shape

    def test_rotary_transformation_properties(self):
        """Test mathematical properties of rotary transformation."""
        layer = DualRotaryPositionEmbedding(head_dim=64, max_seq_len=256)

        # Create deterministic input for testing
        test_input = keras.ops.ones([1, 1, 10, 64])

        # Test both rope types
        for rope_type in ['global', 'local']:
            output = layer(test_input, rope_type=rope_type)

            # Basic checks
            assert not np.any(np.isnan(output.numpy()))
            assert output.shape == test_input.shape

            # Rotations should preserve magnitude (approximately due to numerical precision)
            input_norms = keras.ops.sqrt(keras.ops.sum(keras.ops.square(test_input), axis=-1))
            output_norms = keras.ops.sqrt(keras.ops.sum(keras.ops.square(output), axis=-1))

            # Allow small numerical differences
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(input_norms),
                keras.ops.convert_to_numpy(output_norms),
                rtol=1e-5, atol=1e-6,
                err_msg=f"Rotation should preserve magnitude for {rope_type} RoPE"
            )

    def test_position_dependency(self):
        """Test that output depends on position."""
        layer = DualRotaryPositionEmbedding(head_dim=64, max_seq_len=256)

        # Create identical vectors at different positions
        batch_size, num_heads, seq_len, head_dim = 1, 1, 5, 64
        test_input = keras.ops.ones([batch_size, num_heads, seq_len, head_dim])

        # Test both rope types
        for rope_type in ['global', 'local']:
            output = layer(test_input, rope_type=rope_type)

            # Different positions should produce different outputs
            pos_0 = output[0, 0, 0, :]
            pos_1 = output[0, 0, 1, :]
            pos_2 = output[0, 0, 2, :]

            # Positions should be different (not all close)
            assert not np.allclose(pos_0.numpy(), pos_1.numpy(),
                                   rtol=1e-3), f"Positions 0 and 1 should differ for {rope_type} RoPE"
            assert not np.allclose(pos_1.numpy(), pos_2.numpy(),
                                   rtol=1e-3), f"Positions 1 and 2 should differ for {rope_type} RoPE"

    def test_global_vs_local_frequency_differences(self):
        """Test that global and local RoPE have different frequency characteristics."""
        layer = DualRotaryPositionEmbedding(head_dim=64, max_seq_len=256)

        # Build the layer
        layer.build((None, 8, 100, 64))

        # Compare the cached cos/sin values
        global_cos = layer.cos_global_cached.numpy()
        local_cos = layer.cos_local_cached.numpy()
        global_sin = layer.sin_global_cached.numpy()
        local_sin = layer.sin_local_cached.numpy()

        # They should be different due to different theta_base values
        assert not np.allclose(global_cos, local_cos, rtol=1e-3)
        assert not np.allclose(global_sin, local_sin, rtol=1e-3)

        # But they should have the same shape and range
        assert global_cos.shape == local_cos.shape
        assert global_sin.shape == local_sin.shape
        assert np.all(global_cos >= -1.0) and np.all(global_cos <= 1.0)
        assert np.all(local_cos >= -1.0) and np.all(local_cos <= 1.0)
        assert np.all(global_sin >= -1.0) and np.all(global_sin <= 1.0)
        assert np.all(local_sin >= -1.0) and np.all(local_sin <= 1.0)

    def test_serialization(self):
        """Test serialization and deserialization of the layer."""
        original_layer = DualRotaryPositionEmbedding(
            head_dim=128,
            max_seq_len=1024,
            global_theta_base=2_000_000.0,
            local_theta_base=5_000.0,
        )

        # Build the layer
        input_shape = (None, 8, 256, 128)
        original_layer.build(input_shape)

        # Get configs
        config = original_layer.get_config()
        build_config = original_layer.get_build_config()

        # Recreate the layer
        recreated_layer = DualRotaryPositionEmbedding.from_config(config)
        recreated_layer.build_from_config(build_config)

        # Check configuration matches
        assert recreated_layer.head_dim == original_layer.head_dim
        assert recreated_layer.max_seq_len == original_layer.max_seq_len
        assert recreated_layer.global_theta_base == original_layer.global_theta_base
        assert recreated_layer.local_theta_base == original_layer.local_theta_base

        # Check weights match (shapes and values)
        assert len(recreated_layer.weights) == len(original_layer.weights)
        for w1, w2 in zip(original_layer.weights, recreated_layer.weights):
            assert w1.shape == w2.shape
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(w1),
                keras.ops.convert_to_numpy(w2),
                rtol=1e-6, atol=1e-6,
                err_msg="Recreated weights should match original weights"
            )

    def test_model_integration(self, input_tensor):
        """Test the layer in a model context."""
        # Create a simple model with the dual RoPE layer
        inputs = keras.Input(shape=input_tensor.shape[1:])
        x = DualRotaryPositionEmbedding(head_dim=64, max_seq_len=512)(inputs, rope_type='global')
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
        """Test saving and loading a model with the dual RoPE layer."""
        # Create a model with the dual RoPE layer
        inputs = keras.Input(shape=input_tensor.shape[1:])
        x = DualRotaryPositionEmbedding(head_dim=64, max_seq_len=512, name="dual_rope_layer")(inputs, rope_type='local')
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
                    "DualRotaryPositionEmbedding": DualRotaryPositionEmbedding
                }
            )

            # Generate prediction with loaded model
            loaded_prediction = loaded_model.predict(input_tensor, verbose=0)

            # Check predictions match
            np.testing.assert_allclose(
                original_prediction,
                loaded_prediction,
                rtol=1e-5, atol=1e-6,
                err_msg="Loaded model predictions should match original predictions"
            )

            # Check layer type is preserved
            assert isinstance(loaded_model.get_layer("dual_rope_layer"), DualRotaryPositionEmbedding)

    def test_numerical_stability(self):
        """Test layer stability with extreme input values."""
        layer = DualRotaryPositionEmbedding(head_dim=32, max_seq_len=128)

        # Create inputs with different magnitudes
        batch_size, num_heads, seq_len, head_dim = 2, 4, 50, 32

        test_cases = [
            keras.ops.zeros((batch_size, num_heads, seq_len, head_dim)),  # Zeros
            keras.ops.ones((batch_size, num_heads, seq_len, head_dim)) * 1e-10,  # Very small values
            keras.ops.ones((batch_size, num_heads, seq_len, head_dim)) * 1e5,  # Large values
            keras.random.normal((batch_size, num_heads, seq_len, head_dim)) * 100  # Large random values
        ]

        for rope_type in ['global', 'local']:
            for test_input in test_cases:
                output = layer(test_input, rope_type=rope_type)

                # Check for NaN/Inf values
                assert not np.any(np.isnan(output.numpy())), f"NaN values detected in output for {rope_type} RoPE"
                assert not np.any(np.isinf(output.numpy())), f"Inf values detected in output for {rope_type} RoPE"

    def test_cache_building_both_types(self):
        """Test that cos/sin caches are built correctly for both global and local."""
        layer = DualRotaryPositionEmbedding(
            head_dim=64,
            max_seq_len=128,
            global_theta_base=1_000_000.0,
            local_theta_base=10_000.0
        )

        # Build the layer
        layer.build((None, 8, 100, 64))

        # Check cache shapes
        freq_dim = layer.head_dim
        expected_shape = (128, freq_dim)

        # Check global caches
        assert layer.cos_global_cached.shape == expected_shape
        assert layer.sin_global_cached.shape == expected_shape

        # Check local caches
        assert layer.cos_local_cached.shape == expected_shape
        assert layer.sin_local_cached.shape == expected_shape

        # Check that cached values are reasonable for both types
        for cache_type, cos_cache, sin_cache in [
            ('global', layer.cos_global_cached, layer.sin_global_cached),
            ('local', layer.cos_local_cached, layer.sin_local_cached)
        ]:
            cos_values = cos_cache.numpy()
            sin_values = sin_cache.numpy()

            # All values should be in [-1, 1] range
            assert np.all(cos_values >= -1.0) and np.all(cos_values <= 1.0), f"{cache_type} cos values out of range"
            assert np.all(sin_values >= -1.0) and np.all(sin_values <= 1.0), f"{cache_type} sin values out of range"

            # Check that cos^2 + sin^2 ≈ 1 (trigonometric identity)
            squared_sum = cos_values ** 2 + sin_values ** 2
            np.testing.assert_allclose(
                squared_sum,
                np.ones_like(squared_sum),
                rtol=1e-6, atol=1e-6,
                err_msg=f"cos^2 + sin^2 should equal 1 for {cache_type} cache"
            )

    def test_theta_base_impact_on_frequencies(self):
        """Test that different theta_base values produce different frequency patterns."""
        layer = DualRotaryPositionEmbedding(
            head_dim=64,
            max_seq_len=64,  # Smaller for easier analysis
            global_theta_base=1_000_000.0,  # Higher theta_base -> lower frequencies
            local_theta_base=10_000.0  # Lower theta_base -> higher frequencies
        )

        layer.build((None, 8, 50, 64))

        # Get cached values for first few positions
        global_cos_first_10 = layer.cos_global_cached[:10].numpy()
        local_cos_first_10 = layer.cos_local_cached[:10].numpy()

        # Global RoPE (higher theta_base) should have slower frequency changes
        # Local RoPE (lower theta_base) should have faster frequency changes

        # Check that the patterns are different
        assert not np.allclose(global_cos_first_10, local_cos_first_10, rtol=1e-2)

        # Global should have smaller changes between adjacent positions (slower frequencies)
        global_diff = np.abs(np.diff(global_cos_first_10, axis=0))
        local_diff = np.abs(np.diff(local_cos_first_10, axis=0))

        # Local should generally have larger changes (faster frequencies) for most frequency components
        # This might not hold for all components, but should hold on average
        assert np.mean(local_diff) > np.mean(global_diff), "Local RoPE should have faster frequency changes"

    def test_rope_type_parameter_validation(self):
        """Test that rope_type parameter is properly validated and used."""
        layer = DualRotaryPositionEmbedding(head_dim=32, max_seq_len=64)
        test_input = keras.random.normal([1, 2, 30, 32])

        # Valid rope_types should work
        for rope_type in ['global', 'local']:
            output = layer(test_input, rope_type=rope_type)
            assert output.shape == test_input.shape

        # Invalid rope_types should raise errors
        invalid_types = ['Global', 'Local', 'GLOBAL', 'LOCAL', 'both', 'neither', '']
        for invalid_type in invalid_types:
            with pytest.raises(ValueError, match="rope_type must be 'global' or 'local'"):
                layer(test_input, rope_type=invalid_type)

    def test_consistency_across_calls(self):
        """Test that multiple calls with same inputs produce identical outputs."""
        layer = DualRotaryPositionEmbedding(head_dim=64, max_seq_len=256)
        test_input = keras.random.normal([2, 4, 100, 64])

        for rope_type in ['global', 'local']:
            # Multiple calls should produce identical results
            output1 = layer(test_input, rope_type=rope_type)
            output2 = layer(test_input, rope_type=rope_type)
            output3 = layer(test_input, rope_type=rope_type)

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(output1),
                keras.ops.convert_to_numpy(output2),
                rtol=1e-6, atol=1e-6,
                err_msg=f"Multiple calls should be consistent for {rope_type} RoPE"
            )

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(output1),
                keras.ops.convert_to_numpy(output3),
                rtol=1e-6, atol=1e-6,
                err_msg=f"Multiple calls should be consistent for {rope_type} RoPE"
            )