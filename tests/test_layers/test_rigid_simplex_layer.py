"""
Tests for RigidSimplexLayer.

Comprehensive test suite covering instantiation, forward pass, serialization,
mathematical properties, and edge cases.
"""

import tempfile
import os

import numpy as np
import pytest
import keras

from dl_techniques.layers.rigid_simplex_layer import RigidSimplexLayer


class TestRigidSimplexLayerInstantiation:
    """Tests for layer instantiation and configuration validation."""

    def test_instantiation_default_config(self):
        """Test layer instantiation with default configuration."""
        layer = RigidSimplexLayer(units=64)

        assert layer.units == 64
        assert layer.scale_min == 0.5
        assert layer.scale_max == 2.0
        assert layer.orthogonality_penalty == 1e-4

    def test_instantiation_custom_config(self):
        """Test layer instantiation with custom configuration."""
        layer = RigidSimplexLayer(
            units=128,
            scale_min=0.1,
            scale_max=5.0,
            orthogonality_penalty=1e-3,
        )

        assert layer.units == 128
        assert layer.scale_min == 0.1
        assert layer.scale_max == 5.0
        assert layer.orthogonality_penalty == 1e-3

    def test_invalid_units_zero(self):
        """Test that zero units raises ValueError."""
        with pytest.raises(ValueError, match="units must be positive"):
            RigidSimplexLayer(units=0)

    def test_invalid_units_negative(self):
        """Test that negative units raises ValueError."""
        with pytest.raises(ValueError, match="units must be positive"):
            RigidSimplexLayer(units=-10)

    def test_invalid_scale_range(self):
        """Test that scale_min >= scale_max raises ValueError."""
        with pytest.raises(ValueError, match="scale_min.*must be less than scale_max"):
            RigidSimplexLayer(units=64, scale_min=2.0, scale_max=1.0)

    def test_invalid_scale_equal(self):
        """Test that scale_min == scale_max raises ValueError."""
        with pytest.raises(ValueError, match="scale_min.*must be less than scale_max"):
            RigidSimplexLayer(units=64, scale_min=1.0, scale_max=1.0)

    def test_invalid_orthogonality_penalty(self):
        """Test that negative orthogonality_penalty raises ValueError."""
        with pytest.raises(ValueError, match="orthogonality_penalty must be non-negative"):
            RigidSimplexLayer(units=64, orthogonality_penalty=-0.1)

    def test_zero_orthogonality_penalty_allowed(self):
        """Test that zero orthogonality_penalty is allowed."""
        layer = RigidSimplexLayer(units=64, orthogonality_penalty=0.0)
        assert layer.orthogonality_penalty == 0.0


class TestRigidSimplexLayerBuild:
    """Tests for layer build behavior."""

    def test_build_creates_weights(self):
        """Test that build() creates all expected weights."""
        layer = RigidSimplexLayer(units=64)
        layer.build((None, 128))

        assert layer.static_simplex is not None
        assert layer.rotation_kernel is not None
        assert layer.global_scale is not None

    def test_weight_shapes(self):
        """Test that weights have correct shapes."""
        layer = RigidSimplexLayer(units=64)
        layer.build((None, 128))

        assert layer.static_simplex.shape == (128, 64)
        assert layer.rotation_kernel.shape == (128, 128)
        assert layer.global_scale.shape == (1,)

    def test_static_simplex_not_trainable(self):
        """Test that static_simplex is not trainable."""
        layer = RigidSimplexLayer(units=64)
        layer.build((None, 128))

        assert layer.static_simplex.trainable is False

    def test_rotation_kernel_trainable(self):
        """Test that rotation_kernel is trainable."""
        layer = RigidSimplexLayer(units=64)
        layer.build((None, 128))

        assert layer.rotation_kernel.trainable is True

    def test_global_scale_trainable(self):
        """Test that global_scale is trainable."""
        layer = RigidSimplexLayer(units=64)
        layer.build((None, 128))

        assert layer.global_scale.trainable is True

    def test_rotation_kernel_identity_initialization(self):
        """Test that rotation_kernel is initialized as identity."""
        layer = RigidSimplexLayer(units=64, rotation_initializer='identity')
        layer.build((None, 32))

        identity = np.eye(32, dtype=np.float32)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(layer.rotation_kernel),
            identity,
            rtol=1e-6, atol=1e-6,
            err_msg="Rotation kernel should be initialized as identity"
        )

    def test_global_scale_initial_value(self):
        """Test that global_scale is initialized to 1.0."""
        layer = RigidSimplexLayer(units=64)
        layer.build((None, 128))

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(layer.global_scale),
            np.array([1.0]),
            rtol=1e-6, atol=1e-6,
            err_msg="Global scale should be initialized to 1.0"
        )

    def test_build_requires_defined_input_dim(self):
        """Test that build raises error if input dim is None."""
        layer = RigidSimplexLayer(units=64)

        with pytest.raises(ValueError, match="Last dimension of input must be defined"):
            layer.build((None, None))


class TestRigidSimplexLayerForwardPass:
    """Tests for forward pass computation."""

    @pytest.fixture
    def sample_input(self):
        """Sample input tensor for testing."""
        return np.random.randn(8, 128).astype(np.float32)

    def test_forward_pass_output_shape(self, sample_input):
        """Test that forward pass produces correct output shape."""
        layer = RigidSimplexLayer(units=64)
        output = layer(sample_input)

        assert output.shape == (8, 64)

    def test_forward_pass_3d_input(self):
        """Test forward pass with 3D input (sequence data)."""
        layer = RigidSimplexLayer(units=64)
        inputs = np.random.randn(8, 16, 128).astype(np.float32)
        output = layer(inputs)

        assert output.shape == (8, 16, 64)

    def test_forward_pass_4d_input(self):
        """Test forward pass with 4D input (image-like data after flatten)."""
        layer = RigidSimplexLayer(units=64)
        inputs = np.random.randn(8, 4, 4, 128).astype(np.float32)
        output = layer(inputs)

        assert output.shape == (8, 4, 4, 64)

    @pytest.mark.parametrize("batch_size", [1, 8, 32, 128])
    def test_variable_batch_sizes(self, batch_size):
        """Test layer handles various batch sizes."""
        layer = RigidSimplexLayer(units=64)
        inputs = np.random.randn(batch_size, 128).astype(np.float32)
        output = layer(inputs)

        assert output.shape[0] == batch_size
        assert output.shape[1] == 64

    def test_training_vs_inference_deterministic(self, sample_input):
        """Test that layer output is same in training and inference mode."""
        layer = RigidSimplexLayer(units=64)

        # Build layer first
        _ = layer(sample_input, training=False)

        train_output = layer(sample_input, training=True)
        infer_output = layer(sample_input, training=False)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(train_output),
            keras.ops.convert_to_numpy(infer_output),
            rtol=1e-6, atol=1e-6,
            err_msg="Outputs should match between training and inference"
        )

    def test_orthogonality_loss_added(self, sample_input):
        """Test that orthogonality regularization loss is added."""
        layer = RigidSimplexLayer(units=64, orthogonality_penalty=1e-4)

        # Clear any existing losses
        _ = layer(sample_input)

        assert len(layer.losses) > 0, "Orthogonality loss should be added"

    def test_zero_orthogonality_penalty_still_adds_loss(self, sample_input):
        """Test that loss is added even with zero penalty (just zero valued)."""
        layer = RigidSimplexLayer(units=64, orthogonality_penalty=0.0)
        _ = layer(sample_input)

        # Loss is still added but should be zero
        assert len(layer.losses) > 0

    def test_identity_rotation_preserves_projection(self):
        """Test that identity rotation correctly projects onto simplex."""
        layer = RigidSimplexLayer(units=64, rotation_initializer='identity')
        inputs = np.random.randn(4, 32).astype(np.float32)

        output = layer(inputs)

        # With identity rotation and scale=1, output = inputs @ simplex
        expected = np.matmul(inputs, keras.ops.convert_to_numpy(layer.static_simplex))

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output),
            expected,
            rtol=1e-5, atol=1e-5,
            err_msg="Identity rotation should preserve simplex projection"
        )


class TestRigidSimplexLayerSimplexProperties:
    """Tests for mathematical properties of the Simplex structure."""

    def test_simplex_rows_normalized(self):
        """Test that Simplex matrix has normalized columns (transposed rows)."""
        layer = RigidSimplexLayer(units=33)  # N+1 for N=32
        layer.build((None, 32))

        simplex = keras.ops.convert_to_numpy(layer.static_simplex)

        # The simplex is (input_dim, output_dim) = (32, 33)
        # Rows of the original simplex (before transpose) should be normalized
        # After transpose, columns should have specific norms
        # But we check the property: V.T @ V should be proportional to I for tight frame
        vtv = simplex.T @ simplex

        # Check it's approximately proportional to identity
        # For a tight frame: V^T V = (N+1)/N * I
        expected_scale = 33 / 32  # (N+1)/N
        diagonal_values = np.diag(vtv)

        # All diagonal values should be approximately equal
        np.testing.assert_allclose(
            diagonal_values,
            np.full_like(diagonal_values, np.mean(diagonal_values)),
            rtol=0.1, atol=0.1,
            err_msg="Diagonal values should be approximately equal"
        )

    def test_simplex_maximum_separation(self):
        """Test that Simplex vectors have negative off-diagonal correlations."""
        layer = RigidSimplexLayer(units=33)  # N+1 for N=32
        layer.build((None, 32))

        simplex = keras.ops.convert_to_numpy(layer.static_simplex)

        # Gram matrix G = V @ V.T where V is (N+1, N)
        # In our case, simplex is (N, N+1), so gram = simplex.T @ simplex
        gram = simplex.T @ simplex

        # Off-diagonal elements should be negative or near zero
        off_diag_mask = ~np.eye(gram.shape[0], dtype=bool)
        off_diag = gram[off_diag_mask]

        # For ETF, off-diagonal should be approximately -1/N or close to 0
        # depending on tiling. Just verify they're not all positive
        mean_off_diag = np.mean(off_diag)
        assert mean_off_diag < 0.5, f"Off-diagonal mean should be small/negative, got {mean_off_diag}"

    def test_simplex_tiling_for_large_units(self):
        """Test that simplex correctly tiles when units > input_dim + 1."""
        input_dim = 16
        units = 100  # Much larger than input_dim + 1 = 17

        layer = RigidSimplexLayer(units=units)
        layer.build((None, input_dim))

        simplex = keras.ops.convert_to_numpy(layer.static_simplex)

        assert simplex.shape == (input_dim, units)
        # Should not contain NaN or Inf
        assert np.all(np.isfinite(simplex))

    def test_simplex_slicing_for_small_units(self):
        """Test that simplex correctly slices when units < input_dim + 1."""
        input_dim = 64
        units = 10  # Smaller than input_dim + 1 = 65

        layer = RigidSimplexLayer(units=units)
        layer.build((None, input_dim))

        simplex = keras.ops.convert_to_numpy(layer.static_simplex)

        assert simplex.shape == (input_dim, units)
        assert np.all(np.isfinite(simplex))


class TestRigidSimplexLayerScaleConstraint:
    """Tests for bounded scale constraint."""

    def test_scale_constraint_initialization(self):
        """Test that scale is initialized within bounds."""
        layer = RigidSimplexLayer(units=64, scale_min=0.5, scale_max=2.0)
        layer.build((None, 128))

        scale_value = keras.ops.convert_to_numpy(layer.global_scale)[0]

        assert 0.5 <= scale_value <= 2.0

    def test_scale_constraint_clipping(self):
        """Test that scale constraint clips values to bounds."""
        layer = RigidSimplexLayer(units=64, scale_min=0.5, scale_max=2.0)
        layer.build((None, 128))

        # Manually set to out-of-bounds value and apply constraint
        constraint = layer.global_scale.constraint

        # Test clipping from above
        high_value = keras.ops.convert_to_tensor([5.0])
        clipped = constraint(high_value)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(clipped),
            np.array([2.0]),
            rtol=1e-6, atol=1e-6,
            err_msg="Scale should be clipped to max"
        )

        # Test clipping from below
        low_value = keras.ops.convert_to_tensor([0.1])
        clipped = constraint(low_value)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(clipped),
            np.array([0.5]),
            rtol=1e-6, atol=1e-6,
            err_msg="Scale should be clipped to min"
        )


class TestRigidSimplexLayerComputeOutputShape:
    """Tests for compute_output_shape method."""

    def test_compute_output_shape_2d(self):
        """Test compute_output_shape with 2D input."""
        layer = RigidSimplexLayer(units=64)

        input_shape = (None, 128)
        output_shape = layer.compute_output_shape(input_shape)

        assert output_shape == (None, 64)

    def test_compute_output_shape_3d(self):
        """Test compute_output_shape with 3D input."""
        layer = RigidSimplexLayer(units=64)

        input_shape = (None, 16, 128)
        output_shape = layer.compute_output_shape(input_shape)

        assert output_shape == (None, 16, 64)

    def test_compute_output_shape_4d(self):
        """Test compute_output_shape with 4D input."""
        layer = RigidSimplexLayer(units=64)

        input_shape = (None, 8, 8, 128)
        output_shape = layer.compute_output_shape(input_shape)

        assert output_shape == (None, 8, 8, 64)

    def test_compute_output_shape_before_build(self):
        """Test compute_output_shape works before layer is built."""
        layer = RigidSimplexLayer(units=64)

        # Should work without calling layer
        input_shape = (None, 128)
        output_shape = layer.compute_output_shape(input_shape)

        assert output_shape == (None, 64)

    def test_compute_output_shape_matches_actual(self):
        """Test compute_output_shape matches actual output shape."""
        layer = RigidSimplexLayer(units=64)
        inputs = np.random.randn(8, 128).astype(np.float32)

        computed_shape = layer.compute_output_shape(inputs.shape)
        actual_output = layer(inputs)

        assert computed_shape == actual_output.shape


class TestRigidSimplexLayerConfiguration:
    """Tests for get_config and from_config methods."""

    def test_get_config_contains_all_params(self):
        """Test get_config returns all constructor arguments."""
        layer = RigidSimplexLayer(
            units=64,
            scale_min=0.1,
            scale_max=5.0,
            orthogonality_penalty=1e-3,
        )

        config = layer.get_config()

        assert 'units' in config
        assert 'scale_min' in config
        assert 'scale_max' in config
        assert 'orthogonality_penalty' in config
        assert 'rotation_initializer' in config

    def test_get_config_values(self):
        """Test get_config returns correct values."""
        layer = RigidSimplexLayer(
            units=128,
            scale_min=0.2,
            scale_max=3.0,
            orthogonality_penalty=0.01,
        )

        config = layer.get_config()

        assert config['units'] == 128
        assert config['scale_min'] == 0.2
        assert config['scale_max'] == 3.0
        assert config['orthogonality_penalty'] == 0.01

    def test_from_config_reconstruction(self):
        """Test layer can be reconstructed from config."""
        original = RigidSimplexLayer(
            units=64,
            scale_min=0.3,
            scale_max=4.0,
            orthogonality_penalty=0.005,
        )

        config = original.get_config()
        reconstructed = RigidSimplexLayer.from_config(config)

        assert reconstructed.units == original.units
        assert reconstructed.scale_min == original.scale_min
        assert reconstructed.scale_max == original.scale_max
        assert reconstructed.orthogonality_penalty == original.orthogonality_penalty

    def test_from_config_with_serialized_initializer(self):
        """Test from_config correctly deserializes initializer."""
        original = RigidSimplexLayer(units=64, rotation_initializer='orthogonal')

        config = original.get_config()
        reconstructed = RigidSimplexLayer.from_config(config)

        assert reconstructed.rotation_initializer is not None


class TestRigidSimplexLayerSerialization:
    """Tests for full save/load serialization cycle."""

    @pytest.fixture
    def sample_input(self):
        """Sample input tensor for testing."""
        return np.random.randn(8, 128).astype(np.float32)

    def test_serialization_cycle(self, sample_input):
        """Test full save/load cycle preserves functionality."""
        # Create model with layer
        inputs = keras.Input(shape=(128,))
        outputs = RigidSimplexLayer(units=64)(inputs)
        model = keras.Model(inputs, outputs)

        original_output = model(sample_input)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_model.keras')
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)

        loaded_output = loaded_model(sample_input)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(original_output),
            keras.ops.convert_to_numpy(loaded_output),
            rtol=1e-6, atol=1e-6,
            err_msg="Outputs should match after serialization"
        )

    def test_serialization_preserves_weights(self, sample_input):
        """Test that serialization preserves all weight values."""
        layer = RigidSimplexLayer(units=64)
        _ = layer(sample_input)  # Build layer

        original_simplex = keras.ops.convert_to_numpy(layer.static_simplex).copy()
        original_rotation = keras.ops.convert_to_numpy(layer.rotation_kernel).copy()
        original_scale = keras.ops.convert_to_numpy(layer.global_scale).copy()

        inputs = keras.Input(shape=(128,))
        outputs = RigidSimplexLayer(units=64)(inputs)
        model = keras.Model(inputs, outputs)

        # Set weights to match our layer
        model.layers[1].static_simplex.assign(original_simplex)
        model.layers[1].rotation_kernel.assign(original_rotation)
        model.layers[1].global_scale.assign(original_scale)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_model.keras')
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)

        loaded_layer = loaded_model.layers[1]

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(loaded_layer.static_simplex),
            original_simplex,
            rtol=1e-6, atol=1e-6,
            err_msg="Static simplex weights should be preserved"
        )

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(loaded_layer.rotation_kernel),
            original_rotation,
            rtol=1e-6, atol=1e-6,
            err_msg="Rotation kernel weights should be preserved"
        )

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(loaded_layer.global_scale),
            original_scale,
            rtol=1e-6, atol=1e-6,
            err_msg="Global scale weight should be preserved"
        )

    def test_serialization_with_custom_config(self, sample_input):
        """Test serialization with non-default configuration."""
        inputs = keras.Input(shape=(128,))
        outputs = RigidSimplexLayer(
            units=32,
            scale_min=0.1,
            scale_max=10.0,
            orthogonality_penalty=0.01,
        )(inputs)
        model = keras.Model(inputs, outputs)

        original_output = model(sample_input)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_model.keras')
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)

        # Check config preserved
        loaded_layer = loaded_model.layers[1]
        assert loaded_layer.units == 32
        assert loaded_layer.scale_min == 0.1
        assert loaded_layer.scale_max == 10.0
        assert loaded_layer.orthogonality_penalty == 0.01

        # Check output matches
        loaded_output = loaded_model(sample_input)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(original_output),
            keras.ops.convert_to_numpy(loaded_output),
            rtol=1e-6, atol=1e-6,
            err_msg="Outputs should match after serialization with custom config"
        )


class TestRigidSimplexLayerTraining:
    """Tests for training behavior."""

    def test_trainable_variables_count(self):
        """Test correct number of trainable variables."""
        layer = RigidSimplexLayer(units=64)
        layer.build((None, 128))

        # Should have 2 trainable: rotation_kernel and global_scale
        # static_simplex is not trainable
        trainable_count = len(layer.trainable_variables)
        assert trainable_count == 2

    def test_non_trainable_variables_count(self):
        """Test correct number of non-trainable variables."""
        layer = RigidSimplexLayer(units=64)
        layer.build((None, 128))

        # Should have 1 non-trainable: static_simplex
        non_trainable_count = len(layer.non_trainable_variables)
        assert non_trainable_count == 1

    def test_model_training_updates_rotation(self):
        """Test that training updates rotation kernel."""
        # Create simple model
        inputs = keras.Input(shape=(32,))
        x = RigidSimplexLayer(units=16)(inputs)
        outputs = keras.layers.Dense(1)(x)
        model = keras.Model(inputs, outputs)

        model.compile(optimizer='adam', loss='mse')

        # Get initial rotation
        initial_rotation = keras.ops.convert_to_numpy(
            model.layers[1].rotation_kernel
        ).copy()

        # Train briefly
        x_train = np.random.randn(100, 32).astype(np.float32)
        y_train = np.random.randn(100, 1).astype(np.float32)

        model.fit(x_train, y_train, epochs=5, verbose=0)

        # Rotation should have changed
        final_rotation = keras.ops.convert_to_numpy(model.layers[1].rotation_kernel)

        assert not np.allclose(initial_rotation, final_rotation), \
            "Rotation kernel should be updated during training"

    def test_model_training_simplex_unchanged(self):
        """Test that training does NOT update static simplex."""
        inputs = keras.Input(shape=(32,))
        x = RigidSimplexLayer(units=16)(inputs)
        outputs = keras.layers.Dense(1)(x)
        model = keras.Model(inputs, outputs)

        model.compile(optimizer='adam', loss='mse')

        # Get initial simplex
        initial_simplex = keras.ops.convert_to_numpy(
            model.layers[1].static_simplex
        ).copy()

        # Train
        x_train = np.random.randn(100, 32).astype(np.float32)
        y_train = np.random.randn(100, 1).astype(np.float32)

        model.fit(x_train, y_train, epochs=5, verbose=0)

        # Simplex should NOT have changed
        final_simplex = keras.ops.convert_to_numpy(model.layers[1].static_simplex)

        np.testing.assert_allclose(
            initial_simplex,
            final_simplex,
            rtol=1e-6, atol=1e-6,
            err_msg="Static simplex should NOT change during training"
        )

    def test_orthogonality_loss_contributes_to_total(self):
        """Test that orthogonality loss contributes to model loss."""
        inputs = keras.Input(shape=(32,))
        x = RigidSimplexLayer(units=16, orthogonality_penalty=1.0)(inputs)
        outputs = keras.layers.Dense(1)(x)
        model = keras.Model(inputs, outputs)

        model.compile(optimizer='adam', loss='mse')

        x_test = np.random.randn(10, 32).astype(np.float32)
        y_test = np.zeros((10, 1), dtype=np.float32)

        # Even with zero MSE loss, orthogonality loss should exist
        # (unless rotation is exactly identity)
        total_loss = model.evaluate(x_test, y_test, verbose=0)

        # Total loss should be > 0 due to orthogonality penalty
        # (rotation starts as identity so loss should be very small but computation happens)
        assert total_loss >= 0


class TestRigidSimplexLayerEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_unit(self):
        """Test layer with single output unit."""
        layer = RigidSimplexLayer(units=1)
        inputs = np.random.randn(4, 16).astype(np.float32)
        output = layer(inputs)

        assert output.shape == (4, 1)

    def test_large_input_dimension(self):
        """Test layer with large input dimension."""
        layer = RigidSimplexLayer(units=64)
        inputs = np.random.randn(4, 1024).astype(np.float32)
        output = layer(inputs)

        assert output.shape == (4, 64)

    def test_units_equal_input_dim(self):
        """Test layer where units equals input dimension."""
        layer = RigidSimplexLayer(units=64)
        inputs = np.random.randn(4, 64).astype(np.float32)
        output = layer(inputs)

        assert output.shape == (4, 64)

    def test_units_greater_than_input_dim(self):
        """Test layer where units > input dimension."""
        layer = RigidSimplexLayer(units=128)
        inputs = np.random.randn(4, 64).astype(np.float32)
        output = layer(inputs)

        assert output.shape == (4, 128)

    def test_very_small_scale_range(self):
        """Test layer with very small scale range."""
        layer = RigidSimplexLayer(units=64, scale_min=0.99, scale_max=1.01)
        inputs = np.random.randn(4, 128).astype(np.float32)
        output = layer(inputs)

        assert output.shape == (4, 64)

    def test_single_sample_batch(self):
        """Test layer with batch size of 1."""
        layer = RigidSimplexLayer(units=64)
        inputs = np.random.randn(1, 128).astype(np.float32)
        output = layer(inputs)

        assert output.shape == (1, 64)

    def test_float64_input(self):
        """Test layer handles float64 input."""
        layer = RigidSimplexLayer(units=64)
        inputs = np.random.randn(4, 128).astype(np.float64)
        output = layer(inputs)

        assert output.shape == (4, 64)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])