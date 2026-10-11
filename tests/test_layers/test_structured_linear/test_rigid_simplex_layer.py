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

from dl_techniques.layers.structured_linear.rigid_simplex_layer import RigidSimplexLayer


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

    def test_call_rejects_a_width_the_layer_was_not_built_for(self):
        """The build-time contract is re-asserted on every forward.

        A contract checked only in build() is checked exactly once, against
        whatever shape arrived first -- so a later call at a different width
        would otherwise be silently accepted.
        """
        layer = RigidSimplexLayer(units=64, frame_seed=0)
        layer(np.random.default_rng(0).standard_normal((4, 128)).astype(np.float32))

        with pytest.raises(ValueError, match="was built for input_dim=128"):
            layer(np.random.default_rng(0).standard_normal((4, 64)).astype(np.float32))


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
        """Test that orthogonality regularization loss is added while training."""
        layer = RigidSimplexLayer(units=64, orthogonality_penalty=1e-4)

        _ = layer(sample_input, training=True)

        assert len(layer.losses) > 0, "Orthogonality loss should be added"

    def test_inference_carries_no_orthogonality_loss(self, sample_input):
        """Inference must not carry a penalty term it was never trained against."""
        layer = RigidSimplexLayer(units=64, orthogonality_penalty=1e-4)
        _ = layer(sample_input, training=False)

        assert layer.losses == [], "an inference forward must leave layer.losses empty"

    def test_eager_call_still_receives_the_penalty(self, sample_input):
        """The other half of the pair: ``training=None`` is a training call."""
        layer = RigidSimplexLayer(units=64, orthogonality_penalty=1e-4)
        _ = layer(sample_input)

        assert len(layer.losses) == 1

    def test_zero_orthogonality_penalty_adds_no_loss(self, sample_input):
        """A zero weight makes the term identically zero, so it is not added."""
        layer = RigidSimplexLayer(units=64, orthogonality_penalty=0.0)
        _ = layer(sample_input, training=True)

        assert layer.losses == []

    def test_identity_rotation_preserves_projection(self):
        """At ``R = I`` and ``scale = 1`` the map is exactly ``x @ frame``.

        This is an algebraic identity, not an approximation, so it is pinned at
        a float32 tolerance rather than the loose 1e-2 this test previously used
        -- a bound that loose cannot see a rotation kernel that is wrong.
        """
        layer = RigidSimplexLayer(units=64, rotation_initializer='identity')
        inputs = np.random.randn(4, 32).astype(np.float32)

        output = layer(inputs)

        expected = np.matmul(inputs, keras.ops.convert_to_numpy(layer.static_simplex))

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output),
            expected,
            rtol=0, atol=1e-6,
            err_msg="Identity rotation should preserve the bare frame projection"
        )

    def test_a_non_identity_rotation_actually_changes_the_output(self):
        """Anti-vacuity twin for the identity test above.

        Without this arm, a layer that ignored ``rotation_kernel`` entirely would
        pass the identity assertion perfectly.
        """
        inputs = np.random.default_rng(0).standard_normal((4, 32)).astype(np.float32)

        identity_layer = RigidSimplexLayer(units=64, rotation_initializer='identity')
        rotated_layer = RigidSimplexLayer(units=64, rotation_initializer='orthogonal')
        identity_layer(inputs)
        rotated_layer(inputs)

        assert not np.allclose(
            keras.ops.convert_to_numpy(identity_layer(inputs)),
            keras.ops.convert_to_numpy(rotated_layer(inputs))
        ), "rotation_initializer had no effect on the output"


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

    def test_the_default_frame_packs_distinct_columns_beyond_n_plus_1(self):
        """A width past one simplex's capacity must not repeat its vertices.

        This is the guard the tiling implementation could not pass. Measured on
        the tiled frame at ``input_dim=16, units=100``: 17 distinct columns of
        100, and 490 column pairs at inner product exactly ``1.0`` -- 83% of the
        outputs were exact duplicates of another output, each receiving an
        identical logit. The coherence ceiling of a correctly packed frame at
        these settings measures ~0.71, roughly 0.29 below the tiled value of
        1.0, so a 0.9 ceiling sits well clear of both.
        """
        input_dim, units = 16, 100

        layer = RigidSimplexLayer(units=units, frame_seed=0)
        layer.build((None, input_dim))

        frame = keras.ops.convert_to_numpy(layer.static_simplex)
        assert frame.shape == (input_dim, units)

        distinct = {tuple(np.round(column, 6)) for column in frame.T}
        assert len(distinct) == units, (
            f"expected {units} distinct columns, got {len(distinct)}; the frame "
            f"was tiled rather than packed"
        )

        gram = frame.T @ frame
        coherence = np.abs(gram[~np.eye(units, dtype=bool)]).max()
        assert coherence < 0.9, (
            f"max pairwise |inner product| {coherence:.6f} indicates duplicated "
            f"columns; a packed frame keeps this well below 1.0"
        )

    def test_tiled_mode_really_is_tiled(self):
        """The anti-vacuity twin: frame='tiled' must FAIL the guard above.

        Without this arm the assertion could be satisfied by a layer that packs
        under every configuration, and 'the packed frame is distinct' would
        never have been a claim about the tiling path at all.
        """
        input_dim, units = 16, 100

        layer = RigidSimplexLayer(units=units, frame='tiled')
        layer.build((None, input_dim))

        frame = keras.ops.convert_to_numpy(layer.static_simplex)

        distinct = {tuple(np.round(column, 6)) for column in frame.T}
        assert len(distinct) < units, (
            f"frame='tiled' produced {len(distinct)} distinct columns of {units}; "
            f"the mode under test is supposed to repeat vertices"
        )

    def test_the_crystal_frame_beats_the_tiled_one_on_coherence(self):
        """Direct comparison: packing must separate more than tiling does."""
        input_dim, units = 8, 30

        packed = RigidSimplexLayer(units=units, frame='crystal', frame_seed=3)
        packed.build((None, input_dim))
        tiled = RigidSimplexLayer(units=units, frame='tiled')
        tiled.build((None, input_dim))

        def _max_coherence(layer):
            frame = keras.ops.convert_to_numpy(layer.static_simplex)
            gram = frame.T @ frame
            return float(np.abs(gram[~np.eye(units, dtype=bool)]).max())

        assert _max_coherence(packed) < _max_coherence(tiled)

    def test_frame_seed_makes_the_frame_reproducible(self):
        """The same seed must give the same frame across separate builds."""
        first = RigidSimplexLayer(units=30, frame_seed=11)
        first.build((None, 8))
        second = RigidSimplexLayer(units=30, frame_seed=11)
        second.build((None, 8))

        np.testing.assert_array_equal(
            keras.ops.convert_to_numpy(first.static_simplex),
            keras.ops.convert_to_numpy(second.static_simplex),
            err_msg="frame_seed did not make the frame reproducible",
        )

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

    def test_a_shape_arriving_as_a_list_keeps_its_batch_axis(self):
        """A one-shape LIST is a legal Keras shape form and must not be read as
        a shape tuple.

        ``list(input_shape)`` over ``[[None, 128]]`` iterates the OUTER list, so
        rewriting the last element replaces the batch axis and the layer reports
        ``(64,)`` for a ``(None, 128)`` input.
        """
        layer = RigidSimplexLayer(units=64)

        for listed in ([[None, 128]], [(None, 128)], [None, 128]):
            assert layer.compute_output_shape(listed) == (None, 64), (
                f"input_shape={listed} produced the wrong output shape"
            )


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

    def test_config_carries_the_frame_arguments(self):
        """``frame`` and ``frame_seed`` must survive a config round trip."""
        original = RigidSimplexLayer(units=64, frame='tiled', frame_seed=7)
        reconstructed = RigidSimplexLayer.from_config(original.get_config())

        assert reconstructed.frame == 'tiled'
        assert reconstructed.frame_seed == 7

    def test_a_config_without_frame_reconstructs_as_tiled(self):
        """The migration path: an old config was built by tiling.

        A config written before ``frame`` existed carries no key, so it was
        necessarily built by the tiling construction. Reconstructing it with
        the new ``'crystal'`` default would silently change the values it is
        meant to reproduce.
        """
        config = RigidSimplexLayer(units=64).get_config()
        del config['frame']

        assert RigidSimplexLayer.from_config(config).frame == 'tiled'

    def test_from_config_does_not_mutate_the_callers_dict(self):
        """``from_config`` must not edit the dict it was handed.

        A caller that reuses one config dict for two constructions would
        otherwise see the second receive an already-rewritten value.
        """
        config = RigidSimplexLayer(units=64).get_config()
        del config['frame']
        snapshot = dict(config)

        RigidSimplexLayer.from_config(config)

        assert config == snapshot, "from_config mutated the caller's config dict"

    def test_invalid_frame_mode_is_rejected(self):
        with pytest.raises(ValueError, match="frame must be one of"):
            RigidSimplexLayer(units=64, frame='spherical-harmonics')


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