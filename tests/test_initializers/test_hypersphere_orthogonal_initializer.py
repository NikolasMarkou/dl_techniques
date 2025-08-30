import pytest
import numpy as np
import tempfile
import os
import warnings
from typing import Any, Dict, Tuple

import keras
from dl_techniques.initializers.hypersphere_orthogonal_initializer import OrthogonalHypersphereInitializer


class TestOrthogonalHypersphereInitializer:
    """Comprehensive test suite for OrthogonalHypersphereInitializer."""

    @pytest.fixture
    def basic_config(self) -> Dict[str, Any]:
        """Standard configuration for testing."""
        return {
            'radius': 1.5,
            'seed': 42
        }

    @pytest.fixture
    def feasible_shape(self) -> Tuple[int, ...]:
        """Shape where orthogonality is feasible (num_vectors <= latent_dim)."""
        return (10, 128)  # 10 vectors in 128D space

    @pytest.fixture
    def infeasible_shape(self) -> Tuple[int, ...]:
        """Shape where orthogonality is infeasible (num_vectors > latent_dim)."""
        return (256, 64)  # 256 vectors in 64D space

    @pytest.fixture
    def multi_dim_shape(self) -> Tuple[int, ...]:
        """Multi-dimensional grid shape."""
        return (4, 8, 128)  # 32 vectors (4*8) in 128D space

    def test_initialization(self, basic_config):
        """Test initializer creation and parameter validation."""
        # Basic initialization
        initializer = OrthogonalHypersphereInitializer(**basic_config)

        assert initializer.radius == basic_config['radius']
        assert initializer.seed == basic_config['seed']

    def test_initialization_validation(self):
        """Test parameter validation during initialization."""
        # Valid parameters
        OrthogonalHypersphereInitializer(radius=1.0, seed=None)
        OrthogonalHypersphereInitializer(radius=2.5, seed=123)

        # Invalid radius
        with pytest.raises(ValueError, match="radius must be positive"):
            OrthogonalHypersphereInitializer(radius=0.0)

        with pytest.raises(ValueError, match="radius must be positive"):
            OrthogonalHypersphereInitializer(radius=-1.0)

    def test_call_validation(self, basic_config):
        """Test input validation in __call__ method."""
        initializer = OrthogonalHypersphereInitializer(**basic_config)

        # Valid shapes
        initializer(shape=(10, 64))
        initializer(shape=(5, 5, 32))

        # Invalid shapes
        with pytest.raises(ValueError, match="shape cannot be empty"):
            initializer(shape=())

        with pytest.raises(ValueError, match="latent dimension must be positive"):
            initializer(shape=(10, 0))

        with pytest.raises(ValueError, match="latent dimension must be positive"):
            initializer(shape=(10, -5))

    def test_feasible_orthogonal_generation(self, basic_config, feasible_shape):
        """Test orthogonal vector generation when mathematically feasible."""
        initializer = OrthogonalHypersphereInitializer(**basic_config)

        weights = initializer(shape=feasible_shape)
        weights_np = keras.ops.convert_to_numpy(weights)

        # Check shape
        assert weights.shape == feasible_shape

        # Check radius property - all vectors should have specified radius
        vector_norms = np.linalg.norm(weights_np, axis=1)
        expected_radius = basic_config['radius']
        np.testing.assert_allclose(
            vector_norms,
            np.full_like(vector_norms, expected_radius),
            rtol=1e-5, atol=1e-5,
            err_msg="All vectors should have the specified radius"
        )

        # Check orthogonality - dot products should be near zero
        num_vectors = feasible_shape[0]
        dot_products = np.dot(weights_np, weights_np.T)

        # Extract off-diagonal elements (should be near zero for orthogonal vectors)
        off_diagonal_mask = ~np.eye(num_vectors, dtype=bool)
        off_diagonal_dots = dot_products[off_diagonal_mask]

        np.testing.assert_allclose(
            off_diagonal_dots,
            np.zeros_like(off_diagonal_dots),
            rtol=1e-4, atol=1e-4,
            err_msg="Orthogonal vectors should have near-zero dot products"
        )

        # Check diagonal elements (should be radius squared)
        diagonal_dots = np.diag(dot_products)
        expected_diagonal = expected_radius ** 2
        np.testing.assert_allclose(
            diagonal_dots,
            np.full_like(diagonal_dots, expected_diagonal),
            rtol=1e-5, atol=1e-5,
            err_msg="Self dot products should equal radius squared"
        )

    def test_infeasible_fallback_with_warning(self, basic_config, infeasible_shape):
        """Test fallback to uniform hypersphere when orthogonality is impossible."""
        initializer = OrthogonalHypersphereInitializer(**basic_config)

        # Should generate a warning about impossibility
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            weights = initializer(shape=infeasible_shape)

            # Check that warning was issued
            assert len(warning_list) == 1
            warning = warning_list[0]
            assert issubclass(warning.category, UserWarning)
            assert "Orthogonality constraint violation" in str(warning.message)
            assert "256 orthogonal vectors" in str(warning.message)
            assert "64-dimensional space" in str(warning.message)

        weights_np = keras.ops.convert_to_numpy(weights)

        # Check shape
        assert weights.shape == infeasible_shape

        # Check radius property - all vectors should still have specified radius
        vector_norms = np.linalg.norm(weights_np, axis=1)
        expected_radius = basic_config['radius']
        np.testing.assert_allclose(
            vector_norms,
            np.full_like(vector_norms, expected_radius),
            rtol=1e-5, atol=1e-5,
            err_msg="All vectors should have the specified radius even in fallback mode"
        )

    def test_multi_dimensional_grids(self, basic_config, multi_dim_shape):
        """Test handling of multi-dimensional weight grids."""
        initializer = OrthogonalHypersphereInitializer(**basic_config)

        weights = initializer(shape=multi_dim_shape)
        weights_np = keras.ops.convert_to_numpy(weights)

        # Check shape preservation
        assert weights.shape == multi_dim_shape

        # Flatten to 2D for analysis
        grid_size = multi_dim_shape[0] * multi_dim_shape[1]  # 4 * 8 = 32
        latent_dim = multi_dim_shape[2]  # 128
        weights_2d = weights_np.reshape(grid_size, latent_dim)

        # Check radius property
        vector_norms = np.linalg.norm(weights_2d, axis=1)
        expected_radius = basic_config['radius']
        np.testing.assert_allclose(
            vector_norms,
            np.full_like(vector_norms, expected_radius),
            rtol=1e-5, atol=1e-5,
            err_msg="Multi-dimensional grids should preserve radius property"
        )

    def test_reproducibility_with_seed(self):
        """Test that same seed produces identical results."""
        shape = (8, 64)
        seed = 12345
        radius = 2.0

        # Generate weights twice with same seed
        init1 = OrthogonalHypersphereInitializer(radius=radius, seed=seed)
        init2 = OrthogonalHypersphereInitializer(radius=radius, seed=seed)

        weights1 = init1(shape=shape)
        weights2 = init2(shape=shape)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(weights1),
            keras.ops.convert_to_numpy(weights2),
            rtol=1e-10, atol=1e-10,
            err_msg="Same seed should produce identical results"
        )

        # Different seed should produce different results
        init3 = OrthogonalHypersphereInitializer(radius=radius, seed=seed + 1)
        weights3 = init3(shape=shape)

        with pytest.raises(AssertionError):
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(weights1),
                keras.ops.convert_to_numpy(weights3),
                rtol=1e-5, atol=1e-5
            )

    def test_serialization_cycle(self, basic_config, feasible_shape):
        """CRITICAL TEST: Full serialization cycle with model."""
        # Create a layer using the initializer
        layer = keras.layers.Dense(
            units=feasible_shape[1],
            kernel_initializer=OrthogonalHypersphereInitializer(**basic_config)
        )

        # Create model
        inputs = keras.Input(shape=feasible_shape[1:])
        outputs = layer(inputs)
        model = keras.Model(inputs, outputs)

        # Get original weights
        sample_input = keras.random.normal(shape=(4,) + feasible_shape[1:])
        original_weights = model.get_weights()
        original_prediction = model(sample_input)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_weights = loaded_model.get_weights()
            loaded_prediction = loaded_model(sample_input)

            # Verify weights are identical
            for orig, loaded in zip(original_weights, loaded_weights):
                np.testing.assert_allclose(
                    keras.ops.convert_to_numpy(orig),
                    keras.ops.convert_to_numpy(loaded),
                    rtol=1e-10, atol=1e-10,
                    err_msg="Weights should be identical after serialization"
                )

            # Verify predictions are identical
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_prediction),
                keras.ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions should match after serialization"
            )

    def test_config_completeness(self, basic_config):
        """Test that get_config contains all __init__ parameters."""
        initializer = OrthogonalHypersphereInitializer(**basic_config)
        config = initializer.get_config()

        # Check all config parameters are present
        for key in basic_config:
            assert key in config, f"Missing {key} in get_config()"
            assert config[key] == basic_config[key], f"Config value mismatch for {key}"

    def test_config_reconstruction(self, basic_config, feasible_shape):
        """Test that initializer can be reconstructed from config."""
        # Create original initializer
        original_init = OrthogonalHypersphereInitializer(**basic_config)

        # Get config and reconstruct
        config = original_init.get_config()
        reconstructed_init = OrthogonalHypersphereInitializer(**config)

        # Both should produce identical results
        weights_original = original_init(shape=feasible_shape)
        weights_reconstructed = reconstructed_init(shape=feasible_shape)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(weights_original),
            keras.ops.convert_to_numpy(weights_reconstructed),
            rtol=1e-10, atol=1e-10,
            err_msg="Reconstructed initializer should produce identical results"
        )

    @pytest.mark.parametrize("radius", [0.5, 1.0, 2.0, 10.0])
    def test_different_radii(self, radius):
        """Test initialization with different radius values."""
        shape = (5, 32)
        initializer = OrthogonalHypersphereInitializer(radius=radius, seed=42)

        weights = initializer(shape=shape)
        weights_np = keras.ops.convert_to_numpy(weights)

        # Check that all vectors have the specified radius
        vector_norms = np.linalg.norm(weights_np, axis=1)
        np.testing.assert_allclose(
            vector_norms,
            np.full_like(vector_norms, radius),
            rtol=1e-5, atol=1e-5,
            err_msg=f"All vectors should have radius {radius}"
        )

    def test_edge_case_single_vector(self):
        """Test edge case with single vector."""
        shape = (1, 64)
        initializer = OrthogonalHypersphereInitializer(radius=3.0, seed=42)

        weights = initializer(shape=shape)
        weights_np = keras.ops.convert_to_numpy(weights)

        # Check radius
        vector_norm = np.linalg.norm(weights_np[0])
        np.testing.assert_allclose(
            vector_norm, 3.0, rtol=1e-5, atol=1e-5,
            err_msg="Single vector should have correct radius"
        )

    def test_edge_case_1d_latent(self):
        """Test edge case with 1D latent space."""
        shape = (3, 1)  # 3 vectors in 1D space (infeasible)
        initializer = OrthogonalHypersphereInitializer(radius=2.0, seed=42)

        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            weights = initializer(shape=shape)

            # Should warn about impossibility
            assert len(warning_list) == 1

        weights_np = keras.ops.convert_to_numpy(weights)

        # All vectors should still have correct radius
        vector_norms = np.abs(weights_np.flatten())  # 1D vectors
        np.testing.assert_allclose(
            vector_norms,
            np.full_like(vector_norms, 2.0),
            rtol=1e-5, atol=1e-5,
            err_msg="1D vectors should have correct radius"
        )

    def test_dtype_handling(self, basic_config, feasible_shape):
        """Test handling of different data types."""
        initializer = OrthogonalHypersphereInitializer(**basic_config)

        # Test default dtype
        weights_default = initializer(shape=feasible_shape)
        assert weights_default.dtype == keras.backend.floatx()

        # Test explicit dtype
        weights_float32 = initializer(shape=feasible_shape, dtype='float32')
        assert weights_float32.dtype == 'float32'

    def test_repr_method(self, basic_config):
        """Test string representation."""
        initializer = OrthogonalHypersphereInitializer(**basic_config)
        repr_str = repr(initializer)

        assert "OrthogonalHypersphereInitializer" in repr_str
        assert f"radius={basic_config['radius']}" in repr_str
        assert f"seed={basic_config['seed']}" in repr_str


# Additional integration tests
class TestOrthogonalHypersphereIntegration:
    """Integration tests with Keras layers and models."""

    def test_dense_layer_integration(self):
        """Test integration with Dense layer."""
        initializer = OrthogonalHypersphereInitializer(radius=1.5, seed=42)

        layer = keras.layers.Dense(
            units=64,
            kernel_initializer=initializer
        )

        # Build layer
        layer.build(input_shape=(None, 32))

        # Check weights have correct properties
        weights = layer.get_weights()[0]  # kernel weights, shape: (32, 64)
        # The initializer creates row vectors (along input dimension) with specified radius
        vector_norms = np.linalg.norm(weights, axis=1)  # norm of each input vector (row)

        np.testing.assert_allclose(
            vector_norms,
            np.full_like(vector_norms, 1.5),
            rtol=1e-5, atol=1e-5,
            err_msg="Dense layer weights should have correct radius"
        )

    def test_embedding_layer_integration(self):
        """Test integration with Embedding layer."""
        vocab_size = 100
        embed_dim = 128

        initializer = OrthogonalHypersphereInitializer(radius=2.0, seed=123)

        layer = keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            embeddings_initializer=initializer
        )

        # Build layer
        layer.build(input_shape=())

        # Check embedding weights
        embeddings = layer.get_weights()[0]  # shape: (vocab_size, embed_dim)
        # Each row is an embedding vector for one vocabulary item
        vector_norms = np.linalg.norm(embeddings, axis=1)

        np.testing.assert_allclose(
            vector_norms,
            np.full_like(vector_norms, 2.0),
            rtol=1e-5, atol=1e-5,
            err_msg="Embedding weights should have correct radius"
        )

    def test_model_compilation_and_training(self):
        """Test that models with this initializer can be compiled and trained."""
        # Create simple model
        model = keras.Sequential([
            keras.layers.Dense(
                64,
                activation='relu',
                kernel_initializer=OrthogonalHypersphereInitializer(radius=1.0, seed=42)
            ),
            keras.layers.Dense(10, activation='softmax')
        ])

        # Compile
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Create dummy data
        x_train = keras.random.normal(shape=(100, 32))
        y_train = np.random.randint(0, 10, size=(100,))  # Use numpy instead

        # Train for one epoch (just to verify it works)
        history = model.fit(
            x_train, y_train,
            epochs=1,
            batch_size=32,
            verbose=0
        )

        assert len(history.history['loss']) == 1
        assert isinstance(history.history['loss'][0], float)

# Run tests with: pytest test_orthogonal_hypersphere_initializer.py -v