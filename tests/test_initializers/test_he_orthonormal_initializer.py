"""
Tests for HeOrthonormalInitializer

This module contains comprehensive tests for the HeOrthonormalInitializer class:
- Initialization with default and custom parameters
- He normal followed by orthonormal vector generation and mathematical correctness
- Edge cases and numerical stability
- Serialization and deserialization
- Model integration scenarios
- Error handling for invalid parameters

Tests cover proper orthogonality and normalization, reproducibility, and integration with Keras layers.
"""

import pytest
import numpy as np
import keras
from keras import ops
import tempfile
import os
from typing import Tuple

from dl_techniques.utils.logger import logger
from dl_techniques.utils.tensors import validate_orthonormality
from dl_techniques.initializers.he_orthonormal_initializer import HeOrthonormalInitializer


class TestHeOrthonormalInitializer:
    """Test suite for HeOrthonormalInitializer implementation."""

    @pytest.fixture
    def small_shape(self) -> Tuple[int, int]:
        """Create a small shape for testing.

        Returns:
            Tuple[int, int]: Small shape (5, 10) for basic tests.
        """
        return (5, 10)

    @pytest.fixture
    def square_shape(self) -> Tuple[int, int]:
        """Create a square shape for testing.

        Returns:
            Tuple[int, int]: Square shape (8, 8) for edge case tests.
        """
        return (8, 8)

    @pytest.fixture
    def large_shape(self) -> Tuple[int, int]:
        """Create a large shape for performance testing.

        Returns:
            Tuple[int, int]: Large shape (50, 100) for performance tests.
        """
        return (50, 100)

    @pytest.fixture
    def basic_initializer(self) -> HeOrthonormalInitializer:
        """Create a basic initializer with fixed seed.

        Returns:
            HeOrthonormalInitializer: Initializer with seed=42.
        """
        return HeOrthonormalInitializer(seed=42)

    def test_initialization_defaults(self) -> None:
        """Test initialization with default parameters."""
        initializer = HeOrthonormalInitializer()

        assert initializer.seed is None
        assert hasattr(initializer, '_validate_seed')
        assert hasattr(initializer, '_validate_shape')
        assert hasattr(initializer, '_he_normal')
        assert isinstance(initializer._he_normal, keras.initializers.HeNormal)

    def test_initialization_custom(self) -> None:
        """Test initialization with custom parameters."""
        initializer = HeOrthonormalInitializer(seed=123)

        assert initializer.seed == 123
        assert initializer._he_normal.seed == 123

    def test_invalid_seed_parameters(self) -> None:
        """Test that invalid seed parameters raise appropriate errors."""
        # Test negative seed
        with pytest.raises(ValueError, match="Seed must be non-negative"):
            HeOrthonormalInitializer(seed=-1)

        # Test non-integer seed
        with pytest.raises(ValueError, match="Seed must be an integer"):
            HeOrthonormalInitializer(seed=3.14)

        # Test string seed
        with pytest.raises(ValueError, match="Seed must be an integer"):
            HeOrthonormalInitializer(seed="42")

    def test_invalid_shape_parameters(self, basic_initializer: HeOrthonormalInitializer) -> None:
        """Test that invalid shapes raise appropriate errors."""
        # Test 1D shape
        with pytest.raises(ValueError, match="requires a 2D shape"):
            basic_initializer((10,))

        # Test 3D shape
        with pytest.raises(ValueError, match="requires a 2D shape"):
            basic_initializer((5, 10, 3))

        # Test n_clusters > feature_dims
        with pytest.raises(ValueError, match="Cannot create .* orthogonal vectors"):
            basic_initializer((10, 5))

        # Test zero dimensions
        with pytest.raises(ValueError, match="Shape dimensions must be positive"):
            basic_initializer((0, 10))

        with pytest.raises(ValueError, match="Shape dimensions must be positive"):
            basic_initializer((5, 0))

        # Test negative dimensions
        with pytest.raises(ValueError, match="Shape dimensions must be positive"):
            basic_initializer((-5, 10))

    def test_basic_he_orthonormal_generation(
        self,
        basic_initializer: HeOrthonormalInitializer,
        small_shape: Tuple[int, int]
    ) -> None:
        """Test basic He orthonormal vector generation."""
        vectors = basic_initializer(small_shape)

        # Check shape
        assert ops.shape(vectors)[0] == small_shape[0]
        assert ops.shape(vectors)[1] == small_shape[1]

        # Check that vectors are orthonormal
        assert validate_orthonormality(vectors)

        # Check for NaN or Inf values
        assert not ops.any(ops.isnan(vectors))
        assert not ops.any(ops.isinf(vectors))

    def test_orthogonality_property(
        self,
        basic_initializer: HeOrthonormalInitializer,
        small_shape: Tuple[int, int]
    ) -> None:
        """Test that generated vectors are orthogonal."""
        vectors = basic_initializer(small_shape)

        # Compute Gram matrix (dot products between all pairs)
        gram_matrix = ops.matmul(vectors, ops.transpose(vectors))

        # Off-diagonal elements should be close to zero
        n_vectors = small_shape[0]
        for i in range(n_vectors):
            for j in range(n_vectors):
                if i != j:
                    dot_product = gram_matrix[i, j]
                    assert abs(float(ops.convert_to_numpy(dot_product))) < 1e-5

    def test_normalization_property(
        self,
        basic_initializer: HeOrthonormalInitializer,
        small_shape: Tuple[int, int]
    ) -> None:
        """Test that generated vectors are normalized (unit length)."""
        vectors = basic_initializer(small_shape)

        # Compute norms of each vector
        norms = ops.sqrt(ops.sum(ops.square(vectors), axis=1))

        # All norms should be close to 1.0
        for i in range(small_shape[0]):
            norm_value = float(ops.convert_to_numpy(norms[i]))
            assert abs(norm_value - 1.0) < 1e-5

    def test_seed_reproducibility(self, small_shape: Tuple[int, int]) -> None:
        """Test that same seed produces same results."""
        initializer1 = HeOrthonormalInitializer(seed=42)
        initializer2 = HeOrthonormalInitializer(seed=42)

        vectors1 = initializer1(small_shape)
        vectors2 = initializer2(small_shape)

        # Should be identical
        assert ops.all(ops.isclose(vectors1, vectors2, atol=1e-7))

    def test_different_seeds_different_results(self, small_shape: Tuple[int, int]) -> None:
        """Test that different seeds produce different results."""
        initializer1 = HeOrthonormalInitializer(seed=42)
        initializer2 = HeOrthonormalInitializer(seed=123)

        vectors1 = initializer1(small_shape)
        vectors2 = initializer2(small_shape)

        # Should be different
        assert not ops.all(ops.isclose(vectors1, vectors2, atol=1e-3))

    def test_square_matrix_case(
        self,
        basic_initializer: HeOrthonormalInitializer,
        square_shape: Tuple[int, int]
    ) -> None:
        """Test with square matrices (n_clusters == feature_dims)."""
        vectors = basic_initializer(square_shape)

        # Check shape
        assert ops.shape(vectors)[0] == square_shape[0]
        assert ops.shape(vectors)[1] == square_shape[1]

        # Should still be orthonormal
        assert validate_orthonormality(vectors)

        # For square case, this should be a complete orthogonal basis
        # The determinant of the matrix should be ±1
        det = ops.linalg.det(vectors)
        det_value = abs(float(ops.convert_to_numpy(det)))
        assert abs(det_value - 1.0) < 1e-5

    def test_single_vector_case(self, basic_initializer: HeOrthonormalInitializer) -> None:
        """Test with single vector (n_clusters = 1)."""
        shape = (1, 10)
        vector = basic_initializer(shape)

        # Check shape
        assert ops.shape(vector)[0] == 1
        assert ops.shape(vector)[1] == 10

        # Should be normalized
        norm = ops.sqrt(ops.sum(ops.square(vector)))
        norm_value = float(ops.convert_to_numpy(norm))
        assert abs(norm_value - 1.0) < 1e-5

    def test_dtype_handling(self, basic_initializer: HeOrthonormalInitializer) -> None:
        """Test different dtype specifications."""
        shape = (3, 5)

        # Test float32
        vectors_f32 = basic_initializer(shape, dtype="float32")
        assert vectors_f32.dtype == "float32"

        # Test float64
        vectors_f64 = basic_initializer(shape, dtype="float64")
        assert vectors_f64.dtype == "float64"

        # Both should be orthonormal
        assert validate_orthonormality(vectors_f32)
        assert validate_orthonormality(vectors_f64)

    def test_serialization(self) -> None:
        """Test serialization and deserialization of the initializer."""
        original_initializer = HeOrthonormalInitializer(seed=456)

        # Get config and recreate
        config = original_initializer.get_config()
        recreated_initializer = HeOrthonormalInitializer.from_config(config)

        # Check configuration matches
        assert recreated_initializer.seed == original_initializer.seed

        # Check they produce same results
        shape = (4, 8)
        original_vectors = original_initializer(shape)
        recreated_vectors = recreated_initializer(shape)

        assert ops.all(ops.isclose(original_vectors, recreated_vectors, atol=1e-7))

    def test_serialization_no_seed(self) -> None:
        """Test serialization with no seed specified."""
        original_initializer = HeOrthonormalInitializer()

        # Get config and recreate
        config = original_initializer.get_config()
        recreated_initializer = HeOrthonormalInitializer.from_config(config)

        # Check configuration matches
        assert recreated_initializer.seed is None

    def test_layer_integration(self, small_shape: Tuple[int, int]) -> None:
        """Test the initializer in a layer context."""
        # small_shape is (5, 10) - this means 5 orthonormal vectors in 10D space
        # For Dense layer, kernel shape is (input_dim, output_dim)
        # If we want the initializer to work with shape (5, 10), we need:
        # input_dim=5, output_dim=10, so kernel shape = (5, 10)
        n_clusters, feature_dims = small_shape  # (5, 10)
        input_dim = n_clusters    # 5
        output_dim = feature_dims # 10

        initializer = HeOrthonormalInitializer(seed=42)

        layer = keras.layers.Dense(
            units=output_dim,  # 10 units
            kernel_initializer=initializer,
            input_shape=(input_dim,)  # 5 inputs
        )

        # Build the layer
        layer.build((None, input_dim))

        # Check that weights are orthonormal
        weights = layer.get_weights()[0]  # kernel weights shape: (5, 10)
        weights_tensor = ops.convert_to_tensor(weights)

        # The kernel shape is (5, 10), which matches our small_shape
        # The initializer creates 5 orthonormal vectors in 10D space
        # So the rows should be orthonormal
        assert validate_orthonormality(weights_tensor)

    def test_model_integration(self) -> None:
        """Test the initializer in a model context."""
        # Create a model with He orthonormal initialization
        # Use the corrected model structure with valid dimensions
        model = keras.Sequential([
            keras.layers.Dense(
                units=32,  # 32 outputs from 16 inputs: creates 16 vectors in 32D space (valid: 16 <= 32)
                activation="relu",
                kernel_initializer=HeOrthonormalInitializer(seed=42),
                input_shape=(16,)
            ),
            keras.layers.Dense(
                units=32,  # 32 outputs from 32 inputs: creates 32 vectors in 32D space (valid: 32 <= 32)
                activation="relu",
                kernel_initializer=HeOrthonormalInitializer(seed=123)
            ),
            keras.layers.Dense(
                units=1,
                activation="sigmoid"
            )
        ])

        # Compile the model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.01),
            loss="binary_crossentropy"
        )

        # Generate dummy data
        x_train = np.random.random((100, 16))  # Match the input shape
        y_train = np.random.choice([0, 1], size=(100, 1))

        # Train for a few steps
        history = model.fit(x_train, y_train, epochs=2, batch_size=16, verbose=0)

        # Check that training completed without errors
        assert len(history.history['loss']) == 2
        assert not any(np.isnan(loss) for loss in history.history['loss'])

    def test_model_save_load_with_initializer(self) -> None:
        """Test saving and loading a model with the initializer."""
        # Create a simple model with the initializer
        # kernel shape = (input_dim, output_dim)
        # initializer creates input_dim vectors in output_dim space
        # So we need input_dim <= output_dim
        model = keras.Sequential([
            keras.layers.Dense(
                units=16,  # 16 outputs from 8 inputs: creates 8 vectors in 16D space (valid: 8 <= 16)
                activation="relu",
                kernel_initializer=HeOrthonormalInitializer(seed=42),
                input_shape=(8,)
            ),
            keras.layers.Dense(units=1, activation="sigmoid")
        ])

        # Compile the model
        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss="binary_crossentropy"
        )

        # Generate test data
        x_test = np.random.random((10, 8))  # Match the input shape
        original_predictions = model.predict(x_test, verbose=0)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "he_orthonormal_model.keras")

            # Save the model
            model.save(model_path)

            # Load the model with custom objects
            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={"HeOrthonormalInitializer": HeOrthonormalInitializer}
            )

            # Generate predictions with loaded model
            loaded_predictions = loaded_model.predict(x_test, verbose=0)

            # Predictions should match
            np.testing.assert_allclose(
                original_predictions,
                loaded_predictions,
                rtol=1e-5
            )

            # Check that initializer is preserved
            loaded_initializer = loaded_model.layers[0].kernel_initializer
            assert isinstance(loaded_initializer, HeOrthonormalInitializer)
            assert loaded_initializer.seed == 42

            logger.info("Model save/load test with HeOrthonormalInitializer passed successfully")

    def test_string_representation(self) -> None:
        """Test string representation of the initializer."""
        # Test with seed
        initializer1 = HeOrthonormalInitializer(seed=42)
        repr1 = repr(initializer1)
        assert "HeOrthonormalInitializer" in repr1
        assert "seed=42" in repr1

        str1 = str(initializer1)
        assert "HeOrthonormalInitializer" in str1
        assert "seed=42" in str1

        # Test without seed
        initializer2 = HeOrthonormalInitializer()
        repr2 = repr(initializer2)
        assert "HeOrthonormalInitializer" in repr2
        assert "seed=None" in repr2

    def test_comparison_with_he_normal(self) -> None:
        """Test comparison between HeOrthonormalInitializer and HeNormal."""
        shape = (8, 16)
        seed = 42

        # Create both initializers
        he_ortho_init = HeOrthonormalInitializer(seed=seed)
        he_normal_init = keras.initializers.HeNormal(seed=seed)

        # Generate vectors
        he_ortho_vectors = he_ortho_init(shape)
        he_normal_vectors = he_normal_init(shape)

        # He orthonormal should be orthogonal, He normal should not
        assert validate_orthonormality(he_ortho_vectors)
        assert not validate_orthonormality(he_normal_vectors)

    def test_mathematical_properties(self, basic_initializer: HeOrthonormalInitializer) -> None:
        """Test deeper mathematical properties of generated vectors."""
        shape = (6, 10)
        vectors = basic_initializer(shape)

        # Test that the vectors span a 6-dimensional subspace
        # by checking that the rank of the matrix is 6
        # Note: This is implicitly tested by orthonormality, but we make it explicit

        # Gram matrix should have rank equal to number of vectors
        gram_matrix = ops.matmul(vectors, ops.transpose(vectors))

        # For orthonormal vectors, Gram matrix should be identity
        identity = ops.eye(shape[0])
        assert ops.all(ops.isclose(gram_matrix, identity, atol=1e-5))

        # Test linear independence: no vector should be a linear combination of others
        # This is guaranteed by orthogonality, but we can test by checking
        # that removing any vector changes the span
        for i in range(shape[0]):
            # Create matrix without i-th vector
            indices = [j for j in range(shape[0]) if j != i]
            reduced_vectors = ops.take(vectors, indices, axis=0)

            # The reduced set should still be orthonormal
            assert validate_orthonormality(reduced_vectors)


class TestHeOrthonormalEdgeCases:
    """Test edge cases and numerical stability for HeOrthonormalInitializer."""

    def test_very_small_dimensions(self) -> None:
        """Test with very small dimensions."""
        # Single vector in 1D space
        initializer = HeOrthonormalInitializer(seed=42)
        vectors = initializer((1, 1))

        assert ops.shape(vectors) == (1, 1)
        # Should be ±1
        value = float(ops.convert_to_numpy(vectors[0, 0]))
        assert abs(abs(value) - 1.0) < 1e-7

    def test_boundary_case_equal_dimensions(self) -> None:
        """Test boundary case where n_clusters == feature_dims."""
        shapes_to_test = [(2, 2), (3, 3), (5, 5), (10, 10)]

        for shape in shapes_to_test:
            initializer = HeOrthonormalInitializer(seed=42)
            vectors = initializer(shape)

            assert validate_orthonormality(vectors)

            # For square orthonormal matrices, determinant should be ±1
            det = ops.linalg.det(vectors)
            det_value = abs(float(ops.convert_to_numpy(det)))
            assert abs(det_value - 1.0) < 1e-5

    def test_numerical_stability_large_dimensions(self) -> None:
        """Test numerical stability with large dimensions."""
        # Test with reasonably large dimensions
        shape = (20, 50)
        initializer = HeOrthonormalInitializer(seed=42)
        vectors = initializer(shape)

        # Should still maintain orthonormality
        assert validate_orthonormality(vectors, rtol=1e-4, atol=1e-6)

        # Check for numerical issues
        assert not ops.any(ops.isnan(vectors))
        assert not ops.any(ops.isinf(vectors))

    def test_different_aspect_ratios(self) -> None:
        """Test various aspect ratios of n_clusters to feature_dims."""
        test_cases = [
            (1, 100),   # Very wide
            (2, 50),    # Wide
            (10, 20),   # Moderate
            (15, 16),   # Almost square
        ]

        for n_clusters, feature_dims in test_cases:
            initializer = HeOrthonormalInitializer(seed=42)
            vectors = initializer((n_clusters, feature_dims))

            assert validate_orthonormality(vectors)
            assert not ops.any(ops.isnan(vectors))
            assert not ops.any(ops.isinf(vectors))

    def test_reproducibility_across_calls(self) -> None:
        """Test that multiple calls with same seed are reproducible."""
        shape = (5, 10)
        seed = 42

        # Create multiple initializers with same seed
        results = []
        for _ in range(5):
            initializer = HeOrthonormalInitializer(seed=seed)
            vectors = initializer(shape)
            results.append(vectors)

        # All results should be identical
        for i in range(1, len(results)):
            assert ops.all(ops.isclose(results[0], results[i], atol=1e-7))

    def test_validate_orthonormality_edge_cases(self) -> None:
        """Test the validate_orthonormality function with edge cases."""
        # Test with single vector
        single_vector = ops.cast([[1.0, 0.0, 0.0]], dtype="float32")
        assert validate_orthonormality(single_vector)

        # Test with non-normalized vector
        non_normalized = ops.cast([[2.0, 0.0]], dtype="float32")
        assert not validate_orthonormality(non_normalized)

        # Test with orthogonal but non-normalized vectors
        orthogonal_not_normalized = ops.cast([
            [2.0, 0.0],
            [0.0, 3.0]
        ], dtype="float32")
        assert not validate_orthonormality(orthogonal_not_normalized)

        # Test with normalized but non-orthogonal vectors
        normalized_not_orthogonal = ops.cast([
            [1.0, 0.0],
            [0.5, np.sqrt(0.75)]
        ], dtype="float32")
        assert not validate_orthonormality(normalized_not_orthogonal)

    def test_error_handling_robustness(self) -> None:
        """Test robustness of error handling."""
        initializer = HeOrthonormalInitializer(seed=42)

        # Test with various invalid inputs
        invalid_shapes = [
            (0, 5),      # Zero dimension
            (5, 0),      # Zero dimension
            (-1, 5),     # Negative dimension
            (5, -1),     # Negative dimension
            (10, 5),     # n_clusters > feature_dims
        ]

        for shape in invalid_shapes:
            with pytest.raises(ValueError):
                initializer(shape)

    def test_memory_efficiency(self) -> None:
        """Test that the initializer doesn't use excessive memory."""
        # This test ensures that the implementation doesn't create
        # unnecessarily large intermediate tensors

        # Create a moderately large tensor
        shape = (100, 200)
        initializer = HeOrthonormalInitializer(seed=42)

        # This should complete without memory issues
        vectors = initializer(shape)

        # Verify correctness
        assert validate_orthonormality(vectors, rtol=1e-4, atol=1e-6)
        assert vectors.shape == shape

    def test_performance_consistency(self) -> None:
        """Test that performance is consistent across different shapes."""
        import time

        # Test various shapes and ensure reasonable performance
        test_shapes = [
            (10, 20),
            (20, 40),
            (30, 60),
        ]

        initializer = HeOrthonormalInitializer(seed=42)

        for shape in test_shapes:
            start_time = time.time()
            vectors = initializer(shape)
            end_time = time.time()

            # Should complete in reasonable time (less than 1 second for these sizes)
            assert (end_time - start_time) < 1.0

            # Should still be correct
            assert validate_orthonormality(vectors)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])