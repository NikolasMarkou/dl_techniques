"""
Tests for OrthonormalInitializer

This module contains comprehensive tests for the OrthonormalInitializer class:
- Initialization with default and custom parameters
- Orthonormal vector generation and mathematical correctness
- Edge cases and numerical stability
- Serialization and deserialization
- Model integration scenarios
- Error handling for invalid parameters

Tests cover proper orthogonality and normalization, reproducibility, and integration with Keras layers.
"""

import pytest
import time

import numpy as np
import keras
from keras import ops
import tempfile
import os
from typing import Tuple

from dl_techniques.utils.logger import logger
from dl_techniques.utils.tensors import validate_orthonormality
from dl_techniques.initializers.orthonormal_initializer import OrthonormalInitializer
from tests.optimizer_state import build_optimizer_state


class TestOrthonormalInitializer:
    """Test suite for OrthonormalInitializer implementation."""

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
    def basic_initializer(self) -> OrthonormalInitializer:
        """Create a basic initializer with fixed seed.

        Returns:
            OrthonormalInitializer: Initializer with seed=42.
        """
        return OrthonormalInitializer(seed=42)

    def test_initialization_defaults(self) -> None:
        """Test initialization with default parameters."""
        initializer = OrthonormalInitializer()

        assert initializer.seed is None
        assert hasattr(initializer, '_validate_seed')
        assert hasattr(initializer, '_validate_shape')

    def test_initialization_custom(self) -> None:
        """Test initialization with custom parameters."""
        initializer = OrthonormalInitializer(seed=123)

        assert initializer.seed == 123

    def test_invalid_seed_parameters(self) -> None:
        """Test that invalid seed parameters raise appropriate errors."""
        # Test negative seed
        with pytest.raises(ValueError, match="Seed must be non-negative"):
            OrthonormalInitializer(seed=-1)

        # Test non-integer seed
        with pytest.raises(ValueError, match="Seed must be an integer"):
            OrthonormalInitializer(seed=3.14)

        # Test string seed
        with pytest.raises(ValueError, match="Seed must be an integer"):
            OrthonormalInitializer(seed="42")

    def test_invalid_shape_parameters(self, basic_initializer: OrthonormalInitializer) -> None:
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

    def test_basic_orthonormal_generation(
        self,
        basic_initializer: OrthonormalInitializer,
        small_shape: Tuple[int, int]
    ) -> None:
        """Test basic orthonormal vector generation."""
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
        basic_initializer: OrthonormalInitializer,
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
        basic_initializer: OrthonormalInitializer,
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
        initializer1 = OrthonormalInitializer(seed=42)
        initializer2 = OrthonormalInitializer(seed=42)

        vectors1 = initializer1(small_shape)
        vectors2 = initializer2(small_shape)

        # Should be identical
        assert ops.all(ops.isclose(vectors1, vectors2, atol=1e-7))

    def test_different_seeds_different_results(self, small_shape: Tuple[int, int]) -> None:
        """Test that different seeds produce different results."""
        initializer1 = OrthonormalInitializer(seed=42)
        initializer2 = OrthonormalInitializer(seed=123)

        vectors1 = initializer1(small_shape)
        vectors2 = initializer2(small_shape)

        # Should be different
        assert not ops.all(ops.isclose(vectors1, vectors2, atol=1e-3))

    def test_square_matrix_case(
        self,
        basic_initializer: OrthonormalInitializer,
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

    def test_single_vector_case(self, basic_initializer: OrthonormalInitializer) -> None:
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

    def test_dtype_handling(self, basic_initializer: OrthonormalInitializer) -> None:
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
        original_initializer = OrthonormalInitializer(seed=456)

        # Get config and recreate
        config = original_initializer.get_config()
        recreated_initializer = OrthonormalInitializer.from_config(config)

        # Check configuration matches
        assert recreated_initializer.seed == original_initializer.seed

        # Check they produce same results
        shape = (4, 8)
        original_vectors = original_initializer(shape)
        recreated_vectors = recreated_initializer(shape)

        assert ops.all(ops.isclose(original_vectors, recreated_vectors, atol=1e-7))

    def test_serialization_no_seed(self) -> None:
        """Test serialization with no seed specified."""
        original_initializer = OrthonormalInitializer()

        # Get config and recreate
        config = original_initializer.get_config()
        recreated_initializer = OrthonormalInitializer.from_config(config)

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

        initializer = OrthonormalInitializer(seed=42)

        layer = keras.layers.Dense(
            units=output_dim,  # 10 units
            kernel_initializer=initializer,
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
        # Create a model with orthonormal initialization
        # Use the corrected model structure with valid dimensions
        model = keras.Sequential([
            keras.layers.Dense(
                units=32,  # 32 outputs from 16 inputs: creates 16 vectors in 32D space (valid: 16 <= 32)
                activation="relu",
                kernel_initializer=OrthonormalInitializer(seed=42),
            ),
            keras.layers.Dense(
                units=32,  # 32 outputs from 32 inputs: creates 32 vectors in 32D space (valid: 32 <= 32)
                activation="relu",
                kernel_initializer=OrthonormalInitializer(seed=123)
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
                kernel_initializer=OrthonormalInitializer(seed=42),
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
            model_path = os.path.join(tmpdirname, "orthonormal_model.keras")

            # Save the model
            # The optimizer's slot variables are allocated lazily, so a compiled-but-
            # unfitted model would otherwise save an optimizer the reload cannot match.
            # See tests/optimizer_state.py (D-016).
            build_optimizer_state(model)
            model.save(model_path)

            # Load the model with custom objects
            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={"OrthonormalInitializer": OrthonormalInitializer}
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
            assert isinstance(loaded_initializer, OrthonormalInitializer)
            assert loaded_initializer.seed == 42

            logger.info("Model save/load test with OrthonormalInitializer passed successfully")

    def test_string_representation(self) -> None:
        """Test string representation of the initializer."""
        # Test with seed
        initializer1 = OrthonormalInitializer(seed=42)
        repr1 = repr(initializer1)
        assert "OrthonormalInitializer" in repr1
        assert "seed=42" in repr1

        str1 = str(initializer1)
        assert "OrthonormalInitializer" in str1
        assert "seed=42" in str1

        # Test without seed
        initializer2 = OrthonormalInitializer()
        repr2 = repr(initializer2)
        assert "OrthonormalInitializer" in repr2
        assert "seed=None" in repr2

    def test_mathematical_properties(self, basic_initializer: OrthonormalInitializer) -> None:
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


class TestEdgeCases:
    """Test edge cases and numerical stability for OrthonormalInitializer."""

    def test_very_small_dimensions(self) -> None:
        """Test with very small dimensions."""
        # Single vector in 1D space
        initializer = OrthonormalInitializer(seed=42)
        vectors = initializer((1, 1))

        assert ops.shape(vectors) == (1, 1)
        # Should be ±1
        value = float(ops.convert_to_numpy(vectors[0, 0]))
        assert abs(abs(value) - 1.0) < 1e-7

    def test_boundary_case_equal_dimensions(self) -> None:
        """Test boundary case where n_clusters == feature_dims."""
        shapes_to_test = [(2, 2), (3, 3), (5, 5), (10, 10)]

        for shape in shapes_to_test:
            initializer = OrthonormalInitializer(seed=42)
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
        initializer = OrthonormalInitializer(seed=42)
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
            initializer = OrthonormalInitializer(seed=42)
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
            initializer = OrthonormalInitializer(seed=seed)
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
        initializer = OrthonormalInitializer(seed=42)

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
        initializer = OrthonormalInitializer(seed=42)

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

        initializer = OrthonormalInitializer(seed=42)

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


class TestTheSignConventionAndThinQR:
    """Single-claim guards for the defects this file's review found.

    Measured against the previous implementation:

    * the sign convention keyed on the FIRST ROW of Q rather than diag(R),
      forcing row 0 into the positive orthant on every draw. Over 4000 seeds at
      d=64: P(any entry of row 0 < 0) = 0.000, E[entry] = +0.10010 (the
      theoretical half-normal fold sqrt(2/(pi*d)) = 0.09974), and a mean cosine
      of 0.801 to the all-ones direction. Rows 1+ were unbiased, so the whole
      distortion landed on centroid 0 of every codebook.
    * a full d x d QR was computed even when k << d: 9.08 s and a 67.1 MB buffer
      for k=64, d=4096, against 0.16 s for the thin factorization.
    * np.random.RandomState(None) bypassed keras.utils.set_random_seed.
    * np.int64 shape dims and seeds were rejected with a misleading message,
      while seed=True was accepted and behaved as seed=1.
    * half precision degraded orthonormality to 1.1e-03 (float16) or raised an
      InvalidArgumentError masked as RuntimeError (bfloat16).
    """

    def test_row_zero_is_not_folded_into_the_positive_orthant(self):
        """The headline defect: row 0 had every entry non-negative, always."""
        rows = np.stack([
            keras.ops.convert_to_numpy(OrthonormalInitializer(seed=s)((3, 64)))[0]
            for s in range(400)
        ])

        # Under the old convention this was 0.000.
        assert (rows < 0).any(axis=1).mean() > 0.95
        # Under the old convention this was +0.10010 against a target of 0.
        assert abs(float(rows.mean())) < 0.01

        cosines = rows @ np.ones(64) / np.sqrt(64)
        # Under the old convention this was 0.801.
        assert abs(float(cosines.mean())) < 0.1

    def test_the_convention_matches_keras_orthogonal(self):
        """sign(diag(R)) is the house and Keras convention, not a local choice.

        Anti-vacuity: the same statistic is computed for
        keras.initializers.Orthogonal, which must land in the same place. A test
        that only checked "close to zero" could pass on a constant tensor.
        """
        def row_zero_stats(factory):
            rows = np.stack([
                np.asarray(keras.ops.convert_to_numpy(factory(s)))[0]
                for s in range(400)
            ])
            return float(rows.mean()), float((rows < 0).any(axis=1).mean())

        mine = row_zero_stats(lambda s: OrthonormalInitializer(seed=s)((3, 64)))
        theirs = row_zero_stats(lambda s: keras.initializers.Orthogonal(seed=s)((3, 64)))

        assert abs(mine[0] - theirs[0]) < 0.02
        assert abs(mine[1] - theirs[1]) < 0.05

    def test_rows_beyond_the_first_were_always_unbiased(self):
        """The control that localizes the defect to row 0.

        Haar measure is invariant under column sign flips, so E[q_ij *
        sign(q_0j)] = 0 for i > 0 -- row 1 measured -0.00014 under the OLD
        convention too. Without this arm the test above could be read as
        evidence that the old code was broken everywhere, which it was not.
        """
        rows = np.stack([
            keras.ops.convert_to_numpy(OrthonormalInitializer(seed=s)((3, 64)))[1]
            for s in range(400)
        ])
        assert abs(float(rows.mean())) < 0.01

    def test_the_qr_is_thin(self):
        """Only the requested vectors are factorized, O(d*k^2) not O(d^3).

        A wall-clock bound is a blunt instrument, but the gap here is three
        orders of magnitude: the square factorization measured 9.08 s at
        k=64, d=4096 and the thin one 0.033 s.
        """
        start = time.time()
        vectors = OrthonormalInitializer(seed=0)((64, 4096))
        elapsed = time.time() - start

        assert tuple(vectors.shape) == (64, 4096)
        assert elapsed < 2.0, f"took {elapsed:.2f}s; a full 4096x4096 QR takes ~9s"

    def test_a_seedless_instance_honours_the_global_seed(self):
        """np.random.RandomState(None) ignored keras.utils.set_random_seed."""
        keras.utils.set_random_seed(1234)
        a = keras.ops.convert_to_numpy(OrthonormalInitializer()((4, 16)))
        keras.utils.set_random_seed(1234)
        b = keras.ops.convert_to_numpy(OrthonormalInitializer()((4, 16)))
        np.testing.assert_array_equal(a, b)

        keras.utils.set_random_seed(4321)
        c = keras.ops.convert_to_numpy(OrthonormalInitializer()((4, 16)))
        assert not np.allclose(a, c)

    def test_numpy_integer_shapes_are_accepted(self):
        """isinstance(np.int64(10), int) is False; a TensorShape carries np.int64."""
        vectors = OrthonormalInitializer(seed=0)((np.int64(4), np.int64(16)))
        assert tuple(vectors.shape) == (4, 16)

    def test_numpy_integer_seeds_are_accepted_but_bool_is_not(self):
        """seed=True passed isinstance(True, int) and silently acted as seed=1."""
        assert OrthonormalInitializer(seed=np.int64(3)).seed == 3

        with pytest.raises(ValueError, match="Seed must be an integer"):
            OrthonormalInitializer(seed=True)

    @pytest.mark.parametrize("dtype", ["float64", "float32", "float16", "bfloat16"])
    def test_half_precision_is_decomposed_in_float32(self, dtype):
        """The QR runs in float32 and the result is cast.

        bfloat16 previously raised InvalidArgumentError from TF ("Value for attr
        'T' of bfloat16 is not in the list of allowed values"), masked as
        RuntimeError; float16 "worked" at 1.1e-03 orthonormality against 1.8e-07
        for float32.
        """
        vectors = OrthonormalInitializer(seed=0)((8, 64), dtype=dtype)
        assert keras.backend.standardize_dtype(vectors.dtype) == dtype

        matrix = np.asarray(keras.ops.convert_to_numpy(vectors)).astype("float64")
        gram = matrix @ matrix.T
        np.fill_diagonal(gram, 0.0)
        # Bounded by the OUTPUT dtype's resolution, not by the decomposition's.
        tolerance = {"float64": 1e-12, "float32": 1e-6}.get(dtype, 5e-3)
        assert np.abs(gram).max() < tolerance

    @pytest.mark.parametrize("gain", [1.0, 2.0, np.sqrt(2.0)])
    def test_gain_scales_the_rows(self, gain):
        """There was no way to express the conventional ReLU gain=sqrt(2)."""
        matrix = np.asarray(keras.ops.convert_to_numpy(
            OrthonormalInitializer(gain=gain, seed=0)((4, 16))))
        np.testing.assert_allclose(
            np.linalg.norm(matrix, axis=1), gain, rtol=1e-5, atol=1e-5
        )

    @pytest.mark.parametrize("gain", [0.0, -1.0, float("nan")])
    def test_an_invalid_gain_is_rejected(self, gain):
        with pytest.raises(ValueError, match="gain must be"):
            OrthonormalInitializer(gain=gain)

    def test_gain_roundtrips_through_config(self):
        original = OrthonormalInitializer(gain=2.0, seed=5)
        restored = OrthonormalInitializer.from_config(original.get_config())

        assert restored.gain == 2.0
        np.testing.assert_array_equal(
            keras.ops.convert_to_numpy(original((4, 16))),
            keras.ops.convert_to_numpy(restored((4, 16))),
        )

    def test_the_dead_diagonal_helper_is_gone(self):
        """_extract_diagonal had no callers and its docstring was false.

        It claimed "keras.ops doesn't have a direct diagonal extraction
        function"; keras.ops.diagonal and keras.ops.diag both exist, and the
        correct sign convention needs exactly that operation.
        """
        assert not hasattr(OrthonormalInitializer(), "_extract_diagonal")
        assert hasattr(keras.ops, "diagonal")

    def test_a_bad_shape_raises_value_error_not_runtime_error(self):
        """Validation runs before any backend call and is not wrapped.

        The blanket `except Exception -> RuntimeError` turned an OOM, a
        dtype-unsupported backend error and a genuine bug into one type. The
        4-D case in particular must stay a ValueError: it is pinned by
        tests/test_layers/test_convnext_v1_block.py and its v2 twin.
        """
        for shape in [(3, 3, 64, 1), (10,), (10, 5), (0, 5), (5, -1)]:
            with pytest.raises(ValueError):
                OrthonormalInitializer(seed=0)(shape)

    def test_call_accepts_extra_kwargs(self):
        vectors = OrthonormalInitializer(seed=0)((4, 16), None, partition_shape=None)
        assert tuple(vectors.shape) == (4, 16)


class TestTheValidatorsAreShared:
    """The two orthonormal initializers must not drift apart again.

    HeOrthonormalInitializer carried a verbatim copy of both validators, error
    strings included. They now live in one module and are imported there, so a
    single edit reddens both suites.
    """

    def test_both_classes_use_the_same_validator_objects(self):
        from dl_techniques.initializers import he_orthonormal_initializer as he
        from dl_techniques.initializers import orthonormal_initializer as orth

        assert he.validate_orthonormal_seed is orth.validate_orthonormal_seed
        assert he.validate_orthonormal_shape is orth.validate_orthonormal_shape

    @pytest.mark.parametrize("bad_shape", [(10,), (2, 2, 2), (10, 5), (0, 5)])
    def test_both_classes_reject_the_same_shapes(self, bad_shape):
        from dl_techniques.initializers import HeOrthonormalInitializer

        with pytest.raises(ValueError):
            OrthonormalInitializer(seed=0)(bad_shape)
        with pytest.raises(ValueError):
            HeOrthonormalInitializer(seed=0)(bad_shape)

    def test_the_rank_message_names_the_owning_class(self):
        """The one place the shared validator must not flatten the two."""
        from dl_techniques.initializers import HeOrthonormalInitializer

        with pytest.raises(ValueError, match="OrthonormalInitializer requires"):
            OrthonormalInitializer(seed=0)((2, 2, 2))
        with pytest.raises(ValueError, match="HeOrthonormalInitializer requires"):
            HeOrthonormalInitializer(seed=0)((2, 2, 2))
