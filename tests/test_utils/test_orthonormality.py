"""
Test suite for the validate_orthonormality utility function.

This module provides comprehensive tests for the orthonormality validation function,
covering various scenarios including perfect orthonormal sets, non-orthonormal sets,
edge cases, and error conditions.
"""

import pytest
import numpy as np
import keras
from typing import Any, List, Tuple
from dl_techniques.utils.logger import logger
from dl_techniques.utils.tensors import validate_orthonormality


class TestValidateOrthonormality:
    """Test suite for the validate_orthonormality function."""

    def test_perfect_orthonormal_identity(self) -> None:
        """Test with a perfect identity matrix (orthonormal)."""
        identity_3x3 = np.eye(3, dtype=np.float32)
        result = validate_orthonormality(identity_3x3)
        assert result is True
        logger.info("Perfect identity matrix correctly identified as orthonormal")

    def test_perfect_orthonormal_custom(self) -> None:
        """Test with a custom perfect orthonormal matrix."""
        # Create a rotation matrix (always orthonormal)
        angle = np.pi / 4
        rotation_2d = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ], dtype=np.float32)

        result = validate_orthonormality(rotation_2d)
        assert result is True
        logger.info("Custom orthonormal matrix correctly identified")

    def test_non_orthogonal_vectors(self) -> None:
        """Test with non-orthogonal vectors."""
        # Vectors that are not orthogonal
        non_ortho = np.array([
            [1.0, 0.0],
            [0.5, 0.866]  # Not orthogonal to first vector
        ], dtype=np.float32)

        result = validate_orthonormality(non_ortho)
        assert result is False
        logger.info("Non-orthogonal vectors correctly identified as non-orthonormal")

    def test_non_normalized_vectors(self) -> None:
        """Test with non-normalized vectors."""
        # Orthogonal but not normalized
        non_normalized = np.array([
            [2.0, 0.0],  # Length 2
            [0.0, 1.0]  # Length 1
        ], dtype=np.float32)

        result = validate_orthonormality(non_normalized)
        assert result is False
        logger.info("Non-normalized vectors correctly identified as non-orthonormal")

    def test_single_vector_normalized(self) -> None:
        """Test with a single normalized vector."""
        single_normalized = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        result = validate_orthonormality(single_normalized)
        assert result is True
        logger.info("Single normalized vector correctly identified as orthonormal")

    def test_single_vector_non_normalized(self) -> None:
        """Test with a single non-normalized vector."""
        single_non_normalized = np.array([[2.0, 0.0, 0.0]], dtype=np.float32)
        result = validate_orthonormality(single_non_normalized)
        assert result is False
        logger.info("Single non-normalized vector correctly identified as non-orthonormal")

    def test_empty_matrix(self) -> None:
        """Test with an empty matrix."""
        empty_matrix = np.empty((0, 3), dtype=np.float32)
        result = validate_orthonormality(empty_matrix)
        assert result is True
        logger.info("Empty matrix correctly identified as vacuously orthonormal")

    def test_different_input_types(self) -> None:
        """Test with different input types (numpy, keras tensor, list)."""
        identity_2x2 = [[1.0, 0.0], [0.0, 1.0]]

        # Test with Python list
        result_list = validate_orthonormality(identity_2x2)
        assert result_list is True

        # Test with numpy array
        result_numpy = validate_orthonormality(np.array(identity_2x2, dtype=np.float32))
        assert result_numpy is True

        # Test with keras tensor
        keras_tensor = keras.ops.convert_to_tensor(identity_2x2)
        result_keras = validate_orthonormality(keras_tensor)
        assert result_keras is True

        logger.info("Different input types correctly handled")

    def test_tolerance_parameters(self) -> None:
        """Test with different tolerance parameters."""
        # Create a nearly orthonormal matrix
        nearly_orthonormal = np.array([
            [1.0001, 0.0001],
            [-0.0001, 0.9999]
        ], dtype=np.float32)

        # Should fail with strict tolerance
        result_strict = validate_orthonormality(nearly_orthonormal, rtol=1e-6, atol=1e-8)
        assert result_strict is False

        # Should pass with relaxed tolerance
        result_relaxed = validate_orthonormality(nearly_orthonormal, rtol=1e-2, atol=1e-3)
        assert result_relaxed is True

        logger.info("Tolerance parameters correctly affect validation")

    def test_larger_matrices(self) -> None:
        """Test with larger orthonormal matrices."""
        # Create a larger orthonormal matrix using QR decomposition
        np.random.seed(42)
        random_matrix = np.random.randn(5, 5).astype(np.float32)
        q, _ = np.linalg.qr(random_matrix)

        result = validate_orthonormality(q)
        assert result is True
        logger.info("Larger orthonormal matrix correctly validated")

    def test_rectangular_matrices(self) -> None:
        """Test with rectangular matrices (more columns than rows)."""
        # Create orthonormal rows in a rectangular matrix
        np.random.seed(42)
        random_matrix = np.random.randn(3, 5).astype(np.float32)
        q, _ = np.linalg.qr(random_matrix.T)
        rectangular_ortho = q.T[:3, :]  # Take first 3 rows

        result = validate_orthonormality(rectangular_ortho)
        assert result is True
        logger.info("Rectangular orthonormal matrix correctly validated")

    def test_error_conditions(self) -> None:
        """Test error conditions and edge cases."""
        # Test with non-convertible input
        with pytest.raises(TypeError, match="Input could not be converted to a Keras tensor"):
            validate_orthonormality(object())

        # Test with 1D input
        with pytest.raises(ValueError, match="Input must be a 2D matrix"):
            validate_orthonormality([1, 2, 3])

        # Test with 3D input
        with pytest.raises(ValueError, match="Input must be a 2D matrix"):
            validate_orthonormality(np.random.randn(2, 3, 4))

        logger.info("Error conditions correctly handled")

    def test_numerical_precision_edge_cases(self) -> None:
        """Test numerical precision edge cases."""
        # Test with very small values
        tiny_identity = np.eye(2, dtype=np.float32) * 1e-10
        result = validate_orthonormality(tiny_identity)
        assert result is False  # These are not unit vectors

        # Test with very large values
        large_identity = np.eye(2, dtype=np.float32) * 1e10
        result = validate_orthonormality(large_identity)
        assert result is False  # These are not unit vectors

        logger.info("Numerical precision edge cases correctly handled")

    def test_near_orthonormal_cases(self) -> None:
        """Test cases that are nearly orthonormal."""
        # Create vectors that are almost orthonormal but should fail default tolerance
        # We need to make the error larger than the default tolerances (rtol=1e-3, atol=1e-5)
        eps = 5e-3  # This will create an error larger than default tolerance
        nearly_ortho = np.array([
            [1.0, eps],
            [0.0, 1.0]  # Keep one vector perfectly orthogonal for clearer test
        ], dtype=np.float32)

        # Should fail with default tolerance (rtol=1e-3, atol=1e-5)
        result_default = validate_orthonormality(nearly_ortho)
        assert result_default is False

        # Should pass with more relaxed tolerance
        result_tolerant = validate_orthonormality(nearly_ortho, rtol=1e-2, atol=1e-2)
        assert result_tolerant is True

        logger.info("Near-orthonormal cases correctly handled with tolerance")

    def test_complex_orthonormal_matrices(self) -> None:
        """Test with more complex orthonormal transformations."""
        # Householder reflection matrix
        v = np.array([1, 1, 1], dtype=np.float32)
        v = v / np.linalg.norm(v)  # Normalize
        householder = np.eye(3, dtype=np.float32) - 2 * np.outer(v, v)

        result = validate_orthonormality(householder)
        assert result is True

        # Givens rotation matrix
        c, s = np.cos(np.pi / 6), np.sin(np.pi / 6)
        givens = np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ], dtype=np.float32)

        result = validate_orthonormality(givens)
        assert result is True

        logger.info("Complex orthonormal transformations correctly validated")

    def test_default_tolerance_values(self) -> None:
        """Test that default tolerance values work correctly."""
        # Test with a matrix that should pass with default tolerances
        identity = np.eye(3, dtype=np.float32)
        result = validate_orthonormality(identity)  # Using defaults
        assert result is True

        # Test with a matrix that should fail with default tolerances
        non_ortho = np.array([
            [1.0, 0.1],
            [0.0, 1.0]
        ], dtype=np.float32)
        result = validate_orthonormality(non_ortho)  # Using defaults
        assert result is False

        logger.info("Default tolerance values work correctly")

    @pytest.mark.parametrize("matrix_size", [1, 2, 3, 5, 10])
    def test_various_matrix_sizes(self, matrix_size: int) -> None:
        """Test with various small to medium matrix sizes."""
        identity = np.eye(matrix_size, dtype=np.float32)
        result = validate_orthonormality(identity)
        assert result is True
        logger.info(f"Matrix size {matrix_size}x{matrix_size} correctly validated")

    @pytest.mark.parametrize("matrix_size", [50, 75, 100, 150, 200, 250])
    def test_large_matrix_sizes(self, matrix_size: int) -> None:
        """Test with large matrix sizes (50-250) using QR decomposition for true orthonormality."""
        # Set random seed for reproducible tests
        np.random.seed(42 + matrix_size)  # Different seed for each size

        # Create a truly orthonormal matrix using QR decomposition
        random_matrix = np.random.randn(matrix_size, matrix_size).astype(np.float32)
        q, _ = np.linalg.qr(random_matrix)

        # Test the orthonormal matrix
        result = validate_orthonormality(q)
        assert result is True
        logger.info(f"Large orthonormal matrix {matrix_size}x{matrix_size} correctly validated")

        # Also test identity matrix for this size
        identity = np.eye(matrix_size, dtype=np.float32)
        result_identity = validate_orthonormality(identity)
        assert result_identity is True
        logger.info(f"Large identity matrix {matrix_size}x{matrix_size} correctly validated")

        # Test a non-orthonormal matrix of the same size
        # Create a matrix that's clearly not orthonormal by scaling some rows
        non_orthonormal = identity.copy()
        non_orthonormal[0, :] *= 2.0  # Scale first row, making it non-unit
        result_non_ortho = validate_orthonormality(non_orthonormal)
        assert result_non_ortho is False
        logger.info(f"Large non-orthonormal matrix {matrix_size}x{matrix_size} correctly identified")

    @pytest.mark.parametrize("rows,cols", [
        (50, 100), (50, 150), (75, 200),  # Wide matrices (more columns than rows)
        (100, 50), (150, 75), (200, 100),  # Tall matrices (more rows than columns)
        (60, 120), (80, 160), (100, 250),  # More wide matrices
        (120, 60), (160, 80), (250, 100),  # More tall matrices
    ])
    def test_large_non_square_matrices(self, rows: int, cols: int) -> None:
        """Test with large non-square matrices."""
        # Set random seed for reproducible tests
        np.random.seed(42 + rows + cols)

        if rows <= cols:
            # Wide matrix case: more columns than rows
            # Create orthonormal rows by taking first 'rows' rows from QR decomposition
            random_matrix = np.random.randn(cols, cols).astype(np.float32)
            q, _ = np.linalg.qr(random_matrix)
            orthonormal_wide = q[:rows, :]  # Take first 'rows' rows

            result = validate_orthonormality(orthonormal_wide)
            assert result is True
            logger.info(f"Wide orthonormal matrix {rows}x{cols} correctly validated")

            # Test non-orthonormal case by scaling one row
            non_orthonormal_wide = orthonormal_wide.copy()
            non_orthonormal_wide[0, :] *= 1.5  # Scale first row
            result_non_ortho = validate_orthonormality(non_orthonormal_wide)
            assert result_non_ortho is False
            logger.info(f"Wide non-orthonormal matrix {rows}x{cols} correctly identified")

        else:
            # Tall matrix case: more rows than columns
            # Create orthonormal rows by taking a subset that can be orthonormal
            # We can have at most 'cols' orthonormal rows in a 'cols'-dimensional space
            max_ortho_rows = min(rows, cols)
            random_matrix = np.random.randn(cols, cols).astype(np.float32)
            q, _ = np.linalg.qr(random_matrix)

