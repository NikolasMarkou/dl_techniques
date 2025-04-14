import pytest
import tensorflow as tf
import numpy as np
from numpy.testing import assert_allclose
from typing import Tuple

from dl_techniques.utils.tensors import \
    power_iteration, wt_x_w_normalize, gram_matrix, reshape_to_2d, gaussian_kernel


@pytest.fixture
def random_weights_2d() -> tf.Tensor:
    """Generate random 2D weights for testing."""
    tf.random.set_seed(42)
    return tf.random.normal((10, 5))


@pytest.fixture
def random_weights_4d() -> tf.Tensor:
    """Generate random 4D weights for testing."""
    tf.random.set_seed(42)
    return tf.random.normal((3, 3, 64, 32))


class TestPowerIteration:
    def test_identity_matrix(self):
        """Test with identity matrix (known spectral norm of 1)."""
        matrix = tf.eye(4, dtype=tf.float32)
        result = power_iteration(matrix)
        assert_allclose(result, 1.0, rtol=1e-3)

    def test_zero_matrix(self):
        """Test with zero matrix (known spectral norm of 0)."""
        matrix = tf.zeros((4, 4), dtype=tf.float32)
        result = power_iteration(matrix)
        assert_allclose(result, 0.0, rtol=1e-3)

    def test_simple_2x2(self):
        """Test with a simple 2x2 matrix with known spectral norm."""
        # Matrix [[2, 0], [0, 1]] has spectral norm 2
        matrix = tf.constant([[2.0, 0.0], [0.0, 1.0]], dtype=tf.float32)
        result = power_iteration(matrix)
        assert_allclose(result, 2.0, rtol=1e-3)

    def test_invalid_shape(self):
        """Test that invalid input shapes raise ValueError."""
        invalid_matrix = tf.ones((3,), dtype=tf.float32)  # 1D tensor
        with pytest.raises(ValueError, match="Input matrix must be 2-dimensional"):
            power_iteration(invalid_matrix)

    def test_symmetric_matrix(self):
        """Test with a symmetric matrix where eigenvalues are known."""
        # Symmetric matrix with eigenvalues 3 and 1
        matrix = tf.constant([[2.0, 1.0],
                              [1.0, 2.0]], dtype=tf.float32)
        result = power_iteration(matrix)
        # Largest eigenvalue is 3 for this matrix
        assert_allclose(result, 3.0, rtol=1e-3)

    def test_rectangular_matrix(self):
        """Test with a rectangular matrix."""
        # 3x2 matrix with known singular values
        matrix = tf.constant([[1.0, 0.0],
                              [0.0, 2.0],
                              [0.0, 0.0]], dtype=tf.float32)
        result = power_iteration(matrix)
        # Largest singular value should be 2
        assert_allclose(result, 2.0, rtol=1e-3)

    def test_convergence_iterations(self):
        """Test convergence with different iteration counts."""
        matrix = tf.constant([[3.0, 1.0],
                              [1.0, 3.0]], dtype=tf.float32)
        # Run with different iteration counts
        result_few = power_iteration(matrix, iterations=2)
        result_many = power_iteration(matrix, iterations=20)
        # Results should be close despite different iteration counts
        assert_allclose(result_few, result_many, rtol=1e-1)

    def test_scaled_matrix(self):
        """Test with scaled versions of the same matrix."""
        base_matrix = tf.constant([[2.0, 1.0],
                                   [1.0, 2.0]], dtype=tf.float32)
        scale = 10.0
        scaled_matrix = base_matrix * scale

        base_result = power_iteration(base_matrix)
        scaled_result = power_iteration(scaled_matrix)
        # Spectral norm should scale linearly
        assert_allclose(scaled_result, base_result * scale, rtol=1e-3)

    def test_rotation_matrix(self):
        """Test with a rotation matrix (should have spectral norm 1)."""
        # 2D rotation matrix (45 degrees)
        angle = np.pi / 4
        cos, sin = np.cos(angle), np.sin(angle)
        matrix = tf.constant([[cos, -sin],
                              [sin, cos]], dtype=tf.float32)
        result = power_iteration(matrix)
        # Rotation matrices have spectral norm 1
        assert_allclose(result, 1.0, rtol=1e-3)

    def test_block_diagonal_matrix(self):
        """Test with block diagonal matrix - spectral norm should be max of blocks."""
        # Create a 4x4 block diagonal matrix with blocks [[3,1],[1,3]] and [[2,0],[0,1]]
        matrix = tf.constant([
            [3.0, 1.0, 0.0, 0.0],
            [1.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=tf.float32)
        result = power_iteration(matrix)
        # Largest eigenvalue of first block is 4, second block is 2
        assert_allclose(result, 4.0, rtol=1e-3)

    def test_nilpotent_matrix(self):
        """Test with nilpotent matrix (all eigenvalues zero)."""
        # 3x3 nilpotent matrix
        matrix = tf.constant([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0]
        ], dtype=tf.float32)
        result = power_iteration(matrix)
        # Spectral norm should be 1.0 for this particular nilpotent matrix
        assert_allclose(result, 1.0, rtol=1e-3)

    def test_hadamard_matrix(self):
        """Test with 4x4 Hadamard matrix (normalized)."""
        # 4x4 Hadamard matrix normalized by 1/2
        matrix = tf.constant([
            [1.0, 1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0, 1.0]
        ], dtype=tf.float32) * 0.5
        result = power_iteration(matrix)
        # Normalized Hadamard matrices have spectral norm 1
        assert_allclose(result, 1.0, rtol=1e-3)

    def test_stability_epsilon_variation(self):
        """Test stability with different epsilon values."""
        # Create a well-conditioned matrix
        matrix = tf.constant([
            [4.0, 1.0],
            [1.0, 3.0]
        ], dtype=tf.float32)

        # Test with different epsilon values, spanning several orders of magnitude
        result_large = power_iteration(matrix, epsilon=1e-3)
        result_small = power_iteration(matrix, epsilon=1e-9)
        result_tiny = power_iteration(matrix, epsilon=1e-15)

        # All results should be close to each other
        assert_allclose(result_large, result_small, rtol=1e-3)
        assert_allclose(result_small, result_tiny, rtol=1e-3)
        assert_allclose(result_large, result_tiny, rtol=1e-3)

    def test_stability_large_small_entries(self):
        """Test stability with matrices containing both very large and very small entries."""
        matrix = tf.constant([
            [1e4, 1e-4],
            [1e-4, 1e4]
        ], dtype=tf.float32)

        # Run multiple times with different iteration counts
        result_base = power_iteration(matrix, iterations=10)
        result_more = power_iteration(matrix, iterations=20)
        result_most = power_iteration(matrix, iterations=30)

        # Results should be close to the analytically known spectral norm
        # For this matrix, the spectral norm is approximately 1e4
        expected = 1e4

        assert_allclose(result_base, expected, rtol=1e-3)
        assert_allclose(result_more, expected, rtol=1e-3)
        assert_allclose(result_most, expected, rtol=1e-3)
        # Results should also be close to each other
        assert_allclose(result_base, result_more, rtol=1e-3)
        assert_allclose(result_more, result_most, rtol=1e-3)

class TestReshape:
    def test_reshape_to_2d_with_2d_input(self, random_weights_2d: tf.Tensor) -> None:
        """Test reshape_to_2d with 2D input."""
        reshaped = reshape_to_2d(random_weights_2d)
        assert len(reshaped.shape) == 2
        assert reshaped.shape[0] == random_weights_2d.shape[1]
        assert reshaped.shape[1] == random_weights_2d.shape[0]

    def test_reshape_to_2d_with_4d_input(self, random_weights_4d: tf.Tensor) -> None:
        """Test reshape_to_2d with 4D input."""
        reshaped = reshape_to_2d(random_weights_4d)
        assert len(reshaped.shape) == 2
        assert reshaped.shape[0] == random_weights_4d.shape[3]
        assert reshaped.shape[1] == (random_weights_4d.shape[0] *
                                     random_weights_4d.shape[1] *
                                     random_weights_4d.shape[2])

class TestGram:
    def test_wt_x_w_computation(self, random_weights_2d: tf.Tensor) -> None:
        """Test wt_x_w computation."""
        result = gram_matrix(random_weights_2d)
        expected = tf.matmul(
            tf.transpose(random_weights_2d),
            random_weights_2d
        )
        assert tf.reduce_all(tf.abs(result - expected) < 1e-5)


class TestGaussianKernel:
    def test_gaussian_kernel_shape(self):
        """Test if gaussian_kernel produces correct shapes."""
        kernel_size = (5, 5)
        nsig = (2.0, 2.0)
        kernel = gaussian_kernel(kernel_size, nsig)
        assert kernel.shape == kernel_size


    def test_gaussian_kernel_normalization(self):
        """Test if gaussian_kernel is properly normalized."""
        kernel_size = (7, 7)
        nsig = (1.5, 1.5)
        kernel = gaussian_kernel(kernel_size, nsig)
        assert np.abs(np.sum(kernel) - 1.0) < 1e-6


    def test_gaussian_kernel_symmetry(self):
        """Test if gaussian_kernel is symmetric."""
        kernel_size = (5, 5)
        nsig = (2.0, 2.0)
        kernel = gaussian_kernel(kernel_size, nsig)
        assert np.allclose(kernel, kernel.T)


    def test_gaussian_kernel_invalid_inputs(self):
        """Test if gaussian_kernel handles invalid inputs correctly."""
        with pytest.raises(ValueError):
            gaussian_kernel((3,), (1.0,))  # Invalid tuple lengths
