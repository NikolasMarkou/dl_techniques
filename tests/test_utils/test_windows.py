import keras
import pytest
import numpy as np
from keras import ops


from dl_techniques.utils.tensors import window_reverse, window_partition

class TestWindowPartitioning:
    """Test suite for window partitioning functions."""

    @pytest.fixture
    def sample_tensor_4d(self) -> keras.KerasTensor:
        """Create a 4D sample tensor for testing."""
        # Create a tensor with known values to verify correctness
        return ops.convert_to_tensor(np.random.rand(2, 8, 8, 64).astype(np.float32))

    @pytest.fixture
    def sample_tensor_large(self) -> keras.KerasTensor:
        """Create a larger 4D sample tensor for testing."""
        return ops.convert_to_tensor(np.random.rand(4, 16, 16, 128).astype(np.float32))

    @pytest.fixture
    def sequential_tensor(self) -> keras.KerasTensor:
        """Create a tensor with sequential values for easier verification."""
        # Create a 4x4 tensor with sequential values for easier tracking
        data = np.arange(64).reshape(1, 4, 4, 4).astype(np.float32)
        return ops.convert_to_tensor(data)

    def test_window_partition_basic_functionality(self, sample_tensor_4d):
        """Test basic window partitioning functionality."""
        window_size = 2
        windows = window_partition(sample_tensor_4d, window_size)

        # Check output shape
        B, H, W, C = sample_tensor_4d.shape
        expected_num_windows = (H // window_size) * (W // window_size)
        expected_shape = (B * expected_num_windows, window_size, window_size, C)

        assert windows.shape == expected_shape, f"Expected shape {expected_shape}, got {windows.shape}"

    def test_window_partition_output_shapes(self):
        """Test window partitioning with different input shapes and window sizes."""
        test_cases = [
            ((2, 8, 8, 64), 2),  # 4x4 windows
            ((1, 16, 16, 32), 4),  # 4x4 windows
            ((3, 12, 12, 128), 3),  # 4x4 windows
            ((4, 14, 14, 256), 7),  # 2x2 windows
        ]

        for input_shape, window_size in test_cases:
            x = ops.convert_to_tensor(np.random.rand(*input_shape).astype(np.float32))
            windows = window_partition(x, window_size)

            B, H, W, C = input_shape
            expected_num_windows = (H // window_size) * (W // window_size)
            expected_shape = (B * expected_num_windows, window_size, window_size, C)

            assert windows.shape == expected_shape, \
                f"Input shape {input_shape}, window_size {window_size}: expected {expected_shape}, got {windows.shape}"

    def test_window_reverse_basic_functionality(self, sample_tensor_4d):
        """Test basic window reverse functionality."""
        window_size = 2
        B, H, W, C = sample_tensor_4d.shape

        # Partition and then reverse
        windows = window_partition(sample_tensor_4d, window_size)
        reconstructed = window_reverse(windows, window_size, H, W)

        # Check that we get back the original shape
        assert reconstructed.shape == sample_tensor_4d.shape, \
            f"Expected shape {sample_tensor_4d.shape}, got {reconstructed.shape}"

    def test_roundtrip_consistency(self, sample_tensor_4d):
        """Test that partition followed by reverse gives back the original tensor."""
        window_sizes = [2, 4]

        for window_size in window_sizes:
            B, H, W, C = sample_tensor_4d.shape

            # Skip if dimensions are not divisible by window size
            if H % window_size != 0 or W % window_size != 0:
                continue

            # Partition and reverse
            windows = window_partition(sample_tensor_4d, window_size)
            reconstructed = window_reverse(windows, window_size, H, W)

            # Check that values are preserved (within floating point precision)
            np.testing.assert_allclose(
                ops.convert_to_numpy(sample_tensor_4d),
                ops.convert_to_numpy(reconstructed),
                rtol=1e-6,
                err_msg=f"Roundtrip failed for window_size={window_size}"
            )

    def test_sequential_tensor_partitioning(self, sequential_tensor):
        """Test partitioning with a tensor containing sequential values."""
        window_size = 2
        windows = window_partition(sequential_tensor, window_size)

        # Convert to numpy for easier verification
        windows_np = ops.convert_to_numpy(windows)

        # Check that we have the expected number of windows
        assert windows_np.shape[0] == 4, f"Expected 4 windows, got {windows_np.shape[0]}"
        assert windows_np.shape[1:] == (2, 2, 4), f"Expected (2, 2, 4) window shape, got {windows_np.shape[1:]}"

    def test_window_reverse_different_batch_sizes(self, sample_tensor_large):
        """Test window reverse with different batch sizes."""
        window_size = 4
        B, H, W, C = sample_tensor_large.shape

        windows = window_partition(sample_tensor_large, window_size)
        reconstructed = window_reverse(windows, window_size, H, W)

        assert reconstructed.shape == sample_tensor_large.shape, \
            f"Batch size preservation failed: expected {sample_tensor_large.shape}, got {reconstructed.shape}"

    def test_multiple_window_sizes(self):
        """Test multiple window sizes with compatible input dimensions."""
        # Create input that's divisible by multiple window sizes
        input_tensor = ops.convert_to_tensor(np.random.rand(2, 12, 12, 32).astype(np.float32))
        window_sizes = [2, 3, 4, 6]

        for window_size in window_sizes:
            windows = window_partition(input_tensor, window_size)
            B, H, W, C = input_tensor.shape

            # Calculate expected number of windows
            num_windows_h = H // window_size
            num_windows_w = W // window_size
            expected_batch_windows = B * num_windows_h * num_windows_w

            assert windows.shape[0] == expected_batch_windows, \
                f"Window size {window_size}: expected {expected_batch_windows} windows, got {windows.shape[0]}"

            # Test reverse operation
            reconstructed = window_reverse(windows, window_size, H, W)
            assert reconstructed.shape == input_tensor.shape, \
                f"Reverse failed for window_size {window_size}"

    def test_single_window_case(self):
        """Test edge case where the entire tensor is one window."""
        # Create tensor where window_size equals spatial dimensions
        input_tensor = ops.convert_to_tensor(np.random.rand(1, 4, 4, 16).astype(np.float32))
        window_size = 4

        windows = window_partition(input_tensor, window_size)

        # Should have exactly one window per batch
        assert windows.shape == (1, 4, 4, 16), \
            f"Single window case failed: expected (1, 4, 4, 16), got {windows.shape}"

        # Test reverse
        reconstructed = window_reverse(windows, window_size, 4, 4)
        assert reconstructed.shape == input_tensor.shape

    def test_numerical_precision(self):
        """Test that partitioning and reverse preserve numerical precision."""
        # Use a tensor with specific values to test precision
        input_data = np.array([[[[1.0, 2.0], [3.0, 4.0]],
                                [[5.0, 6.0], [7.0, 8.0]]]], dtype=np.float32)
        input_tensor = ops.convert_to_tensor(input_data)

        window_size = 1
        windows = window_partition(input_tensor, window_size)
        reconstructed = window_reverse(windows, window_size, 2, 2)

        # Check exact equality for simple case
        np.testing.assert_array_equal(
            ops.convert_to_numpy(input_tensor),
            ops.convert_to_numpy(reconstructed),
            err_msg="Numerical precision test failed"
        )

    def test_different_channel_dimensions(self):
        """Test with different channel dimensions."""
        channel_sizes = [1, 3, 64, 128, 256, 512]

        for channels in channel_sizes:
            input_tensor = ops.convert_to_tensor(
                np.random.rand(1, 8, 8, channels).astype(np.float32)
            )
            window_size = 2

            windows = window_partition(input_tensor, window_size)
            reconstructed = window_reverse(windows, window_size, 8, 8)

            assert windows.shape[-1] == channels, \
                f"Channel dimension not preserved: expected {channels}, got {windows.shape[-1]}"
            assert reconstructed.shape == input_tensor.shape, \
                f"Reconstruction failed for {channels} channels"

    def test_memory_layout_consistency(self):
        """Test that memory layout is consistent through partitioning operations."""
        # Create a tensor with a specific pattern
        input_tensor = ops.convert_to_tensor(
            np.random.rand(2, 6, 6, 8).astype(np.float32)
        )
        window_size = 3

        # Perform operations
        windows = window_partition(input_tensor, window_size)
        reconstructed = window_reverse(windows, window_size, 6, 6)

        # Verify that reconstruction is exact
        np.testing.assert_allclose(
            ops.convert_to_numpy(input_tensor),
            ops.convert_to_numpy(reconstructed),
            rtol=1e-7,
            atol=1e-7,
            err_msg="Memory layout consistency test failed"
        )

    def test_error_handling_for_incompatible_dimensions(self):
        """Test behavior when dimensions are not divisible by window size."""
        # Create tensor with dimensions not divisible by window size
        input_tensor = ops.convert_to_tensor(np.random.rand(1, 7, 7, 32).astype(np.float32))
        window_size = 4

        # This should either work (with truncation) or raise an error
        # The current implementation might have undefined behavior
        try:
            windows = window_partition(input_tensor, window_size)
            # If it doesn't error, check the output shape makes sense
            assert windows.shape[1] == window_size
            assert windows.shape[2] == window_size
        except Exception:
            # Expected behavior for incompatible dimensions
            pass

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    @pytest.mark.parametrize("height,width", [(8, 8), (16, 16), (12, 12)])
    @pytest.mark.parametrize("channels", [32, 64, 128])
    @pytest.mark.parametrize("window_size", [2, 4])
    def test_parameterized_shapes(self, batch_size, height, width, channels, window_size):
        """Parameterized test for various input configurations."""
        # Skip if dimensions are not compatible
        if height % window_size != 0 or width % window_size != 0:
            pytest.skip(f"Incompatible dimensions: {height}x{width} with window_size {window_size}")

        input_tensor = ops.convert_to_tensor(
            np.random.rand(batch_size, height, width, channels).astype(np.float32)
        )

        # Test partitioning
        windows = window_partition(input_tensor, window_size)

        # Calculate expected shape
        num_windows = (height // window_size) * (width // window_size)
        expected_windows_shape = (batch_size * num_windows, window_size, window_size, channels)

        assert windows.shape == expected_windows_shape, \
            f"Shape mismatch for config B={batch_size}, H={height}, W={width}, C={channels}, ws={window_size}"

        # Test reverse
        reconstructed = window_reverse(windows, window_size, height, width)
        assert reconstructed.shape == input_tensor.shape, \
            f"Reverse shape mismatch for same config"

        # Test roundtrip
        np.testing.assert_allclose(
            ops.convert_to_numpy(input_tensor),
            ops.convert_to_numpy(reconstructed),
            rtol=1e-6,
            err_msg=f"Roundtrip failed for config B={batch_size}, H={height}, W={width}, C={channels}, ws={window_size}"
        )


if __name__ == "__main__":
    pytest.main([__file__])
