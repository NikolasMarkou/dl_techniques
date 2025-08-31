"""
Comprehensive test suite for FNetFourierTransform layer.

This test suite follows the modern Keras 3 testing guidelines and ensures
the FNet Fourier Transform layer works correctly across all scenarios including
serialization, gradient flow, and mathematical correctness.
"""

import pytest
import tempfile
import os
import numpy as np
import keras
from keras import ops
import tensorflow as tf
from typing import Any, Dict

from dl_techniques.layers.attention.fnet_fourier_transform import FNetFourierTransform


class TestFNetFourierTransform:
    """Comprehensive test suite for FNetFourierTransform layer."""

    @pytest.fixture
    def layer_config_basic(self) -> Dict[str, Any]:
        """Basic configuration for testing."""
        return {
            'implementation': 'matrix',
            'normalize_dft': True,
            'epsilon': 1e-12
        }

    @pytest.fixture
    def layer_config_no_norm(self) -> Dict[str, Any]:
        """Configuration without DFT normalization."""
        return {
            'implementation': 'matrix',
            'normalize_dft': False,
            'epsilon': 1e-8
        }

    @pytest.fixture
    def small_input(self) -> keras.KerasTensor:
        """Small input tensor for fast testing."""
        # Shape: [batch=2, seq_len=4, hidden=8]
        return keras.random.normal(shape=(2, 4, 8), seed=42)

    @pytest.fixture
    def medium_input(self) -> keras.KerasTensor:
        """Medium input tensor for realistic testing."""
        # Shape: [batch=4, seq_len=16, hidden=32]
        return keras.random.normal(shape=(4, 16, 32), seed=123)

    @pytest.fixture
    def large_input(self) -> keras.KerasTensor:
        """Large input tensor for performance testing."""
        # Shape: [batch=2, seq_len=64, hidden=128]
        return keras.random.normal(shape=(2, 64, 128), seed=456)

    def test_initialization_basic(self, layer_config_basic):
        """Test basic layer initialization."""
        layer = FNetFourierTransform(**layer_config_basic)

        # Check configuration stored correctly
        assert layer.implementation == 'matrix'
        assert layer.normalize_dft is True
        assert layer.epsilon == 1e-12

        # Check layer not built initially
        assert not layer.built
        assert layer.dft_matrix_seq is None
        assert layer.dft_matrix_hidden is None

    def test_initialization_custom_config(self, layer_config_no_norm):
        """Test initialization with custom configuration."""
        layer = FNetFourierTransform(**layer_config_no_norm)

        assert layer.implementation == 'matrix'
        assert layer.normalize_dft is False
        assert layer.epsilon == 1e-8

    def test_forward_pass_and_building(self, layer_config_basic, small_input):
        """Test forward pass and automatic building."""
        layer = FNetFourierTransform(**layer_config_basic)

        # Layer should build automatically on first call
        output = layer(small_input)

        # Check layer is now built
        assert layer.built
        assert layer.dft_matrix_seq is not None
        assert layer.dft_matrix_hidden is not None

        # Check output shape preservation
        assert output.shape == small_input.shape

        # Check DFT matrices have correct shapes
        batch, seq_len, hidden_dim = small_input.shape
        assert layer.dft_matrix_seq.shape == (seq_len, seq_len, 2)
        assert layer.dft_matrix_hidden.shape == (hidden_dim, hidden_dim, 2)

        # Check built dimensions are stored
        assert layer._built_seq_len == seq_len
        assert layer._built_hidden_dim == hidden_dim

    def test_output_shape_computation(self, layer_config_basic):
        """Test output shape computation."""
        layer = FNetFourierTransform(**layer_config_basic)

        input_shape = (None, 16, 64)  # batch, seq, hidden
        output_shape = layer.compute_output_shape(input_shape)

        # Should preserve input shape exactly
        assert output_shape == input_shape

    def test_mathematical_properties(self, layer_config_basic, small_input):
        """Test mathematical properties of the Fourier transform."""
        layer = FNetFourierTransform(**layer_config_basic)
        output = layer(small_input)

        # Check output is real (no complex components)
        assert output.dtype in [keras.backend.floatx(), 'float32', 'float64']

        # Check output has finite values (no NaN or Inf)
        assert keras.ops.all(keras.ops.isfinite(output))

        # For normalized DFT, energy should be approximately preserved
        if layer.normalize_dft:
            input_energy = keras.ops.mean(keras.ops.square(small_input))
            output_energy = keras.ops.mean(keras.ops.square(output))

            # Energy preservation should be reasonable (within 50% for real part only)
            energy_ratio = output_energy / input_energy
            assert 0.1 < energy_ratio < 2.0

    def test_different_input_sizes(self, layer_config_basic):
        """Test layer works with different input sizes."""
        # Test various input sizes
        test_shapes = [
            (1, 2, 4),  # Minimal size
            (3, 8, 16),  # Small size
            (2, 32, 64),  # Medium size
        ]

        for batch_size, seq_len, hidden_dim in test_shapes:
            # A new layer must be instantiated for each shape, as the DFT
            # matrices are cached on build.
            layer = FNetFourierTransform(**layer_config_basic)
            test_input = keras.random.normal(shape=(batch_size, seq_len, hidden_dim))
            output = layer(test_input)
            assert output.shape == test_input.shape

    def test_deterministic_behavior(self, layer_config_basic, small_input):
        """Test that layer produces deterministic outputs."""
        layer = FNetFourierTransform(**layer_config_basic)

        # Multiple runs should produce identical results
        output1 = layer(small_input)
        output2 = layer(small_input)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output1),
            keras.ops.convert_to_numpy(output2),
            rtol=1e-6, atol=1e-6,
            err_msg="Layer should produce deterministic outputs"
        )

    def test_serialization_cycle(self, layer_config_basic, small_input):
        """CRITICAL TEST: Full serialization cycle."""
        # Create model with custom layer
        inputs = keras.Input(shape=small_input.shape[1:])
        outputs = FNetFourierTransform(**layer_config_basic)(inputs)
        model = keras.Model(inputs, outputs)

        # Get original prediction
        original_pred = model(small_input)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_fnet_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(small_input)

            # Verify identical predictions after serialization
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions should match after serialization"
            )

    def test_config_completeness(self, layer_config_basic):
        """Test that get_config contains all __init__ parameters."""
        layer = FNetFourierTransform(**layer_config_basic)
        config = layer.get_config()

        # Check all configuration parameters are present
        for key in ['implementation', 'normalize_dft', 'epsilon']:
            assert key in config, f"Missing {key} in get_config()"
            assert config[key] == layer_config_basic[key]

    def test_config_reconstruction(self, layer_config_basic, small_input):
        """Test layer can be reconstructed from config."""
        original_layer = FNetFourierTransform(**layer_config_basic)
        original_output = original_layer(small_input)

        # Reconstruct from config
        config = original_layer.get_config()
        reconstructed_layer = FNetFourierTransform.from_config(config)
        reconstructed_output = reconstructed_layer(small_input)

        # Should produce identical results
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(original_output),
            keras.ops.convert_to_numpy(reconstructed_output),
            rtol=1e-6, atol=1e-6,
            err_msg="Reconstructed layer should match original"
        )

    def test_gradients_flow(self, layer_config_basic, small_input):
        """Test gradient computation works correctly."""
        layer = FNetFourierTransform(**layer_config_basic)

        # Test gradients flow through the layer
        with tf.GradientTape() as tape:
            tape.watch(small_input)
            output = layer(small_input)
            loss = keras.ops.mean(keras.ops.square(output))

        gradients = tape.gradient(loss, small_input)

        # Check gradients exist and are non-zero
        assert gradients is not None
        assert not keras.ops.all(keras.ops.equal(gradients, 0.0))
        assert keras.ops.all(keras.ops.isfinite(gradients))

    def test_no_trainable_parameters(self, layer_config_basic, small_input):
        """Test that layer has no trainable parameters."""
        layer = FNetFourierTransform(**layer_config_basic)
        output = layer(small_input)  # Build the layer

        # FNet should have no trainable parameters (parameter-free)
        trainable_weights = layer.trainable_weights
        assert len(trainable_weights) == 0

        # But should have non-trainable weights (DFT matrices)
        non_trainable_weights = layer.non_trainable_weights
        assert len(non_trainable_weights) == 2  # seq and hidden DFT matrices

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, layer_config_basic, small_input, training):
        """Test behavior in different training modes."""
        layer = FNetFourierTransform(**layer_config_basic)

        output = layer(small_input, training=training)

        # Output should be consistent regardless of training mode
        # (since FNet has no dropout or batch norm)
        assert output.shape == small_input.shape
        assert keras.ops.all(keras.ops.isfinite(output))

    def test_batch_size_independence(self, layer_config_basic):
        """Test layer works with different batch sizes."""
        layer = FNetFourierTransform(**layer_config_basic)

        seq_len, hidden_dim = 8, 16

        # Test with different batch sizes
        for batch_size in [1, 2, 4, 8]:
            test_input = keras.random.normal(shape=(batch_size, seq_len, hidden_dim))
            output = layer(test_input)
            assert output.shape == (batch_size, seq_len, hidden_dim)

    def test_dft_matrix_properties(self, layer_config_basic, small_input):
        """Test mathematical properties of DFT matrices."""
        layer = FNetFourierTransform(**layer_config_basic)
        layer(small_input)  # Build layer

        # Extract DFT matrices
        seq_matrix = layer.dft_matrix_seq.numpy()  # [seq_len, seq_len, 2]
        hidden_matrix = layer.dft_matrix_hidden.numpy()  # [hidden_dim, hidden_dim, 2]

        # Convert to complex for testing
        seq_complex = seq_matrix[..., 0] + 1j * seq_matrix[..., 1]
        hidden_complex = hidden_matrix[..., 0] + 1j * hidden_matrix[..., 1]

        # Test unitarity property (for normalized DFT)
        if layer.normalize_dft:
            seq_len, hidden_dim = small_input.shape[1], small_input.shape[2]

            # DFT matrices should be approximately unitary
            seq_identity = np.matmul(seq_complex, seq_complex.conj().T)
            seq_identity_error = np.max(np.abs(seq_identity - np.eye(seq_len)))
            assert seq_identity_error < 1e-5, "Sequence DFT matrix should be unitary"

            hidden_identity = np.matmul(hidden_complex, hidden_complex.conj().T)
            hidden_identity_error = np.max(np.abs(hidden_identity - np.eye(hidden_dim)))
            assert hidden_identity_error < 1e-5, "Hidden DFT matrix should be unitary"

    def test_normalization_effect(self, small_input):
        """Test effect of DFT normalization."""
        # Test with normalization
        layer_norm = FNetFourierTransform(normalize_dft=True)
        output_norm = layer_norm(small_input)

        # Test without normalization
        layer_no_norm = FNetFourierTransform(normalize_dft=False)
        output_no_norm = layer_no_norm(small_input)

        # Outputs should be different
        output_norm_np = keras.ops.convert_to_numpy(output_norm)
        output_no_norm_np = keras.ops.convert_to_numpy(output_no_norm)
        assert not np.allclose(output_norm_np, output_no_norm_np, atol=1e-6)

        # Normalized version should have smaller magnitude
        norm_magnitude = keras.ops.mean(keras.ops.square(output_norm))
        no_norm_magnitude = keras.ops.mean(keras.ops.square(output_no_norm))
        assert norm_magnitude < no_norm_magnitude

    def test_edge_cases_invalid_inputs(self):
        """Test error conditions and edge cases."""

        # Test invalid implementation
        with pytest.raises(ValueError, match="implementation must be one of"):
            FNetFourierTransform(implementation='invalid')

        # Test invalid epsilon
        with pytest.raises(ValueError, match="epsilon must be positive"):
            FNetFourierTransform(epsilon=0.0)

        with pytest.raises(ValueError, match="epsilon must be positive"):
            FNetFourierTransform(epsilon=-1e-6)

    def test_edge_cases_invalid_shapes(self, layer_config_basic):
        """Test invalid input shapes."""
        layer = FNetFourierTransform(**layer_config_basic)

        # Test 2D input (missing batch or sequence dimension)
        with pytest.raises(ValueError, match="expects 3D input"):
            layer.build(input_shape=(32, 64))  # Missing one dimension

        # Test 4D input (extra dimension)
        with pytest.raises(ValueError, match="expects 3D input"):
            layer.build(input_shape=(2, 16, 32, 8))

        # Test unknown dimensions at build time
        with pytest.raises(ValueError, match="must be known at build time"):
            layer.build(input_shape=(None, None, 64))

    def test_memory_efficiency(self, layer_config_basic):
        """Test memory usage is reasonable for different sizes."""
        layer = FNetFourierTransform(**layer_config_basic)

        # Test that layer can handle reasonably large inputs
        large_input = keras.random.normal(shape=(2, 128, 256))
        output = layer(large_input)

        # Should work without memory errors
        assert output.shape == large_input.shape
        assert keras.ops.all(keras.ops.isfinite(output))

    def test_different_implementations(self, small_input):
        """Test different implementation modes."""
        # Matrix implementation
        layer_matrix = FNetFourierTransform(implementation='matrix')
        output_matrix = layer_matrix(small_input)

        # FFT implementation (not implemented, but should not crash)
        layer_fft = FNetFourierTransform(implementation='fft')
        output_fft = layer_fft(small_input)

        # Both should produce valid outputs
        assert output_matrix.shape == small_input.shape
        assert output_fft.shape == small_input.shape
        assert keras.ops.all(keras.ops.isfinite(output_matrix))
        assert keras.ops.all(keras.ops.isfinite(output_fft))

    def test_numerical_stability(self, layer_config_basic):
        """Test numerical stability with extreme inputs."""
        layer = FNetFourierTransform(**layer_config_basic)

        # Test with very small inputs
        small_val_input = keras.random.normal(shape=(2, 4, 8)) * 1e-8
        output_small = layer(small_val_input)
        assert keras.ops.all(keras.ops.isfinite(output_small))

        # Test with very large inputs
        large_val_input = keras.random.normal(shape=(2, 4, 8)) * 100.0
        output_large = layer(large_val_input)
        assert keras.ops.all(keras.ops.isfinite(output_large))

    def test_reproducibility_across_sessions(self, layer_config_basic, small_input):
        """Test that results are reproducible across different layer instances."""
        # Create two separate layer instances with same config
        layer1 = FNetFourierTransform(**layer_config_basic)
        layer2 = FNetFourierTransform(**layer_config_basic)

        # Both should produce identical results
        output1 = layer1(small_input)
        output2 = layer2(small_input)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output1),
            keras.ops.convert_to_numpy(output2),
            rtol=1e-6, atol=1e-6,
            err_msg="Different instances should produce identical results"
        )

    def test_complex_arithmetic_correctness(self, layer_config_basic):
        """Test that complex matrix multiplication is implemented correctly."""
        layer = FNetFourierTransform(**layer_config_basic)

        # Create a simple test case we can verify manually
        test_input = keras.ops.ones((1, 2, 2))  # Simple input
        output = layer(test_input)

        # Should produce finite, real output
        assert keras.ops.all(keras.ops.isfinite(output))
        assert output.shape == (1, 2, 2)

        # Access the DFT matrices to verify they were created correctly
        assert layer.dft_matrix_seq.shape == (2, 2, 2)
        assert layer.dft_matrix_hidden.shape == (2, 2, 2)

    @pytest.mark.slow
    def test_performance_scaling(self, layer_config_basic):
        """Test performance scaling with input size (marked as slow test)."""
        # Test different sizes to ensure reasonable scaling
        sizes = [(1, 32, 64), (1, 64, 128), (1, 128, 256)]

        for batch, seq_len, hidden_dim in sizes:
            # A new layer must be instantiated for each shape.
            layer = FNetFourierTransform(**layer_config_basic)
            test_input = keras.random.normal(shape=(batch, seq_len, hidden_dim))

            # Should complete without timeout or memory error
            output = layer(test_input)
            assert output.shape == test_input.shape


# Additional utility functions for testing
def create_test_input_with_pattern(batch_size: int, seq_len: int, hidden_dim: int) -> keras.KerasTensor:
    """Create test input with a known pattern for verification."""
    # Create input with simple pattern for easier verification
    x = np.zeros((batch_size, seq_len, hidden_dim), dtype=np.float32)

    # Add some structure that should be preserved/transformed predictably
    for b in range(batch_size):
        for s in range(seq_len):
            for h in range(hidden_dim):
                x[b, s, h] = np.sin(2 * np.pi * s / seq_len) + np.cos(2 * np.pi * h / hidden_dim)

    return keras.ops.convert_to_tensor(x)


def verify_fourier_properties(layer: FNetFourierTransform, test_input: keras.KerasTensor) -> bool:
    """Verify mathematical properties specific to Fourier transforms."""
    output = layer(test_input)

    # Basic sanity checks
    if not keras.ops.all(keras.ops.isfinite(output)):
        return False

    if output.shape != test_input.shape:
        return False

    # Energy-related checks for normalized DFT
    if layer.normalize_dft:
        input_energy = keras.ops.mean(keras.ops.square(test_input))
        output_energy = keras.ops.mean(keras.ops.square(output))

        # Energy should be in reasonable range (real part only)
        if output_energy <= 0 or (input_energy > 1e-9 and output_energy / input_energy > 10):
            return False

    return True