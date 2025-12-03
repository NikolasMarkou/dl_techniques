"""
Comprehensive test suite for ShearletTransform implementation.

This module contains comprehensive unit tests for the ShearletTransform layer using pytest,
following modern Keras 3 testing best practices. Tests cover basic functionality,
input validation, numerical properties, serialization, and frame properties.
"""

import pytest
import numpy as np
import tensorflow as tf
import keras
from keras import ops
import tempfile
import os
from typing import Tuple, Dict, List, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.shearlet_transform import ShearletTransform


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def reconstruct_filter_bank(layer: ShearletTransform, center: bool = True) -> np.ndarray:
    """
    Helper to reconstruct the complex filter bank from the layer's stored weights.

    Args:
        layer: Built ShearletTransform layer.
        center: If True, applies fftshift to move DC component to the image center
                (standard visualization). If False, keeps DC at corners (standard FFT layout).

    Returns:
        Numpy array of shape (num_filters, H, W) with complex64 dtype.
    """
    if layer.filter_bank_real is None or layer.filter_bank_imag is None:
        raise ValueError("Layer is not built or filters are missing.")

    real = ops.convert_to_numpy(layer.filter_bank_real)
    imag = ops.convert_to_numpy(layer.filter_bank_imag)
    complex_bank = real + 1j * imag

    if center:
        # The layer stores filters pre-shifted (ifftshift applied).
        # To center them for analysis, we apply fftshift.
        return np.fft.fftshift(complex_bank, axes=(-2, -1))
    return complex_bank


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture(params=[
    {'scales': 2, 'directions': 4},
    {'scales': 3, 'directions': 8},
    {'scales': 4, 'directions': 12}
])
def transform_config(request) -> Dict[str, int]:
    """Fixture providing different filter configurations."""
    return request.param


@pytest.fixture(params=[(32, 32), (64, 64), (128, 128)])
def input_shape(request) -> Tuple[int, int]:
    """Fixture providing test input shapes."""
    return request.param


@pytest.fixture
def default_config() -> Dict[str, Any]:
    """Provide default configuration for testing."""
    return {
        'scales': 3,
        'directions': 8,
        'alpha': 0.5,
        'high_freq': True
    }


@pytest.fixture
def transform(transform_config: Dict[str, int]) -> ShearletTransform:
    """Fixture creating ShearletTransform instance."""
    return ShearletTransform(**transform_config)


@pytest.fixture
def default_transform(default_config: Dict[str, Any]) -> ShearletTransform:
    """Provide a default ShearletTransform instance."""
    return ShearletTransform(**default_config)


@pytest.fixture
def built_transform(transform: ShearletTransform) -> ShearletTransform:
    """Fixture providing built transform with default size."""
    transform.build(input_shape=(1, 64, 64, 1))
    return transform


def _create_synthetic_batch(shape: Tuple[int, ...]) -> tf.Tensor:
    """Helper function to create a synthetic batch of images."""
    b, h, w, c = shape
    batch = np.zeros(shape, dtype=np.float32)

    x, y = np.meshgrid(np.linspace(-2, 2, w), np.linspace(-2, 2, h))
    gaussian = np.exp(-(x ** 2 + y ** 2) / 0.8)
    pattern = gaussian + 0.3 * np.sin(2 * np.pi * (x + y)) * gaussian

    for i in range(b):
        for j in range(c):
            # Create slightly different patterns for each batch item and channel
            current_pattern = (pattern + i * 0.1) * (1 - j * 0.2)
            batch[i, :, :, j] = current_pattern

    return ops.convert_to_tensor(batch, dtype="float32")


@pytest.fixture
def sample_input(input_shape: Tuple[int, int]) -> Tuple[tf.Tensor, Tuple[int, ...]]:
    """Generate sample single-channel input tensors with realistic patterns."""
    h, w = input_shape
    shape = (2, h, w, 1)
    return _create_synthetic_batch(shape), shape


@pytest.fixture
def multi_channel_sample_input(input_shape: Tuple[int, int]) -> Tuple[tf.Tensor, Tuple[int, ...]]:
    """Generate sample multi-channel input tensors with realistic patterns."""
    h, w = input_shape
    shape = (2, h, w, 3)
    return _create_synthetic_batch(shape), shape


@pytest.fixture
def angle_patterns() -> List[Tuple[float, tf.Tensor]]:
    """Fixture providing test patterns at different angles."""
    x, y = np.meshgrid(
        np.linspace(-1, 1, 64),
        np.linspace(-1, 1, 64)
    )

    patterns = []
    for angle in [0, 45, 90, 135]:
        theta = np.radians(angle)
        pattern = np.sin(8 * np.pi * (x * np.cos(theta) + y * np.sin(theta)))
        pattern = pattern * np.exp(-(x ** 2 + y ** 2) / 0.5)  # Apply window

        tensor = ops.convert_to_tensor(
            pattern[np.newaxis, :, :, np.newaxis],
            dtype="float32"
        )
        patterns.append((angle, tensor))

    return patterns


@pytest.fixture
def scale_patterns() -> List[Tuple[int, tf.Tensor]]:
    """Fixture providing test patterns at different scales."""
    x, y = np.meshgrid(
        np.linspace(-1, 1, 64),
        np.linspace(-1, 1, 64)
    )

    patterns = []
    for scale in [1, 4, 16]:
        pattern = np.sin(scale * np.pi * (x + y))
        pattern = pattern * np.exp(-(x ** 2 + y ** 2) / 0.5)

        tensor = ops.convert_to_tensor(
            pattern[np.newaxis, :, :, np.newaxis],
            dtype="float32"
        )
        patterns.append((scale, tensor))

    return patterns


# ---------------------------------------------------------------------
# Critical Serialization Tests (Most Important)
# ---------------------------------------------------------------------

class TestSerialization:
    """Critical serialization tests following Keras 3 best practices."""

    def test_serialization_cycle(self, default_config: Dict[str, Any], sample_input: Tuple[tf.Tensor, Tuple]) -> None:
        """CRITICAL TEST: Full serialization cycle with prediction comparison.

        This is the most important test for Keras layer compatibility.
        """
        input_tensor, _ = sample_input

        # 1. Create original layer in a model
        inputs = keras.Input(shape=input_tensor.shape[1:])
        layer_output = ShearletTransform(**default_config)(inputs)
        model = keras.Model(inputs, layer_output)

        # 2. Get prediction from original
        original_prediction = model(input_tensor)

        # 3. Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_prediction = loaded_model(input_tensor)

            # 4. Verify identical outputs using numpy comparison
            np.testing.assert_allclose(
                ops.convert_to_numpy(original_prediction),
                ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization"
            )

    def test_config_completeness(self, default_config: Dict[str, Any]) -> None:
        """Test that get_config contains all __init__ parameters."""
        layer = ShearletTransform(**default_config)
        config = layer.get_config()

        # Check all config parameters are present
        for key in default_config:
            assert key in config, f"Missing {key} in get_config()"
            assert config[key] == default_config[key], f"Config mismatch for {key}"

    def test_from_config(self, default_config: Dict[str, Any]) -> None:
        """Test layer reconstruction from configuration."""
        # Create original layer
        original_layer = ShearletTransform(**default_config)

        # Get config and recreate
        config = original_layer.get_config()
        recreated_layer = ShearletTransform.from_config(config)

        # Verify parameters match
        assert recreated_layer.scales == original_layer.scales
        assert recreated_layer.directions == original_layer.directions
        assert recreated_layer.alpha == original_layer.alpha
        assert recreated_layer.high_freq == original_layer.high_freq


# ---------------------------------------------------------------------
# Layer Behavior Tests
# ---------------------------------------------------------------------

class TestLayerBehavior:
    """Test core layer behavior and functionality."""

    def test_forward_pass_and_building(self, default_transform: ShearletTransform,
                                     sample_input: Tuple[tf.Tensor, Tuple]) -> None:
        """Test forward pass and automatic building."""
        input_tensor, _ = sample_input

        # Layer should not be built initially
        assert not default_transform.built

        # Forward pass should trigger building
        output = default_transform(input_tensor)

        # Layer should be built after forward pass
        assert default_transform.built
        assert output.shape[0] == input_tensor.shape[0]  # Batch size preserved

        # Verify all internal attributes are created
        assert default_transform.height is not None
        assert default_transform.width is not None
        # Refactored layer uses split real/imag parts
        assert default_transform.filter_bank_real is not None
        assert default_transform.filter_bank_imag is not None

    def test_output_shape_computation(self, default_transform: ShearletTransform,
                                    sample_input: Tuple[tf.Tensor, Tuple]) -> None:
        """Test output shape computation and consistency for single-channel input."""
        input_tensor, _ = sample_input

        # Build layer
        default_transform.build(input_tensor.shape)

        # Test compute_output_shape
        computed_shape = default_transform.compute_output_shape(input_tensor.shape)

        # Test actual output shape
        output = default_transform(input_tensor)
        actual_shape = output.shape

        # Verify shapes match
        assert computed_shape == tuple(actual_shape), \
            f"Computed shape {computed_shape} doesn't match actual {tuple(actual_shape)}"

        # Verify expected number of output channels
        # Formula: 1 (lowpass) + scales * (directions + 1)
        expected_channels = 1 + default_transform.scales * (default_transform.directions + 1)
        assert computed_shape[-1] == expected_channels

    def test_multi_channel_output_shape(self, default_transform: ShearletTransform,
                                        multi_channel_sample_input: Tuple[tf.Tensor, Tuple]) -> None:
        """Test output shape computation for multi-channel input."""
        input_tensor, _ = multi_channel_sample_input
        num_input_channels = input_tensor.shape[-1]

        output = default_transform(input_tensor)

        num_base_filters = 1 + default_transform.scales * (default_transform.directions + 1)
        expected_channels = num_input_channels * num_base_filters

        assert output.shape[-1] == expected_channels, \
            f"Expected {expected_channels} output channels, got {output.shape[-1]}"

    def test_multi_channel_processing_consistency(self, default_transform: ShearletTransform,
                                                  multi_channel_sample_input: Tuple[tf.Tensor, Tuple]) -> None:
        """Verify that channels are processed independently and correctly."""
        input_tensor, _ = multi_channel_sample_input

        # Process all channels at once
        batch_output = default_transform(input_tensor)

        # Reconstruct bank to count filters
        filter_bank = reconstruct_filter_bank(default_transform)
        num_base_filters = filter_bank.shape[0]

        # Process each channel individually and compare
        for i in range(input_tensor.shape[-1]):
            # Process single channel
            single_channel_input = input_tensor[..., i:i+1]
            single_channel_output = default_transform(single_channel_input)

            # Extract the corresponding slice from the batch output
            start_idx = i * num_base_filters
            end_idx = (i + 1) * num_base_filters
            batch_slice = batch_output[..., start_idx:end_idx]

            np.testing.assert_allclose(
                ops.convert_to_numpy(single_channel_output),
                ops.convert_to_numpy(batch_slice),
                rtol=1e-6, atol=1e-6,
                err_msg=f"Processing for channel {i} is inconsistent between batch and single modes"
            )

    def test_gradients_flow(self, default_transform: ShearletTransform,
                           sample_input: Tuple[tf.Tensor, Tuple]) -> None:
        """Test gradient computation through the layer."""
        input_tensor, _ = sample_input

        # Use tf.GradientTape explicitly as per instructions/backend requirement
        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            output = default_transform(input_tensor)
            loss = ops.mean(ops.square(output))

        # Compute gradients
        gradients = tape.gradient(loss, input_tensor)

        # Verify gradients exist and are non-zero
        assert gradients is not None, "No gradients computed"
        assert not ops.all(ops.equal(gradients, 0.0)), "All gradients are zero"

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, default_transform: ShearletTransform,
                           sample_input: Tuple[tf.Tensor, Tuple], training: bool) -> None:
        """Test behavior in different training modes."""
        input_tensor, _ = sample_input

        output = default_transform(input_tensor, training=training)
        assert output.shape[0] == input_tensor.shape[0]

    def test_multiple_calls_consistency(self, default_transform: ShearletTransform,
                                      sample_input: Tuple[tf.Tensor, Tuple]) -> None:
        """Test that multiple calls produce consistent results."""
        input_tensor, _ = sample_input

        # Multiple calls should produce identical results
        output1 = default_transform(input_tensor)
        output2 = default_transform(input_tensor)

        np.testing.assert_allclose(
            ops.convert_to_numpy(output1),
            ops.convert_to_numpy(output2),
            rtol=1e-6, atol=1e-6,
            err_msg="Multiple calls produced different results"
        )


# ---------------------------------------------------------------------
# Parameter Validation Tests
# ---------------------------------------------------------------------

class TestParameterValidation:
    """Test input validation and error handling."""

    def test_valid_parameters(self) -> None:
        """Test that valid parameters are accepted."""
        # These should not raise errors
        ShearletTransform(scales=1, directions=2, alpha=0.1, high_freq=False)
        ShearletTransform(scales=5, directions=16, alpha=1.0, high_freq=True)

    def test_invalid_scales(self) -> None:
        """Test error handling for invalid scales parameter."""
        with pytest.raises(ValueError, match="scales must be positive"):
            ShearletTransform(scales=0)

        with pytest.raises(ValueError, match="scales must be positive"):
            ShearletTransform(scales=-1)

    def test_invalid_directions(self) -> None:
        """Test error handling for invalid directions parameter."""
        with pytest.raises(ValueError, match="directions must be positive"):
            ShearletTransform(directions=0)

        with pytest.raises(ValueError, match="directions must be positive"):
            ShearletTransform(directions=-2)

    def test_invalid_alpha(self) -> None:
        """Test error handling for invalid alpha parameter."""
        with pytest.raises(ValueError, match="alpha must be in"):
            ShearletTransform(alpha=0.0)

        with pytest.raises(ValueError, match="alpha must be in"):
            ShearletTransform(alpha=1.5)

        with pytest.raises(ValueError, match="alpha must be in"):
            ShearletTransform(alpha=-0.1)

    def test_invalid_input_shape(self, default_transform: ShearletTransform) -> None:
        """Test error handling for invalid input shapes."""
        # 3D input should fail build.
        with pytest.raises(ValueError, match="Expected 4D input shape"):
            default_transform.build((32, 32, 1))

        # None dimensions in H/W should fail for fixed filter bank construction
        with pytest.raises(ValueError, match="Height and width dimensions must be specified"):
            default_transform.build((1, None, 32, 1))


# ---------------------------------------------------------------------
# Filter Creation and Properties Tests
# ---------------------------------------------------------------------

class TestFilterProperties:
    """Test filter creation and mathematical properties."""

    def test_filter_creation(self, transform_config: Dict[str, int], transform: ShearletTransform) -> None:
        """Test basic filter creation properties."""
        transform.build((1, 64, 64, 1))

        # Reconstruct complex filter bank
        filter_bank = reconstruct_filter_bank(transform, center=True)

        # Check number of filters
        expected_filters = 1 + transform_config['scales'] * (transform_config['directions'] + 1)
        assert filter_bank.shape[0] == expected_filters, \
            f"Expected {expected_filters} filters, got {filter_bank.shape[0]}"

        # Check filter shapes and types
        assert filter_bank.shape[1:] == (64, 64), \
            f"Filters have wrong shape: {filter_bank.shape[1:]}"
        # Output of reconstruct is numpy complex64/128
        assert np.iscomplexobj(filter_bank), "Filters should be complex"

    def test_filter_size_adaptation(self, transform: ShearletTransform, input_shape: Tuple[int, int]) -> None:
        """Test if filters adapt to different input sizes."""
        transform.build((1, *input_shape, 1))

        filter_bank = reconstruct_filter_bank(transform)
        assert filter_bank.shape[1:] == input_shape, \
            f"Filter bank shape {filter_bank.shape[1:]} doesn't match input {input_shape}"

    def test_frequency_coverage(self, built_transform: ShearletTransform) -> None:
        """Test if filters provide adequate frequency coverage."""
        # Get centered filter bank for analysis
        filter_bank = reconstruct_filter_bank(built_transform, center=True)

        # Get magnitude responses and normalize
        responses = np.abs(filter_bank)
        responses = responses / (np.max(responses) + 1e-10)
        total_response = np.sum(responses, axis=0)

        # Check if total response is reasonable
        mean_response = np.mean(total_response)
        std_response = np.std(total_response)

        # Relaxed threshold since shearlet coverage is naturally non-uniform
        assert std_response / (mean_response + 1e-10) < 2.0, \
            "Frequency coverage is extremely uneven"

        # Check for severe coverage gaps
        min_response = np.min(total_response)
        assert min_response > 1e-3, \
            f"Found severe coverage gap with minimum response {min_response}"

    def test_low_pass_filter(self, built_transform: ShearletTransform) -> None:
        """Test properties of the low-pass filter."""
        # Must use centered filters to check [32, 32] as DC
        filter_bank = reconstruct_filter_bank(built_transform, center=True)

        low_pass = filter_bank[0]
        response = np.abs(low_pass)

        # Normalize the response
        response = response / (np.max(response) + 1e-10)

        # Check DC response (center point)
        h, w = response.shape
        cy, cx = h // 2, w // 2
        dc_response = response[cy, cx]
        assert np.abs(dc_response - 1.0) < 0.2, \
            f"Low-pass DC response is {dc_response}, expected close to 1.0"

        # Check high-frequency attenuation at corners
        edge_points = np.concatenate([
            response[0:5, 0:5],
            response[-5:, 0:5],
            response[0:5, -5:],
            response[-5:, -5:]
        ], axis=0)
        edge_response = np.mean(edge_points)
        assert edge_response < 0.3, \
            f"Low-pass high-frequency response is {edge_response}, expected <0.3"


# ---------------------------------------------------------------------
# Numerical Properties Tests
# ---------------------------------------------------------------------

class TestNumericalProperties:
    """Test numerical properties and directional sensitivity."""

    def test_directional_sensitivity(self, default_transform: ShearletTransform,
                                   input_shape: Tuple[int, int]) -> None:
        """Test if transform responds correctly to directional patterns."""
        h, w = input_shape

        # Create directional pattern with higher frequency for better testing
        x, y = np.meshgrid(
            np.linspace(-4, 4, w),
            np.linspace(-4, 4, h)
        )

        # Test patterns at distinct angles
        angles = [0, 90]
        responses = []

        for angle in angles:
            # Create pattern at specific angle with windowing
            theta = np.radians(angle)
            window = np.exp(-(x ** 2 + y ** 2) / 8.0)
            pattern = np.sin(4 * np.pi * (x * np.cos(theta) + y * np.sin(theta))) * window

            input_tensor = ops.convert_to_tensor(
                pattern[np.newaxis, :, :, np.newaxis],
                dtype="float32"
            )

            output = default_transform(input_tensor)
            # Use ops for reduction
            response = ops.max(ops.abs(output), axis=[1, 2])
            responses.append(response)

        # Compare responses at orthogonal angles
        # Convert to numpy for correlation calculation
        r0 = ops.convert_to_numpy(responses[0])
        r1 = ops.convert_to_numpy(responses[1])

        correlation = np.mean(r0 * r1) / (np.linalg.norm(r0) * np.linalg.norm(r1))

        assert correlation < 0.85, \
            f"Responses too similar for orthogonal angles: correlation = {correlation:.3f}"

    def test_scale_sensitivity(self, default_transform: ShearletTransform,
                             input_shape: Tuple[int, int]) -> None:
        """Test if transform responds appropriately to different scales."""
        h, w = input_shape

        # Create patterns at different scales
        x, y = np.meshgrid(
            np.linspace(-4, 4, w),
            np.linspace(-4, 4, h)
        )

        # Use distinct scales
        scales = [1, 4]
        responses = []

        for scale in scales:
            # Create pattern with Gaussian windowing
            window = np.exp(-(x ** 2 + y ** 2) / 8.0)
            pattern = np.sin(2 * np.pi * scale * (x + y)) * window

            input_tensor = ops.convert_to_tensor(
                pattern[np.newaxis, :, :, np.newaxis],
                dtype="float32"
            )

            output = default_transform(input_tensor)
            response = ops.max(ops.abs(output), axis=[1, 2])
            responses.append(response)

        # Compare responses at different scales
        r0 = ops.convert_to_numpy(responses[0])
        r1 = ops.convert_to_numpy(responses[1])

        correlation = np.mean(r0 * r1) / (np.linalg.norm(r0) * np.linalg.norm(r1))

        assert correlation < 0.80, \
            f"Responses too similar for distinct scales: correlation = {correlation:.3f}"

    def test_directional_response(self, built_transform: ShearletTransform,
                                angle_patterns: List[Tuple[float, tf.Tensor]]) -> None:
        """Test directional selectivity of the filters."""
        responses = []

        for angle, pattern in angle_patterns:
            output = built_transform(pattern)
            responses.append(ops.max(ops.abs(output), axis=[1, 2]))

        # Compare responses at orthogonal angles (0° vs 90°)
        # Index 0 is 0 deg, Index 2 is 90 deg
        r0 = ops.convert_to_numpy(responses[0])
        r2 = ops.convert_to_numpy(responses[2])

        correlation = np.mean(r0 * r2) / (np.linalg.norm(r0) * np.linalg.norm(r2))

        assert correlation < 0.3, \
            f"High correlation {correlation} between orthogonal directions"

    def test_scale_separation(self, built_transform: ShearletTransform,
                            scale_patterns: List[Tuple[int, tf.Tensor]]) -> None:
        """Test scale selectivity of the filters."""
        responses = []

        for scale, pattern in scale_patterns:
            output = built_transform(pattern)
            responses.append(ops.max(ops.abs(output), axis=[1, 2]))

        # Check scale separation
        for i in range(len(responses) - 1):
            r_i = ops.convert_to_numpy(responses[i])
            r_next = ops.convert_to_numpy(responses[i+1])

            correlation = np.mean(r_i * r_next) / (np.linalg.norm(r_i) * np.linalg.norm(r_next))

            assert correlation < 0.5, \
                f"High correlation {correlation} between scales {scale_patterns[i][0]} and {scale_patterns[i + 1][0]}"


# ---------------------------------------------------------------------
# Frame Properties and Stability Tests
# ---------------------------------------------------------------------

class TestFrameProperties:
    """Test mathematical frame properties of the shearlet system."""

    def test_frame_bounds(self, built_transform: ShearletTransform) -> None:
        """Test if filters satisfy frame bounds properties."""
        # Get squared magnitude responses
        # Shift doesn't affect energy, but consistency is good
        filter_bank = reconstruct_filter_bank(built_transform)
        responses_squared = np.abs(filter_bank) ** 2
        total_energy = np.sum(responses_squared, axis=0)

        # Get frame bounds
        min_energy = np.min(total_energy)
        max_energy = np.max(total_energy)

        # Test frame bounds ratio
        assert max_energy / (min_energy + 1e-5) < 4.0, \
            f"Frame bounds ratio {max_energy / (min_energy + 1e-5)} too large"

        # Test energy preservation
        mean_energy = np.mean(total_energy)
        assert np.abs(mean_energy - 1.0) < 0.2, \
            f"Mean energy {mean_energy} far from 1.0"

        # Test non-zero coverage
        assert min_energy > 1e-3, \
            f"Minimum energy {min_energy} too close to zero"

    def test_shearlet_properties(self) -> None:
        """Test critical properties of the shearlet transform."""
        # Create test instance
        transform = ShearletTransform(scales=3, directions=8)
        transform.build((1, 64, 64, 1))

        # Test frequency coverage
        filter_bank = reconstruct_filter_bank(transform)
        responses = np.abs(filter_bank)
        total_response = np.sum(responses, axis=0)
        min_response = np.min(total_response)
        assert min_response > 1e-3, "Coverage gap detected"

        # Test frame bounds
        responses_squared = np.abs(filter_bank) ** 2
        total_energy = np.sum(responses_squared, axis=0)

        min_energy = np.min(total_energy)
        max_energy = np.max(total_energy)
        frame_ratio = max_energy / (min_energy + 1e-6)

        assert frame_ratio < 4.0, "Frame bounds too large"

        # Test energy preservation
        mean_energy = np.mean(total_energy)
        assert abs(mean_energy - 1.0) < 0.2, "Energy not preserved"


# ---------------------------------------------------------------------
# Edge Cases and Robustness Tests
# ---------------------------------------------------------------------

class TestEdgeCases:
    """Test edge cases and numerical stability."""

    def test_numerical_stability(self, default_transform: ShearletTransform,
                               sample_input: Tuple[tf.Tensor, Tuple]) -> None:
        """Test numerical stability with extreme input values."""
        input_tensor, _ = sample_input

        # Test with very small values
        small_input = input_tensor * 1e-10
        output_small = default_transform(small_input)
        assert not ops.any(ops.isnan(output_small)), \
            "Transform produced NaN values for small input"

        # Test with very large values
        large_input = input_tensor * 1e10
        output_large = default_transform(large_input)
        assert not ops.any(ops.isnan(output_large)), \
            "Transform produced NaN values for large input"

    def test_mixed_precision(self, default_transform: ShearletTransform,
                           sample_input: Tuple[tf.Tensor, Tuple]) -> None:
        """Test behavior with different precision inputs."""
        input_tensor, _ = sample_input

        # Test with float16 input
        input_fp16 = ops.cast(input_tensor, "float16")
        output_fp16 = default_transform(input_fp16)
        # Layer should convert to float32 internally
        assert output_fp16.dtype == "float32", \
            f"Expected float32 output for float16 input, got {output_fp16.dtype}"

        # Test with float64 input
        input_fp64 = ops.cast(input_tensor, "float64")
        output_fp64 = default_transform(input_fp64)
        assert output_fp64.dtype == "float32", \
            f"Expected float32 output for float64 input, got {output_fp64.dtype}"

    def test_zero_input(self, default_transform: ShearletTransform) -> None:
        """Test behavior with zero input."""
        zero_input = ops.zeros((1, 64, 64, 1), dtype="float32")
        output = default_transform(zero_input)

        # Output should not be NaN or infinite
        assert not ops.any(ops.isnan(output)), "NaN output for zero input"

        # Checking infinity using standard math or where
        # Keras ops.isinf exists
        try:
            is_inf = ops.isinf(output)
            assert not ops.any(is_inf), "Infinite output for zero input"
        except AttributeError:
            # Fallback for older keras ops if isinf missing
            assert not ops.any(ops.equal(output, float('inf'))), "Infinite output for zero input"

    def test_single_pixel_input(self, default_transform: ShearletTransform) -> None:
        """Test behavior with single non-zero pixel."""
        # Using numpy to construct easily then convert
        single_pixel_np = np.zeros((1, 64, 64, 1), dtype=np.float32)
        single_pixel_np[0, 32, 32, 0] = 1.0

        single_pixel_input = ops.convert_to_tensor(single_pixel_np)

        output = default_transform(single_pixel_input)

        # Should produce valid output
        assert not ops.any(ops.isnan(output)), "NaN output for single pixel input"
        assert ops.any(ops.not_equal(output, 0.0)), "All zero output for single pixel input"

    def test_batch_consistency(self, default_transform: ShearletTransform) -> None:
        """Test that batch processing is consistent with individual processing."""
        # Create test inputs using standard random from keras if available, or numpy
        input1 = ops.convert_to_tensor(np.random.normal(size=(1, 64, 64, 1)).astype("float32"))
        input2 = ops.convert_to_tensor(np.random.normal(size=(1, 64, 64, 1)).astype("float32"))
        batch_input = ops.concatenate([input1, input2], axis=0)

        # Process individually
        output1 = default_transform(input1)
        output2 = default_transform(input2)

        # Process as batch
        batch_output = default_transform(batch_input)

        # Compare results
        np.testing.assert_allclose(
            ops.convert_to_numpy(output1),
            ops.convert_to_numpy(batch_output[0:1]),
            rtol=1e-6, atol=1e-6,
            err_msg="Batch processing inconsistent with individual processing"
        )

        np.testing.assert_allclose(
            ops.convert_to_numpy(output2),
            ops.convert_to_numpy(batch_output[1:2]),
            rtol=1e-6, atol=1e-6,
            err_msg="Batch processing inconsistent with individual processing"
        )


# ---------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])