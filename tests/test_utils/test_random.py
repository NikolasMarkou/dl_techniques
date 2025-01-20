import pytest
import numpy as np
import tensorflow as tf
from scipy import stats
from typing import Tuple, Optional

from dl_techniques.utils.random import rayleigh, validate_rayleigh_samples


@pytest.fixture(scope="module")
def set_random_seed():
    """Fixture to ensure reproducible test results."""
    tf.random.set_seed(42)
    np.random.seed(42)


def generate_test_samples(
        shape: Tuple[int, ...],
        scale: Optional[float] = None,
        seed: Optional[int] = None
) -> tf.Tensor:
    """Helper function to generate test samples."""
    return rayleigh(shape, scale=scale, dtype=tf.float32, seed=seed)


def test_rayleigh_shape_and_dtype(set_random_seed):
    """Test if the output has correct shape and dtype."""
    # Test cases with different shapes
    test_shapes = [
        (1000,),
        (100, 200),
        (50, 30, 20)
    ]

    for shape in test_shapes:
        samples = generate_test_samples(shape)

        # Check shape
        assert samples.shape == shape, f"Expected shape {shape}, got {samples.shape}"

        # Check dtype
        assert samples.dtype == tf.float32, f"Expected dtype float32, got {samples.dtype}"

        # Check if all values are positive
        assert tf.reduce_all(samples > 0), "All Rayleigh samples should be positive"


def test_rayleigh_scale_parameter(set_random_seed):
    """Test if the scale parameter affects the distribution correctly."""
    n_samples = 100000
    scales = [0.5, 1.0, 2.0, 5.0]

    for scale in scales:
        samples = generate_test_samples((n_samples,), scale=scale)

        # Calculate theoretical moments for Rayleigh distribution
        theoretical_mean = scale * np.sqrt(np.pi / 2)
        theoretical_std = scale * np.sqrt(2 - np.pi / 2)

        # Calculate sample statistics
        sample_mean = tf.reduce_mean(samples)
        sample_std = tf.math.reduce_std(samples)

        # Check if sample statistics are close to theoretical values
        # Using 5% relative tolerance due to random sampling
        np.testing.assert_allclose(
            sample_mean,
            theoretical_mean,
            rtol=0.05,
            err_msg=f"Mean test failed for scale={scale}"
        )
        np.testing.assert_allclose(
            sample_std,
            theoretical_std,
            rtol=0.05,
            err_msg=f"Standard deviation test failed for scale={scale}"
        )


def test_rayleigh_invalid_inputs():
    """Test if the function handles invalid inputs correctly."""
    with pytest.raises(ValueError, match="All dimensions in shape must be non-negative"):
        rayleigh([-1, 100])

    with pytest.raises(ValueError, match="Scale parameter must be positive"):
        rayleigh([100], scale=-1.0)

    with pytest.raises(ValueError, match="Scale parameter must be positive"):
        rayleigh([100], scale=0.0)


def test_rayleigh_distribution_properties(set_random_seed):
    """Test statistical properties of the generated distribution."""
    n_samples = 100000
    scale = 1.0
    samples = generate_test_samples((n_samples,), scale=scale)
    samples_np = samples.numpy()

    # Perform Kolmogorov-Smirnov test against theoretical Rayleigh distribution
    ks_statistic, p_value = stats.kstest(
        samples_np,
        'rayleigh',
        args=(0, scale)  # location (0) and scale parameters
    )

    # Check if p-value is above significance level (0.01)
    # This test confirms if the samples follow a Rayleigh distribution
    assert p_value > 0.01, (
        f"Kolmogorov-Smirnov test failed: p-value={p_value}, "
        f"statistic={ks_statistic}"
    )

    # Check basic properties of Rayleigh distribution
    validation_results = validate_rayleigh_samples(samples, scale=scale)

    # Verify that sample statistics are within expected ranges
    assert abs(validation_results['sample_mean'] - validation_results['theoretical_mean']) / validation_results[
        'theoretical_mean'] < 0.05
    assert abs(validation_results['sample_variance'] - validation_results['theoretical_variance']) / validation_results[
        'theoretical_variance'] < 0.05


def test_rayleigh_reproducibility():
    """Test if the random number generation is reproducible with seeds."""
    shape = (1000,)
    seed = 12345

    # Clear any existing random state
    tf.random.set_seed(None)

    # Generate two sets of samples with the same seed
    samples1 = generate_test_samples(shape, seed=seed)
    samples2 = generate_test_samples(shape, seed=seed)

    # Check if the samples are identical
    # Use np.allclose instead of exact equality due to floating point differences
    tf.debugging.assert_near(samples1, samples2, rtol=1e-5, atol=1e-5)

    # Generate samples with different seed
    samples3 = generate_test_samples(shape, seed=seed + 1)

    # Check if the samples are substantially different
    # We use mean absolute difference to check if samples are different
    mean_abs_diff = tf.reduce_mean(tf.abs(samples1 - samples3))
    assert mean_abs_diff > 0.1, (
        "Samples with different seeds should be substantially different"
    )


if __name__ == "__main__":
    pytest.main([__file__])
