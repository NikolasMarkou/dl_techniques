"""
Test suite for orthogonal regularization implementations.

This module provides comprehensive tests for:
- SoftOrthogonalConstraintRegularizer
- SoftOrthonormalConstraintRegularizer
- Helper functions
- Edge cases and numerical stability
- Serialization/deserialization
"""

import math
import pytest
import tensorflow as tf
import numpy as np
from typing import Any, Dict, Tuple

from dl_techniques.regularizers.soft_orthogonal import (
    SoftOrthogonalConstraintRegularizer,
    SoftOrthonormalConstraintRegularizer,
    DEFAULT_SOFTORTHOGONAL_LAMBDA,
    DEFAULT_SOFTORTHOGONAL_L1,
    DEFAULT_SOFTORTHOGONAL_L2
)


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


@pytest.fixture
def orthogonal_matrix() -> tf.Tensor:
    """Generate a known orthogonal matrix for testing."""
    # Create a simple 2x2 rotation matrix (orthogonal)
    theta = np.pi / 4  # 45 degrees
    matrix = tf.constant([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ], dtype=tf.float32)
    return matrix


@pytest.fixture
def conv_small() -> tf.Tensor:
    """Generate small random 4D conv weights for testing."""
    tf.random.set_seed(42)
    return tf.random.normal((3, 3, 8, 16))  # [h, w, in_channels, out_channels]


@pytest.fixture
def conv_medium() -> tf.Tensor:
    """Generate medium random 4D conv weights for testing."""
    tf.random.set_seed(42)
    return tf.random.normal((3, 3, 16, 32))  # [h, w, in_channels, out_channels]


@pytest.fixture
def conv_large() -> tf.Tensor:
    """Generate large random 4D conv weights for testing."""
    tf.random.set_seed(42)
    return tf.random.normal((3, 3, 32, 64))  # [h, w, in_channels, out_channels]


@pytest.fixture
def dense_small() -> tf.Tensor:
    """Generate small random 2D dense weights for testing."""
    tf.random.set_seed(42)
    return tf.random.normal((64, 32))  # [input_dim, output_dim]


@pytest.fixture
def dense_large() -> tf.Tensor:
    """Generate large random 2D dense weights for testing."""
    tf.random.set_seed(42)
    return tf.random.normal((256, 128))  # [input_dim, output_dim]


def test_soft_orthogonal_default_params() -> None:
    """Test SoftOrthogonalConstraintRegularizer with default parameters."""
    regularizer = SoftOrthogonalConstraintRegularizer()
    assert regularizer._lambda_coefficient == DEFAULT_SOFTORTHOGONAL_LAMBDA
    assert regularizer._l1_coefficient == DEFAULT_SOFTORTHOGONAL_L1
    assert regularizer._l2_coefficient == DEFAULT_SOFTORTHOGONAL_L2


def test_soft_orthogonal_custom_params() -> None:
    """Test SoftOrthogonalConstraintRegularizer with custom parameters."""
    params = {
        "lambda_coefficient": 0.1,
        "l1_coefficient": 0.2,
        "l2_coefficient": 0.3
    }
    regularizer = SoftOrthogonalConstraintRegularizer(**params)
    assert regularizer._lambda_coefficient == params["lambda_coefficient"]
    assert regularizer._l1_coefficient == params["l1_coefficient"]
    assert regularizer._l2_coefficient == params["l2_coefficient"]


def test_soft_orthogonal_zero_for_orthogonal_matrix(orthogonal_matrix: tf.Tensor) -> None:
    """Test that regularization is zero for perfectly orthogonal matrix."""
    regularizer = SoftOrthogonalConstraintRegularizer(
        lambda_coefficient=1.0,
        l1_coefficient=0.0,
        l2_coefficient=0.0
    )
    penalty = regularizer(orthogonal_matrix)
    assert tf.abs(penalty) < 1e-5


def test_soft_orthogonal_nonzero_for_nonorthogonal(random_weights_2d: tf.Tensor) -> None:
    """Test that regularization is nonzero for non-orthogonal matrix."""
    regularizer = SoftOrthogonalConstraintRegularizer(
        lambda_coefficient=1.0,
        l1_coefficient=0.0,
        l2_coefficient=0.0
    )
    penalty = regularizer(random_weights_2d)
    assert penalty > 1e-5


# SoftOrthonormalConstraintRegularizer tests
def test_soft_orthonormal_default_params() -> None:
    """Test SoftOrthonormalConstraintRegularizer with default parameters."""
    regularizer = SoftOrthonormalConstraintRegularizer()
    assert regularizer._lambda_coefficient == DEFAULT_SOFTORTHOGONAL_LAMBDA
    assert regularizer._l1_coefficient == DEFAULT_SOFTORTHOGONAL_L1
    assert regularizer._l2_coefficient == DEFAULT_SOFTORTHOGONAL_L2


def test_soft_orthonormal_custom_params() -> None:
    """Test SoftOrthonormalConstraintRegularizer with custom parameters."""
    params = {
        "lambda_coefficient": 0.1,
        "l1_coefficient": 0.2,
        "l2_coefficient": 0.3
    }
    regularizer = SoftOrthonormalConstraintRegularizer(**params)
    assert regularizer._lambda_coefficient == params["lambda_coefficient"]
    assert regularizer._l1_coefficient == params["l1_coefficient"]
    assert regularizer._l2_coefficient == params["l2_coefficient"]


def test_soft_orthonormal_zero_for_orthonormal_matrix(orthogonal_matrix: tf.Tensor) -> None:
    """Test that regularization is zero for perfectly orthonormal matrix."""
    regularizer = SoftOrthonormalConstraintRegularizer(
        lambda_coefficient=1.0,
        l1_coefficient=0.0,
        l2_coefficient=0.0
    )
    penalty = regularizer(orthogonal_matrix)
    assert tf.abs(penalty) < 1e-5


def test_soft_orthonormal_nonzero_for_nonorthonormal(random_weights_2d: tf.Tensor) -> None:
    """Test that regularization is nonzero for non-orthonormal matrix."""
    regularizer = SoftOrthonormalConstraintRegularizer(
        lambda_coefficient=1.0,
        l1_coefficient=0.0,
        l2_coefficient=0.0
    )
    penalty = regularizer(random_weights_2d)
    assert penalty > 1e-5


# Serialization tests
def test_soft_orthogonal_serialization() -> None:
    """Test serialization/deserialization of SoftOrthogonalConstraintRegularizer."""
    original = SoftOrthogonalConstraintRegularizer(
        lambda_coefficient=0.1,
        l1_coefficient=0.2,
        l2_coefficient=0.3
    )
    config = original.get_config()
    reconstructed = SoftOrthogonalConstraintRegularizer.from_config(config)
    assert original._lambda_coefficient == reconstructed._lambda_coefficient
    assert original._l1_coefficient == reconstructed._l1_coefficient
    assert original._l2_coefficient == reconstructed._l2_coefficient


def test_soft_orthonormal_serialization() -> None:
    """Test serialization/deserialization of SoftOrthonormalConstraintRegularizer."""
    original = SoftOrthonormalConstraintRegularizer(
        lambda_coefficient=0.1,
        l1_coefficient=0.2,
        l2_coefficient=0.3
    )
    config = original.get_config()
    reconstructed = SoftOrthonormalConstraintRegularizer.from_config(config)
    assert original._lambda_coefficient == reconstructed._lambda_coefficient
    assert original._l1_coefficient == reconstructed._l1_coefficient
    assert original._l2_coefficient == reconstructed._l2_coefficient


# Edge cases and numerical stability tests
@pytest.mark.parametrize("shape", [
    (1, 1),
    (100, 100)
])
def test_valid_shapes(shape: Tuple[int, int]) -> None:
    """Test regularizers with valid shapes."""
    weights = tf.random.normal(shape)
    regularizer = SoftOrthogonalConstraintRegularizer()
    penalty = regularizer(weights)
    assert not tf.math.is_nan(penalty)
    assert not tf.math.is_inf(penalty)


def test_numerical_stability_large_values() -> None:
    """Test numerical stability with large values."""
    weights = tf.random.normal((10, 5)) * 1e6
    regularizer = SoftOrthogonalConstraintRegularizer()
    penalty = regularizer(weights)
    assert not tf.math.is_nan(penalty)
    assert not tf.math.is_inf(penalty)


def test_numerical_stability_small_values() -> None:
    """Test numerical stability with small values."""
    weights = tf.random.normal((10, 5)) * 1e-6
    regularizer = SoftOrthogonalConstraintRegularizer()
    penalty = regularizer(weights)
    assert not tf.math.is_nan(penalty)
    assert not tf.math.is_inf(penalty)


# Integration tests
def test_integration_with_keras_layer() -> None:
    """Test regularizer integration with Keras layer."""
    # Enable eager execution for this test
    tf.config.run_functions_eagerly(True)

    try:
        regularizer = SoftOrthogonalConstraintRegularizer()

        # Suppress warnings about input_shape by using Input layer
        inputs = tf.keras.Input(shape=(5,))
        outputs = tf.keras.layers.Dense(
            10,
            kernel_regularizer=regularizer
        )(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        model.compile(optimizer='adam', loss='mse')

        # Generate random data
        x = tf.random.normal((100, 5))
        y = tf.random.normal((100, 10))

        # Check that training works without errors
        with tf.keras.utils.custom_object_scope({
            'SoftOrthogonalConstraintRegularizer': SoftOrthogonalConstraintRegularizer
        }):
            history = model.fit(x, y, epochs=1, verbose=0)
            assert len(history.history['loss']) == 1

    finally:
        # Restore original execution mode
        tf.config.run_functions_eagerly(False)


# ---------------------------------------------------------------------


def test_scaling_effect_on_regularization_strength(conv_small: tf.Tensor, conv_large: tf.Tensor) -> None:
    """
    Test 1: Verify that matrix scaling affects regularization values as expected.

    This test confirms that:
    1. Regularization values differ when scaling is enabled
    2. Without scaling, larger matrices have much higher regularization values
    3. With scaling, regularization values are more proportional to matrix size
    """
    # Create regularizers with and without scaling
    ortho_no_scaling = SoftOrthogonalConstraintRegularizer(
        lambda_coefficient=1.0,
        l1_coefficient=0.0,
        l2_coefficient=0.0,
        use_matrix_scaling=False
    )

    ortho_with_scaling = SoftOrthogonalConstraintRegularizer(
        lambda_coefficient=1.0,
        l1_coefficient=0.0,
        l2_coefficient=0.0,
        use_matrix_scaling=True
    )

    # Calculate regularization values
    small_no_scaling = ortho_no_scaling(conv_small)
    small_with_scaling = ortho_with_scaling(conv_small)
    large_no_scaling = ortho_no_scaling(conv_large)
    large_with_scaling = ortho_with_scaling(conv_large)

    # Calculate ratios of large to small
    ratio_no_scaling = large_no_scaling / small_no_scaling
    ratio_with_scaling = large_with_scaling / small_with_scaling

    # Print values for diagnostic purposes
    print(f"Small Conv - No scaling: {small_no_scaling:.6f}, With scaling: {small_with_scaling:.6f}")
    print(f"Large Conv - No scaling: {large_no_scaling:.6f}, With scaling: {large_with_scaling:.6f}")
    print(f"Ratio - No scaling: {ratio_no_scaling:.6f}, With scaling: {ratio_with_scaling:.6f}")

    # Assert expected behavior
    assert small_no_scaling != small_with_scaling, "Scaling should affect regularization values"
    assert ratio_no_scaling > ratio_with_scaling, "Scaling should reduce the ratio between large and small"

    # Derivation (soft_orthogonal.py module docstring, "Size normalization"):
    # `use_matrix_scaling` divides the ENTIRE regularization value -- orthogonality term
    # plus L1 plus L2 -- by sqrt(rank), where `rank` is the side length of the Gram matrix
    # actually used, i.e. rank = min(units, fan_in).
    #
    #   conv_small  (3, 3,  8, 16):  fan_in = 3*3*8  =  72, units = 16 -> rank = min(16,  72) = 16
    #   conv_large  (3, 3, 32, 64):  fan_in = 3*3*32 = 288, units = 64 -> rank = min(64, 288) = 64
    #
    # The divisor is a per-tensor constant, so toggling the flag is a pure global gain and
    # the large/small ratio transforms EXACTLY:
    #
    #   ratio_with = (large / sqrt(64)) / (small / sqrt(16))
    #              = ratio_no * sqrt(16) / sqrt(64)
    #              = ratio_no * 4 / 8
    #              = ratio_no * 0.5                                  <-- an EQUALITY, not a bound
    #
    # WARNING to a future reader: the superseded assertion here was
    #     assert ratio_with_scaling < ratio_no_scaling * 0.5
    # Under sqrt(rank) that is an exact-boundary STRICT inequality and can NEVER hold. It
    # used to pass only because the OLD divisor was rank**2, which gives a factor of
    #     16**2 / 64**2 = 256 / 4096 = 1/16,
    # comfortably inside the `* 0.5` bound. The test did not drift; it was measuring a
    # divisor that no longer exists. Do NOT "restore" the strict bound.
    expected_factor = math.sqrt(16.0) / math.sqrt(64.0)
    assert expected_factor == 0.5, "sqrt(16)/sqrt(64) is exactly 0.5"
    assert float(ratio_with_scaling) == pytest.approx(
        float(ratio_no_scaling) * expected_factor, rel=1e-6
    ), "Toggling use_matrix_scaling rescales the large/small ratio by exactly sqrt(16)/sqrt(64)"

    # The same identity read one tensor at a time: R(False) / R(True) == sqrt(rank).
    # This inverts a published identity rather than recomputing `_kernel_gram`'s reshape,
    # so it reads `rank` OUT of the implementation instead of re-implementing it.
    assert float(small_no_scaling / small_with_scaling) == pytest.approx(
        math.sqrt(16.0), rel=1e-6
    ), "conv_small rank is min(16, 72) = 16, so the divisor is sqrt(16) = 4"
    assert float(large_no_scaling / large_with_scaling) == pytest.approx(
        math.sqrt(64.0), rel=1e-6
    ), "conv_large rank is min(64, 288) = 64, so the divisor is sqrt(64) = 8"


def test_scaling_across_filter_sizes() -> None:
    """
    Test 2: pin the sqrt(rank) divisor across filter counts, INCLUDING the Gram
    orientation flip at units > fan_in.

    Every kernel here is (3, 3, 3, filters), so fan_in = 3*3*3 = 27 for all of them while
    `units` sweeps past it. `rank` is therefore min(filters, 27) and SATURATES at 27, which
    means the classic premise "scaling always reduces the growth rate" is FALSE for the
    saturated pairs -- their per-pair factor is exactly 1.0. That premise is deliberately
    not asserted here.
    """
    filter_sizes = [8, 16, 32, 64, 128]

    # Derivation: rank = min(units, fan_in) with fan_in = 27 throughout.
    #   filters =   8 -> min(  8, 27) =  8   units <= fan_in: Gram is W^T W, side = units
    #   filters =  16 -> min( 16, 27) = 16   units <= fan_in
    #   filters =  32 -> min( 32, 27) = 27   ORIENTATION FLIP: units > fan_in, Gram is W W^T
    #   filters =  64 -> min( 64, 27) = 27   saturated
    #   filters = 128 -> min(128, 27) = 27   saturated
    expected_ranks = [8, 16, 27, 27, 27]

    results_no_scaling = []
    results_with_scaling = []

    ortho_no_scaling = SoftOrthonormalConstraintRegularizer(
        lambda_coefficient=1.0,
        l1_coefficient=0.0,
        l2_coefficient=0.0,
        use_matrix_scaling=False
    )

    ortho_with_scaling = SoftOrthonormalConstraintRegularizer(
        lambda_coefficient=1.0,
        l1_coefficient=0.0,
        l2_coefficient=0.0,
        use_matrix_scaling=True
    )

    for filters in filter_sizes:
        tf.random.set_seed(42)  # For reproducibility
        weights = tf.random.normal((3, 3, 3, filters))

        val_no_scaling = float(ortho_no_scaling(weights).numpy())
        val_with_scaling = float(ortho_with_scaling(weights).numpy())

        results_no_scaling.append(val_no_scaling)
        results_with_scaling.append(val_with_scaling)

        print(f"Filters: {filters}, No scaling: {val_no_scaling:.6f}, With scaling: {val_with_scaling:.6f}")

    # ---- The rank sequence, read OUT of the implementation via the published identity ----
    # R(use_matrix_scaling=False) / R(use_matrix_scaling=True) == sqrt(rank), because the
    # divisor is a uniform per-tensor constant. Squaring that observable ratio therefore
    # recovers `rank` without re-implementing `_kernel_gram`'s reshape logic (a test that
    # recomputes the code's own math is just a second copy of it).
    implied_ranks = [
        (no / with_) ** 2
        for no, with_ in zip(results_no_scaling, results_with_scaling)
    ]
    print(f"Implied ranks: {implied_ranks}  (expected {expected_ranks})")

    # This is the assertion that pins the Gram ORIENTATION FLIP at units > fan_in: nothing
    # else in this suite can see it. If it fails, the rank model -- not the tolerance -- is
    # wrong, and the correct response is to report it, not to retune this expectation.
    for filters, expected_rank, implied_rank in zip(filter_sizes, expected_ranks, implied_ranks):
        assert implied_rank == pytest.approx(float(expected_rank), rel=1e-5), (
            f"filters={filters}: expected rank min({filters}, 27) = {expected_rank}, "
            f"implied {implied_rank}"
        )

    # ---- Per-pair growth-rate factor ----
    growth_rates_no_scaling = [results_no_scaling[i] / results_no_scaling[i - 1]
                               for i in range(1, len(results_no_scaling))]

    growth_rates_with_scaling = [results_with_scaling[i] / results_with_scaling[i - 1]
                                 for i in range(1, len(results_with_scaling))]

    print(f"Growth rates without scaling: {growth_rates_no_scaling}")
    print(f"Growth rates with scaling: {growth_rates_with_scaling}")

    # Derivation: g_with[i] / g_no[i] = (r_{i-1} / r_i) ** 0.5, since each value carries a
    # 1/sqrt(rank) factor and the ratio keeps only the two ranks involved.
    #   8  -> 16 : sqrt( 8/16) = sqrt(0.5)      = 0.707107
    #   16 -> 32 : sqrt(16/27)                  = 0.769800
    #   32 -> 64 : sqrt(27/27) = 1.0            <-- rank saturated, scaling changes NOTHING
    #   64 -> 128: sqrt(27/27) = 1.0            <-- rank saturated
    expected_pair_factors = [
        math.sqrt(expected_ranks[i - 1] / expected_ranks[i])
        for i in range(1, len(expected_ranks))
    ]
    assert expected_pair_factors[0] == pytest.approx(math.sqrt(0.5), rel=1e-12)
    assert expected_pair_factors[2] == 1.0 and expected_pair_factors[3] == 1.0

    for i, (g_no, g_with, factor) in enumerate(
            zip(growth_rates_no_scaling, growth_rates_with_scaling, expected_pair_factors)):
        assert g_with == pytest.approx(g_no * factor, rel=1e-5), (
            f"pair {filter_sizes[i]}->{filter_sizes[i + 1]}: expected the scaled growth "
            f"rate to be {factor} times the unscaled one"
        )

    # The superseded premise `all(g_no > g_with)` is FALSE once rank saturates: the last
    # two factors are exactly 1.0, so the growth rates are EQUAL there. Assert the true
    # shape instead -- strictly smaller while rank is still growing, equal afterwards.
    assert growth_rates_with_scaling[0] < growth_rates_no_scaling[0]
    assert growth_rates_with_scaling[1] < growth_rates_no_scaling[1]
    assert growth_rates_with_scaling[2] == pytest.approx(growth_rates_no_scaling[2], rel=1e-5)
    assert growth_rates_with_scaling[3] == pytest.approx(growth_rates_no_scaling[3], rel=1e-5)

    # ---- Smallest-to-largest ----
    # ratio_with / ratio_no = sqrt(r_first / r_last) = sqrt(8/27) = 0.544331 < 1, so the
    # end-to-end statement "scaling reduces the largest/smallest ratio" still holds.
    ratio_no_scaling = results_no_scaling[-1] / results_no_scaling[0]
    ratio_with_scaling = results_with_scaling[-1] / results_with_scaling[0]

    print(
        f"Ratio from smallest to largest - No scaling: {ratio_no_scaling:.6f}, With scaling: {ratio_with_scaling:.6f}")
    end_to_end_factor = math.sqrt(8.0 / 27.0)
    assert end_to_end_factor == pytest.approx(0.5443310, abs=1e-6)
    assert ratio_with_scaling == pytest.approx(ratio_no_scaling * end_to_end_factor, rel=1e-5)
    assert ratio_with_scaling < ratio_no_scaling, "Scaling should reduce the ratio between largest and smallest"


@pytest.mark.parametrize("lambda_coefficient,l1_coefficient,l2_coefficient,term_name", [
    (0.0, 1.0, 0.0, "l1-only"),
    (0.0, 0.0, 1.0, "l2-only"),
    (1.0, 0.0, 0.0, "orthogonality-only"),
])
@pytest.mark.parametrize("regularizer_cls", [
    SoftOrthogonalConstraintRegularizer,
    SoftOrthonormalConstraintRegularizer,
])
def test_matrix_scaling_divisor_applies_to_every_term(
        regularizer_cls: Any,
        lambda_coefficient: float,
        l1_coefficient: float,
        l2_coefficient: float,
        term_name: str
) -> None:
    """
    Test the "pure global gain" property: the `use_matrix_scaling` divisor is applied to
    the L1 and L2 terms as well as to the orthogonality term.

    Derivation (soft_orthogonal.py module docstring): the divisor is uniform across terms,
    so for ANY coefficient triple

        R(use_matrix_scaling=False) / R(use_matrix_scaling=True) == sqrt(rank)

    Kernel is (3, 3, 3, 16): fan_in = 3*3*3 = 27, units = 16, rank = min(16, 27) = 16,
    so the expected ratio is sqrt(16) = 4.0 for EVERY term in isolation.

    Zeroing two of the three coefficients is what makes this discriminating. With all three
    terms active the orthogonality term dominates the total at the default coefficients, so
    the measured ratio would read ~sqrt(rank) whether the divisor were uniform or applied
    to the orthogonality term alone. Under the SUPERSEDED behaviour (divisor on the
    orthogonality term only, and rank**2 at that) the l1-only and l2-only rows would read
    exactly 1.0 while the orthogonality row read rank**2 = 256.
    """
    tf.random.set_seed(42)
    weights = tf.random.normal((3, 3, 3, 16))

    expected_ratio = math.sqrt(16.0)  # sqrt(min(16, 27)) = sqrt(16) = 4.0
    assert expected_ratio == 4.0

    no_scaling = regularizer_cls(
        lambda_coefficient=lambda_coefficient,
        l1_coefficient=l1_coefficient,
        l2_coefficient=l2_coefficient,
        use_matrix_scaling=False
    )
    with_scaling = regularizer_cls(
        lambda_coefficient=lambda_coefficient,
        l1_coefficient=l1_coefficient,
        l2_coefficient=l2_coefficient,
        use_matrix_scaling=True
    )

    value_no_scaling = float(no_scaling(weights).numpy())
    value_with_scaling = float(with_scaling(weights).numpy())

    # Anti-vacuity: the isolated term must actually produce a penalty, otherwise the ratio
    # below would be 0/0 and the test would pass without measuring anything.
    assert value_with_scaling > 0.0, f"{term_name} produced no penalty; the ratio is vacuous"

    ratio = value_no_scaling / value_with_scaling
    print(f"{regularizer_cls.__name__} {term_name}: R(False)/R(True) = {ratio:.6f}")
    assert ratio == pytest.approx(expected_ratio, rel=1e-5), (
        f"{term_name}: the sqrt(rank) divisor must apply to this term too"
    )


def test_scaling_consistency_between_dense_and_conv(
        dense_small: tf.Tensor,
        dense_large: tf.Tensor,
        conv_small: tf.Tensor,
        conv_large: tf.Tensor
) -> None:
    """
    Test 3: Verify scaling consistency between dense and convolutional layers.

    This test confirms that:
    1. Scaling behavior is consistent between different layer types
    2. Ratios between large and small layers are similar with scaling enabled
    """
    # Create regularizers
    ortho_with_scaling = SoftOrthogonalConstraintRegularizer(
        lambda_coefficient=1.0,
        l1_coefficient=0.0,
        l2_coefficient=0.0,
        use_matrix_scaling=True
    )

    # Calculate regularization values
    dense_small_val = ortho_with_scaling(dense_small)
    dense_large_val = ortho_with_scaling(dense_large)
    conv_small_val = ortho_with_scaling(conv_small)
    conv_large_val = ortho_with_scaling(conv_large)

    # Calculate ratios
    dense_ratio = dense_large_val / dense_small_val
    conv_ratio = conv_large_val / conv_small_val

    print(f"Dense small: {dense_small_val:.6f}, large: {dense_large_val:.6f}, ratio: {dense_ratio:.6f}")
    print(f"Conv small: {conv_small_val:.6f}, large: {conv_large_val:.6f}, ratio: {conv_ratio:.6f}")

    # The ratios should be reasonably close with scaling enabled
    # We use a relatively loose bound since different layer types have
    # different characteristics
    ratio_difference = abs(dense_ratio.numpy() - conv_ratio.numpy())
    print(f"Ratio difference: {ratio_difference:.6f}")

    # Assert the ratio difference is within reasonable bounds
    assert ratio_difference < max(dense_ratio.numpy(), conv_ratio.numpy()) * 0.5, \
        "Ratios between dense and conv should be reasonably similar with scaling"


def test_orthogonal_vs_orthonormal_scaling(conv_medium: tf.Tensor) -> None:
    """
    Test 4: Compare scaling behavior between orthogonal and orthonormal regularizers.

    This test confirms that:
    1. Both orthogonal and orthonormal regularizers are affected by scaling
    2. The scaling implementation is appropriate for each regularizer type
    """
    # Create regularizers - both with and without scaling
    orthogonal_no_scaling = SoftOrthogonalConstraintRegularizer(
        lambda_coefficient=1.0,
        l1_coefficient=0.0,
        l2_coefficient=0.0,
        use_matrix_scaling=False
    )

    orthogonal_with_scaling = SoftOrthogonalConstraintRegularizer(
        lambda_coefficient=1.0,
        l1_coefficient=0.0,
        l2_coefficient=0.0,
        use_matrix_scaling=True
    )

    orthonormal_no_scaling = SoftOrthonormalConstraintRegularizer(
        lambda_coefficient=1.0,
        l1_coefficient=0.0,
        l2_coefficient=0.0,
        use_matrix_scaling=False
    )

    orthonormal_with_scaling = SoftOrthonormalConstraintRegularizer(
        lambda_coefficient=1.0,
        l1_coefficient=0.0,
        l2_coefficient=0.0,
        use_matrix_scaling=True
    )

    # Calculate regularization values
    orthogonal_no_scale_val = orthogonal_no_scaling(conv_medium)
    orthogonal_with_scale_val = orthogonal_with_scaling(conv_medium)
    orthonormal_no_scale_val = orthonormal_no_scaling(conv_medium)
    orthonormal_with_scale_val = orthonormal_with_scaling(conv_medium)

    print(f"Orthogonal - No scaling: {orthogonal_no_scale_val:.6f}, With scaling: {orthogonal_with_scale_val:.6f}")
    print(f"Orthonormal - No scaling: {orthonormal_no_scale_val:.6f}, With scaling: {orthonormal_with_scale_val:.6f}")

    # Calculate scaling effect ratios
    orthogonal_scale_ratio = orthogonal_no_scale_val / orthogonal_with_scale_val
    orthonormal_scale_ratio = orthonormal_no_scale_val / orthonormal_with_scale_val

    print(
        f"Scaling effect ratio - Orthogonal: {orthogonal_scale_ratio:.6f}, Orthonormal: {orthonormal_scale_ratio:.6f}")

    # Both regularizers should have their values reduced by scaling
    assert orthogonal_with_scale_val < orthogonal_no_scale_val, "Scaling should reduce orthogonal regularization value"
    assert orthonormal_with_scale_val < orthonormal_no_scale_val, "Scaling should reduce orthonormal regularization value"

    # The scaling effect should be substantial for both types
    assert orthogonal_scale_ratio > 2.0, "Scaling should substantially reduce orthogonal regularization"
    assert orthonormal_scale_ratio > 2.0, "Scaling should substantially reduce orthonormal regularization"

    # The scaling effect might differ between types due to diagonal vs. off-diagonal elements
    # but should be in a reasonable range
    ratio_difference = abs(orthogonal_scale_ratio.numpy() - orthonormal_scale_ratio.numpy())
    print(f"Scaling effect ratio difference: {ratio_difference:.6f}")

    # The difference in scaling effect should not be extreme
    max_ratio = max(orthogonal_scale_ratio.numpy(), orthonormal_scale_ratio.numpy())
    assert ratio_difference < max_ratio * 0.5, "Scaling effect should be reasonably similar for both regularizer types"


# ---------------------------------------------------------------------
# Additional coverage tests
# ---------------------------------------------------------------------


class TestNegativeCoefficientValidation:
    """Test that negative coefficients raise ValueError."""

    def test_negative_lambda(self) -> None:
        with pytest.raises(ValueError, match="lambda_coefficient must be non-negative"):
            SoftOrthogonalConstraintRegularizer(lambda_coefficient=-0.1)

    def test_negative_l1(self) -> None:
        with pytest.raises(ValueError, match="l1_coefficient must be non-negative"):
            SoftOrthogonalConstraintRegularizer(l1_coefficient=-0.1)

    def test_negative_l2(self) -> None:
        with pytest.raises(ValueError, match="l2_coefficient must be non-negative"):
            SoftOrthogonalConstraintRegularizer(l2_coefficient=-0.1)

    def test_negative_lambda_orthonormal(self) -> None:
        with pytest.raises(ValueError, match="lambda_coefficient must be non-negative"):
            SoftOrthonormalConstraintRegularizer(lambda_coefficient=-0.1)

    def test_negative_l1_orthonormal(self) -> None:
        with pytest.raises(ValueError, match="l1_coefficient must be non-negative"):
            SoftOrthonormalConstraintRegularizer(l1_coefficient=-0.1)

    def test_negative_l2_orthonormal(self) -> None:
        with pytest.raises(ValueError, match="l2_coefficient must be non-negative"):
            SoftOrthonormalConstraintRegularizer(l2_coefficient=-0.1)


class TestGradientFlow:
    """Test that regularizers produce useful gradients."""

    def test_orthogonal_gradient_nonzero(self) -> None:
        """Verify gradients flow through the orthogonal regularizer."""
        reg = SoftOrthogonalConstraintRegularizer(lambda_coefficient=1e-3, l2_coefficient=0.0)
        weights = tf.Variable(tf.random.normal((8, 4), seed=42))
        with tf.GradientTape() as tape:
            loss = reg(weights)
        grad = tape.gradient(loss, weights)
        assert grad is not None, "Gradient must not be None"
        assert tf.reduce_any(tf.not_equal(grad, 0.0)), "Gradient must be non-zero"

    def test_orthonormal_gradient_nonzero(self) -> None:
        """Verify gradients flow through the orthonormal regularizer."""
        reg = SoftOrthonormalConstraintRegularizer(lambda_coefficient=1e-3, l2_coefficient=0.0)
        weights = tf.Variable(tf.random.normal((8, 4), seed=42))
        with tf.GradientTape() as tape:
            loss = reg(weights)
        grad = tape.gradient(loss, weights)
        assert grad is not None, "Gradient must not be None"
        assert tf.reduce_any(tf.not_equal(grad, 0.0)), "Gradient must be non-zero"

    def test_orthogonal_gradient_zero_for_orthogonal_weights(self) -> None:
        """Orthogonal regularizer gradient should be near-zero for orthogonal weights."""
        reg = SoftOrthogonalConstraintRegularizer(
            lambda_coefficient=1e-3, l1_coefficient=0.0, l2_coefficient=0.0
        )
        theta = np.pi / 4
        weights = tf.Variable(tf.constant([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ], dtype=tf.float32))
        with tf.GradientTape() as tape:
            loss = reg(weights)
        grad = tape.gradient(loss, weights)
        assert grad is not None
        assert tf.reduce_all(tf.abs(grad) < 1e-5), \
            f"Gradient should be near-zero for orthogonal weights, got max={tf.reduce_max(tf.abs(grad)):.6e}"


class TestSerializationRoundTrip:
    """Test full round-trip: create -> get_config -> from_config -> identical output."""

    def test_orthogonal_roundtrip_identical_output(self) -> None:
        original = SoftOrthogonalConstraintRegularizer(
            lambda_coefficient=0.05, l1_coefficient=0.01, l2_coefficient=0.02
        )
        config = original.get_config()
        restored = SoftOrthogonalConstraintRegularizer(**config)

        weights = tf.random.normal((10, 5), seed=42)
        original_loss = original(weights)
        restored_loss = restored(weights)
        np.testing.assert_allclose(
            original_loss.numpy(), restored_loss.numpy(), atol=1e-7,
            err_msg="Round-trip serialization must produce identical loss"
        )

    def test_orthonormal_roundtrip_identical_output(self) -> None:
        original = SoftOrthonormalConstraintRegularizer(
            lambda_coefficient=0.05, l1_coefficient=0.01, l2_coefficient=0.02
        )
        config = original.get_config()
        restored = SoftOrthonormalConstraintRegularizer(**config)

        weights = tf.random.normal((10, 5), seed=42)
        original_loss = original(weights)
        restored_loss = restored(weights)
        np.testing.assert_allclose(
            original_loss.numpy(), restored_loss.numpy(), atol=1e-7,
            err_msg="Round-trip serialization must produce identical loss"
        )
