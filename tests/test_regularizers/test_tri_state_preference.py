"""
Tests for the Tri-State Preference Regularizer implementation.

This test suite verifies the functionality of the TriStatePreferenceRegularizer
including its mathematical properties, integration with Keras, and edge cases.
The regularizer should encourage weights to converge to -1, 0, or 1.
"""

import logging
import math

import pytest
import tensorflow as tf
from keras.api.layers import Dense, Input
from keras.api.models import Sequential
from dl_techniques.regularizers.tri_state_preference import (
    BARRIER_NORMALIZATION,
    TriStatePreferenceRegularizer,
)


@pytest.fixture
def regularizer():
    """Fixture providing a default regularizer instance."""
    return TriStatePreferenceRegularizer(scale=1.0)


def test_regularizer_initialization():
    """Pin the constructor defaults of the current (target-based) API.

    `scale` and `base_coefficient` no longer exist. The old assertion
    `base_coefficient == 32/4.5` literally pinned the superseded barrier
    constant, which was 5.35% too large.
    """
    reg = TriStatePreferenceRegularizer()

    assert reg.target == 1.0
    # `reg.multiplier` is a keras.Variable when annealable=True, and comparing a
    # Variable with `==` yields a truthy tensor rather than a bool -- such an
    # assertion cannot fail. Read the float property instead.
    assert reg.multiplier_value == 1.0
    assert isinstance(reg.multiplier_value, float)
    assert reg.reduction == "sum"
    assert reg.quadratic_tails is False
    assert reg.annealable is True


def test_barrier_normalization_constant():
    """The leading constant is 27/4, derived, not measured.

    L(w) = m * C * w^2 (w-t)^2 (w+t)^2 / t^6.  Substituting u = (w/t)^2 turns
    the shape factor into f(u) = u (u-1)^2, whose derivative is
    f'(u) = (3u - 1)(u - 1); the interior stationary point is u = 1/3, i.e.
    w = +/- t/sqrt(3), where f(1/3) = (1/3)(2/3)^2 = 4/27.
    Normalizing the barrier height to exactly 1 therefore requires
    C = 1 / (4/27) = 27/4 = 6.75.
    """
    assert BARRIER_NORMALIZATION == 27.0 / 4.0


def test_stable_points_zero_cost(regularizer):
    """Test that stable points (-1, 0, 1) produce zero cost."""
    # Create tensor with stable points
    stable_weights = tf.constant([[-1.0, 0.0, 1.0]], dtype=tf.float32)
    cost = regularizer(stable_weights)

    # Cost should be very close to zero
    assert tf.abs(cost) < 1e-6


@pytest.mark.parametrize(
    "reduction, n_elements, expected",
    [
        # Per-element barrier height is EXACTLY `multiplier` (= 1.0 here):
        #   (27/4) * f(1/3) = (27/4) * (4/27) = 1 exactly.
        # "sum" adds one barrier per element; 4 elements -> 4 * 1.0 = 4.0.
        ("sum", 4, 4.0),
        # "mean" averages them -> 1.0, independent of element count.
        ("mean", 4, 1.0),
    ],
)
def test_barrier_height_at_true_maximum(reduction, n_elements, expected):
    """Pin the TRUE barrier height, at w = +/- target/sqrt(3).

    With u = (w/t)^2 the shape factor is f(u) = u (u-1)^2 and
    f'(u) = (3u - 1)(u - 1), so the interior maxima are at u = 1/3, i.e.
    |w| = t / sqrt(3) ~ 0.57735 t -- NOT at 0.5 t, which is what the previous
    (32/4.5) constant encoded. There f(1/3) = 4/27, and the leading 27/4
    cancels it: (27/4) * (4/27) = 1 EXACTLY. So the per-element penalty at the
    maximum is exactly `multiplier`, for every target.
    """
    t = 1.0
    w = t / math.sqrt(3.0)
    reg = TriStatePreferenceRegularizer(
        multiplier=1.0, target=t, reduction=reduction
    )
    weights = tf.constant([[-w, w], [w, -w]], dtype=tf.float32)
    assert int(tf.size(weights)) == n_elements

    cost = float(reg(weights))
    assert abs(cost - expected) < 1e-6


def test_barrier_height_is_target_independent():
    """The barrier height is `multiplier` for ANY target, since t cancels.

    L(t/sqrt(3)) = m * (27/4) * f(1/3) = m, with no residual dependence on t
    (the t^6 in the denominator is exactly consumed by the three squared
    factors in the numerator). Checked at a non-default target so a stray
    t-dependence in the normalization could not hide behind t = 1.
    """
    t = 2.0
    w = t / math.sqrt(3.0)
    reg = TriStatePreferenceRegularizer(
        multiplier=1.0, target=t, reduction="mean"
    )
    weights = tf.constant([[-w, w]], dtype=tf.float32)
    assert abs(float(reg(weights)) - 1.0) < 1e-6


@pytest.mark.parametrize(
    "reduction, expected",
    [
        # Per element at w = 0.5, t = 1:
        #   w^2 = 0.25, (w^2 - t^2)^2 = (0.25 - 1)^2 = 0.5625,
        #   L = 6.75 * 0.25 * 0.5625 = 0.94921875.
        # Two elements under "sum": 2 * 0.94921875 = 1.8984375.
        ("sum", 2.0 * 0.94921875),
        # Under "mean" the average of two identical values is the value itself.
        ("mean", 0.94921875),
    ],
)
def test_value_at_half_target_is_below_the_barrier(reduction, expected):
    """w = +/- 0.5 t is NOT the maximum any more; pin its exact value.

    The superseded 32/4.5 constant placed the maxima at +/- 0.5 t and made the
    height there equal 1.0. Under the corrected 27/4 the maxima moved to
    +/- t/sqrt(3), so w = 0.5 t sits strictly inside the zero well:
    0.94921875 < 1.0 per element.
    """
    reg = TriStatePreferenceRegularizer(
        multiplier=1.0, target=1.0, reduction=reduction
    )
    max_weights = tf.constant([[-0.5, 0.5]], dtype=tf.float32)
    cost = float(reg(max_weights))

    assert abs(cost - expected) < 1e-6
    # And it is strictly under the barrier, unlike under the old constant.
    per_element = cost / (2.0 if reduction == "sum" else 1.0)
    assert per_element < 1.0


def test_watershed_discriminates_the_old_constant():
    """WATERSHED test: a weight at 0.55 t belongs to the ZERO well.

    This is the assertion that actually separates 27/4 from the old 32/4.5.
    The old constant put the watershed (barrier maximum) at 0.5 t, so anything
    at 0.55 t would have been PAST the crest and on its way to +t. The correct
    watershed is at 1/sqrt(3) = 0.57735 t, so 0.55 t is still below the crest.

    Arithmetic at w = 0.55, t = 1, m = 1:
        u = w^2                  = 0.3025
        (u - 1)^2                = (-0.6975)^2 = 0.48650625
        u * (u - 1)^2            = 0.3025 * 0.48650625 = 0.147168140625
        L = 6.75 * that          = 0.99338494921875
    which is strictly BELOW the barrier height of 1.0. The gap is only
    ~0.0066, so this is deliberately a strict inequality against the exact
    derived value, not an approximate comparison.
    """
    reg = TriStatePreferenceRegularizer(
        multiplier=1.0, target=1.0, reduction="mean"
    )
    cost = float(reg(tf.constant([[0.55]], dtype=tf.float32)))

    assert abs(cost - 0.99338494921875) < 1e-6
    # Strictly below the barrier -> 0.55 t is on the zero side of the crest.
    assert cost < 1.0


def test_three_wells_have_zero_cost():
    """L(0) = L(+t) = L(-t) = 0, exactly.

    The penalty is written in the factored form w^2 (w-t)^2 (w+t)^2, so each
    well is a literal root of a factor: at w = 0 the w^2 factor is 0, and at
    w = +/- t the (w -/+ t) factor is 0. With t = 1 every intermediate value is
    exactly representable in float32, so the expected value is exactly 0.0 --
    no tolerance needed.
    """
    reg = TriStatePreferenceRegularizer(
        multiplier=1.0, target=1.0, reduction="sum"
    )
    for w in (-1.0, 0.0, 1.0):
        assert float(reg(tf.constant([[w]], dtype=tf.float32))) == 0.0


def test_multiplier():
    """Test that multiplier factor properly affects the cost."""
    weights = tf.constant([[0.5]], dtype=tf.float32)

    # Test different scales
    reg1 = TriStatePreferenceRegularizer(multiplier=1.0)
    reg2 = TriStatePreferenceRegularizer(multiplier=2.0)

    cost1 = reg1(weights)
    cost2 = reg2(weights)

    # Cost should scale linearly
    assert abs(float(cost2) - 2.0 * float(cost1)) < 1e-6


def test_symmetry(regularizer):
    """Test that cost function is symmetric around x=0."""
    # Test pairs of points symmetric around 0
    points = [
        (-0.8, 0.8),
        (-0.5, 0.5),
        (-0.3, 0.3),
        (-0.1, 0.1)
    ]

    for p1, p2 in points:
        weights1 = tf.constant([[p1]], dtype=tf.float32)
        weights2 = tf.constant([[p2]], dtype=tf.float32)

        cost1 = regularizer(weights1)
        cost2 = regularizer(weights2)

        assert abs(float(cost1) - float(cost2)) < 1e-6


def test_monotonicity_regions():
    """Test monotonicity in regions between stable points and beyond."""
    regularizer = TriStatePreferenceRegularizer()

    # Test increasing cost from -1 to -0.5
    x1 = tf.constant([[-0.9]], dtype=tf.float32)
    x2 = tf.constant([[-0.7]], dtype=tf.float32)
    assert float(regularizer(x1)) < float(regularizer(x2))

    # Test decreasing cost from -0.5 to 0
    x3 = tf.constant([[-0.4]], dtype=tf.float32)
    x4 = tf.constant([[-0.2]], dtype=tf.float32)
    assert float(regularizer(x3)) > float(regularizer(x4))

    # Test increasing cost beyond |1|
    x5 = tf.constant([[1.0]], dtype=tf.float32)
    x6 = tf.constant([[1.5]], dtype=tf.float32)
    assert float(regularizer(x5)) < float(regularizer(x6))


def test_config_serialization_round_trips_every_key():
    """`get_config()` -> `from_config()` must preserve EVERY key it emits.

    The key set is fixed by the constructor signature of the current API:
    multiplier, target, reduction, quadratic_tails, annealable, name.
    `scale` and `base_coefficient` are gone; the old version of this test read
    them and died with AttributeError.
    """
    original = TriStatePreferenceRegularizer(
        multiplier=2.5,
        target=0.5,
        reduction="mean",
        quadratic_tails=True,
        annealable=True,
        name="tri_state_roundtrip",
    )
    config = original.get_config()

    assert set(config) == {
        "multiplier",
        "target",
        "reduction",
        "quadratic_tails",
        "annealable",
        "name",
    }

    rebuilt = TriStatePreferenceRegularizer.from_config(config)
    rebuilt_config = rebuilt.get_config()
    for key in config:
        assert rebuilt_config[key] == config[key], key

    # Config equality alone would not catch a key that is emitted but never
    # consumed by __init__, so also pin the BEHAVIOUR. The two instances must
    # agree bit-for-bit on a fixed input -- an exact 0.0 delta, not a tolerance:
    # both evaluate the identical float32 expression on identical inputs.
    weights = tf.constant([[-0.9, 0.2], [0.6, -0.35]], dtype=tf.float32)
    assert abs(float(original(weights)) - float(rebuilt(weights))) == 0.0


def test_config_reports_the_current_annealed_multiplier():
    """`get_config` emits `multiplier_value`, i.e. the CURRENT multiplier.

    This is a design choice being pinned, not a defect: with `annealable=True`
    the multiplier lives in a non-trainable `keras.Variable` that a
    `TriStatePressureScheduler` ramps during training, and `get_config`
    deliberately reads that variable rather than the constructor argument, so a
    model saved mid-anneal reloads at the pressure it was actually trained at.
    (The class docstring documents the annealing mechanism; it does not spell
    out this serialization consequence, so the pin lives here.)
    """
    reg = TriStatePreferenceRegularizer(multiplier=1.0, annealable=True)
    reg.set_multiplier(3.5)

    assert reg.get_config()["multiplier"] == 3.5
    assert TriStatePreferenceRegularizer.from_config(
        reg.get_config()
    ).multiplier_value == 3.5


def test_legacy_scale_kwarg_maps_to_target_and_warns(caplog):
    """The `scale=` deprecation shim survives, and says so.

    Old semantics pre-multiplied the weights by `scale`, so the wells sat at
    0 and +/- 1/scale. The shim therefore maps scale -> target = 1/scale:
    scale=2.0 -> target = 1/2.0 = 0.5.

    The module warns through the repo logger (`dl_techniques.utils.logger`),
    NOT through `warnings.warn`, so this uses `caplog` rather than
    `pytest.warns`. A shim with no test is a shim that gets deleted by accident.
    """
    with caplog.at_level(logging.WARNING, logger="dl"):
        reg = TriStatePreferenceRegularizer(scale=2.0)

    assert reg.target == 0.5

    warnings_logged = [
        rec.getMessage() for rec in caplog.records if rec.levelno >= logging.WARNING
    ]
    assert any(
        "`scale` is deprecated" in msg and "target=0.5" in msg
        for msg in warnings_logged
    ), warnings_logged


def test_legacy_scale_rejects_non_positive():
    """The shim validates before translating; 1/scale would otherwise blow up."""
    with pytest.raises(ValueError, match="scale must be positive"):
        TriStatePreferenceRegularizer(scale=0.0)


def test_keras_integration():
    """Test integration with Keras model."""
    regularizer = TriStatePreferenceRegularizer(scale=1.0)

    # Create simple model with regularizer
    model = Sequential([
        Input(shape=(2,)),
        Dense(4, kernel_regularizer=regularizer)
    ])

    # Compile model
    model.compile(optimizer='adam', loss='mse')

    # Should compile without errors
    assert model.layers[0].kernel_regularizer is not None


def test_numerical_stability():
    """Test regularizer behavior with extreme values."""
    regularizer = TriStatePreferenceRegularizer(scale=1.0)

    # Test very large and small values
    extreme_weights = tf.constant([
        [-1e5, 1e5],  # Very large values
        [-1 + 1e-10, 1 - 1e-10],  # Very close to -1 and 1
        [-1e-10, 1e-10],  # Very close to 0
        [-0.5 - 1e-10, 0.5 + 1e-10]  # Very close to local maxima
    ], dtype=tf.float32)

    # Should not produce NaN or infinity
    cost = regularizer(extreme_weights)
    assert not tf.math.is_nan(cost)
    assert not tf.math.is_inf(cost)


def test_gradient_computation():
    """Test that gradients can be computed through the regularizer."""
    regularizer = TriStatePreferenceRegularizer(scale=1.0)
    weights = tf.Variable([[-0.3, 0.0, 0.7]], dtype=tf.float32)

    with tf.GradientTape() as tape:
        cost = regularizer(weights)

    # Should be able to compute gradients
    gradients = tape.gradient(cost, weights)
    assert gradients is not None
    assert not tf.reduce_any(tf.math.is_nan(gradients))


if __name__ == '__main__':
    pytest.main([__file__])