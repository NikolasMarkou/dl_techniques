"""
Tests for the Binary Preference Regularizer implementation.

This test suite verifies the functionality of the BinaryPreferenceRegularizer
including its mathematical properties, integration with Keras, and edge cases.
"""

import logging

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras.api.layers import Dense, Input
from keras.api.models import Sequential

from dl_techniques.regularizers.binary_preference import (
    BinaryPreferenceRegularizer,
    BinaryPressureScheduler,
    create_binary_preference_regularizer,
)


@pytest.fixture
def regularizer():
    """Fixture providing a default regularizer instance."""
    return BinaryPreferenceRegularizer(multiplier=1.0)


def test_regularizer_initialization():
    """Pin the constructed multiplier as a FLOAT, not through a tensor compare.

    With the new `annealable=True` default `reg.multiplier` is a
    `keras.Variable`, so the previous `assert reg.multiplier == 1.0` compared a
    Variable against a float. MEASURED on the TF backend at HEAD: that returns
    a SCALAR bool EagerTensor whose truth value is correct, so the old form was
    not in fact vacuous -- it still failed on a wrong value. It was, however,
    fragile: the assertion depends on eager execution, on the variable being
    rank-0 (a non-scalar comparison has an ambiguous truth value and raises),
    and on backend truthiness semantics.

    `multiplier_value` is the documented float accessor (it goes through
    `ops.convert_to_numpy`), so `float(...) == expected` is a plain Python
    comparison that can fail for exactly one reason: a wrong value.
    """
    reg = BinaryPreferenceRegularizer()
    assert float(reg.multiplier_value) == 1.0  # DEFAULT_MULTIPLIER

    reg = BinaryPreferenceRegularizer(multiplier=2.0)
    assert float(reg.multiplier_value) == 2.0


@pytest.mark.parametrize(
    "annealable, expected_type",
    [
        # annealable=True stores the multiplier in a non-trainable
        # `keras.Variable` so a callback can assign to it mid-training.
        (True, keras.Variable),
        # annealable=False folds it in as a Python float constant
        # (`self.multiplier = float(multiplier)`), so `== 1.0` is a real
        # comparison there and a tensor comparison in the other branch.
        (False, float),
    ],
)
def test_multiplier_type_contract(annealable, expected_type):
    """The `annealable` flag decides the TYPE of `.multiplier`.

    This is the type change that turned every `assert reg.multiplier == <x>`
    in this file into a tensor comparison. Pinning the type means a future flip
    of the default surfaces here, at the contract, rather than as a diffuse
    change in how a dozen unrelated assertions evaluate.
    """
    reg = BinaryPreferenceRegularizer(multiplier=1.0, annealable=annealable)
    assert isinstance(reg.multiplier, expected_type)
    # Either way the float accessor reads the same number.
    assert float(reg.multiplier_value) == 1.0


def test_binary_points_zero_cost(regularizer):
    """Test that binary values (0 and 1) produce zero cost."""
    # Create tensor with binary values
    binary_weights = tf.constant([[0.0, 1.0], [1.0, 0.0]], dtype=tf.float32)
    cost = regularizer(binary_weights)

    # Cost should be very close to zero
    assert tf.abs(cost) < 1e-6


@pytest.mark.parametrize(
    "reduction, expected",
    [
        # Derivation (from the module docstring's published formula):
        #   L(w) = m * (w - low)^2 * (w - high)^2 / h^4,  h = (high - low) / 2
        # At the midpoint w = (low + high) / 2 each of the two differences is
        # +/- h, so the numerator is h^2 * h^2 = h^4 and the per-element penalty
        # is EXACTLY m. The barrier height is the multiplier itself,
        # independent of low/high.
        # For the default low=0, high=1: h = 0.5, h^4 = 0.0625; at w = 0.5
        #   (0.5)^2 * (-0.5)^2 / 0.0625 = 0.25 * 0.25 / 0.0625 = 1.0 per element.
        # The tensor is 2x2, so N = 4 elements.
        #   reduction="sum"  -> m * N = 1.0 * 4 = 4.0
        ("sum", 4.0),
        #   reduction="mean" -> m       = 1.0   (what the old test really meant;
        #                                        `mean` was the implicit default)
        ("mean", 1.0),
    ],
)
def test_midpoint_maximum_cost(reduction, expected):
    """Pin the midpoint barrier under BOTH reductions.

    The default reduction changed from an implicit `mean` to `sum`, so a
    "cost at the maximum == 1.0" assertion now scales with the element count.
    """
    reg = BinaryPreferenceRegularizer(multiplier=1.0, reduction=reduction)
    mid_weights = tf.constant([[0.5, 0.5], [0.5, 0.5]], dtype=tf.float32)
    assert int(tf.size(mid_weights)) == 4

    cost = float(reg(mid_weights))
    assert abs(cost - expected) < 1e-6


@pytest.mark.parametrize(
    "low, high, multiplier",
    [
        # Symmetric kernel targets: h = (1 - (-1)) / 2 = 1, h^4 = 1.
        # Midpoint w = 0: (0 - (-1))^2 * (0 - 1)^2 / 1 = 1 * 1 / 1 = 1
        # per element, times m = 2.0 -> 2.0.
        (-1.0, 1.0, 2.0),
        # Narrow gate targets: h = (0.75 - 0.25) / 2 = 0.25, h^4 = 0.00390625.
        # Midpoint w = 0.5: (0.25)^2 * (-0.25)^2 = 0.00390625, and dividing by
        # h^4 gives exactly 1 per element, times m = 0.5 -> 0.5.
        (0.25, 0.75, 0.5),
    ],
)
def test_barrier_height_equals_multiplier_for_any_targets(low, high, multiplier):
    """The barrier height is `multiplier`, independent of `low` and `high`.

    The h^4 divisor exists precisely to cancel the gap width: at the midpoint
    the numerator is always h^4, so L(midpoint) = m for every (low, high).
    Checked at two non-default triples so a residual gap-width dependence
    could not hide behind the canonical low=0, high=1 case. `mean` is used so
    the value read is the per-element barrier rather than N times it.
    """
    reg = BinaryPreferenceRegularizer(
        multiplier=multiplier, low=low, high=high, reduction="mean"
    )
    midpoint = (low + high) / 2.0
    weights = tf.constant([[midpoint, midpoint]], dtype=tf.float32)

    assert abs(float(reg(weights)) - multiplier) < 1e-6


def test_scaling():
    """Test that scaling factor properly affects the cost."""
    weights = tf.constant([[0.5]], dtype=tf.float32)

    # Test different scales
    reg1 = BinaryPreferenceRegularizer(multiplier=1.0)
    reg2 = BinaryPreferenceRegularizer(multiplier=2.0)

    cost1 = reg1(weights)
    cost2 = reg2(weights)

    # Cost should scale linearly
    assert abs(float(cost2) - 2.0 * float(cost1)) < 1e-6


def test_symmetry(regularizer):
    """Test that cost is symmetric around 0.5."""
    # Test pairs of points equidistant from 0.5
    points = [(0.1, 0.9), (0.2, 0.8), (0.3, 0.7), (0.4, 0.6)]

    for p1, p2 in points:
        weights1 = tf.constant([[p1]], dtype=tf.float32)
        weights2 = tf.constant([[p2]], dtype=tf.float32)

        cost1 = regularizer(weights1)
        cost2 = regularizer(weights2)

        assert abs(float(cost1) - float(cost2)) < 1e-6


def test_config_serialization():
    """Test configuration serialization and deserialization."""
    original_reg = BinaryPreferenceRegularizer(multiplier=2.0)
    config = original_reg.get_config()

    # Recreate from config
    new_reg = BinaryPreferenceRegularizer.from_config(config)

    # Read through `multiplier_value` (a float): `.multiplier` is a
    # `keras.Variable` under the default `annealable=True`, so the previous
    # `new_reg.multiplier == original_reg.multiplier` was a Variable-vs-Variable
    # tensor compare. It does discriminate on this backend (measured), but it
    # is eager- and rank-dependent, and it only checked that the two agreed --
    # never that they equal the value that was actually passed in.
    assert float(new_reg.multiplier_value) == float(original_reg.multiplier_value)
    assert float(new_reg.multiplier_value) == 2.0


def test_keras_integration():
    """Test integration with Keras model."""
    # Modern API. The legacy `scale=` kwarg is pinned separately by
    # `test_legacy_scale_kwarg_maps_to_low_high_and_warns`; the mainline of
    # this suite must not depend on a deprecated path.
    regularizer = BinaryPreferenceRegularizer(multiplier=1.0, low=0.0, high=1.0)

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
    regularizer = BinaryPreferenceRegularizer(multiplier=1.0)

    # Test very large and small values
    extreme_weights = tf.constant([
        [-1e5, 1e5],  # Very large values
        [1e-10, 1 - 1e-10],  # Very close to 0 and 1
        [0.5 - 1e-10, 0.5 + 1e-10]  # Very close to 0.5
    ], dtype=tf.float32)

    # Should not produce NaN or infinity
    cost = regularizer(extreme_weights)
    assert not tf.math.is_nan(cost)
    assert not tf.math.is_inf(cost)


def test_gradient_computation():
    """Test that gradients can be computed through the regularizer."""
    regularizer = BinaryPreferenceRegularizer(multiplier=1.0)
    weights = tf.Variable([[0.3, 0.7]], dtype=tf.float32)

    with tf.GradientTape() as tape:
        cost = regularizer(weights)

    # Should be able to compute gradients
    gradients = tape.gradient(cost, weights)
    assert gradients is not None
    assert not tf.reduce_any(tf.math.is_nan(gradients))


# ---------------------------------------------------------------------
# New public surface added by the rewrite: the `scale=` shim, `quadratic_tails`,
# `reduction`, `set_multiplier`, `BinaryPressureScheduler`, the factory, the
# classmethod presets, and the validation paths. None of it had any coverage.
# ---------------------------------------------------------------------


def test_legacy_scale_kwarg_maps_to_low_high_and_warns(caplog):
    """The `scale=` deprecation shim survives, and says so.

    Old semantics placed the wells at 0 and 1/scale, so the shim maps
    scale -> (low=0.0, high=1/scale): scale=2.0 -> low=0.0, high=0.5.

    The module warns through the repo logger (`dl_techniques.utils.logger`,
    the "dl" logger), NOT through `warnings.warn`, so this uses `caplog`
    rather than `pytest.warns`. A shim with no test is a shim that gets
    deleted by accident.
    """
    with caplog.at_level(logging.WARNING, logger="dl"):
        reg = BinaryPreferenceRegularizer(scale=2.0)

    assert reg.low == 0.0
    assert reg.high == 0.5

    warnings_logged = [
        rec.getMessage() for rec in caplog.records if rec.levelno >= logging.WARNING
    ]
    assert any(
        "`scale` is deprecated" in msg and "high=0.5" in msg
        for msg in warnings_logged
    ), warnings_logged


def test_legacy_scale_rejects_non_positive():
    """The shim validates before translating; 1/scale would otherwise blow up."""
    with pytest.raises(ValueError, match="scale must be positive"):
        BinaryPreferenceRegularizer(scale=0.0)


@pytest.mark.parametrize("d", [0.25, 0.5, 1.0, 2.0])
def test_quadratic_tails_growth_is_exactly_quadratic(d):
    """Outside [low, high] the `quadratic_tails=True` branch grows as d^2.

    Derivation (from the docstring): the C2-continuous extension is
        L(high + d) = 4 * m * d^2 / h^2
    which is exactly quadratic in d, so the ratio between two distances that
    differ by a factor of 2 is a CONSTANT 4:
        L(high + 2d) / L(high + d) = (2d)^2 / d^2 = 4     for every d.

    The growth ORDER is what the knob exists to change, so the ratio -- not a
    single value -- is the discriminating oracle. A branch that merely rescaled
    the quartic would reproduce any one value while failing this ratio.
    """
    reg = BinaryPreferenceRegularizer(
        multiplier=1.0, quadratic_tails=True, reduction="mean"
    )
    near = float(reg(tf.constant([[1.0 + d]], dtype=tf.float32)))
    far = float(reg(tf.constant([[1.0 + 2.0 * d]], dtype=tf.float32)))

    assert abs(far / near - 4.0) < 1e-5

    # Cross-check the closed form itself: h = 0.5, so 4/h^2 = 16 and
    # L(high + d) = 16 * d^2. (Measured reference: d=0.5 -> 4.0,
    # d=1.0 -> 16.0, d=2.0 -> 64.0.)
    assert abs(near - 16.0 * d * d) < 1e-4 * max(1.0, 16.0 * d * d)


@pytest.mark.parametrize("d", [0.25, 0.5, 1.0, 2.0])
def test_quartic_tails_grow_strictly_faster_than_quadratic(d):
    """The default `quadratic_tails=False` tail is quartic, not quadratic.

    With low=0, high=1 the unclipped penalty is
        L(w) = m * w^2 * (w - 1)^2 / h^4 = 16 * w^2 * (w - 1)^2,
    so at w = 1 + d it is 16 * (1 + d)^2 * d^2 and the doubling ratio is
        L(1 + 2d) / L(1 + d) = 4 * (1 + 2d)^2 / (1 + d)^2   >  4  for d > 0.
    At d = 1 that is 4 * 9 / 4 = 9 exactly (measured: 64.0 -> 576.0).

    This is the arm that proves the ratio test above DISCRIMINATES: the same
    measurement on the other branch gives a different, d-dependent number.
    """
    reg = BinaryPreferenceRegularizer(
        multiplier=1.0, quadratic_tails=False, reduction="mean"
    )
    near = float(reg(tf.constant([[1.0 + d]], dtype=tf.float32)))
    far = float(reg(tf.constant([[1.0 + 2.0 * d]], dtype=tf.float32)))

    expected_ratio = 4.0 * (1.0 + 2.0 * d) ** 2 / (1.0 + d) ** 2
    assert expected_ratio > 4.0
    assert abs(far / near - expected_ratio) < 1e-4 * expected_ratio


@pytest.mark.parametrize("target_side", ["low", "high"])
def test_quadratic_tails_join_is_continuous(target_side):
    """The tail branch meets the well at value 0 on both targets.

    C2 continuity requires L(target) = 0 exactly (the target is a zero of the
    quartic AND of the quadratic extension, since d = 0 there), and the outside
    branch must approach 0 as d -> 0. At d = 1e-3 the closed form gives
    16 * (1e-3)^2 = 1.6e-05, so the value approaches 0 from outside rather than
    stepping discontinuously.
    """
    reg = BinaryPreferenceRegularizer(
        multiplier=1.0, quadratic_tails=True, reduction="mean"
    )
    target = 0.0 if target_side == "low" else 1.0
    sign = -1.0 if target_side == "low" else 1.0

    at_join = float(reg(tf.constant([[target]], dtype=tf.float32)))
    assert at_join == 0.0

    # Compared RELATIVELY: `target + sign * d` is not representable exactly in
    # float32 (1.0 + 1e-3 rounds), so the input itself carries ~1e-4 relative
    # error before the penalty squares it. 1e-4 relative is the float32 floor
    # here, not a loosened bound -- an absolute 1e-9 is unattainable.
    d = 1e-3
    outside = float(reg(tf.constant([[target + sign * d]], dtype=tf.float32)))
    expected = 16.0 * d * d
    assert abs(outside - expected) < 1e-4 * expected
    assert outside > at_join


@pytest.mark.parametrize("n", [1, 4, 12])
def test_sum_reduction_is_n_times_mean_reduction(n):
    """`sum` and `mean` differ by exactly the element count.

    `mean` divides the summed per-weight cost by N, so for identical weights
    and identical multiplier, sum == N * mean EXACTLY (same summation order,
    one extra division). This is the relationship the docstring's "mean
    divides the per-weight gradient by the parameter count" warning rests on.
    """
    weights = tf.constant([[0.3] * n], dtype=tf.float32)
    assert int(tf.size(weights)) == n

    cost_sum = float(BinaryPreferenceRegularizer(reduction="sum")(weights))
    cost_mean = float(BinaryPreferenceRegularizer(reduction="mean")(weights))

    assert abs(cost_sum - n * cost_mean) < 1e-6 * max(1.0, cost_sum)


def test_invalid_reduction_raises():
    """`reduction` is validated in the constructor, not silently ignored."""
    with pytest.raises(ValueError, match="reduction must be one of"):
        BinaryPreferenceRegularizer(reduction="avg")


def test_set_multiplier_changes_the_computed_cost():
    """`set_multiplier` must move the PENALTY, not merely an attribute.

    The penalty is linear in the multiplier, so doubling it doubles the cost:
        L_m(w) = m * (per-element sum),  hence L_2m / L_m = 2 exactly.
    Asserting on the attribute alone would pass against a multiplier that the
    `__call__` path never reads (a plausible wiring defect once the value moved
    into a `keras.Variable`).
    """
    reg = BinaryPreferenceRegularizer(multiplier=1.0, reduction="mean")
    weights = tf.constant([[0.5, 0.5]], dtype=tf.float32)

    before = float(reg(weights))
    assert abs(before - 1.0) < 1e-6  # barrier height == multiplier == 1.0

    reg.set_multiplier(2.0)
    after = float(reg(weights))

    assert float(reg.multiplier_value) == 2.0
    assert abs(after - 2.0 * before) < 1e-6


def test_set_multiplier_requires_annealable():
    """A constant-folded multiplier cannot be annealed, and says so loudly."""
    reg = BinaryPreferenceRegularizer(multiplier=1.0, annealable=False)
    with pytest.raises(RuntimeError, match="annealable=True"):
        reg.set_multiplier(2.0)


def test_pressure_scheduler_full_sequence_under_real_fit():
    """Pin the WHOLE annealing schedule under a real `model.fit()`.

    Closed form (`on_epoch_begin`):
        value(e) = target * clip((e - warmup_epochs) / ramp_epochs, 0, 1)
    With target=1.0, warmup_epochs=1, ramp_epochs=2 over 5 epochs:
        e=0 -> (0-1)/2 = -0.5 -> clipped to 0 -> 0.0
        e=1 -> (1-1)/2 =  0.0 ->               0.0
        e=2 -> (2-1)/2 =  0.5 ->               0.5
        e=3 -> (3-1)/2 =  1.0 ->               1.0
        e=4 -> (4-1)/2 =  1.5 -> clipped to 1 -> 1.0
    i.e. [0.0, 0.0, 0.5, 1.0, 1.0].

    The FULL sequence is asserted rather than "the value changed": an
    off-by-one at the warmup boundary and a ramp that completes an epoch early
    both leave the endpoint at 1.0 and would pass a change-only check.
    """
    reg = BinaryPreferenceRegularizer(multiplier=0.0, reduction="mean")
    observed = []

    class _Recorder(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            observed.append(reg.multiplier_value)

    model = Sequential([Input(shape=(2,)), Dense(3, kernel_regularizer=reg)])
    model.compile(optimizer="sgd", loss="mse")
    model.fit(
        np.zeros((4, 2), dtype="float32"),
        np.zeros((4, 3), dtype="float32"),
        epochs=5,
        verbose=0,
        callbacks=[
            BinaryPressureScheduler(
                reg, target=1.0, warmup_epochs=1, ramp_epochs=2
            ),
            _Recorder(),
        ],
    )

    assert observed == [0.0, 0.0, 0.5, 1.0, 1.0]


def test_pressure_scheduler_rejects_non_positive_ramp():
    """`ramp_epochs <= 0` would divide by zero in the progress term."""
    reg = BinaryPreferenceRegularizer()
    with pytest.raises(ValueError, match="ramp_epochs must be positive"):
        BinaryPressureScheduler(reg, target=1.0, ramp_epochs=0)


def test_factory_returns_configured_instance():
    """`create_binary_preference_regularizer` forwards every argument.

    Read back through `get_config()` so the check covers what actually got
    stored, not what was passed. The factory is a thin forwarder, so the
    expected config is exactly the arguments plus the constructor defaults
    (`quadratic_tails=False`, `annealable=True`, `name=None`).
    """
    reg = create_binary_preference_regularizer(
        multiplier=0.5, low=-2.0, high=3.0, reduction="mean"
    )

    assert isinstance(reg, BinaryPreferenceRegularizer)
    assert reg.get_config() == {
        "multiplier": 0.5,
        "low": -2.0,
        "high": 3.0,
        "reduction": "mean",
        "quadratic_tails": False,
        "annealable": True,
        "name": None,
    }


@pytest.mark.parametrize(
    "factory, low, high, reduction",
    [
        # `for_gates`: {0, 1} targets for gates/masks initialized inside [0, 1],
        # with reduction="mean" so the loss does not scale with the gate count.
        (BinaryPreferenceRegularizer.for_gates, 0.0, 1.0, "mean"),
        # `for_bipolar_weights`: {-1, +1} for zero-centered kernels; it does NOT
        # override the reduction, so it keeps DEFAULT_REDUCTION == "sum".
        (BinaryPreferenceRegularizer.for_bipolar_weights, -1.0, 1.0, "sum"),
    ],
)
def test_classmethod_presets_targets(factory, low, high, reduction):
    """The two presets differ only in their targets (and gates' reduction).

    Both default `multiplier=0.0` on the documented assumption that the
    pressure is annealed up from zero, and both set `quadratic_tails=True`.
    """
    reg = factory()

    assert reg.low == low
    assert reg.high == high
    assert reg.reduction == reduction
    assert reg.quadratic_tails is True
    assert float(reg.multiplier_value) == 0.0


@pytest.mark.parametrize(
    "kwargs, exc, match",
    [
        # Negative strength is meaningless: the penalty would REWARD leaving
        # the targets.
        ({"multiplier": -1.0}, ValueError, "multiplier must be non-negative"),
        # high == low gives half-gap h = 0 and a division by h^4 = 0.
        ({"low": 1.0, "high": 1.0}, ValueError, "high must be strictly greater"),
        # high < low would invert the well entirely.
        ({"low": 1.0, "high": 0.0}, ValueError, "high must be strictly greater"),
        ({"reduction": "median"}, ValueError, "reduction must be one of"),
        # `Regularizer` defines no __init__, so **kwargs cannot be forwarded;
        # anything but the legacy `scale` is rejected as a TypeError.
        ({"bogus": 1}, TypeError, "unexpected keyword arguments"),
    ],
)
def test_constructor_validation_paths(kwargs, exc, match):
    """Every validation branch raises, with the documented exception type."""
    with pytest.raises(exc, match=match):
        BinaryPreferenceRegularizer(**kwargs)


def test_package_reexports_the_orphaned_binary_scheduler():
    """`BinaryPressureScheduler` is reachable from the package and is the
    same object as the submodule's, not a shadowing re-definition."""
    import dl_techniques.regularizers as R
    import dl_techniques.regularizers.binary_preference as bp

    assert R.BinaryPressureScheduler is bp.BinaryPressureScheduler


if __name__ == '__main__':
    pytest.main([__file__])