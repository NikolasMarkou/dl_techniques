"""Tests for :mod:`dl_techniques.losses.mean_shift_instance_loss`.

The oracle is :func:`_oracle`, a plain-NumPy float64 double loop over ordered point
pairs. It is written from the definition (pull = mean squared distance over
same-instance pairs, push = mean squared hinge over different-instance pairs) and not
from the implementation, which is vectorised. Each load-bearing check is a function of a
callable ``fn(ids, z) -> (B,)`` so the RED proofs can run the same body against a mutated
loss and require it to fail.
"""

import numpy as np
import pytest
import keras
import tensorflow as tf
from sklearn.metrics import adjusted_rand_score

from dl_techniques.layers.mixtures.mean_shift import MeanShiftLayer, assign_clusters
from dl_techniques.losses.mean_shift_instance_loss import MeanShiftInstanceLoss
from tests.numerics import true_float32_matmul

# Measured 2026-10-29 on CPU, float32 inputs, matmul-form squared distance, tiny case:
# the largest deviation from the float64 value is below 1e-6.
FLOAT32_ATOL = 1e-6
FLOAT64_ATOL = 1e-9


# ---------------------------------------------------------------------
# Oracle and helpers
# ---------------------------------------------------------------------


def _oracle(ids, z, margin, pull_w=1.0, push_w=1.0, ignore=-1, swap=False):
    """Per-row loss by an explicit float64 double loop; ``swap`` exchanges the roles."""
    ids = np.asarray(ids).astype(np.int64)
    z = np.asarray(z, dtype=np.float64)
    out = np.zeros(ids.shape[0])
    for b in range(ids.shape[0]):
        pull = push = 0.0
        n_same = n_diff = 0
        for i in range(ids.shape[1]):
            for j in range(ids.shape[1]):
                if i == j or ids[b, i] == ignore or ids[b, j] == ignore:
                    continue
                dist = float(np.linalg.norm(z[b, i] - z[b, j]))
                pulled = (ids[b, i] == ids[b, j]) != swap
                if pulled:
                    pull += dist * dist
                    n_same += 1
                else:
                    push += max(0.0, margin - dist) ** 2
                    n_diff += 1
        out[b] = pull_w * pull / max(n_same, 1) + push_w * push / max(n_diff, 1)
    return out


def _tensor(values, dtype="float32"):
    return keras.ops.convert_to_tensor(np.asarray(values), dtype=dtype)


def _real(loss):
    """Adapter: the real loss as a ``fn(ids, z) -> numpy (B,)``."""

    def fn(ids, z):
        with true_float32_matmul():
            return keras.ops.convert_to_numpy(loss.call(_tensor(ids, "int32"), _tensor(z)))

    return fn


def _random_batch(batch=3, n=9, dim=3, instances=3, ignore_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, instances, size=(batch, n))
    ids = np.where(rng.random((batch, n)) < ignore_frac, -1, ids)
    z = rng.standard_normal((batch, n, dim)) * 0.8
    return ids, z


def _spaced_clusters(spacing=5.0, batch=2, per_instance=3):
    """Instances 0..2 sitting exactly on points ``spacing`` apart, no spread."""
    centres = spacing * np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    ids = np.tile(np.repeat(np.arange(3), per_instance), (batch, 1))
    z = centres[ids]
    return ids, z


# ---------------------------------------------------------------------
# Check bodies (shared with the RED proofs)
# ---------------------------------------------------------------------


def _check_zero_on_separated_clusters(fn):
    ids, z = _spaced_clusters(spacing=5.0)
    value = fn(ids, z)
    np.testing.assert_allclose(
        value, 0.0, atol=1e-6, rtol=0,
        err_msg="perfectly separated, zero-spread clusters must cost nothing",
    )


def _check_ignore_is_inert(fn):
    ids, z = _random_batch(batch=3, n=10, ignore_frac=0.3, seed=11)
    assert (ids == -1).any()
    base = fn(ids, z)
    moved = np.where((ids == -1)[..., None], 40.0 + 7.0 * z, z)
    np.testing.assert_allclose(
        fn(ids, moved), base, atol=1e-5, rtol=0,
        err_msg="moving an ignored point changed the loss",
    )


# ---------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------


def test_zero_loss_on_perfectly_separated_clusters():
    _check_zero_on_separated_clusters(_real(MeanShiftInstanceLoss(bandwidth=1.0)))


def test_zero_loss_survives_ignored_points_between_the_clusters():
    ids, z = _spaced_clusters(spacing=5.0)
    ids = np.concatenate([ids, -np.ones((2, 2), dtype=ids.dtype)], axis=1)
    z = np.concatenate([z, np.full((2, 2, 2), 2.5)], axis=1)
    np.testing.assert_allclose(
        _real(MeanShiftInstanceLoss())(ids, z), 0.0, atol=1e-6, rtol=0
    )


# Hand computation. 1-D points, margin 2. Instance A at 0 and 0.5, instance B at 1.5 and 2.5.
# Same-instance ordered pairs: 2 pairs of squared distance 0.25 and 2 of 1.0, so
# pull = (2 * 0.25 + 2 * 1.0) / 4 = 0.625.
# Different-instance ordered pairs (8) have distances 1.5, 2.5, 1.0, 2.0, hinge squares
# 0.25, 0, 1.0, 0, each twice, so push = 2 * 1.25 / 8 = 0.3125.
HAND_IDS = np.array([[0, 0, 1, 1]])
HAND_Z = np.array([[[0.0], [0.5], [1.5], [2.5]]])


@pytest.mark.parametrize(
    "pull_w,push_w,expected",
    [
        (1.0, 1.0, 0.625 + 0.3125),
        (2.0, 0.5, 2.0 * 0.625 + 0.5 * 0.3125),
        (1.0, 0.0, 0.625),
        (0.0, 1.0, 0.3125),
    ],
)
def test_the_tiny_case_matches_the_hand_computed_value(pull_w, push_w, expected):
    loss = MeanShiftInstanceLoss(bandwidth=1.0, pull_weight=pull_w, push_weight=push_w)
    with true_float32_matmul():
        value32 = keras.ops.convert_to_numpy(
            loss.call(_tensor(HAND_IDS, "int32"), _tensor(HAND_Z))
        )
    value64 = keras.ops.convert_to_numpy(
        loss.call(_tensor(HAND_IDS, "int32"), _tensor(HAND_Z, "float64"))
    )
    assert value64.dtype == np.float64
    np.testing.assert_allclose(value64, [expected], atol=FLOAT64_ATOL, rtol=0)
    np.testing.assert_allclose(value32, [expected], atol=FLOAT32_ATOL, rtol=0)
    np.testing.assert_allclose(
        _oracle(HAND_IDS, HAND_Z, margin=2.0, pull_w=pull_w, push_w=push_w),
        [expected], atol=1e-12, rtol=0,
        err_msg="the NumPy oracle disagrees with the hand value",
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_the_numpy_oracle_with_ignored_points_and_weights(seed):
    ids, z = _random_batch(batch=3, n=11, dim=4, instances=3, ignore_frac=0.25, seed=seed)
    loss = MeanShiftInstanceLoss(
        bandwidth=0.7, margin=1.9, pull_weight=1.5, push_weight=0.6, ignore_label=-1
    )
    expected = _oracle(ids, z, margin=1.9, pull_w=1.5, push_w=0.6)
    got = keras.ops.convert_to_numpy(
        loss.call(_tensor(ids, "int32"), _tensor(z, "float64"))
    )
    np.testing.assert_allclose(got, expected, atol=1e-9, rtol=0)


def test_a_custom_ignore_label_is_honoured():
    ids, z = _random_batch(batch=2, n=8, ignore_frac=0.3, seed=5)
    shifted = np.where(ids == -1, 99, ids)
    a = _real(MeanShiftInstanceLoss(ignore_label=99))(shifted, z)
    b = _real(MeanShiftInstanceLoss(ignore_label=-1))(ids, z)
    np.testing.assert_allclose(a, b, atol=1e-6, rtol=0)


def test_call_returns_one_value_per_row_and_rows_are_independent():
    ids, z = _random_batch(batch=4, n=7, ignore_frac=0.15, seed=3)
    loss = MeanShiftInstanceLoss()
    with true_float32_matmul():
        whole = loss.call(_tensor(ids, "int32"), _tensor(z))
        assert tuple(whole.shape) == (4,)
        for row in range(4):
            single = loss.call(_tensor(ids[row:row + 1], "int32"), _tensor(z[row:row + 1]))
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(whole)[row],
                keras.ops.convert_to_numpy(single)[0],
                atol=1e-6, rtol=0,
            )
    assert len({round(float(v), 4) for v in keras.ops.convert_to_numpy(whole)}) > 1, (
        "rows with different instances should not all cost the same"
    )


def test_float_ids_as_passed_by_fit_are_cast():
    ids, z = _random_batch(batch=2, n=8, seed=9)
    loss = MeanShiftInstanceLoss()
    with true_float32_matmul():
        as_int = loss.call(_tensor(ids, "int32"), _tensor(z))
        as_float = loss.call(_tensor(ids, "float32"), _tensor(z))
        through_call = loss(_tensor(ids, "float32"), _tensor(z))
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(as_float), keras.ops.convert_to_numpy(as_int), atol=0, rtol=0
    )
    np.testing.assert_allclose(
        float(through_call), float(np.mean(keras.ops.convert_to_numpy(as_int))), atol=1e-6, rtol=0
    )


# ---------------------------------------------------------------------
# Ignore label
# ---------------------------------------------------------------------


def test_ignored_points_change_nothing():
    _check_ignore_is_inert(_real(MeanShiftInstanceLoss()))


def test_ignoring_a_point_equals_removing_it():
    rng = np.random.default_rng(21)
    ids = rng.integers(0, 3, size=(2, 10))
    z = rng.standard_normal((2, 10, 3))
    kept = np.array([[0, 1, 2, 4, 5, 7, 8, 9], [0, 2, 3, 4, 5, 6, 8, 9]])
    flagged = ids.copy()
    for row in range(2):
        drop = np.setdiff1d(np.arange(10), kept[row])
        flagged[row, drop] = -1
    removed_ids = np.stack([ids[r, kept[r]] for r in range(2)])
    removed_z = np.stack([z[r, kept[r]] for r in range(2)])
    fn = _real(MeanShiftInstanceLoss())
    np.testing.assert_allclose(fn(flagged, z), fn(removed_ids, removed_z), atol=1e-6, rtol=0)


# ---------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------


def _degenerate_cases():
    z = np.random.default_rng(1).standard_normal((2, 5, 3))
    return {
        "single_instance": (np.zeros((2, 5), dtype=int), z),
        "all_ignored": (-np.ones((2, 5), dtype=int), z),
        "single_valid_point": (np.array([[0, -1, -1, -1, -1], [-1, 3, -1, -1, -1]]), z),
        "n_equals_one": (np.zeros((2, 1), dtype=int), z[:, :1]),
        "one_point_per_instance": (np.arange(5)[None].repeat(2, 0), z),
    }


@pytest.mark.parametrize("case", list(_degenerate_cases()))
def test_degenerate_rows_cost_a_finite_amount_with_a_finite_gradient(case):
    ids, z = _degenerate_cases()[case]
    loss = MeanShiftInstanceLoss()
    variable = tf.Variable(z.astype("float32"))
    with true_float32_matmul(), tf.GradientTape() as tape:
        value = loss.call(_tensor(ids, "int32"), tf.identity(variable))
        total = tf.reduce_sum(value)
    grad = tape.gradient(total, variable)
    value = keras.ops.convert_to_numpy(value)
    assert np.isfinite(value).all(), f"{case}: loss {value}"
    assert np.isfinite(keras.ops.convert_to_numpy(grad)).all(), f"{case}: gradient not finite"
    np.testing.assert_allclose(
        value, _oracle(ids, z, margin=2.0), atol=1e-5, rtol=0, err_msg=case
    )
    if case in ("all_ignored", "single_valid_point", "n_equals_one"):
        np.testing.assert_allclose(value, 0.0, atol=1e-7, rtol=0, err_msg=case)


def test_coincident_points_of_different_instances_have_a_finite_gradient():
    ids = np.array([[0, 1, 0, 1]])
    z = np.zeros((1, 4, 2), dtype="float32")
    loss = MeanShiftInstanceLoss(bandwidth=1.0)
    variable = tf.Variable(z)
    with true_float32_matmul(), tf.GradientTape() as tape:
        value = loss.call(_tensor(ids, "int32"), tf.identity(variable))
    grad = keras.ops.convert_to_numpy(tape.gradient(value, variable))
    assert np.isfinite(grad).all()
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(value), [4.0], atol=1e-4, rtol=0,
        err_msg="coincident different-instance pairs sit at the full margin^2 penalty",
    )


def test_a_variable_a_numpy_array_and_a_tensor_give_the_same_loss():
    ids, z = _random_batch(batch=2, n=7, seed=61)
    z32 = z.astype("float32")
    loss = MeanShiftInstanceLoss()
    with true_float32_matmul():
        from_tensor = keras.ops.convert_to_numpy(loss.call(_tensor(ids, "int32"), _tensor(z32)))
        from_numpy = keras.ops.convert_to_numpy(loss.call(ids.astype("int32"), z32))
        from_variable = keras.ops.convert_to_numpy(
            loss.call(tf.Variable(ids.astype("int32")), tf.Variable(z32))
        )
        scalar = float(loss(tf.Variable(ids.astype("int32")), tf.Variable(z32)))
    np.testing.assert_array_equal(from_numpy, from_tensor)
    np.testing.assert_array_equal(from_variable, from_tensor)
    np.testing.assert_allclose(scalar, from_tensor.mean(), atol=1e-6, rtol=0)


# ---------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------

# Measured 2026-10-29 on CPU against the float64 value of the very float32 points fed in
# (a float32 offset of 1000 quantises a 1e-2 gap to a multiple of 6.1e-5, so the expected
# value is taken from the stored points, not from the nominal gap): the centred loss deviates
# by at most 2e-4 in value and 2e-4 in gradient over the cases below.
TRANSLATION_VALUE_ATOL = 5e-4
TRANSLATION_GRAD_ATOL = 5e-4
TRANSLATION_CASES = [
    (0.0, 1e-3), (10.0, 1e-3), (100.0, 1e-3),
    (0.0, 1e-2), (10.0, 1e-2), (100.0, 1e-2), (1000.0, 1e-2),
]


def _two_close_points(offset, gap):
    """Two points of different instances ``gap`` apart along axis 0, shifted by ``offset``."""
    points = np.zeros((1, 2, 4))
    points[0, 1, 0] = gap
    return (points + offset).astype("float32")


def _value_and_gradient(fn_loss, points):
    variable = tf.Variable(points)
    with true_float32_matmul(), tf.GradientTape() as tape:
        value = fn_loss(tf.constant([[0, 1]], tf.int32), tf.identity(variable))
    return float(keras.ops.convert_to_numpy(value)), keras.ops.convert_to_numpy(
        tape.gradient(value, variable)
    )


def _check_translation_keeps_value_and_push_gradient(fn_loss, cases):
    margin = 2.0
    for offset, gap in cases:
        points = _two_close_points(offset, gap)
        stored_gap = float(points[0, 1, 0]) - float(points[0, 0, 0])
        value, grad = _value_and_gradient(fn_loss, points)
        expected_value = (margin - stored_gap) ** 2
        expected_grad = 2.0 * (margin - stored_gap)
        label = f"offset {offset} gap {gap}"
        np.testing.assert_allclose(
            value, expected_value, atol=TRANSLATION_VALUE_ATOL, rtol=0, err_msg=label
        )
        np.testing.assert_allclose(
            grad[0, 0, 0], expected_grad, atol=TRANSLATION_GRAD_ATOL, rtol=0, err_msg=label
        )
        np.testing.assert_allclose(
            grad[0, 1, 0], -expected_grad, atol=TRANSLATION_GRAD_ATOL, rtol=0, err_msg=label
        )
        assert abs(grad[0, 0, 0]) > 1.0, f"{label}: push gradient vanished"


@pytest.mark.parametrize("offset,gap", TRANSLATION_CASES)
def test_two_close_points_of_different_instances_push_the_same_at_any_offset(offset, gap):
    _check_translation_keeps_value_and_push_gradient(
        MeanShiftInstanceLoss(), [(offset, gap)]
    )


def test_the_value_and_gradient_at_every_offset_agree_with_the_offset_zero_result():
    loss = MeanShiftInstanceLoss()
    base_value, base_grad = _value_and_gradient(loss, _two_close_points(0.0, 1e-2))
    for offset in (10.0, 100.0, 1000.0):
        value, grad = _value_and_gradient(loss, _two_close_points(offset, 1e-2))
        np.testing.assert_allclose(
            value, base_value, atol=TRANSLATION_VALUE_ATOL, rtol=0, err_msg=f"offset {offset}"
        )
        np.testing.assert_allclose(
            grad, base_grad, atol=TRANSLATION_GRAD_ATOL, rtol=0, err_msg=f"offset {offset}"
        )


def test_red_an_uncentred_distance_loses_the_push_at_a_large_offset(monkeypatch):
    import dl_techniques.losses.mean_shift_instance_loss as loss_module

    def uncentred(a, b, key_mask=None):
        a_sq = keras.ops.sum(keras.ops.square(a), axis=-1, keepdims=True)
        b_sq = keras.ops.sum(keras.ops.square(b), axis=-1)[..., None, :]
        cross = keras.ops.matmul(a, keras.ops.swapaxes(b, -1, -2))
        return keras.ops.maximum(a_sq - 2.0 * cross + b_sq, 0.0)

    _check_translation_keeps_value_and_push_gradient(
        MeanShiftInstanceLoss(), TRANSLATION_CASES
    )
    monkeypatch.setattr(loss_module, "pairwise_sq_dist", uncentred)
    with pytest.raises(AssertionError):
        _check_translation_keeps_value_and_push_gradient(
            MeanShiftInstanceLoss(), [(100.0, 1e-2)]
        )
    with pytest.raises(AssertionError):
        _check_translation_keeps_value_and_push_gradient(
            MeanShiftInstanceLoss(), [(1000.0, 1e-2)]
        )


# ---------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------


def test_the_loss_is_positive_and_decreases_under_sgd_on_a_free_embedding():
    rng = np.random.default_rng(4)
    ids = np.tile(np.repeat(np.arange(3), 4), (2, 1))
    embedding = tf.Variable(0.3 * rng.standard_normal((2, 12, 3)).astype("float32"))
    loss = MeanShiftInstanceLoss(bandwidth=1.0)
    ids_t = _tensor(ids, "int32")
    values = []
    for _ in range(80):
        with true_float32_matmul(), tf.GradientTape() as tape:
            value = tf.reduce_mean(loss.call(ids_t, tf.identity(embedding)))
        values.append(float(value))
        embedding.assign_sub(0.5 * tape.gradient(value, embedding))
    assert values[0] > 0.5, f"initial loss {values[0]} should be clearly positive"
    assert all(b < a for a, b in zip(values[:6], values[1:7])), values[:7]
    assert values[-1] < 0.05 * values[0], (values[0], values[-1])


def test_ignored_points_receive_no_gradient():
    ids = np.array([[0, 0, 1, -1]])
    variable = tf.Variable(np.random.default_rng(2).standard_normal((1, 4, 2)).astype("float32"))
    with true_float32_matmul(), tf.GradientTape() as tape:
        value = tf.reduce_sum(MeanShiftInstanceLoss().call(_tensor(ids, "int32"), tf.identity(variable)))
    grad = keras.ops.convert_to_numpy(tape.gradient(value, variable))
    np.testing.assert_array_equal(grad[0, 3], np.zeros(2))
    assert np.abs(grad[0, :3]).sum() > 0


# ---------------------------------------------------------------------
# sample_weight, reduction
# ---------------------------------------------------------------------


def test_zeroing_a_sample_weight_equals_dropping_the_row():
    batch = 4
    ids, z = _random_batch(batch=batch, n=8, ignore_frac=0.1, seed=17)
    ids_t, z_t = _tensor(ids, "float32"), _tensor(z)
    weights = _tensor([1.0, 1.0, 1.0, 0.0])
    loss = MeanShiftInstanceLoss()
    with true_float32_matmul():
        weighted = float(loss(ids_t, z_t, sample_weight=weights))
        dropped = float(loss(ids_t[:-1], z_t[:-1]))
        row = keras.ops.convert_to_numpy(loss.call(ids_t, z_t))
    assert tuple(row.shape) == (batch,)
    np.testing.assert_allclose(
        weighted * batch / (batch - 1), dropped, atol=1e-6, rtol=0,
        err_msg="zeroing a sample_weight is not equivalent to dropping the row",
    )


def test_reduction_none_returns_the_per_row_vector():
    ids, z = _random_batch(batch=3, n=6, seed=8)
    loss = MeanShiftInstanceLoss(reduction="none")
    with true_float32_matmul():
        out = loss(_tensor(ids, "float32"), _tensor(z))
    assert tuple(out.shape) == (3,)


# ---------------------------------------------------------------------
# Margin and arguments
# ---------------------------------------------------------------------


def test_margin_defaults_to_twice_the_bandwidth_and_an_explicit_margin_is_honoured():
    assert MeanShiftInstanceLoss(bandwidth=0.75).resolved_margin == 1.5
    assert MeanShiftInstanceLoss(bandwidth=0.75, margin=0.4).resolved_margin == 0.4
    ids = np.array([[0, 1]])
    z = np.array([[[0.0], [1.0]]])
    # Two different-instance pairs at distance 1: hinge (m - 1)^2 each, mean (m - 1)^2.
    default = _real(MeanShiftInstanceLoss(bandwidth=1.0))(ids, z)
    explicit = _real(MeanShiftInstanceLoss(bandwidth=1.0, margin=3.0))(ids, z)
    small = _real(MeanShiftInstanceLoss(bandwidth=1.0, margin=0.5))(ids, z)
    np.testing.assert_allclose(default, [1.0], atol=1e-6, rtol=0)
    np.testing.assert_allclose(explicit, [4.0], atol=1e-6, rtol=0)
    np.testing.assert_allclose(small, [0.0], atol=1e-6, rtol=0)


@pytest.mark.parametrize(
    "kwargs,needle",
    [
        ({"bandwidth": 0.0}, "bandwidth"),
        ({"bandwidth": -1.0}, "-1.0"),
        ({"bandwidth": "wide"}, "wide"),
        ({"bandwidth": float("nan")}, "bandwidth"),
        ({"margin": 0.0}, "margin"),
        ({"margin": -2.5}, "-2.5"),
        ({"pull_weight": -0.5}, "-0.5"),
        ({"push_weight": -3.0}, "-3.0"),
        ({"pull_weight": float("inf")}, "pull_weight"),
        ({"pull_weight": None}, "pull_weight"),
        ({"max_points": 0}, "max_points"),
        ({"max_points": -4}, "-4"),
        ({"max_points": 2.5}, "2.5"),
        ({"ignore_label": 1.5}, "1.5"),
        ({"ignore_label": True}, "True"),
        ({"ignore_label": None}, "None"),
    ],
)
def test_bad_constructor_arguments_raise_naming_the_value(kwargs, needle):
    with pytest.raises(ValueError, match=needle):
        MeanShiftInstanceLoss(**kwargs)


# ---------------------------------------------------------------------
# Serialization and exports
# ---------------------------------------------------------------------

_KNOBS = dict(
    bandwidth=0.6, margin=1.7, pull_weight=0.8, push_weight=1.3, max_points=16, ignore_label=-7
)


def _fixed_value(loss):
    ids, z = _random_batch(batch=2, n=9, ignore_frac=0.2, seed=31)
    ids = np.where(ids == -1, -7, ids)
    return float(np.mean(_real(loss)(ids, z)))


@pytest.mark.parametrize("margin", [None, 1.7])
def test_get_config_round_trip_preserves_every_argument_and_the_value(margin):
    knobs = dict(_KNOBS, margin=margin)
    original = MeanShiftInstanceLoss(**knobs)
    config = original.get_config()
    for key, value in knobs.items():
        assert config[key] == value, (key, config[key], value)
    assert (config["margin"] is None) == (margin is None)
    restored = MeanShiftInstanceLoss.from_config(config)
    for key, value in knobs.items():
        assert getattr(restored, key) == value
    assert _fixed_value(restored) == pytest.approx(_fixed_value(original), abs=1e-6, rel=0)


def test_keras_registry_serialize_deserialize_round_trip_preserves_the_value():
    original = MeanShiftInstanceLoss(**_KNOBS)
    restored = keras.saving.deserialize_keras_object(
        keras.saving.serialize_keras_object(original)
    )
    assert isinstance(restored, MeanShiftInstanceLoss)
    assert restored.get_config()["max_points"] == 16
    assert _fixed_value(restored) == pytest.approx(_fixed_value(original), abs=1e-6, rel=0)


def test_the_loss_is_exported_from_the_losses_package():
    import dl_techniques.losses as losses_pkg
    from dl_techniques.losses import MeanShiftInstanceLoss as exported

    assert exported is MeanShiftInstanceLoss
    assert "MeanShiftInstanceLoss" in losses_pkg.__all__


# ---------------------------------------------------------------------
# Subsample
# ---------------------------------------------------------------------


def test_the_subsample_is_deterministic_and_equals_the_evenly_spaced_subset():
    batch, n, max_points = 2, 20, 8
    ids, z = _random_batch(batch=batch, n=n, ignore_frac=0.1, seed=41)
    spaced = [0, 2, 5, 7, 10, 12, 15, 17]
    assert spaced == [(k * n) // max_points for k in range(max_points)]
    capped = MeanShiftInstanceLoss(max_points=max_points)
    uncapped = MeanShiftInstanceLoss(max_points=64)
    first = _real(capped)(ids, z)
    second = _real(capped)(ids, z)
    np.testing.assert_array_equal(first, second)
    subset = _real(uncapped)(ids[:, spaced], z[:, spaced])
    np.testing.assert_allclose(first, subset, atol=1e-6, rtol=0)
    assert not np.allclose(first, _real(uncapped)(ids, z), atol=1e-4), (
        "the cap changed nothing, so the subsample is not exercised"
    )


def test_no_subsample_at_or_below_max_points():
    ids, z = _random_batch(batch=2, n=12, seed=43)
    at_cap = _real(MeanShiftInstanceLoss(max_points=12))(ids, z)
    far_below = _real(MeanShiftInstanceLoss(max_points=500))(ids, z)
    np.testing.assert_allclose(at_cap, far_below, atol=1e-6, rtol=0)


def test_an_unknown_point_count_is_not_subsampled_and_traces_in_a_graph():
    ids, z = _random_batch(batch=2, n=12, seed=47)
    loss = MeanShiftInstanceLoss(max_points=4)

    @tf.function(
        input_signature=[
            tf.TensorSpec([None, None], tf.int32),
            tf.TensorSpec([None, None, 3], tf.float32),
        ]
    )
    def graph(i, e):
        return loss.call(i, e)

    with true_float32_matmul():
        traced = keras.ops.convert_to_numpy(graph(_tensor(ids, "int32"), _tensor(z)))
        again = keras.ops.convert_to_numpy(
            graph(_tensor(ids[:, :9], "int32"), _tensor(z[:, :9]))
        )
    np.testing.assert_allclose(
        traced, _real(MeanShiftInstanceLoss(max_points=64))(ids, z), atol=1e-6, rtol=0
    )
    np.testing.assert_allclose(
        again, _real(MeanShiftInstanceLoss(max_points=64))(ids[:, :9], z[:, :9]),
        atol=1e-6, rtol=0,
    )


def _dynamic_loss(loss, ids, z):
    @tf.function(
        input_signature=[
            tf.TensorSpec([None, None], tf.int32),
            tf.TensorSpec([None, None, z.shape[-1]], tf.float32),
        ]
    )
    def graph(i, e):
        return loss.call(i, e)

    with true_float32_matmul():
        return keras.ops.convert_to_numpy(
            graph(_tensor(ids, "int32"), _tensor(z.astype("float32")))
        )


@pytest.mark.parametrize("n", [5, 8, 12])
def test_static_and_unknown_point_counts_agree_when_n_is_at_most_max_points(n):
    ids, z = _random_batch(batch=2, n=n, ignore_frac=0.2, seed=71 + n)
    loss = MeanShiftInstanceLoss(max_points=12)
    np.testing.assert_allclose(
        _real(loss)(ids, z), _dynamic_loss(loss, ids, z), atol=1e-6, rtol=0
    )


def test_a_static_count_above_max_points_drops_a_lone_off_stride_instance_and_an_unknown_count_does_not():
    # Five points, max_points=4: the stride keeps points 0..3, so instance 2 (point 4 only)
    # has no pair in the static case. Measured 2026-10-29: 2.4508 static, 2.4432 unknown N.
    rng = np.random.default_rng(0)
    z = rng.normal(size=(1, 5, 2))
    ids = np.array([[0, 1, 0, 1, 2]])
    capped = MeanShiftInstanceLoss(max_points=4)
    full = _real(MeanShiftInstanceLoss(max_points=64))(ids, z)
    static = _real(capped)(ids, z)
    unknown = _dynamic_loss(capped, ids, z)
    np.testing.assert_allclose(
        static, _real(MeanShiftInstanceLoss(max_points=64))(ids[:, :4], z[:, :4]),
        atol=1e-6, rtol=0,
    )
    np.testing.assert_allclose(unknown, full, atol=1e-6, rtol=0)
    assert abs(float(static[0]) - float(unknown[0])) > 1e-3, (
        "the static and unknown-count values coincide, so the documented difference is not exercised"
    )


# ---------------------------------------------------------------------
# Precision
# ---------------------------------------------------------------------


def test_mixed_float16_inputs_give_a_finite_float32_or_wider_loss():
    ids, z = _random_batch(batch=2, n=10, ignore_frac=0.2, seed=51)
    z16 = _tensor(z * 300.0, "float16")
    previous = keras.mixed_precision.dtype_policy()
    try:
        keras.mixed_precision.set_dtype_policy("mixed_float16")
        loss = MeanShiftInstanceLoss()
        per_row = loss.call(_tensor(ids, "int32"), z16)
        scalar = loss(_tensor(ids, "float32"), z16)
    finally:
        keras.mixed_precision.set_dtype_policy(previous)
    assert np.dtype(per_row.dtype.name) in (np.dtype("float32"), np.dtype("float64"))
    assert np.isfinite(keras.ops.convert_to_numpy(per_row)).all()
    assert np.isfinite(float(scalar))
    # A squared distance of 300-scale points is ~1e5, past the float16 maximum of 65504.
    assert float(np.max(keras.ops.convert_to_numpy(per_row))) > 65504.0


# ---------------------------------------------------------------------
# RED proofs: the check bodies fail on a mutated loss
# ---------------------------------------------------------------------


def test_red_swapping_pull_and_push_breaks_the_zero_loss_check():
    def swapped(ids, z):
        return _oracle(ids, z, margin=2.0, swap=True)

    _check_zero_on_separated_clusters(_real(MeanShiftInstanceLoss()))
    with pytest.raises(AssertionError):
        _check_zero_on_separated_clusters(swapped)


def test_red_dropping_the_ignore_mask_breaks_the_ignore_check():
    mutant = _real(MeanShiftInstanceLoss(ignore_label=-12345))
    _check_ignore_is_inert(_real(MeanShiftInstanceLoss()))
    with pytest.raises(AssertionError):
        _check_ignore_is_inert(mutant)


def test_red_a_cross_row_term_breaks_the_sample_weight_identity():
    def leaky(ids, z):
        per_row = _real(MeanShiftInstanceLoss())(ids, z)
        return per_row + per_row.mean()

    ids, z = _random_batch(batch=4, n=8, seed=17)
    weights = np.array([1.0, 1.0, 1.0, 0.0])
    honest = _real(MeanShiftInstanceLoss())
    assert np.allclose(
        (honest(ids, z) * weights).sum() / 4.0 * 4.0 / 3.0,
        honest(ids[:3], z[:3]).mean(), atol=1e-6, rtol=0,
    )
    assert not np.allclose(
        (leaky(ids, z) * weights).sum() / 4.0 * 4.0 / 3.0,
        leaky(ids[:3], z[:3]).mean(), atol=1e-6, rtol=0,
    )


# ---------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------

_BLOB_SCALE = 0.25
_BLOB_CENTRES = _BLOB_SCALE * np.array([[-4.0, -3.0], [4.0, -3.0], [0.0, 4.0]], dtype="float32")
_POINTS_PER_SAMPLE = 48
_BANDWIDTH = 1.0
_MERGE_RADIUS = 0.5 * _BANDWIDTH


def _blob_batch(rng, batch, ignore_frac):
    ids = rng.integers(0, 3, size=(batch, _POINTS_PER_SAMPLE))
    points = _BLOB_CENTRES[ids] + _BLOB_SCALE * 0.5 * rng.standard_normal(
        (batch, _POINTS_PER_SAMPLE, 2)
    ).astype("float32")
    if ignore_frac:
        drop = rng.random(ids.shape) < ignore_frac
        noise = rng.uniform(-8 * _BLOB_SCALE, 8 * _BLOB_SCALE, size=points.shape)
        points = np.where(drop[..., None], noise, points)
        ids = np.where(drop, -1, ids)
    return points.astype("float32"), ids.astype("float32")


def _clusters(model, points, true_ids):
    shifted = keras.ops.convert_to_numpy(model(points))
    scores, counts = [], []
    for row in range(points.shape[0]):
        labels = assign_clusters(shifted[row], merge_radius=_MERGE_RADIUS)
        scores.append(adjusted_rand_score(true_ids[row].astype(int), labels))
        counts.append(int(labels.max()) + 1)
    return scores, counts


@pytest.mark.parametrize("seed", [0, 1])
def test_end_to_end_three_blobs_are_recovered_by_a_trained_embedding_and_mean_shift(seed):
    """MLP embedding, ``MeanShiftLayer`` and the loss under stock ``fit``, read out by ``assign_clusters``.

    The blobs are scaled so that mean shift at the loss bandwidth merges them all into
    one cluster before training (checked first), so the recovered three clusters are the
    work of the trained embedding and not of the data.
    """
    keras.utils.set_random_seed(seed)
    rng = np.random.default_rng(seed)
    train_x, train_y = _blob_batch(rng, batch=256, ignore_frac=0.1)
    held_x, held_y = _blob_batch(np.random.default_rng(seed + 1000), batch=4, ignore_frac=0.0)

    inputs = keras.Input((_POINTS_PER_SAMPLE, 2))
    hidden = keras.layers.Dense(16, activation="relu")(inputs)
    hidden = keras.layers.Dense(8)(hidden)
    shifted = MeanShiftLayer(
        num_steps=3, bandwidth=_BANDWIDTH, learnable_bandwidth=False
    )(hidden)
    model = keras.Model(inputs, shifted)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-2),
        loss=MeanShiftInstanceLoss(bandwidth=_BANDWIDTH),
    )

    _, counts_before = _clusters(model, held_x, held_y)
    assert counts_before == [1] * 4, f"the untrained embedding already separates: {counts_before}"

    history = model.fit(
        train_x, train_y, batch_size=16, epochs=30, shuffle=True, verbose=0
    )
    losses = history.history["loss"]
    assert losses[-1] < 0.01 * losses[0], (losses[0], losses[-1])

    scores, counts = _clusters(model, held_x, held_y)
    assert counts == [3] * 4, f"cluster counts {counts}"
    assert min(scores) > 0.9, f"adjusted Rand scores {scores}"
