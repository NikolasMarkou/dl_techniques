"""Tests for ``ModePooling`` (``layers/sequence_pooling/mode_pooling.py``).

The layer pools a token set to the dominant mode of its kernel density by
mean-shift iterations. The claims pinned here: it ignores masked and excluded
tokens exactly, a fully masked row is the plain mean and finite, a far-away
minority does not drag it the way it drags the mean, a second start finds the
heavier mode when the mean start falls into the lighter one, and it survives
serialization, mixed precision and an unknown batch and length under
``tf.function``. The assertion bodies of the three claims that a broken layer
could fake (robustness, multi-start, mask isolation) are helper functions, and a
RED proof feeds each one a plain mean and asserts that it raises.

Multi-start permutation invariance is decided by construction, not hoped for:
the extra start points are the kept tokens at evenly spaced ranks of the
sequence, so for ``num_starts > 1`` the choice of starts depends on token order
and only ``num_starts=1`` is asserted invariant.
"""

import os
import re
import tempfile

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.mixtures.mean_shift_ops import kde_log_density
from dl_techniques.layers.sequence_pooling import mode_pooling as mode_pooling_module
from dl_techniques.layers.sequence_pooling import (
    ModePooling,
    STRICT_DROPPED_KEY_MARKER,
    create_sequence_pooling_layer,
    list_sequence_pooling_types,
)

pytestmark = pytest.mark.usefixtures("tf32_disabled")

BLOB_STD = 0.1


def _np(tensor):
    return np.asarray(tensor, dtype="float64")


def _outlier_data(seed=0, dim=8, n_inliers=70, n_outliers=30):
    """Blob A at the origin plus a loose far cluster; returns data and A's centre."""
    rng = np.random.default_rng(seed)
    inliers = BLOB_STD * rng.normal(size=(n_inliers, dim))
    far = np.zeros(dim)
    far[0] = 5.0
    outliers = far + 0.5 * rng.normal(size=(n_outliers, dim))
    tokens = np.concatenate([inliers, outliers])[None].astype("float32")
    return tokens, inliers.mean(axis=0)


def _assert_near_centre(pooled, centre, std=BLOB_STD):
    """Fail unless ``pooled`` is within 0.1 blob std of ``centre``."""
    distance = np.linalg.norm(_np(pooled).reshape(-1) - centre)
    assert distance <= 0.1 * std, f"distance {distance:.4g} > {0.1 * std:.4g}"


def _assert_pulled_away(pooled, centre, std=BLOB_STD):
    """Fail unless ``pooled`` is farther than one blob std from ``centre``."""
    distance = np.linalg.norm(_np(pooled).reshape(-1) - centre)
    assert distance > std, f"distance {distance:.4g} <= {std:.4g}"


def _two_modes_data(seed=0, dim=4):
    """Heavy blob at 0 (50 tokens), light blob at 10 (25), far blob at 40 (25).

    The mean (12.5 along axis 0) lies in the basin of the light blob, and the
    token order puts ranks 0 and 33 in the heavy blob, 66 in the light blob and
    99 in the far blob.
    """
    rng = np.random.default_rng(seed)

    def blob(n, at):
        centre = np.zeros(dim)
        centre[0] = at
        return centre + BLOB_STD * rng.normal(size=(n, dim))

    heavy, light, far = blob(50, 0.0), blob(25, 10.0), blob(25, 40.0)
    tokens = np.concatenate([heavy, light, far])[None].astype("float32")
    return tokens, heavy.mean(axis=0), light.mean(axis=0)


def _assert_heavier_blob(pooled, heavy, light):
    """Fail unless ``pooled`` sits on the heavy blob and not on the light one."""
    pooled = _np(pooled).reshape(-1)
    to_heavy = np.linalg.norm(pooled - heavy)
    to_light = np.linalg.norm(pooled - light)
    assert to_heavy < 0.2 and to_light > 5.0, (to_heavy, to_light)


def _assert_mask_ignored(pool, tokens, mask, position):
    """Fail unless rewriting the masked token at ``position`` leaves ``pool`` unchanged."""
    changed = tokens.copy()
    changed[:, position] += 1e3
    np.testing.assert_array_equal(_np(pool(tokens, mask)), _np(pool(changed, mask)))


def _plain_mean(tokens, mask):
    return np.mean(tokens, axis=1)


class TestRobustness:
    """A minority of far tokens drags the mean but not the mode."""

    @pytest.mark.parametrize("bandwidth", [0.5, "auto"])
    def test_the_mode_stays_on_the_big_blob_while_the_mean_is_pulled(self, bandwidth):
        tokens, centre = _outlier_data()
        _assert_pulled_away(tokens.mean(axis=1), centre)
        pooled = ModePooling(num_steps=10, bandwidth=bandwidth)(tokens)
        _assert_near_centre(pooled, centre)

    def test_red_proof_the_plain_mean_fails_the_robustness_body(self):
        tokens, centre = _outlier_data()
        with pytest.raises(AssertionError):
            _assert_near_centre(tokens.mean(axis=1), centre)
        pooled = ModePooling(num_steps=10, bandwidth=0.5)(tokens)
        with pytest.raises(AssertionError):
            _assert_pulled_away(pooled, centre)

    def test_zero_steps_is_the_mean_and_five_steps_is_not(self):
        tokens, centre = _outlier_data()
        mean = tokens.mean(axis=1)
        still = ModePooling(num_steps=0, bandwidth=0.5)(tokens)
        np.testing.assert_allclose(_np(still), mean, atol=1e-5, rtol=0)
        moved = ModePooling(num_steps=5, bandwidth=0.5)(tokens)
        assert np.linalg.norm(_np(moved) - mean) > 0.5

    def test_clean_unimodal_data_stays_close_to_the_mean(self):
        rng = np.random.default_rng(1)
        tokens = rng.normal(size=(4, 64, 8)).astype("float32")
        layer = ModePooling(num_steps=5)
        pooled = _np(layer(tokens))
        mean = tokens.mean(axis=1)
        pairs = np.linalg.norm(tokens[:, :, None] - tokens[:, None], axis=-1)
        off_diagonal = ~np.eye(64, dtype=bool)
        h = np.median(pairs[:, off_diagonal], axis=1)
        gap = np.linalg.norm(pooled - mean, axis=-1)
        assert (gap < 0.1 * h).all(), (gap, h)

    def test_a_lower_bandwidth_scale_changes_the_output(self):
        tokens, _ = _outlier_data()
        wide = _np(ModePooling(num_steps=5, bandwidth=0.5, bandwidth_scale=1.0)(tokens))
        narrow = _np(ModePooling(num_steps=5, bandwidth=0.5, bandwidth_scale=0.2)(tokens))
        assert np.linalg.norm(wide - narrow) > 0.05


class TestMultiStart:
    """A second start finds the heavier mode that the mean start misses."""

    def test_the_premise_the_mean_lies_in_the_light_basin(self):
        tokens, heavy, light = _two_modes_data()
        mean = tokens.mean(axis=1).reshape(-1)
        assert np.linalg.norm(mean - light) < np.linalg.norm(mean - heavy)

    def test_one_start_returns_the_light_blob_and_four_starts_the_heavy_one(self):
        tokens, heavy, light = _two_modes_data()
        single = ModePooling(num_steps=30, bandwidth=1.0, num_starts=1)(tokens)
        np.testing.assert_allclose(_np(single).reshape(-1), light, atol=0.2)
        with pytest.raises(AssertionError):
            _assert_heavier_blob(single, heavy, light)
        multi = ModePooling(num_steps=30, bandwidth=1.0, num_starts=4)(tokens)
        _assert_heavier_blob(multi, heavy, light)

    def test_the_density_follows_the_selected_start(self):
        tokens, _, _ = _two_modes_data()
        kwargs = dict(num_steps=30, bandwidth=1.0, return_density=True)
        _, one = ModePooling(num_starts=1, **kwargs)(tokens)
        _, four = ModePooling(num_starts=4, **kwargs)(tokens)
        assert float(four[0]) > float(one[0]) + 0.1


class TestMasking:
    """Masked, excluded and absent tokens carry no weight."""

    @pytest.mark.parametrize("num_starts", [1, 3])
    def test_masked_tokens_have_no_influence(self, num_starts):
        rng = np.random.default_rng(2)
        tokens = rng.normal(size=(2, 9, 4)).astype("float32")
        mask = np.ones((2, 9), "float32")
        mask[:, [2, 6]] = 0
        layer = ModePooling(num_steps=4, num_starts=num_starts)
        pool = lambda x, m: layer(x, mask=m)
        _assert_mask_ignored(pool, tokens, mask, 2)
        _assert_mask_ignored(pool, tokens, mask, 6)

    def test_red_proof_the_plain_mean_fails_the_mask_body(self):
        rng = np.random.default_rng(2)
        tokens = rng.normal(size=(2, 9, 4)).astype("float32")
        mask = np.ones((2, 9), "float32")
        mask[:, 2] = 0
        with pytest.raises(AssertionError):
            _assert_mask_ignored(_plain_mean, tokens, mask, 2)

    @pytest.mark.parametrize("num_starts", [1, 3])
    def test_an_excluded_position_is_not_a_key(self, num_starts):
        rng = np.random.default_rng(3)
        tokens = rng.normal(size=(2, 9, 4)).astype("float32")
        layer = ModePooling(num_steps=4, num_starts=num_starts, exclude_positions=[0, 4])
        base = _np(layer(tokens))
        for position in (0, 4):
            changed = tokens.copy()
            changed[:, position] += 1e3
            np.testing.assert_array_equal(base, _np(layer(changed)))

    def test_excluding_a_position_equals_dropping_it(self):
        rng = np.random.default_rng(4)
        tokens = rng.normal(size=(2, 9, 4)).astype("float32")
        excluded = ModePooling(num_steps=4, exclude_positions=[0])(tokens)
        dropped = ModePooling(num_steps=4)(tokens[:, 1:])
        np.testing.assert_allclose(_np(excluded), _np(dropped), atol=1e-5, rtol=0)

    @pytest.mark.parametrize("kwargs", [
        {},
        {"num_starts": 3},
        {"num_starts": 3, "return_density": True},
        {"learnable_bandwidth": True},
    ])
    def test_a_fully_masked_row_is_the_plain_mean_and_finite(self, kwargs):
        rng = np.random.default_rng(5)
        tokens = rng.normal(size=(3, 6, 4)).astype("float32")
        mask = np.ones((3, 6), "float32")
        mask[0] = 0
        out = ModePooling(num_steps=4, **kwargs)(tokens, mask=mask)
        pooled, density = out if isinstance(out, tuple) else (out, None)
        assert np.isfinite(_np(pooled)).all()
        if density is not None:
            assert np.isfinite(_np(density)).all()
        np.testing.assert_allclose(_np(pooled)[0], tokens[0].mean(axis=0), atol=1e-6, rtol=0)

    def test_exclusions_that_remove_every_position_give_the_plain_mean(self):
        tokens = np.random.default_rng(6).normal(size=(2, 3, 4)).astype("float32")
        layer = ModePooling(num_steps=4, exclude_positions=[0, 1, 2])
        np.testing.assert_allclose(
            _np(layer(tokens)), tokens.mean(axis=1), atol=1e-6, rtol=0
        )

    def test_the_unit_sphere_kernel_gives_a_unit_vector_and_a_finite_masked_row(self):
        tokens = np.random.default_rng(7).normal(size=(2, 6, 4)).astype("float32")
        mask = np.ones((2, 6), "float32")
        mask[1] = 0
        out = _np(ModePooling(num_steps=4, kernel="vmf", num_starts=3)(tokens, mask=mask))
        assert np.isfinite(out).all()
        np.testing.assert_allclose(np.linalg.norm(out[0]), 1.0, atol=1e-5)


def test_one_start_is_invariant_to_token_order():
    rng = np.random.default_rng(8)
    tokens = rng.normal(size=(3, 12, 5)).astype("float32")
    permuted = tokens[:, rng.permutation(12)]
    layer = ModePooling(num_steps=5, num_starts=1)
    np.testing.assert_allclose(_np(layer(tokens)), _np(layer(permuted)), atol=1e-5, rtol=0)


class TestGradients:
    @pytest.mark.parametrize("num_starts", [1, 4])
    def test_the_gradient_wrt_the_inputs_is_finite_and_live(self, num_starts):
        tokens = tf.constant(np.random.default_rng(9).normal(size=(2, 10, 4)).astype("float32"))
        layer = ModePooling(num_steps=3, num_starts=num_starts)
        with tf.GradientTape() as tape:
            tape.watch(tokens)
            loss = tf.reduce_sum(layer(tokens) * tf.range(1.0, 5.0))
        grad = _np(tape.gradient(loss, tokens))
        assert np.isfinite(grad).all()
        assert np.abs(grad).max() > 1e-6

    def test_the_learnable_scale_receives_a_nonzero_finite_gradient(self):
        tokens, _ = _outlier_data()
        layer = ModePooling(num_steps=3, learnable_bandwidth=True)
        with tf.GradientTape() as tape:
            out = layer(tf.constant(tokens))
            loss = tf.reduce_sum(out * tf.range(1.0, 9.0))
        grad = _np(tape.gradient(loss, layer.log_scale))
        assert np.isfinite(grad) and abs(grad) > 1e-6
        assert [w.name for w in layer.trainable_weights] == ["log_scale"]

    def test_the_learnable_scale_moves_the_output_when_it_moves(self):
        tokens, _ = _outlier_data()
        layer = ModePooling(num_steps=2, bandwidth=0.5, learnable_bandwidth=True)
        before = _np(layer(tokens))
        layer.log_scale.assign(np.log(0.2).astype("float32"))
        assert np.linalg.norm(_np(layer(tokens)) - before) > 0.05

    def test_a_fixed_bandwidth_layer_has_no_weights(self):
        layer = ModePooling()
        layer.build((None, 5, 4))
        assert layer.weights == []


GRID = 1024.0
DUPLICATE_POINT = -0.5


def _grid_outlier_data(seed=0, dim=8):
    """70 inliers (std 0.05) and 30 outliers (std 0.2) on a 2^-10 grid.

    On this grid a shift by an integer up to 1000 is exact in float32, so only the
    layer's own arithmetic can move the result.
    """
    rng = np.random.default_rng(seed)
    inliers = 0.05 * rng.normal(size=(70, dim))
    far = np.zeros(dim)
    far[0] = 2.0
    outliers = far + 0.2 * rng.normal(size=(30, dim))
    tokens = np.round(np.concatenate([inliers, outliers]) * GRID) / GRID
    return tokens[None].astype("float32")


def _duplicate_data(seed=0, dim=8, n_duplicates=85, n_others=15):
    """More than 71 percent identical tokens, so the median pair distance is 0."""
    rng = np.random.default_rng(seed)
    duplicates = np.full((n_duplicates, dim), DUPLICATE_POINT)
    others = 3.0 + 0.3 * rng.normal(size=(n_others, dim))
    return np.concatenate([duplicates, others])[None].astype("float32")


def _mask_last(total, masked):
    keep = np.ones((1, total), dtype="float32")
    keep[0, total - masked:] = 0.0
    return keep


class TestTranslationAndSmallBandwidths:
    """A constant offset of the tokens, a tiny bandwidth and duplicated tokens."""

    @pytest.mark.parametrize("bandwidth", ["auto", 0.1])
    @pytest.mark.parametrize("offset", [0.0, 10.0, 100.0, 1000.0])
    def test_the_mode_moves_with_a_constant_offset_of_the_tokens(self, bandwidth, offset):
        tokens = _grid_outlier_data()
        shifted = (tokens.astype("float64") + offset).astype("float32")
        assert (shifted.astype("float64") - offset == tokens).all()
        base = _np(ModePooling(num_steps=10, bandwidth=bandwidth)(tokens))
        moved = _np(ModePooling(num_steps=10, bandwidth=bandwidth)(shifted)) - offset
        # Measured at most 0.6 float32 ulp of the shifted value (6.1e-5 near 1000), so
        # the bound is 2 ulp; the uncentred distance form was off by 0.13 at 1000.
        bound = 2.0 * float(np.spacing(np.float32(offset + 4.0))) + 1e-6
        assert np.abs(moved - base).max() <= bound, (offset, np.abs(moved - base).max())

    def test_a_float_bandwidth_of_1e_6_with_a_mask_returns_the_duplicated_point_not_zeros(self):
        tokens = _duplicate_data()
        mask = _mask_last(100, 5)
        layer = ModePooling(num_steps=5, bandwidth=1e-6, return_density=True)
        pooled, density = layer(tokens, mask)
        np.testing.assert_allclose(_np(pooled), DUPLICATE_POINT, rtol=0, atol=1e-6)
        # A float bandwidth is not raised by the data floor, so h is 1e-6 here. The
        # float32 squared distance to the duplicated point then carries rounding noise
        # of about 1e-7 (log density near -6e4); the old -1e9 mask constant would give
        # a density below -1e6, so only that is pinned. The ops tests pin the value.
        assert np.isfinite(_np(density)).all() and _np(density)[0] > -1e6

    @pytest.mark.parametrize("bandwidth", [1e-6, 1e-4])
    def test_a_float_bandwidth_far_below_the_data_scale_with_a_mask_stays_on_the_data(self, bandwidth):
        tokens = _duplicate_data()
        mask = _mask_last(100, 5)
        pooled = _np(ModePooling(num_steps=5, bandwidth=bandwidth)(tokens, mask))
        np.testing.assert_allclose(pooled, DUPLICATE_POINT, rtol=0, atol=1e-6)

    @pytest.mark.parametrize("num_starts", [1, 4])
    @pytest.mark.parametrize("masked", [0, 5])
    def test_mostly_identical_tokens_pool_to_that_token(self, num_starts, masked):
        tokens = _duplicate_data()
        mask = _mask_last(100, masked) if masked else None
        layer = ModePooling(num_steps=5, num_starts=num_starts, return_density=True)
        pooled, density = layer(tokens) if mask is None else layer(tokens, mask)
        np.testing.assert_allclose(_np(pooled), DUPLICATE_POINT, rtol=0, atol=1e-6)
        assert np.isfinite(_np(density)).all()

    @pytest.mark.parametrize("num_starts", [1, 4])
    @pytest.mark.parametrize("masked", [0, 5])
    def test_the_gradients_stay_finite_with_identical_tokens_and_a_mask(self, num_starts, masked):
        tokens = tf.constant(_duplicate_data())
        mask = tf.constant(_mask_last(100, masked)) if masked else None
        layer = ModePooling(num_steps=3, num_starts=num_starts)
        with tf.GradientTape() as tape:
            tape.watch(tokens)
            out = layer(tokens) if mask is None else layer(tokens, mask)
            loss = tf.reduce_sum(out * tf.range(1.0, 9.0))
        grad = _np(tape.gradient(loss, tokens))
        assert np.isfinite(grad).all()
        if masked:
            assert np.abs(grad[:, 95:]).max() == 0.0

    def test_the_gradient_is_finite_for_a_shifted_token_set(self):
        tokens = tf.constant((_grid_outlier_data().astype("float64") + 1000.0).astype("float32"))
        layer = ModePooling(num_steps=3, bandwidth=0.1)
        with tf.GradientTape() as tape:
            tape.watch(tokens)
            loss = tf.reduce_sum(layer(tokens) * tf.range(1.0, 9.0))
        assert np.isfinite(_np(tape.gradient(loss, tokens))).all()

    def test_a_fully_masked_row_stays_the_plain_mean_at_a_tiny_bandwidth(self):
        tokens = _duplicate_data()
        mask = np.zeros((1, 100), dtype="float32")
        pooled = _np(ModePooling(num_steps=5, bandwidth=1e-6)(tokens, mask))
        np.testing.assert_allclose(pooled, tokens.mean(axis=1), rtol=0, atol=1e-6)


def _random_duplicate_batch(rows=40, dim=8, seed=1, value=None):
    """Rows of 75 to 95 percent identical tokens plus loose far tokens, shuffled.

    Returns the tokens ``(rows, 100, dim)``, a mask ``(rows, 100)`` that drops five
    random positions, and the duplicated value of every row ``(rows, dim)``. With
    ``value`` given the duplicated point is that scalar in every coordinate, otherwise
    a random float that float32 does not store exactly.
    """
    rng = np.random.default_rng(seed)
    tokens, mask, points = [], [], []
    for _ in range(rows):
        n_dup = int(rng.integers(75, 96))
        point = np.full(dim, value) if value is not None else rng.normal(size=dim)
        loose = point + 4.0 + rng.normal(size=(100 - n_dup, dim))
        row = np.concatenate([np.tile(point, (n_dup, 1)), loose])
        order = rng.permutation(100)
        keep = np.ones(100)
        keep[rng.choice(100, 5, replace=False)] = 0.0
        tokens.append(row[order])
        mask.append(keep)
        points.append(point)
    return (
        np.stack(tokens).astype("float32"),
        np.stack(mask).astype("float32"),
        np.stack(points).astype("float32"),
    )


def _assert_duplicated_mode_is_pooled(num_starts, masked, value=None):
    """Fail unless every row pools to its duplicated point within 1e-3."""
    tokens, mask, points = _random_duplicate_batch(value=value)
    pool = ModePooling(num_steps=5, num_starts=num_starts)
    pooled = _np(pool(tokens, mask) if masked else pool(tokens))
    error = np.abs(pooled - _np(points)).max(axis=1)
    wrong = int((error > 1e-3).sum())
    assert wrong == 0, f"{wrong} of {len(error)} rows off the duplicated point, worst {error.max():.4g}"


class TestTheAutoBandwidthFloorFollowsTheDataScale:
    """Mostly duplicated tokens give a median distance of 0, then rounding noise sets the mode."""

    @pytest.mark.parametrize("masked", [False, True])
    @pytest.mark.parametrize("num_starts", [1, 2, 4])
    def test_random_duplicated_values_pool_to_the_duplicated_point(self, num_starts, masked):
        _assert_duplicated_mode_is_pooled(num_starts, masked)

    @pytest.mark.parametrize("value", [1.1, 0.3141, -0.7071, DUPLICATE_POINT])
    @pytest.mark.parametrize("num_starts", [2, 4])
    def test_named_duplicated_values_with_a_mask_pool_to_the_duplicated_point(self, num_starts, value):
        _assert_duplicated_mode_is_pooled(num_starts, True, value=value)

    @pytest.mark.parametrize("value", [1.1, 0.3141])
    def test_the_plan_layout_with_a_non_representable_value_pools_to_it(self, value):
        tokens = _duplicate_data()
        tokens[:, :85] = value
        pooled = _np(ModePooling(num_steps=5, num_starts=4)(tokens, _mask_last(100, 5)))
        np.testing.assert_allclose(pooled, value, rtol=0, atol=1e-3)

    def test_ordinary_data_does_not_reach_the_floor(self, monkeypatch):
        tokens, _ = _outlier_data()
        mask = _mask_last(100, 7)
        for num_starts in (1, 3):
            pool = ModePooling(num_steps=6, num_starts=num_starts, bandwidth_scale=0.5)
            with_floor = _np(pool(tokens, mask))
            monkeypatch.setattr(mode_pooling_module, "_BANDWIDTH_DATA_FLOOR", 0.0)
            without_floor = _np(pool(tokens, mask))
            monkeypatch.undo()
            np.testing.assert_allclose(with_floor, without_floor, rtol=0, atol=1e-6)

    @pytest.mark.parametrize("scale", [1e-3, 1e-2, 0.5, 2.0])
    @pytest.mark.parametrize("layout", ["duplicated", "ordinary"])
    def test_the_bandwidth_scale_multiplies_the_floored_bandwidth(self, scale, layout):
        # The data floor bounds the median heuristic; bandwidth_scale acts after it.
        tokens = _duplicate_data() if layout == "duplicated" else _outlier_data()[0]
        keep = keras.ops.ones((1, 100), dtype="bool")

        def bandwidth(bandwidth_scale):
            layer = ModePooling(bandwidth_scale=bandwidth_scale)
            layer.build(tokens.shape)
            h = layer._resolve_bandwidth(keras.ops.convert_to_tensor(tokens), keep, "float32")
            return float(_np(h).reshape(-1)[0])

        np.testing.assert_allclose(bandwidth(scale) / bandwidth(1.0), scale, rtol=1e-6, atol=0)

    @pytest.mark.parametrize("log_scale", [-2.0, 0.7])
    def test_the_learnable_multiplier_scales_the_floored_bandwidth(self, log_scale):
        tokens = _duplicate_data()
        keep = keras.ops.ones((1, 100), dtype="bool")
        layer = ModePooling(learnable_bandwidth=True)
        layer.build(tokens.shape)
        base = float(_np(layer._resolve_bandwidth(keras.ops.convert_to_tensor(tokens), keep, "float32")).reshape(-1)[0])
        layer.log_scale.assign(np.float32(log_scale))
        h = float(_np(layer._resolve_bandwidth(keras.ops.convert_to_tensor(tokens), keep, "float32")).reshape(-1)[0])
        np.testing.assert_allclose(h / base, np.exp(log_scale), rtol=1e-6, atol=0)

    def test_the_learnable_scale_receives_a_nonzero_finite_gradient_when_the_floor_binds(self):
        # Mostly duplicated tokens: the median distance is 0, so the floor sets h.
        tokens = tf.constant(_duplicate_data())
        layer = ModePooling(num_steps=3, learnable_bandwidth=True)
        layer.build(tokens.shape)
        h = layer._resolve_bandwidth(tokens, keras.ops.ones((1, 100), dtype="bool"), "float32")
        assert float(_np(h).reshape(-1)[0]) == pytest.approx(
            mode_pooling_module._BANDWIDTH_DATA_FLOOR
            * float(_np(layer._rms_radius(tokens, keras.ops.ones((1, 100), dtype="bool"), "float32")).reshape(-1)[0]),
            rel=1e-5,
        )
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(layer(tokens) * tf.range(1.0, 9.0))
        grad = _np(tape.gradient(loss, layer.log_scale))
        assert np.isfinite(grad) and abs(grad) > 1e-6, grad

    def test_a_float_bandwidth_is_not_floored_by_the_data_scale(self):
        tokens, _ = _outlier_data()
        layer = ModePooling(bandwidth=1e-5)
        layer.build(tokens.shape)
        h = layer._resolve_bandwidth(
            keras.ops.convert_to_tensor(tokens),
            keras.ops.ones((1, 100), dtype="bool"),
            "float32",
        )
        np.testing.assert_allclose(_np(h), 1e-5, rtol=1e-6)

    @pytest.mark.parametrize("num_starts", [2, 4])
    def test_the_check_fails_when_the_floor_is_the_old_absolute_constant(self, monkeypatch, num_starts):
        monkeypatch.setattr(mode_pooling_module, "_BANDWIDTH_DATA_FLOOR", 0.0)
        with pytest.raises(AssertionError, match="off the duplicated point"):
            _assert_duplicated_mode_is_pooled(num_starts, True, value=1.1)


class TestPrecisionAndGraphs:
    def test_mixed_float16_is_finite_and_close_to_float32(self, mixed_float16_policy):
        rng = np.random.default_rng(10)
        tokens = rng.normal(size=(3, 16, 8)).astype("float32")
        mask = np.ones((3, 16), "float32")
        mask[1, 10:] = 0
        half = ModePooling(num_steps=4, num_starts=3, return_density=True)
        assert half.compute_dtype == "float16"
        pooled, density = half(tokens, mask=mask)
        assert pooled.dtype == "float16"
        keras.mixed_precision.set_global_policy("float32")
        full = ModePooling(num_steps=4, num_starts=3, return_density=True)
        reference, reference_density = full(tokens, mask=mask)
        assert np.isfinite(_np(pooled)).all() and np.isfinite(_np(density)).all()
        np.testing.assert_allclose(_np(pooled), _np(reference), atol=1e-2, rtol=0)
        np.testing.assert_allclose(_np(density), _np(reference_density), atol=2e-2, rtol=0)

    def test_a_graph_with_unknown_batch_and_length_matches_eager_at_two_lengths(self):
        layer = ModePooling(num_steps=4, num_starts=4, exclude_positions=[0], return_density=True)

        @tf.function(input_signature=[
            tf.TensorSpec([None, None, 6], tf.float32),
            tf.TensorSpec([None, None], tf.float32),
        ])
        def run(x, m):
            return layer(x, mask=m)

        rng = np.random.default_rng(11)
        for length, batch in ((7, 3), (13, 2)):
            tokens = rng.normal(size=(batch, length, 6)).astype("float32")
            mask = np.ones((batch, length), "float32")
            mask[0, length // 2:] = 0
            graph_pooled, graph_density = run(tokens, mask)
            eager_pooled, eager_density = layer(tokens, mask=mask)
            np.testing.assert_allclose(_np(graph_pooled), _np(eager_pooled), atol=1e-5, rtol=0)
            np.testing.assert_allclose(_np(graph_density), _np(eager_density), atol=1e-4, rtol=0)


class TestDensity:
    def test_the_density_at_the_mode_beats_the_density_at_the_mean(self):
        tokens, _ = _outlier_data()
        pooled, density = ModePooling(num_steps=10, bandwidth=0.5, return_density=True)(tokens)
        at_mean = kde_log_density(tokens.mean(axis=1, keepdims=True), tokens, 0.5)
        assert density.shape == (1,)
        assert np.isfinite(_np(density)).all()
        assert float(density[0]) > float(at_mean[0, 0]) + 1.0
        direct = kde_log_density(_np(pooled)[:, None].astype("float32"), tokens, 0.5)
        np.testing.assert_allclose(_np(density), _np(direct)[:, 0], atol=1e-4, rtol=0)


class TestShapesAndFactory:
    @pytest.mark.parametrize("return_density", [False, True])
    def test_compute_output_shape_matches_the_symbolic_call(self, return_density):
        layer = ModePooling(return_density=return_density)
        inputs = keras.Input(shape=(5, 8))
        symbolic = layer(inputs)
        declared = layer.compute_output_shape((None, 5, 8))
        if return_density:
            assert (symbolic[0].shape, symbolic[1].shape) == tuple(declared)
        else:
            assert symbolic.shape == declared == (None, 8)

    def test_the_factory_builds_mode_and_lists_it(self):
        assert "mode" in list_sequence_pooling_types()
        layer = create_sequence_pooling_layer("mode", num_steps=3, num_starts=2, name="mode_pool")
        assert isinstance(layer, ModePooling)
        assert (layer.num_steps, layer.num_starts, layer.name) == (3, 2, "mode_pool")

    def test_the_factory_refuses_an_unknown_keyword_and_a_bad_value(self):
        with pytest.raises(ValueError, match=re.escape(STRICT_DROPPED_KEY_MARKER)) as error:
            create_sequence_pooling_layer("mode", bogus_key=1)
        assert "bogus_key" in str(error.value)
        with pytest.raises(ValueError, match="num_starts"):
            create_sequence_pooling_layer("mode", num_starts=0)
        with pytest.raises(ValueError, match="bandwidth_scale"):
            create_sequence_pooling_layer("mode", bandwidth_scale=-1.0)

    @pytest.mark.parametrize("kwargs, text", [
        ({"num_steps": -1}, "-1"),
        ({"num_steps": True}, "True"),
        ({"num_steps": 2.0}, "2.0"),
        ({"bandwidth": "wide"}, "wide"),
        ({"bandwidth": 0.0}, "0.0"),
        ({"bandwidth": -1.0}, "-1.0"),
        ({"bandwidth_scale": 0.0}, "0.0"),
        ({"num_starts": 0}, "0"),
        ({"num_starts": 2.0}, "2.0"),
        ({"kernel": "cosine"}, "cosine"),
        ({"exclude_positions": [-1]}, "-1"),
        ({"exclude_positions": [1, 1]}, "1"),
        ({"learnable_bandwidth": "yes"}, "yes"),
        ({"return_density": 1}, "1"),
    ])
    def test_a_bad_constructor_argument_is_refused_naming_the_value(self, kwargs, text):
        with pytest.raises(ValueError, match=re.escape(text)):
            ModePooling(**kwargs)

    @pytest.mark.parametrize("shape", [(None, 5), (None, 5, None), (None, 3, 4, 2)])
    def test_build_refuses_a_bad_input_shape(self, shape):
        with pytest.raises(ValueError, match="embed_dim"):
            ModePooling().build(shape)


class TestSerialization:
    KWARGS = dict(
        num_steps=3, bandwidth=0.7, bandwidth_scale=1.5, learnable_bandwidth=True,
        num_starts=3, kernel="vmf", exclude_positions=[0], return_density=True,
    )

    def test_get_config_holds_every_constructor_argument(self):
        config = ModePooling(**self.KWARGS).get_config()
        for key, value in self.KWARGS.items():
            assert config[key] == value, key
        assert ModePooling().get_config()["exclude_positions"] == []

    def test_from_config_rebuilds_an_equal_layer(self):
        config = ModePooling(**self.KWARGS).get_config()
        assert ModePooling.from_config(config).get_config() == config

    @pytest.mark.parametrize("kwargs", [
        KWARGS,
        dict(num_steps=4, num_starts=2),
    ], ids=["all_knobs", "defaults_plus"])
    def test_a_keras_file_round_trip_keeps_the_values(self, kwargs):
        tokens = np.random.default_rng(12).normal(size=(3, 6, 4)).astype("float32")
        inputs = keras.Input(shape=(6, 4))
        outputs = ModePooling(**kwargs, name="mode")(inputs)
        model = keras.Model(inputs, outputs)
        if kwargs.get("learnable_bandwidth"):
            model.get_layer("mode").log_scale.assign(0.3)
        expected = model(tokens, training=False)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "mode.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
        actual = loaded(tokens, training=False)
        if isinstance(expected, (list, tuple)):
            for want, got in zip(expected, actual):
                np.testing.assert_allclose(_np(got), _np(want), atol=1e-6, rtol=0)
        else:
            np.testing.assert_allclose(_np(actual), _np(expected), atol=1e-6, rtol=0)
        if kwargs.get("learnable_bandwidth"):
            assert float(loaded.get_layer("mode").log_scale) == pytest.approx(0.3, abs=1e-7)
