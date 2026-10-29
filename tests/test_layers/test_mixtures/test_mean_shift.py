"""Tests for ``MeanShiftLayer`` and ``assign_clusters`` (``layers/mixtures/mean_shift.py``).

The layer is a thin owner of the bandwidth weight, the optional projection and the
mask, around ``mean_shift_ops.mean_shift_iterate``. Its output is therefore checked
against two references that do not share its code: the ops themselves (identity, on
the same inputs and bandwidth) and the float64 oracle of ``mean_shift_oracle.py``
(textbook key-by-key weights). The two identity checks and the mask check are
factored into functions so a RED proof can inject a defect into the layer's call of
the ops and assert that the very same body fails.

The last classes cover the mixtures registration (the factory stays non-strict:
an unknown kwarg warns and is dropped) and the eval-only ``assign_clusters`` readout.
"""

import contextlib
import inspect
import json
import logging
import math
from typing import get_args

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.mixtures import (
    MIXTURE_REGISTRY,
    MeanShiftLayer,
    MixtureType,
    assign_clusters,
    create_mixture_layer,
    get_mixture_info,
    validate_mixture_config,
)
from dl_techniques.layers.mixtures import mean_shift as mean_shift_module
from dl_techniques.layers.mixtures import mean_shift_ops as ops

from .mean_shift_oracle import oracle_iterate

pytestmark = pytest.mark.usefixtures("tf32_disabled")

DIM = 4


def _np(tensor):
    return keras.ops.convert_to_numpy(tensor)


def _points(batch, count, seed, dim=DIM, scale=1.0):
    rng = np.random.default_rng(seed)
    return (scale * rng.normal(size=(batch, count, dim))).astype("float32")


def _make(dim=DIM, **kwargs):
    layer = MeanShiftLayer(**kwargs)
    layer.build((None, None, dim))
    return layer


def _set_log_bandwidth(layer, bandwidth):
    layer.log_bandwidth.assign(np.float32(math.log(bandwidth)))


@contextlib.contextmanager
def _policy(name):
    previous = keras.mixed_precision.global_policy().name
    keras.mixed_precision.set_global_policy(name)
    try:
        yield
    finally:
        keras.mixed_precision.set_global_policy(previous)


# ---------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------

class TestConstructor:
    @pytest.mark.parametrize("kwargs,pattern", [
        ({"num_steps": 0}, "num_steps.*got 0"),
        ({"num_steps": -2}, "num_steps.*got -2"),
        ({"num_steps": 2.5}, r"num_steps.*float \(2.5\)"),
        ({"num_steps": True}, "num_steps.*bool"),
        ({"bandwidth": 0.0}, "bandwidth.*got 0.0"),
        ({"bandwidth": -1.0}, "bandwidth.*got -1.0"),
        ({"bandwidth": float("nan")}, "bandwidth.*nan"),
        ({"bandwidth": True}, "bandwidth.*bool"),
        ({"bandwidth": "wide"}, "bandwidth.*wide"),
        ({"bandwidth": 1e-3}, r"bandwidth must lie in \[0.01001, 99.9\].*got 0.001"),
        ({"bandwidth": 500.0}, r"bandwidth must lie in \[0.01001, 99.9\].*got 500.0"),
        ({"step_size": 0.0}, "step_size.*got 0.0"),
        ({"step_size": 2.0}, "step_size.*got 2.0"),
        ({"step_size": -0.5}, "step_size.*got -0.5"),
        ({"step_size": True}, "step_size.*True"),
        ({"kernel": "laplace"}, "kernel must be one of.*'laplace'"),
        ({"learnable_bandwidth": 1}, "learnable_bandwidth must be a bool, got 1"),
        ({"blurring": "yes"}, "blurring must be a bool, got 'yes'"),
        ({"use_projection": 1}, "use_projection must be a bool, got 1"),
        ({"return_affinity": None}, "return_affinity must be a bool, got None"),
        ({"use_projection": True}, "projection_dim.*None"),
        ({"use_projection": True, "projection_dim": 0}, "projection_dim.*got 0"),
        ({"use_projection": True, "projection_dim": 2.0}, "projection_dim.*float"),
    ])
    def test_a_bad_argument_raises_value_error_naming_the_value(self, kwargs, pattern):
        with pytest.raises(ValueError, match=pattern):
            MeanShiftLayer(**kwargs)

    def test_the_documented_defaults(self):
        layer = MeanShiftLayer()
        assert (layer.num_steps, layer.bandwidth, layer.learnable_bandwidth) == (5, 1.0, True)
        assert (layer.blurring, layer.step_size, layer.kernel) == (True, 1.0, "gaussian")
        assert (layer.use_projection, layer.projection_dim) == (False, None)
        assert layer.return_affinity is False
        assert layer.projection is None

    @pytest.mark.parametrize("bandwidth", [1e-2, 1e2, 1e-2 * 1.0005, 1e2 * 0.9995, 1e-3, 500.0])
    def test_a_bandwidth_at_an_end_of_the_clip_range_or_outside_it_raises(self, bandwidth):
        with pytest.raises(ValueError, match=rf"bandwidth must lie in .*got {bandwidth!r}"):
            MeanShiftLayer(bandwidth=bandwidth)

    @pytest.mark.parametrize("bandwidth", [0.011, 1e-2 * 1.001, 1.0, 1e2 * 0.999, 99.0])
    def test_a_bandwidth_inside_the_margin_is_accepted_and_the_clip_is_inactive(self, bandwidth):
        layer = _make(bandwidth=bandwidth)
        assert layer.bandwidth == bandwidth
        effective = float(_np(layer._current_bandwidth("float32")))
        assert effective == pytest.approx(bandwidth, rel=1e-5)

    def test_numpy_numbers_are_stored_as_python_numbers(self):
        layer = MeanShiftLayer(num_steps=np.int64(3), bandwidth=np.float32(2.0))
        assert type(layer.num_steps) is int and type(layer.bandwidth) is float
        json.dumps(layer.get_config())

    def test_projection_dim_is_ignored_but_stored_without_projection(self):
        layer = MeanShiftLayer(use_projection=False, projection_dim=7)
        assert layer.projection is None and layer.projection_dim == 7
        assert layer.get_config()["projection_dim"] == 7

    def test_the_legal_output_mode_set_is_empty_and_a_frozenset(self):
        assert MeanShiftLayer.VALID_OUTPUT_MODES == frozenset()
        assert isinstance(MeanShiftLayer.VALID_OUTPUT_MODES, frozenset)
        with pytest.raises(ValueError, match="Unrecognized keyword arguments.*output_mode"):
            MeanShiftLayer(output_mode="basis")

    def test_it_is_a_plain_layer_not_a_base_mixture_layer(self):
        from dl_techniques.layers.mixtures.base import BaseMixtureLayer

        assert not issubclass(MeanShiftLayer, BaseMixtureLayer)

    def test_build_rejects_a_wrong_rank_and_an_unknown_width(self):
        with pytest.raises(ValueError, match="rank 2"):
            MeanShiftLayer().build((None, 4))
        with pytest.raises(ValueError, match="rank 4"):
            MeanShiftLayer().build((None, 3, 3, 4))
        with pytest.raises(ValueError, match="last dimension"):
            MeanShiftLayer().build((None, 5, None))


# ---------------------------------------------------------------------
# Shapes, weights
# ---------------------------------------------------------------------

class TestShapesAndWeights:
    @pytest.mark.parametrize("use_projection", [False, True])
    @pytest.mark.parametrize("return_affinity", [False, True])
    def test_output_shapes(self, use_projection, return_affinity):
        width = 3 if use_projection else DIM
        kwargs = {"use_projection": True, "projection_dim": 3} if use_projection else {}
        layer = _make(return_affinity=return_affinity, **kwargs)
        x = _points(2, 6, 0)
        out = layer(x)
        if return_affinity:
            shifted, affinity = out
            assert tuple(shifted.shape) == (2, 6, width)
            assert tuple(affinity.shape) == (2, 6, 6)
            assert layer.compute_output_shape((None, 6, DIM)) == (
                (None, 6, width), (None, 6, 6)
            )
        else:
            assert tuple(out.shape) == (2, 6, width)
            assert layer.compute_output_shape((None, 6, DIM)) == (None, 6, width)

    def test_compute_output_shape_works_before_build(self):
        layer = MeanShiftLayer(use_projection=True, projection_dim=2, return_affinity=True)
        assert layer.compute_output_shape((3, 7, 9)) == ((3, 7, 2), (3, 7, 7))
        assert MeanShiftLayer().compute_output_shape((3, 7, 9)) == (3, 7, 9)

    @pytest.mark.parametrize("kernel", ["gaussian", "vmf"])
    def test_the_affinity_rows_sum_to_one_and_masked_columns_are_zero(self, kernel):
        layer = _make(kernel=kernel, return_affinity=True, num_steps=3)
        x = _points(2, 6, 1)
        mask = np.array([[1, 1, 1, 1, 0, 0], [1] * 6], dtype="float32")
        _, affinity = layer(x, mask=mask)
        affinity = _np(affinity)
        np.testing.assert_allclose(affinity.sum(axis=-1), 1.0, atol=1e-6, rtol=0)
        assert (affinity[0, :, 4:] == 0.0).all()
        assert (affinity >= 0.0).all()

    def test_the_learnable_bandwidth_is_one_scalar_initialised_to_log_h(self):
        layer = _make(bandwidth=2.5)
        assert [w.name for w in layer.trainable_weights] == ["log_bandwidth"]
        assert tuple(layer.log_bandwidth.shape) == ()
        np.testing.assert_allclose(_np(layer.log_bandwidth), math.log(2.5), atol=1e-7)

    def test_a_fixed_bandwidth_has_no_weight(self):
        layer = _make(learnable_bandwidth=False)
        assert layer.weights == [] and layer.trainable_weights == []
        assert layer.count_params() == 0

    def test_the_projection_is_bias_free_and_named(self):
        layer = _make(use_projection=True, projection_dim=3)
        assert layer.projection.name == "projection"
        assert layer.projection.use_bias is False
        assert tuple(layer.projection.kernel.shape) == (DIM, 3)
        assert layer.count_params() == DIM * 3 + 1

    def test_the_bandwidth_is_clipped_to_the_range(self):
        high = _make()
        low = _make()
        high.log_bandwidth.assign(np.float32(50.0))
        low.log_bandwidth.assign(np.float32(-50.0))
        assert float(_np(high._current_bandwidth("float32"))) == pytest.approx(1e2)
        assert float(_np(low._current_bandwidth("float32"))) == pytest.approx(1e-2)
        x = _points(2, 6, 2)
        assert np.isfinite(_np(high(x))).all() and np.isfinite(_np(low(x))).all()


# ---------------------------------------------------------------------
# Identity with the ops and with the float64 oracle
# ---------------------------------------------------------------------

def _check_matches_the_ops(kernel, blurring, num_steps=3, step_size=1.0, projection=False):
    """The layer's output is the ops' iteration on the same (projected) points."""
    extra = {"use_projection": True, "projection_dim": 3} if projection else {}
    layer = _make(
        kernel=kernel, blurring=blurring, num_steps=num_steps, step_size=step_size,
        learnable_bandwidth=False, bandwidth=0.7, **extra,
    )
    x = _points(2, 7, 3)
    mask = np.array([[1, 1, 1, 1, 1, 0, 0], [1] * 7], dtype="float32")
    base = layer.projection(x) if projection else x
    for use_mask in (None, mask):
        expected = ops.mean_shift_iterate(
            base, bandwidth=0.7, num_steps=num_steps, step_size=step_size,
            blurring=blurring, mask=use_mask, kernel=kernel,
        )
        got = layer(x, mask=use_mask)
        # Measured difference: exactly 0.0 (same ops, same constant bandwidth).
        np.testing.assert_allclose(_np(got), _np(expected), atol=1e-7, rtol=0)


def _check_matches_the_oracle(kernel, blurring, num_steps, atol):
    """The layer's output is the float64 key-by-key oracle, to float32 tolerance."""
    layer = _make(
        kernel=kernel, blurring=blurring, num_steps=num_steps,
        learnable_bandwidth=False, bandwidth=0.9, step_size=0.8,
    )
    x = _points(2, 7, 4)
    keep = np.array([[1, 1, 1, 1, 1, 1, 0], [1] * 7], dtype=bool)
    expected = oracle_iterate(
        x.astype("float64"), 0.9, num_steps, step_size=0.8, blurring=blurring,
        keep=keep, kernel=kernel,
    )
    got = _np(layer(x, mask=keep.astype("float32")))
    assert got.dtype == np.float32
    np.testing.assert_allclose(got.astype("float64"), expected, atol=atol, rtol=0)


class TestIdentities:
    @pytest.mark.parametrize("kernel", ["gaussian", "vmf"])
    @pytest.mark.parametrize("blurring", [True, False])
    def test_the_layer_equals_the_ops(self, kernel, blurring):
        _check_matches_the_ops(kernel, blurring)

    @pytest.mark.parametrize("step_size", [0.5, 1.5])
    def test_the_layer_equals_the_ops_with_a_damped_or_over_relaxed_step(self, step_size):
        _check_matches_the_ops("gaussian", True, step_size=step_size)

    def test_the_layer_equals_the_ops_behind_the_projection(self):
        _check_matches_the_ops("gaussian", True, projection=True)

    @pytest.mark.parametrize("kernel", ["gaussian", "vmf"])
    @pytest.mark.parametrize("blurring", [True, False])
    def test_one_step_equals_the_float64_oracle(self, kernel, blurring):
        _check_matches_the_oracle(kernel, blurring, 1, atol=1e-6)

    @pytest.mark.parametrize("kernel", ["gaussian", "vmf"])
    @pytest.mark.parametrize("blurring", [True, False])
    def test_three_steps_equal_the_float64_oracle(self, kernel, blurring):
        # Measured float32 differences: 2.5e-7 (one step), 7.1e-7 (three steps).
        _check_matches_the_oracle(kernel, blurring, 3, atol=3e-6)

    def test_a_learnable_bandwidth_gives_the_same_output_as_the_constant(self):
        x = _points(2, 6, 5)
        fixed = _make(learnable_bandwidth=False, bandwidth=1.7)
        learnable = _make(learnable_bandwidth=True, bandwidth=1.7)
        np.testing.assert_allclose(_np(fixed(x)), _np(learnable(x)), atol=1e-6, rtol=0)

    def test_float64_matches_the_oracle_to_double_precision(self, float64_policy):
        layer = _make(num_steps=3, bandwidth=0.9, learnable_bandwidth=False)
        x = _points(2, 7, 6).astype("float64")
        out = layer(x)
        assert keras.backend.standardize_dtype(out.dtype) == "float64"
        expected = oracle_iterate(x, 0.9, 3)
        np.testing.assert_allclose(_np(out), expected, atol=1e-10, rtol=0)

    def test_the_bandwidth_is_the_kernel_width(self):
        x = _points(1, 5, 7)
        one = _make(num_steps=1, bandwidth=0.5, learnable_bandwidth=False)
        got = _np(one(x))[0].astype("float64")
        expected = oracle_iterate(x.astype("float64"), 0.5, 1)[0]
        other = oracle_iterate(x.astype("float64"), 0.5 * 1.3, 1)[0]
        np.testing.assert_allclose(got, expected, atol=1e-5, rtol=0)
        assert np.abs(got - other).max() > 1e-3


def _defect_wider_bandwidth(real):
    def patched(x, bandwidth, *args, **kwargs):
        return real(x, bandwidth * 1.2, *args, **kwargs)
    return patched


def _defect_ignore_the_mask(real):
    def patched(*args, **kwargs):
        kwargs["mask"] = None
        return real(*args, **kwargs)
    return patched


class TestRedProofs:
    def test_the_ops_identity_fails_when_the_layer_changes_the_bandwidth(self, monkeypatch):
        monkeypatch.setattr(
            mean_shift_module, "mean_shift_iterate",
            _defect_wider_bandwidth(mean_shift_module.mean_shift_iterate),
        )
        with pytest.raises(AssertionError):
            _check_matches_the_ops("gaussian", True)

    def test_the_oracle_identity_fails_when_the_layer_changes_the_bandwidth(self, monkeypatch):
        monkeypatch.setattr(
            mean_shift_module, "mean_shift_iterate",
            _defect_wider_bandwidth(mean_shift_module.mean_shift_iterate),
        )
        with pytest.raises(AssertionError):
            _check_matches_the_oracle("gaussian", False, 1, atol=1e-6)

    def test_the_mask_check_fails_when_the_layer_drops_the_mask(self, monkeypatch):
        monkeypatch.setattr(
            mean_shift_module, "mean_shift_iterate",
            _defect_ignore_the_mask(mean_shift_module.mean_shift_iterate),
        )
        with pytest.raises(AssertionError):
            _check_masked_points_do_not_act_as_keys(blurring=False, num_steps=1, exact=True)


# ---------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------

def _check_masked_points_do_not_act_as_keys(blurring, num_steps, exact, kernel="gaussian"):
    """Perturbing the masked points leaves every kept point's output unchanged."""
    layer = _make(kernel=kernel, blurring=blurring, num_steps=num_steps)
    x = _points(2, 6, 8)
    mask = np.array([[1, 1, 1, 1, 0, 0], [1] * 6], dtype="float32")
    perturbed = x.copy()
    perturbed[0, 4:] = _points(1, 2, 9, scale=5.0)[0]
    kept = mask.astype(bool)
    before = _np(layer(x, mask=mask))
    after = _np(layer(perturbed, mask=mask))
    if exact:
        assert np.array_equal(before[kept], after[kept])
    else:
        np.testing.assert_allclose(before[kept], after[kept], atol=1e-6, rtol=0)
    assert np.abs(before[0, 4:] - after[0, 4:]).max() > 1e-3


class TestMasks:
    def test_one_non_blurring_step_ignores_masked_points_exactly(self):
        _check_masked_points_do_not_act_as_keys(blurring=False, num_steps=1, exact=True)

    @pytest.mark.parametrize("kernel", ["gaussian", "vmf"])
    @pytest.mark.parametrize("blurring", [True, False])
    def test_several_steps_ignore_masked_points(self, kernel, blurring):
        _check_masked_points_do_not_act_as_keys(
            blurring=blurring, num_steps=4, exact=False, kernel=kernel
        )

    def test_a_mask_equal_to_all_ones_changes_nothing(self):
        layer = _make(num_steps=3)
        x = _points(2, 6, 10)
        np.testing.assert_allclose(
            _np(layer(x, mask=np.ones((2, 6), "float32"))), _np(layer(x)),
            atol=1e-6, rtol=0,
        )

    def test_a_boolean_mask_equals_a_float_mask(self):
        layer = _make(num_steps=3)
        x = _points(2, 6, 11)
        keep = np.array([[1, 1, 1, 0, 0, 1], [1] * 6], dtype=bool)
        np.testing.assert_array_equal(
            _np(layer(x, mask=keep)), _np(layer(x, mask=keep.astype("float32")))
        )

    @pytest.mark.parametrize("kernel", ["gaussian", "vmf"])
    @pytest.mark.parametrize("blurring", [True, False])
    def test_a_fully_masked_batch_row_stays_finite(self, kernel, blurring):
        layer = _make(kernel=kernel, blurring=blurring, num_steps=3, return_affinity=True)
        x = _points(2, 6, 12)
        mask = np.array([[0] * 6, [1] * 6], dtype="float32")
        shifted, affinity = layer(x, mask=mask)
        assert np.isfinite(_np(shifted)).all() and np.isfinite(_np(affinity)).all()
        np.testing.assert_allclose(_np(affinity).sum(axis=-1), 1.0, atol=1e-6)

    def test_a_fully_masked_row_returns_the_plain_mean_after_one_non_blurring_step(self):
        layer = _make(num_steps=1, blurring=False)
        x = _points(2, 6, 13)
        mask = np.array([[0] * 6, [1] * 6], dtype="float32")
        out = _np(layer(x, mask=mask))
        np.testing.assert_allclose(
            out[0], np.broadcast_to(x[0].mean(axis=0), (6, DIM)), atol=1e-6, rtol=0
        )

    def test_the_fully_masked_row_does_not_disturb_its_neighbour(self):
        layer = _make(num_steps=3)
        x = _points(2, 6, 14)
        full = _np(layer(x))
        masked = _np(layer(x, mask=np.array([[0] * 6, [1] * 6], dtype="float32")))
        np.testing.assert_allclose(masked[1], full[1], atol=1e-6, rtol=0)


# ---------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------

_FULL = dict(
    num_steps=4, bandwidth=0.6, learnable_bandwidth=True, blurring=False, step_size=0.9,
    kernel="vmf", use_projection=True, projection_dim=3, return_affinity=True,
)


class TestSerialization:
    def test_the_config_carries_every_constructor_argument(self):
        names = set(inspect.signature(MeanShiftLayer.__init__).parameters) - {"self", "kwargs"}
        assert names <= set(MeanShiftLayer(**_FULL).get_config())

    def test_the_config_round_trips_on_values(self):
        layer = MeanShiftLayer(**_FULL, name="ms")
        config = layer.get_config()
        json.dumps(config)
        clone = MeanShiftLayer.from_config(config)
        for key, value in _FULL.items():
            assert getattr(clone, key) == value
        assert clone.get_config() == config
        assert clone.name == "ms"

    def test_the_serializer_round_trip(self):
        layer = MeanShiftLayer(**_FULL)
        clone = keras.layers.deserialize(keras.layers.serialize(layer))
        assert isinstance(clone, MeanShiftLayer) and clone.get_config() == layer.get_config()

    @pytest.mark.parametrize("kwargs", [
        {},
        {"use_projection": True, "projection_dim": 3, "blurring": False},
        {"kernel": "vmf", "return_affinity": True, "learnable_bandwidth": False},
    ])
    def test_the_layer_saves_and_loads_as_a_keras_file(self, kwargs, tmp_path):
        inputs = keras.Input((None, DIM))
        outputs = MeanShiftLayer(num_steps=3, bandwidth=0.8, **kwargs)(inputs)
        model = keras.Model(inputs, outputs)
        layer = model.layers[-1]
        if layer.log_bandwidth is not None:
            layer.log_bandwidth.assign(np.float32(-0.3))
        x = _points(2, 5, 15)
        before = model(x, training=False)
        path = tmp_path / "mean_shift.keras"
        model.save(path)
        loaded = keras.models.load_model(path)
        after = loaded(x, training=False)
        before = before if isinstance(before, (list, tuple)) else [before]
        after = after if isinstance(after, (list, tuple)) else [after]
        for old, new in zip(before, after):
            np.testing.assert_allclose(_np(new), _np(old), atol=1e-6, rtol=0)
        assert len(loaded.weights) == len(model.weights)
        loaded_layer = loaded.layers[-1]
        if layer.log_bandwidth is not None:
            np.testing.assert_allclose(_np(loaded_layer.log_bandwidth), -0.3, atol=1e-7)


# ---------------------------------------------------------------------
# Precision and graph mode
# ---------------------------------------------------------------------

class TestPrecision:
    @pytest.mark.parametrize("kernel", ["gaussian", "vmf"])
    @pytest.mark.parametrize("projection", [False, True])
    def test_mixed_float16_stays_finite_and_close_to_float32(self, kernel, projection):
        extra = {"use_projection": True, "projection_dim": 3} if projection else {}
        x = _points(2, 6, 16)
        mask = np.array([[1, 1, 1, 1, 0, 0], [1] * 6], dtype="float32")
        control = _make(kernel=kernel, num_steps=3, return_affinity=True, **extra)
        with _policy("mixed_float16"):
            mixed = _make(kernel=kernel, num_steps=3, return_affinity=True, **extra)
            assert keras.backend.standardize_dtype(mixed.log_bandwidth.dtype) == "float32"
            if projection:
                mixed.projection.kernel.assign(_np(control.projection.kernel))
            shifted, affinity = mixed(x, mask=mask)
        controls = control(x, mask=mask)
        assert keras.backend.standardize_dtype(shifted.dtype) == "float16"
        assert keras.backend.standardize_dtype(affinity.dtype) == "float16"
        got = _np(shifted).astype("float32")
        assert np.isfinite(got).all() and np.isfinite(_np(affinity).astype("float32")).all()
        scale = np.abs(_np(controls[0])).max()
        error = np.abs(got - _np(controls[0])).max()
        # Measured max error without the projection (reproducible, two runs): 1.3e-3
        # (gaussian) and 2.3e-4 (vmf), on output scales 2.01 and 0.85. With the projection
        # the kernel is drawn unseeded: 0.5e-3 to 1.7e-3 (gaussian, scales 0.7 to 2.9) and
        # 2.3e-4 to 5.5e-4 (vmf, scales 0.74 to 1.0) over 10 draws. The bound is 0.5 percent
        # of the output scale (float16 input and output rounding).
        assert error <= 5e-3 * scale, f"max error {error:.3e} on scale {scale:.3f}"
        np.testing.assert_allclose(
            _np(affinity).astype("float32").sum(axis=-1), 1.0, atol=5e-3
        )

    def test_mixed_float16_does_not_overflow_on_a_large_scale_input(self):
        # exp of a squared distance beyond ~362 overflows float16; the kernel math runs
        # in float32, so a point set of norm ~40 stays finite.
        with _policy("mixed_float16"):
            layer = _make(bandwidth=1.0, num_steps=2)
            out = layer(_points(2, 6, 17, scale=30.0))
        assert np.isfinite(_np(out).astype("float32")).all()

    @pytest.mark.parametrize("policy", ["float32", "mixed_float16"])
    def test_one_concrete_function_serves_two_lengths(self, policy):
        with _policy(policy):
            layer = _make(num_steps=2, return_affinity=True)

            @tf.function
            def run(x, mask):
                return layer(x, mask=mask)

            concrete = run.get_concrete_function(
                tf.TensorSpec([None, None, DIM], tf.float32),
                tf.TensorSpec([None, None], tf.float32),
            )
            atol = 1e-5 if policy == "float32" else 1e-3
            for count in (5, 9):
                x = _points(2, count, 18)
                mask = np.ones((2, count), dtype="float32")
                mask[0, count - 2:] = 0.0
                shifted, affinity = concrete(tf.constant(x), tf.constant(mask))
                eager_shifted, eager_affinity = layer(x, mask=mask)
                assert tuple(shifted.shape) == (2, count, DIM)
                assert tuple(affinity.shape) == (2, count, count)
                np.testing.assert_allclose(
                    _np(shifted).astype("float32"),
                    _np(eager_shifted).astype("float32"), atol=atol, rtol=0,
                )

    def test_a_projection_layer_serves_two_lengths_from_one_function(self):
        layer = _make(use_projection=True, projection_dim=3, num_steps=2)

        @tf.function
        def run(x):
            return layer(x)

        concrete = run.get_concrete_function(tf.TensorSpec([None, None, DIM], tf.float32))
        for count in (4, 11):
            x = _points(2, count, 19)
            np.testing.assert_allclose(
                _np(concrete(tf.constant(x))), _np(layer(x)), atol=1e-5, rtol=0
            )


# ---------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------

class TestGradients:
    @pytest.mark.parametrize("kernel", ["gaussian", "vmf"])
    @pytest.mark.parametrize("blurring", [True, False])
    def test_gradients_are_finite_for_the_inputs_and_the_bandwidth(self, kernel, blurring):
        layer = _make(
            kernel=kernel, blurring=blurring, num_steps=3,
            use_projection=True, projection_dim=3,
        )
        x = tf.constant(_points(2, 6, 20))
        coefficients = tf.constant(_points(2, 6, 21, dim=3))
        mask = tf.constant([[1, 1, 1, 1, 0, 0], [1] * 6], dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x)
            loss = tf.reduce_sum(layer(x, mask=mask) * coefficients)
        grads = tape.gradient(loss, [x, layer.log_bandwidth, layer.projection.kernel])
        for grad in grads:
            assert grad is not None and np.isfinite(_np(grad)).all()
        assert abs(float(_np(grads[1]))) > 1e-6
        assert np.abs(_np(grads[0])).max() > 1e-6

    def test_a_fixed_bandwidth_gets_no_trainable_variable(self):
        layer = _make(learnable_bandwidth=False)
        assert layer.trainable_variables == []

    @pytest.mark.parametrize("bandwidth", [0.011, 0.1, 1.0, 10.0, 99.0])
    def test_the_bandwidth_gradient_is_nonzero_across_the_accepted_range(self, bandwidth):
        layer = _make(bandwidth=bandwidth, num_steps=3)
        x = tf.constant(_points(2, 8, 25))
        coefficients = tf.constant(_points(2, 8, 26))
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(layer(x) * coefficients)
        grad = float(_np(tape.gradient(loss, layer.log_bandwidth)))
        assert np.isfinite(grad) and abs(grad) > 1e-6

    def test_the_bandwidth_gradient_is_zero_outside_the_clip_range(self):
        layer = _make(num_steps=2)
        layer.log_bandwidth.assign(np.float32(50.0))
        x = tf.constant(_points(2, 6, 22))
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(layer(x) ** 2)
        grad = tape.gradient(loss, layer.log_bandwidth)
        assert float(_np(grad)) == 0.0

    def test_a_model_fit_step_moves_the_bandwidth(self):
        inputs = keras.Input((None, DIM))
        layer = MeanShiftLayer(num_steps=3, bandwidth=1.0)
        model = keras.Model(inputs, layer(inputs))
        model.compile(optimizer=keras.optimizers.SGD(0.1), loss="mse")
        x = _points(4, 8, 23)
        target = _points(4, 8, 24, scale=0.2)
        before = float(_np(layer.log_bandwidth))
        model.fit(x, target, epochs=1, batch_size=4, verbose=0)
        assert float(_np(layer.log_bandwidth)) != before


# ---------------------------------------------------------------------
# Liveness
# ---------------------------------------------------------------------

class TestLiveness:
    @pytest.mark.parametrize("blurring", [True, False])
    def test_three_steps_differ_from_one(self, blurring):
        x = _points(2, 8, 25)
        one = _make(num_steps=1, blurring=blurring)
        three = _make(num_steps=3, blurring=blurring)
        assert np.abs(_np(three(x)) - _np(one(x))).max() > 1e-3

    def test_a_small_bandwidth_collapses_nothing(self):
        x = _points(2, 8, 26)
        layer = _make(bandwidth=1.1e-2, learnable_bandwidth=False, num_steps=5)
        np.testing.assert_allclose(_np(layer(x)), x, atol=1e-5, rtol=0)

    def test_a_large_bandwidth_collapses_to_the_mean(self):
        x = _points(2, 8, 27)
        layer = _make(bandwidth=99.0, learnable_bandwidth=False, num_steps=5, blurring=True)
        out = _np(layer(x))
        mean = np.broadcast_to(x.mean(axis=1, keepdims=True), x.shape)
        np.testing.assert_allclose(out, mean, atol=1e-3, rtol=0)
        assert np.abs(x - mean).max() > 0.5

    def test_blurring_contracts_the_set_more_than_not_blurring(self):
        x = _points(2, 10, 28)
        spread = lambda a: np.var(a, axis=1).sum()
        blur = _np(_make(num_steps=4, blurring=True, bandwidth=1.5)(x))
        fixed = _np(_make(num_steps=4, blurring=False, bandwidth=1.5)(x))
        assert spread(blur) < spread(fixed) < spread(x)

    def test_the_projection_changes_the_result(self):
        x = _points(2, 6, 29)
        layer = _make(use_projection=True, projection_dim=DIM, num_steps=2)
        plain = _make(num_steps=2)
        assert np.abs(_np(layer(x)) - _np(plain(x))).max() > 1e-3


# ---------------------------------------------------------------------
# Factory registration
# ---------------------------------------------------------------------

class TestFactory:
    def test_it_builds_with_defaults_and_no_required_parameter(self):
        layer = create_mixture_layer("mean_shift")
        assert isinstance(layer, MeanShiftLayer)
        assert MIXTURE_REGISTRY["mean_shift"]["required_params"] == []
        assert tuple(layer(_points(2, 5, 30)).shape) == (2, 5, DIM)

    def test_the_kwargs_and_the_name_reach_the_layer(self):
        layer = create_mixture_layer(
            "mean_shift", name="ms", num_steps=2, kernel="vmf", use_projection=True,
            projection_dim=3, learnable_bandwidth=False,
        )
        assert layer.name == "ms" and layer.num_steps == 2 and layer.kernel == "vmf"
        assert layer.projection_dim == 3 and layer.learnable_bandwidth is False

    def test_a_base_layer_kwarg_is_forwarded(self):
        layer = create_mixture_layer("mean_shift", dtype="float64")
        assert layer.dtype_policy.name == "float64"

    def test_the_registry_defaults_equal_the_constructor_defaults(self):
        signature = inspect.signature(MeanShiftLayer.__init__)
        declared = MIXTURE_REGISTRY["mean_shift"]["optional_params"]
        names = set(signature.parameters) - {"self", "kwargs"}
        assert set(declared) == names
        for name in names:
            assert declared[name] == signature.parameters[name].default, name

    def test_the_mixture_type_literal_and_the_registry_agree(self):
        assert "mean_shift" in get_args(MixtureType)
        assert set(get_args(MixtureType)) == set(MIXTURE_REGISTRY)

    def test_get_mixture_info_lists_it_with_the_class_by_identity(self):
        info = get_mixture_info()
        assert info["mean_shift"]["class"] is MeanShiftLayer
        assert info["mean_shift"]["optional_params"]["num_steps"] == 5

    def test_an_unknown_kwarg_warns_and_is_dropped_not_raised(self, caplog):
        with caplog.at_level(logging.WARNING, logger="dl"):
            layer = create_mixture_layer("mean_shift", num_step=9)
        assert isinstance(layer, MeanShiftLayer) and layer.num_steps == 5
        dropped = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and "num_step" in r.message
        ]
        assert len(dropped) == 1, [r.message for r in caplog.records]

    @pytest.mark.parametrize("kwargs,pattern", [
        ({"num_steps": 0}, "num_steps must be a positive integer, got 0"),
        ({"num_steps": True}, "num_steps must be a positive integer, got True"),
        ({"num_steps": 2.5}, "num_steps must be a positive integer, got 2.5"),
        ({"projection_dim": -4}, "projection_dim must be a positive integer, got -4"),
        ({"bandwidth": 0}, "bandwidth must be positive, got 0"),
        ({"bandwidth": -1.0}, "bandwidth must be positive, got -1.0"),
        ({"step_size": 0.0}, "step_size must be positive, got 0.0"),
    ])
    def test_validate_mixture_config_rejects_a_bad_value(self, kwargs, pattern):
        with pytest.raises(ValueError, match=pattern):
            validate_mixture_config("mean_shift", **kwargs)

    def test_validate_mixture_config_accepts_the_defaults(self):
        validate_mixture_config("mean_shift", **MIXTURE_REGISTRY["mean_shift"]["optional_params"])

    @pytest.mark.parametrize("mode", ["assignments", "mixture", "basis", "normalized", "x"])
    def test_every_output_mode_is_refused_because_the_legal_set_is_empty(self, mode):
        with pytest.raises(ValueError, match=r"output_mode must be one of \[\]"):
            validate_mixture_config("mean_shift", output_mode=mode)
        with pytest.raises(ValueError, match="output_mode must be one of"):
            create_mixture_layer("mean_shift", output_mode=mode)

    def test_a_constructor_failure_is_rewrapped_as_a_factory_value_error(self):
        with pytest.raises(ValueError, match="Failed to create mean_shift"):
            create_mixture_layer("mean_shift", kernel="laplace")


# ---------------------------------------------------------------------
# assign_clusters
# ---------------------------------------------------------------------

_TWO_CLUSTERS = np.array(
    [[0.0, 0.0], [5.0, 5.0], [0.1, 0.0], [5.1, 5.0], [0.05, 0.05]], dtype="float32"
)


class TestAssignClusters:
    def test_a_hand_built_two_cluster_set(self):
        labels = assign_clusters(_TWO_CLUSTERS, merge_radius=0.5)
        np.testing.assert_array_equal(labels, [0, 1, 0, 1, 0])
        assert labels.dtype == np.int64 and labels.shape == (5,)

    def test_a_batched_input_is_labelled_per_sample(self):
        batch = np.stack([_TWO_CLUSTERS, _TWO_CLUSTERS[::-1]])
        labels = assign_clusters(batch, merge_radius=0.5)
        assert labels.shape == (2, 5)
        np.testing.assert_array_equal(labels[0], [0, 1, 0, 1, 0])
        np.testing.assert_array_equal(labels[1], [0, 1, 0, 1, 0])

    def test_a_large_radius_gives_one_cluster_and_a_tiny_one_gives_one_per_point(self):
        np.testing.assert_array_equal(assign_clusters(_TWO_CLUSTERS, 100.0), [0] * 5)
        np.testing.assert_array_equal(assign_clusters(_TWO_CLUSTERS, 1e-6), np.arange(5))

    def test_the_cluster_center_is_the_point_that_opened_it(self):
        line = np.array([[0.0], [0.4], [0.8], [1.2]], dtype="float32")
        # 0.8 is 0.8 from the opener at 0.0 (more than 0.5), so it opens cluster 1;
        # 1.2 is then within 0.5 of 0.8's cluster and 1.2 from the first.
        np.testing.assert_array_equal(assign_clusters(line, 0.5), [0, 0, 1, 1])

    def test_a_point_exactly_at_the_radius_joins(self):
        pair = np.array([[0.0, 0.0], [3.0, 4.0]], dtype="float64")
        np.testing.assert_array_equal(assign_clusters(pair, 5.0), [0, 0])
        np.testing.assert_array_equal(assign_clusters(pair, 4.999), [0, 1])

    def test_the_first_matching_cluster_wins_over_a_nearer_later_one(self):
        points = np.array([[0.0], [1.0], [0.6]], dtype="float64")
        # 0.6 is within 0.7 of both centers; the first (0.0) wins although 1.0 is nearer.
        np.testing.assert_array_equal(assign_clusters(points, 0.7), [0, 1, 0])

    def test_a_tensor_input_is_accepted(self):
        labels = assign_clusters(tf.constant(_TWO_CLUSTERS), 0.5)
        np.testing.assert_array_equal(labels, [0, 1, 0, 1, 0])

    @pytest.mark.parametrize("radius", [0, 0.0, -1.0, float("nan"), float("inf"), True, "r", None])
    def test_a_bad_merge_radius_raises_value_error_naming_the_value(self, radius):
        with pytest.raises(ValueError, match="assign_clusters: merge_radius"):
            assign_clusters(_TWO_CLUSTERS, radius)

    @pytest.mark.parametrize("shape", [(5,), (2, 2, 3, 4)])
    def test_a_wrong_rank_raises(self, shape):
        with pytest.raises(ValueError, match=r"assign_clusters: shifted.*rank"):
            assign_clusters(np.zeros(shape), 1.0)

    def test_a_masked_outlier_gets_minus_one_and_does_not_open_a_cluster(self):
        points = np.array([[0.0], [10.0], [0.1], [0.2]], dtype="float32")
        np.testing.assert_array_equal(assign_clusters(points, 0.5), [0, 1, 0, 0])
        masked = assign_clusters(points, 0.5, mask=np.array([1, 0, 1, 1]))
        np.testing.assert_array_equal(masked, [0, -1, 0, 0])
        assert masked.dtype == np.int64

    @pytest.mark.parametrize("mask_kind", ["bool", "float", "tensor"])
    def test_every_mask_dtype_is_read_as_a_keep_mask(self, mask_kind):
        points = np.array([[0.0], [10.0], [0.1], [20.0]], dtype="float32")
        keep = np.array([True, False, True, False])
        mask = {"bool": keep, "float": keep.astype("float32"),
                "tensor": tf.constant(keep.astype("float32"))}[mask_kind]
        np.testing.assert_array_equal(assign_clusters(points, 0.5, mask=mask), [0, -1, 0, -1])

    @pytest.mark.parametrize("batched", [False, True])
    def test_the_unmasked_points_are_labelled_as_if_the_masked_ones_were_removed(self, batched):
        rng = np.random.default_rng(41)
        samples = 3 if batched else 1
        points = rng.normal(size=(samples, 30, 2)) * 2.0
        keep = rng.random(size=(samples, 30)) > 0.4
        keep[:, 0] = False
        keep[:, 1] = True
        if batched:
            labels = assign_clusters(points, 1.0, mask=keep)
        else:
            labels = assign_clusters(points[0], 1.0, mask=keep[0])[None]
        for row in range(samples):
            expected = assign_clusters(points[row][keep[row]], 1.0)
            np.testing.assert_array_equal(labels[row][keep[row]], expected)
            assert np.all(labels[row][~keep[row]] == -1)

    def test_a_sample_with_no_kept_point_is_all_minus_one(self):
        batch = np.stack([_TWO_CLUSTERS, _TWO_CLUSTERS])
        mask = np.array([[1, 1, 1, 1, 1], [0, 0, 0, 0, 0]])
        labels = assign_clusters(batch, 0.5, mask=mask)
        np.testing.assert_array_equal(labels[0], [0, 1, 0, 1, 0])
        np.testing.assert_array_equal(labels[1], [-1] * 5)

    def test_a_mask_of_ones_changes_nothing(self):
        np.testing.assert_array_equal(
            assign_clusters(_TWO_CLUSTERS, 0.5, mask=np.ones(5)),
            assign_clusters(_TWO_CLUSTERS, 0.5),
        )

    @pytest.mark.parametrize("shape,mask_shape", [
        ((5, 2), (4,)), ((5, 2), (1, 5)), ((2, 5, 2), (5,)), ((2, 5, 2), (2, 4)),
        ((2, 5, 2), (2, 5, 1)),
    ])
    def test_a_mask_whose_shape_does_not_match_the_leading_axes_raises(self, shape, mask_shape):
        with pytest.raises(ValueError, match=r"assign_clusters: mask must have shape"):
            assign_clusters(np.zeros(shape), 1.0, mask=np.ones(mask_shape))

    def test_it_reads_two_labels_off_the_shifted_output_of_two_blobs(self):
        rng = np.random.default_rng(31)
        blob_a = rng.normal(loc=-4.0, scale=0.3, size=(20, 2))
        blob_b = rng.normal(loc=4.0, scale=0.3, size=(20, 2))
        order = rng.permutation(40)
        x = np.concatenate([blob_a, blob_b])[order][None].astype("float32")
        truth = np.concatenate([np.zeros(20, int), np.ones(20, int)])[order]
        layer = _make(dim=2, num_steps=10, bandwidth=1.0, blurring=True)
        shifted = _np(layer(x))
        labels = assign_clusters(shifted, merge_radius=0.5)
        assert labels.shape == (1, 40) and set(labels[0]) == {0, 1}
        agree = (labels[0] == truth).all() or (labels[0] == 1 - truth).all()
        assert agree, "labels must split the two blobs exactly"
        raw = assign_clusters(x, merge_radius=0.05)
        assert len(set(raw[0])) > 10, "the unshifted points are not already merged"
