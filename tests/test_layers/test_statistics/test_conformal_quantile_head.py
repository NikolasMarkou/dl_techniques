"""Test suite for :class:`ConformalQuantileHead` (Conformalized Quantile Regression head).

Covers construction and registration, validation, the build layout, the forward pass
and its dtype/XLA arms, gradient flow per variable, the order-statistic calibration
(``calibrate``) including its small-``n`` ``+inf`` case, ``predict_intervals`` building
lazily and applying ``q_hat``, the ``.keras`` and config round trips of a CALIBRATED
head, and the initializer spec property (two heads from one seedless instance draw
different kernels).
"""

import json

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.statistics.conformal_quantile_head import ConformalQuantileHead

FL, OUT, DIM, B = 5, 3, 10, 4
CAL_Q = 0.37


def _x(n=B, seed=0, dtype="float32"):
    return np.random.default_rng(seed).standard_normal((n, DIM)).astype(dtype)


def _head(**kw):
    return ConformalQuantileHead(forecast_length=FL, output_dim=OUT, **kw)


def _np(t):
    return keras.ops.convert_to_numpy(t)


class TestConstruction:

    @pytest.fixture(autouse=True)
    def _seed(self):
        keras.utils.set_random_seed(1234)

    def test_registration_and_unbuilt_state(self, registration_contract):
        registration_contract(ConformalQuantileHead)
        head = _head()
        assert head.q_hat is None and head.projection is not None
        assert not head.projection.built  # built by the layer's build(), not before

    def test_build_materializes_exactly_the_projection_and_the_offset(self):
        head = _head(name="cqh")
        head(_x())
        assert head.projection.built
        paths = sorted(w.path for w in head.weights)
        assert paths == [
            "cqh/conformal_q",
            "cqh/quantile_projection/bias",
            "cqh/quantile_projection/kernel",
        ]
        assert [w.path.split("/")[-1] for w in head.non_trainable_weights] == ["conformal_q"]
        assert head.q_hat.shape == ()
        assert head.projection.kernel.shape == (DIM, FL * OUT * 3)

    @pytest.mark.parametrize("kwargs, message", [
        (dict(forecast_length=0, output_dim=2), "forecast_length must be positive, got 0"),
        (dict(forecast_length=2, output_dim=-1), "output_dim must be positive, got -1"),
        (dict(forecast_length=2, output_dim=2, q_hat_init=float("nan")), "q_hat_init must not be NaN"),
    ])
    def test_arguments_are_validated_naming_the_value(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            ConformalQuantileHead(**kwargs)

    @pytest.mark.parametrize("shape", [(None, 4, 3), (None, None)])
    def test_build_refuses_a_bad_input_shape(self, shape):
        with pytest.raises(ValueError, match="defined input_dim"):
            _head().build(shape)

    def test_two_heads_from_one_seedless_initializer_instance_draw_differently(self):
        init = keras.initializers.GlorotUniform()
        a, b = _head(kernel_initializer=init), _head(kernel_initializer=init)
        a(_x()), b(_x())
        assert not np.array_equal(_np(a.projection.kernel), _np(b.projection.kernel))


class TestForward:

    def test_shape_finite_and_unbuilt_compute_output_shape(self):
        head = _head()
        assert head.compute_output_shape((None, DIM)) == (None, FL, OUT, 3)
        out = _np(head(_x()))
        assert out.shape == (B, FL, OUT, 3) and np.isfinite(out).all()

    def test_call_never_applies_q_hat(self):
        head = _head()
        before = _np(head(_x()))
        head.calibrate(np.array([5.0] * 20), alpha=0.1)
        np.testing.assert_array_equal(_np(head(_x())), before)

    def test_dtype_arms_construct_and_run(self, dtype_policy):
        head = _head()
        out = head(_x())
        assert np.isfinite(_np(out).astype("float64")).all()
        assert head.q_hat.dtype == ("float64" if dtype_policy == "float64" else "float32")
        median, lower, upper = head.predict_intervals(_x())  # q_hat dtype != compute dtype
        assert np.isfinite(_np(lower).astype("float64")).all()
        assert lower.dtype == upper.dtype == median.dtype == out.dtype

    def test_xla_matches_eager(self):
        head = _head()
        x = _x()
        eager = _np(head(x))
        fn = tf.function(lambda t: head(t, training=False), jit_compile=True)
        np.testing.assert_allclose(_np(fn(tf.constant(x))), eager, rtol=1e-5, atol=1e-6)

    def test_every_trainable_variable_gets_a_nonzero_gradient_and_q_hat_is_frozen(self):
        head = _head()
        x = tf.constant(_x())
        head(x)
        assert len(head.trainable_variables) > 0
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(head(x, training=True) ** 2)
        grads = tape.gradient(loss, head.trainable_variables)
        for var, g in zip(head.trainable_variables, grads):
            assert g is not None, var.path
            assert float(tf.reduce_max(tf.abs(g))) > 0.0, var.path
        assert head.q_hat.trainable is False
        assert all(v is not head.q_hat for v in head.trainable_variables)


class TestCalibration:

    def test_q_hat_is_the_order_statistic_not_an_interpolation(self):
        # n = 9, alpha = 0.1: k = ceil(10 * 0.9) = 9, the LARGEST score. The previous
        # interpolated quantile of level 0.99 sat below it and under-covered.
        head = _head()
        head(_x())
        scores = np.array([0.1, 0.5, 0.2, 0.9, 0.3, 0.8, 0.4, 0.7, 0.6])
        head.calibrate(scores, alpha=0.1)
        assert float(_np(head.q_hat)) == pytest.approx(0.9, abs=1e-6)

    def test_too_few_scores_give_unbounded_intervals(self):
        head = _head()
        head(_x())
        head.calibrate(np.array([0.2, 0.4, 0.1]), alpha=0.1)  # k = 4 > n = 3
        assert float(_np(head.q_hat)) == np.inf
        _, lower, upper = head.predict_intervals(_x())
        assert np.isneginf(_np(lower)).all() and np.isposinf(_np(upper)).all()

    def test_calibrate_before_build_is_refused(self):
        with pytest.raises(ValueError, match="not built"):
            _head().calibrate(np.array([0.1]), alpha=0.1)

    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1])
    def test_alpha_outside_the_open_unit_interval_is_refused(self, alpha):
        head = _head()
        head(_x())
        with pytest.raises(ValueError, match="alpha must be in"):
            head.calibrate(np.array([0.1, 0.2]), alpha=alpha)

    def test_the_calibrated_interval_reaches_the_target_where_the_raw_one_does_not(self):
        # Raw quantile interval [-0.5, 0.5] for y ~ N(0, 1): covers ~38 %, far from 90 %.
        rng = np.random.default_rng(0)
        y_cal, y_test = rng.standard_normal(2000), rng.standard_normal(20000)
        lo, hi = -0.5, 0.5
        scores = np.maximum(lo - y_cal, y_cal - hi)
        head = _head()
        head(_x())
        head.calibrate(scores, alpha=0.1)
        q = float(_np(head.q_hat))
        raw = np.mean((y_test >= lo) & (y_test <= hi))
        cal = np.mean((y_test >= lo - q) & (y_test <= hi + q))
        assert raw < 0.5  # anti-vacuity
        assert cal >= 0.9 - 0.01

    def test_predict_intervals_widens_by_exactly_q_hat_and_keeps_the_median(self):
        head = _head()
        raw = _np(head(_x()))
        head.q_hat.assign(CAL_Q)
        median, lower, upper = map(_np, head.predict_intervals(_x()))
        np.testing.assert_array_equal(median, raw[..., 1])
        np.testing.assert_allclose(lower, raw[..., 0] - CAL_Q, rtol=0, atol=1e-6)
        np.testing.assert_allclose(upper, raw[..., 2] + CAL_Q, rtol=0, atol=1e-6)

    def test_predict_intervals_builds_an_unbuilt_head(self):
        head = _head()
        median, lower, upper = head.predict_intervals(_x())
        assert head.built and tuple(median.shape) == (B, FL, OUT)
        np.testing.assert_array_equal(_np(lower), _np(head(_x()))[..., 0])  # q_hat = 0


class TestSerialization:

    def _calibrated_model(self, q=CAL_Q):
        inp = keras.Input((DIM,))
        head = _head(name="cqh")
        model = keras.Model(inp, head(inp))
        head.q_hat.assign(q)
        return model, head

    @pytest.mark.parametrize("q", [CAL_Q, float("inf")], ids=["finite", "unbounded"])
    def test_keras_round_trip_on_values_with_a_calibrated_offset(self, tmp_path, q):
        model, head = self._calibrated_model(q)
        x = _x()
        pre = _np(model(x, training=False))
        path = tmp_path / "m.keras"
        model.save(path)
        loaded = keras.models.load_model(
            path, custom_objects={"ConformalQuantileHead": ConformalQuantileHead}
        )
        rel = next(l for l in loaded.layers if isinstance(l, ConformalQuantileHead))
        # weights first, before the loaded model's first call
        for a, b in zip(head.weights, rel.weights):
            assert a.path == b.path
            np.testing.assert_array_equal(_np(a), _np(b))
        assert float(_np(rel.q_hat)) == float(np.float32(q))
        np.testing.assert_array_equal(_np(loaded(x, training=False)), pre)
        for a, b in zip(head.predict_intervals(x), rel.predict_intervals(x)):
            np.testing.assert_array_equal(_np(a), _np(b))

    def test_config_round_trip_keeps_the_calibration_and_is_json_safe(self):
        head = _head(kernel_initializer=keras.initializers.HeNormal(),
                     kernel_regularizer=keras.regularizers.L2(1e-3))
        head(_x())
        head.q_hat.assign(CAL_Q)
        cfg = head.get_config()
        json.dumps(cfg)  # an instance would not serialize
        assert cfg["q_hat_init"] == pytest.approx(CAL_Q, abs=1e-6)
        assert isinstance(cfg["kernel_initializer"], dict)
        clone = ConformalQuantileHead.from_config(cfg)
        clone(_x())
        assert float(_np(clone.q_hat)) == pytest.approx(CAL_Q, abs=1e-6)
        assert clone.get_config() == cfg or clone.get_config().keys() == cfg.keys()
        assert clone.projection.kernel_regularizer is not None

    def test_every_constructor_argument_is_in_the_config(self):
        cfg = _head().get_config()
        for key in ("forecast_length", "output_dim", "kernel_initializer",
                    "kernel_regularizer", "q_hat_init"):
            assert key in cfg
