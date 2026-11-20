"""Tests for GeoNorm.

The oracle recomputes the update in float64 NumPy from the paper's equations. The
decay and clamp tests read the rotation angle off the output (``arccos`` of the
cosine between ``x`` and the result), because the released Appendix D listing it
comes from has a harmonic branch that assigns to the wrong variable and therefore
never decays: a test that only checks shapes and finiteness cannot see that.

RED record (research/2026_keras_custom_models_instructions_v2.md, house rule 2). Each mutation was
injected into ``geo_norm.py`` in turn, this file run, and the file restored. Measured 2026-11-20, CPU,
125 tests in the file, baseline 125 passed:

====================================================  ======  ===================================
injection                                             failed  first-line evidence
====================================================  ======  ===================================
no step decay (``theta = alpha * ratio``)              40     schedule, angle, clamp-decay tests
no tangent projection                                  66     perpendicular/parallel twin, oracle
drop the ``radius`` factor of the tangent term         66     homogeneity error 0.21 (c=0.5), 0.14 (c=3)
gradient blind to the projection (stop_gradient)        1     central differences: 8.2e-02 vs 1.4e-11
statistics in the input dtype (float16 overflow)        1     half-precision statistics test
clamp ignored                                          29     clamp, default-clamp, oracle tests
no static contract re-assertion in ``call``             1     mismatched-pair test
====================================================  ======  ===================================

The half-precision statistics test was first written without the ``mixed_float16`` policy, passed with
the statistics narrowed to float16 (Keras autocasts the inputs to float32 before ``call`` under the
default policy), and was repaired after that injection exposed it.
"""

import inspect
import math
import os
import tempfile

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.norms import GeoNorm

# ---------------------------------------------------------------------

B, L, D = 3, 5, 16

# Tolerance of the float64 NumPy oracle, relative to max(1, output absmax). Measured 2026-11-20, CPU,
# float32, schedule harmonic, layer_idx 2: max|layer - oracle| / max(1, absmax) = 5.2e-08. Defects it
# must catch: alpha 1.01 moves the output by 2.0e-03, a layer_idx off by one by 5.0e-02. 1e-5 is
# 190x above the floor and 200x below the smallest defect.
ORACLE_RTOL = 1e-5


def _pair(seed=0, scale_s=1.0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(B, L, D)).astype("float32")
    s = (scale_s * rng.normal(size=(B, L, D))).astype("float32")
    return x, s


def _run(layer, x, s, **kwargs):
    return keras.ops.convert_to_numpy(layer([x, s], **kwargs)).astype("float64")


def _decay(schedule, k, total=None):
    return {"harmonic": 1.0 / (k + 1), "sqrt": 1.0 / math.sqrt(k + 1),
            "linear": None if total is None else (total - k) / total}[schedule]


def _oracle(x, s, schedule="harmonic", alpha=1.0, clamp=math.pi / 4, k=0, total=None):
    x = x.astype("float64")
    s = s.astype("float64")
    radius = np.linalg.norm(x, axis=-1, keepdims=True)
    v = s - (x * s).sum(-1, keepdims=True) / radius ** 2 * x
    vnorm = np.linalg.norm(v, axis=-1, keepdims=True)
    ratio = vnorm / radius
    if clamp is not None:
        ratio = np.minimum(ratio, clamp)
    theta = alpha * _decay(schedule, k, total) * ratio
    return np.cos(theta) * x + radius * np.sin(theta) * v / vnorm


def _angle(x, y):
    x = x.astype("float64")
    cos = (x * y).sum(-1) / (np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1))
    return np.arccos(np.clip(cos, -1.0, 1.0))


# ---------------------------------------------------------------------

class TestConstruction:

    def test_defaults(self):
        layer = GeoNorm()
        assert (layer.schedule, layer.alpha, layer.total_layers) == ("harmonic", 1.0, None)
        assert layer.clamp == pytest.approx(math.pi / 4)

    @pytest.mark.parametrize("bad", [
        dict(schedule="cosine"), dict(schedule="harnomic"),
        dict(alpha=0.0), dict(alpha=-1.0),
        dict(clamp=0.0), dict(clamp=-0.5),
        dict(epsilon=0.0),
        dict(total_layers=0), dict(total_layers=-3), dict(total_layers=2.5), dict(total_layers=True),
        dict(schedule="linear"),
    ])
    def test_invalid_configuration_raises(self, bad):
        with pytest.raises(ValueError):
            GeoNorm(**bad)

    def test_clamp_can_be_switched_off(self):
        assert GeoNorm(clamp=None).clamp is None

    def test_linear_with_total_layers_is_valid(self):
        assert GeoNorm(schedule="linear", total_layers=12).total_layers == 12

    def test_has_no_weights(self):
        layer = GeoNorm()
        layer(list(_pair()))
        assert layer.weights == [] and layer.count_params() == 0


class TestBuild:

    @pytest.mark.parametrize("shapes", [
        (B, L, D), [(B, L, D)], [(B, L, D), (B, L, D), (B, L, D)],
        [(B, L, D), (B, L)], [(B, L, D), (B, L, D + 1)],
    ])
    def test_malformed_input_shapes_raise(self, shapes):
        with pytest.raises(ValueError):
            GeoNorm().build(shapes)

    def test_dynamic_dimensions_are_accepted(self):
        GeoNorm().build([(None, None, D), (None, None, D)])

    def test_compute_output_shape_is_the_state_shape(self):
        assert GeoNorm().compute_output_shape([(None, 7, D), (None, 7, D)]) == (None, 7, D)


# ---------------------------------------------------------------------

class TestAgainstTheOracle:

    @pytest.mark.parametrize("schedule,total", [("harmonic", None), ("sqrt", None), ("linear", 6)])
    @pytest.mark.parametrize("k", [0, 1, 4])
    @pytest.mark.parametrize("scale_s", [0.05, 1.0, 40.0])
    def test_matches_the_equations(self, schedule, total, k, scale_s):
        x, s = _pair(scale_s=scale_s)
        layer = GeoNorm(schedule=schedule, total_layers=total)
        got = _run(layer, x, s, layer_idx=k)
        want = _oracle(x, s, schedule=schedule, k=k, total=total)
        np.testing.assert_allclose(got, want, rtol=0, atol=ORACLE_RTOL * max(1.0, np.abs(want).max()))

    @pytest.mark.parametrize("clamp", [None, 0.1, math.pi / 2])
    @pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
    def test_matches_the_equations_for_clamp_and_alpha(self, clamp, alpha):
        x, s = _pair(scale_s=10.0)
        got = _run(GeoNorm(alpha=alpha, clamp=clamp), x, s, layer_idx=2)
        want = _oracle(x, s, alpha=alpha, clamp=clamp, k=2)
        np.testing.assert_allclose(got, want, rtol=0, atol=ORACLE_RTOL * max(1.0, np.abs(want).max()))


class TestGeometry:

    @pytest.mark.parametrize("clamp", [None, math.pi / 4])
    @pytest.mark.parametrize("k", [0, 3])
    def test_the_state_stays_on_its_sphere(self, clamp, k):
        x, s = _pair(scale_s=5.0)
        y = _run(GeoNorm(clamp=clamp), x, s, layer_idx=k)
        np.testing.assert_allclose(
            np.linalg.norm(y, axis=-1), np.linalg.norm(x.astype("float64"), axis=-1), rtol=2e-6, atol=0)

    def test_a_step_parallel_to_the_state_does_not_move_it(self):
        x, _ = _pair()
        y = _run(GeoNorm(), x, 3.7 * x)
        np.testing.assert_allclose(y, x, rtol=0, atol=2e-5)

    def test_a_zero_update_does_not_move_it(self):
        x, _ = _pair()
        y = _run(GeoNorm(), x, np.zeros_like(x))
        np.testing.assert_allclose(y, x, rtol=0, atol=1e-6)

    def test_a_small_step_follows_the_tangent(self):
        """Without a clamp, a tiny update moves x by alpha * decay * v, to second order."""
        x, s = _pair(scale_s=1e-3)
        y = _run(GeoNorm(clamp=None), x, s, layer_idx=0)
        radius_sq = (x.astype("float64") ** 2).sum(-1, keepdims=True)
        v = s - (x * s).sum(-1, keepdims=True) / radius_sq * x
        np.testing.assert_allclose(y - x, v, rtol=0, atol=1e-5)


class TestStepDecay:
    """Read the angle off the output; a schedule that is wired to nothing leaves it unchanged."""

    @pytest.mark.parametrize("schedule,total,expected", [
        ("harmonic", None, lambda k: 1.0 / (k + 1)),
        ("sqrt", None, lambda k: 1.0 / math.sqrt(k + 1)),
        ("linear", 8, lambda k: (8 - k) / 8),
    ])
    @pytest.mark.parametrize("k", [0, 1, 3, 7])
    def test_the_angle_follows_the_schedule(self, schedule, total, expected, k):
        x, s = _pair(scale_s=0.2)
        layer = GeoNorm(schedule=schedule, total_layers=total, clamp=None)
        radius = np.linalg.norm(x.astype("float64"), axis=-1)
        radius_sq = radius ** 2
        v = s - (x * s).sum(-1, keepdims=True) / radius_sq[..., None] * x
        ratio = np.linalg.norm(v, axis=-1) / radius
        angle = _angle(x, _run(layer, x, s, layer_idx=k))
        np.testing.assert_allclose(angle, expected(k) * ratio, rtol=1e-3, atol=1e-5)

    def test_a_deeper_block_takes_a_smaller_step(self):
        x, s = _pair(scale_s=0.2)
        layer = GeoNorm(clamp=None)
        angles = [_angle(x, _run(layer, x, s, layer_idx=k)).mean() for k in range(5)]
        assert all(a > b for a, b in zip(angles, angles[1:]))

    def test_alpha_scales_the_angle(self):
        x, s = _pair(scale_s=0.1)
        one = _angle(x, _run(GeoNorm(alpha=1.0, clamp=None), x, s)).mean()
        half = _angle(x, _run(GeoNorm(alpha=0.5, clamp=None), x, s)).mean()
        assert half == pytest.approx(0.5 * one, rel=1e-3)


class TestClamp:

    @pytest.mark.parametrize("k", [0, 2])
    def test_the_default_clamp_caps_the_angle_before_the_decay(self, k):
        x, s = _pair(scale_s=1e4)
        angle = _angle(x, _run(GeoNorm(), x, s, layer_idx=k))
        np.testing.assert_allclose(angle, (math.pi / 4) / (k + 1), rtol=1e-4)

    def test_a_custom_clamp(self):
        x, s = _pair(scale_s=1e4)
        angle = _angle(x, _run(GeoNorm(clamp=0.3), x, s))
        np.testing.assert_allclose(angle, 0.3, rtol=1e-4)

    def test_clamp_none_lets_a_huge_step_rotate_further(self):
        x, s = _pair(scale_s=1e4)
        clamped = _angle(x, _run(GeoNorm(), x, s)).mean()
        free = _angle(x, _run(GeoNorm(clamp=None), x, s)).mean()
        assert clamped == pytest.approx(math.pi / 4, rel=1e-4)
        assert free > clamped + 0.1

    def test_a_small_step_is_untouched_by_the_clamp(self):
        x, s = _pair(scale_s=0.01)
        np.testing.assert_allclose(
            _run(GeoNorm(), x, s), _run(GeoNorm(clamp=None), x, s), rtol=0, atol=1e-7)


# ---------------------------------------------------------------------

class TestRobustness:

    def test_zero_state_and_zero_update_are_finite(self):
        z = np.zeros((B, L, D), "float32")
        x, s = _pair()
        for a, b in ((z, z), (z, s), (x, z)):
            assert np.isfinite(_run(GeoNorm(), a, b)).all()

    @pytest.mark.parametrize("which", ["both_zero", "state_zero", "update_zero", "generic"])
    def test_gradients_are_finite(self, which):
        x, s = _pair()
        z = np.zeros_like(x)
        a, b = {"both_zero": (z, z), "state_zero": (z, s), "update_zero": (x, z), "generic": (x, s)}[which]
        a, b = tf.constant(a), tf.constant(b)
        layer = GeoNorm()
        with tf.GradientTape() as tape:
            tape.watch([a, b])
            loss = tf.reduce_sum(layer([a, b], layer_idx=1) ** 2 * tf.range(1.0, D + 1.0))
        grads = tape.gradient(loss, [a, b])
        assert all(g is not None and np.isfinite(g.numpy()).all() for g in grads)

    def test_a_huge_magnitude_stays_finite(self):
        x, s = _pair(scale_s=1e18)
        assert np.isfinite(_run(GeoNorm(), x * 1e10, s)).all()

    def test_a_masked_position_is_not_special(self):
        x, s = _pair()
        full = _run(GeoNorm(), x, s)
        np.testing.assert_allclose(_run(GeoNorm(), x[:, :2], s[:, :2]), full[:, :2], rtol=0, atol=1e-6)

    def test_layer_idx_may_be_a_tensor(self):
        x, s = _pair()
        a = _run(GeoNorm(), x, s, layer_idx=3)
        b = _run(GeoNorm(), x, s, layer_idx=tf.constant(3))
        np.testing.assert_allclose(a, b, rtol=0, atol=0)

    def test_mixed_precision_is_finite_and_keeps_the_dtype(self, dtype_policy):
        x, s = _pair(scale_s=3.0)
        y = GeoNorm()([tf.constant(x, tf.as_dtype(
            {"float32": "float32", "mixed_float16": "float16", "float64": "float64"}[dtype_policy])),
            tf.constant(s, tf.as_dtype(
                {"float32": "float32", "mixed_float16": "float16", "float64": "float64"}[dtype_policy]))])
        assert keras.backend.standardize_dtype(y.dtype) == {
            "float32": "float32", "mixed_float16": "float16", "float64": "float64"}[dtype_policy]
        y = keras.ops.convert_to_numpy(y).astype("float64")
        assert np.isfinite(y).all()
        tolerance = {"float32": 2e-6, "mixed_float16": 2e-3, "float64": 1e-12}[dtype_policy]
        np.testing.assert_allclose(
            np.linalg.norm(y, axis=-1), np.linalg.norm(x.astype("float64"), axis=-1), rtol=tolerance * 4, atol=0)

    def test_the_statistics_do_not_overflow_in_half_precision(self, mixed_float16_policy):
        """A float16 sum of squares of norm-300 vectors is 9e4, past the float16 maximum of 65504.

        Needs the mixed_float16 policy: under the default float32 policy Keras autocasts the float16
        inputs to float32 before call(), so the hazard never reaches the layer (the first version of
        this test did exactly that and passed with the statistics narrowed to float16)."""
        x = (np.random.default_rng(0).normal(size=(2, 4, 64)) * 40.0).astype("float16")
        s = np.random.default_rng(1).normal(size=(2, 4, 64)).astype("float16")
        assert float((x.astype("float32") ** 2).sum(-1).max()) > float(np.finfo(np.float16).max)
        y = GeoNorm()([tf.constant(x), tf.constant(s)])
        assert keras.backend.standardize_dtype(y.dtype) == "float16"
        y = keras.ops.convert_to_numpy(y).astype("float32")
        assert np.isfinite(y).all()
        np.testing.assert_allclose(
            np.linalg.norm(y, axis=-1), np.linalg.norm(x.astype("float32"), axis=-1), rtol=2e-3, atol=0)


# ---------------------------------------------------------------------

class TestSerialization:

    def test_get_config_covers_every_constructor_argument(self):
        config = GeoNorm().get_config()
        params = {n for n in inspect.signature(GeoNorm.__init__).parameters if n not in ("self", "kwargs")}
        assert params <= set(config)

    def test_from_config_round_trip(self):
        layer = GeoNorm(schedule="linear", alpha=0.7, clamp=None, total_layers=9, epsilon=1e-5, name="geo")
        assert GeoNorm.from_config(layer.get_config()).get_config() == layer.get_config()

    def test_a_saved_model_reloads_and_reproduces_its_output(self):
        x_in, s_in = keras.Input((L, D)), keras.Input((L, D))
        model = keras.Model([x_in, s_in], GeoNorm(schedule="sqrt", name="geo")([x_in, s_in]))
        x, s = _pair()
        before = model.predict([x, s], verbose=0)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "geo.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
        assert loaded.get_layer("geo").schedule == "sqrt"
        np.testing.assert_allclose(loaded.predict([x, s], verbose=0), before, rtol=0, atol=0)

    def test_a_dynamic_sequence_length_in_a_functional_model(self):
        x_in, s_in = keras.Input((None, D)), keras.Input((None, D))
        model = keras.Model([x_in, s_in], GeoNorm()([x_in, s_in]))
        for length in (3, 11):
            x = np.ones((2, length, D), "float32")
            assert model.predict([x, x * 0.5], verbose=0).shape == (2, length, D)

    def test_a_training_step_runs_under_jit_compile(self):
        x_in, s_in = keras.Input((L, D)), keras.Input((L, D))
        dense = keras.layers.Dense(D)
        model = keras.Model([x_in, s_in], GeoNorm()([x_in, dense(s_in)]))
        model.compile(optimizer="adam", loss="mse", jit_compile=True)
        x, s = _pair()
        history = model.fit([x, s], x, epochs=1, batch_size=B, verbose=0)
        assert np.isfinite(history.history["loss"]).all()


# ---------------------------------------------------------------------
# Compliance with research/2026_keras_custom_models_instructions_v2.md, section 16.1 and 16.3.
# ---------------------------------------------------------------------

HOMOGENEITY_RTOL = 1e-5
HOMOGENEITY_SCALES = (0.5, 3.0)


def _homogeneity_error(layer, x, s, c):
    f = _run(layer, x, s)
    fc = _run(layer, (c * x).astype("float32"), (c * s).astype("float32"))
    denominator = float(np.abs(c * f).max())
    if denominator == 0.0:
        return float("inf")          # a dead layer must not read as perfectly homogeneous
    return float(np.abs(fc - c * f).max() / denominator)


class TestInvariants:
    """Section 10.3 and 13.2.9: pin the invariant, not the shape."""

    @pytest.mark.parametrize("c", HOMOGENEITY_SCALES)
    @pytest.mark.parametrize("clamp", [math.pi / 4, None])
    def test_the_update_is_jointly_homogeneous(self, c, clamp):
        """GeoNorm(c x, c s) = c GeoNorm(x, s): the angle depends on ||v|| / ||x|| only.

        Measured 2026-11-20, CPU, float32, this pair: 0.0 at c=0.5 and c=3 (1.6e-07 at c=3 for a
        unit-scale update). A tolerance of 1e-5 is 64x above the larger floor; dropping the ``radius``
        factor of the tangent term measures 0.21 and 0.14 (RED record at the top of this file)."""
        x, s = _pair(scale_s=3.0)
        error = _homogeneity_error(GeoNorm(clamp=clamp), x, s, c)
        assert error < HOMOGENEITY_RTOL, error

    def test_the_homogeneity_probe_rejects_a_dead_layer(self):
        class Dead(GeoNorm):
            def call(self, inputs, layer_idx=0, training=None):
                return inputs[0] * 0.0

        x, s = _pair()
        assert _homogeneity_error(Dead(), x, s, 3.0) == float("inf")

    def test_the_sign_of_the_update_is_honoured(self):
        """The two sides of the great circle are mirror images: out(x, s) + out(x, -s) = 2 cos(theta) x."""
        x, s = _pair(scale_s=0.5)
        layer = GeoNorm(clamp=None)
        plus, minus = _run(layer, x, s), _run(layer, x, -s)
        radius_sq = (x.astype("float64") ** 2).sum(-1, keepdims=True)
        v = s - (x * s).sum(-1, keepdims=True) / radius_sq * x
        theta = np.linalg.norm(v, axis=-1, keepdims=True) / np.sqrt(radius_sq)
        np.testing.assert_allclose(plus + minus, 2.0 * np.cos(theta) * x, rtol=0, atol=2e-6)
        assert np.abs(plus - minus).max() > 1e-2                   # the twin: the two sides differ

    def test_a_perpendicular_step_moves_where_a_parallel_one_does_not(self):
        """The twin of test_a_step_parallel_to_the_state_does_not_move_it."""
        x, s = _pair()
        moved = _run(GeoNorm(), x, s)
        assert np.abs(moved - x).max() > 1e-2
        assert np.abs(_run(GeoNorm(), x, 3.7 * x) - x).max() < 2e-5

    def test_the_closed_form_gradient_matches_central_differences(self):
        """float64 directional derivative against autodiff. Measured 2026-11-20, CPU: relative
        disagreement 1.4e-11 at h=1e-6; the bound 1e-6 sits 5 orders above that and 5 below the 8.2e-02
        of a gradient that ignored the tangent projection (RED record)."""
        rng = np.random.default_rng(3)
        x = tf.constant(rng.normal(size=(2, 3, 8)))
        s = tf.constant(rng.normal(size=(2, 3, 8)))
        direction_x = tf.constant(rng.normal(size=(2, 3, 8)))
        direction_s = tf.constant(rng.normal(size=(2, 3, 8)))
        weights = tf.constant(rng.normal(size=(2, 3, 8)))
        layer = GeoNorm(dtype="float64", clamp=None)

        def scalar(a, b):
            return tf.reduce_sum(weights * layer([a, b], layer_idx=1))

        with tf.GradientTape() as tape:
            tape.watch([x, s])
            value = scalar(x, s)
        gx, gs = tape.gradient(value, [x, s])
        analytic = float(tf.reduce_sum(gx * direction_x) + tf.reduce_sum(gs * direction_s))
        h = 1e-6
        numeric = float(scalar(x + h * direction_x, s + h * direction_s)
                        - scalar(x - h * direction_x, s - h * direction_s)) / (2 * h)
        assert abs(analytic - numeric) / max(1.0, abs(numeric)) < 1e-6


class TestStaticContractInCall:
    """Section 3.5: re-asserted in call(), because build() sees only the first shapes."""

    def test_a_built_layer_refuses_a_mismatched_pair(self):
        layer = GeoNorm()
        layer(list(_pair()))
        x, s = _pair()
        with pytest.raises(ValueError, match="same shape"):
            layer([x, s[:, :, :-1]])
        with pytest.raises(ValueError, match="same shape"):
            layer([x, s[:, 0]])


class TestKnobs:
    """Section 12.5: every knob varied with a measured difference, and a control that does not move.

    The layer owns no weights, so the knob-sensitivity oracle's weight signature is empty by design;
    the pins are direct output differences at configurations where the knob is observable."""

    @staticmethod
    def _delta(a, b, x, s, **call):
        return float(np.abs(_run(a, x, s, **call) - _run(b, x, s, **call)).max())

    def test_alpha(self):
        x, s = _pair(scale_s=0.3)
        assert self._delta(GeoNorm(alpha=1.0), GeoNorm(alpha=0.5), x, s) > 1e-2
        assert self._delta(GeoNorm(alpha=1.0), GeoNorm(alpha=1.0), x, s) == 0.0

    def test_clamp(self):
        x, s = _pair(scale_s=40.0)
        assert self._delta(GeoNorm(clamp=math.pi / 4), GeoNorm(clamp=math.pi / 8), x, s) > 1e-2
        assert self._delta(GeoNorm(clamp=math.pi / 4), GeoNorm(clamp=None), x, s) > 1e-2
        small_x, small_s = _pair(scale_s=0.01)           # the control: below the clamp, it is inert
        assert self._delta(GeoNorm(clamp=math.pi / 8), GeoNorm(clamp=None), small_x, small_s) == 0.0

    def test_schedule(self):
        x, s = _pair(scale_s=0.3)
        harmonic, sqrt = GeoNorm(schedule="harmonic"), GeoNorm(schedule="sqrt")
        linear = GeoNorm(schedule="linear", total_layers=8)
        assert self._delta(harmonic, sqrt, x, s, layer_idx=3) > 1e-3
        assert self._delta(sqrt, linear, x, s, layer_idx=3) > 1e-3
        assert self._delta(harmonic, sqrt, x, s, layer_idx=0) < 1e-6          # all three start at 1

    def test_total_layers(self):
        x, s = _pair(scale_s=0.3)
        a, b = GeoNorm(schedule="linear", total_layers=4), GeoNorm(schedule="linear", total_layers=40)
        assert self._delta(a, b, x, s, layer_idx=2) > 1e-3

    def test_epsilon_acts_on_a_vanishing_radius_and_is_inert_elsewhere(self):
        tiny = (np.random.default_rng(0).normal(size=(2, 3, 8)) * 1e-5).astype("float32")
        s = np.random.default_rng(1).normal(size=(2, 3, 8)).astype("float32")
        assert self._delta(GeoNorm(epsilon=1e-6), GeoNorm(epsilon=1e-3), tiny, s) > 1e-6
        x, s = _pair()
        assert self._delta(GeoNorm(epsilon=1e-6), GeoNorm(epsilon=1e-3), x, s) == 0.0

    def test_the_tangent_floor_differs_from_the_paper_only_below_float32_resolution(self):
        """The paper floors ||v|| at 1e-8 and ||x|| at 1e-6; this layer floors both at epsilon. Measured
        2026-11-20, CPU, float32, ||x||=4, ||v||=1e-7 and 1e-5: max|out(1e-6) - out(1e-8)| = 0.0 at
        both, because the displacement (~1e-8 relative) is below half an ulp of x (6e-8)."""
        x = np.ones((1, 1, 16), "float32")
        for size in (1e-7, 1e-5):
            v = np.zeros_like(x)
            v[0, 0, 0] = size
            assert self._delta(GeoNorm(epsilon=1e-6), GeoNorm(epsilon=1e-8), x, v) == 0.0


class TestDegenerateAndGraphModes:
    """Section 10.2 and 13.2.7."""

    @pytest.mark.parametrize("length", [0, 1])
    def test_static_degenerate_lengths_are_finite(self, length):
        x = np.ones((2, length, D), "float32")
        y = _run(GeoNorm(), x, x * 0.5)
        assert y.shape == (2, length, D) and np.isfinite(y).all()

    @pytest.mark.parametrize("length", [0, 1, L])
    def test_a_none_dimension_trace_is_finite_and_equals_eager(self, length):
        layer = GeoNorm()

        @tf.function(input_signature=[tf.TensorSpec([None, None, D], tf.float32)] * 2)
        def traced(a, b):
            return layer([a, b], layer_idx=2)

        rng = np.random.default_rng(0)
        x = rng.normal(size=(2, length, D)).astype("float32")
        s = rng.normal(size=(2, length, D)).astype("float32")
        out = traced(tf.constant(x), tf.constant(s)).numpy()
        assert np.isfinite(out).all()
        np.testing.assert_array_equal(out, keras.ops.convert_to_numpy(layer([x, s], layer_idx=2)))

    def test_xla_matches_eager(self):
        """Measured 2026-11-20, CPU, float32, output absmax 3.7: max|diff| 2.4e-07. The bound 1e-5 is 40x
        above that and ~200x below the 2e-3 a one percent change of alpha moves the output."""
        layer = GeoNorm()
        x, s = _pair()
        eager = keras.ops.convert_to_numpy(layer([x, s], layer_idx=1))

        @tf.function(jit_compile=True)
        def compiled(a, b):
            return layer([a, b], layer_idx=1)

        np.testing.assert_allclose(compiled(tf.constant(x), tf.constant(s)).numpy(), eager, rtol=0, atol=1e-5)


class TestPrecisionArms2:
    """Section 13.2.6: a control on the same input, and a realised dtype."""

    def test_float64_agrees_with_float32_to_rounding(self, float64_policy):
        """Measured 2026-11-20, CPU: float32 against float64 differs by 1.9e-07 at output absmax 3.7."""
        x, s = _pair(scale_s=3.0)
        control = _run(GeoNorm(), x, s, layer_idx=2)
        x64 = keras.ops.convert_to_tensor(x.astype("float64"))
        assert keras.backend.standardize_dtype(x64.dtype) == "float64"
        y = GeoNorm()([x64, keras.ops.convert_to_tensor(s.astype("float64"))], layer_idx=2)
        assert keras.backend.standardize_dtype(y.dtype) == "float64"
        np.testing.assert_allclose(keras.ops.convert_to_numpy(y), control, rtol=0, atol=5e-6)

    def test_half_precision_norm_is_preserved_to_the_float16_rounding(self, mixed_float16_policy):
        """Measured 2026-11-20, CPU, float16 in and out: max relative norm error 2.7e-04 (float16 unit
        roundoff 4.9e-04). Bound 2e-3 is 4 roundoffs; a layer that normalized in float16 and lost
        the tail would sit orders above."""
        x, s = _pair(scale_s=3.0)
        y = GeoNorm()([tf.constant(x, tf.float16), tf.constant(s, tf.float16)])
        assert keras.backend.standardize_dtype(y.dtype) == "float16"
        got = np.linalg.norm(keras.ops.convert_to_numpy(y).astype("float64"), axis=-1)
        want = np.linalg.norm(x.astype("float16").astype("float64"), axis=-1)
        assert np.abs(got / want - 1.0).max() < 2e-3
