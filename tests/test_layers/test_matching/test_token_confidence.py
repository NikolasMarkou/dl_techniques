"""Tests for MatchTokenConfidence: oracle parity, the stop-gradient guard, serialization."""

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.matching.token_confidence import MatchTokenConfidence
from tests import lightglue_reference_numpy as ref
from tests.test_models.test_sam.dead_component_oracle import (
    component_response,
    fit_one_step_moved_variables,
    no_op_kill,
    zeroed_variables,
)

pytestmark = pytest.mark.usefixtures("tf32_disabled")

_EPS = {"float16": float(np.finfo(np.float16).eps), "float32": float(np.finfo(np.float32).eps),
        "float64": float(np.finfo(np.float64).eps)}
_NP = {"float16": np.float16, "bfloat16": np.float32, "float32": np.float32, "float64": np.float64}
PFX = "token_confidence.0"
D = 16


def _round(a, cdt):
    return a.astype(_NP[cdt]).astype(np.float64)


def _layer(cdt, seed=0, cls=MatchTokenConfidence):
    weights = {k: _round(v, cdt) for k, v in ref.random_weights(np.random.RandomState(seed), D, 2, 2).items()}
    layer = cls(dim=D)
    layer.build((None, None, D), (None, None, D))
    layer.token_0.kernel.assign(np.asarray(weights[PFX + ".token.0.weight"].T, dtype=layer.token_0.kernel.dtype))
    layer.token_0.bias.assign(np.asarray(weights[PFX + ".token.0.bias"], dtype=layer.token_0.bias.dtype))
    return layer, weights


class _NoStopGradient(MatchTokenConfidence):
    """MUTANT: the confidence head without the detach."""

    def _logit(self, desc):
        return keras.ops.cast(self.token_0(desc), "float32")[..., 0]


def _input_gradient(layer, x0, x1):
    a, b = tf.constant(x0, tf.float32), tf.constant(x1, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch([a, b])
        c0, c1 = layer(a, b)
        loss = tf.reduce_sum(c0) + tf.reduce_sum(c1)
    grads = tape.gradient(loss, [a, b])
    # TF reports a stop_gradient'd path as None; None IS an exactly-zero gradient
    return [np.zeros(x.shape, np.float32) if g is None else g.numpy() for g, x in zip(grads, (a, b))]


class TestParity:
    def test_matches_oracle(self, dtype_policy):
        cdt = keras.mixed_precision.global_policy().compute_dtype
        layer, weights = _layer(cdt)
        rng = np.random.RandomState(1)
        d0, d1 = _round(rng.normal(size=(2, 5, D)), cdt), _round(rng.normal(size=(2, 8, D)), cdt)
        c0, c1 = layer(d0.astype(_NP[cdt]), d1.astype(_NP[cdt]))
        w0 = np.stack([ref.token_confidence(d0[b], weights, PFX) for b in range(2)])
        w1 = np.stack([ref.token_confidence(d1[b], weights, PFX) for b in range(2)])
        c0, c1 = [keras.ops.convert_to_numpy(c).astype(np.float64) for c in (c0, c1)]
        assert c0.shape == (2, 5) and c1.shape == (2, 8)
        # Dense fan-in D at unit roundoff eps, sigmoid is 0.25-Lipschitz, plus float32 slack
        tol = 1e-10 if cdt == "float64" else (D + 1) * _EPS[cdt] * 0.25 * (
            1.0 + float(np.abs(weights[PFX + ".token.0.weight"]).sum() * 4.0)) + 8 * _EPS["float32"]
        assert np.abs(c0 - w0).max() <= tol and np.abs(c1 - w1).max() <= tol
        assert np.all((c0 > 0) & (c0 < 1))

    def test_output_dtype_is_at_least_float32(self, dtype_policy):
        cdt = keras.mixed_precision.global_policy().compute_dtype
        layer, _ = _layer(cdt)
        x = np.zeros((1, 3, D), _NP[cdt])
        c0, _ = layer(x, x)
        assert keras.backend.standardize_dtype(c0.dtype) == ("float64" if cdt == "float64" else "float32")


class TestStopGradient:
    def _data(self):
        rng = np.random.RandomState(2)
        return rng.normal(size=(2, 5, D)).astype("float32"), rng.normal(size=(2, 6, D)).astype("float32")

    def test_input_gradient_is_exactly_zero(self):
        layer, _ = _layer("float32")
        g0, g1 = _input_gradient(layer, *self._data())
        assert np.all(g0 == 0.0) and np.all(g1 == 0.0)

    def test_head_parameters_still_receive_gradient(self):
        layer, _ = _layer("float32")
        a, b = [tf.constant(x) for x in self._data()]
        with tf.GradientTape() as tape:
            c0, c1 = layer(a, b)
            loss = tf.reduce_sum(c0) + tf.reduce_sum(c1)
        grads = tape.gradient(loss, layer.trainable_variables)
        assert all(g is not None and np.abs(g.numpy()).max() > 0 for g in grads)

    def test_guard_is_red_against_a_version_without_stop_gradient(self):
        layer, _ = _layer("float32", cls=_NoStopGradient)
        g0, g1 = _input_gradient(layer, *self._data())
        assert np.abs(g0).max() > 0.0 and np.abs(g1).max() > 0.0


class _LogitsFromProbability(MatchTokenConfidence):
    """MUTANT: the 'logits' are recovered from the clipped probability, so they saturate."""

    def call(self, desc0, desc1, return_logits=False):
        c0, c1 = super().call(desc0, desc1)
        if not return_logits:
            return c0, c1
        eps = 1e-7
        inv = lambda p: keras.ops.log(keras.ops.clip(p, eps, 1 - eps) / (1 - keras.ops.clip(p, eps, 1 - eps)))
        return c0, c1, inv(c0), inv(c1)


class TestLogits:
    def _data(self, scale=1.0):
        rng = np.random.RandomState(5)
        return (rng.normal(size=(2, 5, D)).astype("float32") * scale,
                rng.normal(size=(2, 6, D)).astype("float32") * scale)

    def test_default_call_is_the_two_tuple(self):
        layer, _ = _layer("float32")
        assert len(layer(*self._data())) == 2

    def test_logits_shape_dtype_and_sigmoid_consistency(self):
        layer, _ = _layer("float32")
        c0, c1, l0, l1 = layer(*self._data(), return_logits=True)
        assert tuple(l0.shape) == (2, 5) and tuple(l1.shape) == (2, 6)
        assert keras.backend.standardize_dtype(l0.dtype) == "float32"
        np.testing.assert_allclose(keras.ops.convert_to_numpy(c0),
                                   1.0 / (1.0 + np.exp(-keras.ops.convert_to_numpy(l0))), atol=1e-6)
        d0, d1 = layer(*self._data())
        np.testing.assert_array_equal(keras.ops.convert_to_numpy(d0), keras.ops.convert_to_numpy(c0))

    def test_logits_match_dense_head(self):
        layer, _ = _layer("float32")
        x0, x1 = self._data()
        _, _, l0, _ = layer(x0, x1, return_logits=True)
        want = x0 @ layer.token_0.kernel.numpy()[:, 0] + layer.token_0.bias.numpy()[0]
        np.testing.assert_allclose(keras.ops.convert_to_numpy(l0), want, atol=1e-5)

    @staticmethod
    def _logit_grad_at_saturation(cls):
        """d logit / d kernel stays 1 per element even when sigmoid(logit) rounds to 1."""
        layer, _ = _layer("float32", cls=cls)
        layer.token_0.kernel.assign(np.ones((D, 1), np.float32))
        x = np.full((1, 2, D), 3.0, np.float32)           # logit = 48 + bias, far past 17
        a, b = tf.constant(x), tf.constant(x)
        with tf.GradientTape() as tape:
            _, _, l0, l1 = layer(a, b, return_logits=True)
            loss = tf.reduce_sum(l0) + tf.reduce_sum(l1)
        g = tape.gradient(loss, layer.token_0.kernel).numpy()
        return g

    def test_logit_gradient_survives_saturation(self):
        g = self._logit_grad_at_saturation(MatchTokenConfidence)
        np.testing.assert_allclose(g[:, 0], 12.0, rtol=1e-6)    # 4 tokens x input 3.0

    def test_gradient_guard_is_red_when_logits_come_from_the_clipped_probability(self):
        g = self._logit_grad_at_saturation(_LogitsFromProbability)
        assert np.abs(g).max() == 0.0

    def test_input_gradient_still_zero_with_logits(self):
        layer, _ = _layer("float32")
        a, b = [tf.constant(x) for x in self._data()]
        with tf.GradientTape() as tape:
            tape.watch([a, b])
            _, _, l0, l1 = layer(a, b, return_logits=True)
            loss = tf.reduce_sum(l0) + tf.reduce_sum(l1)
        grads = tape.gradient(loss, [a, b])
        assert all(g is None or np.all(g.numpy() == 0) for g in grads)


class TestConstruction:
    @pytest.mark.parametrize("dim", [0, -1])
    def test_bad_dim_raises(self, dim):
        with pytest.raises(ValueError):
            MatchTokenConfidence(dim=dim)

    def test_bad_build_shape_raises(self):
        with pytest.raises(ValueError):
            MatchTokenConfidence(dim=16).build((None, 5, 17), (None, 5, 16))

    def test_sublayer_name_follows_the_torch_state_dict(self):
        layer = MatchTokenConfidence(dim=16)
        layer.build((None, None, 16), (None, None, 16))
        assert tuple(layer.token_0.kernel.shape) == (16, 1)
        assert [w.path.split("/")[-2] for w in layer.weights] == ["token_0", "token_0"]

    def test_get_config_round_trip(self):
        config = MatchTokenConfidence(dim=24, name="tc").get_config()
        assert config["dim"] == 24 and MatchTokenConfidence.from_config(config).get_config() == config

    def test_compute_output_shape(self):
        assert MatchTokenConfidence(dim=16).compute_output_shape((None, 9, 16), (None, 7, 16)) == ((None, 9), (None, 7))


def _wrapper(m=6, n=5):
    d0, d1 = keras.Input((m, D)), keras.Input((n, D))
    c0, c1 = MatchTokenConfidence(dim=D)(d0, d1)
    return keras.Model([d0, d1], [c0, c1])


class TestSerializationAndGradientFlow:
    def test_save_load_round_trip(self, tmp_path):
        rng = np.random.RandomState(3)
        data = [rng.normal(size=(3, 6, D)).astype("float32"), rng.normal(size=(3, 5, D)).astype("float32")]
        model = _wrapper()
        expected = model.predict(data, verbose=0)
        path = str(tmp_path / "tc.keras")
        model.save(path)
        for a, b in zip(keras.models.load_model(path).predict(data, verbose=0), expected):
            np.testing.assert_array_equal(a, b)

    def test_head_variables_move_and_zeroing_responds(self):
        rng = np.random.RandomState(4)
        data = [rng.normal(size=(4, 6, D)).astype("float32"), rng.normal(size=(4, 5, D)).astype("float32")]
        targets = [rng.uniform(size=(4, 6)).astype("float32"), rng.uniform(size=(4, 5)).astype("float32")]
        model = _wrapper()
        model.compile(optimizer=keras.optimizers.Adam(1e-2), loss=["mse", "mse"])
        report = fit_one_step_moved_variables(model, data, targets)
        assert report.total == 2 and report.unmoved == (), report.unmoved
        layer = next(l for l in model.layers if isinstance(l, MatchTokenConfidence))

        def metric():
            return float(sum(np.abs(o).sum() for o in model.predict(data, verbose=0)))

        control = component_response(metric, no_op_kill, name="no-op control")
        assert not control.moved and control.delta == 0.0
        result = component_response(metric, lambda: zeroed_variables(layer.weights), name="token_0", atol=1e-4)
        assert result.moved, result.summary()
