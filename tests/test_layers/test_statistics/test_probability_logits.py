"""Test suite for :class:`ProbabilityLogits`.

Covers construction and registration, the forward rule against a float64 numpy oracle,
monotonicity (the class order is the input's own), finiteness at 0 and 1, the epsilon
validation, the config path, a ``.keras`` round trip with a NON-DEFAULT epsilon, the
dtype arms, XLA against eager, and the layer under the softmax of
``ConformalPredictionSet`` (the reason it exists).
"""

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.statistics.conformal import ConformalPredictionSet
from dl_techniques.layers.statistics.probability_logits import ProbabilityLogits


def _np_logit(p, epsilon=1e-6):
    p = np.clip(np.asarray(p, dtype=np.float64), epsilon, 1.0 - epsilon)
    return np.log(p) - np.log1p(-p)


def _run(layer, p):
    return keras.ops.convert_to_numpy(layer(np.asarray(p, dtype="float32")))


class TestProbabilityLogits:

    @pytest.fixture(autouse=True)
    def _seed(self) -> None:
        keras.utils.set_random_seed(1234)
        np.random.seed(1234)

    def test_construction(self, registration_contract) -> None:
        registration_contract(ProbabilityLogits)
        layer = ProbabilityLogits()
        layer(np.full((1, 3), 0.5, dtype="float32"))
        assert layer.weights == [] and layer.epsilon == 1e-6

    def test_forward_on_hand_numbers(self) -> None:
        out = _run(ProbabilityLogits(), [[0.5, 0.75, 0.25]])
        np.testing.assert_allclose(out, [[0.0, np.log(3.0), -np.log(3.0)]], atol=1e-6)

    def test_forward_matches_the_float64_oracle(self) -> None:
        p = np.random.default_rng(0).uniform(0.001, 0.999, size=(32, 17))
        np.testing.assert_allclose(
            _run(ProbabilityLogits(), p), _np_logit(p), rtol=0, atol=1e-4
        )

    def test_the_class_order_is_the_inputs_own(self) -> None:
        p = np.random.default_rng(1).uniform(0.0, 1.0, size=(64, 10)).astype("float32")
        out = _run(ProbabilityLogits(), p)
        assert np.array_equal(np.argsort(out, axis=-1), np.argsort(p, axis=-1))

    @pytest.mark.parametrize("p, bound", [(0.0, -1), (1.0, 1)])
    def test_the_ends_are_finite_and_clipped(self, p, bound) -> None:
        epsilon = 1e-4
        out = _run(ProbabilityLogits(epsilon=epsilon), [[p]])
        assert np.isfinite(out).all()
        assert out[0, 0] == pytest.approx(bound * np.log((1 - epsilon) / epsilon), abs=1e-3)

    def test_the_output_has_the_input_shape(self) -> None:
        layer = ProbabilityLogits()
        assert layer.compute_output_shape((None, 7)) == (None, 7)
        assert tuple(layer(np.full((2, 3, 4), 0.5, dtype="float32")).shape) == (2, 3, 4)

    @pytest.mark.parametrize("bad", [0.0, -1e-3, 0.5, 0.7, float("nan"), "0.1", None, True])
    def test_an_epsilon_outside_the_open_half_interval_is_refused(self, bad) -> None:
        with pytest.raises(ValueError, match="epsilon must be a number in"):
            ProbabilityLogits(epsilon=bad)

    def test_the_refusal_names_the_value(self) -> None:
        with pytest.raises(ValueError, match="got 0.75"):
            ProbabilityLogits(epsilon=0.75)

    def test_config_round_trip(self) -> None:
        layer = ProbabilityLogits(epsilon=0.01, name="pl")
        clone = ProbabilityLogits.from_config(layer.get_config())
        assert clone.epsilon == 0.01 and clone.name == "pl"

    def test_a_keras_round_trip_keeps_a_non_default_epsilon(self, tmp_path) -> None:
        inputs = keras.Input((5,))
        model = keras.Model(inputs, ProbabilityLogits(epsilon=0.01)(inputs))
        p = np.array([[0.0, 0.2, 0.5, 0.8, 1.0]], dtype="float32")
        path = tmp_path / "m.keras"
        model.save(path)
        loaded = keras.models.load_model(path)
        np.testing.assert_array_equal(
            keras.ops.convert_to_numpy(loaded(p)), keras.ops.convert_to_numpy(model(p))
        )
        assert loaded.layers[-1].epsilon == 0.01
        assert float(keras.ops.convert_to_numpy(loaded(p))[0, 0]) == pytest.approx(
            _np_logit(0.0, 0.01), abs=1e-4
        )

    def test_dtype_arms_stay_finite_and_agree_with_the_oracle(self, dtype_policy) -> None:
        p = np.random.default_rng(2).uniform(0.05, 0.95, size=(16, 12)).astype("float32")
        out = _run(ProbabilityLogits(), p)
        assert np.isfinite(out).all()
        # mixed_float16 rounds the probabilities to 11 significant bits before the layer
        # sees them: 3e-2 in the logit near p = 0.95 (d logit = dp / (p (1 - p))).
        atol = 1e-4 if dtype_policy != "mixed_float16" else 3e-2
        np.testing.assert_allclose(out, _np_logit(p), rtol=0, atol=atol)

    def test_mixed_float16_does_not_round_the_clip_to_one(self, mixed_float16_policy) -> None:
        # 1 - 1e-6 is exactly 1.0 in float16; the arithmetic must run in float32.
        out = _run(ProbabilityLogits(), [[1.0, 0.0]])
        assert np.isfinite(out).all() and out[0, 0] > 10.0 and out[0, 1] < -10.0

    def test_xla_matches_eager(self) -> None:
        p = np.random.default_rng(4).uniform(0.0, 1.0, size=(32, 10)).astype("float32")
        layer = ProbabilityLogits()
        eager = keras.ops.convert_to_numpy(layer(p))
        fn = tf.function(lambda t: layer(t), jit_compile=True)
        np.testing.assert_allclose(eager, fn(tf.constant(p)).numpy(), rtol=0, atol=1e-5)

    def test_a_softmax_over_the_raw_values_is_flat_and_over_the_logits_is_not(self) -> None:
        # The reason the layer exists: 1000 lengths in [0, 1] differ by at most 1.
        p = np.full((1, 1000), 0.01, dtype="float32")
        p[0, 7] = 0.99
        raw = keras.ops.convert_to_numpy(keras.ops.softmax(p, axis=-1))
        sharp = keras.ops.convert_to_numpy(
            keras.ops.softmax(keras.ops.convert_to_tensor(_run(ProbabilityLogits(), p)), axis=-1)
        )
        assert raw.max() < 0.01
        assert sharp.max() > 0.9
        probs, _ = ConformalPredictionSet()(_run(ProbabilityLogits(), p))
        assert int(np.argmax(keras.ops.convert_to_numpy(probs))) == 7

    def test_a_trace_with_an_unknown_batch_matches_the_static_one(self) -> None:
        layer = ProbabilityLogits()
        p = np.random.default_rng(5).uniform(size=(3, 6)).astype("float32")
        fn = tf.function(
            lambda t: layer(t), input_signature=[tf.TensorSpec((None, 6), tf.float32)]
        )
        np.testing.assert_allclose(fn(tf.constant(p)).numpy(), _run(layer, p), atol=1e-6)
