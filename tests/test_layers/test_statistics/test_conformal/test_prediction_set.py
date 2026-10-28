"""Test suite for :class:`ConformalPredictionSet` and :func:`attach_conformal_sets`.

Covers construction and registration, the forward rule on hand numbers, agreement with
the numpy ``predict_sets`` (the two implementations pin each other), calibration,
a ``.keras`` round trip of a NON-DEFAULT ``tau``, the config path, and attaching the
layer to a Functional and to a subclassed classifier.
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.statistics.conformal import (
    ConformalPredictionSet,
    attach_conformal_sets,
)
from dl_techniques.utils.conformal_classification import (
    calibrate_aps,
    predict_sets,
)

CAL_Q_HAT = 0.83


def _mask(layer, logits):
    probs, in_set = layer(np.asarray(logits, dtype="float32"))
    return keras.ops.convert_to_numpy(probs), keras.ops.convert_to_numpy(in_set)


class TestConformalPredictionSet:

    @pytest.fixture(autouse=True)
    def _seed(self) -> None:
        keras.utils.set_random_seed(1234)
        np.random.seed(1234)

    def test_construction(self, registration_contract) -> None:
        registration_contract(ConformalPredictionSet)
        layer = ConformalPredictionSet()
        assert layer.trainable is False and layer.q_hat is None
        layer(np.zeros((1, 4), dtype="float32"))
        assert len(layer.trainable_weights) == 0
        assert [w.name for w in layer.non_trainable_weights] == ["conformal_q_hat"]

    def test_uncalibrated_keeps_every_label(self) -> None:
        layer = ConformalPredictionSet()
        _, in_set = _mask(layer, np.random.normal(size=(6, 9)))
        assert in_set.all()

    def test_forward_on_hand_numbers(self) -> None:
        # p = [0.1, 0.6, 0.3]: cumulative through each label in sorted order is
        # 0.6 (class 1), 0.9 (class 2), 1.0 (class 0).
        logits = np.log([[0.1, 0.6, 0.3]])
        # thresholds sit between the cumulative values, away from float32 rounding
        layer = ConformalPredictionSet(q_hat_init=0.95)
        probs, in_set = _mask(layer, logits)
        np.testing.assert_allclose(probs, [[0.1, 0.6, 0.3]], atol=1e-6)
        assert in_set.tolist() == [[0.0, 1.0, 1.0]]
        layer.set_q_hat(0.65)
        assert _mask(layer, logits)[1].tolist() == [[0.0, 1.0, 0.0]]

    def test_a_label_whose_mass_equals_q_hat_is_in_the_set(self) -> None:
        # [0.5, 0.5]: the cumulative masses 0.5 and 1.0 are exact in float32.
        layer = ConformalPredictionSet(q_hat_init=0.5)
        assert _mask(layer, np.zeros((1, 2)))[1].tolist() == [[1.0, 0.0]]

    def test_a_q_hat_below_the_top_probability_gives_an_empty_set(self) -> None:
        layer = ConformalPredictionSet(q_hat_init=0.3)
        assert _mask(layer, np.log([[0.1, 0.6, 0.3]]))[1].sum() == 0

    def test_it_matches_the_numpy_prediction_sets(self) -> None:
        rng = np.random.default_rng(0)
        logits = (rng.normal(size=(600, 12)) * 3).astype("float32")
        labels = rng.integers(0, 12, 600)
        tau = calibrate_aps(logits[:300], labels[:300], 0.1, randomized=False)
        layer = ConformalPredictionSet()
        layer(logits[:1])
        layer.set_q_hat(tau)
        got = _mask(layer, logits[300:])[1].astype(bool)
        want = predict_sets(logits[300:], tau, randomized=False)
        assert np.array_equal(got, want)

    def test_ties_follow_class_index_like_the_numpy_rule(self) -> None:
        logits = np.zeros((1, 400), dtype="float32")
        layer = ConformalPredictionSet(q_hat_init=0.514)
        got = _mask(layer, logits)[1].astype(bool)
        assert got.sum() == 205 and got[0, :205].all()
        assert np.array_equal(got, predict_sets(logits, 0.514, randomized=False))

    def test_set_q_hat_caps_an_infinite_value_and_refuses_nan(self) -> None:
        layer = ConformalPredictionSet()
        layer(np.zeros((1, 3), dtype="float32"))
        layer.set_q_hat(float("inf"))
        assert float(keras.ops.convert_to_numpy(layer.q_hat)) == 2.0
        with pytest.raises(ValueError, match="non-negative, got nan"):
            layer.set_q_hat(float("nan"))

    def test_logits_must_be_rank_2(self) -> None:
        with pytest.raises(ValueError, match="batch, num_classes"):
            ConformalPredictionSet()(np.zeros((1, 2, 3), dtype="float32"))

    def test_compute_output_shape(self) -> None:
        a, b = ConformalPredictionSet().compute_output_shape((None, 7))
        assert a == b == (None, 7)

    def test_round_trip_keras(self, tmp_path) -> None:
        inp = keras.Input(shape=(10,))
        probs, in_set = ConformalPredictionSet()(inp)
        model = keras.Model(inp, [probs, in_set])
        layer = next(l for l in model.layers if isinstance(l, ConformalPredictionSet))
        layer.set_q_hat(CAL_Q_HAT)
        x = np.random.normal(size=(5, 10)).astype("float32") * 2
        pre = [keras.ops.convert_to_numpy(t) for t in model(x)]

        path = tmp_path / "m.keras"
        model.save(path)
        reloaded = keras.models.load_model(
            path, custom_objects={"ConformalPredictionSet": ConformalPredictionSet}
        )
        rel = next(l for l in reloaded.layers if isinstance(l, ConformalPredictionSet))
        assert float(keras.ops.convert_to_numpy(rel.q_hat)) == pytest.approx(CAL_Q_HAT, abs=1e-6)
        for a, b in zip(pre, [keras.ops.convert_to_numpy(t) for t in reloaded(x)]):
            np.testing.assert_allclose(a, b, atol=1e-6)

    def test_config_round_trip(self) -> None:
        layer = ConformalPredictionSet()
        layer(np.zeros((1, 3), dtype="float32"))
        layer.set_q_hat(CAL_Q_HAT)
        cfg = layer.get_config()
        assert cfg["q_hat_init"] == pytest.approx(CAL_Q_HAT, abs=1e-6)
        layer2 = ConformalPredictionSet.from_config(cfg)
        layer2(np.zeros((1, 3), dtype="float32"))
        assert float(keras.ops.convert_to_numpy(layer2.q_hat)) == pytest.approx(CAL_Q_HAT, abs=1e-6)


class _Subclassed(keras.Model):
    def __init__(self, classes=5, **kw):
        super().__init__(**kw)
        self.classes = classes
        self.dense = keras.layers.Dense(classes)

    def call(self, x):
        return self.dense(x)


class TestAttach:

    def test_a_functional_classifier_keeps_its_logits(self) -> None:
        inp = keras.Input((4,))
        base = keras.Model(inp, keras.layers.Dense(6)(inp))
        wrapped = attach_conformal_sets(base, 0.9, (4,))
        x = np.random.normal(size=(3, 4)).astype("float32")
        logits, probs, in_set = wrapped(x)
        np.testing.assert_allclose(keras.ops.convert_to_numpy(logits), base(x), atol=1e-6)
        assert keras.ops.convert_to_numpy(in_set).shape == (3, 6)

    def test_a_subclassed_classifier_attaches_and_shares_weights(self) -> None:
        base = _Subclassed()
        base(np.zeros((1, 4), dtype="float32"))
        wrapped = attach_conformal_sets(base, 0.9, (4,))
        x = np.random.normal(size=(3, 4)).astype("float32")
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(wrapped(x)[0]), base(x), atol=1e-6
        )

    def test_the_q_hat_reaches_the_attached_layer(self) -> None:
        inp = keras.Input((4,))
        base = keras.Model(inp, keras.layers.Dense(6)(inp))
        layer = attach_conformal_sets(base, CAL_Q_HAT, (4,)).get_layer("conformal_prediction_set")
        assert float(keras.ops.convert_to_numpy(layer.q_hat)) == pytest.approx(CAL_Q_HAT, abs=1e-6)

    def test_a_multi_output_model_is_refused(self) -> None:
        inp = keras.Input((4,))
        d = keras.layers.Dense(3)(inp)
        with pytest.raises(ValueError, match="single-output"):
            attach_conformal_sets(keras.Model(inp, [d, d]), 0.9, (4,))


# ---------------------------------------------------------------------
# v2 guide section 16.1 / 16.3 arms: validation, dtype, XLA, shapes, round trip on values
# ---------------------------------------------------------------------

import tensorflow as tf  # noqa: E402


class TestV2Arms:

    def test_a_negative_q_hat_is_refused_at_construction_and_assignment(self) -> None:
        with pytest.raises(ValueError, match="q_hat_init must be non-negative, got -0.1"):
            ConformalPredictionSet(q_hat_init=-0.1)
        layer = ConformalPredictionSet()
        layer(np.zeros((1, 3), dtype="float32"))
        with pytest.raises(ValueError, match="q_hat must be non-negative, got -0.5"):
            layer.set_q_hat(-0.5)

    def test_dtype_arms_agree_with_the_float32_control(self, dtype_policy) -> None:
        rng = np.random.default_rng(3)
        logits = (rng.normal(size=(64, 12)) * 3).astype("float32")
        layer = ConformalPredictionSet(q_hat_init=0.8)
        probs, in_set = layer(logits)
        probs, in_set = keras.ops.convert_to_numpy(probs), keras.ops.convert_to_numpy(in_set)
        assert np.isfinite(probs).all() and set(np.unique(in_set)) <= {0.0, 1.0}
        # The numpy rule on the same logits is the control for every policy. Under
        # mixed_float16 the logits reach the layer rounded to float16, which can move a
        # label across the threshold: demand >= 99% agreement, exact at float32/float64.
        want = predict_sets(logits, 0.8, randomized=False)
        agreement = np.mean(in_set.astype(bool) == want)
        assert agreement >= (1.0 if dtype_policy == "float32" else 0.99), agreement

    def test_xla_matches_eager(self) -> None:
        rng = np.random.default_rng(4)
        logits = (rng.normal(size=(32, 10)) * 3).astype("float32")
        layer = ConformalPredictionSet(q_hat_init=0.85)
        eager = [keras.ops.convert_to_numpy(t) for t in layer(logits)]
        fn = tf.function(lambda t: layer(t), jit_compile=True)
        for a, b in zip(eager, fn(tf.constant(logits))):
            np.testing.assert_allclose(a, b.numpy(), rtol=0, atol=1e-6)

    @pytest.mark.parametrize("batch, classes", [(1, 2), (1, 5), (3, 2)])
    def test_degenerate_batch_and_class_counts(self, batch, classes) -> None:
        layer = ConformalPredictionSet(q_hat_init=0.99)
        logits = np.random.default_rng(0).normal(size=(batch, classes)).astype("float32")
        probs, in_set = layer(logits)
        assert tuple(in_set.shape) == (batch, classes)
        assert np.isfinite(keras.ops.convert_to_numpy(probs)).all()

    def test_a_single_class_is_refused(self) -> None:
        # softmax over one class is always 1: the set is trivially everything.
        with pytest.raises(ValueError, match="at least 2 classes, got 1"):
            ConformalPredictionSet()(np.zeros((3, 1), dtype="float32"))

    def test_a_trace_with_an_unknown_batch_matches_the_static_one(self) -> None:
        layer = ConformalPredictionSet(q_hat_init=0.85)
        logits = (np.random.default_rng(5).normal(size=(7, 9)) * 3).astype("float32")
        static = keras.ops.convert_to_numpy(layer(logits)[1])
        fn = tf.function(lambda t: layer(t)[1], input_signature=[tf.TensorSpec([None, 9], tf.float32)])
        np.testing.assert_array_equal(fn(tf.constant(logits)).numpy(), static)

    def test_weights_are_identical_before_the_reloaded_model_is_called(self, tmp_path) -> None:
        inp = keras.Input(shape=(10,))
        model = keras.Model(inp, ConformalPredictionSet()(inp))
        layer = next(l for l in model.layers if isinstance(l, ConformalPredictionSet))
        layer.set_q_hat(CAL_Q_HAT)
        path = tmp_path / "w.keras"
        model.save(path)
        loaded = keras.models.load_model(
            path, custom_objects={"ConformalPredictionSet": ConformalPredictionSet}
        )
        rel = next(l for l in loaded.layers if isinstance(l, ConformalPredictionSet))
        for a, b in zip(layer.weights, rel.weights):
            assert a.path.split("/")[-1] == b.path.split("/")[-1]
            np.testing.assert_array_equal(
                keras.ops.convert_to_numpy(a), keras.ops.convert_to_numpy(b)
            )
        x = np.random.default_rng(6).normal(size=(5, 10)).astype("float32") * 3
        for a, b in zip(model(x, training=False), loaded(x, training=False)):
            np.testing.assert_array_equal(
                keras.ops.convert_to_numpy(a), keras.ops.convert_to_numpy(b)
            )

    def test_the_layer_has_exactly_one_weight_and_it_is_frozen(self) -> None:
        layer = ConformalPredictionSet(name="cps")
        layer(np.zeros((2, 4), dtype="float32"))
        assert [w.path for w in layer.weights] == ["cps/conformal_q_hat"]
        assert layer.trainable_weights == [] and layer.weights[0].trainable is False


def test_the_threshold_is_not_rounded_to_float16_under_mixed_precision(mixed_float16_policy) -> None:
    # logits [0, -1] are exact in float16; the top label's mass is 0.7310586. A threshold
    # of 0.7311 includes it. Rounded to float16 it would become 0.7309570 and exclude it.
    logits = np.array([[0.0, -1.0]], dtype="float32")
    layer = ConformalPredictionSet()
    layer(logits)
    layer.set_q_hat(0.7311)
    assert layer.q_hat.dtype == "float32"
    in_set = keras.ops.convert_to_numpy(layer(logits)[1])
    assert in_set.tolist() == [[1.0, 0.0]]
