"""Tests for ``metrics/keypoint_matching.py``.

Expected counts are derived by hand in the comments of each case. Log assignments are
built from probabilities (``_la``) so that the mutual pairs and their scores are known.
"""

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.losses.lightglue_loss import LightGlueLoss, pack_matches
from dl_techniques.metrics.keypoint_matching import KeypointMatchMetric
from dl_techniques.models.vision.keypoints.lightglue.model import LightGlue

LOW = 1e-3  # exp(score) of a non-pair, below the 0.1 threshold


def _la(pairs, m, n, high=0.9, batch_axis=True):
    """``(1, M+1, N+1)`` log assignment whose only confident entries are ``pairs``."""
    p = np.full((m + 1, n + 1), LOW, "float32")
    for i, j in pairs:
        p[i, j] = high
    out = np.log(p)
    return out[None] if batch_axis else out


def _packed(labels0, labels1):
    return np.concatenate([np.asarray(labels0, "int32"), np.asarray(labels1, "int32")])[None]


def _run(mode, y_true, y_pred, **kw):
    metric = KeypointMatchMetric(mode)
    metric.update_state(y_true, y_pred, **kw)
    return metric, float(metric.result())


# Shared hand case, M = N = 5.
# predicted pairs (mutual, score 0.9): (0,0) (1,1) (2,2)
# labels0 = [0, 2, -2, 3, 4]   labels1 = [0, -1, 1, 3, 4]
#   i=0: label 0 == predicted 0                    -> TP, counts as prediction
#   i=1: label 2 != predicted 1                    -> FP, counts as prediction
#   i=2: label -2 (ignored)                        -> excluded from everything
#   i=3, i=4: labels 3, 4, nothing predicted       -> false negatives
# predicted = 2, TP = 1, ground truth (labels0 >= 0) = 4 (i = 0, 1, 3, 4)
# precision = 1/2, recall = 1/4, f1 = 2*1/(2+4) = 1/3
HAND_PRED = [(0, 0), (1, 1), (2, 2)]
HAND_L0 = [0, 2, -2, 3, 4]
HAND_L1 = [0, -1, 1, 3, 4]


class TestHandCounts:
    def test_precision(self):
        _, v = _run("precision", _packed(HAND_L0, HAND_L1), _la(HAND_PRED, 5, 5))
        assert v == pytest.approx(0.5)

    def test_recall(self):
        _, v = _run("recall", _packed(HAND_L0, HAND_L1), _la(HAND_PRED, 5, 5))
        assert v == pytest.approx(0.25)

    def test_f1(self):
        _, v = _run("f1", _packed(HAND_L0, HAND_L1), _la(HAND_PRED, 5, 5))
        assert v == pytest.approx(1.0 / 3.0)

    def test_counters(self):
        met, _ = _run("precision", _packed(HAND_L0, HAND_L1), _la(HAND_PRED, 5, 5))
        assert (float(met.true_positives), float(met.predicted), float(met.ground_truth)) == (1, 2, 4)

    def test_ignored_partner_side_excluded(self):
        # same predictions but labels1[1] = -2: pair (1,1) lands on an ignored image-1
        # keypoint, so it is no prediction: predicted = 1 (only i=0), TP = 1 -> precision 1
        l1 = [0, -2, 1, 3, 4]
        _, v = _run("precision", _packed(HAND_L0, l1), _la(HAND_PRED, 5, 5))
        assert v == pytest.approx(1.0)

    def test_padded_slots_excluded(self):
        # M = N = 4, the last slot of each image is padding (label -2 both sides) and
        # carries a confident pair (3,3) as an unmasked model output would.
        # pairs (0,0) (3,3); labels0 = [0,-1,-1,-2]; labels1 = [0,-1,-1,-2]
        # predicted = 1 (pair (3,3) excluded), TP = 1, ground truth = 1
        l0, l1 = [0, -1, -1, -2], [0, -1, -1, -2]
        for mode in ("precision", "recall"):
            _, v = _run(mode, _packed(l0, l1), _la([(0, 0), (3, 3)], 4, 4))
            assert v == pytest.approx(1.0)

    def test_dustbin_label_prediction_is_false_positive(self):
        # pair (0,0) predicted, label0[0] = -1 (dustbin): predicted = 1, TP = 0
        met, v = _run("precision", _packed([-1, -1], [-1, -1]), _la([(0, 0)], 2, 2))
        assert v == 0.0 and float(met.predicted) == 1

    def test_threshold_is_strict_on_exp_score(self):
        # exp(score) = 0.9 passes 0.5 and fails 0.95
        for thr, want in ((0.5, 1.0), (0.95, 0.0)):
            metric = KeypointMatchMetric("recall", threshold=thr)
            metric.update_state(_packed([0], [0]), _la([(0, 0)], 1, 1))
            assert float(metric.result()) == want

    def test_four_dim_last_layer_is_scored(self):
        good = _la([(0, 0)], 1, 1)
        bad = _la([], 1, 1)
        stacked = np.stack([good, bad, good], axis=1)       # (1, 3, 2, 2), last = good
        stacked_bad_last = np.stack([good, good, bad], axis=1)
        y = _packed([0], [0])
        assert _run("recall", y, stacked)[1] == 1.0
        assert _run("recall", y, stacked_bad_last)[1] == 0.0

    def test_sample_weight(self):
        # two images: first hand case (TP 1, pred 2, gt 4), second a perfect single pair.
        # weights (2, 1): TP = 2 + 1, predicted = 4 + 1, gt = 8 + 1 -> precision 3/5
        l0 = [[0, 2, -2, 3, 4], [0, -1, -1, -1, -1]]
        l1 = [[0, -1, 1, 3, 4], [0, -1, -1, -1, -1]]
        y = np.concatenate([np.array(l0, "int32"), np.array(l1, "int32")], axis=1)
        y_pred = np.concatenate([_la(HAND_PRED, 5, 5), _la([(0, 0)], 5, 5)], axis=0)
        metric = KeypointMatchMetric("precision")
        metric.update_state(y, y_pred, sample_weight=np.array([2.0, 1.0], "float32"))
        assert float(metric.result()) == pytest.approx(3.0 / 5.0)

    def test_masks_keep_padded_slots_from_competing(self):
        # padded slot j=2 has score 0.99 against real i=0 (unmasked model output), the
        # real pair (0,0) scores 0.9. Without masks i=0 prefers padded j=2 and recall
        # is 0; with mask1 = [1,1,0] the real pair wins.
        p = np.full((4, 4), LOW, "float32")
        p[0, 0], p[0, 2] = 0.9, 0.99
        y_pred = np.log(p)[None]
        y = _packed([0, -1, -2], [0, -1, -2])
        assert _run("recall", y, y_pred)[1] == 0.0
        mask = np.array([[1, 1, 0]], "float32")
        assert _run("recall", y, y_pred, mask0=mask, mask1=mask)[1] == 1.0


class TestEdges:
    def test_no_predictions_precision_zero_finite(self):
        _, v = _run("precision", _packed([0, -1], [0, -1]), _la([], 2, 2))
        assert v == 0.0 and np.isfinite(v)

    def test_no_ground_truth_recall_zero_finite(self):
        _, v = _run("recall", _packed([-1, -1], [-1, -1]), _la([(0, 0)], 2, 2))
        assert v == 0.0 and np.isfinite(v)

    def test_f1_empty_is_zero(self):
        _, v = _run("f1", _packed([-1, -1], [-1, -1]), _la([], 2, 2))
        assert v == 0.0 and np.isfinite(v)

    def test_fresh_metric_is_zero(self):
        assert float(KeypointMatchMetric("recall").result()) == 0.0

    def test_unknown_mode_and_bad_threshold(self):
        with pytest.raises(ValueError):
            KeypointMatchMetric("accuracy")
        with pytest.raises(ValueError):
            KeypointMatchMetric("recall", threshold=1.5)

    def test_wrong_rank_raises(self):
        with pytest.raises(ValueError):
            KeypointMatchMetric().update_state(np.zeros((1, 4), "int32"), np.zeros((1, 3), "float32"))


def _random_batch(rng, batch, m, n):
    la = np.log(rng.uniform(0.01, 1.0, (batch, m + 1, n + 1))).astype("float32")
    l0 = rng.randint(-2, n, (batch, m)).astype("int32")
    l1 = rng.randint(-2, m, (batch, n)).astype("int32")
    return np.concatenate([l0, l1], axis=1), la


class TestStatefulness:
    @pytest.mark.parametrize("mode", ["precision", "recall", "f1"])
    def test_two_updates_equal_one_concatenated(self, mode):
        rng = np.random.RandomState(3)
        y_a, la_a = _random_batch(rng, 3, 6, 5)
        y_b, la_b = _random_batch(rng, 2, 6, 5)
        two = KeypointMatchMetric(mode)
        two.update_state(y_a, la_a)
        two.update_state(y_b, la_b)
        one = KeypointMatchMetric(mode)
        one.update_state(np.concatenate([y_a, y_b]), np.concatenate([la_a, la_b]))
        assert float(two.result()) == pytest.approx(float(one.result()))
        for a, b in zip(two.variables, one.variables):
            assert float(a) == float(b)

    def test_reset_state(self):
        met, v = _run("precision", _packed(HAND_L0, HAND_L1), _la(HAND_PRED, 5, 5))
        assert v > 0
        met.reset_state()
        assert float(met.result()) == 0.0
        assert all(float(x) == 0.0 for x in met.variables)
        met.update_state(_packed(HAND_L0, HAND_L1), _la(HAND_PRED, 5, 5))
        assert float(met.result()) == pytest.approx(0.5)


class TestSerialization:
    def test_get_config_from_config(self):
        met = KeypointMatchMetric("recall", threshold=0.3, name="r")
        cfg = met.get_config()
        assert cfg["mode"] == "recall" and cfg["threshold"] == 0.3 and cfg["name"] == "r"
        clone = KeypointMatchMetric.from_config(cfg)
        assert clone.get_config() == cfg

    def test_keras_serialize_round_trip(self):
        met = KeypointMatchMetric("f1", threshold=0.25)
        clone = keras.saving.deserialize_keras_object(keras.saving.serialize_keras_object(met))
        assert isinstance(clone, KeypointMatchMetric)
        assert clone.get_config() == met.get_config()

    def test_default_name_follows_mode(self):
        assert KeypointMatchMetric("recall").name == "match_recall"


class TestModes:
    def test_graph_mode_matches_eager(self):
        rng = np.random.RandomState(5)
        y, la = _random_batch(rng, 4, 7, 6)
        eager, _ = _run("precision", y, la)
        graph = KeypointMatchMetric("precision")

        @tf.function
        def step(a, b):
            graph.update_state(a, b)

        step(tf.constant(y), tf.constant(la))
        step(tf.constant(y), tf.constant(la))
        eager.update_state(y, la)
        assert float(graph.result()) == pytest.approx(float(eager.result()))
        assert float(graph.predicted) == float(eager.predicted)

    def test_float16_predictions(self):
        y, la = _packed(HAND_L0, HAND_L1), _la(HAND_PRED, 5, 5)
        _, v32 = _run("precision", y, la)
        _, v16 = _run("precision", y, la.astype("float16"))
        assert v16 == pytest.approx(v32)

    def test_mixed_float16_policy(self):
        keras.mixed_precision.set_global_policy("mixed_float16")
        try:
            met = KeypointMatchMetric("recall")
            met.update_state(_packed(HAND_L0, HAND_L1).astype("float16"),
                             tf.constant(_la(HAND_PRED, 5, 5), "float16"))
            assert float(met.result()) == pytest.approx(0.25)
            assert met.true_positives.dtype == "float32"
        finally:
            keras.mixed_precision.set_global_policy("float32")


class TestStockFit:
    """Probe result (decisions.md D-013): the dict-keyed compile route surfaces the metric."""

    def _setup(self):
        from tests.test_models.test_lightglue.weight_loading import random_inputs

        d, m, n, batch = 16, 8, 7, 4
        rng = np.random.RandomState(0)
        data = random_inputs(rng, batch, m, n, d)
        l0 = -np.ones((batch, m), "int32")
        l1 = -np.ones((batch, n), "int32")
        for s in range(batch):                       # 3 positives per image
            for k in range(3):
                l0[s, k], l1[s, k] = k, k
        model = LightGlue(input_dim=d, descriptor_dim=d, num_layers=2, num_heads=2)
        model(data)
        return model, data, np.asarray(pack_matches(l0, l1))

    def test_fit_and_evaluate_report_metrics(self):
        model, data, y = self._setup()
        prec = KeypointMatchMetric("precision", name="p")
        rec = KeypointMatchMetric("recall", name="r")
        model.compile(optimizer=keras.optimizers.SGD(0.1),
                      loss={"log_assignments": LightGlueLoss()},
                      metrics={"log_assignments": [prec, rec]}, jit_compile=False)
        hist = model.fit(data, {"log_assignments": y}, batch_size=2, epochs=1, verbose=0)
        assert "log_assignments_p" in hist.history and "log_assignments_r" in hist.history
        assert np.isfinite(hist.history["log_assignments_r"][0])
        # wiring check: every image contributes 3 ground-truth positives, 4 images
        assert float(rec.ground_truth) == 12.0
        ev = model.evaluate(data, {"log_assignments": y}, batch_size=2, verbose=0, return_dict=True)
        assert 0.0 <= ev["log_assignments_r"] <= 1.0

    def test_list_metrics_on_multi_output_model_is_rejected(self):
        # documents the probe: a bare list does not work against the dict-output model
        model, data, y = self._setup()
        model.compile(optimizer="sgd", loss={"log_assignments": LightGlueLoss()},
                      metrics=[KeypointMatchMetric()], jit_compile=False)
        with pytest.raises(ValueError, match="as many entries"):
            model.fit(data, {"log_assignments": y}, batch_size=2, epochs=1, verbose=0)

    def test_metric_attribute_updated_in_call_surfaces_in_fit_logs(self):
        class Holder(keras.Model):
            def __init__(self, **kw):
                super().__init__(**kw)
                self.dense = keras.layers.Dense(1)
                self.m = keras.metrics.Mean(name="inner")

            def call(self, x, training=None):
                out = self.dense(x)
                self.m.update_state(7.0 + 0.0 * keras.ops.mean(out))
                return out

        model = Holder()
        model.compile(optimizer="sgd", loss="mse", jit_compile=False)
        hist = model.fit(np.ones((8, 3), "float32"), np.ones((8, 1), "float32"),
                         batch_size=4, epochs=1, verbose=0)
        assert hist.history["inner"] == [7.0]
