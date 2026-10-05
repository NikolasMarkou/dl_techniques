"""Tests for the DeepSORT appearance net and the tracking runtime."""

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.models.vision.deepsort import (
    CosineClassifier,
    DeepSortAppearanceNet,
    DeepSortResidualBlock,
    DeepSortTracker,
    Detection,
    KalmanFilter,
    NearestNeighborDistanceMetric,
    TrackState,
    create_deepsort_embedding,
    gate_cost_matrix,
    iou,
    iou_cost,
    matching_cascade,
    min_cost_matching,
)


def _unit_vectors(n: int, dim: int = 128, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vectors = rng.random((n, dim)).astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


class TestAppearanceNet:
    def test_forward_trunk_shape(self):
        model = create_deepsort_embedding()
        out = model(np.zeros((2, 128, 64, 3), dtype="float32"), training=False)
        assert tuple(out.shape) == (2, 128)

    def test_forward_top_shapes(self):
        model = create_deepsort_embedding(include_top=True, num_classes=10)
        features, logits = model(
            np.zeros((2, 128, 64, 3), dtype="float32"), training=False
        )
        assert tuple(features.shape) == (2, 128)
        assert tuple(logits.shape) == (2, 10)

    def test_descriptors_are_unit_norm_in_training_mode(self):
        # At random weights inference-mode activations vanish (transcribed
        # 1e-3 init + fresh BN stats); batch statistics rescue them, so the
        # unit-norm property holds in training mode. See the module note.
        model = create_deepsort_embedding()
        rng = np.random.default_rng(0)
        x = rng.random((4, 128, 64, 3)).astype("float32")
        norms = np.linalg.norm(
            np.asarray(model(x, training=True)), axis=1
        )
        np.testing.assert_allclose(norms, 1.0, atol=1e-5, rtol=0)

    def test_top_requires_num_classes(self):
        with pytest.raises(ValueError):
            DeepSortAppearanceNet(include_top=True)

    def test_invalid_args_raise(self):
        with pytest.raises(ValueError):
            DeepSortResidualBlock(filters=0)
        with pytest.raises(ValueError):
            DeepSortResidualBlock(filters=32, stride=3)
        with pytest.raises(ValueError):
            CosineClassifier(num_classes=0)

    def test_explicit_build_matches_lazy_build(self):
        def _relative(m):
            return sorted(w.path.split("/", 1)[-1] for w in m.weights)

        explicit = create_deepsort_embedding(include_top=True, num_classes=5)
        explicit.build((None, 128, 64, 3))
        lazy = create_deepsort_embedding(include_top=True, num_classes=5)
        lazy(np.zeros((1, 128, 64, 3), dtype="float32"))
        assert _relative(explicit) == _relative(lazy)
        assert len(explicit.weights) > 0

    def test_no_top_builds_no_head_weights(self):
        model = create_deepsort_embedding(include_top=False)
        model(np.zeros((1, 128, 64, 3), dtype="float32"))
        assert not [w for w in model.weights if "cosine_head" in w.path]

    def test_no_projection_builds_no_projection_weights(self):
        block = DeepSortResidualBlock(32, use_projection=False, name="blk")
        block.build((None, 32, 32, 32))
        assert not [w for w in block.weights if "projection" in w.path]
        assert [w for w in block.weights if "conv1" in w.path]

    def test_serialization_round_trip_values(self, tmp_path):
        model = create_deepsort_embedding()
        x = np.ones((1, 128, 64, 3), dtype="float32")
        original = ops.convert_to_numpy(model(x, training=False))
        path = str(tmp_path / "deepsort.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        restored = ops.convert_to_numpy(loaded(x, training=False))
        np.testing.assert_allclose(original, restored, atol=0.0, rtol=0)

    def test_get_config_round_trip(self):
        model = create_deepsort_embedding(include_top=True, num_classes=7)
        rebuilt = DeepSortAppearanceNet.from_config(dict(model.get_config()))
        assert rebuilt.include_top and rebuilt.num_classes == 7

    def test_pretrained_true_raises(self):
        with pytest.raises(NotImplementedError):
            create_deepsort_embedding(pretrained=True)

    def test_gradient_flows_to_trunk(self):
        import tensorflow as tf

        model = create_deepsort_embedding()
        x = tf.zeros((1, 128, 64, 3))
        with tf.GradientTape() as tape:
            loss = ops.sum(model(x, training=True))
        grads = tape.gradient(loss, model.trainable_variables)
        assert any(g is not None for g in grads)


class TestKalmanFilter:
    def test_initiate_shape_and_zero_velocity(self):
        kf = KalmanFilter()
        mean, cov = kf.initiate(np.array([10.0, 20.0, 1.0, 40.0]))
        assert mean.shape == (8,) and cov.shape == (8, 8)
        np.testing.assert_allclose(mean[4:], 0.0, atol=0, rtol=0)

    def test_predict_advances_by_velocity(self):
        kf = KalmanFilter()
        mean, cov = kf.initiate(np.array([10.0, 20.0, 1.0, 40.0]))
        mean[4] = 5.0  # vx
        predicted, _ = kf.predict(mean, cov)
        assert predicted[0] == pytest.approx(15.0)

    def test_update_pulls_toward_measurement(self):
        kf = KalmanFilter()
        mean, cov = kf.initiate(np.array([10.0, 20.0, 1.0, 40.0]))
        mean, cov = kf.predict(mean, cov)
        updated, _ = kf.update(mean, cov, np.array([30.0, 20.0, 1.0, 40.0]))
        assert 10.0 < updated[0] < 30.0

    def test_gating_distance_nonnegative(self):
        kf = KalmanFilter()
        mean, cov = kf.initiate(np.array([10.0, 20.0, 1.0, 40.0]))
        d = kf.gating_distance(
            mean, cov, np.array([[10.0, 20.0, 1.0, 40.0], [500.0, 500.0, 1.0, 40.0]])
        )
        assert d.shape == (2,)
        assert bool(np.all(d >= 0.0))
        assert d[1] > d[0]

    def test_initiate_predict_update_matches_transcribed_oracle(self):
        # Pinned outputs of the reference implementation on a fixed input
        # (differential run: initiate/predict/update/gating bit-identical
        # over 5 randomized trials). Any drift in the transcription fails here.
        kf = KalmanFilter()
        mean, cov = kf.initiate(np.array([10.0, 20.0, 1.0, 40.0]))
        mean, cov = kf.predict(mean, cov)
        mean, cov = kf.update(mean, cov, np.array([12.0, 19.0, 1.0, 41.0]))
        np.testing.assert_allclose(
            mean,
            [11.7355371901, 19.1322314050, 1.0, 40.8677685950, 0.4132231405,
             -0.2066115702, 0.0, 0.2066115702],
            atol=1e-9, rtol=0,
        )
        np.testing.assert_allclose(
            np.diag(cov),
            [3.4710743802, 3.4710743802, 0.0001960785, 3.4710743802,
             5.0211776860, 5.0211776860, 0.0, 5.0211776860],
            atol=1e-9, rtol=0,
        )
        gate = kf.gating_distance(
            mean, cov, np.array([[12.0, 19.0, 1.0, 41.0]])
        )
        np.testing.assert_allclose(gate, [0.0137200968], atol=1e-9, rtol=0)


class TestMatching:
    def test_iou_known_values(self):
        box = np.array([0.0, 0.0, 10.0, 10.0])
        cands = np.array([[0.0, 0.0, 10.0, 10.0], [5.0, 0.0, 10.0, 10.0]])
        out = iou(box, cands)
        np.testing.assert_allclose(out[0], 1.0, atol=1e-9, rtol=0)
        np.testing.assert_allclose(out[1], 50.0 / 150.0, atol=1e-9, rtol=0)

    def test_gallery_budget_evicts_oldest(self):
        metric = NearestNeighborDistanceMetric("cosine", 0.2, budget=2)
        feats = _unit_vectors(3)
        metric.partial_fit(feats, np.array([1, 1, 1]), [1])
        assert len(metric.samples[1]) == 2
        # A kept sample matches itself exactly.
        d = metric.distance(feats[2:3], [1])
        np.testing.assert_allclose(d[0, 0], 0.0, atol=1e-6, rtol=0)

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError):
            NearestNeighborDistanceMetric("manhattan", 0.2)

    def test_gate_cost_matrix_infinities_far_pairs(self):
        from dl_techniques.models.vision.deepsort import Track

        kf = KalmanFilter()
        mean, cov = kf.initiate(np.array([10.0, 20.0, 1.0, 40.0]))
        track = Track(mean, cov, 1, n_init=3, max_age=30)
        near = Detection(np.array([-10.0, 0.0, 40.0, 40.0]), 0.9, _unit_vectors(1)[0])
        far = Detection(np.array([900.0, 900.0, 40.0, 40.0]), 0.9, _unit_vectors(1)[0])
        cost = np.zeros((1, 2))
        out = gate_cost_matrix(kf, cost, [track], [near, far], [0], [0, 1])
        assert out[0, 0] == 0.0
        assert out[0, 1] == 1e5  # chi-square gated to INFTY_COST

    def test_cascade_prefers_recent_tracks(self):
        # After prediction, track A was seen last frame (time_since_update 1)
        # and track B missed one (time_since_update 2). One detection: the
        # cascade matches A at level 0 before B is even considered.
        kf = KalmanFilter()
        from dl_techniques.models.vision.deepsort import Track

        mean_a, cov_a = kf.initiate(np.array([0.0, 0.0, 1.0, 10.0]))
        mean_b, cov_b = kf.initiate(np.array([0.0, 0.0, 1.0, 10.0]))
        recent = Track(mean_a, cov_a, 1, n_init=3, max_age=30)
        recent.predict(kf)
        stale = Track(mean_b, cov_b, 2, n_init=3, max_age=30)
        stale.predict(kf)
        stale.predict(kf)
        assert recent.time_since_update == 1 and stale.time_since_update == 2
        det = Detection(
            np.array([0.0, 0.0, 10.0, 10.0]), 0.9, _unit_vectors(1, seed=3)[0]
        )

        def zero_metric(tracks, dets, rows, cols):
            return np.zeros((len(rows), len(cols)))

        matches, unmatched, _ = matching_cascade(
            zero_metric, 10.0, 5, [recent, stale], [det]
        )
        assert (0, 0) in matches
        assert 1 in unmatched


class TestTrackerEndToEnd:
    def _tracker(self, **kwargs):
        metric = NearestNeighborDistanceMetric("cosine", 0.2, 100)
        return DeepSortTracker(metric, **kwargs)

    def _detections(self, boxes, feats):
        return [
            Detection(np.asarray(b, dtype=float), 0.9, f)
            for b, f in zip(boxes, feats)
        ]

    def test_two_targets_keep_stable_ids(self):
        tracker = self._tracker(n_init=2)
        feats = _unit_vectors(2, seed=11)
        seen = []
        for t in range(6):
            boxes = [
                [50 + 4 * t, 100, 40, 80],   # target A drifts right
                [400 - 3 * t, 300, 40, 80],  # target B drifts left
            ]
            tracker.predict()
            tracker.update(self._detections(boxes, feats))
            confirmed = sorted(
                (tr.track_id for tr in tracker.tracks if tr.is_confirmed())
            )
            seen.append(confirmed)
        assert seen[-1] == [1, 2]
        # IDs never switch once both are confirmed.
        assert all(s == [1, 2] for s in seen[2:])

    def test_short_gap_keeps_id_long_gap_reassigns(self):
        box = [[200, 200, 40, 80]]
        feat = _unit_vectors(1, seed=21)
        tracker = self._tracker(n_init=1, max_age=2)
        for _ in range(2):
            tracker.predict()
            tracker.update(self._detections(box, feat))
        first_id = [t.track_id for t in tracker.tracks if t.is_confirmed()]
        assert first_id == [1]
        # One missed frame: within max_age, the same track re-acquires.
        tracker.predict()
        tracker.update([])
        tracker.predict()
        tracker.update(self._detections(box, feat))
        assert [t.track_id for t in tracker.tracks if t.is_confirmed()] == [1]
        # Three missed frames: over max_age, a fresh identity starts.
        for _ in range(3):
            tracker.predict()
            tracker.update([])
        tracker.predict()
        tracker.update(self._detections(box, feat))
        assert [t.track_id for t in tracker.tracks] == [2]
