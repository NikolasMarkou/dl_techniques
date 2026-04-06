import numpy as np
import pytest
import keras

from dl_techniques.metrics.multi_label_metrics import MultiLabelMetrics


class TestMultiLabelMetrics:
    """Tests for MultiLabelMetrics."""

    def test_init(self):
        metric = MultiLabelMetrics(num_classes=5)
        assert metric.num_classes == 5
        assert metric.threshold == 0.5
        assert metric.exclude_background is False

    def test_perfect_predictions(self):
        metric = MultiLabelMetrics(num_classes=3)
        y_true = np.array([[1, 0, 1], [0, 1, 0]], dtype="float32")
        y_pred = np.array([[0.9, 0.1, 0.8], [0.1, 0.9, 0.2]], dtype="float32")

        metric.update_state(y_true, y_pred)
        f1 = float(metric.result())
        assert abs(f1 - 1.0) < 1e-5

    def test_worst_predictions(self):
        metric = MultiLabelMetrics(num_classes=2)
        # All predictions are opposite of ground truth
        y_true = np.array([[1, 0], [0, 1]], dtype="float32")
        y_pred = np.array([[0.1, 0.9], [0.9, 0.1]], dtype="float32")

        metric.update_state(y_true, y_pred)
        f1 = float(metric.result())
        assert f1 < 0.01  # Should be near zero

    def test_precision_recall_f1(self):
        metric = MultiLabelMetrics(num_classes=2)
        # y_true=[[1,1],[0,1]], y_pred thresholded=[[1,1],[1,0]]
        # Class 0: TP=1, FP=1, FN=0 → precision=0.5, recall=1.0
        # Class 1: TP=1, FP=0, FN=1 → precision=1.0, recall=0.5
        y_true = np.array([[1, 1], [0, 1]], dtype="float32")
        y_pred = np.array([[0.9, 0.9], [0.9, 0.1]], dtype="float32")

        metric.update_state(y_true, y_pred)

        precision = keras.ops.convert_to_numpy(metric.compute_precision())
        recall = keras.ops.convert_to_numpy(metric.compute_recall())
        f1 = keras.ops.convert_to_numpy(metric.compute_f1())

        assert abs(precision[0] - 0.5) < 1e-5
        assert abs(precision[1] - 1.0) < 1e-5
        assert abs(recall[0] - 1.0) < 1e-5
        assert abs(recall[1] - 0.5) < 1e-5
        # F1 = 2 * P * R / (P + R)
        assert abs(f1[0] - 2.0 * 0.5 * 1.0 / 1.5) < 1e-4
        assert abs(f1[1] - 2.0 * 1.0 * 0.5 / 1.5) < 1e-4

    def test_exclude_background(self):
        metric_with_bg = MultiLabelMetrics(num_classes=3, exclude_background=False)
        metric_no_bg = MultiLabelMetrics(num_classes=3, exclude_background=True)

        y_true = np.array([[1, 0, 1], [0, 1, 0]], dtype="float32")
        y_pred = np.array([[0.9, 0.1, 0.8], [0.1, 0.9, 0.2]], dtype="float32")

        metric_with_bg.update_state(y_true, y_pred)
        metric_no_bg.update_state(y_true, y_pred)

        # Both should work but may differ
        f1_with_bg = float(metric_with_bg.result())
        f1_no_bg = float(metric_no_bg.result())
        assert 0.0 <= f1_with_bg <= 1.0
        assert 0.0 <= f1_no_bg <= 1.0

    def test_sample_weight_unweighted_unchanged(self):
        """Verify that uniform weights produce the same result as no weights."""
        metric_no_weight = MultiLabelMetrics(num_classes=3)
        metric_uniform = MultiLabelMetrics(num_classes=3)

        y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]], dtype="float32")
        y_pred = np.array([[0.9, 0.1, 0.8], [0.1, 0.9, 0.2], [0.8, 0.7, 0.1]], dtype="float32")

        metric_no_weight.update_state(y_true, y_pred)
        metric_uniform.update_state(y_true, y_pred, sample_weight=np.ones(3, dtype="float32"))

        assert abs(float(metric_no_weight.result()) - float(metric_uniform.result())) < 1e-5

    def test_sample_weight_correctness(self):
        """Verify that zero-weight samples are effectively excluded."""
        metric = MultiLabelMetrics(num_classes=2)

        # Two samples: first is perfect for both classes, second is terrible
        y_true = np.array([[1, 1], [0, 0]], dtype="float32")
        y_pred = np.array([[0.9, 0.9], [0.9, 0.9]], dtype="float32")
        # Zero weight on the bad sample → only the perfect sample counts
        weights = np.array([1.0, 0.0], dtype="float32")

        metric.update_state(y_true, y_pred, sample_weight=weights)

        # With only sample 1 (weight=1): TP=[1,1], FP=[0,0], FN=[0,0]
        # precision=1.0, recall=1.0, F1=1.0 for both classes
        f1 = float(metric.result())
        assert abs(f1 - 1.0) < 1e-4

    def test_reset_state(self):
        metric = MultiLabelMetrics(num_classes=3)
        y_true = np.array([[1, 0, 1]], dtype="float32")
        y_pred = np.array([[0.9, 0.1, 0.8]], dtype="float32")

        metric.update_state(y_true, y_pred)
        metric.reset_state()

        tp = keras.ops.convert_to_numpy(metric.true_positives)
        assert np.all(tp == 0.0)

    def test_accumulation(self):
        metric = MultiLabelMetrics(num_classes=2)
        y_true1 = np.array([[1, 0]], dtype="float32")
        y_pred1 = np.array([[0.9, 0.1]], dtype="float32")
        y_true2 = np.array([[0, 1]], dtype="float32")
        y_pred2 = np.array([[0.1, 0.9]], dtype="float32")

        metric.update_state(y_true1, y_pred1)
        metric.update_state(y_true2, y_pred2)
        f1 = float(metric.result())
        assert abs(f1 - 1.0) < 1e-5

    def test_get_config_and_from_config(self):
        metric = MultiLabelMetrics(
            num_classes=5,
            threshold=0.3,
            exclude_background=True,
            epsilon=1e-8,
            name="test_ml",
        )
        config = metric.get_config()
        assert config["num_classes"] == 5
        assert config["threshold"] == 0.3
        assert config["exclude_background"] is True
        assert config["epsilon"] == 1e-8

        restored = MultiLabelMetrics.from_config(config)
        assert restored.num_classes == 5
        assert restored.threshold == 0.3
        assert restored.exclude_background is True

    def test_serialization_round_trip(self):
        metric = MultiLabelMetrics(num_classes=3)
        config = metric.get_config()
        restored = MultiLabelMetrics.from_config(config)

        y_true = np.array([[1, 0, 1], [0, 1, 0]], dtype="float32")
        y_pred = np.array([[0.9, 0.1, 0.8], [0.1, 0.9, 0.2]], dtype="float32")

        metric.update_state(y_true, y_pred)
        restored.update_state(y_true, y_pred)
        assert abs(float(metric.result()) - float(restored.result())) < 1e-6
