import numpy as np
import pytest
import keras

from dl_techniques.metrics.perplexity_metric import Perplexity, perplexity


class TestPerplexity:
    """Tests for Perplexity metric."""

    def test_init_default(self):
        metric = Perplexity()
        assert metric.name == "perplexity"
        assert metric.from_logits is True
        assert metric.ignore_class is None

    def test_init_custom(self):
        metric = Perplexity(from_logits=False, ignore_class=0, name="ppl")
        assert metric.from_logits is False
        assert metric.ignore_class == 0
        assert metric.name == "ppl"

    def test_perfect_prediction_from_logits(self):
        """Perfect logits (very high for correct class) should give perplexity ≈ 1."""
        metric = Perplexity(from_logits=True)
        # 2 samples, 4 classes
        y_true = np.array([0, 2], dtype="int32")
        # Very confident correct predictions
        y_pred = np.array([
            [100.0, -100.0, -100.0, -100.0],
            [-100.0, -100.0, 100.0, -100.0],
        ], dtype="float32")

        metric.update_state(y_true, y_pred)
        result = float(metric.result())
        assert abs(result - 1.0) < 0.1

    def test_uniform_prediction(self):
        """Uniform predictions over V classes should give perplexity ≈ V."""
        num_classes = 4
        metric = Perplexity(from_logits=True)
        y_true = np.array([0, 1, 2, 3], dtype="int32")
        # All logits equal → uniform distribution
        y_pred = np.zeros((4, num_classes), dtype="float32")

        metric.update_state(y_true, y_pred)
        result = float(metric.result())
        assert abs(result - num_classes) < 0.5

    def test_from_probabilities(self):
        """Test with probability inputs (from_logits=False)."""
        metric = Perplexity(from_logits=False)
        y_true = np.array([0], dtype="int32")
        y_pred = np.array([[1.0, 0.0, 0.0]], dtype="float32")

        metric.update_state(y_true, y_pred)
        result = float(metric.result())
        assert abs(result - 1.0) < 0.1

    def test_ignore_class(self):
        """Tokens with ignore_class should not affect perplexity."""
        metric = Perplexity(from_logits=True, ignore_class=-100)
        # First token is real, second should be ignored
        y_true = np.array([0, -100], dtype="int32")
        y_pred = np.array([
            [100.0, -100.0, -100.0],  # perfect
            [-100.0, -100.0, -100.0],  # ignored
        ], dtype="float32")

        metric.update_state(y_true, y_pred)
        result = float(metric.result())
        assert abs(result - 1.0) < 0.1

    def test_perplexity_always_gte_1(self):
        """Perplexity should always be >= 1."""
        metric = Perplexity(from_logits=True)
        y_true = np.array([0, 1, 2], dtype="int32")
        y_pred = np.random.randn(3, 5).astype("float32")

        metric.update_state(y_true, y_pred)
        result = float(metric.result())
        assert result >= 1.0 - 1e-3

    def test_reset_state(self):
        metric = Perplexity(from_logits=True)
        y_true = np.array([0], dtype="int32")
        y_pred = np.array([[1.0, 0.0, 0.0]], dtype="float32")

        metric.update_state(y_true, y_pred)
        metric.reset_state()
        # exp(0/0) via divide_no_nan → exp(0) = 1
        result = float(metric.result())
        assert abs(result - 1.0) < 1e-5

    def test_accumulation(self):
        metric = Perplexity(from_logits=True)
        y_true1 = np.array([0], dtype="int32")
        y_pred1 = np.array([[10.0, -10.0, -10.0]], dtype="float32")
        y_true2 = np.array([1], dtype="int32")
        y_pred2 = np.array([[-10.0, 10.0, -10.0]], dtype="float32")

        metric.update_state(y_true1, y_pred1)
        metric.update_state(y_true2, y_pred2)
        result = float(metric.result())
        assert abs(result - 1.0) < 0.1

    def test_get_config_and_from_config(self):
        metric = Perplexity(from_logits=False, ignore_class=0, name="test_ppl")
        config = metric.get_config()
        assert config["from_logits"] is False
        assert config["ignore_class"] == 0
        assert config["name"] == "test_ppl"

        restored = Perplexity.from_config(config)
        assert restored.from_logits is False
        assert restored.ignore_class == 0

    def test_serialization_round_trip(self):
        metric = Perplexity(from_logits=True)
        config = metric.get_config()
        restored = Perplexity.from_config(config)

        y_true = np.array([0, 1], dtype="int32")
        y_pred = np.random.randn(2, 4).astype("float32")

        metric.update_state(y_true, y_pred)
        restored.update_state(y_true, y_pred)
        assert abs(float(metric.result()) - float(restored.result())) < 1e-5


class TestPerplexityFunction:
    """Tests for the functional perplexity interface."""

    def test_perfect_prediction(self):
        y_true = np.array([0, 1], dtype="int32")
        y_pred = np.array([
            [100.0, -100.0, -100.0],
            [-100.0, 100.0, -100.0],
        ], dtype="float32")

        result = float(perplexity(y_true, y_pred, from_logits=True))
        assert abs(result - 1.0) < 0.1

    def test_uniform_prediction(self):
        num_classes = 4
        y_true = np.array([0, 1, 2, 3], dtype="int32")
        y_pred = np.zeros((4, num_classes), dtype="float32")

        result = float(perplexity(y_true, y_pred, from_logits=True))
        assert abs(result - num_classes) < 0.5

    def test_ignore_class(self):
        y_true = np.array([0, -100], dtype="int32")
        y_pred = np.array([
            [100.0, -100.0, -100.0],
            [-100.0, -100.0, -100.0],
        ], dtype="float32")

        result = float(perplexity(y_true, y_pred, from_logits=True, ignore_class=-100))
        assert abs(result - 1.0) < 0.1
