import numpy as np
import pytest
import keras

from dl_techniques.metrics.hrm_metrics import HRMMetrics


class TestHRMMetrics:
    """Tests for HRMMetrics aggregator."""

    def _make_inputs(
        self,
        batch_size=4,
        seq_len=8,
        vocab_size=100,
        all_correct=False,
        halted=None,
        steps=None,
    ):
        """Create synthetic inputs for HRMMetrics.

        Returns (y_true, y_pred) dicts.
        """
        ignore_index = -100
        # Random labels with some ignored positions
        labels = np.random.randint(0, vocab_size, (batch_size, seq_len)).astype("int32")
        # Mask out last 2 positions
        labels[:, -2:] = ignore_index

        if all_correct:
            # Make logits that predict the correct labels
            logits = np.full((batch_size, seq_len, vocab_size), -100.0, dtype="float32")
            for b in range(batch_size):
                for s in range(seq_len):
                    if labels[b, s] != ignore_index:
                        logits[b, s, labels[b, s]] = 100.0
        else:
            logits = np.random.randn(batch_size, seq_len, vocab_size).astype("float32")

        # Q-halt logits: positive means "predict correct"
        q_halt_logits = np.random.randn(batch_size).astype("float32")

        y_true = {"labels": labels}
        if halted is not None:
            y_true["halted"] = halted
        if steps is not None:
            y_true["steps"] = steps

        y_pred = {"logits": logits, "q_halt_logits": q_halt_logits}
        return y_true, y_pred

    def test_init(self):
        metrics = HRMMetrics()
        assert metrics.ignore_index == -100

    def test_init_custom_ignore(self):
        metrics = HRMMetrics(ignore_index=-1)
        assert metrics.ignore_index == -1

    def test_update_and_result_basic(self):
        metrics = HRMMetrics()
        y_true, y_pred = self._make_inputs()
        metrics.update_state(y_true, y_pred)
        result = metrics.result()

        assert isinstance(result, dict)
        assert "accuracy" in result
        assert "exact_accuracy" in result
        assert "q_halt_accuracy" in result
        assert "avg_steps" in result
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_perfect_accuracy(self):
        metrics = HRMMetrics()
        y_true, y_pred = self._make_inputs(all_correct=True)
        metrics.update_state(y_true, y_pred)
        result = metrics.result()

        assert abs(result["accuracy"] - 1.0) < 1e-5
        assert abs(result["exact_accuracy"] - 1.0) < 1e-5

    def test_with_halted_mask(self):
        metrics = HRMMetrics()
        halted = np.array([True, True, False, False])
        y_true, y_pred = self._make_inputs(halted=halted)
        metrics.update_state(y_true, y_pred)
        result = metrics.result()

        assert isinstance(result, dict)
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_with_steps(self):
        metrics = HRMMetrics()
        halted = np.array([True, True, True, True])
        steps = np.array([3, 5, 2, 4], dtype="int32")
        y_true, y_pred = self._make_inputs(halted=halted, steps=steps)
        metrics.update_state(y_true, y_pred)
        result = metrics.result()

        assert result["avg_steps"] > 0.0

    def test_reset_state(self):
        metrics = HRMMetrics()
        y_true, y_pred = self._make_inputs()
        metrics.update_state(y_true, y_pred)
        metrics.reset_state()
        result = metrics.result()

        assert result["accuracy"] == 0.0
        assert result["exact_accuracy"] == 0.0

    def test_accumulation(self):
        metrics = HRMMetrics()
        y_true1, y_pred1 = self._make_inputs(batch_size=2, all_correct=True)
        y_true2, y_pred2 = self._make_inputs(batch_size=2, all_correct=False)

        metrics.update_state(y_true1, y_pred1)
        metrics.update_state(y_true2, y_pred2)
        result = metrics.result()

        # Mix of perfect and random → accuracy between 0 and 1
        assert 0.0 < result["accuracy"] <= 1.0

    def test_get_config(self):
        metrics = HRMMetrics(ignore_index=-1)
        config = metrics.get_config()
        assert config["ignore_index"] == -1

    def test_recreate_from_config(self):
        metrics = HRMMetrics(ignore_index=-1)
        config = metrics.get_config()
        restored = HRMMetrics(**config)
        assert restored.ignore_index == -1
