import keras
from typing import Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


class HRMMetrics:
    """
    Metrics for Hierarchical Reasoning Model.

    Computes accuracy, exact accuracy, Q-learning metrics, and step statistics.
    """

    def __init__(self, ignore_index: int = -100):
        self.ignore_index = ignore_index

        # Initialize metrics
        self.accuracy = keras.metrics.Mean(name="accuracy")
        self.exact_accuracy = keras.metrics.Mean(name="exact_accuracy")
        self.q_halt_accuracy = keras.metrics.Mean(name="q_halt_accuracy")
        self.avg_steps = keras.metrics.Mean(name="avg_steps")
        self.lm_loss = keras.metrics.Mean(name="lm_loss")
        self.q_halt_loss = keras.metrics.Mean(name="q_halt_loss")
        self.q_continue_loss = keras.metrics.Mean(name="q_continue_loss")

    def update_state(self, y_true: Dict, y_pred: Dict):
        """Update metric states."""
        # Extract components
        labels = y_true["labels"]
        halted = y_true.get("halted", None)
        steps = y_true.get("steps", None)

        logits = y_pred["logits"]
        q_halt_logits = y_pred["q_halt_logits"]

        # Token-level accuracy
        pred_labels = keras.ops.argmax(logits, axis=-1)
        valid_mask = keras.ops.not_equal(labels, self.ignore_index)
        correct_tokens = valid_mask & keras.ops.equal(pred_labels, labels)

        # Compute per-sequence metrics
        valid_counts = keras.ops.sum(keras.ops.cast(valid_mask, "float32"), axis=-1)
        valid_counts = keras.ops.maximum(valid_counts, 1.0)

        token_accuracy = (
                keras.ops.sum(
                    keras.ops.cast(correct_tokens, "float32"), axis=-1) / valid_counts)

        # Exact sequence accuracy
        seq_correct = keras.ops.equal(
            keras.ops.sum(keras.ops.cast(correct_tokens, "float32"), axis=-1),
            valid_counts
        )

        # Q-halt accuracy (predicting sequence correctness)
        q_halt_pred = keras.ops.greater_equal(q_halt_logits, 0.0)
        q_halt_correct = keras.ops.equal(q_halt_pred, seq_correct)

        # Update metrics (only for halted sequences if available)
        if halted is not None:
            valid_mask_seq = halted & keras.ops.greater(valid_counts, 0)
            if keras.ops.any(valid_mask_seq):
                self.accuracy.update_state(token_accuracy, sample_weight=keras.ops.cast(valid_mask_seq, "float32"))
                self.exact_accuracy.update_state(keras.ops.cast(seq_correct, "float32"),
                                                 sample_weight=keras.ops.cast(valid_mask_seq, "float32"))
                self.q_halt_accuracy.update_state(keras.ops.cast(q_halt_correct, "float32"),
                                                  sample_weight=keras.ops.cast(valid_mask_seq, "float32"))

                if steps is not None:
                    self.avg_steps.update_state(keras.ops.cast(steps, "float32"),
                                                sample_weight=keras.ops.cast(valid_mask_seq, "float32"))
        else:
            # Update for all sequences
            valid_mask_seq = keras.ops.greater(valid_counts, 0)
            if keras.ops.any(valid_mask_seq):
                self.accuracy.update_state(token_accuracy, sample_weight=keras.ops.cast(valid_mask_seq, "float32"))
                self.exact_accuracy.update_state(keras.ops.cast(seq_correct, "float32"),
                                                 sample_weight=keras.ops.cast(valid_mask_seq, "float32"))
                self.q_halt_accuracy.update_state(keras.ops.cast(q_halt_correct, "float32"),
                                                  sample_weight=keras.ops.cast(valid_mask_seq, "float32"))

    def result(self) -> Dict[str, float]:
        """Get current metric results."""
        return {
            "accuracy": float(self.accuracy.result()),
            "exact_accuracy": float(self.exact_accuracy.result()),
            "q_halt_accuracy": float(self.q_halt_accuracy.result()),
            "avg_steps": float(self.avg_steps.result()),
        }

    def reset_state(self):
        """Reset all metrics."""
        self.accuracy.reset_state()
        self.exact_accuracy.reset_state()
        self.q_halt_accuracy.reset_state()
        self.avg_steps.reset_state()
        self.lm_loss.reset_state()
        self.q_halt_loss.reset_state()
        self.q_continue_loss.reset_state()

# ---------------------------------------------------------------------
