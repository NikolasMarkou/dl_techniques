import keras
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------


class HRMMetrics:
    """Metrics aggregator for Hierarchical Reasoning Model.

    Computes accuracy, exact accuracy, Q-learning metrics, and step statistics.
    This is a standalone metrics container for custom training loops, not a
    Keras Metric subclass (it returns a dict from result(), not a scalar tensor).

    Args:
        ignore_index: Token index to ignore in accuracy computation.
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

    def update_state(self, y_true: Dict[str, Any], y_pred: Dict[str, Any]):
        """Update metric states.

        Args:
            y_true: Dictionary with keys 'labels', optionally 'halted' and 'steps'.
            y_pred: Dictionary with keys 'logits' and 'q_halt_logits'.
        """
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
        valid_counts = keras.ops.sum(
            keras.ops.cast(valid_mask, "float32"), axis=-1
        )
        valid_counts = keras.ops.maximum(valid_counts, 1.0)

        token_accuracy = (
            keras.ops.sum(
                keras.ops.cast(correct_tokens, "float32"), axis=-1
            )
            / valid_counts
        )

        # Exact sequence accuracy
        seq_correct = keras.ops.equal(
            keras.ops.sum(keras.ops.cast(correct_tokens, "float32"), axis=-1),
            valid_counts,
        )

        # Q-halt accuracy (predicting sequence correctness)
        q_halt_pred = keras.ops.greater_equal(q_halt_logits, 0.0)
        q_halt_correct = keras.ops.equal(q_halt_pred, seq_correct)

        # Build sequence-level validity mask
        valid_mask_seq = keras.ops.greater(valid_counts, 0)
        if halted is not None:
            valid_mask_seq = halted & valid_mask_seq

        # Apply weights — use the mask as sample_weight so we don't branch
        # on a tensor value (which breaks graph mode)
        weight = keras.ops.cast(valid_mask_seq, "float32")

        self.accuracy.update_state(token_accuracy, sample_weight=weight)
        self.exact_accuracy.update_state(
            keras.ops.cast(seq_correct, "float32"), sample_weight=weight
        )
        self.q_halt_accuracy.update_state(
            keras.ops.cast(q_halt_correct, "float32"), sample_weight=weight
        )

        if halted is not None and steps is not None:
            halted_weight = keras.ops.cast(halted, "float32") * keras.ops.cast(
                keras.ops.greater(valid_counts, 0), "float32"
            )
            self.avg_steps.update_state(
                keras.ops.cast(steps, "float32"), sample_weight=halted_weight
            )

    def result(self) -> Dict[str, float]:
        """Get current metric results.

        Returns:
            Dictionary with accuracy, exact_accuracy, q_halt_accuracy, avg_steps.
        """
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

    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary."""
        return {"ignore_index": self.ignore_index}


# ---------------------------------------------------------------------
