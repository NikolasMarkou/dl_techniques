"""
A composite, multi-task loss function for training the Hierarchical Reasoning Model.

This loss function combines the two primary objectives of the HRM:
1.  **Language Modeling (LM):** To accurately predict the next token in a sequence.
2.  **Q-Learning for Halting:** To train the Adaptive Computation Time (ACT)
    mechanism to decide when to stop the reasoning process.

The total loss is a weighted sum of the LM loss and two distinct Q-learning losses.

The loss components are:
1.  **LM Loss:**
    - The primary language modeling objective.
    - Can be configured to use either the custom `StableMaxCrossEntropy` or the
      standard `SparseCategoricalCrossentropy`.

2.  **Q-Halt Loss:**
    - This trains the `q_halt` head of the model.
    - The goal is to predict whether the model's current prediction for the
      entire sequence is already correct.
    - The target is a binary value (1 if the generated sequence is perfect, 0 otherwise).
      This encourages the model to halt when it is confident in its answer.

3.  **Q-Continue Loss:**
    - This trains the `q_continue` head using a bootstrapping (Temporal Difference)
      approach, which is standard in Q-learning.
    - The goal is to predict the expected future value of *not* halting.
    - The target is calculated from the maximum Q-value of the *next* computational
      step, allowing the model to learn the long-term benefit of continuing to reason.

The `q_loss_weight` hyperparameter controls the balance between the language modeling
task and the task of learning when to halt.
"""

import keras

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class StableMaxCrossEntropy(keras.losses.Loss):
    """
    Stable max cross entropy loss as used in the original HRM.

    This is a numerically stable version of cross entropy that uses
    a modified softmax function.
    """

    def __init__(self,
                 from_logits: bool = True,
                 ignore_index: int = -100,
                 epsilon: float = 1e-30,
                 **kwargs):
        super().__init__(**kwargs)
        self.from_logits = from_logits
        self.ignore_index = ignore_index
        self.epsilon = epsilon

    def _stable_function(self, x):
        """Stable function s(x) as defined in original code."""
        return keras.ops.where(
            x < 0,
            1.0 / (1.0 - x + self.epsilon),
            x + 1.0
        )

    def _log_stablemax(self, logits, axis=-1):
        """Log of stable softmax."""
        s_x = self._stable_function(logits)
        return keras.ops.log(s_x / keras.ops.sum(s_x, axis=axis, keepdims=True))

    def call(self, y_true, y_pred):
        """
        Compute stable max cross entropy loss.

        Args:
            y_true: True labels (batch_size, seq_len)
            y_pred: Predicted logits (batch_size, seq_len, vocab_size)

        Returns:
            Loss tensor (batch_size, seq_len)
        """
        # Cast to float64 for numerical stability
        y_pred = keras.ops.cast(y_pred, "float64")
        y_true = keras.ops.cast(y_true, "int64")

        # Compute log probabilities
        log_probs = self._log_stablemax(y_pred, axis=-1)

        # Create mask for valid labels
        valid_mask = keras.ops.not_equal(y_true, self.ignore_index)

        # Get predictions for true labels
        y_true_safe = keras.ops.where(valid_mask, y_true, 0)
        pred_log_probs = keras.ops.take_along_axis(
            log_probs,
            keras.ops.expand_dims(y_true_safe, axis=-1),
            axis=-1
        )
        pred_log_probs = keras.ops.squeeze(pred_log_probs, axis=-1)

        # Apply mask and return negative log likelihood
        loss = keras.ops.where(valid_mask, -pred_log_probs, 0.0)

        return keras.ops.cast(loss, y_pred.dtype)

    def get_config(self):
        """Get loss configuration."""
        config = super().get_config()
        config.update({
            "from_logits": self.from_logits,
            "ignore_index": self.ignore_index,
            "epsilon": self.epsilon,
        })
        return config

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class HRMLoss(keras.losses.Loss):
    """
    Combined loss function for Hierarchical Reasoning Model.

    Combines language modeling loss, Q-learning losses for halt/continue
    decisions, and computes accuracy metrics.

    Args:
        lm_loss_type: Type of language modeling loss ("stable_max" or "sparse_categorical_crossentropy")
        q_loss_weight: Weight for Q-learning losses
        ignore_index: Index to ignore in loss computation
        **kwargs: Additional loss arguments
    """

    def __init__(self,
                 lm_loss_type: str = "stable_max",
                 q_loss_weight: float = 0.5,
                 ignore_index: int = -100,
                 **kwargs):
        super().__init__(**kwargs)

        self.lm_loss_type = lm_loss_type
        self.q_loss_weight = q_loss_weight
        self.ignore_index = ignore_index

        # Language modeling loss
        if lm_loss_type == "stable_max":
            self.lm_loss_fn = StableMaxCrossEntropy(ignore_index=ignore_index)
        else:
            self.lm_loss_fn = keras.losses.SparseCategoricalCrossentropy(
                from_logits=True,
                ignore_class=ignore_index if ignore_index >= 0 else None
            )

        # Q-learning loss
        self.q_loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)

        # Metrics
        self.accuracy_metric = keras.metrics.SparseCategoricalAccuracy()
        self.exact_accuracy_metric = keras.metrics.BinaryAccuracy()

    def call(self, y_true, y_pred):
        """
        Compute combined HRM loss.

        Args:
            y_true: Dict with "labels", "halted", "steps"
            y_pred: Dict with "logits", "q_halt_logits", "q_continue_logits", optionally "target_q_continue"

        Returns:
            Total loss scalar
        """
        # Extract components
        labels = y_true["labels"]
        halted = y_true.get("halted", None)

        logits = y_pred["logits"]
        q_halt_logits = y_pred["q_halt_logits"]
        q_continue_logits = y_pred.get("q_continue_logits", None)
        target_q_continue = y_pred.get("target_q_continue", None)

        # Language modeling loss
        if self.lm_loss_type == "stable_max":
            lm_losses = self.lm_loss_fn(labels, logits)
        else:
            # Flatten for standard sparse categorical crossentropy
            labels_flat = keras.ops.reshape(labels, [-1])
            logits_flat = keras.ops.reshape(logits, [-1, keras.ops.shape(logits)[-1]])
            lm_losses = self.lm_loss_fn(labels_flat, logits_flat)
            lm_losses = keras.ops.reshape(lm_losses, keras.ops.shape(labels))

        # Compute valid token mask
        valid_mask = keras.ops.not_equal(labels, self.ignore_index)
        valid_counts = keras.ops.sum(keras.ops.cast(valid_mask, "float32"), axis=-1)
        valid_counts = keras.ops.maximum(valid_counts, 1.0)  # Avoid division by zero

        # Average LM loss per sequence
        lm_loss = keras.ops.sum(lm_losses) / keras.ops.sum(valid_counts)

        # Compute sequence-level correctness for Q-learning targets
        pred_labels = keras.ops.argmax(logits, axis=-1)
        correct_tokens = valid_mask & keras.ops.equal(pred_labels, labels)
        seq_correct = keras.ops.equal(
            keras.ops.sum(keras.ops.cast(correct_tokens, "float32"), axis=-1),
            valid_counts
        )

        # Q-halt loss (predict sequence correctness)
        q_halt_targets = keras.ops.cast(seq_correct, "float32")
        q_halt_loss = self.q_loss_fn(q_halt_targets, q_halt_logits)

        # Q-continue loss (bootstrapping target)
        q_continue_loss = 0.0
        if target_q_continue is not None and q_continue_logits is not None:
            q_continue_loss = self.q_loss_fn(target_q_continue, q_continue_logits)

        # Total loss
        total_loss = lm_loss + self.q_loss_weight * (q_halt_loss + q_continue_loss)

        return total_loss

    def get_config(self):
        """Get loss configuration."""
        config = super().get_config()
        config.update({
            "lm_loss_type": self.lm_loss_type,
            "q_loss_weight": self.q_loss_weight,
            "ignore_index": self.ignore_index,
        })
        return config

# ---------------------------------------------------------------------

def create_hrm_loss(
        lm_loss_type: str = "stable_max",
        q_loss_weight: float = 0.5,
        ignore_index: int = -100
) -> HRMLoss:
    """
    Create HRM loss function.

    Args:
        lm_loss_type: Type of language modeling loss
        q_loss_weight: Weight for Q-learning losses
        ignore_index: Index to ignore in loss computation

    Returns:
        Configured HRMLoss instance
    """
    return HRMLoss(
        lm_loss_type=lm_loss_type,
        q_loss_weight=q_loss_weight,
        ignore_index=ignore_index
    )

# ---------------------------------------------------------------------
