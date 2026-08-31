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
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.losses.hrm_loss")
class StableMaxCrossEntropy(keras.losses.Loss):
    """
    Stable max cross entropy loss as used in the original HRM.

    This is a numerically stable version of cross entropy that uses
    a modified softmax function.

    **``call()`` returns PER-TOKEN values of shape ``(batch, seq_len)``, and that
    is deliberate — it is not the premature-scalar defect and it is not a missing
    reduction.** ``keras.losses.SparseCategoricalCrossentropy`` returns the same
    shape for the same inputs; both are token-level losses and neither reduces
    the sequence axis. Consequences, all MEASURED 2026-08-31 rather than assumed:

    *   A ``sample_weight`` of shape ``(batch,)`` RAISES
        ``InvalidArgumentError: Incompatible shapes: [4,5] vs. [4]``. So does
        stock ``SparseCategoricalCrossentropy`` on the identical inputs — this is
        the Keras convention for a token-level loss, not a bug in this class. Pass
        ``(batch, 1)`` or ``(batch, seq_len)`` instead; both work and both select
        rows correctly (2.00308299 under ``[1,1,1,0]`` against an unweighted
        2.56611896, where the broken broadcast would have given 1.92458922).
    *   Reducing the sequence axis here was evaluated and REJECTED. It would break
        :class:`HRMLoss`, which needs per-token values in order to divide by its
        OWN valid-token count (``sum(lm_losses) / sum(valid_counts)``), and it
        would leave this class shaped differently from the
        ``sparse_categorical_crossentropy`` branch it is interchangeable with.
        The token-pool per-sample form additionally changes the value: under
        masking the current default reduction reports 1.71106493 (the mean over
        ALL positions, masked ones contributing zero) where a per-valid-token
        mean is 2.44437834, 42.9% apart. See ``decisions.md`` D-002 of
        ``plan-2026-08-31T045723-c0d5ffa9``.

    Known and deliberately NOT fixed here: ``call()`` casts ``y_pred`` to float64
    for numerical stability and then returns ``ops.cast(loss, y_pred.dtype)``,
    where ``y_pred`` is by then the float64 copy — so a float32 caller gets a
    float64 loss back. Recorded 2026-08-31; out of scope for the shape work.
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

        # DECISION plan-2026-08-31T045723-c0d5ffa9/D-002
        # The token axis is deliberately NOT reduced here. Do NOT "finish the
        # premature-scalar fix" by adding a `mean(axis=-1)` or the token-pool
        # per-sample form: stock SparseCategoricalCrossentropy is token-shaped on
        # these same inputs and raises the same error on a (batch,) sample_weight,
        # HRMLoss below needs per-token values to divide by its own valid-token
        # count, and the token-pool form moves the reported value by 42.9% under
        # masking. Pass a (batch, 1) or (batch, seq_len) sample_weight instead.
        # See the class docstring and decisions.md D-002.
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


@register_dl_technique("dl_techniques.losses.hrm_loss")
class HRMLoss(keras.losses.Loss):
    """
    Combined loss function for Hierarchical Reasoning Model.

    Combines language modeling loss, Q-learning losses for halt/continue
    decisions, and computes accuracy metrics.

    ``call()`` returns PER-SAMPLE values of shape ``(batch,)``, not a scalar.
    All three terms are decomposed so that each one's own mean is the scalar it
    used to contribute, which means Keras' default reduction reproduces the
    total this loss has always reported. The LM term's row value is proportional
    to that row's valid-token COUNT as well as to its token losses -- faithful to
    the batch-wide token mean it replaces, and deliberately not a per-sequence
    mean.

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
            # DECISION plan-2026-08-31T045723-c0d5ffa9/D-001
            # `reduction="none"` is LOAD-BEARING, not cosmetic. `call()` below does
            # `sum(lm_losses) / sum(valid_counts)`, which is only correct if this
            # sub-loss hands back PER-TOKEN values. Without it, `Loss.__call__`
            # returns an already-averaged scalar and that line divides it by the
            # token count a SECOND time, under-weighting the LM term by exactly
            # `sum(valid_counts)` (MEASURED: 24x at 4x6, 512x at 8x64, 4096x at
            # 8x512). The LM objective then all but vanishes next to the Q-learning
            # terms it is summed with, while the loss curve still looks healthy.
            # The `else` branch below has always passed `reduction="none"`; this
            # branch -- the DEFAULT -- did not. See decisions.md D-001.
            self.lm_loss_fn = StableMaxCrossEntropy(
                ignore_index=ignore_index,
                reduction="none",
            )
        else:
            self.lm_loss_fn = keras.losses.SparseCategoricalCrossentropy(
                from_logits=True,
                ignore_class=ignore_index if ignore_index >= 0 else None,
                reduction="none",
            )

        # Q-learning loss. `reduction="none"` matches the LM sub-losses above and
        # is what makes the Q terms per-sample; see the reshape in `call()` for
        # the trap that comes with it.
        self.q_loss_fn = keras.losses.BinaryCrossentropy(
            from_logits=True,
            reduction="none",
        )

    def call(self, y_true, y_pred):
        """
        Compute combined HRM loss.

        Args:
            y_true: Dict with "labels", "halted", "steps"
            y_pred: Dict with "logits", "q_halt_logits", "q_continue_logits", optionally "target_q_continue"

        Returns:
            Per-sample total loss with shape ``(batch,)``: the LM term plus
            ``q_loss_weight`` times the two Q terms, each decomposed so that its
            OWN mean is the scalar it used to contribute. Keras' default
            reduction recovers the total this used to return; keeping the batch
            axis is what lets ``sample_weight`` and ``reduction=`` select rows.
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
            # DECISION plan-2026-08-31T045723-c0d5ffa9/D-003
            # CLAMP the ignored labels to a VALID class id before the sub-loss sees
            # them, then zero those positions afterwards. Passing them through raw
            # CRASHES: `__init__` sets `ignore_class=ignore_index if ignore_index >= 0
            # else None`, so at the DEFAULT `ignore_index=-100` the sub-loss is told to
            # ignore nothing and TF raises
            # `Received a label value of -100 which is outside the valid range`.
            # The masking below is what actually removes those positions, so the clamp
            # only has to make the kernel's range check pass -- class 0 is arbitrary and
            # never reaches the returned value. See decisions.md D-003.
            safe_labels = keras.ops.where(
                keras.ops.not_equal(labels, self.ignore_index),
                labels,
                keras.ops.zeros_like(labels),
            )
            labels_flat = keras.ops.reshape(safe_labels, [-1])
            logits_flat = keras.ops.reshape(logits, [-1, keras.ops.shape(logits)[-1]])
            lm_losses = self.lm_loss_fn(labels_flat, logits_flat)
            lm_losses = keras.ops.reshape(lm_losses, keras.ops.shape(labels))
            # Zero the clamped positions so they contribute nothing to the token sum.
            lm_losses = keras.ops.where(
                keras.ops.not_equal(labels, self.ignore_index),
                lm_losses,
                keras.ops.zeros_like(lm_losses),
            )

        # Compute valid token mask
        valid_mask = keras.ops.not_equal(labels, self.ignore_index)
        valid_counts = keras.ops.sum(keras.ops.cast(valid_mask, "float32"), axis=-1)
        valid_counts = keras.ops.maximum(valid_counts, 1.0)  # Avoid division by zero

        # LM term, PER SAMPLE. This used to be the scalar
        # `sum(lm_losses) / sum(valid_counts)` -- the mean over every valid token
        # in the batch. Returning each sequence's OWN mean instead is a different
        # number whenever the valid-token counts differ, so each row's TOKEN SUM
        # is scaled by the batch-global token count and by the batch size, which
        # makes Keras' `sum_over_batch_size` reproduce that mean EXACTLY.
        total_valid = keras.ops.sum(valid_counts)
        batch_size = keras.ops.cast(
            keras.ops.shape(valid_counts)[0], lm_losses.dtype
        )
        lm_loss = (
            keras.ops.sum(lm_losses, axis=-1)
            / keras.ops.cast(total_valid, lm_losses.dtype)
            * batch_size
        )

        # Compute sequence-level correctness for Q-learning targets
        pred_labels = keras.ops.argmax(logits, axis=-1)
        correct_tokens = valid_mask & keras.ops.equal(pred_labels, labels)
        seq_correct = keras.ops.equal(
            keras.ops.sum(keras.ops.cast(correct_tokens, "float32"), axis=-1),
            valid_counts
        )

        # Q-halt loss (predict sequence correctness), PER SAMPLE.
        #
        # The `(batch, 1)` reshape is LOAD-BEARING and is the trap in this
        # decomposition. `BinaryCrossentropy` means over the LAST axis, so with
        # `(batch,)` inputs that axis IS the batch axis and the result collapses
        # to a scalar even under `reduction="none"` -- MEASURED: `call()` returns
        # shape `()` for `(4,)` inputs and shape `(4,)` for `(4, 1)` inputs, with
        # identical means. Do not "simplify" the reshape away.
        q_halt_targets = keras.ops.cast(seq_correct, "float32")
        q_halt_loss = self.q_loss_fn(
            keras.ops.reshape(q_halt_targets, (-1, 1)),
            keras.ops.reshape(q_halt_logits, (-1, 1)),
        )

        # Q-continue loss (bootstrapping target), PER SAMPLE. Absent, it is a
        # ZERO VECTOR rather than the scalar 0.0 -- a scalar would still add
        # correctly here, but only by broadcasting, and the shape of what this
        # branch contributes should not depend on whether it is present.
        q_continue_loss = keras.ops.zeros_like(q_halt_loss)
        if target_q_continue is not None and q_continue_logits is not None:
            q_continue_loss = self.q_loss_fn(
                keras.ops.reshape(target_q_continue, (-1, 1)),
                keras.ops.reshape(q_continue_logits, (-1, 1)),
            )

        # Total loss, per sample. Each of the three terms is a `(batch,)` vector
        # whose own mean is the scalar it used to contribute, so the mean of the
        # sum is the sum of the old scalars -- proven per TERM, not only on the
        # total, because three terms whose errors cancel would pass a total-only
        # check over a broken decomposition.
        q_halt_loss = keras.ops.cast(q_halt_loss, lm_loss.dtype)
        q_continue_loss = keras.ops.cast(q_continue_loss, lm_loss.dtype)
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
