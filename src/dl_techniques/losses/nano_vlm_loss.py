"""
Autoregressive cross-entropy loss for language modeling.

This loss function implements the core training objective for the language
generation component of a Vision-Language Model (VLM). It is designed to
train a model to predict the next token in a sequence, conditioned on the
preceding tokens and, implicitly, the visual input that has been processed
by the model's encoder.

Conceptual Overview:
    The fundamental principle is autoregressive sequence generation, where the
    model learns a probability distribution over a vocabulary for the next
    token at each position in a sequence. By maximizing the likelihood of the
    ground-truth next token at every step, the model learns the statistical
    patterns of language, enabling it to generate coherent text. This is the
    standard objective used to train large-scale generative language models
    like GPT.

Architectural Design:
    To implement the "next-token prediction" objective, this loss function
    relies on a crucial alignment shift between the model's predictions and
    the ground-truth labels.
    -   The predictions for sequence positions `[0, 1, ..., N-1]` are used.
    -   The ground-truth labels for sequence positions `[1, 2, ..., N]` are
        used as targets.
    This effectively aligns the model's output at time step `t` with the
    target token at `t+1`, forcing the model to predict one step into the
    future.

    NanoVLM's logits cover more positions than its labels: the model puts
    its projected vision tokens BEFORE the text tokens and runs one causal
    decoder over both, so its output is ``[batch, vision_len + text_len,
    vocab]`` against ``[batch, text_len]`` labels. The text positions are
    the trailing ``text_len`` ones, so the loss scores that suffix and
    ignores the vision prefix (:func:`text_position_logits`). The logit at
    text position ``t`` has seen the image and text ``0..t`` only.

    A key architectural component is the handling of padding. Since sequences
    in a batch are padded to a uniform length, an `ignore_index` is used to
    create a mask. This mask ensures that the loss is only computed for
    actual tokens in the sequence, and the gradients are not influenced by
    the arbitrary padding tokens.

Mathematical Formulation:
    The loss is based on the standard cross-entropy between the model's
    predicted probability distribution and the one-hot encoded ground-truth
    token. For a given sequence position, the loss is the negative
    log-probability of the correct target token:

        L_token = -log(p_target)

    where `p_target` is the probability assigned by the model to the correct
    next token. In practice, this is computed using the model's raw output
    logits `z` and the target token's index `y`:

        L_token = -z_y + log( Σ_j exp(z_j) )

    The final loss is the average of `L_token` over all non-masked (i.e.,
    non-padding) tokens across all sequences in the batch. This formulation
    may also incorporate label smoothing, a regularization technique that
    replaces the hard one-hot target with a soft distribution, preventing
    the model from becoming overconfident.

References:
    -   Radford, A., et al. (2018). "Improving Language Understanding by
        Generative Pre-Training." Foundational for large-scale autoregressive
        language model pre-training.
    -   Vaswani, A., et al. (2017). "Attention Is All You Need." Introduced the
        Transformer architecture where this loss is predominantly used.
"""

import keras

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


def text_position_logits(
        y_true: keras.KerasTensor,
        y_pred: keras.KerasTensor
) -> keras.KerasTensor:
    """Return the logits at the label positions: the trailing ``seq_len``.

    Interface contract (callers: ``NanoVLMLoss.call``): ``y_true`` is
    ``[batch, seq_len]``, ``y_pred`` is ``[batch, pred_len, vocab]`` with
    ``pred_len >= seq_len``. NanoVLM puts its non-text positions (the
    projected vision tokens) BEFORE the text, so the text logits are the
    last ``seq_len`` positions. Equal static lengths return ``y_pred`` unchanged.
    ``pred_len`` may be unknown at trace time (NanoVLM's sequence axis is
    dynamic under ``fit`` when the batch size is unknown); the suffix slice
    needs only ``seq_len``, and the length check runs when both are static.

    Args:
        y_true: Target token IDs, ``[batch, seq_len]``.
        y_pred: Logits, ``[batch, pred_len, vocab_size]``.

    Returns:
        The last ``seq_len`` positions of ``y_pred``.

    Raises:
        ValueError: If ``y_true`` is not rank 2, ``y_pred`` is not rank 3, or
            a static ``pred_len`` is smaller than a static ``seq_len``.
    """
    y_true_shape = keras.ops.shape(y_true)
    y_pred_shape = keras.ops.shape(y_pred)
    if len(y_true_shape) != 2 or len(y_pred_shape) != 3:
        raise ValueError(
            f"Expected y_true shape [batch, seq_len] and y_pred shape "
            f"[batch, pred_len, vocab_size], got {y_true_shape} and {y_pred_shape}"
        )
    seq_len_true = y_true_shape[1]
    seq_len_pred = y_pred_shape[1]
    if isinstance(seq_len_true, int) and isinstance(seq_len_pred, int):
        if seq_len_pred < seq_len_true:
            raise ValueError(
                f"Sequence length mismatch: y_pred has {seq_len_pred} positions, "
                f"fewer than the {seq_len_true} labels in y_true"
            )
        if seq_len_pred == seq_len_true:
            return y_pred
    return y_pred[:, -seq_len_true:, :]

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.losses.nano_vlm_loss")
class NanoVLMLoss(keras.losses.Loss):
    """Autoregressive language modeling loss for vision_heads-language training.

    This loss function implements the standard autoregressive language modeling
    objective with proper token masking for vision_heads-language models. It computes
    the cross-entropy loss between predicted and target tokens while ignoring
    padding tokens and applying optional label smoothing.

    The loss shifts the prediction and target sequences to implement the
    autoregressive objective: predict the next token given all previous tokens.

    Args:
        ignore_index: Token index to ignore in loss computation. Typically
            used for padding tokens (the trainer's ``PAD_ID`` is 0, so 0 is the
            documented default and drops a real token 0 too). A negative
            value such as -1 is allowed: labels are clamped before the
            cross-entropy. Default: 0.
        label_smoothing: Label smoothing factor between 0.0 and 1.0.
            Higher values apply more smoothing. Default: 0.0.
        from_logits: Whether predictions are logits or probabilities.
            Default: True.
        reduction: Type of reduction to apply to loss. One of 'sum_over_batch_size',
            'sum', or 'none'. Default: 'sum_over_batch_size'.
        name: Name of the loss function. Default: "nanovlm_loss".
        **kwargs: Additional keyword arguments for Loss base class.

    Example:
        >>> loss_fn = NanoVLMLoss(ignore_index=0, label_smoothing=0.1)
        >>> # y_true: [batch_size, seq_len] - target token IDs
        >>> # y_pred: [batch_size, seq_len, vocab_size] - predicted logits
        >>> loss = loss_fn(y_true, y_pred)

    Note:
        The input sequences are automatically shifted for autoregressive training:
        - Predictions use tokens [0:seq_len-1] to predict [1:seq_len]
        - This implements the standard language modeling objective

        The per-row values share one batch-wide token count, so the rows are
        coupled: a zero ``sample_weight`` removes a row's contribution but not
        its tokens from the denominator, and is not the same as dropping the
        row (see ``call``).
    """

    def __init__(
            self,
            ignore_index: int = 0,
            label_smoothing: float = 0.0,
            from_logits: bool = True,
            reduction: str = "sum_over_batch_size",
            name: str = "nanovlm_loss",
            **kwargs
    ) -> None:
        super().__init__(name=name, reduction=reduction, **kwargs)

        # Validate parameters
        if not 0.0 <= label_smoothing <= 1.0:
            raise ValueError(
                f"label_smoothing must be between 0.0 and 1.0, got {label_smoothing}"
            )

        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.from_logits = from_logits

        # Initialize the underlying sparse categorical crossentropy loss
        # Note: SparseCategoricalCrossentropy does not support label_smoothing;
        # label smoothing is applied manually in call() when needed
        self.sparse_ce = keras.losses.SparseCategoricalCrossentropy(
            from_logits=from_logits,
            reduction='none',  # We handle reduction manually for masking
        )

        logger.info(
            f"Initialized {self.__class__.__name__} with ignore_index={ignore_index}, "
            f"label_smoothing={label_smoothing}, from_logits={from_logits}"
        )

    def call(
            self,
            y_true: keras.KerasTensor,
            y_pred: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Compute the autoregressive language modeling loss.

        Args:
            y_true: Target token IDs of shape [batch_size, seq_len].
                Should contain integer token indices.
            y_pred: Predicted logits or probabilities of shape
                [batch_size, pred_len, vocab_size], ``pred_len >= seq_len``;
                only the trailing ``seq_len`` positions are scored.

        Returns:
            Per-sample loss of shape ``[batch_size]`` for every ``reduction``.
            Row ``i`` is ``S_i / N * batch_size``, where ``S_i`` is row ``i``'s
            token-loss sum and ``N`` is the number of non-masked tokens in the
            WHOLE batch, so the default ``sum_over_batch_size`` reduction is the
            batch-wide mean over non-masked tokens EXACTLY. Under
            ``reduction="sum"`` the result is therefore ``batch_size`` times
            that mean, not the raw token sum.

            The batch axis lets ``sample_weight`` and ``reduction=`` act per
            row, but the rows stay COUPLED through ``N``: ``call()`` never sees
            the weights, so a zero ``sample_weight`` zeroes that row's
            contribution while its tokens still count in ``N``. It is NOT
            equivalent to dropping the row: with ``n_d`` valid tokens in the
            zero-weighted row, ``loss(w) == loss(batch without the row) *
            (N - n_d) / N`` (``(B - 1) / B`` only when every row has the same
            valid-token count).

        Raises:
            ValueError: If a rank is wrong or ``y_pred`` has fewer positions
                than ``y_true``.
        """
        # Score only the label positions (NanoVLM's vision prefix is dropped)
        y_pred = text_position_logits(y_true, y_pred)

        # Shift sequences for autoregressive prediction
        # Predict next token given previous tokens
        y_pred_shifted = y_pred[:, :-1, :]  # [batch, seq_len-1, vocab_size]
        y_true_shifted = y_true[:, 1:]  # [batch, seq_len-1]

        # Clamp ignored targets so a negative `ignore_index` never reaches the
        # sparse CE as an invalid label; the mask below removes their loss.
        safe_labels = keras.ops.maximum(y_true_shifted, 0)

        # Per-token cross-entropy: [batch, seq_len-1]
        loss_per_token = self.sparse_ce(safe_labels, y_pred_shifted)

        # Apply label smoothing manually if configured
        if self.label_smoothing > 0.0:
            if self.from_logits:
                log_probs = y_pred_shifted - keras.ops.logsumexp(
                    y_pred_shifted, axis=-1, keepdims=True
                )
            else:
                log_probs = keras.ops.log(y_pred_shifted + 1e-8)
            smooth_loss = -keras.ops.mean(log_probs, axis=-1)
            loss_per_token = (
                (1.0 - self.label_smoothing) * loss_per_token
                + self.label_smoothing * smooth_loss
            )

        mask = keras.ops.cast(
            keras.ops.not_equal(y_true_shifted, self.ignore_index),
            dtype=loss_per_token.dtype
        )

        # DECISION plan-2026-11-11T131349-13b28b37/D-011
        # Return one value per ROW, scaled so Keras' `sum_over_batch_size`
        # reproduces the batch-wide token mean exactly. Do NOT return that
        # scalar: `Loss.__call__` multiplies call()'s output by
        # `sample_weight` before reducing, so a scalar made every weighted
        # value `unweighted * mean(sample_weight)` whichever rows were
        # weighted. Do NOT return each row's OWN mean either: a mean of means
        # re-weights short sequences up (see masked_causal_lm_loss.py). Do NOT
        # branch in Python on the sequence length: it is a tensor under a
        # ragged `padded_batch`; a one-token sequence has no valid target and
        # row_token_sum = 0 already scores it 0.
        #
        # DECISION plan-2026-11-11T131349-13b28b37/D-046
        # `total_valid_tokens` is BATCH-WIDE, so the rows are coupled and a
        # zero `sample_weight` is not a row drop. That is documented, not
        # fixed: `call()` receives no `sample_weight` (Keras multiplies it in
        # afterwards), so the denominator cannot be restricted to weighted rows
        # here. WHAT NOT TO DO: do not write "sample_weight selects rows" in a
        # docstring for this form, and do not "fix" the coupling by dividing by
        # a per-row count (that is the mean-of-means rejected above). The float64
        # reference and the exact-ratio pin are in
        # test_the_nano_vlm_loss_scores_the_text_positions.py.
        row_token_sum = keras.ops.sum(loss_per_token * mask, axis=-1)
        total_valid_tokens = keras.ops.sum(mask) + 1e-8
        batch_size = keras.ops.cast(keras.ops.shape(mask)[0], row_token_sum.dtype)
        return row_token_sum / total_valid_tokens * batch_size

    def get_config(self) -> dict:
        """Get the loss function configuration.

        Returns:
            Dictionary containing the configuration parameters.
        """
        config = super().get_config()
        config.update({
            "ignore_index": self.ignore_index,
            "label_smoothing": self.label_smoothing,
            "from_logits": self.from_logits,
        })
        return config

    @classmethod
    def from_config(cls, config: dict) -> "NanoVLMLoss":
        """Create loss function from configuration.

        Args:
            config: Configuration dictionary from get_config().

        Returns:
            New NanoVLMLoss instance.
        """
        return cls(**config)

# ---------------------------------------------------------------------
