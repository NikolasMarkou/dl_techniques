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
            used for padding tokens. Default: 0.
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

        if ignore_index < 0:
            logger.warning(
                f"ignore_index is negative ({ignore_index}). "
                "This may cause unexpected behavior."
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
            Scalar loss value averaged over non-masked tokens.

        Raises:
            ValueError: If a rank is wrong or ``y_pred`` has fewer positions
                than ``y_true``.
        """
        # Score only the label positions (NanoVLM's vision prefix is dropped)
        y_pred = text_position_logits(y_true, y_pred)
        y_pred_shape = keras.ops.shape(y_pred)
        seq_len_true = keras.ops.shape(y_true)[1]

        # Ensure minimum sequence length for autoregressive training
        if seq_len_true < 2:
            logger.warning(
                f"Sequence length is {seq_len_true}, which is too short for "
                "autoregressive training. Consider using longer sequences."
            )
            return keras.ops.zeros([], dtype=y_pred.dtype)

        # Shift sequences for autoregressive prediction
        # Predict next token given previous tokens
        y_pred_shifted = y_pred[:, :-1, :]  # [batch, seq_len-1, vocab_size]
        y_true_shifted = y_true[:, 1:]  # [batch, seq_len-1]

        # Flatten for loss computation
        batch_size = y_pred_shape[0]
        vocab_size = y_pred_shape[2]
        seq_len_shifted = seq_len_true - 1

        y_pred_flat = keras.ops.reshape(
            y_pred_shifted,
            [batch_size * seq_len_shifted, vocab_size]
        )
        y_true_flat = keras.ops.reshape(
            y_true_shifted,
            [batch_size * seq_len_shifted]
        )

        # Compute per-token loss
        loss_per_token = self.sparse_ce(y_true_flat, y_pred_flat)

        # Apply label smoothing manually if configured
        if self.label_smoothing > 0.0:
            if self.from_logits:
                log_probs = y_pred_flat - keras.ops.logsumexp(
                    y_pred_flat, axis=-1, keepdims=True
                )
            else:
                log_probs = keras.ops.log(y_pred_flat + 1e-8)
            smooth_loss = -keras.ops.mean(log_probs, axis=-1)
            loss_per_token = (
                (1.0 - self.label_smoothing) * loss_per_token
                + self.label_smoothing * smooth_loss
            )

        # Create mask to ignore specified tokens (e.g., padding)
        mask = keras.ops.cast(
            keras.ops.not_equal(y_true_flat, self.ignore_index),
            dtype=loss_per_token.dtype
        )

        # Apply mask to loss
        masked_loss = loss_per_token * mask

        # Compute final loss based on reduction type
        total_loss = keras.ops.sum(masked_loss)
        total_valid_tokens = keras.ops.sum(mask)

        # Avoid division by zero
        mean_loss = keras.ops.where(
            keras.ops.greater(total_valid_tokens, 0),
            total_loss / keras.ops.maximum(total_valid_tokens, 1.0),
            keras.ops.zeros_like(total_loss)
        )

        # Handle different reduction types
        if self.reduction == "sum":
            return total_loss
        elif self.reduction == "none":
            # Return per-sample loss (reshape back to [batch_size, seq_len-1])
            return keras.ops.reshape(masked_loss, [batch_size, seq_len_shifted])
        else:  # sum_over_batch_size
            return mean_loss

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
