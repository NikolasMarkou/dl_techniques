"""
Focal Causal Language Modeling Loss.

A focal variant of token-level cross-entropy for causal language models.
Down-weights easy (well-predicted) tokens and focuses training on harder,
more informative tokens — improving learning efficiency on long-tail
vocabulary distributions typical of natural language.

.. code-block:: text

    Standard CE:     loss_t = -log(p_t)
    Focal CE:        loss_t = -(1 - p_t)^γ · log(p_t)
                                ↑ modulating factor
                     γ = 0  →  standard cross-entropy
                     γ > 0  →  down-weights easy tokens (high p_t)

    ┌─────────────────────────────────────────────────────────────┐
    │  Token:      "the"   "cat"   "sat"   "on"   "the"   "mat"   │
    │  p_t:         0.95    0.12    0.30    0.88    0.93    0.05  │
    │                                                             │
    │  CE weight:   1.0     1.0     1.0     1.0     1.0     1.0   │
    │  Focal(γ=2):  0.0025  0.774   0.490   0.014   0.005   0.90  │
    │               ↑ easy, suppressed        ↑ hard, amplified   │
    └─────────────────────────────────────────────────────────────┘

Motivation
----------
Standard cross-entropy treats all token positions equally. In natural
language, most tokens are highly predictable (function words, common
patterns) while a small fraction carry the bulk of semantic content.
Focal loss automatically focuses training on the informative minority,
which:

1. **Reduces repetition** — common tokens that dominate the loss surface
   receive less gradient, preventing the model from over-optimizing
   for high-frequency patterns at the expense of diversity.
2. **Improves rare token prediction** — low-frequency content words and
   proper nouns receive proportionally more training signal.
3. **Stabilizes small models** — particularly beneficial for models with
   limited capacity (< 100M params) training from scratch, where the
   loss budget must be spent wisely.

The ``alpha`` (class weighting) parameter can optionally balance the
loss further, e.g. up-weighting rare tokens by inverse frequency.

References
----------
- Lin et al. (2017). "Focal Loss for Dense Object Detection."
  *ICCV*. (Original focal loss for vision.)
- Ghosh et al. (2024). "Beyond Accuracy Optimization: Computer Vision
  Losses for Large Language Model Fine-Tuning." *EMNLP Findings*.
  (+42% exact match improvement using focal loss for LLM fine-tuning.)
- Zhang et al. (2024). "MiLe Loss: a New Entropy-Weighed Loss for
  Mitigating the Bias of Learning Difficulties in LLMs." *arXiv:2310.19531*.
"""

import keras
from keras import ops
from typing import Optional

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class FocalCausalLMLoss(keras.losses.Loss):
    """Focal cross-entropy for causal language modeling with ignore index.

    Combines focal loss modulation with PAD masking for autoregressive
    language model training. Down-weights easy tokens (high predicted
    probability) and focuses on hard, informative tokens.

    .. code-block:: text

        loss_t = -α · (1 - p_t)^γ · log(p_t)

        where p_t = softmax(logits)[y_true] is the predicted probability
        of the correct token at position t.

    :param gamma: Focusing parameter γ ≥ 0. Higher values increase
        focus on hard tokens. ``γ=0`` recovers standard cross-entropy.
        Recommended: ``1.0-3.0`` for LM training. Default ``2.0``.
    :param alpha: Optional class weight. When a float, applied uniformly.
        Default ``None`` (no weighting).
    :param ignore_index: Label value to ignore (PAD positions).
        Default ``-1``.
    :param label_smoothing: Smoothing factor in ``[0, 1)``. Blends
        one-hot target with uniform distribution. Default ``0.0``.
    :param from_logits: Whether ``y_pred`` contains raw logits.
        Default ``True``.
    :param name: Name for the loss instance.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[float] = None,
        ignore_index: int = -1,
        label_smoothing: float = 0.0,
        from_logits: bool = True,
        name: str = "focal_causal_lm_loss",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.from_logits = from_logits

    def call(
        self,
        y_true: keras.KerasTensor,
        y_pred: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """Compute focal causal LM loss.

        :param y_true: Ground truth token IDs ``(batch, seq_len)``.
            Positions with ``ignore_index`` are excluded.
        :param y_pred: Predicted logits ``(batch, seq_len, vocab_size)``.
        :return: Scalar loss averaged over non-masked positions.
        """
        # Mask: 1.0 for real tokens, 0.0 for ignored
        mask = ops.cast(y_true != self.ignore_index, "float32")
        safe_labels = ops.maximum(y_true, 0)

        # Convert logits to log-probabilities and probabilities
        if self.from_logits:
            log_probs = ops.log_softmax(y_pred, axis=-1)
        else:
            log_probs = ops.log(y_pred + 1e-8)

        # Gather log-prob of the correct token at each position
        # safe_labels: (batch, seq_len) → one_hot: (batch, seq_len, V)
        one_hot = ops.one_hot(safe_labels, ops.shape(y_pred)[-1])
        # log_p_t: (batch, seq_len) — log probability of correct token
        log_p_t = ops.sum(log_probs * one_hot, axis=-1)
        # p_t: probability of correct token
        p_t = ops.exp(log_p_t)

        # Focal modulating factor: (1 - p_t)^gamma
        focal_weight = ops.power(1.0 - p_t, self.gamma)

        # Per-token focal loss: -alpha * (1 - p_t)^gamma * log(p_t)
        per_token_loss = -focal_weight * log_p_t

        # Optional alpha weighting
        if self.alpha is not None:
            per_token_loss = self.alpha * per_token_loss

        # Optional label smoothing
        if self.label_smoothing > 0.0:
            # Uniform distribution component: -mean(log_probs) per position
            smooth_loss = -ops.mean(log_probs, axis=-1)
            per_token_loss = (
                (1.0 - self.label_smoothing) * per_token_loss
                + self.label_smoothing * smooth_loss
            )

        # Masked mean: average only over real token positions
        numerator = ops.sum(per_token_loss * mask)
        denominator = ops.sum(mask) + 1e-8
        return numerator / denominator

    def get_config(self) -> dict:
        """Get loss configuration for serialization."""
        config = super().get_config()
        config.update({
            "gamma": self.gamma,
            "alpha": self.alpha,
            "ignore_index": self.ignore_index,
            "label_smoothing": self.label_smoothing,
            "from_logits": self.from_logits,
        })
        return config
