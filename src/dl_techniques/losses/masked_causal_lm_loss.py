"""
Masked Causal Language Modeling Loss.

A token-level cross-entropy loss that supports an ignore index for masked
positions (e.g. padding tokens). Only non-masked positions contribute to
the loss, giving an accurate measure of language modeling performance
without PAD token inflation.

.. code-block:: text

    ┌─────────────────────────────────────────────────────┐
    │  Labels:   [  42,  17, 305,  -1,  -1,  -1,  -1 ]    │
    │  Logits:   [  ▓▓,  ▓▓,  ▓▓,  ░░,  ░░,  ░░,  ░░ ]    │
    │                                                     │
    │  Mask:     [  1,    1,   1,   0,   0,   0,   0  ]   │
    │                                                     │
    │  Loss = mean(CE[42,▓▓], CE[17,▓▓], CE[305,▓▓])      │
    │          ↑ only real tokens, PAD ignored            W│
    └─────────────────────────────────────────────────────┘

This is the standard approach used by GPT-2, GPT-3, LLaMA, and most
modern causal language models. The ``ignore_index`` convention (default
``-1``) follows PyTorch's ``CrossEntropyLoss(ignore_index=-100)``
pattern adapted for Keras.

Use Cases
---------
- **CLM pre-training**: next-token prediction with padded sequences
- **CLM fine-tuning**: domain adaptation with variable-length texts
- **Instruction tuning**: mask prompt tokens, only compute loss on response
- **Prefix LM**: mask prefix positions, train on continuation only

References
----------
- Radford et al. (2019). "Language Models are Unsupervised Multitask
  Learners." (GPT-2)
- Brown et al. (2020). "Language Models are Few-Shot Learners." (GPT-3)
"""

import keras
from keras import ops
from typing import Optional

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class MaskedCausalLMLoss(keras.losses.Loss):
    """Token-level cross-entropy with an ignore index for masked positions.

    Computes sparse categorical cross-entropy per token, masks out
    positions where ``y_true == ignore_index``, and averages only
    over the remaining (real) positions.

    .. code-block:: text

        Input
        ─────
        y_true : (batch, seq_len)       int32, with ignore_index for PAD
        y_pred : (batch, seq_len, V)    float32, raw logits

        Output
        ──────
        scalar : mean CE over non-masked positions

    :param ignore_index: Label value to ignore in loss computation.
        Positions with this label are masked out. Default ``-1``.
    :param label_smoothing: Smoothing factor in ``[0, 1)``. When > 0,
        blends the one-hot target with a uniform distribution over the
        vocabulary: ``target = (1 - α) * one_hot + α / V``.
        Helps prevent overconfident predictions. Default ``0.0``.
    :param from_logits: Whether ``y_pred`` contains raw logits (True)
        or probabilities (False). Default ``True``.
    :param name: Name for the loss instance.
    """

    def __init__(
        self,
        ignore_index: int = -1,
        label_smoothing: float = 0.0,
        from_logits: bool = True,
        name: str = "masked_causal_lm_loss",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.from_logits = from_logits
        self._base_loss = keras.losses.SparseCategoricalCrossentropy(
            from_logits=from_logits,
            reduction="none",
        )

    def call(
        self,
        y_true: keras.KerasTensor,
        y_pred: keras.KerasTensor,
    ) -> keras.KerasTensor:
        """Compute masked causal LM loss.

        :param y_true: Ground truth token IDs with shape
            ``(batch, seq_len)``. Positions set to ``ignore_index``
            are excluded from the loss.
        :param y_pred: Predicted logits with shape
            ``(batch, seq_len, vocab_size)``.
        :return: Scalar loss averaged over non-masked positions.
        """
        # Build mask: 1.0 for real tokens, 0.0 for ignored positions
        mask = ops.cast(y_true != self.ignore_index, "float32")

        # Replace ignore_index with 0 to avoid index-out-of-range
        safe_labels = ops.maximum(y_true, 0)

        # Per-token cross-entropy: (batch, seq_len)
        per_token_loss = self._base_loss(safe_labels, y_pred)

        # Optional label smoothing applied manually
        if self.label_smoothing > 0.0:
            vocab_size = ops.cast(ops.shape(y_pred)[-1], "float32")
            if self.from_logits:
                # Use log-softmax for numerical stability
                log_probs = y_pred - ops.logsumexp(y_pred, axis=-1, keepdims=True)
                smooth_loss = -ops.mean(log_probs, axis=-1)
            else:
                smooth_loss = -ops.mean(
                    ops.log(y_pred + 1e-8), axis=-1,
                )
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
            "ignore_index": self.ignore_index,
            "label_smoothing": self.label_smoothing,
            "from_logits": self.from_logits,
        })
        return config


@keras.saving.register_keras_serializable()
class PrefixMaskedCausalLMLoss(MaskedCausalLMLoss):
    """Causal LM loss that also masks a variable-length prefix per sample.

    Useful for instruction tuning where the prompt should not contribute
    to the loss — only the model's response is scored.

    The prefix length per sample is communicated via ``y_true``:
    set prompt token positions to ``ignore_index`` in the labels during
    preprocessing. This class adds no extra logic over the base — it
    exists as a semantic alias with documentation for the prefix use case.

    .. code-block:: text

        Labels:  [ -1, -1, -1,  42,  17, 305,  -1 ]
                   ↑ prompt ↑    ↑ response ↑   ↑pad

    :param ignore_index: Label value to ignore. Default ``-1``.
    :param label_smoothing: Smoothing factor. Default ``0.0``.
    :param from_logits: Whether ``y_pred`` is logits. Default ``True``.
    :param name: Name for the loss instance.
    """

    def __init__(
        self,
        ignore_index: int = -1,
        label_smoothing: float = 0.0,
        from_logits: bool = True,
        name: str = "prefix_masked_causal_lm_loss",
        **kwargs,
    ):
        super().__init__(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            from_logits=from_logits,
            name=name,
            **kwargs,
        )
