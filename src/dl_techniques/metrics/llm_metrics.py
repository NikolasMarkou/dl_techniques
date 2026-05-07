"""
LLM-oriented evaluation metrics for causal language modeling pipelines.

This module provides a small, DRY collection of language-model evaluation
metrics that fit into ``model.compile(metrics=...)`` plus a handful of
pure-Python text-quality helpers that are intended to be invoked from a
generation-probe ``_post_generate_hook`` extension point.

## Compile-time metrics (per-batch on GPU)

- ``BitsPerToken`` -- accumulates fp32 cross-entropy and returns
  ``(total_loss / count) / ln(2)``. Equivalently ``log2(perplexity)``.
- ``BitsPerCharacter`` -- same accumulator, divided by a
  ``chars_per_token`` config constant. Display-time approximation; for
  GPT-2 BPE on English the empirical ratio is ~3.6-4.0 chars/token.

Both metrics:

- inherit ``keras.metrics.Metric``,
- accept ``from_logits`` (default True), ``ignore_class`` (default None),
- carry a fp32 accumulator (AMP-safe even if upstream activations are
  ``mixed_float16``),
- are decorated with ``@keras.saving.register_keras_serializable``,
- expose full ``get_config`` round-trip.

The CE math mirrors ``dl_techniques.metrics.perplexity_metric.Perplexity``
exactly so that ``BitsPerToken == log2(Perplexity)`` to floating-point
precision.

## Probe-time helpers (free functions, NLTK-free)

- ``self_bleu(texts, n=4)`` -- mean self-BLEU over a small list of probe
  outputs. Pure-Python n-gram overlap with the corpus-BLEU brevity
  penalty omitted for stability on very short generations. Returns a
  scalar in ``[0.0, 1.0]``; high means repetitive.
- ``distinct_n(texts, n=2)`` -- fraction of distinct n-grams across the
  concatenated probe outputs. Returns a scalar in ``[0.0, 1.0]``; high
  means lexically diverse.
- ``aggregate_probe_metrics(results)`` -- in-place augments a probe
  ``results`` dict (the ``_post_generate_hook`` payload) with three new
  keys: ``self_bleu``, ``distinct_2``, ``mean_tok_per_s``. Schema-tolerant:
  defensive ``.get(...)`` reads, log-and-skip on shape error.

The helpers are deliberately deps-free: NLTK is **not** in this project's
dependency set.

## Mathematical notes

Given per-token cross-entropy ``H`` (in nats), the standard relations are::

    PPL = exp(H)
    BPT = H / ln(2)         # bits-per-token, == log2(PPL)
    BPC = BPT / cpt         # cpt = chars_per_token (config constant)

The metric reduction is mean cross-entropy over non-ignored tokens
(``total_loss / count``), matching ``Perplexity`` and standard CLM
``SparseCategoricalCrossentropy(from_logits=True)`` reduction.

Example::

    from dl_techniques.metrics.llm_metrics import BitsPerToken, BitsPerCharacter
    from dl_techniques.metrics.perplexity_metric import Perplexity
    import keras

    model.compile(
        optimizer=keras.optimizers.AdamW(...),
        loss={"logits": MaskedCausalLMLoss(...)},
        metrics={"logits": [
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            Perplexity(ignore_class=-1),
            BitsPerToken(ignore_class=-1),
            BitsPerCharacter(chars_per_token=4.0, ignore_class=-1),
        ]},
    )
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import keras

from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------
# Internal CE helper -- shared by BitsPerToken and BitsPerCharacter.
# Mirrors dl_techniques.metrics.perplexity_metric.Perplexity.update_state.
# ---------------------------------------------------------------------

def _accumulate_cross_entropy(
        y_true: keras.KerasTensor,
        y_pred: keras.KerasTensor,
        from_logits: bool,
        ignore_class: Optional[int],
        sample_weight: Optional[keras.KerasTensor],
):
    """Compute (sum_cross_entropy, valid_count) tensors in fp32.

    The accumulator dtype is fp32 to remain AMP-safe even if upstream
    activations are ``mixed_float16``.
    """
    if from_logits:
        y_pred = keras.ops.softmax(y_pred, axis=-1)

    epsilon = keras.backend.epsilon()
    y_pred = keras.ops.clip(y_pred, epsilon, 1.0 - epsilon)

    y_true_int = keras.ops.cast(y_true, "int32")
    num_classes = keras.ops.shape(y_pred)[-1]
    y_true_one_hot = keras.ops.one_hot(y_true_int, num_classes)

    # Cast to fp32 for numerically-stable accumulation.
    y_true_one_hot = keras.ops.cast(y_true_one_hot, "float32")
    y_pred_f32 = keras.ops.cast(y_pred, "float32")

    cross_entropy = -keras.ops.sum(
        y_true_one_hot * keras.ops.log(y_pred_f32), axis=-1,
    )

    if ignore_class is not None:
        mask = keras.ops.not_equal(y_true_int, ignore_class)
        mask = keras.ops.cast(mask, "float32")
        cross_entropy = cross_entropy * mask
        if sample_weight is not None:
            sample_weight = keras.ops.cast(sample_weight, "float32")
            cross_entropy = cross_entropy * sample_weight
            valid_count = keras.ops.sum(mask * sample_weight)
        else:
            valid_count = keras.ops.sum(mask)
    else:
        if sample_weight is not None:
            sample_weight = keras.ops.cast(sample_weight, "float32")
            cross_entropy = cross_entropy * sample_weight
            valid_count = keras.ops.sum(sample_weight)
        else:
            valid_count = keras.ops.cast(
                keras.ops.size(cross_entropy), "float32",
            )

    total_cross_entropy = keras.ops.sum(cross_entropy)
    return total_cross_entropy, valid_count


# ---------------------------------------------------------------------
# BitsPerToken
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable(package="dl_techniques.metrics")
class BitsPerToken(keras.metrics.Metric):
    """Bits-per-token (== ``log2(perplexity)``) for CLM evaluation.

    Args:
        from_logits: Whether ``y_pred`` is logits (True) or probabilities
            (False). Defaults to True.
        ignore_class: Optional integer class id to mask out (e.g. a
            ``label_pad_id`` like -1 or -100). Defaults to None.
        name: Metric display name. Defaults to ``'bits_per_token'``.
        dtype: Output dtype. Internal accumulator is always fp32.

    Example::

        model.compile(metrics={"logits": [BitsPerToken(ignore_class=-1)]})
    """

    def __init__(
            self,
            from_logits: bool = True,
            ignore_class: Optional[int] = None,
            name: str = "bits_per_token",
            dtype: Optional[str] = None,
            **kwargs,
    ):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.from_logits = from_logits
        self.ignore_class = ignore_class

        # Explicit fp32 accumulator -- AMP-safe.
        self.total_loss = self.add_weight(
            name="total_loss", initializer="zeros", dtype="float32",
        )
        self.count = self.add_weight(
            name="count", initializer="zeros", dtype="float32",
        )

    def update_state(
            self,
            y_true: keras.KerasTensor,
            y_pred: keras.KerasTensor,
            sample_weight: Optional[keras.KerasTensor] = None,
    ) -> None:
        total_ce, valid_count = _accumulate_cross_entropy(
            y_true, y_pred, self.from_logits, self.ignore_class, sample_weight,
        )
        self.total_loss.assign_add(total_ce)
        self.count.assign_add(valid_count)

    def result(self) -> keras.KerasTensor:
        avg_ce = keras.ops.divide_no_nan(self.total_loss, self.count)
        # nats -> bits
        ln2 = keras.ops.cast(math.log(2.0), "float32")
        return avg_ce / ln2

    def reset_state(self) -> None:
        self.total_loss.assign(0.0)
        self.count.assign(0.0)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "from_logits": self.from_logits,
            "ignore_class": self.ignore_class,
        })
        return config


# ---------------------------------------------------------------------
# BitsPerCharacter
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable(package="dl_techniques.metrics")
class BitsPerCharacter(keras.metrics.Metric):
    """Bits-per-character (== ``BitsPerToken / chars_per_token``).

    ``chars_per_token`` is a display-time configuration constant (no
    decode happens inside the training loop). Defaults to 4.0, the
    typical empirical ratio for GPT-2 BPE on English text. Override per
    encoding/dataset as needed.

    Args:
        chars_per_token: Float divisor applied to bits-per-token.
            Defaults to 4.0.
        from_logits: Whether ``y_pred`` is logits. Defaults to True.
        ignore_class: Optional integer class id to mask. Defaults to None.
        name: Metric display name. Defaults to ``'bits_per_character'``.
        dtype: Output dtype. Internal accumulator is always fp32.
    """

    def __init__(
            self,
            chars_per_token: float = 4.0,
            from_logits: bool = True,
            ignore_class: Optional[int] = None,
            name: str = "bits_per_character",
            dtype: Optional[str] = None,
            **kwargs,
    ):
        super().__init__(name=name, dtype=dtype, **kwargs)
        if chars_per_token <= 0.0:
            raise ValueError(
                f"chars_per_token must be > 0, got {chars_per_token}",
            )
        self.chars_per_token = float(chars_per_token)
        self.from_logits = from_logits
        self.ignore_class = ignore_class

        self.total_loss = self.add_weight(
            name="total_loss", initializer="zeros", dtype="float32",
        )
        self.count = self.add_weight(
            name="count", initializer="zeros", dtype="float32",
        )

    def update_state(
            self,
            y_true: keras.KerasTensor,
            y_pred: keras.KerasTensor,
            sample_weight: Optional[keras.KerasTensor] = None,
    ) -> None:
        total_ce, valid_count = _accumulate_cross_entropy(
            y_true, y_pred, self.from_logits, self.ignore_class, sample_weight,
        )
        self.total_loss.assign_add(total_ce)
        self.count.assign_add(valid_count)

    def result(self) -> keras.KerasTensor:
        avg_ce = keras.ops.divide_no_nan(self.total_loss, self.count)
        ln2 = keras.ops.cast(math.log(2.0), "float32")
        bpt = avg_ce / ln2
        return bpt / keras.ops.cast(self.chars_per_token, "float32")

    def reset_state(self) -> None:
        self.total_loss.assign(0.0)
        self.count.assign(0.0)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "chars_per_token": self.chars_per_token,
            "from_logits": self.from_logits,
            "ignore_class": self.ignore_class,
        })
        return config


# ---------------------------------------------------------------------
# Probe-time helpers (pure Python, NLTK-free).
# ---------------------------------------------------------------------

def _whitespace_tokenize(text: str) -> List[str]:
    """Split on whitespace -- adequate for diversity-style metrics over
    short probe outputs. Avoids importing tiktoken into the helpers."""
    if not text:
        return []
    return text.split()


def _ngrams(tokens: Sequence[str], n: int) -> List[tuple]:
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def self_bleu(texts: List[str], n: int = 4) -> float:
    """Mean self-BLEU(n) over a small list of probe outputs.

    For each text, treat the others as references and compute the
    fraction of n-grams in the candidate that also appear in the
    reference set. The brevity penalty is omitted (probe outputs are
    short and length-controlled by ``max_tokens``). Returns a value in
    ``[0.0, 1.0]``; higher means more repetition across samples.

    Args:
        texts: List of generated strings. Length < 2 returns 0.0.
        n: N-gram order. Defaults to 4.

    Returns:
        Float self-BLEU score.
    """
    if not texts or len(texts) < 2 or n <= 0:
        return 0.0

    tokenized = [_whitespace_tokenize(t) for t in texts]
    scores: List[float] = []
    for i, cand_tokens in enumerate(tokenized):
        cand_ngrams = _ngrams(cand_tokens, n)
        if not cand_ngrams:
            continue
        ref_set = set()
        for j, ref_tokens in enumerate(tokenized):
            if i == j:
                continue
            ref_set.update(_ngrams(ref_tokens, n))
        if not ref_set:
            continue
        overlap = sum(1 for g in cand_ngrams if g in ref_set)
        scores.append(overlap / float(len(cand_ngrams)))

    if not scores:
        return 0.0
    return sum(scores) / float(len(scores))


def distinct_n(texts: List[str], n: int = 2) -> float:
    """Distinct-n diversity over the concatenated probe outputs.

    Returns ``len(set(ngrams)) / len(ngrams)`` across all texts, in
    ``[0.0, 1.0]``. Higher means more lexically diverse.

    Args:
        texts: List of generated strings.
        n: N-gram order. Defaults to 2.

    Returns:
        Float distinct-n score, or 0.0 when no n-grams exist.
    """
    if not texts or n <= 0:
        return 0.0
    all_ngrams: List[tuple] = []
    for t in texts:
        all_ngrams.extend(_ngrams(_whitespace_tokenize(t), n))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / float(len(all_ngrams))


def aggregate_probe_metrics(results: dict) -> None:
    """In-place augment a generation-probe ``results`` dict with diversity
    and aggregate-throughput metrics.

    Designed to be bound to ``GenerationProbeCallback._post_generate_hook``
    in a single line per trainer::

        probe_cb = GenerationProbeCallback(...)
        probe_cb._post_generate_hook = augment_probe_results

    Reads:
      - ``results["generations"]`` -- expected list of dicts each with
        keys ``output`` (str) and ``tok_per_s`` (float). Defensive
        ``.get(...)`` reads tolerate schema drift.

    Writes (in place):
      - ``results["self_bleu"]``       -- mean self-BLEU(4) over outputs.
      - ``results["distinct_2"]``      -- distinct-2 over outputs.
      - ``results["mean_tok_per_s"]``  -- mean tok/s across generations.

    Logging-only; never raises on shape mismatch.
    """
    try:
        generations = results.get("generations", []) if isinstance(
            results, dict,
        ) else []
        if not isinstance(generations, list):
            logger.warning(
                "aggregate_probe_metrics: generations not a list; skipping",
            )
            return

        outputs: List[str] = []
        tok_per_s_vals: List[float] = []
        for entry in generations:
            if not isinstance(entry, dict):
                continue
            out = entry.get("output", "")
            if isinstance(out, str):
                outputs.append(out)
            tps = entry.get("tok_per_s")
            if isinstance(tps, (int, float)):
                tok_per_s_vals.append(float(tps))

        sb = self_bleu(outputs, n=4)
        d2 = distinct_n(outputs, n=2)
        mean_tps = (
            sum(tok_per_s_vals) / float(len(tok_per_s_vals))
            if tok_per_s_vals else 0.0
        )

        results["self_bleu"] = round(sb, 4)
        results["distinct_2"] = round(d2, 4)
        results["mean_tok_per_s"] = round(mean_tps, 2)

        logger.info(
            f"Probe diversity: self_bleu={sb:.3f} distinct_2={d2:.3f} "
            f"mean_tok_per_s={mean_tps:.1f}",
        )
    except Exception as exc:  # noqa: BLE001 -- probe must never kill train
        logger.warning(
            f"aggregate_probe_metrics: skipped due to error: {exc}",
        )


# ---------------------------------------------------------------------
