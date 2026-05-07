"""Tests for ``dl_techniques.metrics.llm_metrics`` and the
``train.common.nlp.build_clm_metrics`` builder.

Coverage:
  - BitsPerToken matches log2(Perplexity) on synthetic logits.
  - BitsPerCharacter == BitsPerToken / chars_per_token.
  - ignore_class excludes pad positions consistently with Perplexity.
  - Serialization round-trip (get_config/from_config) for both metrics.
  - self_bleu monotonic on diverse-vs-repeated text.
  - distinct_n on empty/single-token/many-tokens.
  - aggregate_probe_metrics augments a sample results dict in place.
  - build_clm_metrics returns [Acc, PPL, BPT, BPC] in order.
"""

from __future__ import annotations

import math

import keras
import numpy as np
import pytest

from dl_techniques.metrics.perplexity_metric import Perplexity
from dl_techniques.metrics.llm_metrics import (
    BitsPerToken,
    BitsPerCharacter,
    self_bleu,
    distinct_n,
    aggregate_probe_metrics,
)


# ---------------------------------------------------------------------
# Compile-time metrics
# ---------------------------------------------------------------------

class TestBitsPerToken:
    """Verify BPT == log2(PPL) on synthetic logits + labels."""

    def test_matches_log2_ppl(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 8, size=(4, 16)).astype("int32")
        y_pred = rng.standard_normal((4, 16, 8)).astype("float32")

        bpt = BitsPerToken(from_logits=True)
        ppl = Perplexity(from_logits=True)
        bpt.update_state(y_true, y_pred)
        ppl.update_state(y_true, y_pred)

        bpt_val = float(bpt.result())
        ppl_val = float(ppl.result())
        assert math.isfinite(bpt_val)
        assert math.isfinite(ppl_val)
        # log2(PPL) == BPT (within fp32 tolerance)
        assert abs(bpt_val - math.log2(ppl_val)) < 1e-5

    def test_perfect_prediction(self):
        """Confident-correct logits -> BPT close to 0."""
        y_true = np.array([0, 1, 2], dtype="int32")
        y_pred = np.array([
            [100.0, -100.0, -100.0],
            [-100.0, 100.0, -100.0],
            [-100.0, -100.0, 100.0],
        ], dtype="float32")
        bpt = BitsPerToken(from_logits=True)
        bpt.update_state(y_true, y_pred)
        assert float(bpt.result()) < 0.05

    def test_uniform_prediction(self):
        """Uniform logits over V classes -> BPT close to log2(V)."""
        num_classes = 8
        y_true = np.arange(num_classes, dtype="int32")
        y_pred = np.zeros((num_classes, num_classes), dtype="float32")
        bpt = BitsPerToken(from_logits=True)
        bpt.update_state(y_true, y_pred)
        assert abs(float(bpt.result()) - math.log2(num_classes)) < 0.05

    def test_ignore_class_excludes_pad(self):
        """Setting ignore_class on some labels matches Perplexity behaviour."""
        rng = np.random.default_rng(1)
        y_true = rng.integers(0, 4, size=(2, 8)).astype("int32")
        # Force some positions to ignore_id=-1
        y_true[0, 0] = -1
        y_true[1, 3] = -1
        y_pred = rng.standard_normal((2, 8, 4)).astype("float32")

        bpt = BitsPerToken(from_logits=True, ignore_class=-1)
        ppl = Perplexity(from_logits=True, ignore_class=-1)
        bpt.update_state(y_true, y_pred)
        ppl.update_state(y_true, y_pred)
        bpt_val = float(bpt.result())
        ppl_val = float(ppl.result())
        assert abs(bpt_val - math.log2(ppl_val)) < 1e-5

    def test_reset_state(self):
        bpt = BitsPerToken(from_logits=True)
        y_true = np.array([0], dtype="int32")
        y_pred = np.array([[1.0, 0.0, 0.0]], dtype="float32")
        bpt.update_state(y_true, y_pred)
        bpt.reset_state()
        # No data -> result is divide_no_nan -> 0.0
        assert float(bpt.result()) == pytest.approx(0.0)

    def test_serialization_roundtrip(self):
        m = BitsPerToken(
            from_logits=True, ignore_class=-100, name="my_bpt",
        )
        cfg = m.get_config()
        m2 = BitsPerToken.from_config(cfg)
        assert m2.from_logits is True
        assert m2.ignore_class == -100
        assert m2.name == "my_bpt"


class TestBitsPerCharacter:
    """Verify BPC == BPT / chars_per_token."""

    def test_division_by_chars_per_token(self):
        rng = np.random.default_rng(2)
        y_true = rng.integers(0, 5, size=(3, 12)).astype("int32")
        y_pred = rng.standard_normal((3, 12, 5)).astype("float32")

        cpt = 4.0
        bpt = BitsPerToken(from_logits=True)
        bpc = BitsPerCharacter(chars_per_token=cpt, from_logits=True)
        bpt.update_state(y_true, y_pred)
        bpc.update_state(y_true, y_pred)

        bpt_val = float(bpt.result())
        bpc_val = float(bpc.result())
        assert abs(bpc_val - bpt_val / cpt) < 1e-5

    def test_rejects_non_positive_chars_per_token(self):
        with pytest.raises(ValueError):
            BitsPerCharacter(chars_per_token=0.0)
        with pytest.raises(ValueError):
            BitsPerCharacter(chars_per_token=-1.0)

    def test_serialization_roundtrip(self):
        m = BitsPerCharacter(
            chars_per_token=3.5, from_logits=True, ignore_class=-1,
            name="my_bpc",
        )
        cfg = m.get_config()
        m2 = BitsPerCharacter.from_config(cfg)
        assert m2.chars_per_token == pytest.approx(3.5)
        assert m2.from_logits is True
        assert m2.ignore_class == -1
        assert m2.name == "my_bpc"


# ---------------------------------------------------------------------
# Probe-time helpers
# ---------------------------------------------------------------------

class TestSelfBleu:

    def test_empty_returns_zero(self):
        assert self_bleu([]) == 0.0
        assert self_bleu(["only one"]) == 0.0

    def test_repeated_high_score(self):
        repeated = ["the quick brown fox jumps over"] * 4
        assert self_bleu(repeated, n=2) > 0.9

    def test_diverse_lower_than_repeated(self):
        diverse = [
            "alpha beta gamma delta epsilon zeta",
            "rouge violet jaune indigo cyan magenta",
            "norse celtic gaelic latin germanic slavic",
            "iron copper zinc nickel cobalt manganese",
        ]
        repeated = ["the quick brown fox jumps over the lazy dog"] * 4
        assert self_bleu(diverse, n=4) < self_bleu(repeated, n=4)


class TestDistinctN:

    def test_empty(self):
        assert distinct_n([]) == 0.0

    def test_single_short_text_below_n(self):
        # Single token -> no bigrams -> 0.0
        assert distinct_n(["hello"], n=2) == 0.0

    def test_all_distinct(self):
        # All bigrams unique
        texts = ["a b c d e f"]
        assert distinct_n(texts, n=2) == pytest.approx(1.0)

    def test_repeats_lower_diversity(self):
        repeated = ["the the the the the the"]
        diverse = ["one two three four five six"]
        assert distinct_n(repeated, n=2) < distinct_n(diverse, n=2)


class TestAggregateProbeMetrics:

    def test_augments_in_place(self):
        results = {
            "step": 100,
            "generations": [
                {"prompt": "p1", "output": "alpha beta gamma delta",
                 "tok_per_s": 50.0},
                {"prompt": "p2", "output": "rouge violet jaune indigo",
                 "tok_per_s": 60.0},
                {"prompt": "p3", "output": "iron copper zinc nickel",
                 "tok_per_s": 40.0},
            ],
        }
        aggregate_probe_metrics(results)
        assert "self_bleu" in results
        assert "distinct_2" in results
        assert "mean_tok_per_s" in results
        assert isinstance(results["self_bleu"], float)
        assert isinstance(results["distinct_2"], float)
        assert results["mean_tok_per_s"] == pytest.approx(50.0, abs=0.1)

    def test_tolerates_missing_keys(self):
        results = {"generations": [{"prompt": "p"}]}  # no output, no tok/s
        # Must not raise.
        aggregate_probe_metrics(results)
        assert results["mean_tok_per_s"] == 0.0

    def test_tolerates_non_dict(self):
        # Schema-tolerant: bad shape -> just skip.
        results = {"generations": "not-a-list"}
        aggregate_probe_metrics(results)  # no raise
        # No augmented keys should appear.
        assert "self_bleu" not in results


# ---------------------------------------------------------------------
# build_clm_metrics smoke
# ---------------------------------------------------------------------

class TestBuildClmMetrics:

    def test_returns_expected_metric_list(self):
        from train.common.nlp import build_clm_metrics

        metrics = build_clm_metrics("gpt2")
        assert len(metrics) == 4
        assert isinstance(metrics[0], keras.metrics.SparseCategoricalAccuracy)
        assert isinstance(metrics[1], Perplexity)
        assert isinstance(metrics[2], BitsPerToken)
        assert isinstance(metrics[3], BitsPerCharacter)
        # Metric names match the canonical CLM dashboard.
        names = [m.name for m in metrics]
        assert names == [
            "accuracy", "perplexity", "bits_per_token", "bits_per_character",
        ]

    def test_ignore_index_propagates(self):
        from train.common.nlp import build_clm_metrics

        metrics = build_clm_metrics("gpt2", ignore_index=-100)
        assert metrics[1].ignore_class == -100  # PPL
        assert metrics[2].ignore_class == -100  # BPT
        assert metrics[3].ignore_class == -100  # BPC

    def test_chars_per_token_override(self):
        from train.common.nlp import build_clm_metrics

        metrics = build_clm_metrics("gpt2", chars_per_token=5.0)
        assert metrics[3].chars_per_token == pytest.approx(5.0)

    def test_augment_probe_results_reexport(self):
        from train.common.nlp import augment_probe_results
        from dl_techniques.metrics.llm_metrics import (
            aggregate_probe_metrics,
        )
        # Re-export should be the same object.
        assert augment_probe_results is aggregate_probe_metrics

    def test_fresh_instances_per_call(self):
        from train.common.nlp import build_clm_metrics

        a = build_clm_metrics("gpt2")
        b = build_clm_metrics("gpt2")
        # Distinct instances (Keras requires unique state per compile).
        for ma, mb in zip(a, b):
            assert ma is not mb
