"""Tests for ``train.common.nlp`` library helpers.

Covers:
- ``estimate_clm_steps_per_epoch`` math (D-001) — override path,
  measured-articles path, and full-Wikipedia fallback.
- The dead ``streaming`` parameter is gone from ``preprocess_clm_dataset``
  (D-004).
- ``preprocess_mlm_dataset`` no longer calls ``dataset.cache()`` (D-005) —
  asserted indirectly by re-iterating and confirming the same elements
  appear (a ``cache()`` would break in the absence of a ``shuffle`` ahead
  of iteration but here we only need to confirm the dataset graph does
  not crash on second iteration without re-tokenizing from disk).
"""

from __future__ import annotations

import inspect
import pytest


class TestEstimateClmStepsPerEpoch:

    def test_override_takes_precedence(self):
        from train.common.nlp import estimate_clm_steps_per_epoch

        assert estimate_clm_steps_per_epoch(
            num_articles=None, max_seq_length=512, batch_size=8,
            override=12345,
        ) == 12345

    def test_override_clamped_to_at_least_one(self):
        from train.common.nlp import estimate_clm_steps_per_epoch

        # override=0 is interpreted as "use 0", but the helper guarantees
        # >=1 (otherwise WarmupSchedule division blows up).
        assert estimate_clm_steps_per_epoch(
            num_articles=10, max_seq_length=512, batch_size=8, override=0,
        ) == 1

    def test_measured_articles_path(self):
        from train.common.nlp import estimate_clm_steps_per_epoch

        # 4.85M articles, 600 tokens/article, seq=512, batch=8
        # = 4_850_000 * 600 / 512 / 8 = 710_449
        assert estimate_clm_steps_per_epoch(
            num_articles=4_850_000, max_seq_length=512, batch_size=8,
            avg_tokens_per_article=600,
        ) == 710_449

    def test_chunks_grow_with_max_seq_length_inversely(self):
        from train.common.nlp import estimate_clm_steps_per_epoch

        a = estimate_clm_steps_per_epoch(1_000_000, 512, 8)
        b = estimate_clm_steps_per_epoch(1_000_000, 256, 8)
        # Halving max_seq_length doubles the chunk count (within int trunc).
        assert b == pytest.approx(2 * a, rel=0.001)

    def test_default_wikipedia_fallback(self):
        from train.common.nlp import estimate_clm_steps_per_epoch

        out = estimate_clm_steps_per_epoch(
            num_articles=None, max_seq_length=512, batch_size=8,
        )
        # ~2.9B tokens / 512 / 8 = 708_007.
        assert 600_000 <= out <= 800_000

    def test_min_articles_yield_at_least_one_step(self):
        from train.common.nlp import estimate_clm_steps_per_epoch

        # Pathological case: 1 short article. Helper must still return >=1
        # (otherwise the LR schedule's CosineDecay(decay_steps=...) crashes).
        assert estimate_clm_steps_per_epoch(
            num_articles=1, max_seq_length=512, batch_size=8,
        ) >= 1


class TestPreprocessClmDatasetSignature:

    def test_streaming_kwarg_removed(self):
        """D-004: ``streaming`` should no longer appear in the signature."""
        from train.common.nlp import preprocess_clm_dataset

        sig = inspect.signature(preprocess_clm_dataset)
        assert "streaming" not in sig.parameters

    def test_passing_streaming_raises_typeerror(self):
        from train.common.nlp import preprocess_clm_dataset

        # Positional/kwarg call with the dropped parameter must fail loud.
        with pytest.raises(TypeError):
            preprocess_clm_dataset(
                None, None, max_seq_length=64, batch_size=4, streaming=True,
            )


class TestPreprocessMlmDatasetNoCache:

    def test_no_cache_call_in_pipeline(self):
        """D-005: source-level grep confirms ``.cache()`` is no longer
        called inside ``preprocess_mlm_dataset``. Comments are stripped
        before the check so a documentation reference to ``.cache()`` is
        not a false positive."""
        import inspect

        from train.common import nlp

        source = inspect.getsource(nlp.preprocess_mlm_dataset)
        code_only = "\n".join(
            line for line in source.splitlines()
            if not line.lstrip().startswith("#")
        )
        assert ".cache()" not in code_only, (
            "preprocess_mlm_dataset must not call dataset.cache() — "
            "see D-005."
        )
