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
import numpy as np
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


class TestDictKeyedCompile:
    """Coverage for ``prepare_dict_keyed_compile`` (D-001).

    The helper makes ``model.compile(metrics={"logits": [...]})`` actually
    fire its metrics on subclassed Keras models that return a dict from
    ``call()``. Without it, Keras silently no-ops the metric list — the
    bug this plan fixes.

    Tests c/d run a real ``model.fit`` on a tiny config to assert all 4
    CLM metrics appear in ``history.history`` with clean (non-prefixed)
    names. CPU only — no GPU required.
    """

    def test_helper_sets_output_names_on_subclassed_dict_model(self):
        """SC-1 (a): empty -> ['logits']."""
        import keras

        from train.common.nlp import prepare_dict_keyed_compile

        from dl_techniques.models.gpt2.gpt2 import GPT2

        m = GPT2.from_variant("tiny", vocab_size=64, max_seq_len=8)
        # Build the model so internal state is populated.
        _ = m({"input_ids": np.zeros((1, 4), dtype=np.int32)}, training=False)
        # Pre-condition: subclassed dict-output model has no output_names.
        assert not getattr(m, "output_names", None)

        prepare_dict_keyed_compile(m)
        assert m.output_names == ["logits"]

    def test_helper_is_idempotent(self):
        """SC-1 (b): no-op when already populated."""
        import keras

        from train.common.nlp import prepare_dict_keyed_compile

        m = keras.Sequential([keras.layers.Dense(2)])
        m.build((None, 3))
        prepare_dict_keyed_compile(m)  # may or may not set, depending on Keras
        first = list(m.output_names) if m.output_names else None
        # Pretend it was set explicitly with a different value -> stays put.
        m.output_names = ["custom"]
        prepare_dict_keyed_compile(m)
        assert m.output_names == ["custom"]

    def _assert_clm_metric_history(self, history):
        keys = set(history.history.keys())
        required = {"loss", "accuracy", "perplexity",
                    "bits_per_token", "bits_per_character"}
        missing = required - keys
        assert not missing, (
            f"missing metric keys in history: {missing}; got {sorted(keys)}"
        )
        # No `logits_` prefix on metrics — the whole point of the fix.
        prefixed = [k for k in keys if k.startswith("logits_")]
        assert not prefixed, (
            f"unexpected logits_-prefixed keys in history: {prefixed}"
        )
        # SC-4: loss is finite.
        loss_seq = history.history["loss"]
        assert len(loss_seq) >= 1
        assert np.isfinite(loss_seq[-1]), f"non-finite loss: {loss_seq}"

    def test_fit_gpt2_emits_all_metrics(self):
        """SC-2: real ``model.fit`` on tiny GPT2 emits all 4 metric keys."""
        import keras

        from train.common.nlp import (
            prepare_dict_keyed_compile,
            build_clm_metrics,
        )
        from dl_techniques.models.gpt2.gpt2 import GPT2
        from dl_techniques.losses import MaskedCausalLMLoss

        vocab_size = 64
        seq_in = 7  # input length after the causal shift (chunk-1)
        batch = 2
        n_samples = 4

        m = GPT2.from_variant(
            "tiny", vocab_size=vocab_size, max_seq_len=seq_in,
        )
        prepare_dict_keyed_compile(m)
        m.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss={"logits": MaskedCausalLMLoss(ignore_index=-1)},
            metrics={"logits": build_clm_metrics("gpt2")},
        )
        rng = np.random.default_rng(0)
        x = rng.integers(0, vocab_size, size=(n_samples, seq_in), dtype=np.int32)
        y = rng.integers(0, vocab_size, size=(n_samples, seq_in), dtype=np.int32)
        history = m.fit(x, y, batch_size=batch, epochs=1, verbose=0)
        self._assert_clm_metric_history(history)

    def test_fit_cliffordnet_lm_emits_all_metrics(self):
        """SC-3: real ``model.fit`` on tiny CliffordNetLM (single-key dict)."""
        import keras

        from train.common.nlp import (
            prepare_dict_keyed_compile,
            build_clm_metrics,
        )
        from dl_techniques.models.cliffordnet.lm import CliffordNetLM
        from dl_techniques.losses import MaskedCausalLMLoss

        vocab_size = 64
        seq_in = 8
        batch = 2
        n_samples = 4

        m = CliffordNetLM.from_variant(
            "nano", vocab_size=vocab_size, max_seq_length=seq_in,
        )
        prepare_dict_keyed_compile(m)
        m.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss={"logits": MaskedCausalLMLoss(ignore_index=-1)},
            metrics={"logits": build_clm_metrics("gpt2")},
        )
        rng = np.random.default_rng(0)
        x = rng.integers(0, vocab_size, size=(n_samples, seq_in), dtype=np.int32)
        y = rng.integers(0, vocab_size, size=(n_samples, seq_in), dtype=np.int32)
        history = m.fit(x, y, batch_size=batch, epochs=1, verbose=0)
        self._assert_clm_metric_history(history)
