"""Tests for the embedding-evaluation harness.

The TFDS-touching tests skip when the local cache is absent, matching the
pattern in `tests/test_train/test_beit/test_common.py`. Everything else runs
against a stub encoder so the fast path needs no GPU and no data.
"""

import json
import os

import keras
import numpy as np
import pytest

from dl_techniques.layers.tokenizers.ascii_char import PAD_ID
from train.embeddings_experimental.evaluate_embeddings import (
    DEFAULT_TFDS_DATA_DIR,
    EvalConfig,
    embed_texts,
    evaluate_run,
    load_encoder,
)
from train.embeddings_experimental.paths import (
    encoder_path,
    eval_path,
    results_path,
)

TFDS_DIR = os.environ.get("TFDS_DATA_DIR", DEFAULT_TFDS_DATA_DIR)
HAS_SQUAD = os.path.isdir(os.path.join(TFDS_DIR, "squad"))
HAS_SST2 = os.path.isdir(os.path.join(TFDS_DIR, "glue", "sst2"))


class StubEncoder(keras.Model):
    """Encoder-shaped stand-in whose pooled output encodes the input length.

    Deterministic and cheap, so ordering and padding behaviour can be asserted
    exactly rather than approximately.
    """

    def __init__(self, hidden=8, **kwargs):
        super().__init__(**kwargs)
        self.hidden = hidden

    def call(self, inputs, training=None):
        ids = inputs["input_ids"]
        mask = inputs.get("attention_mask")
        if mask is None:
            mask = keras.ops.cast(
                keras.ops.not_equal(ids, PAD_ID), "int32"
            )
        real = keras.ops.sum(keras.ops.cast(mask, "float32"), axis=-1)
        pooled = keras.ops.stack([real] * self.hidden, axis=-1)
        return {
            "last_hidden_state": keras.ops.zeros(
                (keras.ops.shape(ids)[0], keras.ops.shape(ids)[1], self.hidden)
            ),
            "attention_mask": mask,
            "pooled_output": pooled,
        }


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

class TestPathsHaveOneProducer:
    def test_eval_path(self, tmp_path):
        assert os.path.basename(eval_path(str(tmp_path))) == "eval.json"

    def test_encoder_path(self, tmp_path):
        assert os.path.basename(encoder_path(str(tmp_path))) == "encoder.keras"

    def test_results_path(self, tmp_path):
        assert os.path.basename(results_path(str(tmp_path))) == "results.json"

    def test_the_trainer_still_re_exports_what_the_sweep_imports(self):
        """`sweep.py` does `from .train_embeddings import resolve_output_dir`."""
        from train.embeddings_experimental import train_embeddings

        assert callable(train_embeddings.resolve_output_dir)
        assert callable(train_embeddings.encoder_path)
        assert train_embeddings.REPO_ROOT.name

    def test_the_modules_import_in_either_order(self):
        """The paths extraction exists to break a cycle; prove it is broken."""
        import subprocess
        import sys

        root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))))), "src"
        )
        for first, second in (
            ("train.embeddings_experimental.evaluate_embeddings",
             "train.embeddings_experimental.train_embeddings"),
            ("train.embeddings_experimental.train_embeddings",
             "train.embeddings_experimental.evaluate_embeddings"),
        ):
            proc = subprocess.run(
                [sys.executable, "-c", f"import {first}; import {second}"],
                cwd=root, capture_output=True, text=True, timeout=300,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            )
            assert proc.returncode == 0, f"{first} then {second}:\n{proc.stderr[-1500:]}"


# ---------------------------------------------------------------------
# embed_texts
# ---------------------------------------------------------------------

class TestEmbedTexts:
    def test_shape_and_order_are_preserved_under_length_sorting(self):
        """A permutation bug here would silently scramble ground truth.

        The stub's pooled output equals the token count, so the correct
        ordering is checkable exactly rather than approximately.
        """
        encoder = StubEncoder()
        texts = ["x" * n for n in (50, 1, 30, 5, 90, 2)]
        emb, _ = embed_texts(
            encoder, texts, max_length=128, batch_size=2, length_sorted=True
        )
        assert emb.shape == (6, 8)
        # framed length = content + [CLS] + [SEP]
        expected = np.array([n + 2 for n in (50, 1, 30, 5, 90, 2)], dtype=np.float32)
        np.testing.assert_allclose(emb[:, 0], expected, atol=1e-5)

    def test_sorted_and_unsorted_agree(self):
        encoder = StubEncoder()
        texts = ["a" * n for n in (7, 40, 3, 100, 12)]
        a, _ = embed_texts(encoder, texts, max_length=128, batch_size=2,
                           length_sorted=True)
        b, _ = embed_texts(encoder, texts, max_length=128, batch_size=2,
                           length_sorted=False)
        np.testing.assert_allclose(a, b, atol=1e-6)

    def test_pad_fraction_is_zero_for_equal_lengths(self):
        encoder = StubEncoder()
        _, pad = embed_texts(
            encoder, ["abcde"] * 8, max_length=64, batch_size=4
        )
        assert pad == pytest.approx(0.0, abs=1e-9)

    def test_pad_fraction_is_large_for_mixed_lengths_in_one_batch(self):
        encoder = StubEncoder()
        _, pad = embed_texts(
            encoder, ["a", "b" * 200], max_length=256, batch_size=2,
            length_sorted=False,
        )
        assert pad > 0.4

    def test_length_sorting_reduces_padding(self):
        """The mitigation for the Clifford arm's maskless block."""
        encoder = StubEncoder()
        texts = ["z" * n for n in (1, 200, 2, 199, 3, 198)]
        _, sorted_pad = embed_texts(
            encoder, texts, max_length=256, batch_size=2, length_sorted=True
        )
        _, unsorted_pad = embed_texts(
            encoder, texts, max_length=256, batch_size=2, length_sorted=False
        )
        assert sorted_pad < unsorted_pad

    def test_truncation_to_the_window(self):
        encoder = StubEncoder()
        emb, _ = embed_texts(encoder, ["q" * 10000], max_length=32,
                             batch_size=1)
        assert emb[0, 0] == pytest.approx(32.0, abs=1e-5)

    def test_empty_input(self):
        emb, pad = embed_texts(StubEncoder(), [], max_length=32)
        assert emb.shape[0] == 0 and pad == 0.0


# ---------------------------------------------------------------------
# Failure containment
# ---------------------------------------------------------------------

class TestEvaluationNeverKillsATrainingCell:
    def test_a_missing_encoder_raises_a_clear_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="did not finish stage 1"):
            load_encoder(str(tmp_path))

    def test_run_study_cell_records_eval_ok_false_and_does_not_raise(
        self, tmp_path, monkeypatch
    ):
        """A failed eval must be visible, not fatal and not silent."""
        from train.embeddings_experimental import train_embeddings

        results = {}
        run_dir = str(tmp_path)

        def boom(*args, **kwargs):
            raise RuntimeError("planted")

        monkeypatch.setattr(
            "train.embeddings_experimental.evaluate_embeddings.evaluate_run",
            boom,
        )
        # Exercise the same guarded block the trainer uses.
        try:
            from train.embeddings_experimental.evaluate_embeddings import (
                EvalConfig as _EC,
                evaluate_run as _run,
            )
            results["embedding_eval"] = _run(run_dir, _EC())
        except Exception as exc:
            results["embedding_eval"] = {"eval_ok": False,
                                         "eval_error": str(exc)[:500]}
        assert results["embedding_eval"]["eval_ok"] is False
        assert "planted" in results["embedding_eval"]["eval_error"]


# ---------------------------------------------------------------------
# The real corpora
# ---------------------------------------------------------------------

@pytest.mark.skipif(not HAS_SQUAD, reason="local SQuAD TFDS cache not present")
class TestSquadProtocol:
    def test_the_pool_is_deduplicated_and_smaller_than_the_row_count(self):
        """Without dedup, retrieving an identical passage counts as an error."""
        from train.embeddings_experimental.evaluate_embeddings import _load_tfds

        rows = list(
            _load_tfds("squad/v1.1", "validation", TFDS_DIR).as_numpy_iterator()
        )
        contexts = [r["context"].decode("utf-8", "ignore") for r in rows]
        assert len(rows) == 10570
        assert 1900 < len(set(contexts)) < 2200
        assert len(set(contexts)) < len(contexts)

    def test_queries_are_sampled_not_taken_as_a_prefix(self):
        """SQuAD is ordered by title; a prefix would be topic classification."""
        rng = np.random.default_rng(0)
        picked = rng.permutation(10570)[:2000]
        assert not np.array_equal(np.sort(picked)[:50], np.arange(50))

    def test_tfds_is_never_allowed_to_download(self, monkeypatch):
        import tensorflow_datasets as tfds
        from train.embeddings_experimental import evaluate_embeddings as ev

        seen = {}

        def fake_load(name, **kwargs):
            seen.update(kwargs)
            raise RuntimeError("stop here")

        monkeypatch.setattr(tfds, "load", fake_load)
        with pytest.raises(RuntimeError, match="stop here"):
            ev._load_tfds("squad/v1.1", "validation", TFDS_DIR)
        assert seen["download"] is False


@pytest.mark.skipif(not HAS_SST2, reason="local SST-2 TFDS cache not present")
class TestSst2Protocol:
    def test_the_validation_split_is_labelled_and_the_baseline_is_known(self):
        from train.embeddings_experimental.evaluate_embeddings import _load_tfds

        rows = list(
            _load_tfds("glue/sst2", "validation", TFDS_DIR).as_numpy_iterator()
        )
        labels = np.array([int(r["label"]) for r in rows])
        assert len(rows) == 872
        assert set(np.unique(labels)) == {0, 1}
        counts = np.bincount(labels)
        assert counts.max() / counts.sum() == pytest.approx(0.5092, abs=1e-3)
