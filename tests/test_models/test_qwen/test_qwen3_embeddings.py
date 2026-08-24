"""
Test suite for Qwen3 embedding + reranker models.

Both consume a dict {'input_ids': (B,T), 'attention_mask': (B,T)} and perform
last-token pooling. Regression note: last-token pooling used a 2-D index into a
3-D tensor for ops.take_along_axis (dead-on-forward); fixed to broadcast the
per-row index to (B, 1, D). These tests pin the forward + M2 round-trip.
"""

import os
import keras
import pytest
import numpy as np

from dl_techniques.models.qwen.qwen3_embeddings import (
    Qwen3EmbeddingModel, Qwen3RerankerModel,
)

VOCAB = 1000
HID = 64


def _inputs(batch=2, seq=16):
    rng = np.random.default_rng(0)
    return {
        "input_ids": rng.integers(0, VOCAB, (batch, seq)).astype("int32"),
        "attention_mask": np.ones((batch, seq), dtype="int32"),
    }


def _embed_model():
    return Qwen3EmbeddingModel(vocab_size=VOCAB, hidden_size=HID, num_layers=2,
                               num_heads=4, intermediate_size=128, max_seq_len=64)


def _rerank_model():
    return Qwen3RerankerModel(vocab_size=VOCAB, hidden_size=HID, num_layers=2,
                              num_heads=4, intermediate_size=128, max_seq_len=64,
                              yes_token_id=11, no_token_id=22)


class TestQwen3EmbeddingModel:

    def test_forward_pooled(self):
        out = _embed_model()(_inputs(), training=False)
        assert tuple(out.shape) == (2, HID)
        assert not np.any(np.isnan(keras.ops.convert_to_numpy(out)))

    def test_keras_round_trip(self, tmp_path):
        model = _embed_model()
        x = _inputs()
        before = keras.ops.convert_to_numpy(model(x, training=False))
        path = os.path.join(str(tmp_path), "qwen3_emb.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        after = keras.ops.convert_to_numpy(loaded(x, training=False))
        np.testing.assert_allclose(before, after, atol=1e-4,
                                   err_msg="Qwen3EmbeddingModel differs after round-trip")


class TestQwen3RerankerModel:

    def test_forward_score(self):
        out = _rerank_model()(_inputs(), training=False)
        assert int(out.shape[0]) == 2
        assert not np.any(np.isnan(keras.ops.convert_to_numpy(out)))

    def test_keras_round_trip(self, tmp_path):
        model = _rerank_model()
        x = _inputs()
        before = keras.ops.convert_to_numpy(model(x, training=False))
        path = os.path.join(str(tmp_path), "qwen3_rerank.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        after = keras.ops.convert_to_numpy(loaded(x, training=False))
        np.testing.assert_allclose(before, after, atol=1e-4,
                                   err_msg="Qwen3RerankerModel differs after round-trip")


class _Tap:
    """Plain callable (not a Layer) so it can replace an attribute on an
    already-built layer, which Keras 3 refuses for tracked sub-layers."""

    def __init__(self, inner, sink):
        self.inner = inner
        self.sink = sink

    def __call__(self, x, *args, **kwargs):
        self.sink.append(keras.ops.convert_to_numpy(x))
        return self.inner(x, *args, **kwargs)


def _hidden_states_pair(layer, attr, seq=12, perturb_at=6):
    """Run ``layer`` twice on ids differing only at ``perturb_at`` and return
    the two per-position sequences that reached ``attr``."""
    rng = np.random.default_rng(0)
    ids = rng.integers(1, VOCAB, (2, seq)).astype("int32")
    mask = np.ones((2, seq), dtype="int32")
    layer({"input_ids": ids, "attention_mask": mask})  # build

    sink = []
    object.__setattr__(layer, attr, _Tap(getattr(layer, attr), sink))

    other = ids.copy()
    other[:, perturb_at] = (other[:, perturb_at] + 7) % (VOCAB - 1) + 1
    layer({"input_ids": ids, "attention_mask": mask}, training=False)
    layer({"input_ids": other, "attention_mask": mask}, training=False)
    return np.asarray(sink[0]), np.asarray(sink[1])


class TestQwen3Causality:
    """The reranker is causal; the embedding tower deliberately is not.

    The reranker reads its own LM head at the last real position, which makes
    it a next-token prediction: without a causal mask that position has
    already attended to the tokens it is scoring. Measured on the unmasked
    implementation, perturbing index 6 of a 12-token sequence moved the
    hidden states at indices < 6 by ``3.421534e-01``; it is now exactly 0.0.

    The embedding tower predicts nothing, so there is no target to leak and
    bidirectional attention is strictly more informative. That is a decision,
    not an oversight, and the test below pins it so a future "consistency"
    edit has to argue with a test rather than with a comment.
    """

    PERTURB_AT = 6

    def _reranker(self):
        from dl_techniques.models.qwen.qwen3_embeddings import Qwen3RerankerLayer
        return Qwen3RerankerLayer(
            vocab_size=VOCAB, hidden_size=32, num_layers=2, num_heads=4,
            intermediate_size=64, max_seq_len=12, yes_token_id=5,
            no_token_id=6,
        )

    def _embedder(self):
        from dl_techniques.models.qwen.qwen3_embeddings import Qwen3EmbeddingLayer
        return Qwen3EmbeddingLayer(
            vocab_size=VOCAB, hidden_size=32, num_layers=2, num_heads=4,
            intermediate_size=64, max_seq_len=12, normalize=False,
        )

    def test_reranker_does_not_leak_the_future(self):
        a, b = _hidden_states_pair(self._reranker(), "lm_head")
        past = np.abs(a[:, :self.PERTURB_AT] - b[:, :self.PERTURB_AT]).max()
        assert past == 0.0, f"reranker future leak: {past:.6e} (must be 0.0)"

    def test_reranker_still_responds(self):
        a, b = _hidden_states_pair(self._reranker(), "lm_head")
        future = np.abs(a[:, self.PERTURB_AT:] - b[:, self.PERTURB_AT:]).max()
        assert future > 1e-3, f"reranker is inert: {future:.6e}"

    def test_embedding_tower_is_bidirectional_on_purpose(self):
        a, b = _hidden_states_pair(self._embedder(), "final_norm")
        past = np.abs(a[:, :self.PERTURB_AT] - b[:, :self.PERTURB_AT]).max()
        assert past > 1e-3, (
            "the embedding tower is expected to be BIDIRECTIONAL: an encoder "
            "predicts nothing, so there is no target to leak. If a causal mask "
            "was added here deliberately, change this test and say why."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
