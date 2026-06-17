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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
