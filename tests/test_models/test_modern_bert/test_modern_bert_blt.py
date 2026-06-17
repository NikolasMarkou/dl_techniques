"""
M2 .keras round-trip test for ModernBertBLT.

ModernBertBLT is the byte-level ModernBERT with ngram-hash embeddings. call()
accepts an int32 (B, T) byte-token tensor and returns a dict with
last_hidden_state + attention_mask. A small config is used to avoid the
documented ngram-hash OOM at large seq_len / hash_vocab_size (§L2-5).
"""

import os
import keras
import pytest
import numpy as np

from dl_techniques.models.modern_bert.modern_bert_blt import ModernBertBLT


def _model():
    return ModernBertBLT(
        vocab_size=260, hidden_size=64, num_layers=2, num_heads=2,
        intermediate_size=128, hash_vocab_size=1000, max_seq_len=64,
        global_attention_interval=2, local_attention_window_size=16,
    )


def _tokens(batch=2, seq=16):
    return np.random.default_rng(0).integers(0, 260, (batch, seq)).astype("int32")


class TestModernBertBLT:

    def test_forward_dict(self):
        out = _model()(_tokens(), training=False)
        assert "last_hidden_state" in out
        assert tuple(out["last_hidden_state"].shape) == (2, 16, 64)

    def test_keras_round_trip(self, tmp_path):
        model = _model()
        x = _tokens()
        before = keras.ops.convert_to_numpy(model(x, training=False)["last_hidden_state"])

        path = os.path.join(str(tmp_path), "modern_bert_blt.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        after = keras.ops.convert_to_numpy(loaded(x, training=False)["last_hidden_state"])

        # GPU fp32 reduction noise -> atol 1e-4 (SYSTEM invariant)
        np.testing.assert_allclose(before, after, atol=1e-4,
                                   err_msg="ModernBertBLT differs after .keras round-trip")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
