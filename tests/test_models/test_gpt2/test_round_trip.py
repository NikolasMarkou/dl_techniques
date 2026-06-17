"""
M2 .keras round-trip test for GPT2.

GPT2.call() accepts an int32 (B, T) token tensor and returns a dict with
`logits` + `last_hidden_state`. Pins numerically-identical outputs across a
full save -> load cycle.
"""

import os
import keras
import pytest
import numpy as np

from dl_techniques.models.gpt2.gpt2 import GPT2


def _model():
    return GPT2(vocab_size=256, embed_dim=64, depth=2, num_heads=4,
                max_seq_len=32, dropout_rate=0.0, attention_dropout_rate=0.0)


def _tokens(batch=2, seq=16):
    return np.random.default_rng(0).integers(0, 256, (batch, seq)).astype("int32")


class TestGPT2RoundTrip:

    def test_forward_dict(self):
        out = _model()(_tokens(), training=False)
        assert {"logits", "last_hidden_state"} <= set(out)
        assert tuple(out["logits"].shape) == (2, 16, 256)

    def test_keras_round_trip(self, tmp_path):
        model = _model()
        x = _tokens()
        before = keras.ops.convert_to_numpy(model(x, training=False)["logits"])

        path = os.path.join(str(tmp_path), "gpt2.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        after = keras.ops.convert_to_numpy(loaded(x, training=False)["logits"])

        # GPU fp32 reduction noise -> atol 1e-4 (SYSTEM invariant)
        np.testing.assert_allclose(before, after, atol=1e-4,
                                   err_msg="GPT2 differs after .keras round-trip")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
