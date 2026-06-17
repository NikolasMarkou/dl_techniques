"""
Test suite for DistilBERT (encoder foundation model).

Covers construction, a forward pass, and the M2 full .keras
save -> load -> identical-output round-trip. call() accepts an int32 (B, T)
token tensor and returns a dict with `last_hidden_state` + `attention_mask`.
"""

import os
import keras
import pytest
import numpy as np

from dl_techniques.models.distilbert.model import DistilBERT


def _model():
    return DistilBERT(
        vocab_size=256,
        hidden_size=64,
        num_layers=2,
        num_heads=2,
        intermediate_size=128,
        max_position_embeddings=64,
    )


def _tokens(batch=2, seq=16):
    return np.random.default_rng(0).integers(0, 256, (batch, seq)).astype("int32")


class TestDistilBERT:

    def test_forward_dict(self):
        out = _model()(_tokens(), training=False)
        assert "last_hidden_state" in out
        assert tuple(out["last_hidden_state"].shape) == (2, 16, 64)

    def test_keras_round_trip(self, tmp_path):
        model = _model()
        x = _tokens()
        before = keras.ops.convert_to_numpy(model(x, training=False)["last_hidden_state"])

        path = os.path.join(str(tmp_path), "distilbert.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        after = keras.ops.convert_to_numpy(loaded(x, training=False)["last_hidden_state"])

        # GPU fp32 reduction noise -> atol 1e-4 (SYSTEM invariant)
        np.testing.assert_allclose(before, after, atol=1e-4,
                                   err_msg="DistilBERT differs after .keras round-trip")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
