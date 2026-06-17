"""
Test suite for Qwen3-MEGA (memory + entity-graph augmented Qwen3).

create_qwen3_mega(variant) builds the model; call() takes int32 (B, T) token ids
and returns LM logits (B, T, vocab). Covers a forward pass and the M2 full
.keras save -> load -> identical-output round-trip.
"""

import os
import keras
import pytest
import numpy as np

from dl_techniques.models.qwen.qwen3_mega import create_qwen3_mega, Qwen3MEGA


def _tokens(batch=2, seq=16):
    return np.random.default_rng(0).integers(0, 1000, (batch, seq)).astype("int32")


class TestQwen3MEGA:

    def test_factory_construction(self):
        assert isinstance(create_qwen3_mega("tiny"), Qwen3MEGA)

    def test_forward_logits(self):
        out = create_qwen3_mega("tiny")(_tokens(), training=False)
        y = out[0] if isinstance(out, (list, tuple)) else (
            out["logits"] if isinstance(out, dict) and "logits" in out else out)
        assert int(y.shape[0]) == 2
        assert not np.any(np.isnan(keras.ops.convert_to_numpy(y)))

    def test_keras_round_trip(self, tmp_path):
        model = create_qwen3_mega("tiny")
        x = _tokens()

        def primary(o):
            if isinstance(o, dict):
                return o.get("logits", list(o.values())[0])
            return o[0] if isinstance(o, (list, tuple)) else o

        before = keras.ops.convert_to_numpy(primary(model(x, training=False)))
        path = os.path.join(str(tmp_path), "qwen3_mega.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        after = keras.ops.convert_to_numpy(primary(loaded(x, training=False)))
        np.testing.assert_allclose(before, after, atol=1e-4,
                                   err_msg="Qwen3MEGA differs after round-trip")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
