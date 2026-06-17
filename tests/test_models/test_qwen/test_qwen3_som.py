"""
Test suite for Qwen3-SOM (self-organizing-map augmented Qwen3).

create_qwen3som(variant, task_type) returns a functional Keras model taking a
dict {'input_ids': (B,T), 'attention_mask': (B,T)}. Generation returns LM logits
(B, T, vocab); classification returns (B, num_labels). Covers forward + M2
round-trip for both task types.
"""

import os
import keras
import pytest
import numpy as np

from dl_techniques.models.qwen.qwen3_som import create_qwen3som


def _inputs(batch=2, seq=16):
    rng = np.random.default_rng(0)
    return {
        "input_ids": rng.integers(0, 1000, (batch, seq)).astype("int32"),
        "attention_mask": np.ones((batch, seq), dtype="int32"),
    }


def _round_trip(model, x, tmp_path, name):
    before = keras.ops.convert_to_numpy(model(x, training=False))
    path = os.path.join(str(tmp_path), f"{name}.keras")
    model.save(path)
    loaded = keras.models.load_model(path)
    after = keras.ops.convert_to_numpy(loaded(x, training=False))
    np.testing.assert_allclose(before, after, atol=1e-4,
                               err_msg=f"{name} differs after round-trip")


class TestQwen3SOM:

    def test_generation_forward_and_round_trip(self, tmp_path):
        model = create_qwen3som("tiny_som", task_type="generation")
        x = _inputs()
        out = model(x, training=False)
        assert out.shape[0] == 2 and out.shape[1] == 16
        _round_trip(model, x, tmp_path, "qwen3_som_gen")

    def test_classification_forward_and_round_trip(self, tmp_path):
        model = create_qwen3som("tiny_som", task_type="classification", num_labels=5)
        x = _inputs()
        out = model(x, training=False)
        assert tuple(out.shape) == (2, 5)
        _round_trip(model, x, tmp_path, "qwen3_som_cls")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
