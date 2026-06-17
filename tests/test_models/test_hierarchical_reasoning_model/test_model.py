"""
Construction + M2 .keras round-trip test for the Hierarchical Reasoning Model.

HRM has a build() override (SAM D-008 pattern). call() dispatches a dict batch
({token_ids (B,T), puzzle_ids (B,)}) and returns a dict with logits +
q_halt_logits + q_continue_logits. Pins identical outputs across save -> load.
"""

import os
import keras
import pytest
import numpy as np

from dl_techniques.models.hierarchical_reasoning_model.model import (
    create_hierarchical_reasoning_model,
)

SEQ_LEN = 16


def _model():
    return create_hierarchical_reasoning_model(
        vocab_size=256, seq_len=SEQ_LEN, variant="micro")


def _batch(batch=2):
    rng = np.random.default_rng(0)
    return {
        "token_ids": rng.integers(0, 256, (batch, SEQ_LEN)).astype("int32"),
        "puzzle_ids": rng.integers(0, 1000, (batch,)).astype("int32"),
    }


class TestHRM:

    def test_forward_dict(self):
        out = _model()(_batch(), training=False)
        assert "logits" in out
        for v in out.values():
            arr = keras.ops.convert_to_numpy(v)
            assert not np.any(np.isnan(arr))

    def test_keras_round_trip(self, tmp_path):
        model = _model()
        x = _batch()
        before = keras.ops.convert_to_numpy(model(x, training=False)["logits"])

        path = os.path.join(str(tmp_path), "hrm.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        after = keras.ops.convert_to_numpy(loaded(x, training=False)["logits"])

        # GPU fp32 reduction noise -> atol 1e-4 (SYSTEM invariant)
        np.testing.assert_allclose(before, after, atol=1e-4,
                                   err_msg="HRM differs after .keras round-trip")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
