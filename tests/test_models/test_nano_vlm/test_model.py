"""
Test suite for NanoVLM (vision-language model).

call() consumes a dict {'images': (B,224,224,3), 'text_tokens': (B,T)} and
returns LM logits (B, combined_seq, vocab_size). Covers construction via the
create_nanovlm factory, a forward pass, and the M2 full .keras
save -> load -> identical-output round-trip.
"""

import os
import keras
import pytest
import numpy as np

from dl_techniques.models.nano_vlm.model import create_nanovlm, NanoVLM

VOCAB = 256


def _model():
    return create_nanovlm("mini", vocab_size=VOCAB)


def _inputs(batch=2, seq=16):
    rng = np.random.default_rng(0)
    return {
        "images": rng.random((batch, 224, 224, 3)).astype("float32"),
        "text_tokens": rng.integers(0, VOCAB, (batch, seq)).astype("int32"),
    }


class TestNanoVLM:

    def test_factory_construction(self):
        model = _model()
        assert isinstance(model, NanoVLM)

    def test_forward_logits(self):
        out = _model()(_inputs(), training=False)
        assert out.shape[0] == 2 and out.shape[-1] == VOCAB
        assert not np.any(np.isnan(keras.ops.convert_to_numpy(out)))

    def test_keras_round_trip(self, tmp_path):
        model = _model()
        x = _inputs()
        before = keras.ops.convert_to_numpy(model(x, training=False))

        path = os.path.join(str(tmp_path), "nano_vlm.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        after = keras.ops.convert_to_numpy(loaded(x, training=False))

        # GPU fp32 reduction noise -> atol 1e-4 (SYSTEM invariant)
        np.testing.assert_allclose(before, after, atol=1e-4,
                                   err_msg="NanoVLM differs after .keras round-trip")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
