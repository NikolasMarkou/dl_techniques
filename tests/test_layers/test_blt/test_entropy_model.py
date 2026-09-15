"""Tests for ``EntropyModel`` (``dl_techniques.layers.blt.entropy_model``)."""

import os

import keras
import numpy as np

from dl_techniques.layers.blt.entropy_model import EntropyModel

VOCAB, HID, SEQ = 32, 16, 10
B = 2


def _tokens():
    return keras.ops.convert_to_tensor(
        np.random.default_rng(0).integers(0, VOCAB, size=(B, SEQ)).astype("int32")
    )


# ---------------------------------------------------------------------
# EntropyModel (single-tensor call -> functional .keras round-trip)
# ---------------------------------------------------------------------

class TestEntropyModel:

    def _make(self):
        return EntropyModel(vocab_size=VOCAB, hidden_dim=HID, num_layers=1,
                            num_heads=2, max_seq_len=64)

    def test_forward_pass(self):
        out = self._make()(_tokens())
        assert tuple(out.shape) == (B, SEQ, VOCAB)

    def test_compute_output_shape(self):
        assert self._make().compute_output_shape((B, SEQ)) == (B, SEQ, VOCAB)

    def test_serialization_round_trip(self, tmp_path):
        inp = keras.Input(shape=(SEQ,), dtype="int32")
        out = self._make()(inp)
        model = keras.Model(inp, out)
        toks = _tokens()
        y0 = model(toks)
        path = os.path.join(tmp_path, "entropy.keras")
        model.save(path)
        loaded = keras.models.load_model(path, custom_objects={"EntropyModel": EntropyModel})
        y1 = loaded(toks)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y0), keras.ops.convert_to_numpy(y1),
            rtol=1e-5, atol=1e-5,
        )
