"""Tests for ``LocalDecoder`` (``dl_techniques.layers.blt.local_decoder``)."""

import os

import keras
import numpy as np

from dl_techniques.layers.blt.local_decoder import LocalDecoder

VOCAB, HID, SEQ, NP_, GDIM = 32, 16, 10, 8, 16
B = 2


def _tokens():
    return keras.ops.convert_to_tensor(
        np.random.default_rng(0).integers(0, VOCAB, size=(B, SEQ)).astype("int32")
    )


def _patch_ids():
    return keras.ops.convert_to_tensor(
        np.random.default_rng(1).integers(0, NP_, size=(B, SEQ)).astype("int32")
    )


# ---------------------------------------------------------------------
# LocalDecoder (multi-arg call -> Model-wrapper round-trip)
# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class _DecWrapper(keras.Model):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.dec = LocalDecoder(
            vocab_size=VOCAB, local_dim=HID, global_dim=GDIM, num_local_layers=1,
            num_heads_local=2,
        )

    def call(self, inputs, training=None):
        return self.dec(inputs[0], inputs[1], inputs[2], training=training)


class TestLocalDecoder:

    def _gctx(self):
        return keras.ops.convert_to_tensor(
            np.random.default_rng(2).standard_normal((B, NP_, GDIM)).astype("float32")
        )

    def test_forward_pass(self):
        dec = LocalDecoder(vocab_size=VOCAB, local_dim=HID, global_dim=GDIM,
                          num_local_layers=1, num_heads_local=2)
        out = dec(_tokens(), self._gctx(), _patch_ids())
        assert tuple(out.shape) == (B, SEQ, VOCAB)

    def test_serialization_round_trip(self, tmp_path):
        model = _DecWrapper()
        inputs = [_tokens(), self._gctx(), _patch_ids()]
        y0 = model(inputs)
        path = os.path.join(tmp_path, "dec.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        y1 = loaded(inputs)
        assert tuple(y0.shape) == tuple(y1.shape) == (B, SEQ, VOCAB)
